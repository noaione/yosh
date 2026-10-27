//! OGSOV: histogram features followed by five averaged MLPs.
//!
//! The oracle is `OGSOV.compute_image_vectors_hist()` plus `predict()`, not
//! the older `compute_image_vectors()` path: the two disagree because the
//! older one takes its top-17 values in NumPy partition order and computes its
//! statistics with different precision.  Only the histogram path is ported.
//!
//! Feature positions are fixed by the reference's concatenation order:
//! `0..17` the descending top values, then standard deviation, quantized
//! median, skew, kurtosis, IQR and L2 norm, then the five quantiles, then the
//! four capped mask-hit ratios.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::lookup::LookupTable;
use crate::math;
use crate::rgb::RgbView;
use crate::weights::{OGSOV_CLASSIFIERS, OGSOV_HIDDEN, OGSOV_INPUT, OgsovModel};

mod simd;

/// The reference removes this many of the largest samples before computing the
/// statistics.
pub const TOP_N: usize = 17;

/// Smallest number of pixels the trimmed statistics are defined for.
///
/// The reference computes `m = pixels - 17` and then indexes `elem(m - 1)`, so
/// 17 pixels would leave `m = 0` and no usable remainder.  Frames of at most
/// this size are rejected with a descriptive error instead.
pub const MIN_PIXELS: u64 = (TOP_N + 1) as u64;

/// Largest possible value of `4*(R-G)^2 + (R+G-2B)^2` for 8-bit channels.
pub const MAX_D2: usize = 325_125;

/// Number of histogram bins.
pub const HISTOGRAM_LEN: usize = MAX_D2 + 1;

/// Feature vector width.
pub const FEATURE_COUNT: usize = 32;

/// The reference only ever uses `min(hits, 1024)`.
pub const MASK_HIT_CAP: u32 = 1024;

/// The reference multiplies a value by this before rounding it to an integer.
const MEDIAN_SCALER: f64 = 1000.0;

/// The histogram's float32 value table, built once per process.
///
/// `vals[i] = min(sqrt(float32(i)) * 0.5, 285) / 285`, evaluated in `f32` with
/// the reference's exact operation order so the table is bit-identical to the
/// one NumPy builds.
pub fn value_table() -> &'static [f32] {
    static TABLE: OnceLock<Box<[f32]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut values = Vec::with_capacity(HISTOGRAM_LEN);
        for index in 0..HISTOGRAM_LEN {
            let root = (index as f32).sqrt() * 0.5f32;
            values.push(root.min(285.0f32) / 285.0f32);
        }
        values.into_boxed_slice()
    })
}

/// What one `debug` extraction measured, per stage.
///
/// The classifier core has no host handle and therefore logs nothing itself:
/// it leaves these numbers on the workspace it was handed, and the caller
/// decides what to do with them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OgsovStages {
    /// The pixel walk, which counts every pixel into the histogram.
    pub scan: Duration,
    /// Everything after it: the top-N descent, the order statistics, the
    /// moments and the feature assembly — and, on the collected path, the
    /// read-back reduction itself, since no scan ran on that side.
    pub tail: Duration,
    /// The five MLPs, measured by [`predict`].
    pub mlp: Duration,
}

/// One reusable OGSOV workspace.
///
/// A full histogram is `HISTOGRAM_LEN` counters wide, so a workspace is
/// allocated per active frame rather than per call; the value table above is
/// shared and immutable.
pub struct OgsovScratch {
    histogram: Vec<u32>,
    cumulative: Vec<u64>,
    /// Filled in only by a `debug` run; zero otherwise.
    stages: OgsovStages,
    /// The largest `d2` the previous frame reached, taken from its top-N
    /// descent.
    ///
    /// Bins above it hold no samples and were never incremented, so clearing
    /// and scanning stop there: a monochrome page costs one bin instead of
    /// 325,126 without the pixel loop paying for the bound.
    highest: usize,
    top: [f32; TOP_N],
    hidden1: [f32; OGSOV_HIDDEN],
    hidden2: [f32; OGSOV_HIDDEN],
    logit: [f32; 1],
    ratio_bias: [f32; 1],
}

impl Default for OgsovScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl OgsovScratch {
    /// Allocate a workspace.
    pub fn new() -> Self {
        OgsovScratch {
            histogram: vec![0u32; HISTOGRAM_LEN],
            cumulative: vec![0u64; HISTOGRAM_LEN],
            stages: OgsovStages::default(),
            highest: 0,
            top: [0.0f32; TOP_N],
            hidden1: [0.0f32; OGSOV_HIDDEN],
            hidden2: [0.0f32; OGSOV_HIDDEN],
            logit: [0.0f32; 1],
            ratio_bias: [0.0f32; 1],
        }
    }

    /// The stages the last `debug` extraction measured.
    ///
    /// Every field is zero unless the call that filled it was given
    /// `debug = true`.
    #[must_use]
    pub fn stages(&self) -> OgsovStages {
        self.stages
    }

    /// Bytes this workspace keeps alive.
    pub fn resident_bytes(&self) -> usize {
        self.histogram.len() * std::mem::size_of::<u32>()
            + self.cumulative.len() * std::mem::size_of::<u64>()
    }

    /// Reduce the histogram this workspace's own pixel walk filled.
    fn reduce(&mut self, pixels: u64, hits: u32) -> [f32; FEATURE_COUNT] {
        let OgsovScratch {
            histogram,
            cumulative,
            top,
            highest,
            ..
        } = self;
        reduce_histogram(histogram, cumulative, top, highest, pixels, hits)
    }

    /// Reduce a histogram collected elsewhere, without copying it in.
    ///
    /// The two paths differ only in which slice the bins come from: the scan
    /// wrote this workspace's own, and a dispatch wrote one the caller read
    /// back.  Neither is modified.
    ///
    /// The clear bound is *not* lowered here.  This workspace's own bins were
    /// not written on this path, so whatever an earlier scan left above this
    /// frame's largest distance is still in them, and the next scan clears up
    /// to `highest`: keeping the wider of the two is what keeps the invariant
    /// that every bin above it is zero.
    fn reduce_collected(
        &mut self,
        histogram: &[u32],
        pixels: u64,
        hits: u32,
    ) -> [f32; FEATURE_COUNT] {
        let previous = self.highest;
        let features = {
            let OgsovScratch {
                cumulative,
                top,
                highest,
                ..
            } = self;
            reduce_histogram(histogram, cumulative, top, highest, pixels, hits)
        };
        self.highest = self.highest.max(previous);
        features
    }
}

/// The reference's `d2` for one pixel.
///
/// `d2 = 4*(R-G)^2 + (R+G-2B)^2`, which is exactly four times the squared
/// chroma distance the model is built on.  Intermediate values stay inside
/// `i32` for 8-bit inputs.
#[inline]
pub fn distance_squared(red: u8, green: u8, blue: u8) -> usize {
    let red = red as i32;
    let green = green as i32;
    let blue = blue as i32;
    let rg = red - green;
    let yb = red + green - 2 * blue;
    (4 * rg * rg + yb * yb) as usize
}

/// Extract the 32 OGSOV features from an RGB frame.
///
/// `full_frame` probes every pixel of the frame for a lookup hit instead of
/// skipping the pixels whose channels are too close together to be in the
/// table.  Those pixels were never hits, so the features do not change; it
/// exists to be able to check that on real frames, and it is slower.
///
/// `debug` times the two stages this function owns and leaves them on
/// `scratch` ([`OgsovScratch::stages`]).  A `debug` run is slower by
/// construction, and the numbers it reports include that cost.
pub fn extract_features(
    rgb: &RgbView<'_>,
    lookup: &LookupTable,
    scratch: &mut OgsovScratch,
    full_frame: bool,
    debug: bool,
) -> Result<[f32; FEATURE_COUNT]> {
    let pixels = rgb.pixel_count()?;
    if pixels < MIN_PIXELS {
        return Err(Error::TooFewPixels {
            at_most: MIN_PIXELS - 1,
            actual: pixels,
        });
    }
    if pixels > u32::MAX as u64 {
        return Err(Error::DimensionOverflow {
            detail: format!(
                "OGSOV histogram counters are 32-bit and this frame has {pixels} pixels"
            ),
        });
    }

    // Only the bins the previous frame touched need clearing: a bin above the
    // largest distance that frame reached was never incremented, so it is
    // still zero.  The reference draws the same line, since its histogram is
    // `max(d2) + 1` wide.
    scratch.histogram[..=scratch.highest].fill(0);

    // The distances and the lookup hits are the same walk over the frame, so
    // one call does both; `simd` runs it a vector at a time on aarch64 and a
    // pixel at a time anywhere else.
    let scan_started = debug.then(Instant::now);
    let hits = simd::scan(
        rgb,
        &mut scratch.histogram,
        lookup,
        lookup.spread_threshold(full_frame),
    );
    let scan_finished = debug.then(Instant::now);

    // The pixel walk is over, and what it left behind is a histogram like any
    // other: reduce it with the same function a collected histogram uses.
    let features = scratch.reduce(pixels, hits);

    if let (Some(started), Some(walked), Some(finished)) =
        (scan_started, scan_finished, debug.then(Instant::now))
    {
        scratch.stages = OgsovStages {
            scan: walked - started,
            tail: finished - walked,
            mlp: Duration::ZERO,
        };
    }
    Ok(features)
}

/// One bin's contribution to the cumulative counts and the first two weighted
/// moments.
///
/// Split out of the pass that walks them so the spans between the bins the
/// top-N descent touched run the body with no per-bin test; see
/// [`reduce_histogram`].
#[inline]
fn accumulate_bin(
    bin: usize,
    remaining: u32,
    cumulative: &mut [u64],
    values: &[f32],
    running: &mut u64,
    s1: &mut f64,
    s2: &mut f64,
) {
    *running += u64::from(remaining);
    cumulative[bin] = *running;
    let weight = f64::from(remaining);
    if weight == 0.0 {
        return;
    }
    let value = f64::from(values[bin]);
    *s1 += weight * value;
    *s2 += weight * (value * value);
}

/// One bin's contribution to the third and fourth weighted moments.
#[inline]
fn accumulate_moment(
    bin: usize,
    remaining: u32,
    values: &[f32],
    mean: f64,
    m2: &mut f64,
    m3: &mut f64,
    m4: &mut f64,
) {
    let weight = f64::from(remaining);
    if weight == 0.0 {
        return;
    }
    let centered = f64::from(values[bin]) - mean;
    let squared = centered * centered;
    *m2 += weight * squared;
    *m3 += weight * (squared * centered);
    *m4 += weight * (squared * squared);
}

/// The half of the extraction that follows the pixel walk: histogram to
/// features.
///
/// Both paths go through this.  A pixel scan and a histogram collected
/// elsewhere fill the same bins with the same integer arithmetic, so one
/// reduction serves both and neither can drift from the other.
///
/// `histogram` is borrowed and never written: the top-[`TOP_N`] descent takes
/// samples *out of the statistics* rather than out of the bins, and the two
/// ascending passes below subtract again exactly what it took.  That is what
/// lets the collected path hand its own words straight in instead of copying a
/// 1.3 MB histogram into the workspace first.
///
/// `pixels` is the number of samples the bins hold — the frame's pixel count,
/// which the caller knows.  The descent removes the largest [`TOP_N`] of them,
/// which is why the caller may not reduce the same histogram twice: the second
/// descent would take them again.
fn reduce_histogram(
    histogram: &[u32],
    cumulative: &mut [u64],
    top: &mut [f32; TOP_N],
    highest: &mut usize,
    pixels: u64,
    hits: u32,
) -> [f32; FEATURE_COUNT] {
    let values = value_table();
    debug_assert!(
        pixels >= MIN_PIXELS,
        "a frame of {pixels} pixels has no remainder after the top {TOP_N}"
    );
    let sample_count = pixels - TOP_N as u64;

    // 1. The descending top-N values, removed from the statistics by bin order
    //    even where the clip maps several bins to the same value.  The bins
    //    themselves are left alone: what the descent took is recorded here and
    //    subtracted by the ascending passes below.
    let mut collected = 0usize;
    // The first non-empty bin this descent meets is the largest distance in
    // the frame, so every bin above it is empty and every later scan stops
    // there.  The reference builds a `max(d2) + 1` histogram for the same
    // reason; taking the bound from this scan costs the pixel loop nothing.
    let mut bound = 0usize;
    // One entry per bin the descent touched, in the descending order it met
    // them: the bin, and how many of its samples were taken.
    let mut removed = [(0usize, 0u32); TOP_N];
    let mut touched = 0usize;
    for bin in (0..HISTOGRAM_LEN).rev() {
        let count = histogram[bin];
        if count == 0 {
            continue;
        }
        if collected == 0 {
            bound = bin;
        }
        let take = count.min((TOP_N - collected) as u32);
        for slot in &mut top[collected..collected + take as usize] {
            *slot = values[bin];
        }
        collected += take as usize;
        removed[touched] = (bin, take);
        touched += 1;
        if collected == TOP_N {
            break;
        }
    }
    debug_assert_eq!(collected, TOP_N);
    // Bounds the next frame's clear and this frame's scans.
    *highest = bound;

    // The same entries in ascending bin order, which is the order the passes
    // below meet them in.
    let mut adjustments = [(0usize, 0u32); TOP_N];
    for (slot, &(bin, take)) in adjustments.iter_mut().zip(removed[..touched].iter().rev()) {
        *slot = (bin, take);
    }

    // 2. The cumulative counts and the first two weighted moments in one
    //    ascending pass: all three start at the lowest bin, stop at `bound`, and
    //    read the same remaining count.  The reference's own order is kept —
    //    every term is added in ascending `bin`.
    //
    //    The pass runs in the spans *between* the bins the descent touched, so
    //    the inner loop carries no test for one of them: a dependent load and a
    //    branch on every bin cost 0.08 ms on a noise frame, which is more than
    //    the pass it guards (measured; see the plan's rejected shapes).
    let mut running = 0u64;
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    let mut start = 0usize;
    for &(bin, take) in &adjustments[..touched] {
        for (offset, &count) in histogram[start..bin].iter().enumerate() {
            accumulate_bin(
                start + offset,
                count,
                cumulative,
                values,
                &mut running,
                &mut s1,
                &mut s2,
            );
        }
        accumulate_bin(
            bin,
            histogram[bin] - take,
            cumulative,
            values,
            &mut running,
            &mut s1,
            &mut s2,
        );
        start = bin + 1;
    }
    for (offset, &count) in histogram[start..=bound].iter().enumerate() {
        accumulate_bin(
            start + offset,
            count,
            cumulative,
            values,
            &mut running,
            &mut s1,
            &mut s2,
        );
    }
    debug_assert_eq!(running, sample_count);

    // The value at 0-based sorted position `k`.
    let element = |k: u64| -> f64 {
        let bin = cumulative[..=bound].partition_point(|&count| count <= k);
        debug_assert!(bin <= bound, "position {k} is past the histogram");
        values[bin] as f64
    };
    let lerp = |percent: f64| -> f64 {
        let position = percent / 100.0 * (sample_count as f64 - 1.0);
        let low = position.floor();
        let high = position.ceil();
        let (a, b) = (element(low as u64), element(high as u64));
        a + (position - low) * (b - a)
    };

    let quantiles = [
        element(0),
        lerp(25.0),
        lerp(50.0),
        lerp(75.0),
        element(sample_count - 1),
    ];
    let iqr = quantiles[3] - quantiles[1];

    let count = sample_count as f64;
    let mean = s1 / count;
    let l2_norm = s2.sqrt();

    // 3. The higher moments in float64, over the bins that still hold samples.
    //    The remaining counts are the first pass's, so the same spans are walked
    //    again from the lowest bin.
    let (mut m2, mut m3, mut m4) = (0.0f64, 0.0f64, 0.0f64);
    let mut start = 0usize;
    for &(bin, take) in &adjustments[..touched] {
        for (offset, &count) in histogram[start..bin].iter().enumerate() {
            accumulate_moment(
                start + offset,
                count,
                values,
                mean,
                &mut m2,
                &mut m3,
                &mut m4,
            );
        }
        accumulate_moment(
            bin,
            histogram[bin] - take,
            values,
            mean,
            &mut m2,
            &mut m3,
            &mut m4,
        );
        start = bin + 1;
    }
    for (offset, &count) in histogram[start..=bound].iter().enumerate() {
        accumulate_moment(
            start + offset,
            count,
            values,
            mean,
            &mut m2,
            &mut m3,
            &mut m4,
        );
    }
    m2 /= count;
    m3 /= count;
    m4 /= count;

    let std = m2.sqrt();
    let (skew, kurtosis) = if m2 > 0.0 {
        (m3 / m2.powf(1.5), m4 / (m2 * m2) - 3.0)
    } else {
        (0.0, 0.0)
    };

    // 4. The separately quantized median: ties-to-even rounding of value*1000,
    //    averaging the two middle samples when the count is even.
    let median = if sample_count % 2 == 1 {
        (element(sample_count / 2) * MEDIAN_SCALER).round_ties_even() / MEDIAN_SCALER
    } else {
        ((element(sample_count / 2 - 1) * MEDIAN_SCALER).round_ties_even()
            + (element(sample_count / 2) * MEDIAN_SCALER).round_ties_even())
            / 2.0
            / MEDIAN_SCALER
    };

    // 5. The capped hit ratios, then the reference's concatenation order.
    let hits = hits as f64;
    let mut features = [0.0f32; FEATURE_COUNT];
    features[..TOP_N].copy_from_slice(top);
    let scalars = [std, median, skew, kurtosis, iqr, l2_norm];
    for (offset, value) in scalars.iter().enumerate() {
        features[TOP_N + offset] = *value as f32;
    }
    for (offset, value) in quantiles.iter().enumerate() {
        features[TOP_N + scalars.len() + offset] = *value as f32;
    }
    for (offset, denominator) in [2.0f64, 16.0, 128.0, 1024.0].iter().enumerate() {
        features[TOP_N + scalars.len() + quantiles.len() + offset] =
            (hits.min(*denominator) / denominator) as f32;
    }

    // The reference runs nan_to_num with every non-finite replacement at zero.
    for value in features.iter_mut() {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    features
}

/// Extract the 32 features from a histogram collected elsewhere.
///
/// This is [`extract_features`] without the pixel walk: the compute dispatch
/// A caller can fill the same `HISTOGRAM_LEN` bins by a walk of its own,
/// so everything after the walk — the order statistics, the moments, the hit
/// ratios — is the same function of them and the features are bit-identical.
///
/// `histogram` is borrowed and left as it was: the reduction takes the top
/// [`TOP_N`] samples out of the statistics without writing the bins, so the
/// caller's words are read where they lie.  `pixels` is how many samples the
/// bins hold — the frame's pixel count, which the walk's caller knows and
/// which the reduction cannot recover from a bound it has not found yet.
///
/// `debug` times this function's own work into [`OgsovStages::tail`], with
/// `scan` left at zero because no pixel walk happens here; [`predict`] times
/// the MLP afterwards and overwrites its field.  A `debug` run is slower by
/// construction, and the number it reports includes that cost.
///
/// # Errors
///
/// Fails when the bins do not cover [`HISTOGRAM_LEN`] samples, or when the
/// frame has fewer than [`MIN_PIXELS`] pixels — the two checks the pixel-walk
/// path makes before it starts walking.
pub fn features_from_histogram(
    scratch: &mut OgsovScratch,
    histogram: &[u32],
    pixels: u64,
    hits: u32,
    debug: bool,
) -> Result<[f32; FEATURE_COUNT]> {
    let started = debug.then(Instant::now);
    if histogram.len() < HISTOGRAM_LEN {
        return Err(Error::input(format!(
            "the collected histogram has {} bins, expected {HISTOGRAM_LEN}",
            histogram.len()
        )));
    }
    if pixels < MIN_PIXELS {
        return Err(Error::TooFewPixels {
            at_most: MIN_PIXELS - 1,
            actual: pixels,
        });
    }

    let features = scratch.reduce_collected(&histogram[..HISTOGRAM_LEN], pixels, hits);
    if let (Some(started), Some(finished)) = (started, debug.then(Instant::now)) {
        scratch.stages = OgsovStages {
            scan: Duration::ZERO,
            tail: finished - started,
            mlp: Duration::ZERO,
        };
    }
    Ok(features)
}

/// What one OGSOV prediction reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OgsovPrediction {
    /// Per-classifier sigmoid probabilities, in reference order.
    pub probabilities: [f32; OGSOV_CLASSIFIERS],
    /// The mean probability, which is OGSOV's raw score.
    pub score: f32,
    /// The predicted label: `score >= 0.5`.
    pub is_color: bool,
    /// The confidence percentage, as a float.
    ///
    /// The reference reports `confidence * 100` without truncating or
    /// rounding it, so a port that rounded to an integer would silently lose
    /// the distinction between 99.96% and 100%.
    pub confidence: f64,
    /// True when an exact-gray frame was answered without running the model.
    ///
    /// The verdict is certain (a gray frame is not a colour page) and no
    /// probabilities were computed, so `probabilities` and `score` are zero.
    /// This mirrors the host's gray fast path in the upstream plugin, where a
    /// gray frame carries `_MangaColorFastGray` and no score.
    pub fast_gray: bool,
}

impl OgsovPrediction {
    /// The verdict for a frame whose every pixel is exactly gray.
    #[must_use]
    pub fn fast_gray() -> Self {
        OgsovPrediction {
            probabilities: [0.0; OGSOV_CLASSIFIERS],
            score: 0.0,
            is_color: false,
            confidence: 100.0,
            fast_gray: true,
        }
    }
}

impl Default for OgsovPrediction {
    fn default() -> Self {
        Self::fast_gray()
    }
}

/// Run the five classifiers and average their probabilities.
///
/// `debug` times the stage this function owns and leaves it on `scratch`
/// ([`OgsovScratch::stages`]).
pub fn predict(
    model: &OgsovModel,
    features: &[f32; FEATURE_COUNT],
    scratch: &mut OgsovScratch,
    debug: bool,
) -> OgsovPrediction {
    debug_assert_eq!(model.classifiers.len(), OGSOV_CLASSIFIERS);
    let started = debug.then(Instant::now);
    let mut probabilities = [0.0f32; OGSOV_CLASSIFIERS];
    for (index, classifier) in model.classifiers.iter().enumerate() {
        math::dense_transposed(
            features,
            &classifier.w1_transposed,
            OGSOV_HIDDEN,
            OGSOV_INPUT,
            &classifier.b1,
            &mut scratch.hidden1,
        );
        math::tanh_in_place(&mut scratch.hidden1);

        math::dense_transposed(
            &scratch.hidden1,
            &classifier.w2_transposed,
            OGSOV_HIDDEN,
            OGSOV_HIDDEN,
            &classifier.b2,
            &mut scratch.hidden2,
        );
        math::tanh_in_place(&mut scratch.hidden2);

        scratch.ratio_bias[0] = classifier.b3;
        math::dense_transposed(
            &scratch.hidden2,
            &classifier.w3_transposed,
            1,
            OGSOV_HIDDEN,
            &scratch.ratio_bias,
            &mut scratch.logit,
        );
        probabilities[index] = math::sigmoid(scratch.logit[0]);
    }

    let mut total = 0.0f32;
    for probability in probabilities {
        total += probability;
    }
    let score = total / OGSOV_CLASSIFIERS as f32;
    if let Some(started) = started {
        scratch.stages.mlp = started.elapsed();
    }
    let is_color = score >= 0.5;
    let confidence_value = if is_color { score } else { 1.0 - score };
    OgsovPrediction {
        probabilities,
        score,
        is_color,
        // The reference widens the float32 tensor value to a Python float and
        // scales it by 100 in float64, keeping the fraction.
        confidence: confidence_value as f64 * 100.0,
        fast_gray: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rgb::PlaneView;

    #[test]
    fn the_largest_distance_is_the_documented_bound() {
        // For fixed R and G the `(R+G-2B)` term is extremal at B = 0 or 255.
        let mut maximum = 0usize;
        for red in 0..=255u8 {
            for green in 0..=255u8 {
                for blue in [0u8, 255] {
                    maximum = maximum.max(distance_squared(red, green, blue));
                }
            }
        }
        assert_eq!(maximum, MAX_D2);
    }

    #[test]
    fn the_value_table_starts_at_zero_and_saturates() {
        let table = value_table();
        assert_eq!(table[0], 0.0);
        assert_eq!(table.len(), HISTOGRAM_LEN);
        // sqrt(324900)/570 == 1, so everything from there on is clipped.
        assert_eq!(table[324_900], 1.0);
        assert_eq!(table[MAX_D2], 1.0);
        assert!(table[1] > 0.0 && table[1] < 1.0);
    }

    #[test]
    fn a_tiny_frame_is_rejected() {
        let red = [0u8; 17];
        let green = [0u8; 17];
        let blue = [0u8; 17];
        let view = RgbView::new(
            PlaneView::new(&red, 17, 17, 1).expect("r"),
            PlaneView::new(&green, 17, 17, 1).expect("g"),
            PlaneView::new(&blue, 17, 17, 1).expect("b"),
        )
        .expect("view");
        let table = LookupTable::from_bool_slice(&vec![false; crate::lookup::MASK_ELEMENTS], 256);
        let error = extract_features(&view, &table, &mut OgsovScratch::new(), false, false)
            .expect_err("must reject");
        assert!(matches!(error, Error::TooFewPixels { .. }), "{error}");
    }

    /// A reused workspace must not carry the previous frame's histogram bound
    /// into a frame that reaches fewer bins.
    #[test]
    fn reuse_across_frames_gives_the_same_features_as_a_fresh_workspace() {
        let table = LookupTable::from_bool_slice(&vec![false; crate::lookup::MASK_ELEMENTS], 256);
        // Wide: alternating red and blue, so `d2` reaches 260,100.
        let mut wide = [[0u8; 32]; 3];
        // Narrow: exact gray and one step away from it, so `d2` stays below 8.
        let mut narrow = [[0u8; 32]; 3];
        for index in 0..32 {
            let (r, g, b) = if index % 2 == 0 {
                (255, 0, 0)
            } else {
                (0, 0, 255)
            };
            wide[0][index] = r;
            wide[1][index] = g;
            wide[2][index] = b;
            let (r, g, b) = if index % 2 == 0 {
                (128, 128, 128)
            } else {
                (129, 128, 128)
            };
            narrow[0][index] = r;
            narrow[1][index] = g;
            narrow[2][index] = b;
        }
        fn view(planes: &[[u8; 32]; 3]) -> RgbView<'_> {
            RgbView::new(
                PlaneView::new(&planes[0], 32, 32, 1).expect("r"),
                PlaneView::new(&planes[1], 32, 32, 1).expect("g"),
                PlaneView::new(&planes[2], 32, 32, 1).expect("b"),
            )
            .expect("view")
        }

        let mut reused = OgsovScratch::new();
        let first =
            extract_features(&view(&wide), &table, &mut reused, false, false).expect("wide");
        let second =
            extract_features(&view(&narrow), &table, &mut reused, false, false).expect("narrow");
        let fresh = extract_features(
            &view(&narrow),
            &table,
            &mut OgsovScratch::new(),
            false,
            false,
        )
        .expect("narrow, fresh");
        assert_eq!(second, fresh, "a reused workspace changed an answer");
        // The two frames really do reach different bins: feature 3 is ten times
        // the largest chroma in the frame.
        assert!(
            first[3] > second[3],
            "the wide frame should reach a larger distance: {} vs {}",
            first[3],
            second[3]
        );
    }

    /// The collected path does not write the workspace's own bins, so it must
    /// not narrow the bound the next scan clears up to: a count an earlier
    /// scan left above that bound would be added to the next frame's.
    #[test]
    fn a_collected_reduction_does_not_narrow_the_clear_bound() {
        let table = LookupTable::from_bool_slice(&vec![false; crate::lookup::MASK_ELEMENTS], 256);
        // One mid-range chroma and 31 exact grays: the frame reaches bin 500,
        // far below the wide frame's 325,125.
        let mid = |index: usize| {
            if index == 0 {
                (10u8, 0u8, 0u8)
            } else {
                (0, 0, 0)
            }
        };
        let narrow = [
            std::array::from_fn(|index| mid(index).0),
            std::array::from_fn(|index| mid(index).1),
            std::array::from_fn(|index| mid(index).2),
        ];
        let wide: [[u8; 32]; 3] = [
            std::array::from_fn(|index| if index % 2 == 0 { 255 } else { 0 }),
            std::array::from_fn(|_| 0),
            std::array::from_fn(|index| if index % 2 == 0 { 0 } else { 255 }),
        ];
        fn view(planes: &[[u8; 32]; 3]) -> RgbView<'_> {
            RgbView::new(
                PlaneView::new(&planes[0], 32, 32, 1).expect("r"),
                PlaneView::new(&planes[1], 32, 32, 1).expect("g"),
                PlaneView::new(&planes[2], 32, 32, 1).expect("b"),
            )
            .expect("view")
        }

        // A wide frame first, so the workspace's own bins hold counts above
        // the distance the narrow frame reaches.
        let mut reused = OgsovScratch::new();
        extract_features(&view(&wide), &table, &mut reused, false, false).expect("wide");

        // A collected histogram whose largest bin is bin 0: the reduce must
        // not tell the next scan to clear only up to it.
        let mut collected = vec![0u32; HISTOGRAM_LEN];
        collected[0] = 32;
        features_from_histogram(&mut reused, &collected, 32, 0, false).expect("collected");

        let reused_features =
            extract_features(&view(&narrow), &table, &mut reused, false, false).expect("narrow");
        let fresh_features = extract_features(
            &view(&narrow),
            &table,
            &mut OgsovScratch::new(),
            false,
            false,
        )
        .expect("narrow, fresh");
        assert_eq!(
            reused_features, fresh_features,
            "a collected reduction narrowed the clear bound"
        );
    }

    /// A caller can hand the reduction a histogram it did not scan itself,
    /// and the hit count in uncapped where the pixel walk stops at the cap.
    /// Both have to reach the features the scan reaches.
    #[test]
    fn a_collected_histogram_gives_the_features_the_scan_gives() {
        let table = LookupTable::from_bool_slice(&vec![false; crate::lookup::MASK_ELEMENTS], 256);
        let mut planes = [[0u8; 32]; 3];
        for (index, plane) in planes.iter_mut().enumerate() {
            plane[0] = (index * 7) as u8;
            plane[1] = (index * 3) as u8;
            plane[2] = (index * 11) as u8;
        }
        let view = RgbView::new(
            PlaneView::new(&planes[0], 32, 32, 1).expect("r"),
            PlaneView::new(&planes[1], 32, 32, 1).expect("g"),
            PlaneView::new(&planes[2], 32, 32, 1).expect("b"),
        )
        .expect("view");
        let pixels = view.pixel_count().expect("pixels");
        let scanned =
            extract_features(&view, &table, &mut OgsovScratch::new(), false, false).expect("scan");

        // The same histogram a dispatch would leave: exact `u32` counts, read
        // back into a workspace the reduction owns.
        let mut collected = OgsovScratch::new();
        let hits = simd::scan(
            &view,
            &mut collected.histogram,
            &table,
            table.spread_threshold(false),
        );
        let bins = collected.histogram.clone();
        let collected =
            features_from_histogram(&mut OgsovScratch::new(), &bins, pixels, hits, false)
                .expect("reduce");
        assert_eq!(
            collected, scanned,
            "the collected histogram moved the features"
        );

        // The scalar loop stops counting at the cap; a collector may count on.
        let capped = features_from_histogram(
            &mut OgsovScratch::new(),
            &bins,
            pixels,
            hits.min(MASK_HIT_CAP),
            false,
        )
        .expect("capped");
        assert_eq!(capped, scanned, "the cap changed the features");

        // The collected path times itself: no scan ran on this side, and the
        // number the debug line reports as `tail=` is the reduction rather
        // than a zero standing in for work that did happen.
        let mut timed = OgsovScratch::new();
        features_from_histogram(&mut timed, &bins, pixels, hits, true).expect("timed");
        let stages = timed.stages();
        assert_eq!(stages.scan, Duration::ZERO, "{stages:?}");
        assert!(stages.tail > Duration::ZERO, "{stages:?}");
        assert_eq!(stages.mlp, Duration::ZERO, "{stages:?}");
    }

    /// A table with one true entry, so the shortcut has something to skip.
    fn one_hit_table() -> LookupTable {
        let mut values = vec![false; crate::lookup::MASK_ELEMENTS];
        // One hit at `(b, g, r) = (3, 2, 1)`, whose spread of 2 is the
        // threshold the table then carries.
        values[(3 << 16) | (2 << 8) | 1] = true;
        LookupTable::from_bool_slice(&values, 2)
    }

    /// Thirty-two pixels of the four kinds that make the shortcut visible:
    /// exactly gray, just inside the threshold, the single hit, and something
    /// that cannot be one.
    fn option_view() -> ([[u8; 32]; 3], LookupTable) {
        let pixel = |index: usize| match index % 4 {
            0 => (128u8, 128u8, 128u8),
            1 => (1, 2, 3),
            2 => (7, 7, 6),
            _ => (240, 30, 200),
        };
        let planes = [
            std::array::from_fn(|index| pixel(index).0),
            std::array::from_fn(|index| pixel(index).1),
            std::array::from_fn(|index| pixel(index).2),
        ];
        (planes, one_hit_table())
    }

    fn option_rgb_view(planes: &[[u8; 32]; 3]) -> RgbView<'_> {
        RgbView::new(
            PlaneView::new(&planes[0], 32, 32, 1).expect("r"),
            PlaneView::new(&planes[1], 32, 32, 1).expect("g"),
            PlaneView::new(&planes[2], 32, 32, 1).expect("b"),
        )
        .expect("view")
    }

    /// Probing the whole frame cannot change an answer.
    #[test]
    fn checking_the_whole_frame_gives_the_same_features() {
        let (planes, table) = option_view();
        assert_eq!(table.min_spread(), 2);
        // The fixture has to hold pixels the shortcut skips, or this proves
        // nothing at all.
        let skipped = (0..32)
            .filter(|index| {
                let (red, green, blue) = (planes[0][*index], planes[1][*index], planes[2][*index]);
                let high = red.max(green).max(blue) as i32;
                let low = red.min(green).min(blue) as i32;
                high - low < table.min_spread() as i32
            })
            .count();
        assert!(skipped > 0, "the fixture must hold skipped pixels");

        let view = option_rgb_view(&planes);
        let shortcut = extract_features(&view, &table, &mut OgsovScratch::new(), false, false)
            .expect("shortcut");
        let whole =
            extract_features(&view, &table, &mut OgsovScratch::new(), true, false).expect("whole");
        assert_eq!(
            shortcut, whole,
            "checking the whole frame changed a feature"
        );
    }

    /// `debug` times the stages and changes nothing else.
    #[test]
    fn debug_fills_the_stages_and_nothing_else() {
        let (planes, table) = option_view();
        let view = option_rgb_view(&planes);

        let mut quiet = OgsovScratch::new();
        let unwatched = extract_features(&view, &table, &mut quiet, false, false).expect("quiet");
        assert_eq!(
            quiet.stages(),
            OgsovStages::default(),
            "a run that was not asked for timings measured some"
        );

        let mut watched = OgsovScratch::new();
        let timed = extract_features(&view, &table, &mut watched, false, true).expect("watched");
        assert_eq!(unwatched, timed, "debug changed a feature");
        let stages = watched.stages();
        assert!(stages.scan > Duration::ZERO, "{stages:?}");
        assert!(stages.tail > Duration::ZERO, "{stages:?}");
        assert_eq!(stages.mlp, Duration::ZERO, "the MLP has not run yet");
    }

    #[test]
    fn a_gray_frame_has_no_chroma() {
        // 32 exactly-gray pixels: every d2 is zero, so the top 17 values and
        // every statistic derived from the remainder are zero too.
        let plane = [128u8; 32];
        let view = RgbView::new(
            PlaneView::new(&plane, 32, 32, 1).expect("r"),
            PlaneView::new(&plane, 32, 32, 1).expect("g"),
            PlaneView::new(&plane, 32, 32, 1).expect("b"),
        )
        .expect("view");
        let table = LookupTable::from_bool_slice(&vec![false; crate::lookup::MASK_ELEMENTS], 256);
        let features = extract_features(&view, &table, &mut OgsovScratch::new(), false, false)
            .expect("features");
        assert!(features.iter().all(|value| *value == 0.0), "{features:?}");
    }
}

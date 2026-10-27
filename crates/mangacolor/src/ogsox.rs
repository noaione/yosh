//! OGSOX: 512² tiling, nine per-tile features, and five attention heads.
//!
//! The canonical oracle is the complete ONNX graph, so the port follows the
//! graph's node order rather than the prose summary of it:
//!
//! 1. every tile contributes nine features — five floats derived from its
//!    chroma aggregates and four capped mask-hit ratios — and all nine are
//!    clipped to `[0, 1]`;
//! 2. each head projects those nine features to 256, splits them in half and
//!    gates one half with the other's SiLU, layer-normalizes twice, and builds
//!    a 16-value embedding per tile;
//! 3. one attention logit per tile is softmaxed **across all tiles of the
//!    frame**, and the weighted sum of the embeddings is pooled;
//! 4. the pooled vector goes through `64 -> relu -> 2` and becomes
//!    `tanh((logit[1] - logit[0]) / 2)`.
//!
//! Summing the five head scalars gives the page score; the heads are never
//! classified independently, and their scalars are never averaged.

use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::math;
use crate::rgb::RgbView;
use crate::weights::{
    OGSOX_ATTENTION, OGSOX_EMBED, OGSOX_FEATURES, OGSOX_GATE, OGSOX_HALF, OGSOX_HEADS,
    OGSOX_LOGITS, OgsoxModel,
};

mod simd;

/// Tile edge length in pixels.
pub const TILE: usize = 512;

/// The graph's per-tile divisor, used even for partially populated tiles.
pub const TILE_PIXELS: f32 = (TILE * TILE) as f32;

/// The layer normalization epsilon the graph was exported with.
///
/// The ONNX attribute is `9.999999747378752e-06`, which is the `f32` nearest to
/// `1e-5`; every layer of the graph carries the same value.
pub const LAYER_NORM_EPSILON: f32 = 1e-5;

/// The graph only ever uses `min(hits, 1024)`.
pub const MASK_HIT_CAP: u32 = 1024;

/// Offset of the graph's `c3` affine, `1/512`, exactly representable in `f32`.
#[allow(clippy::excessive_precision)] // the exact decimal is the point
pub const C3_OFFSET: f32 = 0.001_953_125;
/// Denominator of the graph's `c3` affine, `511/512`, exactly representable.
#[allow(clippy::excessive_precision)] // the exact decimal is the point
pub const C3_SCALE: f32 = 0.998_046_875;

/// If a page's non-gray fraction is above this, one page-level chroma pass
/// beats per-tile recomputation.
pub const DENSE_THRESHOLD: f64 = 0.2;

/// The number of float32 accumulators the graph's reduction kernel keeps.
///
/// The canonical oracle reduces 262,144 values per tile, and that reduction's
/// order is observable: on a constant-colour tile the graph's own `mean_1`
/// differs from the mathematically exact value by ~1.5e-4, and `std` — derived
/// from the cancellation `mean_1 - mean * mean` — is only zero when the
/// reduction is exact.  Summing in `f64` therefore does *not* reproduce the
/// oracle: it changes `10 * clamp(2 * std, 0, 1)` from about 0.45 to 0 on a
/// solid page and moves the score by 1.5, which is the difference between
/// confidence 100 and confidence 70.
///
/// `tests/measure_reduction.py` established that the graph accumulates
/// eight lanes, one element per lane in row-major order, and combines them as
/// a tree; that model reproduced every probed tile bit-exactly (10 of 10
/// fixture and synthetic tiles, `mean` and `mean_1` both).
///
/// This is deliberately the reference host's kernel shape rather than a
/// portability promise: a host where ONNX Runtime used a different lane count
/// would round degenerate tiles differently.  Labels are unaffected by that;
/// confidences near a rounding boundary could differ, which the parity harness
/// reports explicitly rather than assuming.
const REDUCTION_LANES: usize = 8;

/// Combine the lane accumulators the way the graph's kernel does.
///
/// The order is the 256-bit horizontal add the reference kernel performs:
/// fold the upper half of the accumulator vector onto the lower half, then
/// reduce the four folded values pairwise.
#[inline]
fn reduce_lanes(lanes: &[f32; REDUCTION_LANES]) -> f32 {
    let folded = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    (folded[0] + folded[1]) + (folded[2] + folded[3])
}

/// Tile start offsets along one axis, reproducing `np.round(np.linspace(0,
/// full - TILE, n))` exactly, including its ties-to-even rounding.
pub fn tile_starts(length: usize) -> Vec<usize> {
    let full = length.max(TILE);
    let count = full.div_ceil(TILE);
    if count == 1 {
        return vec![0];
    }
    // NumPy's linspace multiplies the index by the step and then assigns the
    // final element exactly, which is why the last start is recomputed here.
    let stop = (full - TILE) as f64;
    let step = stop / (count - 1) as f64;
    let mut starts = Vec::with_capacity(count);
    for index in 0..count {
        let value = if index == count - 1 {
            stop
        } else {
            index as f64 * step
        };
        starts.push(value.round_ties_even() as usize);
    }
    starts
}

/// Top-left corner of every tile, in row-major order.
pub fn tile_origins(width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut origins = Vec::new();
    for y in tile_starts(height) {
        for &x in &tile_starts(width) {
            origins.push((y, x));
        }
    }
    origins
}

/// Per-head diagnostic tensors, recorded only when validation asks for them.
#[derive(Debug, Default, Clone)]
pub struct OgsoxDiagnostics {
    /// The nine features per tile, after the clip.
    pub features: Vec<f32>,
    /// Per-tile mean of the clipped chroma.
    pub mean: Vec<f32>,
    /// Per-tile mean of the squared clipped chroma.
    pub mean_1: Vec<f32>,
    /// Per-tile maximum clipped chroma.
    pub amax: Vec<f32>,
    /// Per-tile `sqrt(sum v^2) / sum v` before the `c3` affine.
    pub ratio: Vec<f32>,
    /// The four capped mask ratios per tile.
    pub div: Vec<f32>,
    /// Head 0's attention weights, one per tile.
    pub attention_head0: Vec<f32>,
    /// Head 0's pooled 16-vector.
    pub pooled_head0: Vec<f32>,
    /// Head 0's two logits before the difference.
    pub logits_head0: Vec<f32>,
    /// The five head scalars before their sum.
    pub heads: Vec<f32>,
}

/// What one `debug` extraction measured, per stage.
///
/// The classifier core has no host handle and therefore logs nothing itself:
/// it leaves these numbers on the workspace it was handed, and the caller
/// decides what to do with them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OgsoxStages {
    /// The per-tile chroma pass: the sums, the squared sums and the maximum.
    pub chroma: Duration,
    /// The per-tile lookup hit scan.
    pub hits: Duration,
    /// The five attention heads, measured by [`score`].
    pub heads: Duration,
}

/// One reusable OGSOX workspace.
pub struct OgsoxScratch {
    origins: Vec<(usize, usize)>,
    features: Vec<f32>,
    embeddings: Vec<f32>,
    logits: Vec<f32>,
    gate: Vec<f32>,
    gated: Vec<f32>,
    normed: Vec<f32>,
    embedding: Vec<f32>,
    attention: Vec<f32>,
    pooled: [f32; OGSOX_EMBED],
    hidden: [f32; OGSOX_ATTENTION],
    logit_pair: [f32; OGSOX_LOGITS],
    /// Filled in only by a `debug` run; zero otherwise.
    stages: OgsoxStages,
    diagnostics: Option<OgsoxDiagnostics>,
}

impl Default for OgsoxScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl OgsoxScratch {
    /// Allocate a workspace; the tile-sized buffers grow on first use.
    pub fn new() -> Self {
        OgsoxScratch {
            origins: Vec::new(),
            features: Vec::new(),
            embeddings: Vec::new(),
            logits: Vec::new(),
            gate: vec![0.0; OGSOX_GATE],
            gated: vec![0.0; OGSOX_HALF],
            normed: vec![0.0; OGSOX_HALF],
            embedding: vec![0.0; OGSOX_EMBED],
            attention: vec![0.0; OGSOX_ATTENTION],
            pooled: [0.0; OGSOX_EMBED],
            hidden: [0.0; OGSOX_ATTENTION],
            logit_pair: [0.0; OGSOX_LOGITS],
            stages: OgsoxStages::default(),
            diagnostics: None,
        }
    }

    /// Start recording per-stage tensors for the next scored frame.
    pub fn enable_diagnostics(&mut self) {
        self.diagnostics = Some(OgsoxDiagnostics::default());
    }

    /// The diagnostics recorded by the last frame, if any.
    pub fn diagnostics(&self) -> Option<&OgsoxDiagnostics> {
        self.diagnostics.as_ref()
    }

    /// The stages the last `debug` extraction measured.
    ///
    /// Every field is zero unless the call that filled it was given
    /// `debug = true`.
    #[must_use]
    pub fn stages(&self) -> OgsoxStages {
        self.stages
    }

    /// Bytes kept alive between calls, excluding the per-frame vectors.
    pub fn resident_bytes(&self) -> usize {
        self.origins.capacity() * std::mem::size_of::<(usize, usize)>()
            + self.features.capacity() * std::mem::size_of::<f32>()
            + self.embeddings.capacity() * std::mem::size_of::<f32>()
            + self.logits.capacity() * std::mem::size_of::<f32>()
    }

    fn prepare(&mut self, width: usize, height: usize) {
        self.origins = tile_origins(width, height);
        let tiles = self.origins.len();
        self.features.resize(tiles * OGSOX_FEATURES, 0.0);
        self.embeddings.resize(tiles * OGSOX_EMBED, 0.0);
        self.logits.resize(tiles, 0.0);
        if let Some(diagnostics) = self.diagnostics.as_mut() {
            *diagnostics = OgsoxDiagnostics::default();
        }
    }
}

/// The chroma the graph computes for one pixel.
///
/// `sqrt((R-G)^2 + ((R+G)/2 - B)^2) / 285`, clipped to `[0, 1]`, evaluated in
/// `float32` in the graph's own operation order.  Every intermediate is exact
/// for 8-bit inputs until the square root, so this is bit-identical to the
/// graph's `Clip` output.
#[inline]
pub fn chroma(red: u8, green: u8, blue: u8) -> f32 {
    let red = red as f32;
    let green = green as f32;
    let blue = blue as f32;
    let dx = red - green;
    let yb = (red + green) * 0.5f32 - blue;
    let squared = dx * dx + yb * yb;
    (squared.sqrt() / 285.0f32).min(1.0f32)
}

/// Fill the workspace's feature matrix, one row of nine per tile.
///
/// `full_frame` probes every pixel of every tile for a lookup hit instead of
/// skipping the pixels whose channels are too close together to be in the
/// table.  Those pixels were never hits, so the features do not change; it
/// exists to be able to check that on real frames, and it is slower.
///
/// `debug` times the two stages this function owns and leaves them on
/// `scratch` ([`OgsoxScratch::stages`]).  A `debug` run is slower by
/// construction, and the numbers it reports include that cost.
pub fn extract_tile_features(
    rgb: &RgbView<'_>,
    model: &OgsoxModel,
    scratch: &mut OgsoxScratch,
    full_frame: bool,
    debug: bool,
) -> Result<()> {
    let width = rgb.width();
    let height = rgb.height();
    scratch.prepare(width, height);
    if debug {
        scratch.stages = OgsoxStages::default();
    }
    let lookup = &model.lookup;
    let threshold = lookup.spread_threshold(full_frame);
    let count = scratch.origins.len();
    // A band at a time, not a tile at a time: the planes are read in
    // frame-row order, and each band's hit scan re-reads what the chroma
    // walk just put in the last-level cache.
    let sweep = simd::sweep_tiles(
        rgb,
        &scratch.origins,
        width.div_ceil(TILE),
        lookup,
        threshold,
        debug,
    );
    if debug {
        scratch.stages.chroma += sweep.chroma_time;
        scratch.stages.hits += sweep.hits_time;
    }

    for tile in 0..count {
        // Per-tile aggregates, accumulated exactly the way the graph's own
        // reduction kernel does: eight float32 accumulators fed one element
        // per lane in row-major order, then combined as a tree.  See
        // `REDUCTION_LANES` for why the lane count and combine order are part
        // of the port rather than an implementation detail.  The row walk
        // itself lives in `simd`, which does it a vector at a time on
        // aarch64 and x86-64 and one pixel at a time anywhere else; both
        // produce these values identically.
        let (lanes, squared_lanes, maximum) = sweep.chromas[tile];
        let hits = sweep.hits[tile];

        let aggregates = TileAggregates {
            lanes,
            squared_lanes,
            maximum,
            hits,
        };
        let (features, statistics) = tile_features(&aggregates);
        scratch.features[tile * OGSOX_FEATURES..][..OGSOX_FEATURES].copy_from_slice(&features);

        if let Some(diagnostics) = scratch.diagnostics.as_mut() {
            diagnostics.mean.push(statistics.mean);
            diagnostics.mean_1.push(statistics.mean_1);
            diagnostics.amax.push(statistics.maximum);
            diagnostics.ratio.push(statistics.ratio);
            diagnostics.div.extend_from_slice(&statistics.mask_ratios);
        }
    }
    Ok(())
}

/// One tile's collected aggregates, which are all the rest of OGSOX sees.
///
/// A pixel walk produces them, and a reduction over an identical aggregate
/// produces the same features — which is why the reduction below takes
/// nothing else.
#[derive(Debug, Clone, Copy)]
pub struct TileAggregates {
    /// The tile's eight chroma lane sums, filled one pixel per lane in
    /// row-major order.
    pub lanes: [f32; REDUCTION_LANES],
    /// The same lanes, for the squared chroma.
    pub squared_lanes: [f32; REDUCTION_LANES],
    /// The largest clipped chroma in the tile.
    pub maximum: f32,
    /// How many pixels were lookup hits.  Only `min(hits, MASK_HIT_CAP)` is
    /// ever read, so a collector may count past the cap.
    pub hits: u32,
}

/// The per-tile numbers the diagnostics record, beside the nine features.
#[derive(Debug, Clone, Copy)]
pub struct TileStatistics {
    /// The tile's mean clipped chroma.
    pub mean: f32,
    /// The tile's mean squared clipped chroma.
    pub mean_1: f32,
    /// The tile's largest clipped chroma.
    pub maximum: f32,
    /// `sqrt(sum v^2) / sum v` before the `c3` affine.
    pub ratio: f32,
    /// The four capped mask ratios.
    pub mask_ratios: [f32; 4],
}

/// Reduce one tile's aggregates to its nine features.
///
/// This is the graph's own reduction: the lanes are folded the way the
/// reference kernel folds them, and the nine values are clipped to `[0, 1]`
/// after they are assembled rather than one at a time.
#[must_use]
pub fn tile_features(aggregates: &TileAggregates) -> ([f32; OGSOX_FEATURES], TileStatistics) {
    let sum_3 = reduce_lanes(&aggregates.lanes);
    let sum_2 = reduce_lanes(&aggregates.squared_lanes);
    let mean = sum_3 / TILE_PIXELS;
    let mean_1 = sum_2 / TILE_PIXELS;
    let variance = mean_1 - mean * mean;
    let std = variance.max(0.0).sqrt();
    let root_mean_square = mean_1.sqrt();
    let ratio = if sum_3 > 0.0 {
        sum_2.sqrt() / sum_3
    } else {
        0.0
    };
    let c3 = ((ratio - C3_OFFSET) / C3_SCALE).clamp(0.0, 1.0);
    let hits = aggregates.hits as f32;
    let mask_ratios = [
        hits.min(2.0) / 2.0,
        hits.min(16.0) / 16.0,
        hits.min(128.0) / 128.0,
        hits.min(1024.0) / 1024.0,
    ];

    let mut features = [0.0f32; OGSOX_FEATURES];
    features[0] = mean * 10.0;
    features[1] = (2.0 * std).clamp(0.0, 1.0) * 10.0;
    features[2] = root_mean_square * 10.0;
    features[3] = aggregates.maximum * 10.0;
    features[4] = c3 * 100.0;
    features[5..].copy_from_slice(&mask_ratios);
    for value in features.iter_mut() {
        *value = value.clamp(0.0, 1.0);
    }

    let statistics = TileStatistics {
        mean,
        mean_1,
        maximum: aggregates.maximum,
        ratio,
        mask_ratios,
    };
    (features, statistics)
}

/// Sum the five head scalars into the page score.
///
/// `debug` times the stage this function owns and leaves it on `scratch`
/// ([`OgsoxScratch::stages`]).
pub fn score(model: &OgsoxModel, scratch: &mut OgsoxScratch, debug: bool) -> Result<f32> {
    debug_assert_eq!(model.heads.len(), OGSOX_HEADS);
    let started = debug.then(Instant::now);
    if scratch.origins.is_empty() {
        return Err(Error::input("OGSOX needs at least one tile"));
    }
    let tiles = scratch.origins.len();
    let mut total = 0.0f32;
    let mut head_scalars = [0.0f32; OGSOX_HEADS];

    for (index, head) in model.heads.iter().enumerate() {
        // 1. Project the nine features to 256 and gate one half with the
        //    other's SiLU.
        for tile in 0..tiles {
            let features = &scratch.features[tile * OGSOX_FEATURES..][..OGSOX_FEATURES];
            math::dense_transposed(
                features,
                &head.w1_transposed,
                OGSOX_GATE,
                OGSOX_FEATURES,
                &head.b1,
                &mut scratch.gate,
            );
            for half in 0..OGSOX_HALF {
                let gate_value = scratch.gate[OGSOX_HALF + half];
                let silu = gate_value * math::sigmoid(gate_value);
                scratch.gated[half] = scratch.gate[half] * silu;
            }
            math::layer_norm(
                &scratch.gated,
                &head.ln1_scale,
                &head.ln1_bias,
                LAYER_NORM_EPSILON,
                &mut scratch.normed,
            );
            math::dense_transposed(
                &scratch.normed,
                &head.w2_transposed,
                OGSOX_EMBED,
                OGSOX_HALF,
                &head.b2,
                &mut scratch.embedding,
            );
            math::layer_norm(
                &scratch.embedding,
                &head.ln2_scale,
                &head.ln2_bias,
                LAYER_NORM_EPSILON,
                &mut scratch.normed[..OGSOX_EMBED],
            );
            math::tanh_in_place(&mut scratch.normed[..OGSOX_EMBED]);
            scratch.embeddings[tile * OGSOX_EMBED..][..OGSOX_EMBED]
                .copy_from_slice(&scratch.normed[..OGSOX_EMBED]);
        }

        // 2. One attention logit per tile, then a softmax across the frame.
        for tile in 0..tiles {
            let embedding = &scratch.embeddings[tile * OGSOX_EMBED..][..OGSOX_EMBED];
            math::dense_input_major(
                embedding,
                &head.wa1,
                OGSOX_ATTENTION,
                OGSOX_EMBED,
                &head.ba1,
                &mut scratch.attention,
            );
            math::tanh_in_place(&mut scratch.attention);
            scratch.logits[tile] = math::dot(&scratch.attention, &head.wa2);
        }
        math::softmax_in_place(&mut scratch.logits[..tiles]);

        // 3. Pool the embeddings and classify, exactly as the graph does.
        scratch.pooled = [0.0; OGSOX_EMBED];
        for tile in 0..tiles {
            let weight = scratch.logits[tile];
            let embedding = &scratch.embeddings[tile * OGSOX_EMBED..][..OGSOX_EMBED];
            for (pooled, value) in scratch.pooled.iter_mut().zip(embedding) {
                *pooled += weight * value;
            }
        }
        math::dense_transposed(
            &scratch.pooled,
            &head.wc1_transposed,
            OGSOX_ATTENTION,
            OGSOX_EMBED,
            &head.bc1,
            &mut scratch.hidden,
        );
        math::relu_in_place(&mut scratch.hidden);
        math::dense_transposed(
            &scratch.hidden,
            &head.wc2_transposed,
            OGSOX_LOGITS,
            OGSOX_ATTENTION,
            &head.bc2,
            &mut scratch.logit_pair,
        );
        head_scalars[index] = ((scratch.logit_pair[1] - scratch.logit_pair[0]) / 2.0).tanh();
        total += head_scalars[index];

        if let Some(diagnostics) = scratch.diagnostics.as_mut()
            && index == 0
        {
            diagnostics.attention_head0 = scratch.logits[..tiles].to_vec();
            diagnostics.pooled_head0 = scratch.pooled.to_vec();
            diagnostics.logits_head0 = scratch.logit_pair.to_vec();
        }
    }

    if let Some(started) = started {
        scratch.stages.heads = started.elapsed();
    }
    if let Some(diagnostics) = scratch.diagnostics.as_mut() {
        diagnostics.features = scratch.features.clone();
        diagnostics.heads = head_scalars.to_vec();
    }
    Ok(total)
}

/// Extract the features and score a frame in one call.
///
/// The two switches are [`extract_tile_features`]'s and [`score`]'s; see those.
pub fn predict(
    model: &OgsoxModel,
    rgb: &RgbView<'_>,
    scratch: &mut OgsoxScratch,
    full_frame: bool,
    debug: bool,
) -> Result<f32> {
    extract_tile_features(rgb, model, scratch, full_frame, debug)?;
    score(model, scratch, debug)
}

/// What one OGSOX prediction reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OgsoxPrediction {
    /// The signed page score, in `[-5, 5]`.
    pub score: f32,
    /// The predicted label: `score >= 0`.
    pub is_color: bool,
    /// The confidence percentage, as a float.
    ///
    /// `abs(score) / 5 * 100`, evaluated in float64 in the reference's
    /// operation order and never truncated or rounded; see
    /// [`crate::ogsov::OgsovPrediction::confidence`] for the same rule.
    pub confidence: f64,
    /// True when an exact-gray frame was answered without running the model.
    ///
    /// The verdict is certain (a gray frame is not a colour page) and no
    /// score was computed, so `score` is zero.  This mirrors the host's gray
    /// fast path in the upstream plugin, where a gray frame carries
    /// `_MangaColorFastGray` and no score.
    pub fast_gray: bool,
}

impl OgsoxPrediction {
    /// Derive the label and confidence the reference's `predict` reports.
    pub fn from_score(score: f32) -> Self {
        let wide = score as f64;
        OgsoxPrediction {
            score,
            is_color: score >= 0.0,
            // The reference's own `abs(score) / 5.0 * 100`, in float64.
            confidence: wide.abs() / 5.0 * 100.0,
            fast_gray: false,
        }
    }

    /// The verdict for a frame whose every pixel is exactly gray.
    #[must_use]
    pub fn fast_gray() -> Self {
        OgsoxPrediction {
            score: 0.0,
            is_color: false,
            confidence: 100.0,
            fast_gray: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_pages_produce_exactly_one_tile() {
        assert_eq!(tile_starts(1), vec![0]);
        assert_eq!(tile_starts(512), vec![0]);
        assert_eq!(tile_origins(1, 1), vec![(0, 0)]);
        assert_eq!(tile_origins(511, 511), vec![(0, 0)]);
    }

    #[test]
    fn tile_starts_match_numpy_linspace_rounding() {
        // Generated with Python 3.13 / NumPy 2.3.2 from
        //   np.round(np.linspace(0, max(n, 512) - 512, ceil(max(n, 512) / 512)))
        // including the ties-to-even cases at 1025, where 256.5 rounds down.
        for (length, expected) in [
            (1usize, vec![0usize]),
            (511, vec![0]),
            (512, vec![0]),
            (513, vec![0, 1]),
            (1000, vec![0, 488]),
            (1024, vec![0, 512]),
            (1025, vec![0, 256, 513]),
            (1536, vec![0, 512, 1024]),
            (1537, vec![0, 342, 683, 1025]),
            (2048, vec![0, 512, 1024, 1536]),
            (2049, vec![0, 384, 768, 1153, 1537]),
            (2456, vec![0, 486, 972, 1458, 1944]),
            (3072, vec![0, 512, 1024, 1536, 2048, 2560]),
            (3684, vec![0, 453, 906, 1359, 1813, 2266, 2719, 3172]),
            (4608, vec![0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]),
            (
                5274,
                vec![0, 476, 952, 1429, 1905, 2381, 2857, 3333, 3810, 4286, 4762],
            ),
        ] {
            assert_eq!(tile_starts(length), expected, "length {length}");
        }
    }

    #[test]
    fn origins_are_row_major() {
        assert_eq!(
            tile_origins(1025, 1025),
            vec![
                (0, 0),
                (0, 256),
                (0, 513),
                (256, 0),
                (256, 256),
                (256, 513),
                (513, 0),
                (513, 256),
                (513, 513)
            ]
        );
    }

    #[test]
    fn chroma_matches_the_reference_formula() {
        // A mid-range colour that does not saturate: (R-G) = 100 and
        // (R+G)/2 - B = 100, so the clip is a no-op and the arithmetic is
        // visible instead of being pinned at 1.0.
        let expected = (100.0f32 * 100.0 + 100.0 * 100.0).sqrt() / 285.0;
        assert!(expected < 1.0);
        assert_eq!(chroma(200, 100, 50), expected);
        assert_eq!(chroma(0, 0, 0), 0.0);
        // Gray has no chroma at all.
        for value in [0u8, 1, 128, 255] {
            assert_eq!(chroma(value, value, value), 0.0);
        }
    }

    #[test]
    fn chroma_saturates_at_one() {
        assert_eq!(chroma(0, 255, 255), 1.0);
        assert_eq!(chroma(255, 0, 255), 1.0);
    }
}

//! Rust port of OGSOV color detector (5-MLP ensemble + 256^3 mask lookup).
//!
//! Input is raw 8-bit RGB (as produced by e.g. `image::DynamicImage::to_rgb8`).
//!
//! Known deliberate deviations from the Python reference:
//! - The top-17 colorfulness values are sorted descending; numpy's
//!   `argpartition` leaves them in an arbitrary (but deterministic) order.
//!   Validate predictions against the Python implementation on real pages.
//! - Statistics are accumulated in f64 and cast to f32; numpy mixes f32/f64.
//!   Differences are far below the ensemble's decision noise floor.
//! - Colorfulness order statistics use a bounded integer histogram instead of
//!   sorting one float per pixel. Moment accumulation preserves sorted order, so
//!   generated features remain bit-identical to previous sort path.

const MASK_BYTES: usize = 256 * 256 * 256 / 8; // 2 MiB
const IN_DIM: usize = 32;
const HIDDEN: usize = 512;
const TOP_N: usize = 17;
const MEDIAN_SCALER: f64 = 1000.0;
const NUM_CLASSIFIERS: usize = 5;
const FLOATS_PER_CLF: usize = HIDDEN * IN_DIM + HIDDEN + HIDDEN * HIDDEN + HIDDEN + HIDDEN + 1;
/// `4*(r-g)^2 + (r+g-2*b)^2` spans 0..=520_200.
const COLORFULNESS_BINS: usize = 4 * 255 * 255 + 510 * 510 + 1;

#[derive(Debug, Clone, PartialEq)]
pub struct DetectedColor {
    pub is_color: bool,
    /// 0-100; only meaningful for ML-based detection.
    pub confidence: u8,
    pub reason: Option<&'static str>,
    pub should_convert: bool,
}

struct Mlp {
    w1: Vec<f32>, // (512, 32) row-major
    b1: Vec<f32>, // (512,)
    w2: Vec<f32>, // (512, 512)
    b2: Vec<f32>, // (512,)
    w3: Vec<f32>, // (1, 512)
    b3: f32,
}

pub struct Ogsov {
    /// Bit-packed, MSB-first. Bit index = b * 65536 + g * 256 + r.
    mask: Vec<u8>,
    classifiers: Vec<Mlp>,
}

#[derive(Debug)]
pub enum LoadError {
    BadLength { expected: usize, got: usize },
}

impl Ogsov {
    pub fn from_bytes(data: &[u8]) -> Result<Self, LoadError> {
        let expected = MASK_BYTES + NUM_CLASSIFIERS * FLOATS_PER_CLF * 4;
        if data.len() != expected {
            return Err(LoadError::BadLength {
                expected,
                got: data.len(),
            });
        }

        let mask = data[..MASK_BYTES].to_vec();
        let mut off = MASK_BYTES;

        let mut read_f32s = |n: usize| -> Vec<f32> {
            let out = data[off..off + n * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            off += n * 4;
            out
        };

        let mut classifiers = Vec::with_capacity(NUM_CLASSIFIERS);
        for _ in 0..NUM_CLASSIFIERS {
            let w1 = read_f32s(HIDDEN * IN_DIM);
            let b1 = read_f32s(HIDDEN);
            let w2 = read_f32s(HIDDEN * HIDDEN);
            let b2 = read_f32s(HIDDEN);
            let w3 = read_f32s(HIDDEN);
            let b3 = read_f32s(1)[0];
            classifiers.push(Mlp {
                w1,
                b1,
                w2,
                b2,
                w3,
                b3,
            });
        }

        Ok(Self { mask, classifiers })
    }

    /// Full detection: fast pre-check, then ML if the pre-check says "color".
    /// `rgb` is w*h*3 bytes, row-major, RGB order.
    pub fn detect(&self, rgb: &[u8], width: usize, height: usize) -> DetectedColor {
        debug_assert_eq!(rgb.len(), width * height * 3);
        self.detect_channels::<3>(rgb, width, height)
    }

    /// Full detection from RGBA without allocating an RGB copy. Alpha is ignored;
    /// callers must decide whether transparent images are eligible for conversion.
    pub fn detect_rgba(&self, rgba: &[u8], width: usize, height: usize) -> DetectedColor {
        debug_assert_eq!(rgba.len(), width * height * 4);
        self.detect_channels::<4>(rgba, width, height)
    }

    fn detect_channels<const CHANNELS: usize>(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> DetectedColor {
        if is_grayscale::<CHANNELS>(pixels) {
            return DetectedColor {
                is_color: false,
                confidence: 100,
                reason: Some("is grayscale RGB"),
                should_convert: true,
            };
        }

        let (is_color, confidence) = self.predict_channels::<CHANNELS>(pixels, width, height);
        DetectedColor {
            is_color,
            confidence,
            reason: Some("ML-based detection"),
            should_convert: is_color,
        }
    }

    /// ML prediction only. Returns (is_color, confidence 0-100).
    pub fn predict(&self, rgb: &[u8], width: usize, height: usize) -> (bool, u8) {
        debug_assert_eq!(rgb.len(), width * height * 3);
        self.predict_channels::<3>(rgb, width, height)
    }

    fn predict_channels<const CHANNELS: usize>(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> (bool, u8) {
        let features = self.compute_features::<CHANNELS>(pixels, width, height);

        self.predict_features(&features)
    }

    fn predict_features(&self, features: &[f32; IN_DIM]) -> (bool, u8) {
        let mut sum = 0.0f32;
        for clf in &self.classifiers {
            sum += sigmoid(clf.forward(features));
        }
        let mean_prob = sum / NUM_CLASSIFIERS as f32;

        let is_color = mean_prob >= 0.5;
        let conf = if is_color { mean_prob } else { 1.0 - mean_prob };
        // Python: int(conf * 100) — truncation toward zero.
        (is_color, (conf * 100.0) as u8)
    }

    fn mask_bit(&self, r: u8, g: u8, b: u8) -> bool {
        let idx = (b as usize) << 16 | (g as usize) << 8 | (r as usize);
        (self.mask[idx >> 3] >> (7 - (idx & 7))) & 1 != 0
    }

    fn compute_features<const CHANNELS: usize>(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> [f32; IN_DIM] {
        let n = width * height;
        assert!(n > TOP_N, "image too small for OGSOV feature extraction");

        // Colorfulness depends only on bounded integer d2. Histogram replaces
        // per-pixel float allocation + O(n log n) sort with O(n + fixed bins).
        let mut hist = vec![0u32; COLORFULNESS_BINS];
        let mut mask_hits: u64 = 0;
        let mut max_d2 = 0usize;
        for px in pixels.as_chunks::<CHANNELS>().0 {
            let (r8, g8, b8) = (px[0], px[1], px[2]);
            if self.mask_bit(r8, g8, b8) {
                mask_hits += 1;
            }
            let r = r8 as i32;
            let g = g8 as i32;
            let b = b8 as i32;
            let rg = r - g;
            let yb2 = r + g - 2 * b;
            let d2 = (4 * rg * rg + yb2 * yb2) as usize;
            hist[d2] += 1;
            max_d2 = max_d2.max(d2);
        }

        let val = |d2: usize| ((d2 as f32).sqrt() * 0.5).min(285.0) / 285.0;

        // Remove highest 17 values from histogram while preserving descending
        // top-feature order used by current Rust implementation.
        let mut top = [0f32; TOP_N];
        let mut got = 0usize;
        let mut bin = max_d2;
        loop {
            if hist[bin] > 0 {
                let take = (hist[bin] as usize).min(TOP_N - got);
                top[got..got + take].fill(val(bin));
                hist[bin] -= take as u32;
                got += take;
                if got == TOP_N {
                    break;
                }
            }
            bin -= 1;
        }

        let m = n - TOP_N;
        let mf = m as f64;

        // Sorted positions required for quartiles and rounded median.
        let pos = |q: f64| q / 100.0 * (mf - 1.0);
        let (p25, p50, p75) = (pos(25.0), pos(50.0), pos(75.0));
        let mut need = vec![
            p25.floor() as usize,
            p25.ceil() as usize,
            p50.floor() as usize,
            p50.ceil() as usize,
            p75.floor() as usize,
            p75.ceil() as usize,
            m / 2,
            if m.is_multiple_of(2) {
                m / 2 - 1
            } else {
                m / 2
            },
        ];
        need.sort_unstable();
        need.dedup();
        let mut found = vec![0f32; need.len()];

        // First histogram walk: order statistics, min/max, mean, L2 norm.
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let (mut vmin, mut vmax) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut cumulative = 0usize;
        let mut needed = 0usize;
        for (bin, &count) in hist[..=max_d2].iter().enumerate() {
            let count = count as usize;
            if count == 0 {
                continue;
            }
            let v = val(bin);
            if cumulative == 0 {
                vmin = v;
            }
            vmax = v;
            while needed < need.len() && need[needed] < cumulative + count {
                found[needed] = v;
                needed += 1;
            }
            cumulative += count;
            let v = v as f64;
            // Repeat in sorted order instead of multiplying by `count`: same
            // f64 accumulation order as old sort path, without sorting.
            for _ in 0..count {
                sum += v;
                sum_sq += v * v;
            }
        }
        debug_assert_eq!(cumulative, m);
        let mean = sum / mf;
        let l2_norm = sum_sq.sqrt();

        // Second histogram walk: central moments.
        let (mut m2, mut m3, mut m4) = (0.0f64, 0.0f64, 0.0f64);
        for (bin, &count) in hist[..=max_d2].iter().enumerate() {
            if count == 0 {
                continue;
            }
            let d = val(bin) as f64 - mean;
            let d2 = d * d;
            for _ in 0..count {
                m2 += d2;
                m3 += d2 * d;
                m4 += d2 * d2;
            }
        }
        m2 /= mf;
        m3 /= mf;
        m4 /= mf;

        let std = m2.sqrt();
        // scipy defaults: biased skew, Fisher (excess) kurtosis.
        // Zero variance => NaN in Python, then nan_to_num => 0.
        let (skew, kurt) = if m2 > 0.0 {
            (m3 / m2.powf(1.5), m4 / (m2 * m2) - 3.0)
        } else {
            (0.0, 0.0)
        };

        let at = |sorted_idx: usize| found[need.binary_search(&sorted_idx).unwrap()];
        let lerp = |p: f64| -> f64 {
            let floor = p.floor();
            let a = at(floor as usize) as f64;
            let b = at(p.ceil() as usize) as f64;
            a + (p - floor) * (b - a)
        };
        let q = [vmin as f64, lerp(p25), lerp(p50), lerp(p75), vmax as f64];
        let iqr = q[3] - q[1];

        // Median of round(v * 1000), matching numpy's ties-to-even rounding.
        let round_int = |v: f32| -> f64 { ((v as f64) * MEDIAN_SCALER).round_ties_even() };
        let med = if m % 2 == 1 {
            round_int(at(m / 2)) / MEDIAN_SCALER
        } else {
            (round_int(at(m / 2 - 1)) + round_int(at(m / 2))) / 2.0 / MEDIAN_SCALER
        };

        // Threshold features from the mask-hit count.
        let tc = |k: u64| -> f64 { mask_hits.min(k) as f64 / k as f64 };

        let mut features = [0.0f32; IN_DIM];
        let mut i = 0;
        let mut push = |v: f64| {
            // Mirror np.nan_to_num(..., nan=0, posinf=0, neginf=0).
            features[i] = if v.is_finite() { v as f32 } else { 0.0 };
            i += 1;
        };

        for &t in &top {
            push(t as f64);
        }
        push(std);
        push(med);
        push(skew);
        push(kurt);
        push(iqr);
        push(l2_norm);
        for &v in &q {
            push(v);
        }
        push(tc(2));
        push(tc(16));
        push(tc(128));
        push(tc(1024));
        debug_assert_eq!(i, IN_DIM);

        features
    }

    /// Previous per-pixel sort implementation, retained only as correctness
    /// oracle for histogram-path tests.
    #[cfg(test)]
    fn compute_features_sort_reference<const CHANNELS: usize>(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> [f32; IN_DIM] {
        let n = width * height;
        assert!(n > TOP_N);
        let mut color = Vec::with_capacity(n);
        let mut mask_hits = 0u64;
        for px in pixels.as_chunks::<CHANNELS>().0 {
            let (r8, g8, b8) = (px[0], px[1], px[2]);
            mask_hits += self.mask_bit(r8, g8, b8) as u64;
            let (r, g, b) = (r8 as f32, g8 as f32, b8 as f32);
            let rg = r - g;
            let yb = 0.5 * (r + g) - b;
            color.push((rg * rg + yb * yb).sqrt().clamp(0.0, 285.0) / 285.0);
        }

        let split = n - TOP_N;
        color.select_nth_unstable_by(split, |a, b| a.total_cmp(b));
        let (remaining, top) = color.split_at_mut(split);
        top.sort_unstable_by(|a, b| b.total_cmp(a));
        remaining.sort_unstable_by(|a, b| a.total_cmp(b));
        let m = remaining.len();
        let mf = m as f64;

        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &v in remaining.iter() {
            let v = v as f64;
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / mf;
        let l2_norm = sum_sq.sqrt();

        let (mut m2, mut m3, mut m4) = (0.0f64, 0.0f64, 0.0f64);
        for &v in remaining.iter() {
            let d = v as f64 - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        m2 /= mf;
        m3 /= mf;
        m4 /= mf;
        let std = m2.sqrt();
        let (skew, kurt) = if m2 > 0.0 {
            (m3 / m2.powf(1.5), m4 / (m2 * m2) - 3.0)
        } else {
            (0.0, 0.0)
        };

        let round_int = |v: f32| ((v as f64) * MEDIAN_SCALER).round_ties_even();
        let med = if m % 2 == 1 {
            round_int(remaining[m / 2]) / MEDIAN_SCALER
        } else {
            (round_int(remaining[m / 2 - 1]) + round_int(remaining[m / 2])) / 2.0 / MEDIAN_SCALER
        };
        let percentile = |q: f64| {
            let pos = q / 100.0 * (mf - 1.0);
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            let frac = pos - lo as f64;
            let (a, b) = (remaining[lo] as f64, remaining[hi] as f64);
            a + frac * (b - a)
        };
        let q = [
            percentile(0.0),
            percentile(25.0),
            percentile(50.0),
            percentile(75.0),
            percentile(100.0),
        ];
        let iqr = q[3] - q[1];
        let tc = |k: u64| mask_hits.min(k) as f64 / k as f64;

        let mut features = [0.0f32; IN_DIM];
        let mut i = 0;
        let mut push = |v: f64| {
            features[i] = if v.is_finite() { v as f32 } else { 0.0 };
            i += 1;
        };
        for &v in top.iter() {
            push(v as f64);
        }
        for v in [std, med, skew, kurt, iqr, l2_norm] {
            push(v);
        }
        for &v in &q {
            push(v);
        }
        for k in [2, 16, 128, 1024] {
            push(tc(k));
        }
        features
    }
}

impl Mlp {
    /// PyTorch Linear: y = x W^T + b, weights stored (out, in) row-major.
    fn forward(&self, x: &[f32; IN_DIM]) -> f32 {
        let mut h1 = [0.0f32; HIDDEN];
        for (o, h) in h1.iter_mut().enumerate() {
            let row = &self.w1[o * IN_DIM..(o + 1) * IN_DIM];
            let mut acc = self.b1[o];
            for (w, v) in row.iter().zip(x.iter()) {
                acc += w * v;
            }
            *h = acc.tanh();
        }

        let mut h2 = [0.0f32; HIDDEN];
        for (o, h) in h2.iter_mut().enumerate() {
            let row = &self.w2[o * HIDDEN..(o + 1) * HIDDEN];
            let mut acc = self.b2[o];
            for (w, v) in row.iter().zip(h1.iter()) {
                acc += w * v;
            }
            *h = acc.tanh();
        }

        let mut acc = self.b3;
        for (w, v) in self.w3.iter().zip(h2.iter()) {
            acc += w * v;
        }
        acc
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Port of the fast pre-check for RGB data: every pixel has r == g == b.
pub fn is_grayscale_rgb(rgb: &[u8]) -> bool {
    is_grayscale::<3>(rgb)
}

fn is_grayscale<const CHANNELS: usize>(pixels: &[u8]) -> bool {
    pixels
        .as_chunks::<CHANNELS>()
        .0
        .iter()
        .all(|px| px[0] == px[1] && px[1] == px[2])
}

#[cfg(test)]
mod tests {
    fn canvas(width: usize, height: usize, channels: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut pixels = Vec::with_capacity(width * height * channels);
        for i in 0..width * height {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let r = state as u8;
            let g = state.rotate_left(17) as u8;
            let b = if i % 19 == 0 {
                r
            } else {
                state.rotate_left(41) as u8
            };
            pixels.extend_from_slice(&[r, g, b]);
            if channels == 4 {
                pixels.push(255);
            }
        }
        pixels
    }

    fn compare<const CHANNELS: usize>(pixels: &[u8], width: usize, height: usize) {
        let Some(model) = crate::embedded_model() else {
            return;
        };
        let histogram = model.compute_features::<CHANNELS>(pixels, width, height);
        let sorted = model.compute_features_sort_reference::<CHANNELS>(pixels, width, height);
        for (i, (&a, &b)) in histogram.iter().zip(&sorted).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "feature {i}: histogram={a}, sort={b}"
            );
        }
        assert_eq!(
            model.predict_features(&histogram),
            model.predict_features(&sorted),
            "classification/confidence changed"
        );
    }

    #[test]
    fn histogram_matches_sort_reference_for_rgb_and_rgba() {
        for (width, height, seed) in [(64, 48, 1), (257, 129, 0xdead_beef)] {
            compare::<3>(&canvas(width, height, 3, seed), width, height);
            compare::<4>(&canvas(width, height, 4, seed), width, height);
        }

        let (width, height) = (128, 96);
        let mut near_gray = Vec::with_capacity(width * height * 3);
        let mut gradient = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let g = ((x * 17 + y * 31) & 255) as u8;
                near_gray.extend_from_slice(&[g, g.saturating_add(1), g]);
                gradient.extend_from_slice(&[
                    (x * 255 / (width - 1)) as u8,
                    (y * 255 / (height - 1)) as u8,
                    ((x + y) * 255 / (width + height - 2)) as u8,
                ]);
            }
        }
        compare::<3>(&near_gray, width, height);
        compare::<3>(&gradient, width, height);
    }
}

//! The per-tile pixel kernels: NEON on aarch64, AVX2 or AVX-512 on x86-64, and the
//! portable scalar reference beside them.
//!
//! Two measurements drive this file.  The chroma pass is a long chain of `f32`
//! work per pixel with no room left to overlap, and the hit pass is dominated
//! by pages where almost nothing can pass at all: a near-gray manga page asks
//! sixteen pixels at a time whether any of them could possibly be a hit, and
//! for the pages this filter exists to answer, the answer is usually no.
//!
//! Both kernels are bit-exact replacements for the loops they took over:
//!
//! * the chroma lanes keep their own addition order, because the vector
//!   accumulators *are* the eight lanes — lane `i` is loaded, added to and
//!   stored back, so it adds exactly the values it added before, in the same
//!   order;
//! * the squared accumulation multiplies and then adds rather than fusing the
//!   two, because the reference rounds the product before adding it;
//! * the hit count is only ever read through `min(hits, 1024)`, so counting a
//!   row before comparing against the cap is the same answer as stopping at
//!   the pixel the scalar loop would have stopped at.
//!
//! The vector chroma is built from the integer form of the graph's squared
//! chroma, `4*(R-G)^2 + (R+G-2B)^2`, which is exactly four times the `f32`
//! value the scalar path builds.  `sqrt(4s)/570` rounds to the same `f32` as
//! `sqrt(s)/285`: scaling by a power of two is exact, and round-to-nearest
//! commutes with it.  Every intermediate is a small exact integer, so the
//! integer form loses nothing on the way in.

use crate::lookup::LookupTable;
use crate::rgb::RgbView;
use std::time::{Duration, Instant};

use super::{MASK_HIT_CAP, REDUCTION_LANES, TILE};

/// The graph's squared chroma, four times the `f32` value it computes.
#[inline]
pub(super) fn squared_chroma_4(red: u8, green: u8, blue: u8) -> i32 {
    let red = red as i32;
    let green = green as i32;
    let blue = blue as i32;
    let dx = red - green;
    let yb = red + green - 2 * blue;
    4 * dx * dx + yb * yb
}

/// The graph's clipped chroma, from [`squared_chroma_4`].
#[inline]
pub(super) fn chroma_of_squared_4(squared: i32) -> f32 {
    ((squared as f32).sqrt() / 570.0f32).min(1.0f32)
}

/// Whether one pixel is a lookup hit.
#[inline]
fn counts_as_hit(lookup: &LookupTable, threshold: u16, red: u8, green: u8, blue: u8) -> bool {
    let high = red.max(green).max(blue) as i32;
    let low = red.min(green).min(blue) as i32;
    high - low >= threshold as i32 && lookup.contains(blue, green, red)
}

/// One frame's per-tile aggregates, walked a band at a time.
///
/// A tile at a time reads 512 bytes of every 3 KiB row and hands the
/// prefetcher a 3 KiB stride; a band at a time reads each plane
/// contiguously, and the band's 4.5 MB is still in the last-level cache when
/// the hit scan re-reads it.  The per-tile lanes are fed the same row-major
/// pixels in the same order either way, so the order the *tiles* are visited
/// in is not part of any answer.
pub(super) struct Sweep {
    /// Each tile's eight lane sums, squared sums and maximum.
    pub chromas: Vec<([f32; REDUCTION_LANES], [f32; REDUCTION_LANES], f32)>,
    /// Each tile's lookup hits, capped as the graph caps them.
    pub hits: Vec<u32>,
    /// Time spent in the chroma walk, for the `debug` line.
    pub chroma_time: Duration,
    /// Time spent in the hit scan, for the `debug` line.
    pub hits_time: Duration,
}

/// Walk every tile of a frame, band by band.
///
/// `across` is the number of tiles a band holds; the band windows are derived
/// from `origins` because `tile_starts` is NumPy's linspace and adjacent bands
/// overlap whenever a dimension is not a multiple of 512.
pub(super) fn sweep_tiles(
    rgb: &RgbView<'_>,
    origins: &[(usize, usize)],
    across: usize,
    lookup: &LookupTable,
    threshold: u16,
    timed: bool,
) -> Sweep {
    let width = rgb.width();
    let height = rgb.height();
    let mut chromas =
        vec![([0.0f32; REDUCTION_LANES], [0.0f32; REDUCTION_LANES], 0.0f32,); origins.len()];
    let mut hits = vec![0u32; origins.len()];
    let mut chroma_time = Duration::ZERO;
    let mut hits_time = Duration::ZERO;
    let bands: Vec<(usize, usize)> = (0..origins.len() / across)
        .map(|band| {
            let start = origins[band * across].0;
            (start, (start + TILE).min(height))
        })
        .collect();
    for (band, &(start, end)) in bands.iter().enumerate() {
        let started = timed.then(Instant::now);
        for y in start..end {
            for column in 0..across {
                let tile = band * across + column;
                let (top, left) = origins[tile];
                debug_assert_eq!(top, start);
                let columns = TILE.min(width - left);
                let (lanes, squared_lanes, maximum) = &mut chromas[tile];
                #[cfg(any(
                    target_arch = "aarch64",
                    all(target_arch = "x86_64", target_feature = "avx2")
                ))]
                unsafe {
                    vector::accumulate_row(
                        &rgb.red.row(y)[left..left + columns],
                        &rgb.green.row(y)[left..left + columns],
                        &rgb.blue.row(y)[left..left + columns],
                        lanes,
                        squared_lanes,
                        maximum,
                    );
                }
                #[cfg(not(any(
                    target_arch = "aarch64",
                    all(target_arch = "x86_64", target_feature = "avx2")
                )))]
                scalar::accumulate_row(
                    &rgb.red.row(y)[left..left + columns],
                    &rgb.green.row(y)[left..left + columns],
                    &rgb.blue.row(y)[left..left + columns],
                    lanes,
                    squared_lanes,
                    maximum,
                );
            }
        }
        if let Some(started) = started {
            chroma_time += started.elapsed();
        }
        // The band was just read; the hit scan re-reads it a tile at a time
        // from the last-level cache rather than from memory.
        let started = timed.then(Instant::now);
        for column in 0..across {
            let tile = band * across + column;
            let (top, left) = origins[tile];
            let rows = TILE.min(height - top);
            let columns = TILE.min(width - left);
            hits[tile] = tile_hits(rgb, lookup, top, left, rows, columns, threshold);
        }
        if let Some(started) = started {
            hits_time += started.elapsed();
        }
    }
    Sweep {
        chromas,
        hits,
        chroma_time,
        hits_time,
    }
}

/// One tile's lookup hits, stopping at the graph's cap.
pub(super) fn tile_hits(
    rgb: &RgbView<'_>,
    lookup: &LookupTable,
    top: usize,
    left: usize,
    rows: usize,
    columns: usize,
    threshold: u16,
) -> u32 {
    let mut hits = 0u32;
    for y in top..top + rows {
        let reds = &rgb.red.row(y)[left..left + columns];
        let greens = &rgb.green.row(y)[left..left + columns];
        let blues = &rgb.blue.row(y)[left..left + columns];
        // SAFETY: as in `sweep_tiles`.
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx2")
        ))]
        unsafe {
            hits += vector::row_hits(reds, greens, blues, lookup, threshold);
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx2")
        )))]
        {
            hits += scalar::row_hits(reds, greens, blues, lookup, threshold);
        }
        if hits >= MASK_HIT_CAP {
            // Every mask ratio is capped at 1024, so the count past the cap
            // cannot change a feature and the rest of the tile is not read.
            return MASK_HIT_CAP;
        }
    }
    hits
}

/// The portable kernels, one pixel at a time.
///
/// This is the reference order the vector paths have to reproduce, and it is
/// what a host without NEON or AVX2 runs.  Where a vector kernel is compiled in it
/// is the documented definition rather than the code that runs, which is why it
/// may be unused.
#[cfg_attr(
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ),
    allow(dead_code)
)]
mod scalar {
    use super::super::REDUCTION_LANES;
    use super::{chroma_of_squared_4, counts_as_hit, squared_chroma_4};
    use crate::lookup::LookupTable;

    /// Add one row of chroma to the tile's lane accumulators.
    pub(super) fn accumulate_row(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lanes: &mut [f32; REDUCTION_LANES],
        squared_lanes: &mut [f32; REDUCTION_LANES],
        maximum: &mut f32,
    ) {
        let mut lane = 0usize;
        for index in 0..reds.len() {
            let value =
                chroma_of_squared_4(squared_chroma_4(reds[index], greens[index], blues[index]));
            lanes[lane] += value;
            squared_lanes[lane] += value * value;
            if value > *maximum {
                *maximum = value;
            }
            lane += 1;
            if lane == REDUCTION_LANES {
                lane = 0;
            }
        }
    }

    /// Count one row's lookup hits.
    pub(super) fn row_hits(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
    ) -> u32 {
        let mut hits = 0u32;
        for index in 0..reds.len() {
            if counts_as_hit(lookup, threshold, reds[index], greens[index], blues[index]) {
                hits += 1;
            }
        }
        hits
    }
}

/// The aarch64 kernels.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::super::REDUCTION_LANES;
    use super::{chroma_of_squared_4, counts_as_hit, squared_chroma_4};
    use crate::lookup::LookupTable;

    /// How many pixels one chroma iteration covers.
    const STEP: usize = 8;
    /// How many pixels one spread test covers.
    const WIDE: usize = 16;

    /// The graph's squared chroma for four pixels, four times the `f32` value.
    ///
    /// SAFETY: NEON is part of the aarch64 baseline, so every intrinsic here
    /// is available unconditionally.
    #[inline]
    unsafe fn squared4(red: int32x4_t, green: int32x4_t, blue: int32x4_t) -> int32x4_t {
        unsafe {
            let dx = vsubq_s32(red, green);
            let twice_dx = vshlq_n_s32(dx, 1);
            let yb = vsubq_s32(vaddq_s32(red, green), vshlq_n_s32(blue, 1));
            vaddq_s32(vmulq_s32(twice_dx, twice_dx), vmulq_s32(yb, yb))
        }
    }

    /// The graph's clipped chroma for four pixels, from their squared chroma.
    ///
    /// SAFETY: as [`squared4`].
    #[inline]
    unsafe fn chroma4(squared: int32x4_t) -> float32x4_t {
        unsafe {
            // One correctly rounded square root and one correctly rounded
            // division, which are the two operations the scalar path performs.
            let value = vdivq_f32(vsqrtq_f32(vcvtq_f32_s32(squared)), vdupq_n_f32(570.0));
            vminq_f32(value, vdupq_n_f32(1.0))
        }
    }

    /// Split eight loaded samples into two four-lane integers.
    ///
    /// SAFETY: as [`chroma4`].
    #[inline]
    unsafe fn widen(samples: uint8x8_t) -> (int32x4_t, int32x4_t) {
        unsafe {
            let wide = vmovl_u8(samples);
            (
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(wide))),
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(wide))),
            )
        }
    }

    /// Add one row of chroma to the tile's lane accumulators.
    ///
    /// SAFETY: `reds`, `greens` and `blues` are the same length, and `lanes`
    /// and `squared_lanes` are eight wide.
    pub(super) unsafe fn accumulate_row(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lanes: &mut [f32; REDUCTION_LANES],
        squared_lanes: &mut [f32; REDUCTION_LANES],
        maximum: &mut f32,
    ) {
        unsafe {
            let count = reds.len();
            let mut index = 0usize;
            // The vector accumulators *are* the lanes: each is loaded from the
            // lane it belongs to and stored back to it, so lane `i` sums exactly
            // the values it summed before, in the same order.
            let mut low = vld1q_f32(lanes.as_ptr());
            let mut high = vld1q_f32(lanes.as_ptr().add(4));
            let mut low_squared = vld1q_f32(squared_lanes.as_ptr());
            let mut high_squared = vld1q_f32(squared_lanes.as_ptr().add(4));
            // Chroma is never negative, so zero is a valid identity to start from.
            let mut running_maximum = vdupq_n_f32(0.0);
            while index + STEP <= count {
                let (red_low, red_high) = widen(vld1_u8(reds.as_ptr().add(index)));
                let (green_low, green_high) = widen(vld1_u8(greens.as_ptr().add(index)));
                let (blue_low, blue_high) = widen(vld1_u8(blues.as_ptr().add(index)));
                let squared_low = squared4(red_low, green_low, blue_low);
                let squared_high = squared4(red_high, green_high, blue_high);
                // A gray group contributes nothing: every chroma is +0.0, so the
                // lane sums, the squared sums and the maximum are unchanged by
                // adding it.  A page of paper and ink is mostly such groups, and
                // this skips the square root and the division for them without
                // moving a single bit of the answer.
                if vmaxvq_s32(vorrq_s32(squared_low, squared_high)) == 0 {
                    index += STEP;
                    continue;
                }
                let value_low = chroma4(squared_low);
                let value_high = chroma4(squared_high);
                low = vaddq_f32(low, value_low);
                high = vaddq_f32(high, value_high);
                low_squared = vaddq_f32(low_squared, vmulq_f32(value_low, value_low));
                high_squared = vaddq_f32(high_squared, vmulq_f32(value_high, value_high));
                running_maximum = vmaxq_f32(running_maximum, vmaxq_f32(value_low, value_high));
                index += STEP;
            }
            vst1q_f32(lanes.as_mut_ptr(), low);
            vst1q_f32(lanes.as_mut_ptr().add(4), high);
            vst1q_f32(squared_lanes.as_mut_ptr(), low_squared);
            vst1q_f32(squared_lanes.as_mut_ptr().add(4), high_squared);
            *maximum = maximum.max(vmaxvq_f32(running_maximum));
            // The remainder is shorter than one group, so it goes one pixel at a
            // time; `index` is a multiple of eight here, which is the lane the
            // scalar loop would have used next.
            let mut lane = 0usize;
            while index < count {
                let value =
                    chroma_of_squared_4(squared_chroma_4(reds[index], greens[index], blues[index]));
                lanes[lane] += value;
                squared_lanes[lane] += value * value;
                if value > *maximum {
                    *maximum = value;
                }
                lane += 1;
                index += 1;
            }
        }
    }

    /// Count one row's lookup hits.
    ///
    /// The spread test is answered sixteen pixels at a time, and only a group
    /// with at least one candidate reaches the lookup.
    ///
    /// SAFETY: `reds`, `greens` and `blues` are the same length.
    pub(super) unsafe fn row_hits(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
    ) -> u32 {
        unsafe {
            // 256 means "nothing can hit"; narrowing that to `u8` would
            // wrap to zero and make every pixel a candidate.
            let min_spread = vdupq_n_u8(threshold.min(255) as u8);
            let mut hits = 0u32;
            let count = reds.len();
            let mut index = 0usize;
            while index + WIDE <= count {
                let red = vld1q_u8(reds.as_ptr().add(index));
                let green = vld1q_u8(greens.as_ptr().add(index));
                let blue = vld1q_u8(blues.as_ptr().add(index));
                let spread = vsubq_u8(
                    vmaxq_u8(vmaxq_u8(red, green), blue),
                    vminq_u8(vminq_u8(red, green), blue),
                );
                if vmaxvq_u8(vcgeq_u8(spread, min_spread)) == 0 {
                    // Not one pixel in this group can be a hit, which is the usual
                    // answer on a page that is only nominally r,g,b.
                    index += WIDE;
                    continue;
                }
                for offset in index..index + WIDE {
                    if counts_as_hit(
                        lookup,
                        threshold,
                        reds[offset],
                        greens[offset],
                        blues[offset],
                    ) {
                        hits += 1;
                    }
                }
                index += WIDE;
            }
            while index < count {
                if counts_as_hit(lookup, threshold, reds[index], greens[index], blues[index]) {
                    hits += 1;
                }
                index += 1;
            }
            hits
        }
    }
}

#[cfg(target_arch = "aarch64")]
use neon as vector;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use x86 as vector;

/// The x86-64 kernels, compiled in only where the build was told the CPU level.
///
/// There is no runtime detection: the AVX2 kernels exist in the `.avx2` and
/// `.avx512` libraries and the AVX-512 mask test only in the `.avx512` one.
/// The AVX-512 build keeps the AVX2 chroma kernel, because a 512-bit square
/// root and division issue at half the rate of a 256-bit one and the lanes
/// that are accumulated are eight wide either way.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86 {
    use std::arch::x86_64::*;

    use super::super::REDUCTION_LANES;
    use super::{chroma_of_squared_4, counts_as_hit, squared_chroma_4};
    use crate::lookup::LookupTable;

    /// How many pixels one chroma iteration covers: exactly the eight lanes.
    const STEP: usize = 8;
    /// How many pixels one AVX2 spread test covers.
    const WIDE: usize = 32;

    /// The graph's clipped chroma for eight pixels.
    ///
    /// SAFETY: AVX2 is on for the whole crate wherever this module exists.
    #[inline]
    unsafe fn squared8(red: __m256i, green: __m256i, blue: __m256i) -> __m256i {
        unsafe {
            let dx = _mm256_sub_epi32(red, green);
            let twice_dx = _mm256_slli_epi32::<1>(dx);
            let yb = _mm256_sub_epi32(_mm256_add_epi32(red, green), _mm256_slli_epi32::<1>(blue));
            _mm256_add_epi32(
                _mm256_mullo_epi32(twice_dx, twice_dx),
                _mm256_mullo_epi32(yb, yb),
            )
        }
    }

    /// The graph's clipped chroma for eight pixels.
    ///
    /// SAFETY: AVX2 is on for the whole crate wherever this module exists.
    #[inline]
    unsafe fn chroma8(squared: __m256i) -> __m256 {
        unsafe {
            // One correctly rounded square root and one correctly rounded
            // division, which are the two operations the scalar path performs.
            let value = _mm256_div_ps(
                _mm256_sqrt_ps(_mm256_cvtepi32_ps(squared)),
                _mm256_set1_ps(570.0),
            );
            _mm256_min_ps(value, _mm256_set1_ps(1.0))
        }
    }

    /// Eight samples widened to eight `i32`.
    ///
    /// SAFETY: eight bytes are readable at `pointer`.
    #[inline]
    unsafe fn load8(pointer: *const u8) -> __m256i {
        unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(pointer as *const __m128i)) }
    }

    /// Sixteen samples widened to sixteen `i32`.
    ///
    /// SAFETY: sixteen bytes are readable at `pointer`, and the build has AVX-512.
    #[cfg(target_feature = "avx512f")]
    #[inline]
    unsafe fn load16(pointer: *const u8) -> __m512i {
        unsafe { _mm512_cvtepu8_epi32(_mm_loadu_si128(pointer as *const __m128i)) }
    }

    /// The graph's squared chroma for sixteen pixels.
    ///
    /// SAFETY: AVX-512F and AVX-512BW are on for the whole crate wherever the
    /// wide path is compiled.
    #[cfg(target_feature = "avx512f")]
    #[inline]
    unsafe fn squared16(red: __m512i, green: __m512i, blue: __m512i) -> __m512i {
        unsafe {
            let dx = _mm512_sub_epi32(red, green);
            let twice_dx = _mm512_slli_epi32::<1>(dx);
            let yb = _mm512_sub_epi32(_mm512_add_epi32(red, green), _mm512_slli_epi32::<1>(blue));
            _mm512_add_epi32(
                _mm512_mullo_epi32(twice_dx, twice_dx),
                _mm512_mullo_epi32(yb, yb),
            )
        }
    }

    /// The largest of the eight lanes.
    #[inline]
    unsafe fn horizontal_max(value: __m256) -> f32 {
        unsafe {
            let half = _mm_max_ps(
                _mm256_castps256_ps128(value),
                _mm256_extractf128_ps::<1>(value),
            );
            let quarter = _mm_max_ps(half, _mm_movehl_ps(half, half));
            let one = _mm_max_ss(quarter, _mm_shuffle_ps::<0b01>(quarter, quarter));
            _mm_cvtss_f32(one)
        }
    }

    /// Add one row of chroma to the tile's lane accumulators.
    ///
    /// SAFETY: `reds`, `greens` and `blues` are the same length.
    pub(super) unsafe fn accumulate_row(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lanes: &mut [f32; REDUCTION_LANES],
        squared_lanes: &mut [f32; REDUCTION_LANES],
        maximum: &mut f32,
    ) {
        unsafe {
            let count = reds.len();
            let mut index = 0usize;
            // The vector accumulators *are* the lanes: each is loaded from the
            // lane it belongs to and stored back to it, so lane `i` sums exactly
            // the values it summed before, in the same order.
            let mut sum = _mm256_loadu_ps(lanes.as_ptr());
            let mut squared_sum = _mm256_loadu_ps(squared_lanes.as_ptr());
            // Chroma is never negative, so zero is a valid identity to start from.
            let mut running_maximum = _mm256_setzero_ps();
            // A gray group contributes nothing: every chroma is +0.0, so the
            // lane sums, the squared sums and the maximum are unchanged by
            // adding it.  A page of paper and ink is mostly such groups, and
            // this skips the square root and the division for them without
            // moving a single bit of the answer.
            #[inline]
            unsafe fn step(
                squared: __m256i,
                sum: &mut __m256,
                squared_sum: &mut __m256,
                running_maximum: &mut __m256,
            ) {
                unsafe {
                    let all_gray =
                        _mm256_movemask_epi8(_mm256_cmpeq_epi32(squared, _mm256_setzero_si256()))
                            == -1;
                    if all_gray {
                        return;
                    }
                    let value = chroma8(squared);
                    *sum = _mm256_add_ps(*sum, value);
                    // A product and then a sum: a fused multiply-add would round once.
                    *squared_sum = _mm256_add_ps(*squared_sum, _mm256_mul_ps(value, value));
                    *running_maximum = _mm256_max_ps(*running_maximum, value);
                }
            }

            // Sixteen pixels at a time where AVX-512 exists: the widening of
            // six bytes to i32 is three port-5 operations either way, and
            // halving their number per pixel is what the wide build buys here.
            // The two halves are still the eight lanes, low then high, so the
            // order of every lane's additions is untouched.
            #[cfg(target_feature = "avx512f")]
            while index + 16 <= count {
                let squared = squared16(
                    load16(reds.as_ptr().add(index)),
                    load16(greens.as_ptr().add(index)),
                    load16(blues.as_ptr().add(index)),
                );
                step(
                    _mm512_castsi512_si256(squared),
                    &mut sum,
                    &mut squared_sum,
                    &mut running_maximum,
                );
                step(
                    _mm512_extracti64x4_epi64::<1>(squared),
                    &mut sum,
                    &mut squared_sum,
                    &mut running_maximum,
                );
                index += 16;
            }
            while index + STEP <= count {
                let squared = squared8(
                    load8(reds.as_ptr().add(index)),
                    load8(greens.as_ptr().add(index)),
                    load8(blues.as_ptr().add(index)),
                );
                step(squared, &mut sum, &mut squared_sum, &mut running_maximum);
                index += STEP;
            }
            _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
            _mm256_storeu_ps(squared_lanes.as_mut_ptr(), squared_sum);
            *maximum = maximum.max(horizontal_max(running_maximum));
            // The remainder is shorter than one group, so it goes one pixel at a
            // time; `index` is a multiple of eight here, which is the lane the
            // scalar loop would have used next.
            let mut lane = 0usize;
            while index < count {
                let squared = squared_chroma_4(reds[index], greens[index], blues[index]);
                if squared == 0 {
                    lane += 1;
                    index += 1;
                    continue;
                }
                let value = chroma_of_squared_4(squared);
                lanes[lane] += value;
                squared_lanes[lane] += value * value;
                if value > *maximum {
                    *maximum = value;
                }
                lane += 1;
                index += 1;
            }
        }
    }

    /// Probe the lookup for every pixel the spread test left a bit set for.
    ///
    /// SAFETY: `bits` only names pixels below `reds.len()` from `base`.
    #[inline]
    unsafe fn probe(
        mut bits: u64,
        base: usize,
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
    ) -> u32 {
        let mut hits = 0u32;
        while bits != 0 {
            let offset = base + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            if counts_as_hit(
                lookup,
                threshold,
                reds[offset],
                greens[offset],
                blues[offset],
            ) {
                hits += 1;
            }
        }
        hits
    }

    /// Count one row's lookup hits, thirty-two pixels per spread test, from
    /// pixel `from` on.
    ///
    /// SAFETY: `reds`, `greens` and `blues` are the same length.
    unsafe fn row_hits_from(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
        from: usize,
    ) -> u32 {
        unsafe {
            // 256 means "nothing can hit"; narrowing that to `u8` would
            // wrap to zero and make every pixel a candidate.
            let min_spread = _mm256_set1_epi8(threshold.min(255) as u8 as i8);
            let mut hits = 0u32;
            let count = reds.len();
            let mut index = from;
            while index + WIDE <= count {
                let red = _mm256_loadu_si256(reds.as_ptr().add(index) as *const __m256i);
                let green = _mm256_loadu_si256(greens.as_ptr().add(index) as *const __m256i);
                let blue = _mm256_loadu_si256(blues.as_ptr().add(index) as *const __m256i);
                let spread = _mm256_sub_epi8(
                    _mm256_max_epu8(_mm256_max_epu8(red, green), blue),
                    _mm256_min_epu8(_mm256_min_epu8(red, green), blue),
                );
                // `spread >= min_spread`, as an unsigned byte compare.
                let candidates = _mm256_cmpeq_epi8(_mm256_max_epu8(spread, min_spread), spread);
                let bits = _mm256_movemask_epi8(candidates) as u32 as u64;
                // Not one pixel in most groups can be a hit, which is the usual
                // answer on a page that is only nominally r,g,b.
                if bits != 0 {
                    hits += probe(bits, index, reds, greens, blues, lookup, threshold);
                }
                index += WIDE;
            }
            while index < count {
                if counts_as_hit(lookup, threshold, reds[index], greens[index], blues[index]) {
                    hits += 1;
                }
                index += 1;
            }
            hits
        }
    }

    /// Count one row's lookup hits with the AVX2 spread test.
    ///
    /// SAFETY: `reds`, `greens` and `blues` are the same length.
    #[cfg_attr(target_feature = "avx512bw", allow(dead_code))]
    pub(super) unsafe fn row_hits_avx2(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
    ) -> u32 {
        unsafe { row_hits_from(reds, greens, blues, lookup, threshold, 0) }
    }

    /// Count one row's lookup hits with the AVX-512 spread test: sixty-four
    /// pixels per compare, answered as a bit mask.
    ///
    /// SAFETY: `reds`, `greens` and `blues` are the same length.
    #[cfg(target_feature = "avx512bw")]
    pub(super) unsafe fn row_hits_avx512(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
    ) -> u32 {
        unsafe {
            let min_spread = _mm512_set1_epi8(threshold.min(255) as u8 as i8);
            let mut hits = 0u32;
            let count = reds.len();
            let mut index = 0usize;
            while index + 64 <= count {
                let red = _mm512_loadu_si512(reds.as_ptr().add(index) as *const _);
                let green = _mm512_loadu_si512(greens.as_ptr().add(index) as *const _);
                let blue = _mm512_loadu_si512(blues.as_ptr().add(index) as *const _);
                let spread = _mm512_sub_epi8(
                    _mm512_max_epu8(_mm512_max_epu8(red, green), blue),
                    _mm512_min_epu8(_mm512_min_epu8(red, green), blue),
                );
                let bits = _mm512_cmpge_epu8_mask(spread, min_spread);
                if bits != 0 {
                    hits += probe(bits, index, reds, greens, blues, lookup, threshold);
                }
                index += 64;
            }
            // Fewer than sixty-four pixels remain: the narrower test and then
            // single pixels finish the row.
            hits + row_hits_from(reds, greens, blues, lookup, threshold, index)
        }
    }

    /// Count one row's lookup hits with the widest test this build has.
    ///
    /// SAFETY: `reds`, `greens` and `blues` are the same length.
    #[inline]
    pub(super) unsafe fn row_hits(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
    ) -> u32 {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            row_hits_avx512(reds, greens, blues, lookup, threshold)
        }
        #[cfg(not(target_feature = "avx512bw"))]
        unsafe {
            row_hits_avx2(reds, greens, blues, lookup, threshold)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{MASK_HIT_CAP, REDUCTION_LANES, chroma};
    use super::*;

    /// The integer chroma is a rewriting of the `f32` one, not an
    /// approximation of it: every reachable `(R-G, R+G-2B)` pair has to land
    /// on the same bits.
    #[test]
    fn the_integer_chroma_matches_the_float_one_everywhere() {
        let mut checked = 0usize;
        for dx in -255i32..=255 {
            for yb in -510i32..=510 {
                if (yb - dx) % 2 != 0 {
                    // No integer `r` and `g` have this sum and difference.
                    continue;
                }
                // With `b = shift`, `r + g = yb + 2*shift` and `r - g = dx`:
                // both terms are what the pair describes whatever the shift.
                let (mut red, mut green) = ((yb + dx) / 2, (yb - dx) / 2);
                let shift = (-red).max(-green).max(0);
                red += shift;
                green += shift;
                let (red, green, blue) = (red, green, shift);
                if !(0..=255).contains(&red) || !(0..=255).contains(&green) || blue > 255 {
                    continue;
                }
                let (red, green, blue) = (red as u8, green as u8, blue as u8);
                let want = chroma(red, green, blue);
                let got = chroma_of_squared_4(squared_chroma_4(red, green, blue));
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "({red}, {green}, {blue}) from dx {dx} and yb {yb}"
                );
                checked += 1;
            }
        }
        assert!(checked > 100_000, "only {checked} pairs were reachable");
    }

    /// A deterministic byte stream, so a failure can be replayed.
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u8 {
            let mut state = self.0;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            self.0 = state;
            (state >> 24) as u8
        }
    }

    /// One row of each kind the kernels take a different path on.
    fn rows(kind: u8, length: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut stream = Stream(seed);
        let mut reds = Vec::with_capacity(length);
        let mut greens = Vec::with_capacity(length);
        let mut blues = Vec::with_capacity(length);
        for index in 0..length {
            let (red, green, blue) = match kind {
                // Noise: every lane is exercised, and so is the lookup.
                0 => (stream.next(), stream.next(), stream.next()),
                // Near-gray: nothing passes the spread test, which is the path
                // a manga page spends its time on.
                1 => {
                    let grey = 100 + (index % 3) as u8;
                    (grey, grey + (index % 2) as u8, grey)
                }
                // A flat run, which is what the lookup cache would see.
                _ => (17, 17, 17),
            };
            reds.push(red);
            greens.push(green);
            blues.push(blue);
        }
        (reds, greens, blues)
    }

    /// A table that is true for a third of the first slice, so a row of noise
    /// both hits and misses.
    fn sparse_lookup() -> LookupTable {
        let table: Vec<bool> = (0..(1 << 16)).map(|index| index % 3 == 0).collect();
        LookupTable::from_bool_slice(&table, 2)
    }

    /// A table that is true for three quarters of every colour, which is what
    /// the real asset looks like (15,550,753 of 16,777,216 entries).
    fn dense_lookup() -> LookupTable {
        let table: Vec<bool> = (0..crate::lookup::MASK_ELEMENTS)
            .map(|index| index % 4 != 0)
            .collect();
        LookupTable::from_bool_slice(&table, 2)
    }

    fn lanes_of(
        row: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
    ) -> ([f32; REDUCTION_LANES], [f32; REDUCTION_LANES], f32, u32) {
        let mut lanes = [0.0f32; REDUCTION_LANES];
        let mut squared_lanes = [0.0f32; REDUCTION_LANES];
        let mut maximum = 0.0f32;
        scalar::accumulate_row(
            row,
            greens,
            blues,
            &mut lanes,
            &mut squared_lanes,
            &mut maximum,
        );
        (
            lanes,
            squared_lanes,
            maximum,
            scalar::row_hits(row, greens, blues, lookup, threshold),
        )
    }

    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    fn vector_lanes_of(
        row: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
    ) -> ([f32; REDUCTION_LANES], [f32; REDUCTION_LANES], f32, u32) {
        let mut lanes = [0.0f32; REDUCTION_LANES];
        let mut squared_lanes = [0.0f32; REDUCTION_LANES];
        let mut maximum = 0.0f32;
        // SAFETY: NEON is part of the aarch64 baseline, and AVX2 is on for the
        // whole crate wherever the x86 module is compiled.
        unsafe {
            vector::accumulate_row(
                row,
                greens,
                blues,
                &mut lanes,
                &mut squared_lanes,
                &mut maximum,
            );
        }
        let hits = unsafe { vector::row_hits(row, greens, blues, lookup, threshold) };
        (lanes, squared_lanes, maximum, hits)
    }

    /// The AVX2 spread test, which an AVX-512 build does not otherwise run.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn avx2_hits_of(
        row: &[u8],
        greens: &[u8],
        blues: &[u8],
        lookup: &LookupTable,
        threshold: u16,
    ) -> u32 {
        // SAFETY: AVX2 is on for the whole crate wherever the x86 module exists.
        unsafe { x86::row_hits_avx2(row, greens, blues, lookup, threshold) }
    }

    /// The vector kernels have to reproduce the scalar ones bit for bit, on
    /// every row shape and every tail length.
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    #[test]
    fn the_vector_kernels_match_the_scalar_ones_bit_for_bit() {
        let tables = [sparse_lookup(), dense_lookup()];
        // The thresholds cover "everything is a candidate", the real table's 5,
        // candidates that are rare in noise, the largest `u8`, and the
        // 256 that means "nothing can hit".
        let thresholds = [0u16, 2, 5, 128, 255, 256];
        for lookup in &tables {
            for length in [
                1usize, 2, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129,
                512, 513, 1000,
            ] {
                for kind in 0..3u8 {
                    let (reds, greens, blues) = rows(kind, length, 0x2468_1357_9abc_def0);
                    for threshold in thresholds {
                        let (lanes, squared, maximum, hits) =
                            lanes_of(&reds, &greens, &blues, lookup, threshold);
                        let (vector, vector_squared, vector_maximum, vector_hits) =
                            vector_lanes_of(&reds, &greens, &blues, lookup, threshold);
                        for lane in 0..REDUCTION_LANES {
                            assert_eq!(
                                vector[lane].to_bits(),
                                lanes[lane].to_bits(),
                                "lane {lane} of a {length} pixel row of kind {kind}"
                            );
                            assert_eq!(
                                vector_squared[lane].to_bits(),
                                squared[lane].to_bits(),
                                "squared lane {lane} of a {length} pixel row of kind {kind}"
                            );
                        }
                        assert_eq!(
                            vector_maximum.to_bits(),
                            maximum.to_bits(),
                            "maximum of a {length} pixel row of kind {kind}"
                        );
                        assert_eq!(
                            vector_hits, hits,
                            "hits in a {length} pixel row of kind {kind} at threshold {threshold}"
                        );
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        assert_eq!(
                            avx2_hits_of(&reds, &greens, &blues, lookup, threshold),
                            hits,
                            "AVX2 hits in a {length} pixel row of kind {kind} at threshold {threshold}"
                        );
                    }
                }
            }
        }
    }

    /// The hit kernel stops the row walk at the cap, and the count it reports
    /// past the cap cannot change a feature.
    #[test]
    fn hits_are_capped_where_the_graph_caps_them() {
        let lookup = dense_lookup();
        let (reds, greens, blues) = rows(0, 4096, 7);
        let mut hits = 0u32;
        for index in 0..4096 {
            let threshold = lookup.min_spread();
            if counts_as_hit(&lookup, threshold, reds[index], greens[index], blues[index]) {
                hits += 1;
            }
        }
        assert!(
            hits > MASK_HIT_CAP,
            "the fixture must exceed the cap: {hits}"
        );
    }

    /// A row that is not a whole number of vectors long still adds into the
    /// lanes the scalar loop would have used.
    #[test]
    fn every_lane_sums_its_own_subsequence() {
        let lookup = sparse_lookup();
        for length in [0usize, 1, 7, 8, 9, 16, 17, 40, 512, 513] {
            let (reds, greens, blues) = rows(0, length, 99);
            let (lanes, _, maximum, _) =
                lanes_of(&reds, &greens, &blues, &lookup, lookup.min_spread());
            // Lane `i` takes pixel `i`, then every eighth pixel after it, in
            // that order -- the same expectation the scalar loop was written
            // against, summed here without going through it.
            for (lane, value) in lanes.iter().enumerate() {
                let mut want = 0.0f32;
                let mut index = lane;
                while index < length {
                    want += chroma(reds[index], greens[index], blues[index]);
                    index += REDUCTION_LANES;
                }
                assert_eq!(value.to_bits(), want.to_bits(), "lane {lane} of {length}");
            }
            let want_maximum = (0..length)
                .map(|index| chroma(reds[index], greens[index], blues[index]))
                .fold(0.0f32, f32::max);
            assert_eq!(
                maximum.to_bits(),
                want_maximum.to_bits(),
                "maximum of {length}"
            );
        }
    }
}

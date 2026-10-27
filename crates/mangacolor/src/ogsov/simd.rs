//! The OGSOV pixel scan: NEON on aarch64, AVX2 or AVX-512 on x86-64, portable
//! scalar elsewhere.
//!
//! The scalar loop is the reference shape, and it is measured: forming
//! `4*(R-G)^2 + (R+G-2B)^2` and counting it into the histogram is one random
//! memory access per pixel, and a vector form of it was measured *slower* than
//! the scalar one.  The counts have to come back out of the vector registers
//! one at a time before they can be scattered, and that round trip costs more
//! than the arithmetic it saves.
//!
//! What the vector kernels do instead is answer the mask test sixteen (NEON),
//! thirty-two (AVX2) or sixty-four (AVX-512) pixels at a time.  A pixel whose
//! channels are closer together than the table's own
//! `min_spread` cannot be a hit, so a group where no pixel passes that test —
//! which is what a near-gray manga page produces, pixel after pixel — skips the
//! per-pixel maximum, minimum and table probe entirely.
//!
//! Both forms produce the same answer: the histogram holds integer counts, so
//! the order pixels are counted in cannot change a bin, and the hit count is
//! only ever read through `min(hits, 1024)`.

use crate::lookup::LookupTable;
use crate::rgb::RgbView;

/// One frame's distances and lookup hits.
///
/// The histogram is indexed by [`super::distance_squared`], whose largest value
/// is [`super::MAX_D2`] — the last bin — and that bound is asserted by a test.
pub(super) fn scan(
    rgb: &RgbView<'_>,
    histogram: &mut [u32],
    lookup: &LookupTable,
    threshold: u16,
) -> u32 {
    let height = rgb.height();
    let mut hits = 0u32;
    for y in 0..height {
        let reds = rgb.red.row(y);
        let greens = rgb.green.row(y);
        let blues = rgb.blue.row(y);
        // SAFETY: NEON is part of the aarch64 baseline, AVX2 is on for the whole
        // crate wherever the x86 module is compiled, and the kernel reads only the
        // rows it is handed.
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx2")
        ))]
        unsafe {
            hits = vector::row(reds, greens, blues, histogram, lookup, threshold, hits);
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx2")
        )))]
        {
            hits = scalar::row(reds, greens, blues, histogram, lookup, threshold, hits);
        }
    }
    hits
}

/// The portable scan, which is the measured reference shape.
///
/// Where a vector kernel is compiled in it runs instead and this is kept as the
/// definition it has to match, which is why it may be unused.
#[cfg_attr(
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ),
    allow(dead_code)
)]
mod scalar {
    use super::super::{MASK_HIT_CAP, distance_squared};
    use crate::lookup::LookupTable;

    /// Count one row, continuing from the hits counted so far.
    pub(super) fn row(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        histogram: &mut [u32],
        lookup: &LookupTable,
        threshold: u16,
        mut hits: u32,
    ) -> u32 {
        let min_spread = threshold as i32;
        for index in 0..reds.len() {
            let (red, green, blue) = (reds[index], greens[index], blues[index]);
            histogram[distance_squared(red, green, blue)] += 1;
            if hits < MASK_HIT_CAP {
                let high = red.max(green).max(blue) as i32;
                let low = red.min(green).min(blue) as i32;
                if high - low >= min_spread && lookup.contains(blue, green, red) {
                    hits += 1;
                }
            }
        }
        hits
    }
}

/// The aarch64 scan.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::super::{MASK_HIT_CAP, distance_squared};
    use crate::lookup::LookupTable;

    /// How many pixels one mask test covers.
    const WIDE: usize = 16;

    /// Count one row, continuing from the hits counted so far.
    ///
    /// SAFETY: the three slices are the same length, and the histogram holds
    /// one bin per possible distance.
    pub(super) unsafe fn row(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        histogram: &mut [u32],
        lookup: &LookupTable,
        threshold: u16,
        mut hits: u32,
    ) -> u32 {
        unsafe {
            let count = reds.len();
            // 256 means "nothing can hit"; narrowing that to `u8` would wrap
            // to zero and make every pixel a candidate.
            let min_spread = vdupq_n_u8(threshold.min(255) as u8);
            let min_spread_wide = threshold as i32;
            let mut index = 0usize;
            while index + WIDE <= count {
                // The distance and its count, in the scalar loop's own shape: one
                // random access per pixel, which is what the histogram costs.
                for offset in index..index + WIDE {
                    let (red, green, blue) = (reds[offset], greens[offset], blues[offset]);
                    // SAFETY: the largest distance is `MAX_D2`, the last bin.
                    *histogram.get_unchecked_mut(distance_squared(red, green, blue)) += 1;
                }
                if hits < MASK_HIT_CAP {
                    let red = vld1q_u8(reds.as_ptr().add(index));
                    let green = vld1q_u8(greens.as_ptr().add(index));
                    let blue = vld1q_u8(blues.as_ptr().add(index));
                    let spread = vsubq_u8(
                        vmaxq_u8(vmaxq_u8(red, green), blue),
                        vminq_u8(vminq_u8(red, green), blue),
                    );
                    if vmaxvq_u8(vcgeq_u8(spread, min_spread)) != 0 {
                        // Only a group with a candidate is walked pixel by pixel.
                        for offset in index..index + WIDE {
                            let (red, green, blue) = (reds[offset], greens[offset], blues[offset]);
                            let high = red.max(green).max(blue) as i32;
                            let low = red.min(green).min(blue) as i32;
                            if high - low >= min_spread_wide && lookup.contains(blue, green, red) {
                                hits += 1;
                            }
                        }
                        hits = hits.min(MASK_HIT_CAP);
                    }
                }
                index += WIDE;
            }
            // The remainder is shorter than one group, so it goes one pixel at a
            // time.
            while index < count {
                let (red, green, blue) = (reds[index], greens[index], blues[index]);
                // SAFETY: as above.
                *histogram.get_unchecked_mut(distance_squared(red, green, blue)) += 1;
                if hits < MASK_HIT_CAP {
                    let high = red.max(green).max(blue) as i32;
                    let low = red.min(green).min(blue) as i32;
                    if high - low >= min_spread_wide && lookup.contains(blue, green, red) {
                        hits += 1;
                    }
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

/// The x86-64 scan, compiled in only where the build was told the CPU level.
///
/// The scatter is the scalar one and only the mask test is vectorised, as on
/// NEON.  The AVX2 build answers it thirty-two pixels at a time with a byte
/// compare and `movemask`; the AVX-512 build sixty-four at a time with a
/// compare straight into a mask register.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86 {
    use std::arch::x86_64::*;

    use super::super::{MASK_HIT_CAP, distance_squared};
    use crate::lookup::LookupTable;

    /// How many pixels one mask test covers.
    #[cfg(target_feature = "avx512bw")]
    const WIDE: usize = 64;
    #[cfg(not(target_feature = "avx512bw"))]
    const WIDE: usize = 32;

    /// The candidates in one group: bit `i` is set when pixel `index + i` has a
    /// channel spread of at least `min_spread`.
    ///
    /// SAFETY: `WIDE` bytes are readable at `index` in every slice.
    #[cfg(not(target_feature = "avx512bw"))]
    #[inline]
    unsafe fn candidates(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        index: usize,
        min_spread: __m256i,
    ) -> u64 {
        unsafe {
            let red = _mm256_loadu_si256(reds.as_ptr().add(index) as *const __m256i);
            let green = _mm256_loadu_si256(greens.as_ptr().add(index) as *const __m256i);
            let blue = _mm256_loadu_si256(blues.as_ptr().add(index) as *const __m256i);
            let spread = _mm256_sub_epi8(
                _mm256_max_epu8(_mm256_max_epu8(red, green), blue),
                _mm256_min_epu8(_mm256_min_epu8(red, green), blue),
            );
            // `spread >= min_spread`, as an unsigned byte compare.
            let mask = _mm256_cmpeq_epi8(_mm256_max_epu8(spread, min_spread), spread);
            _mm256_movemask_epi8(mask) as u32 as u64
        }
    }

    /// As above, sixty-four pixels into a mask register.
    ///
    /// SAFETY: `WIDE` bytes are readable at `index` in every slice.
    #[cfg(target_feature = "avx512bw")]
    #[inline]
    unsafe fn candidates(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        index: usize,
        min_spread: __m512i,
    ) -> u64 {
        unsafe {
            let red = _mm512_loadu_si512(reds.as_ptr().add(index) as *const _);
            let green = _mm512_loadu_si512(greens.as_ptr().add(index) as *const _);
            let blue = _mm512_loadu_si512(blues.as_ptr().add(index) as *const _);
            let spread = _mm512_sub_epi8(
                _mm512_max_epu8(_mm512_max_epu8(red, green), blue),
                _mm512_min_epu8(_mm512_min_epu8(red, green), blue),
            );
            _mm512_cmpge_epu8_mask(spread, min_spread)
        }
    }

    /// Count thirty-two pixels into the histogram.
    ///
    /// The distances are formed in vector registers: `4*(R-G)^2 + (R+G-2B)^2` is
    /// `(2(R-G))^2 + (R+G-2B)^2`, and both terms fit in an `i16`, so one
    /// `_mm256_madd_epi16` over the interleaved pair gives the sum of both
    /// squares as an exact `i32`.  The lanes come out in a different order from
    /// the pixels, which does not matter: the histogram only holds counts.
    ///
    /// A group whose thirty-two distances are all the same bin -- a white, black
    /// or flat gray area, which is most of a manga page -- is added with one
    /// store instead of thirty-two read-modify-writes to the same word, which
    /// would each wait for the one before.
    ///
    /// SAFETY: thirty-two bytes are readable at each pointer, and the histogram
    /// holds one bin per possible distance.
    #[inline]
    unsafe fn count32(reds: *const u8, greens: *const u8, blues: *const u8, histogram: *mut u32) {
        unsafe {
            let zero = _mm256_setzero_si256();
            let red = _mm256_loadu_si256(reds as *const __m256i);
            let green = _mm256_loadu_si256(greens as *const __m256i);
            let blue = _mm256_loadu_si256(blues as *const __m256i);

            #[inline(always)]
            unsafe fn squares(red: __m256i, green: __m256i, blue: __m256i) -> (__m256i, __m256i) {
                unsafe {
                    let twice_dx = _mm256_slli_epi16::<1>(_mm256_sub_epi16(red, green));
                    let yb = _mm256_sub_epi16(
                        _mm256_add_epi16(red, green),
                        _mm256_slli_epi16::<1>(blue),
                    );
                    let low = _mm256_unpacklo_epi16(twice_dx, yb);
                    let high = _mm256_unpackhi_epi16(twice_dx, yb);
                    (_mm256_madd_epi16(low, low), _mm256_madd_epi16(high, high))
                }
            }

            let (d0, d1) = squares(
                _mm256_unpacklo_epi8(red, zero),
                _mm256_unpacklo_epi8(green, zero),
                _mm256_unpacklo_epi8(blue, zero),
            );
            let (d2, d3) = squares(
                _mm256_unpackhi_epi8(red, zero),
                _mm256_unpackhi_epi8(green, zero),
                _mm256_unpackhi_epi8(blue, zero),
            );

            let first = _mm256_broadcastd_epi32(_mm256_castsi256_si128(d0));
            let same = _mm256_and_si256(
                _mm256_and_si256(_mm256_cmpeq_epi32(d0, first), _mm256_cmpeq_epi32(d1, first)),
                _mm256_and_si256(_mm256_cmpeq_epi32(d2, first), _mm256_cmpeq_epi32(d3, first)),
            );
            if _mm256_movemask_epi8(same) == -1 {
                let bin = _mm_cvtsi128_si32(_mm256_castsi256_si128(d0)) as usize;
                *histogram.add(bin) += 32;
                return;
            }

            let mut distances = [0i32; 32];
            _mm256_storeu_si256(distances.as_mut_ptr() as *mut __m256i, d0);
            _mm256_storeu_si256(distances.as_mut_ptr().add(8) as *mut __m256i, d1);
            _mm256_storeu_si256(distances.as_mut_ptr().add(16) as *mut __m256i, d2);
            _mm256_storeu_si256(distances.as_mut_ptr().add(24) as *mut __m256i, d3);
            for distance in distances {
                *histogram.add(distance as usize) += 1;
            }
        }
    }

    /// Count one row, continuing from the hits counted so far.
    ///
    /// SAFETY: the three slices are the same length, and the histogram holds
    /// one bin per possible distance.
    pub(super) unsafe fn row(
        reds: &[u8],
        greens: &[u8],
        blues: &[u8],
        histogram: &mut [u32],
        lookup: &LookupTable,
        threshold: u16,
        mut hits: u32,
    ) -> u32 {
        unsafe {
            let count = reds.len();
            // 256 means "nothing can hit"; narrowing that to `u8` would wrap
            // to zero and make every pixel a candidate.
            #[cfg(target_feature = "avx512bw")]
            let min_spread = _mm512_set1_epi8(threshold.min(255) as u8 as i8);
            #[cfg(not(target_feature = "avx512bw"))]
            let min_spread = _mm256_set1_epi8(threshold.min(255) as u8 as i8);
            let min_spread_wide = threshold as i32;
            let mut index = 0usize;
            while index + WIDE <= count {
                // The distances are formed a vector at a time and counted from a
                // small array, or in one store when the group is all one bin.
                let mut offset = index;
                while offset < index + WIDE {
                    // SAFETY: the largest distance is `MAX_D2`, the last bin.
                    count32(
                        reds.as_ptr().add(offset),
                        greens.as_ptr().add(offset),
                        blues.as_ptr().add(offset),
                        histogram.as_mut_ptr(),
                    );
                    offset += 32;
                }
                if hits < MASK_HIT_CAP {
                    let mut bits = candidates(reds, greens, blues, index, min_spread);
                    // Only the candidates are looked at, one set bit each.
                    while bits != 0 {
                        let offset = index + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let (red, green, blue) = (reds[offset], greens[offset], blues[offset]);
                        let high = red.max(green).max(blue) as i32;
                        let low = red.min(green).min(blue) as i32;
                        if high - low >= min_spread_wide && lookup.contains(blue, green, red) {
                            hits += 1;
                        }
                    }
                    hits = hits.min(MASK_HIT_CAP);
                }
                index += WIDE;
            }
            // The remainder is shorter than one group, so it goes one pixel at a
            // time.
            while index < count {
                let (red, green, blue) = (reds[index], greens[index], blues[index]);
                // SAFETY: as above.
                *histogram.get_unchecked_mut(distance_squared(red, green, blue)) += 1;
                if hits < MASK_HIT_CAP {
                    let high = red.max(green).max(blue) as i32;
                    let low = red.min(green).min(blue) as i32;
                    if high - low >= min_spread_wide && lookup.contains(blue, green, red) {
                        hits += 1;
                    }
                }
                index += 1;
            }
            hits
        }
    }
}

#[cfg(all(
    test,
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )
))]
mod tests {
    use super::super::{HISTOGRAM_LEN, MASK_HIT_CAP};
    use super::*;

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

    fn row_of(kind: u8, length: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut stream = Stream(seed);
        let mut reds = Vec::with_capacity(length);
        let mut greens = Vec::with_capacity(length);
        let mut blues = Vec::with_capacity(length);
        for index in 0..length {
            let (red, green, blue) = match kind {
                0 => (stream.next(), stream.next(), stream.next()),
                1 => {
                    let grey = 100 + (index % 3) as u8;
                    (grey, grey + (index % 2) as u8, grey)
                }
                _ => (17, 17, 17),
            };
            reds.push(red);
            greens.push(green);
            blues.push(blue);
        }
        (reds, greens, blues)
    }

    fn tables() -> [LookupTable; 2] {
        let sparse: Vec<bool> = (0..(1 << 16)).map(|index| index % 3 == 0).collect();
        let dense: Vec<bool> = (0..crate::lookup::MASK_ELEMENTS)
            .map(|index| index % 4 != 0)
            .collect();
        [
            LookupTable::from_bool_slice(&sparse, 2),
            LookupTable::from_bool_slice(&dense, 2),
        ]
    }

    /// The vector scan has to leave the same histogram and report the same
    /// capped hit count as the scalar one, on every row shape and tail length,
    /// including a count that starts at or beyond the cap.
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    #[test]
    fn the_vector_scan_matches_the_scalar_one() {
        let thresholds = [0u16, 2, 5, 128, 255, 256];
        for lookup in &tables() {
            for length in [
                1usize, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 512, 513, 1000,
                4096,
            ] {
                for kind in 0..3u8 {
                    let (reds, greens, blues) = row_of(kind, length, 0x2468_1357_9abc_def0);
                    for threshold in thresholds {
                        for start in [0u32, MASK_HIT_CAP - 3, MASK_HIT_CAP] {
                            let mut want = vec![0u32; HISTOGRAM_LEN];
                            let mut got = vec![0u32; HISTOGRAM_LEN];
                            let expected = scalar::row(
                                &reds, &greens, &blues, &mut want, lookup, threshold, start,
                            );
                            // SAFETY: the slices are one length and the histogram has
                            // a bin per distance; the CPU level is the build's.
                            let actual = unsafe {
                                vector::row(
                                    &reds, &greens, &blues, &mut got, lookup, threshold, start,
                                )
                            };
                            assert_eq!(
                                actual, expected,
                                "hits of a {length} pixel row of kind {kind} at {threshold} from {start}"
                            );
                            assert!(want == got, "histogram of a {length} pixel row");
                        }
                    }
                }
            }
        }
    }
}

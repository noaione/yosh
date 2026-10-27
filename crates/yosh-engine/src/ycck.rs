//! Adobe YCCK (`Y, Cb, Cr, K` with K inverted) to opaque RGBA8.
//!
//! One fixed-point definition feeds every path, so the NEON/AVX2 kernels and the
//! portable loop write identical bytes:
//!
//! ```text
//! R = Y + ((91881 * (Cr - 128) + 32768) >> 16)              // 1.40200
//! G = Y - ((22554 * (Cb - 128) + 46802 * (Cr - 128) + 32768) >> 16)
//! B = Y + ((116130 * (Cb - 128) + 32768) >> 16)             // 1.77200
//! out = (t + (t >> 8)) >> 8        where t = channel * K + 128   // ~channel*K/255
//! ```
//!
//! K is inverted (255 = no black ink), which is the Adobe convention, so each
//! display channel is scaled by `K / 255`.
//!
//! NEON is part of the aarch64 baseline, so that kernel is used unconditionally
//! there. On x86-64 the AVX2 kernel is compiled in and selected by a runtime
//! `is_x86_feature_detected!` check, so a baseline-built binary still uses it on
//! an AVX2 CPU.

/// Convert one frame's YCCK samples to opaque RGBA8.
///
/// `ycck` is the raw four-sample run zune-jpeg hands back for `ColorSpace::YCCK`
/// (one sample per component per pixel, row-major). Interior bytes are ignored.
pub(super) fn to_rgba(ycck: &[u8]) -> Vec<u8> {
    let pixels = ycck.len() / 4;
    let mut out = vec![0u8; pixels * 4];

    #[cfg(target_arch = "aarch64")]
    let done = neon::convert(ycck, &mut out);

    #[cfg(not(target_arch = "aarch64"))]
    let done = {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: the detection above proves AVX2 is available, and
                // both slices hold `pixels` whole samples.
                unsafe { avx2::convert(ycck, &mut out) }
            } else {
                0
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            0
        }
    };

    scalar::convert(&ycck[done * 4..], &mut out[done * 4..]);
    out
}

/// The portable reference, one pixel at a time.
mod scalar {
    pub(super) fn convert(input: &[u8], output: &mut [u8]) {
        for (px, dst) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<4>().0)
        {
            let (r, g, b) = ycbcr_to_rgb(px[0], px[1], px[2]);
            let coverage = px[3] as u32;
            dst[0] = scale(r, coverage);
            dst[1] = scale(g, coverage);
            dst[2] = scale(b, coverage);
            dst[3] = 255;
        }
    }

    /// BT.601 full-range YCbCr -> RGB in Q16 fixed point.
    #[inline]
    pub(super) fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
        let y = y as i32;
        let cb = cb as i32 - 128;
        let cr = cr as i32 - 128;
        let clamp = |value: i32| value.clamp(0, 255) as u8;
        (
            clamp(y + ((91881 * cr + 32768) >> 16)),
            clamp(y - ((22554 * cb + 46802 * cr + 32768) >> 16)),
            clamp(y + ((116130 * cb + 32768) >> 16)),
        )
    }

    /// Scale one display channel by the inverted black coverage.
    #[inline]
    pub(super) fn scale(value: u8, coverage: u32) -> u8 {
        let t = value as u32 * coverage + 128;
        ((t + (t >> 8)) >> 8) as u8
    }
}

/// The aarch64 kernel: sixteen pixels at a time.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Pixels one iteration consumes; `vld4q_u8`/`vst4q_u8` move them at once.
    const WIDE: usize = 16;

    /// Convert whole sixteen-pixel blocks; returns how many pixels were written.
    pub(super) fn convert(input: &[u8], output: &mut [u8]) -> usize {
        let pixels = input.len() / 4;
        let mut index = 0usize;
        // SAFETY: `index + WIDE <= pixels` guarantees the loads/stores address
        // `WIDE` whole samples inside both slices, and NEON is part of the
        // aarch64 baseline.
        unsafe {
            while index + WIDE <= pixels {
                let block = vld4q_u8(input.as_ptr().add(index * 4));
                let low = rgba8(
                    vget_low_u8(block.0),
                    vget_low_u8(block.1),
                    vget_low_u8(block.2),
                    vget_low_u8(block.3),
                );
                let high = rgba8(
                    vget_high_u8(block.0),
                    vget_high_u8(block.1),
                    vget_high_u8(block.2),
                    vget_high_u8(block.3),
                );
                let out = uint8x16x4_t(
                    vcombine_u8(low.0, high.0),
                    vcombine_u8(low.1, high.1),
                    vcombine_u8(low.2, high.2),
                    vdupq_n_u8(255),
                );
                vst4q_u8(output.as_mut_ptr().add(index * 4), out);
                index += WIDE;
            }
        }
        index
    }

    /// Eight pixels -> `(R, G, B)`, with the coverage applied.
    ///
    /// # Safety
    ///
    /// The caller guarantees the vectors are live NEON registers; no memory is
    /// addressed here.
    #[inline]
    unsafe fn rgba8(
        y: uint8x8_t,
        cb: uint8x8_t,
        cr: uint8x8_t,
        k: uint8x8_t,
    ) -> (uint8x8_t, uint8x8_t, uint8x8_t) {
        unsafe {
            let y16 = vmovl_u8(y);
            let k16 = vmovl_u8(k);
            let cb16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb)), vdupq_n_s16(128));
            let cr16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr)), vdupq_n_s16(128));

            let (r0, g0, b0) = rgb4(vget_low_u16(y16), vget_low_s16(cb16), vget_low_s16(cr16));
            let (r1, g1, b1) = rgb4(vget_high_u16(y16), vget_high_s16(cb16), vget_high_s16(cr16));

            let coverage = vcombine_u16(vget_low_u16(k16), vget_high_u16(k16));
            let red = vcombine_u16(vqmovun_s32(r0), vqmovun_s32(r1));
            let green = vcombine_u16(vqmovun_s32(g0), vqmovun_s32(g1));
            let blue = vcombine_u16(vqmovun_s32(b0), vqmovun_s32(b1));
            (
                narrow_scale(red, coverage),
                narrow_scale(green, coverage),
                narrow_scale(blue, coverage),
            )
        }
    }

    /// Four pixels' YCbCr -> clamped RGB, one i32 lane each.
    ///
    /// # Safety
    ///
    /// Vectors are live NEON registers; no memory is addressed here.
    #[inline]
    unsafe fn rgb4(
        y: uint16x4_t,
        cb: int16x4_t,
        cr: int16x4_t,
    ) -> (int32x4_t, int32x4_t, int32x4_t) {
        unsafe {
            let y = vreinterpretq_s32_u32(vmovl_u16(y));
            let cb = vmovl_s16(cb);
            let cr = vmovl_s16(cr);
            let round = vdupq_n_s32(32768);
            let red = vaddq_s32(
                y,
                vshrq_n_s32::<16>(vaddq_s32(vmulq_s32(cr, vdupq_n_s32(91881)), round)),
            );
            let green_vector = vaddq_s32(
                vmulq_s32(cb, vdupq_n_s32(22554)),
                vmulq_s32(cr, vdupq_n_s32(46802)),
            );
            let green = vsubq_s32(y, vshrq_n_s32::<16>(vaddq_s32(green_vector, round)));
            let blue = vaddq_s32(
                y,
                vshrq_n_s32::<16>(vaddq_s32(vmulq_s32(cb, vdupq_n_s32(116130)), round)),
            );
            let zero = vdupq_n_s32(0);
            let max = vdupq_n_s32(255);
            (
                vminq_s32(vmaxq_s32(red, zero), max),
                vminq_s32(vmaxq_s32(green, zero), max),
                vminq_s32(vmaxq_s32(blue, zero), max),
            )
        }
    }

    /// `(t + (t >> 8)) >> 8` for `t = value * coverage + 128`, narrowed to u8.
    ///
    /// # Safety
    ///
    /// Vectors are live NEON registers; no memory is addressed here.
    #[inline]
    unsafe fn narrow_scale(value: uint16x8_t, coverage: uint16x8_t) -> uint8x8_t {
        unsafe {
            let t = vaddq_u16(vmulq_u16(value, coverage), vdupq_n_u16(128));
            vshrn_n_u16::<8>(vaddq_u16(t, vshrq_n_u16::<8>(t)))
        }
    }
}

/// The x86-64 kernel: eight pixels at a time, selected at runtime.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Pixels one iteration consumes.
    const WIDE: usize = 8;

    /// Convert whole eight-pixel blocks; returns how many pixels were written.
    ///
    /// # Safety
    ///
    /// The caller must have verified (e.g. `is_x86_feature_detected!`) that the
    /// CPU supports AVX2, and both slices must hold `input.len() / 4` whole
    /// samples.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn convert(input: &[u8], output: &mut [u8]) -> usize {
        unsafe {
            let pixels = input.len() / 4;
            let mut index = 0usize;
            let alpha = _mm256_set1_epi32(255);
            let transpose = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2,
                6, 10, 14, 3, 7, 11, 15,
            );
            while index + WIDE <= pixels {
                let block = _mm256_loadu_si256(input.as_ptr().add(index * 4) as *const __m256i);
                let y = channel(block, 0);
                let cb = _mm256_sub_epi32(channel(block, 1), _mm256_set1_epi32(128));
                let cr = _mm256_sub_epi32(channel(block, 2), _mm256_set1_epi32(128));
                let coverage = channel(block, 3);

                let red = clamp(_mm256_add_epi32(y, scaled_delta(cr, 91881)));
                let green = clamp(_mm256_sub_epi32(
                    y,
                    _mm256_srai_epi32::<16>(_mm256_add_epi32(
                        _mm256_add_epi32(
                            _mm256_mullo_epi32(cb, _mm256_set1_epi32(22554)),
                            _mm256_mullo_epi32(cr, _mm256_set1_epi32(46802)),
                        ),
                        _mm256_set1_epi32(32768),
                    )),
                ));
                let blue = clamp(_mm256_add_epi32(y, scaled_delta(cb, 116130)));

                let red = apply_coverage(red, coverage);
                let green = apply_coverage(green, coverage);
                let blue = apply_coverage(blue, coverage);

                // Pack i32 -> u16 -> u8. `packus` works per 128-bit lane, so the
                // channel bytes arrive as [r0..r3, g0..g3, b0..b3, a0..a3] in
                // each lane; the final shuffle interleaves them into RGBA.
                let rg = _mm256_packus_epi32(red, green);
                let ba = _mm256_packus_epi32(blue, alpha);
                let packed = _mm256_packus_epi16(rg, ba);
                let rgba = _mm256_shuffle_epi8(packed, transpose);
                _mm256_storeu_si256(output.as_mut_ptr().add(index * 4) as *mut __m256i, rgba);
                index += WIDE;
            }
            index
        }
    }

    /// Gather one component's eight bytes (in pixel order) and widen to i32.
    ///
    /// # Safety
    ///
    /// The caller guarantees AVX2; `index` is a component byte offset 0..=3.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn channel(block: __m256i, index: i32) -> __m256i {
        // Each 128-bit lane holds four pixels (16 bytes). Pull this lane's
        // component bytes to the front, then merge both lanes into one
        // 64-bit group: bytes 0..7 become the eight pixels in order.
        let mask = _mm256_setr_epi8(
            index as i8,
            (index + 4) as i8,
            (index + 8) as i8,
            (index + 12) as i8,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            (index + 16) as i8,
            (index + 20) as i8,
            (index + 24) as i8,
            (index + 28) as i8,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        );
        let shuffled = _mm256_shuffle_epi8(block, mask);
        let merged =
            _mm256_permutevar8x32_epi32(shuffled, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_cvtepu8_epi32(_mm256_castsi256_si128(merged))
    }

    /// `((value * coefficient + 32768) >> 16)` for the red/blue channels.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn scaled_delta(values: __m256i, coefficient: i32) -> __m256i {
        _mm256_srai_epi32::<16>(_mm256_add_epi32(
            _mm256_mullo_epi32(values, _mm256_set1_epi32(coefficient)),
            _mm256_set1_epi32(32768),
        ))
    }

    /// Clamp to `0..=255`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn clamp(values: __m256i) -> __m256i {
        _mm256_min_epi32(
            _mm256_max_epi32(values, _mm256_setzero_si256()),
            _mm256_set1_epi32(255),
        )
    }

    /// `(t + (t >> 8)) >> 8` for `t = value * coverage + 128`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn apply_coverage(values: __m256i, coverage: __m256i) -> __m256i {
        let t = _mm256_add_epi32(_mm256_mullo_epi32(values, coverage), _mm256_set1_epi32(128));
        _mm256_srli_epi32::<8>(_mm256_add_epi32(t, _mm256_srli_epi32::<8>(t)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic samples, mostly no-ink K but with a share of partial coverage.
    fn samples(count: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(count * 4);
        for _ in 0..count {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let coverage = match (state >> 40) & 3 {
                0 => 255,
                1 => 0,
                2 => (state >> 48) as u8,
                _ => 128,
            };
            out.extend_from_slice(&[
                (state >> 8) as u8,
                (state >> 16) as u8,
                (state >> 24) as u8,
                coverage,
            ]);
        }
        out
    }

    #[test]
    fn the_fast_path_matches_the_portable_one() {
        for (count, seed) in [
            (0usize, 1u64),
            (1, 2),
            (7, 3),
            (8, 4),
            (9, 5),
            (15, 6),
            (16, 7),
            (17, 8),
            (31, 9),
            (32, 10),
            (33, 11),
            (100, 12),
            (257, 13),
        ] {
            let input = samples(count, seed);
            let mut want = vec![0u8; count * 4];
            scalar::convert(&input, &mut want);
            assert_eq!(to_rgba(&input), want, "{count} pixels, seed {seed}");
        }
    }

    /// The differential test above would still pass if the vector kernel
    /// silently handled nothing, so pin that it consumes whole blocks.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_neon_kernel_consumes_whole_blocks() {
        let input = samples(32, 99);
        let mut out = vec![0u8; 32 * 4];
        assert_eq!(neon::convert(&input, &mut out), 32);
    }

    /// On x86-64 the AVX2 kernel is compile-time optional and selected at
    /// runtime; when the CPU has it, it must actually run.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_avx2_kernel_consumes_whole_blocks_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let input = samples(24, 7);
        let mut out = vec![0u8; 24 * 4];
        // SAFETY: the runtime check above proves AVX2 is available.
        assert_eq!(unsafe { avx2::convert(&input, &mut out) }, 24);
    }

    #[test]
    fn the_known_values_are_exact() {
        let cases: &[([u8; 4], [u8; 4])] = &[
            ([255, 128, 128, 255], [255, 255, 255, 255]), // white, no black ink
            ([0, 128, 128, 0], [0, 0, 0, 255]),           // black, full black ink
            ([179, 171, 1, 255], [1, 255, 255, 255]),     // cyan
            ([255, 128, 128, 128], [128, 128, 128, 255]), // white at half coverage
        ];
        for (input, want) in cases {
            assert_eq!(&to_rgba(input)[..], &want[..], "{input:?}");
        }
    }
}

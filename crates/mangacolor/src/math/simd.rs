//! The transposed dense kernel: NEON on aarch64, AVX2 or AVX-512 on x86-64,
//! portable scalar elsewhere.
//!
//! The scalar loop below is the definition.  Every vector kernel here has to
//! reproduce it bit for bit, and the reason it can is the *shape* of the work
//! rather than any numerical trick: one output neuron owns one lane of the
//! accumulators, and that lane adds `w[c] * x[c]` for `c` ascending, from the
//! same two values, in the same order, with a multiply and then an add — never a
//! fused one.  A lane's sum is therefore the scalar loop's sum, bit for bit, no
//! matter how many outputs are in flight beside it.
//!
//! The weight layout is what makes it a vector kernel: transposed to `(cols,
//! rows)` — the copy [`crate::weights`] builds once at load — the outputs a
//! group of lanes covers are a contiguous run of `f32`, so a group is one load a
//! column.  In the asset's own output-major layout the same group would be four
//! strided loads, which is what the scalar `dense` in `math.rs` costs.
//!
//! See [`super::dense_transposed`] for the contract and the callers.

/// The portable kernel, which every vector one is measured against.
///
/// On a target with a vector kernel this is kept for the differential test and
/// otherwise unused.
#[cfg_attr(
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ),
    allow(dead_code)
)]
mod scalar {
    /// `out[row] = sum_c w[c * rows + row] * x[c] + bias[row]`.
    ///
    /// This is [`crate::math::dense_transposed`]'s definition: one accumulator
    /// per output neuron, the products added in ascending `c`, the bias last.
    pub(super) fn dense_transposed(
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        bias: &[f32],
        out: &mut [f32],
    ) {
        for row in 0..rows {
            let mut accumulator = 0.0f32;
            for column in 0..cols {
                accumulator += w[column * rows + row] * x[column];
            }
            out[row] = accumulator + bias[row];
        }
    }
}

/// The aarch64 kernel: four output neurons per vector of accumulators.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// How many output neurons one vector of accumulators covers.
    const LANES: usize = 4;

    /// One group of [`LANES`] outputs, accumulated together.
    ///
    /// # Safety
    ///
    /// `row + LANES <= rows`, the slices hold what their lengths say, and `w` is
    /// `(cols, rows)`.
    #[inline]
    unsafe fn group(
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        bias: &[f32],
        out: &mut [f32],
        row: usize,
    ) {
        unsafe {
            let mut accumulator = vdupq_n_f32(0.0);
            for column in 0..cols {
                // The four outputs this group covers are adjacent in the
                // transposed layout, so they are one contiguous load.
                let weights = vld1q_f32(w.as_ptr().add(column * rows + row));
                let value = vld1q_dup_f32(x.as_ptr().add(column));
                accumulator = vaddq_f32(accumulator, vmulq_f32(weights, value));
            }
            let mut sums = [0.0f32; LANES];
            vst1q_f32(sums.as_mut_ptr(), accumulator);
            for (offset, (&sum, &bias_value)) in
                sums.iter().zip(&bias[row..row + LANES]).enumerate()
            {
                *out.get_unchecked_mut(row + offset) = sum + bias_value;
            }
        }
    }

    /// The row walk; see the module documentation.
    ///
    /// # Safety
    ///
    /// The slices hold `rows`, `rows * cols`, `rows` and `rows` values
    /// respectively, and the CPU is aarch64 with NEON, which the architecture
    /// guarantees.
    pub(super) unsafe fn dense_transposed(
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        bias: &[f32],
        out: &mut [f32],
    ) {
        unsafe {
            let mut row = 0usize;
            while row + LANES <= rows {
                group(x, w, rows, cols, bias, out, row);
                row += LANES;
            }
            // Fewer outputs left than one vector covers: the scalar loop's own
            // shape, on the rows the vector groups did not take.
            while row < rows {
                let mut accumulator = 0.0f32;
                for column in 0..cols {
                    accumulator += *w.get_unchecked(column * rows + row) * *x.get_unchecked(column);
                }
                *out.get_unchecked_mut(row) = accumulator + *bias.get_unchecked(row);
                row += 1;
            }
        }
    }
}

/// The x86-64 kernel, compiled in only where the build was told the CPU level.
///
/// AVX2 covers eight output neurons per vector and an AVX-512 build sixteen;
/// both keep one neuron a lane.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86 {
    use std::arch::x86_64::*;

    /// How many output neurons one vector of accumulators covers.
    #[cfg(target_feature = "avx512f")]
    const LANES: usize = 16;
    #[cfg(not(target_feature = "avx512f"))]
    const LANES: usize = 8;

    /// One group of [`LANES`] outputs, accumulated together.
    ///
    /// # Safety
    ///
    /// `row + LANES <= rows`, the slices hold what their lengths say, and `w` is
    /// `(cols, rows)`.
    #[cfg(not(target_feature = "avx512f"))]
    #[inline]
    unsafe fn group(
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        bias: &[f32],
        out: &mut [f32],
        row: usize,
    ) {
        unsafe {
            let mut accumulator = _mm256_setzero_ps();
            for column in 0..cols {
                let weights = _mm256_loadu_ps(w.as_ptr().add(column * rows + row));
                let value = _mm256_broadcast_ss(x.get_unchecked(column));
                accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(weights, value));
            }
            let mut sums = [0.0f32; LANES];
            _mm256_storeu_ps(sums.as_mut_ptr(), accumulator);
            for (offset, (&sum, &bias_value)) in
                sums.iter().zip(&bias[row..row + LANES]).enumerate()
            {
                *out.get_unchecked_mut(row + offset) = sum + bias_value;
            }
        }
    }

    /// As above, sixteen outputs at a time.
    ///
    /// # Safety
    ///
    /// As above.
    #[cfg(target_feature = "avx512f")]
    #[inline]
    unsafe fn group(
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        bias: &[f32],
        out: &mut [f32],
        row: usize,
    ) {
        unsafe {
            let mut accumulator = _mm512_setzero_ps();
            for column in 0..cols {
                let weights = _mm512_loadu_ps(w.as_ptr().add(column * rows + row));
                let value = _mm512_set1_ps(*x.get_unchecked(column));
                accumulator = _mm512_add_ps(accumulator, _mm512_mul_ps(weights, value));
            }
            let mut sums = [0.0f32; LANES];
            _mm512_storeu_ps(sums.as_mut_ptr(), accumulator);
            for (offset, (&sum, &bias_value)) in
                sums.iter().zip(&bias[row..row + LANES]).enumerate()
            {
                *out.get_unchecked_mut(row + offset) = sum + bias_value;
            }
        }
    }

    /// The row walk; see the module documentation.
    ///
    /// # Safety
    ///
    /// The slices hold `rows`, `rows * cols`, `rows` and `rows` values
    /// respectively, and the build was told the CPU level the module compiles
    /// for.
    pub(super) unsafe fn dense_transposed(
        x: &[f32],
        w: &[f32],
        rows: usize,
        cols: usize,
        bias: &[f32],
        out: &mut [f32],
    ) {
        unsafe {
            let mut row = 0usize;
            while row + LANES <= rows {
                group(x, w, rows, cols, bias, out, row);
                row += LANES;
            }
            while row < rows {
                let mut accumulator = 0.0f32;
                for column in 0..cols {
                    accumulator += *w.get_unchecked(column * rows + row) * *x.get_unchecked(column);
                }
                *out.get_unchecked_mut(row) = accumulator + *bias.get_unchecked(row);
                row += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
use neon as vector;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use x86 as vector;

/// Run [`super::dense_transposed`] on whatever kernel this build has.
///
/// # Safety
///
/// The caller has already checked the four slice lengths against `rows` and
/// `cols`, and `w` really is the transposed `(cols, rows)` layout.
pub(super) unsafe fn dense_transposed(
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    // SAFETY: the caller checked the geometry and the build is the CPU level the
    // `vector` module was compiled for.
    unsafe {
        vector::dense_transposed(x, w, rows, cols, bias, out)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    scalar::dense_transposed(x, w, rows, cols, bias, out)
}

#[cfg(all(
    test,
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )
))]
mod tests {
    use super::*;

    /// A deterministic stream of `f32`s, so a failure can be replayed.
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> f32 {
            let mut state = self.0;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            self.0 = state;
            // A spread of magnitudes and signs, so a summation order that
            // differed would have something to differ about.
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            (unit - 0.5) * 8.0
        }
    }

    fn inputs(rows: usize, cols: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut stream = Stream(seed);
        (
            (0..cols).map(|_| stream.next()).collect(),
            (0..rows * cols).map(|_| stream.next()).collect(),
            (0..rows).map(|_| stream.next()).collect(),
        )
    }

    /// The vector kernel has to reproduce the scalar one bit for bit, on every
    /// row count — including the ones no vector divides — and every column
    /// count, including zero.
    #[test]
    fn the_vector_kernel_matches_the_scalar_one_bit_for_bit() {
        for rows in [
            0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 63, 64, 65, 127, 128,
            129, 256, 257, 512, 513,
        ] {
            for cols in [0usize, 1, 2, 3, 5, 7, 8, 9, 16, 32, 64] {
                let (x, w, bias) = inputs(rows, cols, 0x9e37_79b9_7f4a_7c15);
                let mut want = vec![0.0f32; rows];
                let mut got = vec![0.0f32; rows];
                scalar::dense_transposed(&x, &w, rows, cols, &bias, &mut want);
                // SAFETY: the slices are the sizes the kernel reads, built above.
                unsafe { vector::dense_transposed(&x, &w, rows, cols, &bias, &mut got) };
                for row in 0..rows {
                    assert_eq!(
                        got[row].to_bits(),
                        want[row].to_bits(),
                        "row {row} of a {rows}x{cols} matrix"
                    );
                }
            }
        }
    }

    /// The lane a group covers is the output neuron at that offset, not a sum
    /// spread across lanes: a matrix with one non-zero weight per output is the
    /// clearest way to say so.
    #[test]
    fn every_lane_holds_one_output_neuron() {
        let rows = 11usize;
        let cols = 5usize;
        let x = [1.0f32, 2.0, 4.0, 8.0, 16.0];
        let mut w = vec![0.0f32; rows * cols];
        let mut bias = vec![0.0f32; rows];
        for row in 0..rows {
            // Output `row` reads only column `row % cols`, so its value is
            // `x[row % cols]` and nothing else can reach it.
            w[(row % cols) * rows + row] = 1.0;
            bias[row] = row as f32;
        }
        let mut got = vec![0.0f32; rows];
        // SAFETY: the slices are the sizes the kernel reads, built above.
        unsafe { vector::dense_transposed(&x, &w, rows, cols, &bias, &mut got) };
        for row in 0..rows {
            assert_eq!(got[row], x[row % cols] + row as f32, "row {row}");
        }
    }
}

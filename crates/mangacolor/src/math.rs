//! Shared dense and activation kernels.
//!
//! The dense kernels are deliberately plain scalar loops, so that the numeric
//! contract is easy to read: what is written here is exactly what the classifier
//! runs, and every vector kernel has to reproduce it bit for bit (or be measured
//! against the reference like any other change).  The per-level vector twin of
//! [`dense_transposed`] lives in `math/simd.rs`, the same way each model's
//! per-pixel kernel keeps its scalar reference beside it.
//!
//! The accumulation strategy is part of the contract rather than an
//! implementation detail: a dense layer accumulates a `cols`-term dot product in
//! `f32` per output neuron, in ascending `c`, with the bias added last, matching
//! the reference `Linear` layers and the OGSOX graph's `Gemm` nodes, and the
//! per-tile chroma aggregates use the graph's own lane structure (see
//! `crate::ogsox`).
//!
//! Two weight layouts reach those kernels.  [`dense`] takes the asset's own
//! output-major `(rows, cols)` weights; [`dense_transposed`] takes the
//! `(cols, rows)` copy [`crate::weights`] builds once at load, which is what lets
//! one vector lane own one output neuron without changing a single sum.

mod simd;

/// `out[i] = dot(w[i], x) + bias[i]`, with `w` row-major `(rows, cols)`.
///
/// This is the layout `Gemm(transB=1)` consumes, i.e. every weight in both
/// generated assets except `wa1`.  It is the reference the vector kernel is
/// measured against; the models run [`dense_transposed`] over the transposed
/// copy of the same weights, which is the same arithmetic in a layout that
/// vectorizes.
pub fn dense(x: &[f32], w: &[f32], rows: usize, cols: usize, bias: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(bias.len(), rows);
    debug_assert_eq!(out.len(), rows);
    for row in 0..rows {
        let weights = &w[row * cols..row * cols + cols];
        let mut accumulator = 0.0f32;
        for column in 0..cols {
            accumulator += weights[column] * x[column];
        }
        out[row] = accumulator + bias[row];
    }
}

/// `dense` from the transposed weight layout: `w[c * rows + r]` is the weight
/// `dense` reads as `w[r * cols + c]`.
///
/// Every output neuron still sums its `cols` products in ascending `c` and adds
/// its bias last, from the same `f32`s, so this is bit-identical to
/// [`dense`] over the row-major weights — the differential tests pin that on
/// random data, and `math/simd.rs` pins the vector kernel to the scalar loop.
///
/// The layout is the whole point: transposed, a run of output neurons is a
/// contiguous run of weights, so a group of lanes reads one load a column
/// instead of one strided load an output.  That is what moves the five OGSOV
/// classifiers and the OGSOX heads off their dependency chain.
///
/// # Panics
///
/// If the slices do not match `rows` and `cols`.  The scalar kernel pays a
/// bounds check per term; this one reads what it was handed, so the geometry is
/// checked once, here, and the kernel is free of them.
pub fn dense_transposed(
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    let expected = rows
        .checked_mul(cols)
        .expect("dense_transposed: the matrix dimensions overflow usize");
    assert_eq!(x.len(), cols, "dense_transposed: x is not cols long");
    assert_eq!(
        w.len(),
        expected,
        "dense_transposed: w is not rows*cols long"
    );
    assert_eq!(bias.len(), rows, "dense_transposed: bias is not rows long");
    assert_eq!(out.len(), rows, "dense_transposed: out is not rows long");
    // SAFETY: the four lengths above are the ones the kernel reads.
    unsafe { simd::dense_transposed(x, w, rows, cols, bias, out) }
}

/// `out[j] = sum_i x[i] * w[i * rows + j] + bias[j]`, with `w` input-major.
///
/// The OGSOX attention value projection is stored `(16, 64)` because the graph
/// reaches it through `MatMul` rather than `Gemm(transB=1)`.
pub fn dense_input_major(
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(bias.len(), rows);
    debug_assert_eq!(out.len(), rows);
    out.copy_from_slice(bias);
    for column in 0..cols {
        let value = x[column];
        let weights = &w[column * rows..column * rows + rows];
        for row in 0..rows {
            out[row] += weights[row] * value;
        }
    }
}

/// Dot product of two equal-length slices, accumulating in `f32`.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut accumulator = 0.0f32;
    for index in 0..a.len() {
        accumulator += a[index] * b[index];
    }
    accumulator
}

/// Element-wise `tanh` in place.
#[inline]
pub fn tanh_in_place(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = value.tanh();
    }
}

/// Element-wise `relu` in place.
#[inline]
pub fn relu_in_place(values: &mut [f32]) {
    for value in values.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

/// The logistic function, matching the reference's `torch.sigmoid`.
#[inline]
pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Layer normalization over the whole slice, with population variance.
///
/// `y[i] = (x[i] - mean) / sqrt(var + epsilon) * scale[i] + bias[i]`, computed
/// in `f32` because the graph's `LayerNormalization` nodes use
/// `stash_type = 1`.
pub fn layer_norm(x: &[f32], scale: &[f32], bias: &[f32], epsilon: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), scale.len());
    debug_assert_eq!(x.len(), bias.len());
    debug_assert_eq!(x.len(), out.len());
    let count = x.len() as f32;
    let mut mean = 0.0f32;
    for value in x {
        mean += value;
    }
    mean /= count;

    let mut variance = 0.0f32;
    for value in x {
        let centered = value - mean;
        variance += centered * centered;
    }
    variance /= count;

    let inverse = 1.0 / (variance + epsilon).sqrt();
    for index in 0..x.len() {
        out[index] = (x[index] - mean) * inverse * scale[index] + bias[index];
    }
}

/// Numerically stable in-place softmax over the whole slice.
pub fn softmax_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let mut maximum = values[0];
    for value in &values[1..] {
        if *value > maximum {
            maximum = *value;
        }
    }
    let mut total = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        total += *value;
    }
    let inverse = 1.0 / total;
    for value in values.iter_mut() {
        *value *= inverse;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_is_output_major() {
        // A 2x2 identity-ish weight: rows are (1, 0) and (0, 2).
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 0.0, 0.0, 2.0];
        let bias = [0.5f32, -1.0];
        let mut out = [0.0f32; 2];
        dense(&x, &w, 2, 2, &bias, &mut out);
        assert_eq!(out, [3.5, 7.0]);
    }

    #[test]
    fn dense_input_major_matches_a_transposed_dense() {
        let x = [3.0f32, 4.0];
        // (rows=2, cols=2) input-major: column 0 is (1, 0), column 1 is (0, 2).
        let w = [1.0f32, 0.0, 0.0, 2.0];
        let bias = [0.5f32, -1.0];
        let mut out = [0.0f32; 2];
        dense_input_major(&x, &w, 2, 2, &bias, &mut out);
        assert_eq!(out, [3.5, 7.0]);
    }

    #[test]
    fn layer_norm_normalizes_to_unit_variance() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let scale = [1.0f32; 4];
        let bias = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        layer_norm(&x, &scale, &bias, 1e-5, &mut out);
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6, "mean {mean}");
        let variance: f32 = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!((variance - 1.0).abs() < 1e-3, "variance {variance}");
    }

    /// The transposed kernel is the same arithmetic: same products, same
    /// ascending `c`, same bias last — so every bit has to match `dense` over
    /// the row-major weights, on the shapes the models actually use and on the
    /// ones no vector divides.
    #[test]
    fn dense_transposed_reproduces_dense_bit_for_bit() {
        // A deterministic stream, in the spirit of the per-model tests.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            (unit - 0.5) * 4.0
        };
        for (rows, cols) in [
            (512usize, 32usize),
            (512, 512),
            (1, 512),
            (256, 9),
            (16, 128),
            (64, 16),
            (2, 64),
            (1, 1),
            (3, 5),
            (5, 3),
            (7, 0),
        ] {
            let x: Vec<f32> = (0..cols).map(|_| next()).collect();
            let w: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
            let bias: Vec<f32> = (0..rows).map(|_| next()).collect();
            let mut transposed = vec![0.0f32; rows * cols];
            for row in 0..rows {
                for column in 0..cols {
                    transposed[column * rows + row] = w[row * cols + column];
                }
            }

            let mut want = vec![0.0f32; rows];
            let mut got = vec![0.0f32; rows];
            dense(&x, &w, rows, cols, &bias, &mut want);
            dense_transposed(&x, &transposed, rows, cols, &bias, &mut got);
            for row in 0..rows {
                assert_eq!(
                    got[row].to_bits(),
                    want[row].to_bits(),
                    "row {row} of a {rows}x{cols} matrix"
                );
            }
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut values = [1.0f32, 2.0, 3.0];
        softmax_in_place(&mut values);
        let total: f32 = values.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "total {total}");
        assert!(values[2] > values[1] && values[1] > values[0]);
    }

    #[test]
    fn softmax_is_stable_for_large_inputs() {
        let mut values = [1000.0f32, 1001.0, 1002.0];
        softmax_in_place(&mut values);
        assert!(values.iter().all(|value| value.is_finite()));
        let total: f32 = values.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}

//! Decoders for the generated model assets.
//!
//! Two little-endian formats live here.  Both start with a magic, a version
//! and the lookup table's `min_spread`, then the float32 parameters, then the
//! run-length encoded lookup:
//!
//! * `OGSV` version 1 — five OGSOV MLPs, layer order 0, 3, 6.
//!   `5 x (w1 512x32, b1 512, w2 512x512, b2 512, w3 1x512, b3 1)`.
//! * `OGSX` version 2 — five OGSOX attention heads in `HEAD_ORDER`, then the
//!   runs.  `wa1` is `(16, 64)` input-major because the reference graph uses
//!   `MatMul` there; every other weight is output-major because the graph uses
//!   `Gemm(transB=1)`.
//!
//! Everything is validated before it is used: magic, version, exact sizes,
//! finite parameters, a run sum of exactly `256³`, and no trailing bytes.

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::lookup::LookupTable;

/// How many OGSOV classifiers the reference averages.
pub const OGSOV_CLASSIFIERS: usize = 5;
/// OGSOV feature vector width (the reference's 32 features).
pub const OGSOV_INPUT: usize = 32;
/// OGSOV hidden width.
pub const OGSOV_HIDDEN: usize = 512;

/// How many OGSOX attention heads the reference sums.
pub const OGSOX_HEADS: usize = 5;
/// The nine per-tile features the graph builds before its heads.
pub const OGSOX_FEATURES: usize = 9;
/// The gated projection's width (split into two halves of 128).
pub const OGSOX_GATE: usize = 256;
/// Half of [`OGSOX_GATE`].
pub const OGSOX_HALF: usize = 128;
/// The per-tile embedding the attention pools.
pub const OGSOX_EMBED: usize = 16;
/// The attention / classifier hidden width.
pub const OGSOX_ATTENTION: usize = 64;
/// The two logits a head produces before its scalar difference.
pub const OGSOX_LOGITS: usize = 2;

/// Magic bytes of the OGSOV asset.
pub const OGSOV_MAGIC: &[u8; 4] = b"OGSV";
/// Magic bytes of the OGSOX asset.
pub const OGSOX_MAGIC: &[u8; 4] = b"OGSX";
/// The only accepted OGSOV version.
pub const OGSOV_VERSION: u32 = 1;
/// The only accepted OGSOX version.
pub const OGSOX_VERSION: u32 = 2;

/// Bytes one OGSOV classifier occupies in the asset.
pub const OGSOV_CLASSIFIER_BYTES: usize = 4
    * (OGSOV_HIDDEN * OGSOV_INPUT
        + OGSOV_HIDDEN
        + OGSOV_HIDDEN * OGSOV_HIDDEN
        + OGSOV_HIDDEN
        + OGSOV_HIDDEN
        + 1);

/// Floats one OGSOX head occupies in the asset.
pub const OGSOX_HEAD_FLOATS: usize = OGSOX_GATE * OGSOX_FEATURES // w1
    + OGSOX_GATE // b1
    + OGSOX_HALF // ln1_scale
    + OGSOX_HALF // ln1_bias
    + OGSOX_EMBED * OGSOX_HALF // w2
    + OGSOX_EMBED // b2
    + OGSOX_EMBED // ln2_scale
    + OGSOX_EMBED // ln2_bias
    + OGSOX_EMBED * OGSOX_ATTENTION // wa1 (16, 64) input-major
    + OGSOX_ATTENTION // ba1
    + OGSOX_ATTENTION // wa2
    + OGSOX_ATTENTION * OGSOX_EMBED // wc1
    + OGSOX_ATTENTION // bc1
    + OGSOX_LOGITS * OGSOX_ATTENTION // wc2
    + OGSOX_LOGITS; // bc2

/// One `Linear -> tanh -> Linear -> tanh -> Linear` OGSOV classifier.
#[derive(Debug, Clone, PartialEq)]
pub struct OgsovClassifier {
    /// (512, 32) output-major.
    pub w1: Box<[f32]>,
    /// 512.
    pub b1: Box<[f32]>,
    /// (512, 512) output-major.
    pub w2: Box<[f32]>,
    /// 512.
    pub b2: Box<[f32]>,
    /// (1, 512) output-major.
    pub w3: Box<[f32]>,
    /// One.
    pub b3: f32,
    /// `w1` transposed: `(32, 512)`, the layout the dense kernel walks.
    pub w1_transposed: Box<[f32]>,
    /// `w2` transposed: `(512, 512)`.
    pub w2_transposed: Box<[f32]>,
    /// `w3` transposed: `(512, 1)`.
    pub w3_transposed: Box<[f32]>,
}

impl OgsovClassifier {
    /// Build the transposed copies of the three `Gemm` weights.
    ///
    /// This is the one thing the kernel needs that the asset does not carry:
    /// with `w` output-major four consecutive output neurons are four strided
    /// loads, and the vector kernel reads one output per lane.  Transposing
    /// once at load leaves the assets, and every number in them, untouched.
    fn transpose_weights(&mut self) {
        self.w1_transposed = transpose(&self.w1, OGSOV_HIDDEN, OGSOV_INPUT);
        self.w2_transposed = transpose(&self.w2, OGSOV_HIDDEN, OGSOV_HIDDEN);
        self.w3_transposed = transpose(&self.w3, 1, OGSOV_HIDDEN);
    }
}

impl OgsovClassifier {
    /// How many float parameters this classifier holds.
    pub fn parameter_count(&self) -> usize {
        OGSOV_HIDDEN * OGSOV_INPUT
            + OGSOV_HIDDEN
            + OGSOV_HIDDEN * OGSOV_HIDDEN
            + OGSOV_HIDDEN
            + OGSOV_HIDDEN
            + 1
    }
}

/// The five OGSOV classifiers and the table they share with OGSOX.
#[derive(Debug, Clone)]
pub struct OgsovModel {
    /// One entry per reference classifier, in reference order.
    pub classifiers: Vec<OgsovClassifier>,
    /// Decoded lookup table.
    pub lookup: Arc<LookupTable>,
}

/// Transpose an output-major `(rows, cols)` weight into `(cols, rows)`.
///
/// The decoder keeps the asset's layout and this copy of it: the kernel wants
/// one output neuron per lane, which is a contiguous run only in the transposed
/// layout.  The copy holds the same `f32`s, and [`crate::math::dense_transposed`]
/// adds them in the same per-output order, so nothing about a score moves.
fn transpose(weight: &[f32], rows: usize, cols: usize) -> Box<[f32]> {
    debug_assert_eq!(weight.len(), rows * cols);
    let mut transposed = vec![0.0f32; weight.len()].into_boxed_slice();
    for row in 0..rows {
        for column in 0..cols {
            transposed[column * rows + row] = weight[row * cols + column];
        }
    }
    transposed
}

/// One OGSOX attention head.
///
/// Shapes follow the exported asset: `w1` is `(256, 9)` output-major, `wa1` is
/// `(16, 64)` input-major, and the rest are output-major.
#[derive(Debug, Clone, PartialEq)]
pub struct OgsoxHead {
    /// Gated projection, (256, 9).
    pub w1: Box<[f32]>,
    /// 256.
    pub b1: Box<[f32]>,
    /// First layer norm scale, 128.
    pub ln1_scale: Box<[f32]>,
    /// First layer norm bias, 128.
    pub ln1_bias: Box<[f32]>,
    /// (16, 128).
    pub w2: Box<[f32]>,
    /// 16.
    pub b2: Box<[f32]>,
    /// Second layer norm scale, 16.
    pub ln2_scale: Box<[f32]>,
    /// Second layer norm bias, 16.
    pub ln2_bias: Box<[f32]>,
    /// Attention value projection, (16, 64) input-major.
    pub wa1: Box<[f32]>,
    /// 64.
    pub ba1: Box<[f32]>,
    /// Attention logit vector, 64.
    pub wa2: Box<[f32]>,
    /// Classifier input projection, (64, 16).
    pub wc1: Box<[f32]>,
    /// 64.
    pub bc1: Box<[f32]>,
    /// (2, 64).
    pub wc2: Box<[f32]>,
    /// 2.
    pub bc2: Box<[f32]>,
    /// `w1` transposed: `(9, 256)`.
    pub w1_transposed: Box<[f32]>,
    /// `w2` transposed: `(128, 16)`.
    pub w2_transposed: Box<[f32]>,
    /// `wc1` transposed: `(16, 64)`.
    pub wc1_transposed: Box<[f32]>,
    /// `wc2` transposed: `(64, 2)`.
    pub wc2_transposed: Box<[f32]>,
}

impl OgsoxHead {
    /// How many float parameters this head holds.
    pub fn parameter_count(&self) -> usize {
        self.w1.len()
            + self.b1.len()
            + self.ln1_scale.len()
            + self.ln1_bias.len()
            + self.w2.len()
            + self.b2.len()
            + self.ln2_scale.len()
            + self.ln2_bias.len()
            + self.wa1.len()
            + self.ba1.len()
            + self.wa2.len()
            + self.wc1.len()
            + self.bc1.len()
            + self.wc2.len()
            + self.bc2.len()
    }
}

impl OgsoxHead {
    /// Build the transposed copies of the four `Gemm` weights.
    ///
    /// `wa1` is already input-major, so it has no copy; see
    /// [`OgsovClassifier::transpose_weights`] for why the rest do.
    fn transpose_weights(&mut self) {
        self.w1_transposed = transpose(&self.w1, OGSOX_GATE, OGSOX_FEATURES);
        self.w2_transposed = transpose(&self.w2, OGSOX_EMBED, OGSOX_HALF);
        self.wc1_transposed = transpose(&self.wc1, OGSOX_ATTENTION, OGSOX_EMBED);
        self.wc2_transposed = transpose(&self.wc2, OGSOX_LOGITS, OGSOX_ATTENTION);
    }
}

/// The five OGSOX heads and the lookup table they share with OGSOV.
#[derive(Debug, Clone)]
pub struct OgsoxModel {
    /// One entry per reference head, in reference order.
    pub heads: Vec<OgsoxHead>,
    /// Decoded lookup table.
    pub lookup: Arc<LookupTable>,
}

/// Bounds-checked little-endian reader with asset-aware error messages.
struct Reader<'a> {
    asset: &'static str,
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(asset: &'static str, data: &'a [u8]) -> Self {
        Reader {
            asset,
            data,
            offset: 0,
        }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| Error::asset(self.asset, "asset length overflows usize"))?;
        if end > self.data.len() {
            return Err(Error::asset(
                self.asset,
                format!(
                    "truncated: {} more bytes needed at offset {}, only {} remain",
                    count,
                    self.offset,
                    self.data.len().saturating_sub(self.offset)
                ),
            ));
        }
        let slice = &self.data[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn magic(&mut self) -> Result<[u8; 4]> {
        let bytes = self.take(4)?;
        Ok([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    /// Decode `count` little-endian float32 values, rejecting non-finite ones.
    ///
    /// The bytes are decoded explicitly instead of being reinterpreted, so no
    /// alignment assumption is made about the embedded asset.
    fn floats(&mut self, count: usize) -> Result<Box<[f32]>> {
        let bytes = self.take(count * 4)?;
        let mut values = Vec::with_capacity(count);
        for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
            let value = f32::from_le_bytes(*chunk);
            if !value.is_finite() {
                return Err(Error::asset(
                    self.asset,
                    format!("parameter {index} is not finite ({value})"),
                ));
            }
            values.push(value);
        }
        Ok(values.into_boxed_slice())
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.offset
    }

    /// Read the trailing run-length lookup payload.
    fn lookup(&mut self, min_spread: u32) -> Result<LookupTable> {
        let run_count = self.u32()? as usize;
        let available = self.remaining() / 4;
        if run_count > available {
            return Err(Error::asset(
                self.asset,
                format!("declares {run_count} lookup runs but only {available} fit"),
            ));
        }
        let mut runs = Vec::with_capacity(run_count);
        for _ in 0..run_count {
            runs.push(self.u32()?);
        }
        let table = LookupTable::from_runs(self.asset, &runs, min_spread)?;
        if self.remaining() != 0 {
            return Err(Error::asset(
                self.asset,
                format!("{} trailing bytes after the lookup runs", self.remaining()),
            ));
        }
        Ok(table)
    }
}

/// Read the shared four-byte magic, version and `min_spread` header.
fn header<'a>(
    bytes: &'a [u8],
    asset: &'static str,
    magic: &[u8; 4],
    version: u32,
) -> Result<(Reader<'a>, u32)> {
    if bytes.len() < 4 {
        return Err(Error::asset(
            asset,
            format!("only {} bytes, too short for a header", bytes.len()),
        ));
    }
    let mut reader = Reader::new(asset, bytes);
    let found = reader.magic()?;
    if &found != magic {
        return Err(Error::asset(
            asset,
            format!(
                "magic is {:?}, expected {:?}",
                String::from_utf8_lossy(&found),
                String::from_utf8_lossy(magic)
            ),
        ));
    }
    let found_version = reader.u32()?;
    if found_version != version {
        return Err(Error::asset(
            asset,
            format!("version {found_version} is not supported (expected {version})"),
        ));
    }
    let min_spread = reader.u32()?;
    Ok((reader, min_spread))
}

/// Decode the `OGSV` asset.
pub fn load_ogsov(asset: &'static str, bytes: &[u8]) -> Result<OgsovModel> {
    let (mut reader, min_spread) = header(bytes, asset, OGSOV_MAGIC, OGSOV_VERSION)?;
    let mut classifiers = Vec::with_capacity(OGSOV_CLASSIFIERS);
    for index in 0..OGSOV_CLASSIFIERS {
        let w1 = reader.floats(OGSOV_HIDDEN * OGSOV_INPUT)?;
        let b1 = reader.floats(OGSOV_HIDDEN)?;
        let w2 = reader.floats(OGSOV_HIDDEN * OGSOV_HIDDEN)?;
        let b2 = reader.floats(OGSOV_HIDDEN)?;
        let w3 = reader.floats(OGSOV_HIDDEN)?;
        let b3 = reader.floats(1)?[0];
        if w1.len() != OGSOV_HIDDEN * OGSOV_INPUT {
            return Err(Error::asset(
                asset,
                format!("classifier {index} has a short w1"),
            ));
        }
        let mut classifier = OgsovClassifier {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            w1_transposed: Box::default(),
            w2_transposed: Box::default(),
            w3_transposed: Box::default(),
        };
        classifier.transpose_weights();
        classifiers.push(classifier);
    }
    let lookup = reader.lookup(min_spread)?;
    Ok(OgsovModel {
        classifiers,
        lookup: Arc::new(lookup),
    })
}

/// Decode the `OGSX` asset.
pub fn load_ogsox(asset: &'static str, bytes: &[u8]) -> Result<OgsoxModel> {
    let (mut reader, min_spread) = header(bytes, asset, OGSOX_MAGIC, OGSOX_VERSION)?;
    let mut heads = Vec::with_capacity(OGSOX_HEADS);
    for _ in 0..OGSOX_HEADS {
        // The four `Gemm` weights arrive in the asset's own layout and are
        // transposed below; `wa1` is input-major already.
        let mut head = OgsoxHead {
            w1: reader.floats(OGSOX_GATE * OGSOX_FEATURES)?,
            b1: reader.floats(OGSOX_GATE)?,
            ln1_scale: reader.floats(OGSOX_HALF)?,
            ln1_bias: reader.floats(OGSOX_HALF)?,
            w2: reader.floats(OGSOX_EMBED * OGSOX_HALF)?,
            b2: reader.floats(OGSOX_EMBED)?,
            ln2_scale: reader.floats(OGSOX_EMBED)?,
            ln2_bias: reader.floats(OGSOX_EMBED)?,
            wa1: reader.floats(OGSOX_EMBED * OGSOX_ATTENTION)?,
            ba1: reader.floats(OGSOX_ATTENTION)?,
            wa2: reader.floats(OGSOX_ATTENTION)?,
            wc1: reader.floats(OGSOX_ATTENTION * OGSOX_EMBED)?,
            bc1: reader.floats(OGSOX_ATTENTION)?,
            wc2: reader.floats(OGSOX_LOGITS * OGSOX_ATTENTION)?,
            bc2: reader.floats(OGSOX_LOGITS)?,
            w1_transposed: Box::default(),
            w2_transposed: Box::default(),
            wc1_transposed: Box::default(),
            wc2_transposed: Box::default(),
        };
        head.transpose_weights();
        heads.push(head);
    }
    let lookup = reader.lookup(min_spread)?;
    Ok(OgsoxModel {
        heads,
        lookup: Arc::new(lookup),
    })
}

/// Point both models at one decoded table once their bytes are proven equal.
///
/// The two generated assets carry the same lookup; sharing it is a memory
/// decision, so it happens only after comparing the decoded tables.
pub fn unify_lookup(ogsov: &mut OgsovModel, ogsox: &mut OgsoxModel) -> Result<bool> {
    if Arc::ptr_eq(&ogsov.lookup, &ogsox.lookup) {
        return Ok(true);
    }
    if !ogsov.lookup.same_content(&ogsox.lookup) {
        return Err(Error::asset(
            "ogsov.bin/ogsox.bin",
            "the two assets carry different lookup tables",
        ));
    }
    ogsox.lookup = Arc::clone(&ogsov.lookup);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ogsox_head_float_count_matches_the_exporter() {
        // Five heads of 7,282 parameters each, as tools/export_weights.py
        // computes from HEAD_SHAPES.
        assert_eq!(OGSOX_HEAD_FLOATS, 7282);
    }

    #[test]
    fn ogsov_classifier_byte_count_matches_the_exporter() {
        assert_eq!(OGSOV_CLASSIFIER_BYTES, 280_065 * 4);
    }

    #[test]
    fn a_bad_magic_is_rejected() {
        let error =
            load_ogsov("t.bin", b"NOPE\x01\x00\x00\x00\x05\x00\x00\x00").expect_err("must reject");
        assert!(matches!(error, Error::Asset { .. }));
    }

    #[test]
    fn a_bad_version_is_rejected() {
        let mut bytes = Vec::from(*OGSOV_MAGIC);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        let error = load_ogsov("t.bin", &bytes).expect_err("must reject");
        assert!(matches!(error, Error::Asset { .. }));
    }

    #[test]
    fn a_truncated_asset_is_rejected() {
        let mut bytes = Vec::from(*OGSOX_MAGIC);
        bytes.extend_from_slice(&OGSOX_VERSION.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        let error = load_ogsox("t.bin", &bytes).expect_err("must reject");
        assert!(matches!(error, Error::Asset { .. }));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        // A complete OGSX header plus one head of zeros, then an empty lookup
        // with a stray byte.
        let mut bytes = Vec::from(*OGSOX_MAGIC);
        bytes.extend_from_slice(&OGSOX_VERSION.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend(std::iter::repeat_n(
            0u8,
            4 * OGSOX_HEAD_FLOATS * OGSOX_HEADS,
        ));
        bytes.extend_from_slice(&0u32.to_le_bytes()); // zero runs
        bytes.push(0);
        let error = load_ogsox("t.bin", &bytes).expect_err("must reject");
        assert!(matches!(error, Error::Asset { .. }), "{error}");
    }
}

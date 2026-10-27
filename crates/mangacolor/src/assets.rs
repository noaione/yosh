//! Generated model assets, embedded at compile time.
//!
//! `build.rs` converts the source weights (`ogsov_weights.npz` and
//! `ogsox_classifier_optimized.onnx`) into the two binary assets below and
//! embeds them.  When the sources are absent the build still succeeds: the
//! files are empty and the `*_EMBEDDED` flags in `models_gen.rs` are `false`,
//! so a caller can offer a clear "weights missing" state instead of silently
//! running a zeroed model.

/// The `OGSV` version 1 asset: five OGSOV MLPs plus the shared lookup table.
pub static OGSOV_BIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ogsov.bin"));

/// The `OGSX` version 2 asset: five OGSOX heads plus the shared lookup table.
pub static OGSOX_BIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ogsox.bin"));

/// File name of the OGSOV asset, for error messages and diagnostics.
pub const OGSOV_ASSET: &str = "ogsov.bin";

/// File name of the OGSOX asset, for error messages and diagnostics.
pub const OGSOX_ASSET: &str = "ogsox.bin";

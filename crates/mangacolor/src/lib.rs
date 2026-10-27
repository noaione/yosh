//! CPU manga colour-page detection: OGSOV and OGSOX.
//!
//! Ported from the `vs-mangacolor` VapourSynth plugin (MIT, noaione), keeping
//! its CPU algorithm core and SIMD kernels and dropping the VapourSynth host,
//! YUV conversion and GPU paths.  The two generations answer the same
//! question — *is this a colour page?* — with different models:
//!
//! * [`ogsov`] — a colourfulness histogram, 32 statistics and five MLPs whose
//!   sigmoid probabilities are averaged;
//! * [`ogsox`] — 512² tiling, nine per-tile features and five attention heads
//!   whose signed scalars are summed.
//!
//! The core is host-free: frames reach it as an [`rgb::RgbView`] (three plane
//! rows, explicit strides) or through the interleaved-RGB/RGBA helpers below,
//! which de-interleave once into reusable scratch planes and then run exactly
//! the same kernels.
//!
//! Weights are embedded at build time when the build script can find them; see
//! [`embedded_ogsov`] and [`embedded_ogsox`].
//!
//! # Score and confidence
//!
//! The two models report different raw scores and derive confidence from them
//! differently, exactly as their references do:
//!
//! * OGSOV's score is the mean of five sigmoid probabilities in `[0, 1]`;
//! * OGSOX's score is the signed sum of five head scalars in `[-5, 5]`.
//!
//! Both report `confidence` as an unrounded percentage (`0..=100`), so a
//! caller that only wants an integer must truncate deliberately.

use std::cell::RefCell;
use std::sync::OnceLock;

pub mod assets;
pub mod error;
pub mod lookup;
pub mod math;
pub mod ogsov;
pub mod ogsox;
pub mod rgb;
pub mod weights;

pub use error::{Error, Result};
pub use rgb::{PlanarScratch, PlaneView, RgbView};
pub use weights::{OgsovModel, OgsoxModel};

// `OGSOV_EMBEDDED` / `OGSOX_EMBEDDED`, written by the build script.
include!(concat!(env!("OUT_DIR"), "/models_gen.rs"));

/// Whether this binary carries the OGSOV weights.
pub fn ogsov_available() -> bool {
    OGSOV_EMBEDDED
}

/// Whether this binary carries the OGSOX weights.
pub fn ogsox_available() -> bool {
    OGSOX_EMBEDDED
}

/// The two decoded models, loaded at most once per process.
struct EmbeddedModels {
    ogsov: Option<OgsovModel>,
    ogsox: Option<OgsoxModel>,
}

static MODELS: OnceLock<EmbeddedModels> = OnceLock::new();

fn models() -> &'static EmbeddedModels {
    MODELS.get_or_init(|| {
        let mut ogsov = OGSOV_EMBEDDED.then(|| {
            weights::load_ogsov(assets::OGSOV_ASSET, assets::OGSOV_BIN)
                .expect("the embedded OGSOV asset is malformed")
        });
        let mut ogsox = OGSOX_EMBEDDED.then(|| {
            weights::load_ogsox(assets::OGSOX_ASSET, assets::OGSOX_BIN)
                .expect("the embedded OGSOX asset is malformed")
        });
        // The two generated assets carry the same lookup table. Sharing the decoded
        // table saves its 2 MiB in a process that has both models, and
        // `unify_lookup` only shares it after proving the tables are equal.
        if let (Some(ogsov), Some(ogsox)) = (&mut ogsov, &mut ogsox) {
            weights::unify_lookup(ogsov, ogsox)
                .expect("the embedded OGSOV and OGSOX lookup tables differ");
        }
        EmbeddedModels { ogsov, ogsox }
    })
}

/// The embedded OGSOV model, if this build has weights.
pub fn embedded_ogsov() -> Option<&'static OgsovModel> {
    models().ogsov.as_ref()
}

/// The embedded OGSOX model, if this build has weights.
pub fn embedded_ogsox() -> Option<&'static OgsoxModel> {
    models().ogsox.as_ref()
}

/// The error a caller gets when it asks for a model this build does not carry.
pub fn missing_model(asset: &'static str) -> Error {
    Error::asset(
        asset,
        "this build has no embedded weights (place ogsov_weights.npz and \
         ogsox_classifier_optimized.onnx in the crate or workspace root, or set \
         OGSOV_WEIGHTS_PATH / OGSOX_WEIGHTS_PATH, then rebuild)",
    )
}

// --- interleaved input -----------------------------------------------------
//
// yosh decodes pages to interleaved RGB/RGBA. The scratch below is per thread
// and reused across pages: a decode pool worker that classifies page after page
// pays the allocation once, not once per page, and the two models never share
// state.

struct OgsovSession {
    scratch: ogsov::OgsovScratch,
    planes: PlanarScratch,
}

impl OgsovSession {
    fn new() -> Self {
        OgsovSession {
            scratch: ogsov::OgsovScratch::new(),
            planes: PlanarScratch::new(),
        }
    }
}

struct OgsoxSession {
    scratch: ogsox::OgsoxScratch,
    planes: PlanarScratch,
}

impl OgsoxSession {
    fn new() -> Self {
        OgsoxSession {
            scratch: ogsox::OgsoxScratch::new(),
            planes: PlanarScratch::new(),
        }
    }
}

thread_local! {
    static OGSOV_SESSION: RefCell<Option<OgsovSession>> = const { RefCell::new(None) };
    static OGSOX_SESSION: RefCell<Option<OgsoxSession>> = const { RefCell::new(None) };
}

/// Run OGSOV on an interleaved RGB (`channels = 3`) or RGBA (`channels = 4`)
/// frame, using this thread's reusable workspace.
///
/// The alpha byte of an RGBA frame is ignored; a caller decides whether a
/// transparent frame is eligible for detection.
pub fn detect_ogsov(
    pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<ogsov::OgsovPrediction> {
    let model = embedded_ogsov().ok_or_else(|| missing_model(assets::OGSOV_ASSET))?;
    OGSOV_SESSION.with(|session| {
        let mut session = session.borrow_mut();
        let session = session.get_or_insert_with(OgsovSession::new);
        session.planes.adopt(pixels, width, height, channels)?;
        // A frame that is exactly gray cannot be a colour page, and the
        // de-interleave above already compared every channel: answer without
        // running the model, as the old detector did.
        if session.planes.is_gray() {
            return Ok(ogsov::OgsovPrediction::fast_gray());
        }
        let view = session.planes.view()?;
        let features =
            ogsov::extract_features(&view, &model.lookup, &mut session.scratch, false, false)?;
        Ok(ogsov::predict(
            model,
            &features,
            &mut session.scratch,
            false,
        ))
    })
}

/// Run OGSOX on an interleaved RGB (`channels = 3`) or RGBA (`channels = 4`)
/// frame, using this thread's reusable workspace.
pub fn detect_ogsox(
    pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<ogsox::OgsoxPrediction> {
    let model = embedded_ogsox().ok_or_else(|| missing_model(assets::OGSOX_ASSET))?;
    OGSOX_SESSION.with(|session| {
        let mut session = session.borrow_mut();
        let session = session.get_or_insert_with(OgsoxSession::new);
        session.planes.adopt(pixels, width, height, channels)?;
        if session.planes.is_gray() {
            return Ok(ogsox::OgsoxPrediction::fast_gray());
        }
        let view = session.planes.view()?;
        let score = ogsox::predict(model, &view, &mut session.scratch, false, false)?;
        Ok(ogsox::OgsoxPrediction::from_score(score))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_model_is_reported_clearly() {
        let error = missing_model("ogsov.bin");
        let message = error.to_string();
        assert!(message.contains("ogsov.bin"), "{message}");
    }

    #[test]
    fn interleaved_rgb_and_rgba_agree() {
        let (width, height) = (37, 23);
        let mut rgb = Vec::with_capacity(width * height * 3);
        let mut rgba = Vec::with_capacity(width * height * 4);
        for index in 0..width * height {
            let r = (index * 7) as u8;
            let g = (index * 11) as u8;
            let b = (index * 13) as u8;
            rgb.extend_from_slice(&[r, g, b]);
            rgba.extend_from_slice(&[r, g, b, 0]);
        }
        let mut planes = PlanarScratch::new();
        planes.adopt(&rgb, width, height, 3).expect("rgb");
        let rgb_red: Vec<u8> = planes.view().expect("view").red.row(0).to_vec();
        assert!(!planes.is_gray());
        planes.adopt(&rgba, width, height, 4).expect("rgba");
        let rgba = planes.view().expect("view");
        assert_eq!(rgba.red.row(0), rgb_red.as_slice());
        assert_eq!(rgba.pixel_count().expect("count"), (width * height) as u64);
    }

    #[test]
    fn a_short_interleaved_frame_is_rejected() {
        let mut planes = PlanarScratch::new();
        let error = planes.adopt(&[0u8; 11], 2, 2, 3).expect_err("must reject");
        assert!(matches!(error, Error::UnsupportedInput { .. }), "{error}");
        let error = planes.adopt(&[0u8; 12], 2, 2, 2).expect_err("must reject");
        assert!(matches!(error, Error::UnsupportedInput { .. }), "{error}");
    }
}

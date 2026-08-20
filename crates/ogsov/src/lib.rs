pub mod detect;

include!(concat!(env!("OUT_DIR"), "/ogsov_gen.rs"));

static EMBEDDED_WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ogsov.bin"));

use std::sync::OnceLock;
static MODEL: OnceLock<Option<detect::Ogsov>> = OnceLock::new();

/// The embedded model, if weights were available at build time.
/// Parsed once on first use; `None` if this build has no weights.
pub fn embedded_model() -> Option<&'static detect::Ogsov> {
    MODEL
        .get_or_init(|| {
            if OGSOV_EMBEDDED {
                Some(
                    detect::Ogsov::from_bytes(EMBEDDED_WEIGHTS)
                        .expect("embedded OGSOV weights are malformed"),
                )
            } else {
                None
            }
        })
        .as_ref()
}

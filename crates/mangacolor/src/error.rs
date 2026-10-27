//! Actionable error types for the detection core.
//!
//! Every variant carries enough context to name the asset, frame or parameter
//! that was rejected, because these messages are what a caller sees when a page
//! is outside the documented input contract.

use std::fmt;

/// A failure the core can report to the host instead of panicking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A generated model asset is missing, truncated or inconsistent.
    Asset {
        /// Which generated asset, as it is named on disk.
        asset: &'static str,
        /// What specifically did not validate.
        detail: String,
    },
    /// The frame's format or dimensions are outside the input contract.
    UnsupportedInput {
        /// What specifically is unsupported, naming the value that was seen.
        detail: String,
    },
    /// OGSOV cannot define a remainder for this many pixels.
    TooFewPixels {
        /// The largest pixel count the trimmed statistics are undefined for.
        at_most: u64,
        /// How many pixels the rejected frame actually has.
        actual: u64,
    },
    /// Checked dimension arithmetic overflowed, or a frame is too large for a
    /// workspace the core is willing to allocate.
    DimensionOverflow {
        /// Which computation or allocation failed.
        detail: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Asset { asset, detail } => write!(
                formatter,
                "the embedded model asset {asset} is unusable: {detail}. Place the \
                 source weights where crates/mangacolor/build.rs looks for them and \
                 rebuild (see crates/mangacolor/BENCHMARKS.md)."
            ),
            Error::UnsupportedInput { detail } => {
                write!(formatter, "unsupported input: {detail}")
            }
            Error::TooFewPixels { at_most, actual } => write!(
                formatter,
                "OGSOV needs more than {at_most} pixels to define its trimmed \
                 statistics (the histogram path removes that many samples before \
                 computing any), but this frame has {actual}"
            ),
            Error::DimensionOverflow { detail } => {
                write!(formatter, "frame dimensions are out of range: {detail}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Convenience constructor for asset failures.
    pub(crate) fn asset(asset: &'static str, detail: impl Into<String>) -> Self {
        Error::Asset {
            asset,
            detail: detail.into(),
        }
    }

    /// Convenience constructor for input-contract failures.
    pub(crate) fn input(detail: impl Into<String>) -> Self {
        Error::UnsupportedInput {
            detail: detail.into(),
        }
    }
}

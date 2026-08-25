//! ICC color management: pull the embedded profile out of an encoded image,
//! read its human-readable name, and color-manage decoded pixels to sRGB.
//!
//! yosh otherwise treats decoded bytes as sRGB. An image tagged with a wider
//! profile (e.g. Display P3) would then render desaturated, so before downscale
//! we convert such color images to sRGB with `qcms` (pure-Rust, the color
//! manager Firefox uses for image display). Untagged or already-sRGB images, and
//! the grayscale path, are left untouched — only wide-gamut color pages pay any
//! cost.

use std::io::Cursor;

use image::ImageDecoder;

/// Extract the embedded ICC profile from encoded image bytes (JPEG APP2 / PNG
/// iCCP / WebP ICCP), reusing the `image` crate's decoder. Reads metadata only —
/// it does not decode pixels. None if absent or unsupported.
pub fn extract_icc(bytes: &[u8]) -> Option<Vec<u8>> {
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?;
    let mut decoder = reader.into_decoder().ok()?;
    decoder.icc_profile().ok().flatten()
}

/// True if the profile is an sRGB profile (by description) — used to skip a
/// pointless sRGB→sRGB transform.
pub fn is_srgb(profile: &[u8]) -> bool {
    describe(profile).is_some_and(|d| d.to_ascii_lowercase().contains("srgb"))
}

/// True if the profile's data colour space is grayscale (`GRAY`, at ICC header
/// offset 16) — e.g. a "Dot Gain" output profile on a monochrome (yuv400) AVIF.
/// Such a profile must NOT be fed to `to_srgb_rgba`: it builds an RGBA8 qcms
/// transform, and a 1-channel gray profile against 4-channel data corrupts the
/// image (renders white). Grayscale pages get their tone from the gray decode
/// path instead, so the right move is to skip color management here.
pub fn is_gray(profile: &[u8]) -> bool {
    profile.get(16..20) == Some(b"GRAY")
}

/// True if the profile's data colour space is CMYK (`CMYK`, at ICC header
/// offset 16) — a printer/output profile describing device CMYK. Such a profile
/// must NOT be fed to `to_srgb_rgba` (RGBA data against a CMYK profile is
/// nonsense); it is the only valid source for `cmyk_to_srgb_rgb8`.
pub fn is_cmyk(profile: &[u8]) -> bool {
    profile.get(16..20) == Some(b"CMYK")
}

/// Human-readable profile name (e.g. "Display P3") from the ICC `desc` tag.
/// Handles the v2 `textDescriptionType` (ASCII) and v4 `mluc`
/// (multiLocalizedUnicodeType, UTF-16BE) encodings.
pub fn describe(profile: &[u8]) -> Option<String> {
    let be32 = |b: &[u8], i: usize| -> Option<usize> {
        b.get(i..i + 4)
            .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
    };
    // Tag table: count at offset 128, then 12-byte entries (sig, offset, size).
    let count = be32(profile, 128)?;
    for k in 0..count {
        let e = 132 + k * 12;
        if profile.get(e..e + 4)? == b"desc" {
            let off = be32(profile, e + 4)?;
            let size = be32(profile, e + 8)?;
            let tag = profile.get(off..off.checked_add(size)?)?;
            return parse_desc(tag);
        }
    }
    None
}

fn parse_desc(tag: &[u8]) -> Option<String> {
    let be32 = |i: usize| -> Option<usize> {
        tag.get(i..i + 4)
            .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
    };
    match tag.get(0..4)? {
        // textDescriptionType: [type:4][reserved:4][ascii-count:4][ascii…]
        b"desc" => {
            let n = be32(8)?;
            let s = tag.get(12..12usize.checked_add(n)?)?;
            let s = s.split(|&c| c == 0).next().unwrap_or(s); // up to NUL
            let s = String::from_utf8_lossy(s).trim().to_string();
            (!s.is_empty()).then_some(s)
        }
        // multiLocalizedUnicodeType: [type:4][reserved:4][count:4][recsize:4]
        // then records; first record at 16: [lang:2][country:2][len:4][offset:4],
        // string data is UTF-16BE at `offset` from the start of the tag.
        b"mluc" => {
            let len = be32(20)?;
            let off = be32(24)?;
            let raw = tag.get(off..off.checked_add(len)?)?;
            let u16s: Vec<u16> = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_be_bytes(*c))
                .collect();
            let s = String::from_utf16_lossy(&u16s);
            let s = s
                .trim_matches(|c: char| c == '\0' || c.is_whitespace())
                .to_string();
            (!s.is_empty()).then_some(s)
        }
        _ => None,
    }
}

/// Color-manage an RGBA8 buffer in place from `profile` to sRGB. No-op if the
/// profile can't be parsed or a transform can't be built (falls back to the
/// previous behavior of treating the bytes as sRGB).
pub fn to_srgb_rgba(profile: &[u8], rgba: &mut [u8]) {
    let Some(input) = qcms::Profile::new_from_slice(profile, false) else {
        return;
    };
    let mut output = qcms::Profile::new_sRGB();
    output.precache_output_transform();
    let Some(xfm) = qcms::Transform::new(
        &input,
        &output,
        qcms::DataType::RGBA8,
        qcms::Intent::Perceptual,
    ) else {
        return;
    };
    xfm.apply(rgba);
}

/// Fallible CMYK8 → sRGB RGB8 conversion via qcms. `cmyk` holds conventional
/// CMYK samples (0 = no ink, 255 = full ink) and `rgb` receives the converted
/// pixels. Returns a descriptive error when the profile can't be parsed, qcms
/// can't build a transform (e.g. a CMYK profile without a usable A2B table), or
/// the buffer lengths are inconsistent — the caller falls back to a generic
/// conversion. Lengths are validated before conversion because qcms's
/// `Transform::convert` panics on mismatched buffers.
pub fn cmyk_to_srgb_rgb8(profile: &[u8], cmyk: &[u8], rgb: &mut [u8]) -> Result<(), String> {
    let n = cmyk.len() / 4;
    if !cmyk.len().is_multiple_of(4) {
        return Err(format!(
            "cmyk icc: incomplete CMYK pixels ({} bytes)",
            cmyk.len()
        ));
    }
    if rgb.len() != n * 3 {
        return Err(format!(
            "cmyk icc: destination {} bytes for {n} CMYK pixels (need {})",
            rgb.len(),
            n * 3
        ));
    }
    let Some(input) = qcms::Profile::new_from_slice(profile, false) else {
        return Err("cmyk icc: unparsable profile".into());
    };
    let mut output = qcms::Profile::new_sRGB();
    output.precache_output_transform();
    let Some(xfm) = qcms::Transform::new_to(
        &input,
        &output,
        qcms::DataType::CMYK,
        qcms::DataType::RGB8,
        qcms::Intent::Perceptual,
    ) else {
        return Err("cmyk icc: unsupported profile (no usable A2B table)".into());
    };
    xfm.convert(cmyk, rgb);
    Ok(())
}

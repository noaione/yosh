//! Decode + downscale a page's encoded bytes to a display-resolution buffer.
//!
//! Routes by magic bytes: PNG → `png`, JPEG → `jpeg-decoder`, JPEG XL →
//! `jxl-oxide` (pure Rust), else → `image` crate fallback (WebP/GIF/BMP/AVIF/…).
//! Normalizes to single-channel R8 (gray) or
//! RGBA8 (color), then downscales with a high-quality, content-aware filter
//! (inspired by MangaJaNaiConverterGui's final-resize strategy):
//!   - **color** → Lanczos3 in gamma space (no color conversion),
//!   - **grayscale** → Catmull-Rom in **true 16-bit linear light**: linearize
//!     sRGB → linear luminance, resample, then re-encode through the Dot Gain 20%
//!     curve so screentones stay inky (see `tone.rs`). Linear-light resampling is
//!     what suppresses halftone moiré.
//!
//! Color-stored pages can be left untouched, classified with MangaJaNai's
//! traditional threshold, or classified by the OGSOV model before resize.

use std::sync::atomic::{AtomicU32, Ordering};

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::codecs::gif::GifDecoder;
use image::codecs::webp::WebPDecoder;
use image::{AnimationDecoder, ImageDecoder};

use crate::icc;
use crate::tone;

const PNG_SIG: [u8; 4] = [0x89, 0x50, 0x4E, 0x47];

/// MangaJaNai's default `GrayscaleDetectionThreshold` (its slider spans 0..24).
/// Higher = more tolerant of slight color casts when deciding "is this gray?".
const GRAYSCALE_THRESHOLD: i32 = 12;

/// Strategy used to decide whether a color-stored, opaque page should use the
/// color path or be collapsed to luminance and use the grayscale path.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ColorDetection {
    /// Skip color detection. Color-stored pages remain RGBA.
    Off,
    /// MangaJaNai-compatible channel-difference threshold.
    #[default]
    Traditional,
    /// OGSOV ML ensemble. If this build has no embedded weights, conservatively
    /// retain color rather than silently falling back to another strategy.
    Ml,
}

/// Decode behavior shared by direct callers and decode-pool workers.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DecodeOptions {
    pub color_detection: ColorDetection,
}

/// Color-page classification attached to decoded output for info overlays.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorDetectionOutcome {
    NotRun,
    SourceGray,
    Off,
    Traditional { is_color: bool },
    Ogsov { is_color: bool, confidence: u8 },
    OgsovUnavailable,
    Transparent,
}

impl ColorDetectionOutcome {
    fn is_color(self) -> Option<bool> {
        match self {
            Self::SourceGray => Some(false),
            Self::Traditional { is_color } | Self::Ogsov { is_color, .. } => Some(is_color),
            _ => None,
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::NotRun => "not run".to_string(),
            Self::SourceGray => "source grayscale · is_color: false".to_string(),
            Self::Off => "off".to_string(),
            Self::Traditional { is_color } => {
                format!("traditional · is_color: {is_color}")
            }
            Self::Ogsov {
                is_color,
                confidence,
            } => format!("OGSOV · is_color: {is_color} · confidence: {confidence}%"),
            Self::OgsovUnavailable => "OGSOV unavailable (weights missing)".to_string(),
            Self::Transparent => "not run (transparent image)".to_string(),
        }
    }
}

/// Whether this binary contains OGSOV weights required by [`ColorDetection::Ml`].
pub fn ml_color_detection_available() -> bool {
    ogsov::OGSOV_EMBEDDED
}

/// GPU `max_texture_dimension_2d`, published by `gpu.rs` at startup. A decoded page
/// can't exceed this in either dimension (it's one texture), so pages that would
/// go over are rejected with a clear error rather than downscaled. wgpu's default
/// (8192) until set; modern GPUs report 16384.
pub static MAX_TEX_DIM: AtomicU32 = AtomicU32::new(8192);

/// Decoded size for a source of `(w, h)` at a desired `target_h`: scale to
/// `target_h` (never upscaling past the source), preserving aspect.
fn target_dims(w: u32, h: u32, target_h: u32) -> (u32, u32) {
    let th = target_h.min(h).max(1);
    let tw = (((w as f64) * (th as f64) / (h as f64)).round() as u32).max(1);
    (tw, th)
}

/// Reject a page whose decoded size won't fit in a single GPU texture (full
/// resolution is preserved up to the limit; we never silently downscale past it).
fn check_fits(tw: u32, th: u32) -> Result<(), String> {
    let max = MAX_TEX_DIM.load(Ordering::Relaxed);
    if tw > max || th > max {
        Err(format!("image too large for the GPU ({tw}x{th}; max {max} px per side)"))
    } else {
        Ok(())
    }
}

/// Which CPU resize path produced a page. Surfaced in the info overlay so the
/// active pipeline — and whether the GPU then has to resize at all — is visible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResizePath {
    /// No downscale: decoded at native (target ≥ source). The GPU draws it 1:1, or
    /// upscales for zoom-past-native magnification.
    None,
    /// Grayscale source, resampled in true linear light (Catmull-Rom) then
    /// re-encoded through the Dot Gain 20% curve (the screentone-safe path).
    GrayLinear,
    /// Color source detected as visually gray: collapsed to luma + the gray path.
    GrayFromColor,
    /// Color source: Lanczos3 in gamma space (ICC→sRGB first if the page was tagged).
    Color,
}

impl ResizePath {
    pub fn label(self) -> &'static str {
        match self {
            ResizePath::None => "none (native res)",
            ResizePath::GrayLinear => "gray linear-light (Catmull-Rom + Dot Gain)",
            ResizePath::GrayFromColor => "gray-from-color (Catmull-Rom + Dot Gain)",
            ResizePath::Color => "color (Lanczos3)",
        }
    }
}

/// A decoded, downscaled page ready for GPU upload.
pub struct DecodedImage {
    pub w: u32,
    pub h: u32,
    /// Native (pre-downscale) source dimensions, kept so the UI can report the
    /// zoom level relative to the original image (the texture is decoded to ~the
    /// display size, so `w`/`h` alone can't reveal it).
    pub src_w: u32,
    pub src_h: u32,
    /// true => single-channel R8; false => RGBA8.
    pub gray: bool,
    /// Which CPU resize path produced this image (for the info overlay).
    pub path: ResizePath,
    /// Color classifier outcome, retained through GPU upload for info overlays.
    pub color_detection: ColorDetectionOutcome,
    pub pixels: Vec<u8>,
}

fn rgb_to_rgba(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let n = (w as usize) * (h as usize);
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        out[i * 4] = rgb[i * 3];
        out[i * 4 + 1] = rgb[i * 3 + 1];
        out[i * 4 + 2] = rgb[i * 3 + 2];
        out[i * 4 + 3] = 255;
    }
    out
}

fn ga_to_gray(ga: &[u8]) -> Vec<u8> {
    ga.iter().step_by(2).copied().collect()
}

/// Invert CMYK samples in place (`v = 255 - v`).
fn invert_cmyk(buf: &mut [u8]) {
    for v in buf {
        *v = 255 - *v;
    }
}

/// The CMYK storage convention of a 4-component JPEG, read from its Adobe
/// APP14 marker: returns `(adobe, ycck)`.
///
/// - Adobe files (APP14 `"Adobe"` transform 0 = CMYK, 2 = YCCK) store
///   **inverted** samples (0 = 100% ink). jpeg-decoder outputs `255 − stored`,
///   so for them its output is already conventional ink.
/// - US files (no Adobe marker) store conventional samples (0 = no ink), so
///   jpeg-decoder's output is complemented (255 − ink).
///
/// YCCK's three color channels are display RGB decoded from YCbCr in either
/// convention; only the K polarity follows the file.
fn jpeg_cmyk_convention(bytes: &[u8]) -> (bool, bool) {
    let mut i = 2;
    while i + 4 <= bytes.len() {
        if bytes[i] == 0xFF && bytes[i + 1] == 0xEE {
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            let payload = bytes.get(i + 4..i + 4 + len.saturating_sub(2));
            if let Some(p) = payload
                && p.len() >= 12
                && &p[..5] == b"Adobe"
            {
                // Payload: "Adobe" + version(2) + flags0(2) + flags1(2) + transform(1).
                return (true, p[11] == 2);
            }
            i += 2 + len;
        } else {
            i += 1;
        }
    }
    (false, false)
}

/// Generic fallback for CMYK JPEGs without a usable embedded profile: convert
/// conventional-ink samples to opaque RGBA8 with the classic ink model
/// (poppler's `cmyk2rgb`): `R = (255 − C)·(255 − K)/255` — white paper stays
/// white, full black stays black, pure inks keep their expected RGB hues.
/// For YCCK the RGB channels are display colors decoded from YCbCr (not ink),
/// so only the K factor applies: `R = R_disp·(255 − K)/255`.
///
/// This is a deterministic approximate display conversion — an untagged CMYK
/// image has no uniquely correct appearance (it depends on the printing
/// condition), so we define one. Division uses the fast `(t + (t >> 8)) >> 8`
/// ≈ t/255 trick (exact to ±1, like poppler); u32 math cannot overflow
/// (`c·nk + 128 ≤ 65153`) and `nk − div ≥ 0`, so no clamping is needed.
fn cmyk_fallback_to_rgba(ink: &[u8], ycck: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(ink.len());
    for px in ink.chunks_exact(4) {
        let (c, m, y, k) = (px[0] as u32, px[1] as u32, px[2] as u32, px[3] as u32);
        let nk = 255 - k;
        let scale = |v: u32| {
            let t = v * nk + 128;
            (t + (t >> 8)) >> 8
        };
        if ycck {
            out.extend_from_slice(&[scale(c) as u8, scale(m) as u8, scale(y) as u8, 255]);
        } else {
            out.extend_from_slice(&[
                (nk - scale(c)) as u8,
                (nk - scale(m)) as u8,
                (nk - scale(y)) as u8,
                255,
            ]);
        }
    }
    out
}

/// Convert conventional CMYK8 via an embedded ICC to opaque sRGB RGBA8. None
/// when the profile or qcms transform is unusable — the caller falls back to
/// the generic conversion.
fn cmyk_icc_to_rgba(profile: &[u8], cmyk: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    let n = (w as usize) * (h as usize);
    let mut rgb = vec![0u8; n * 3];
    icc::cmyk_to_srgb_rgb8(profile, cmyk, &mut rgb).ok()?;
    Some(rgb_to_rgba(&rgb, w, h))
}

/// True if every RGBA pixel is fully opaque (alpha 255).
fn is_opaque(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).all(|px| px[3] == 255)
}

/// Premultiply R,G,B by A in place (gamma space — consistent with the rest of the
/// decode/resize path). Needed before downscaling and before the GPU's
/// premultiplied-alpha blend: it zeroes the garbage RGB encoders leave in
/// fully-transparent pixels and lets the bilinear sampler interpolate edges
/// without colour fringing.
fn premultiply_alpha(rgba: &mut [u8]) {
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3] as u32;
        px[0] = ((px[0] as u32 * a + 127) / 255) as u8;
        px[1] = ((px[1] as u32 * a + 127) / 255) as u8;
        px[2] = ((px[2] as u32 * a + 127) / 255) as u8;
    }
}

/// Returns (w, h, gray, normalized pixels [1ch gray or 4ch rgba], icc profile).
/// The ICC profile (if any) is read from the same decode — no second parse.
type Decoded = (u32, u32, bool, Vec<u8>, Option<Vec<u8>>);

fn decode_png(bytes: &[u8]) -> Result<Decoded, String> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    // Normalize to 8-bit colour: down-convert 16-bit → 8-bit and expand
    // palette / sub-8-bit grayscale / tRNS. Without this a 16-bit PNG reports 6
    // (RGB16) or 8 (RGBA16) bytes-per-pixel below and fails to decode (and a
    // paletted PNG would be misread as 1-channel gray). No-op for plain 8-bit.
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png read_info: {e}"))?;
    let size = reader.output_buffer_size().ok_or("png: no buffer size")?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png next_frame: {e}"))?;
    let (w, h) = (info.width, info.height);
    let ch = info.buffer_size() / ((w as usize) * (h as usize));
    let icc = reader.info().icc_profile.as_deref().map(<[u8]>::to_vec);
    match ch {
        1 => Ok((w, h, true, buf, icc)),
        2 => Ok((w, h, true, ga_to_gray(&buf), icc)),
        3 => Ok((w, h, false, rgb_to_rgba(&buf, w, h), icc)),
        4 => Ok((w, h, false, buf, icc)),
        other => Err(format!("png: unsupported channel count {other}")),
    }
}

fn decode_jpeg(bytes: &[u8]) -> Result<Decoded, String> {
    use jpeg_decoder::PixelFormat;
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let pixels = d.decode().map_err(|e| format!("jpeg decode: {e}"))?;
    let info = d.info().ok_or("jpeg: no info")?;
    let (w, h) = (info.width as u32, info.height as u32);
    let icc = d.icc_profile();
    match info.pixel_format {
        PixelFormat::L8 => Ok((w, h, true, pixels, icc)),
        PixelFormat::RGB24 => Ok((w, h, false, rgb_to_rgba(&pixels, w, h), icc)),
        // CMYK32 (ordinary CMYK and Adobe YCCK are both normalized to this by
        // jpeg-decoder). jpeg-decoder outputs 255 − stored, so the sample
        // polarity mirrors the file's convention: Adobe files (APP14) store
        // inverted CMYK and therefore already yield conventional ink, while
        // US files yield complemented samples. Prefer the embedded CMYK ICC
        // via qcms when it is usable (converted to conventional ink first);
        // otherwise use the documented generic fallback. Either way the
        // result is opaque sRGB RGBA8 and the CMYK profile is consumed — the
        // downstream RGBA ICC path must never see a CMYK profile applied to
        // already-converted pixels. YCCK's RGB channels are display colors
        // from YCbCr (not ink), so it never takes the ICC path.
        PixelFormat::CMYK32 => {
            let n = (w as usize) * (h as usize);
            if pixels.len() != n * 4 {
                return Err(format!(
                    "jpeg: cmyk buffer {} bytes for {w}x{h} ({n} px)",
                    pixels.len()
                ));
            }
            let (adobe, ycck) = jpeg_cmyk_convention(bytes);
            let mut cmyk = pixels;
            // Normalize the decoder output to conventional ink (0 = no ink),
            // which is what the ICC profiles describe and qcms expects: US
            // files make jpeg-decoder's 255 − stored complemented (invert);
            // Adobe files already yield ink.
            if !adobe {
                invert_cmyk(&mut cmyk);
            }
            if !ycck
                && let Some(p) = &icc
                && icc::is_cmyk(p)
                && let Some(rgba) = cmyk_icc_to_rgba(p, &cmyk, w, h)
            {
                return Ok((w, h, false, rgba, None));
            }
            Ok((w, h, false, cmyk_fallback_to_rgba(&cmyk, ycck), None))
        }
        other => Err(format!("jpeg: unsupported pixel format {other:?}")),
    }
}

/// Decode JPEG XL via the pure-Rust `jxl-oxide` (with the pure-Rust `moxcms`
/// CMS enabled for CMYK sources). Renders the first frame (ignores animation)
/// and normalizes to the same gray/RGBA8 + ICC contract as the others.
///
/// CMYK and CMYKA sources are profile-described (an embedded CMYK ICC), which
/// jxl-oxide's basic color path cannot process — we request an sRGB color
/// encoding with perceptual intent before rendering, so jxl-oxide + moxcms
/// convert CMYK → sRGB while samples are still floating point (avoiding a
/// CMYK8 round trip). Non-CMYK images render in their own encoding and keep the
/// original ICC for the downstream qcms → sRGB step, exactly like JPEG/PNG.
fn decode_jxl(bytes: &[u8]) -> Result<Decoded, String> {
    use jxl_oxide::{EnumColourEncoding, JxlImage, PixelFormat, RenderingIntent};
    let mut image = JxlImage::builder()
        .read(std::io::Cursor::new(bytes))
        .map_err(|e| format!("jxl read: {e}"))?;
    let (w, h) = (image.width(), image.height());
    // The *source* pixel format, captured before any color-encoding request
    // (pixel_format() reflects the requested encoding).
    let fmt = image.pixel_format();
    let is_cmyk = matches!(fmt, PixelFormat::Cmyk | PixelFormat::Cmyka);
    let icc = image.original_icc().map(<[u8]>::to_vec);
    if is_cmyk {
        image.request_color_encoding(EnumColourEncoding::srgb(RenderingIntent::Perceptual));
    }
    // A CMYK JXL without a usable embedded CMYK ICC fails here with the CMS
    // error surfaced by the render (a descriptive failure, not a silent
    // fallback: JXL CMYK is explicitly profile-described).
    let render = image
        .render_frame(0)
        .map_err(|e| format!("jxl{} render: {e}", if is_cmyk { " cmyk" } else { "" }))?;
    // `stream()` yields only the display channels (color + black when still
    // CMYK + alpha when present) — unrelated extra channels (spot colors,
    // depth, …) never corrupt the stride. After the sRGB request a CMYK(A)
    // source streams RGB(A): the black channel is folded into RGB by the CMS.
    let mut stream = render.stream();
    let ch = stream.channels();
    let expect = match fmt {
        PixelFormat::Gray => 1,
        PixelFormat::Graya => 2,
        PixelFormat::Rgb | PixelFormat::Cmyk => 3,
        PixelFormat::Rgba | PixelFormat::Cmyka => 4,
    };
    if ch != expect {
        return Err(format!(
            "jxl: stream has {ch} channels, expected {expect} for {fmt:?}"
        ));
    }
    let n = (w as usize) * (h as usize);
    let mut samples = vec![0f32; n * expect as usize];
    let mut got = 0;
    loop {
        let k = stream.write_to_buffer(&mut samples[got..]);
        if k == 0 {
            break;
        }
        got += k;
    }
    if got != samples.len() {
        return Err(format!(
            "jxl: stream wrote {got} of {} samples",
            samples.len()
        ));
    }
    // jxl-oxide samples are f32 (≈[0,1] for SDR); clamp handles any HDR overshoot.
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    match expect {
        // 1ch gray — map straight through.
        1 => Ok((w, h, true, samples.iter().map(|&v| to_u8(v)).collect(), icc)),
        // Gray+alpha — drop alpha to match the 1ch gray contract (like `ga_to_gray`).
        2 => Ok((
            w,
            h,
            true,
            samples.chunks_exact(2).map(|px| to_u8(px[0])).collect(),
            icc,
        )),
        // RGB(A) → RGBA8. CMYK(A) sources converted to sRGB here must not
        // return the original CMYK profile as though it described the RGB
        // pixels (the downstream qcms path would corrupt them) — no profile
        // means "treat as sRGB".
        ch @ (3 | 4) => {
            let mut pixels = vec![0u8; n * 4];
            for (px, out) in samples
                .chunks_exact(ch as usize)
                .zip(pixels.chunks_exact_mut(4))
            {
                out[0] = to_u8(px[0]);
                out[1] = to_u8(px[1]);
                out[2] = to_u8(px[2]);
                out[3] = if ch == 4 { to_u8(px[3]) } else { 255 };
            }
            Ok((w, h, false, pixels, if is_cmyk { None } else { icc }))
        }
        _ => unreachable!("channel count validated against the pixel format"),
    }
}

/// Decode a Photoshop document to its flattened composite (the "merged image
/// data" Photoshop stores), via the pure-Rust `psd` crate. Handles 8-bit RGB(A)
/// documents only — CMYK / 16-bit / grayscale-mode / PSB files error out and the
/// page shows as failed. yosh reads PSD when browsing a folder/archive but is not
/// a registered `.psd` handler (that stays Photoshop's).
fn decode_psd(bytes: &[u8]) -> Result<Decoded, String> {
    let psd = psd::Psd::from_bytes(bytes).map_err(|e| format!("psd: {e:?}"))?;
    let (w, h) = (psd.width(), psd.height());
    // rgba() is the pre-composited final image: [R,G,B,A, …], len = w*h*4.
    Ok((w, h, false, psd.rgba(), None))
}

fn decode_other(bytes: &[u8]) -> Result<Decoded, String> {
    let guessed = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image: {e}"))?;
    // TGA has no magic bytes, so content-guessing yields no format. Since the file
    // already passed the image-extension allowlist, fall back to decoding it as TGA.
    let reader = if guessed.format().is_some() {
        guessed
    } else {
        image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Tga)
    };
    let mut decoder = reader.into_decoder().map_err(|e| format!("image: {e}"))?;
    let icc = decoder.icc_profile().ok().flatten();
    let img = image::DynamicImage::from_decoder(decoder).map_err(|e| format!("image: {e}"))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((w, h, false, rgba.into_raw(), icc))
}

/// Expand a grayscale (R8) image to RGBA8 (r=g=b, a=255). Used for egui
/// thumbnails, since egui samples textures as RGBA (an R8 texture would render
/// red). No-op for images that are already color.
pub fn to_rgba_image(img: DecodedImage) -> DecodedImage {
    if !img.gray {
        return img;
    }
    let mut pixels = Vec::with_capacity(img.pixels.len() * 4);
    for &g in &img.pixels {
        pixels.extend_from_slice(&[g, g, g, 255]);
    }
    DecodedImage {
        w: img.w,
        h: img.h,
        src_w: img.src_w,
        src_h: img.src_h,
        gray: false,
        path: img.path,
        color_detection: img.color_detection,
        pixels,
    }
}

/// JPEG XL signature: either a bare codestream (`FF 0A`) or the ISOBMFF
/// container's 12-byte JXL box (`\0\0\0\x0C JXL \r \n \x87 \n`).
fn is_jxl(bytes: &[u8]) -> bool {
    const JXL_BOX: [u8; 12] =
        [0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A];
    bytes.starts_with(&[0xFF, 0x0A]) || bytes.starts_with(&JXL_BOX)
}

/// The smallest `jpeg-decoder` IDCT scale (in eighths: 1, 2, 4 or 8) whose output
/// height still **covers** `target_h`. The decoder can run a reduced-size inverse
/// DCT — 1/8, 1/4, 1/2 or full — which costs a fraction of the full IDCT and skips
/// the corresponding share of the upsample/color-convert work. Choosing the
/// smallest covering scale means the CPU resize still does a real (never an
/// upscaling) reduction to the exact target, so the LQ tier's output size is
/// unchanged — only the work to get there shrinks (≈4–16× less IDCT on a thumbnail).
fn idct_eighths(src_h: u32, target_h: u32) -> u32 {
    [1u32, 2, 4]
        .into_iter()
        .find(|&s| src_h.saturating_mul(s).div_ceil(8) >= target_h)
        .unwrap_or(8)
}

/// JPEG decode for the **LQ tier only**, asking the decoder for an IDCT-reduced
/// image when the target is far below native. Returns the usual [`Decoded`] (whose
/// `w`/`h` are the *reduced* buffer's) plus the file's true source dimensions,
/// which the caller must restore onto the [`DecodedImage`] — `src_w`/`src_h` drive
/// the zoom readout and the 1:1 decode target, and must always describe the file,
/// not the buffer.
fn decode_jpeg_scaled(bytes: &[u8], target_h: u32) -> Result<(Decoded, (u32, u32)), String> {
    use jpeg_decoder::PixelFormat;
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    d.read_info().map_err(|e| format!("jpeg read_info: {e}"))?;
    let info = d.info().ok_or("jpeg: no info")?;
    // 4-component (CMYK / YCCK) JPEGs go to the `image` crate, as in `decode_jpeg`.
    if info.pixel_format == PixelFormat::CMYK32 {
        let d = decode_other(bytes)?;
        let src = (d.0, d.1);
        return Ok((d, src));
    }
    let (src_w, src_h) = (info.width as u32, info.height as u32);
    let s = idct_eighths(src_h, target_h);
    if s < 8 {
        // Request the reduced size explicitly rather than passing the caller's
        // target: `choose_idct_size` matches on *either* axis, so an aspect-correct
        // request is what makes it land on the scale computed above.
        let req = |v: u32| v.saturating_mul(s).div_ceil(8).clamp(1, u16::MAX as u32) as u16;
        d.scale(req(src_w), req(src_h)).map_err(|e| format!("jpeg scale: {e}"))?;
    }
    let pixels = d.decode().map_err(|e| format!("jpeg decode: {e}"))?;
    let info = d.info().ok_or("jpeg: no info")?; // output size (post-scale)
    let (w, h) = (info.width as u32, info.height as u32);
    let icc = d.icc_profile();
    let decoded = match info.pixel_format {
        PixelFormat::L8 => (w, h, true, pixels, icc),
        PixelFormat::RGB24 => (w, h, false, rgb_to_rgba(&pixels, w, h), icc),
        other => return Err(format!("jpeg: unsupported pixel format {other:?}")),
    };
    Ok((decoded, (src_w, src_h)))
}

/// Decode to full resolution (no resize), normalized to gray (1ch) or RGBA8.
fn decode_raw(bytes: &[u8]) -> Result<Decoded, String> {
    if bytes.starts_with(&PNG_SIG) {
        decode_png(bytes)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        decode_jpeg(bytes)
    } else if is_jxl(bytes) {
        decode_jxl(bytes)
    } else if bytes.starts_with(b"8BPS") {
        decode_psd(bytes)
    } else {
        decode_other(bytes)
    }
}

/// Decide whether an RGBA buffer is *effectively* grayscale within `threshold`.
/// Port of MangaJaNai's `cv_image_is_grayscale` (run_upscale.py): for every pixel
/// that isn't pure black or pure white, sum the (saturating) channel-pair
/// differences beyond `threshold`; the image is gray if the mean per-channel
/// difference is `<= threshold / 12`. Channel order is irrelevant (all pairs).
fn rgba_is_grayscale(rgba: &[u8], threshold: i32) -> bool {
    let mut diff_sum: u64 = 0;
    let mut non_bw: u64 = 0;
    for px in rgba.chunks_exact(4) {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        if (r == 0 && g == 0 && b == 0) || (r == 255 && g == 255 && b == 255) {
            continue; // exclude pure black / pure white
        }
        non_bw += 1;
        // cv2.subtract saturates at 0: max(|a-b| - threshold, 0).
        let rg = ((r - g).abs() - threshold).max(0);
        let rb = ((r - b).abs() - threshold).max(0);
        let gb = ((g - b).abs() - threshold).max(0);
        diff_sum += (rg + rb + gb) as u64;
    }
    if non_bw == 0 {
        return false; // entirely pure black/white → treat as color (MJN does)
    }
    let ratio = diff_sum as f64 / (non_bw as f64 * 3.0);
    ratio <= threshold as f64 / 12.0
}

/// Classify an opaque RGBA page. `true` means it is safe to collapse to one
/// luminance channel. OGSOV accepts RGB, so alpha is stripped only for ML mode.
fn detect_color(
    rgba: &[u8],
    w: u32,
    h: u32,
    gray_by_channels: bool,
    opaque: bool,
    mode: ColorDetection,
) -> ColorDetectionOutcome {
    if gray_by_channels {
        return ColorDetectionOutcome::SourceGray;
    }
    if !opaque {
        return ColorDetectionOutcome::Transparent;
    }
    match mode {
        ColorDetection::Off => ColorDetectionOutcome::Off,
        ColorDetection::Traditional => ColorDetectionOutcome::Traditional {
            is_color: !rgba_is_grayscale(rgba, GRAYSCALE_THRESHOLD),
        },
        ColorDetection::Ml => {
            // OGSOV feature extraction removes its top 17 pixels. Tiny icons and
            // malformed zero-sized images cannot be classified by that model.
            if (w as usize).saturating_mul(h as usize) <= 17 {
                return ColorDetectionOutcome::OgsovUnavailable;
            }
            let Some(model) = ogsov::embedded_model() else {
                return ColorDetectionOutcome::OgsovUnavailable;
            };
            let result = model.detect_rgba(rgba, w as usize, h as usize);
            ColorDetectionOutcome::Ogsov {
                is_color: result.is_color,
                confidence: result.confidence,
            }
        }
    }
}

/// Collapse RGBA to a single luminance channel (ITU-R 601, matching cv2's
/// `COLOR_BGR2GRAY`): `Y = 0.299R + 0.587G + 0.114B`.
fn rgba_to_luma(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .map(|px| {
            let y = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
            y.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Grayscale strategy: resample in **true linear light** to kill halftone moiré.
/// Linearize the sRGB-encoded source to 16-bit linear luminance, Catmull-Rom
/// resample in that space, then re-encode through the Dot Gain 20% curve (which
/// darkens, keeping screentones inky). 16-bit intermediate avoids shadow banding.
fn downscale_gray(
    gray: &[u8],
    w: u32,
    h: u32,
    tw: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    // sRGB device → 16-bit linear luminance (native-endian U16 byte buffer).
    let mut lin = Vec::with_capacity(gray.len() * 2);
    for &v in gray {
        lin.extend_from_slice(&tone::SRGB_TO_LINEAR[v as usize].to_ne_bytes());
    }
    let src = ImageRef::new(w, h, &lin, PixelType::U16).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, PixelType::U16);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom)),
        )
        .map_err(|e| format!("resize: {e}"))?;
    // Linear luminance → Dot Gain 20% device (8-bit).
    let enc = tone::linear_to_dotgain();
    let bytes = dst.into_vec();
    let pixels: Vec<u8> = bytes
        .chunks_exact(2)
        .map(|c| enc[u16::from_ne_bytes([c[0], c[1]]) as usize])
        .collect();
    Ok(DecodedImage {
        w: tw,
        h: target_h,
        src_w: w,
        src_h: h,
        gray: true,
        path: ResizePath::GrayLinear,
        color_detection: ColorDetectionOutcome::NotRun,
        pixels,
    })
}

/// Color strategy (MangaJaNai `standard_resize`): Lanczos3 in gamma space, no
/// color conversion.
fn downscale_color(
    rgba: &[u8],
    w: u32,
    h: u32,
    tw: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let src = ImageRef::new(w, h, rgba, PixelType::U8x4).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, PixelType::U8x4);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3)),
        )
        .map_err(|e| format!("resize: {e}"))?;
    Ok(DecodedImage {
        w: tw,
        h: target_h,
        src_w: w,
        src_h: h,
        gray: false,
        path: ResizePath::Color,
        color_detection: ColorDetectionOutcome::NotRun,
        pixels: dst.into_vec(),
    })
}

/// LQ grayscale: a fast 8-bit Bilinear downscale in gamma space. Skips the HQ
/// cost — the 32M-px sRGB→linear pass, the U16 Catmull-Rom, and the Dot-Gain
/// re-encode — so it's several times faster (and shows some screentone moiré).
/// Used transiently while seeking; the page re-decodes via `downscale_gray` on
/// settle.
fn downscale_gray_fast(
    gray: &[u8],
    w: u32,
    h: u32,
    tw: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let src = ImageRef::new(w, h, gray, PixelType::U8).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, PixelType::U8);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )
        .map_err(|e| format!("resize: {e}"))?;
    Ok(DecodedImage {
        w: tw,
        h: target_h,
        src_w: w,
        src_h: h,
        gray: true,
        path: ResizePath::GrayLinear,
        color_detection: ColorDetectionOutcome::NotRun,
        pixels: dst.into_vec(),
    })
}

/// LQ color: a fast 8-bit Bilinear downscale (vs the HQ Lanczos3).
fn downscale_color_fast(
    rgba: &[u8],
    w: u32,
    h: u32,
    tw: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let src = ImageRef::new(w, h, rgba, PixelType::U8x4).map_err(|e| format!("resize src: {e}"))?;
    let mut dst = Image::new(tw, target_h, PixelType::U8x4);
    resizer
        .resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )
        .map_err(|e| format!("resize: {e}"))?;
    Ok(DecodedImage {
        w: tw,
        h: target_h,
        src_w: w,
        src_h: h,
        gray: false,
        path: ResizePath::Color,
        color_detection: ColorDetectionOutcome::NotRun,
        pixels: dst.into_vec(),
    })
}

/// Decode page bytes (any supported format) and downscale to `target_h` on CPU,
/// picking the grayscale or color resize strategy.
pub fn decode_and_downscale(
    bytes: &[u8],
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    decode_and_downscale_with_options(bytes, target_h, DecodeOptions::default(), resizer)
}

/// Option-aware sibling of [`decode_and_downscale`].
pub fn decode_and_downscale_with_options(
    bytes: &[u8],
    target_h: u32,
    options: DecodeOptions,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let (w, h, gray_by_channels, mut full, profile) = decode_raw(bytes)?;
    // Color-manage to sRGB before any resampling: a color page tagged with a
    // wider profile (e.g. Display P3) would otherwise render desaturated. The
    // profile comes from the same decode (no second parse); only color images
    // carrying a non-sRGB profile pay the transform — grayscale/untagged pages
    // are untouched, so seek throughput is unaffected. A *grayscale* ICC (e.g. a
    // Dot Gain profile on a monochrome AVIF) is skipped: it can't be applied to
    // the RGBA buffer (channel mismatch → white), and the gray resize path below
    // handles its tone instead. A *CMYK* ICC (a print-sourced JPEG/TIFF) is
    // skipped for the same reason — see `icc::is_cmyk`, which fails silently
    // rather than loudly — and because the decoder already converted those pixels
    // to RGB, so the profile no longer describes the buffer.
    if !gray_by_channels
        && let Some(p) = &profile
            && !icc::is_srgb(p) && !icc::is_gray(p) && !icc::is_cmyk(p) {
                icc::to_srgb_rgba(p, &mut full);
            }
    // Decoded size = scale to the display height (never upscaling past the source).
    // A page bigger than one GPU texture is rejected (full res is preserved up to
    // the limit; we don't silently downscale a 16k-px image to a blurry one).
    let (tw, th) = target_dims(w, h, target_h);
    check_fits(tw, th)?;

    // A color page may carry transparency; opaque pages (the manga norm) keep the
    // unchanged fast path. Computed once and reused for the routing decisions.
    let opaque = gray_by_channels || is_opaque(&full);
    let color_detection = detect_color(
        &full,
        w,
        h,
        gray_by_channels,
        opaque,
        options.color_detection,
    );

    // No downscale needed (source already fits the target and the GPU limit):
    // show the decoded pixels unaltered — no resampling, no tone remap.
    if tw == w && th == h {
        if !opaque {
            premultiply_alpha(&mut full);
        }
        return Ok(DecodedImage {
            w,
            h,
            src_w: w,
            src_h: h,
            gray: gray_by_channels,
            path: ResizePath::None,
            color_detection,
            pixels: full,
        });
    }

    let mut img = if gray_by_channels {
        // Already single-channel (1ch / GA PNG, L8 JPEG) — no scan needed.
        downscale_gray(&full, w, h, tw, th, resizer)
    } else if color_detection.is_color() == Some(false) {
        // Color-stored but visually gray (and opaque) → collapse to luma, gray
        // strategy. Transparent images skip this so their alpha is preserved.
        let mut img = downscale_gray(&rgba_to_luma(&full), w, h, tw, th, resizer)?;
        img.path = ResizePath::GrayFromColor; // same gray path, but source was color
        Ok(img)
    } else {
        if !opaque {
            premultiply_alpha(&mut full);
        }
        downscale_color(&full, w, h, tw, th, resizer)
    }?;
    img.color_detection = color_detection;
    Ok(img)
}

/// LQ sibling of `decode_and_downscale`: decode + a cheap gamma-space Bilinear
/// resize, skipping ICC color management, the visually-grayscale detection, and
/// the linear-light path. The fast tier shown while seeking; a native-sized page
/// (no downscale) returns the same pixels HQ would, so nothing is lost there.
///
/// JPEGs additionally decode through a reduced-size IDCT (see `idct_eighths`) —
/// the biggest single win for the whole-volume thumbnail fill, which used to
/// full-res-decode every page of a volume just to shrink it to 540 px. **The HQ
/// path never does this**: its output must be the one exact resample of the full
/// source data.
fn decode_and_downscale_lq(
    bytes: &[u8],
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    let ((w, h, gray_by_channels, full, _profile), (src_w, src_h)) =
        if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            decode_jpeg_scaled(bytes, target_h)?
        } else {
            let d = decode_raw(bytes)?;
            let src = (d.0, d.1);
            (d, src)
        };
    let (tw, th) = target_dims(w, h, target_h);
    check_fits(tw, th)?;
    let mut img = if tw == w && th == h {
        DecodedImage { w, h, src_w: w, src_h: h, gray: gray_by_channels, path: ResizePath::None, color_detection: ColorDetectionOutcome::NotRun, pixels: full }
    } else if gray_by_channels {
        downscale_gray_fast(&full, w, h, tw, th, resizer)?
    } else {
        downscale_color_fast(&full, w, h, tw, th, resizer)?
    };
    // The IDCT hint means the decoded buffer may be smaller than the file, so the
    // source dims have to come from the header, not from what we decoded.
    img.src_w = src_w;
    img.src_h = src_h;
    Ok(img)
}

/// A decoded page: a single still image (the common case), an animation as an
/// ordered list of `(frame, delay_ms)` (GIF/WebP — auto-plays), or a set of
/// static layers (an `.ico`'s multiple resolutions — stepped manually, no
/// playback). Single-frame/-layer inputs collapse to `Still`.
pub enum DecodedPage {
    Still(DecodedImage),
    Animated(Vec<(DecodedImage, u32)>),
    Layered(Vec<DecodedImage>),
}

/// Downscale one already-decoded, canvas-composited RGBA frame to `target_h` via
/// the color (Lanczos3) path — the strategy for animated GIF frames (palette
/// color, not manga screentone, so the gray linear-light path doesn't apply).
fn downscale_rgba_frame(
    mut rgba: Vec<u8>,
    w: u32,
    h: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedImage, String> {
    // GIF/WebP frames can be transparent — premultiply before resize/return.
    if !is_opaque(&rgba) {
        premultiply_alpha(&mut rgba);
    }
    let (tw, th) = target_dims(w, h, target_h);
    check_fits(tw, th)?;
    if tw == w && th == h {
        return Ok(DecodedImage {
            w,
            h,
            src_w: w,
            src_h: h,
            gray: false,
            path: ResizePath::None,
            color_detection: ColorDetectionOutcome::NotRun,
            pixels: rgba,
        });
    }
    downscale_color(&rgba, w, h, tw, th, resizer)
}

/// Turn an animation's decoded frames into a `DecodedPage`: downscale each frame
/// (color path) and keep its delay. The decoder (`GifDecoder` / `WebPDecoder`)
/// hands back each frame **pre-composited to the full canvas** (disposal already
/// applied), so each is a complete same-size RGBA image. A single frame collapses
/// to `Still` so a non-animated file pays no animation overhead.
fn frames_to_page(
    frames: Vec<image::Frame>,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<DecodedPage, String> {
    if frames.is_empty() {
        return Err("animation: no frames".into());
    }
    let mut out: Vec<(DecodedImage, u32)> = Vec::with_capacity(frames.len());
    for frame in frames {
        // Delay as a ms ratio (numer/denom). Clamp tiny/zero delays to 100ms,
        // matching browsers (which treat <20ms as 100ms) so a 0ms frame can't pin
        // the loop.
        let (num, den) = frame.delay().numer_denom_ms();
        let ms = num.checked_div(den).unwrap_or(0);
        let delay = if ms < 20 { 100 } else { ms };
        let buf = frame.into_buffer(); // RgbaImage, full canvas
        let (w, h) = buf.dimensions();
        out.push((downscale_rgba_frame(buf.into_raw(), w, h, target_h, resizer)?, delay));
    }
    if out.len() == 1 {
        Ok(DecodedPage::Still(out.pop().unwrap().0))
    } else {
        Ok(DecodedPage::Animated(out))
    }
}

/// Decode every image inside an `.ico` (its multiple resolutions / "layers"),
/// largest first. Each entry decodes to RGBA8 at its own native size; they are
/// kept native (icons are tiny) and shown scaled to the page box.
fn decode_ico(bytes: &[u8]) -> Result<Vec<DecodedImage>, String> {
    let dir = ico::IconDir::read(std::io::Cursor::new(bytes)).map_err(|e| format!("ico: {e}"))?;
    let mut out: Vec<DecodedImage> = Vec::with_capacity(dir.entries().len());
    for entry in dir.entries() {
        let img = entry.decode().map_err(|e| format!("ico entry: {e}"))?;
        let (w, h) = (img.width(), img.height());
        let mut pixels = img.rgba_data().to_vec();
        // Icons are typically transparent — premultiply so the GPU blend composites
        // them over the background instead of showing garbage in clear areas.
        if !is_opaque(&pixels) {
            premultiply_alpha(&mut pixels);
        }
        out.push(DecodedImage {
            w,
            h,
            src_w: w,
            src_h: h,
            gray: false,
            path: ResizePath::None,
            color_detection: ColorDetectionOutcome::NotRun,
            pixels,
        });
    }
    if out.is_empty() {
        return Err("ico: no entries".into());
    }
    // Largest first, so the default-shown layer is the highest resolution.
    out.sort_by_key(|i| std::cmp::Reverse((i.w as u64) * (i.h as u64)));
    Ok(out)
}

/// Decode a page to a still image, an animation (GIF/WebP), or layered (`.ico`).
/// This is the entry point the decode pool uses; stills go through the unchanged
/// `decode_and_downscale` hot path.
pub fn decode_page(
    bytes: &[u8],
    target_h: u32,
    lq: bool,
    resizer: &mut Resizer,
) -> Result<DecodedPage, String> {
    decode_page_with_options(bytes, target_h, lq, DecodeOptions::default(), resizer)
}

/// Option-aware sibling of [`decode_page`].
pub fn decode_page_with_options(
    bytes: &[u8],
    target_h: u32,
    lq: bool,
    options: DecodeOptions,
    resizer: &mut Resizer,
) -> Result<DecodedPage, String> {
    // ICO: expose every contained image as a steppable layer (1 entry → still).
    if bytes.starts_with(&[0x00, 0x00, 0x01, 0x00]) {
        let mut layers = decode_ico(bytes)?;
        return Ok(if layers.len() == 1 {
            DecodedPage::Still(layers.pop().unwrap())
        } else {
            DecodedPage::Layered(layers)
        });
    }
    // GIF is always frame-decoded (a 1-frame GIF collapses back to a still).
    if bytes.starts_with(b"GIF8") {
        let frames = GifDecoder::new(std::io::Cursor::new(bytes))
            .map_err(|e| format!("gif: {e}"))?
            .into_frames()
            .collect_frames()
            .map_err(|e| format!("gif frames: {e}"))?;
        return frames_to_page(frames, target_h, resizer);
    }
    // WebP: frame-decode only when it's actually animated; a static WebP takes the
    // normal still path (with ICC color management) like any other image.
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
        && let Ok(dec) = WebPDecoder::new(std::io::Cursor::new(bytes))
            && dec.has_animation()
        {
            let frames = dec
                .into_frames()
                .collect_frames()
                .map_err(|e| format!("webp frames: {e}"))?;
            return frames_to_page(frames, target_h, resizer);
        }
    // Stills: the seek hot path. LQ uses the fast gamma-space resize; HQ is the
    // unchanged linear-light pipeline. (Animations/ICO above always decode HQ —
    // rare, and not the seek bottleneck.)
    let img = if lq {
        decode_and_downscale_lq(bytes, target_h, resizer)?
    } else {
        decode_and_downscale_with_options(bytes, target_h, options, resizer)?
    };
    Ok(DecodedPage::Still(img))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, Rgba, RgbaImage};

    fn encode_gif(frames: Vec<Frame>) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut buf);
            for f in frames {
                enc.encode_frame(f).unwrap();
            }
        } // drop encoder → flush trailer
        buf
    }

    fn frame(rgba: [u8; 4]) -> Frame {
        let img = RgbaImage::from_pixel(4, 4, Rgba(rgba));
        Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(100, 1))
    }

    fn encode_rgb_png(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, w, h);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().unwrap();
            writer
                .write_image_data(&rgb.repeat((w * h) as usize))
                .unwrap();
        }
        bytes
    }

    #[test]
    fn color_detection_off_preserves_color_storage() {
        let bytes = encode_rgb_png(32, 32, [120, 121, 120]);
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            &bytes,
            16,
            DecodeOptions {
                color_detection: ColorDetection::Off,
            },
            &mut resizer,
        )
        .unwrap();
        assert!(!img.gray);
        assert_eq!(img.path, ResizePath::Color);
        assert_eq!(img.color_detection, ColorDetectionOutcome::Off);
    }

    #[test]
    fn traditional_detection_keeps_current_gray_from_color_path() {
        let bytes = encode_rgb_png(32, 32, [120, 121, 120]);
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            &bytes,
            16,
            DecodeOptions {
                color_detection: ColorDetection::Traditional,
            },
            &mut resizer,
        )
        .unwrap();
        assert!(img.gray);
        assert_eq!(img.path, ResizePath::GrayFromColor);
        assert_eq!(
            img.color_detection,
            ColorDetectionOutcome::Traditional { is_color: false }
        );
    }

    #[test]
    fn ml_detection_is_conservative_without_weights() {
        if ml_color_detection_available() {
            return;
        }
        let bytes = encode_rgb_png(32, 32, [120, 120, 120]);
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            &bytes,
            16,
            DecodeOptions {
                color_detection: ColorDetection::Ml,
            },
            &mut resizer,
        )
        .unwrap();
        assert!(!img.gray);
        assert_eq!(img.path, ResizePath::Color);
        assert_eq!(img.color_detection, ColorDetectionOutcome::OgsovUnavailable);
    }

    #[test]
    fn ml_detection_routes_exact_rgb_gray_when_model_available() {
        if !ml_color_detection_available() {
            return;
        }
        let bytes = encode_rgb_png(32, 32, [120, 120, 120]);
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            &bytes,
            16,
            DecodeOptions {
                color_detection: ColorDetection::Ml,
            },
            &mut resizer,
        )
        .unwrap();
        assert!(img.gray);
        assert_eq!(img.path, ResizePath::GrayFromColor);
        assert_eq!(
            img.color_detection,
            ColorDetectionOutcome::Ogsov {
                is_color: false,
                confidence: 100,
            }
        );
        assert_eq!(
            img.color_detection.label(),
            "OGSOV · is_color: false · confidence: 100%"
        );
    }

    #[test]
    fn ml_detection_is_reported_at_native_size() {
        if !ml_color_detection_available() {
            return;
        }
        let bytes = encode_rgb_png(32, 32, [120, 120, 120]);
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            &bytes,
            32,
            DecodeOptions {
                color_detection: ColorDetection::Ml,
            },
            &mut resizer,
        )
        .unwrap();
        assert!(!img.gray, "native-size pixels remain as-is");
        assert_eq!(img.path, ResizePath::None);
        assert_eq!(
            img.color_detection,
            ColorDetectionOutcome::Ogsov {
                is_color: false,
                confidence: 100,
            }
        );
    }

    /// The IDCT-scale chooser must never pick a reduction that lands *below* the
    /// requested height — the CPU resize is a downscale-only path, so undershooting
    /// would mean upscaling a thumbnail (blurry) instead of reducing it. It must
    /// also actually reduce when there is headroom, which is the entire win.
    #[test]
    fn idct_scale_covers_the_target_and_still_reduces() {
        for src_h in [540u32, 1024, 1600, 2048, 4096, 5207] {
            for target in [90u32, 180, 360, 540, 1080, 2160] {
                let s = idct_eighths(src_h, target);
                assert!((1..=8).contains(&s), "src {src_h} → {target}: scale {s}");
                let out = src_h.saturating_mul(s).div_ceil(8);
                assert!(
                    out >= target.min(src_h),
                    "src {src_h} → {target}: reduced to {out}, below the target"
                );
                // And it is the *smallest* such scale (no wasted IDCT work).
                if s > 1 {
                    let smaller = src_h.saturating_mul(s / 2).div_ceil(8);
                    assert!(smaller < target, "src {src_h} → {target}: {s}/8 is bigger than needed");
                }
            }
        }
        // A page already at or below the target decodes at full scale.
        assert_eq!(idct_eighths(400, 540), 8);
        // A 540px thumb of a 4096px page: 1/8 (512) is short, 2/8 (1024) covers it.
        assert_eq!(idct_eighths(4096, 540), 2);
        // A huge page for a tiny thumb takes the cheapest IDCT there is.
        assert_eq!(idct_eighths(5207, 90), 1);
        // `u32::MAX` (the uncached 1:1 target) can't overflow into a bogus scale.
        assert_eq!(idct_eighths(5207, u32::MAX), 8);
    }

    #[test]
    fn multiframe_gif_decodes_as_animation() {
        let bytes = encode_gif(vec![frame([255, 0, 0, 255]), frame([0, 0, 255, 255])]);
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 4, false, &mut resizer).unwrap() {
            DecodedPage::Animated(fs) => {
                assert_eq!(fs.len(), 2, "both frames preserved");
                assert!(fs.iter().all(|(img, _)| img.w == 4 && img.h == 4 && !img.gray));
                // 100ms round-trips through the centisecond GIF delay field.
                assert!(fs.iter().all(|(_, d)| *d == 100), "delays = {:?}", fs.iter().map(|(_, d)| *d).collect::<Vec<_>>());
            }
            _ => panic!("expected an animation"),
        }
    }

    #[test]
    fn single_frame_gif_is_a_still() {
        let bytes = encode_gif(vec![frame([0, 255, 0, 255])]);
        let mut resizer = Resizer::new();
        assert!(matches!(
            decode_page(&bytes, 4, false, &mut resizer).unwrap(),
            DecodedPage::Still(_)
        ));
    }

    /// A minimal 4×4, 8-bit RGB PSD with raw (uncompressed) merged image data:
    /// header → empty color-mode/resources/layer sections → planar R,G,B planes.
    fn minimal_rgb_psd() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"8BPS"); // signature
        b.extend_from_slice(&1u16.to_be_bytes()); // version 1 = PSD
        b.extend_from_slice(&[0u8; 6]); // reserved
        b.extend_from_slice(&3u16.to_be_bytes()); // channels = RGB
        b.extend_from_slice(&4u32.to_be_bytes()); // height
        b.extend_from_slice(&4u32.to_be_bytes()); // width
        b.extend_from_slice(&8u16.to_be_bytes()); // depth = 8
        b.extend_from_slice(&3u16.to_be_bytes()); // color mode = RGB
        b.extend_from_slice(&0u32.to_be_bytes()); // color mode data: none
        b.extend_from_slice(&0u32.to_be_bytes()); // image resources: none
        b.extend_from_slice(&0u32.to_be_bytes()); // layer & mask info: none
        b.extend_from_slice(&0u16.to_be_bytes()); // compression = raw
        b.extend(std::iter::repeat_n(255u8, 16)); // R plane
        b.extend(std::iter::repeat_n(0u8, 16)); // G plane
        b.extend(std::iter::repeat_n(0u8, 16)); // B plane
        b
    }

    #[test]
    fn psd_decodes_flattened_composite() {
        let bytes = minimal_rgb_psd();
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 4, false, &mut resizer).unwrap() {
            DecodedPage::Still(img) => {
                assert_eq!((img.w, img.h), (4, 4));
                assert!(!img.gray);
                assert_eq!(&img.pixels[0..4], &[255, 0, 0, 255], "opaque red");
            }
            _ => panic!("psd should be a still"),
        }
    }

    #[test]
    fn png_16bit_decodes() {
        // A 16-bit RGBA PNG (e.g. from ImageMagick/Photoshop) — before the
        // normalize-to-8-bit transform this failed to decode (reported 8 channels).
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, 2, 2);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Sixteen);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[0xFFu8; 2 * 2 * 4 * 2]).unwrap(); // 2×2 RGBA16
        }
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 2, false, &mut resizer).unwrap() {
            DecodedPage::Still(img) => {
                assert_eq!((img.w, img.h), (2, 2));
                assert!(!img.gray);
                assert_eq!(img.pixels.len(), 2 * 2 * 4, "down-converted to RGBA8");
            }
            _ => panic!("png is a still"),
        }
    }

    #[test]
    fn ico_decodes_as_layers() {
        use ico::{IconDir, IconDirEntry, IconImage, ResourceType};
        let mut dir = IconDir::new(ResourceType::Icon);
        for sz in [16u32, 32u32] {
            let img = IconImage::from_rgba_data(sz, sz, vec![0xFFu8; (sz * sz * 4) as usize]);
            dir.add_entry(IconDirEntry::encode(&img).unwrap());
        }
        let mut bytes = Vec::new();
        dir.write(&mut bytes).unwrap();
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 64, false, &mut resizer).unwrap() {
            DecodedPage::Layered(layers) => {
                assert_eq!(layers.len(), 2);
                assert_eq!((layers[0].w, layers[0].h), (32, 32), "largest layer first");
                assert_eq!((layers[1].w, layers[1].h), (16, 16));
            }
            _ => panic!("multi-entry ico should be Layered"),
        }
    }

    #[test]
    fn transparent_png_is_premultiplied() {
        // 2x2 RGBA: opaque red, a fully-transparent pixel with garbage white RGB,
        // opaque green, and a half-alpha pixel.
        let data: [u8; 16] = [
            255, 0, 0, 255, // opaque red
            255, 255, 255, 0, // transparent — garbage RGB must premultiply to 0
            0, 255, 0, 255, // opaque green
            200, 100, 50, 128, // half alpha
        ];
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, 2, 2);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&data).unwrap();
        }
        let mut resizer = Resizer::new();
        match decode_page(&bytes, 2, false, &mut resizer).unwrap() {
            DecodedPage::Still(img) => {
                assert!(!img.gray, "transparent image keeps the color/alpha path");
                assert_eq!(&img.pixels[4..8], &[0, 0, 0, 0], "transparent RGB zeroed");
                assert_eq!(&img.pixels[0..4], &[255, 0, 0, 255], "opaque unchanged");
                assert_eq!(&img.pixels[8..12], &[0, 255, 0, 255], "opaque unchanged");
                let p = &img.pixels[12..16]; // ~ rgb * 128/255
                assert_eq!(p[3], 128);
                assert!((p[0] as i32 - 100).abs() <= 1 && (p[1] as i32 - 50).abs() <= 1);
            }
            _ => panic!("png is a still"),
        }
    }

    #[test]
    fn tiff_and_qoi_decode_via_image_crate() {
        use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
        // These formats have no dedicated decoder in yosh — they round-trip through
        // the `image`-crate fallback (decode_other), proving the easy-batch support.
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(5, 3, Rgba([10, 20, 30, 255])));
        for fmt in [ImageFormat::Tiff, ImageFormat::Qoi, ImageFormat::Tga] {
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, fmt).unwrap();
            let bytes = buf.into_inner();
            let mut resizer = Resizer::new();
            match decode_page(&bytes, 3, false, &mut resizer).unwrap() {
                DecodedPage::Still(d) => assert_eq!((d.w, d.h), (5, 3), "{fmt:?} dims"),
                _ => panic!("{fmt:?} should be a still"),
            }
        }
    }

    // ---------------------------------------------------------------------
    // CMYK support (plan: docs/CMYK_IMPLEMENTATION_PLAN.md). Fixtures live in
    // `testdata/` and are generated by `testdata/make_fixtures.py`.
    // ---------------------------------------------------------------------

    macro_rules! fixture {
        ($name:literal) => {
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/", $name))
        };
    }

    /// The 4×4 grid of conventional-CMYK patches in the `cmyk_*.jpg`/`cmyk*.jxl`
    /// fixtures, plus each patch's sRGB reference: lcms2 (Pillow ImageCms,
    /// USWebCoatedSWOP, perceptual intent) printed by the generator script.
    const CMY: &[(u8, u8, u8, u8, [u8; 3])] = &[
        (0, 0, 0, 0, [255, 255, 255]),
        (0, 0, 0, 255, [35, 31, 32]),
        (255, 0, 0, 0, [0, 174, 239]),
        (0, 255, 0, 0, [236, 0, 140]),
        (0, 0, 255, 0, [255, 242, 0]),
        (255, 255, 0, 0, [46, 48, 146]),
        (255, 0, 255, 0, [0, 166, 80]),
        (0, 255, 255, 0, [237, 28, 36]),
        (64, 48, 48, 0, [191, 193, 194]),
        (128, 96, 96, 0, [139, 146, 149]),
        (192, 144, 144, 0, [90, 111, 115]),
        (32, 32, 32, 160, [107, 105, 106]),
        (255, 192, 64, 32, [4, 74, 124]),
        (64, 128, 192, 48, [164, 116, 72]),
        (16, 240, 32, 0, [224, 46, 131]),
        (200, 200, 200, 40, [80, 71, 69]),
    ];

    /// Index of patch `i`'s center pixel in the 64×64 fixture buffers (16px
    /// patches, 4×4 grid).
    fn patch_center(i: usize) -> usize {
        ((i / 4) * 16 + 8) * 64 + (i % 4) * 16 + 8
    }

    /// Per-channel tolerance vs the lcms2 references: qcms and moxcms differ
    /// from Little CMS in interpolation and intent handling, and the lossy
    /// JPEG / JXL fixtures add small codec error. The tests assert the colors
    /// are visibly right, not pixel-identical across CMS implementations.
    const CMS_TOLERANCE: i32 = 16;

    fn rgba_at(rgba: &[u8], px: usize) -> [u8; 4] {
        [
            rgba[px * 4],
            rgba[px * 4 + 1],
            rgba[px * 4 + 2],
            rgba[px * 4 + 3],
        ]
    }

    fn near(expected: [u8; 3], got: [u8; 4], tol: i32) -> bool {
        (0..3).all(|c| (expected[c] as i32 - got[c] as i32).abs() <= tol)
    }

    #[test]
    fn complemented_cmyk_fallback_maps_expected_colors() {
        // Inputs are conventional ink samples (0 = no ink) — the normalized
        // form decode_jpeg hands the fallback. `ycck` says the RGB channels
        // are display colors decoded from YCbCr rather than ink.
        let cases: &[([u8; 4], bool, [u8; 4])] = &[
            // CMYK: R = (255 − C)·(255 − K)/255.
            ([0, 0, 0, 0], false, [255, 255, 255, 255]), // white paper
            ([0, 0, 0, 255], false, [0, 0, 0, 255]),     // full black
            ([255, 0, 0, 0], false, [0, 255, 255, 255]), // cyan
            ([0, 255, 0, 0], false, [255, 0, 255, 255]), // magenta
            ([0, 0, 255, 0], false, [255, 255, 0, 255]), // yellow
            ([128, 128, 128, 128], false, [63, 63, 63, 255]), // (127·127)/255 = 63.2 → 63
            ([1, 1, 1, 0], false, [254, 254, 254, 255]), // (254·255)/255 = 254.5 → 254
            ([100, 200, 150, 0], false, [155, 55, 105, 255]), // (255−v)·255/255
            // YCCK: display RGB scaled by (1 − K).
            ([255, 255, 255, 0], true, [255, 255, 255, 255]), // white, no K
            ([0, 0, 0, 255], true, [0, 0, 0, 255]),           // black
            ([255, 128, 128, 128], true, [127, 64, 64, 255]), // 255·127/255 = 127
            ([255, 255, 255, 128], true, [127, 127, 127, 255]),
        ];
        for (input, ycck, want) in cases {
            let out = cmyk_fallback_to_rgba(input, *ycck);
            assert_eq!(
                &out[..],
                &want[..],
                "fallback({input:?}, ycck={ycck}) — RGBA ordering + alpha 255"
            );
        }
    }

    #[test]
    fn icc_signatures_are_detected_safely() {
        let sig = |s: &[u8]| {
            let mut p = vec![0u8; 20];
            p[16..20].copy_from_slice(s);
            p
        };
        assert!(icc::is_cmyk(&sig(b"CMYK")));
        assert!(!icc::is_cmyk(&sig(b"RGB ")));
        assert!(!icc::is_cmyk(&sig(b"GRAY")));
        assert!(icc::is_gray(&sig(b"GRAY")));
        assert!(!icc::is_gray(&sig(b"RGB ")));
        // Truncated / empty profiles must be rejected safely (no panics).
        assert!(!icc::is_cmyk(b""));
        assert!(!icc::is_cmyk(&[0u8; 16]));
        assert!(!icc::is_cmyk(&[0u8; 19]));
        assert!(!icc::is_gray(&[0u8; 19]));
        // The real fixture profile is recognized as CMYK.
        assert!(icc::is_cmyk(fixture!("USWebCoatedSWOP.icc")));
    }

    #[test]
    fn qcms_cmyk_transform_matches_lcms_references() {
        let profile = fixture!("USWebCoatedSWOP.icc");
        let mut cmyk = Vec::new();
        for (c, m, y, k, _) in CMY.iter().copied() {
            cmyk.extend_from_slice(&[c, m, y, k]);
        }
        let mut rgb = vec![0u8; CMY.len() * 3];
        icc::cmyk_to_srgb_rgb8(profile, &cmyk, &mut rgb).unwrap();
        for (i, (_, _, _, _, want)) in CMY.iter().enumerate() {
            let got = [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255];
            assert!(
                near(*want, got, CMS_TOLERANCE),
                "patch {i} ({:?}): qcms {got:?} vs lcms {want:?}",
                CMY[i]
            );
        }
    }

    #[test]
    fn cmyk_transform_rejects_bad_buffers() {
        let profile = fixture!("USWebCoatedSWOP.icc");
        // Incomplete CMYK pixel (len % 4 != 0).
        let mut rgb = vec![0u8; 3];
        assert!(icc::cmyk_to_srgb_rgb8(profile, &[0, 0, 0], &mut rgb).is_err());
        // Mismatched destination length — must not panic.
        let mut short = vec![0u8; 7];
        assert!(icc::cmyk_to_srgb_rgb8(profile, &[0u8; 4], &mut short).is_err());
        let mut long = vec![0u8; 4];
        assert!(icc::cmyk_to_srgb_rgb8(profile, &[0u8; 4], &mut long).is_err());
        // Malformed / non-CMYK profile data → descriptive error, not a panic.
        assert!(icc::cmyk_to_srgb_rgb8(b"not a profile", &[0u8; 4], &mut rgb).is_err());
    }

    #[test]
    fn cmyk_icc_jpeg_decodes_to_srgb() {
        let bytes = fixture!("cmyk_icc.jpg");
        let mut resizer = Resizer::new();
        let img = decode_page(bytes, 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("cmyk jpeg should be a still")
        };
        assert_eq!((img.w, img.h), (64, 64));
        assert!(!img.gray, "CMYK source is color-stored, not SourceGray");
        assert_eq!(img.pixels.len(), 64 * 64 * 4);
        assert_ne!(
            img.color_detection,
            ColorDetectionOutcome::SourceGray,
            "CMYK source must not report SourceGray"
        );
        for (i, (_, _, _, _, want)) in CMY.iter().enumerate() {
            let got = rgba_at(&img.pixels, patch_center(i));
            assert!(
                near(*want, got, CMS_TOLERANCE),
                "icc jpeg patch {i}: got {got:?} vs lcms {want:?}"
            );
        }
    }

    #[test]
    fn untagged_cmyk_jpeg_uses_generic_fallback() {
        let bytes = fixture!("cmyk_untagged.jpg");
        let mut resizer = Resizer::new();
        let img = decode_page(bytes, 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("still")
        };
        assert_eq!((img.w, img.h), (64, 64));
        assert!(!img.gray);
        assert_eq!(img.pixels.len(), 64 * 64 * 4);
        // The generic conversion must not invert the image: white stays white,
        // black stays black, and pure cyan keeps a blue-dominant hue.
        let white = rgba_at(&img.pixels, patch_center(0));
        let black = rgba_at(&img.pixels, patch_center(1));
        let cyan = rgba_at(&img.pixels, patch_center(2));
        assert!(near([255, 255, 255], white, 8), "white: {white:?}");
        assert!(near([0, 0, 0], black, 8), "black: {black:?}");
        assert!(
            cyan[0] <= 32 && cyan[1] >= 224 && cyan[2] >= 224,
            "cyan: {cyan:?} (inverted output would be red-ish)"
        );
    }

    #[test]
    fn us_convention_cmyk_jpeg_uses_generic_fallback() {
        // No Adobe APP14 marker: the file stores conventional CMYK (0 = no
        // ink), so jpeg-decoder complements it and the fallback must invert
        // back — white stays white, black stays black, cyan stays blue-ish.
        let bytes = fixture!("cmyk_us.jpg");
        let mut resizer = Resizer::new();
        let img = decode_page(bytes, 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("still")
        };
        assert_eq!((img.w, img.h), (64, 64));
        assert!(!img.gray);
        let white = rgba_at(&img.pixels, patch_center(0));
        let black = rgba_at(&img.pixels, patch_center(1));
        let cyan = rgba_at(&img.pixels, patch_center(2));
        assert!(near([255, 255, 255], white, 8), "white: {white:?}");
        assert!(near([0, 0, 0], black, 8), "black: {black:?}");
        assert!(
            cyan[0] <= 32 && cyan[1] >= 224 && cyan[2] >= 224,
            "cyan: {cyan:?} (inverted output would be red-ish)"
        );
    }

    #[test]
    fn ycck_jpeg_follows_the_cmyk32_path() {
        let bytes = fixture!("cmyk_ycck.jpg");
        let mut resizer = Resizer::new();
        let img = decode_page(bytes, 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("still")
        };
        assert_eq!((img.w, img.h), (64, 64), "YCCK decodes via CMYK32");
        assert!(!img.gray);
        assert_eq!(img.pixels.len(), 64 * 64 * 4);
        let white = rgba_at(&img.pixels, patch_center(0));
        let black = rgba_at(&img.pixels, patch_center(1));
        let cyan = rgba_at(&img.pixels, patch_center(2));
        assert!(near([255, 255, 255], white, 12), "white: {white:?}");
        assert!(near([0, 0, 0], black, 12), "black: {black:?}");
        assert!(
            cyan[0] <= 32 && cyan[1] >= 224 && cyan[2] >= 224,
            "cyan: {cyan:?}"
        );
    }

    #[test]
    fn cmyk_jxl_decodes_through_moxcms() {
        let bytes = fixture!("cmyk.jxl");
        let mut resizer = Resizer::new();
        let img = decode_page(bytes, 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("still")
        };
        assert_eq!((img.w, img.h), (64, 64));
        assert!(!img.gray, "CMYK JXL is color-stored");
        assert_eq!(img.pixels.len(), 64 * 64 * 4);
        assert_ne!(img.color_detection, ColorDetectionOutcome::SourceGray);
        for (i, (_, _, _, _, want)) in CMY.iter().enumerate() {
            let got = rgba_at(&img.pixels, patch_center(i));
            assert!(
                near(*want, got, CMS_TOLERANCE),
                "cmyk.jxl patch {i}: got {got:?} vs lcms {want:?}"
            );
        }
    }

    #[test]
    fn cmyka_jxl_preserves_alpha() {
        let bytes = fixture!("cmyka.jxl");
        let mut resizer = Resizer::new();
        let img = decode_page(bytes, 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("still")
        };
        assert_eq!((img.w, img.h), (64, 64));
        assert!(!img.gray);
        // The fixture's alpha is 0 on every 16th pixel, 255 elsewhere; at native
        // size the not-fully-opaque buffer is premultiplied (RGB zeroed where
        // alpha is 0) exactly like any other transparent image.
        let transparent = rgba_at(&img.pixels, 0);
        let opaque = rgba_at(&img.pixels, 1);
        assert_eq!(transparent, [0, 0, 0, 0], "alpha 0 pixel premultiplied");
        assert_eq!(opaque[3], 255, "alpha preserved");
        assert!(near([255, 255, 255], opaque, CMS_TOLERANCE), "{opaque:?}");
    }

    #[test]
    fn jxl_plain_formats_unchanged() {
        let mut resizer = Resizer::new();
        // Gray JXL stays single-channel.
        let img = decode_page(fixture!("gray.jxl"), 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("still")
        };
        assert!(
            img.gray && img.pixels.len() == 64 * 64,
            "gray.jxl → 1ch gray"
        );
        assert!(
            (img.pixels[0] as i32 - 128).abs() <= 3,
            "gray value preserved (lossy), got {}",
            img.pixels[0]
        );
        // Gray+alpha JXL drops alpha to the 1ch gray contract (like `ga_to_gray`).
        let img = decode_page(fixture!("graya.jxl"), 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("still")
        };
        assert!(
            img.gray && img.pixels.len() == 64 * 64,
            "graya.jxl → 1ch gray"
        );
        // RGB JXL → opaque RGBA.
        let img = decode_page(fixture!("rgb.jxl"), 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("still")
        };
        assert!(!img.gray && img.pixels.len() == 64 * 64 * 4);
        let blue = rgba_at(&img.pixels, 0);
        assert!(near([0, 0, 255], blue, 8), "blue, opaque: {blue:?}");
        // RGBA JXL: alpha (an extra channel) must not corrupt the stride.
        let img = decode_page(fixture!("rgba.jxl"), 64, false, &mut resizer).unwrap();
        let DecodedPage::Still(img) = img else {
            panic!("still")
        };
        assert!(!img.gray && img.pixels.len() == 64 * 64 * 4);
        assert_eq!(
            rgba_at(&img.pixels, 0),
            [0, 0, 0, 0],
            "alpha 0 premultiplied"
        );
        let opaque = rgba_at(&img.pixels, 1);
        // The lossy -d 1 encoder rings slightly at the hard alpha checkerboard
        // edges; the stride/alpha contract is what this test asserts, so only
        // red (the flat color) gets a tight bound.
        assert!(
            (opaque[0] as i32 - 255).abs() <= 8
                && (opaque[1] as i32 - 0).abs() <= 8
                && (opaque[2] as i32 - 0).abs() <= 32
                && opaque[3] == 255,
            "opaque red, alpha intact: {opaque:?}"
        );
    }

    #[test]
    fn malformed_cmyk_jxl_errors_descriptively() {
        // Truncating mid-container leaves a file whose header reads but whose
        // frame cannot render — the error must mention the jxl path.
        let truncated = &fixture!("cmyk.jxl")[..200];
        let mut resizer = Resizer::new();
        let err = match decode_page(truncated, 64, false, &mut resizer) {
            Ok(_) => panic!("truncated cmyk jxl unexpectedly decoded"),
            Err(e) => e,
        };
        assert!(err.contains("jxl"), "descriptive jxl error, got: {err}");
    }

    #[test]
    fn neutral_cmyk_traditional_routes_gray_from_color() {
        let bytes = fixture!("cmyk_neutral.jpg");
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            bytes,
            32,
            DecodeOptions {
                color_detection: ColorDetection::Traditional,
            },
            &mut resizer,
        )
        .unwrap();
        assert!(img.gray, "converted neutral CMYK collapses to luma");
        assert_eq!(img.path, ResizePath::GrayFromColor);
        assert_eq!(
            img.color_detection,
            ColorDetectionOutcome::Traditional { is_color: false }
        );
    }

    #[test]
    fn neutral_cmyk_ml_detection_receives_rgba() {
        if !ml_color_detection_available() {
            return;
        }
        let bytes = fixture!("cmyk_neutral.jpg");
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            bytes,
            32,
            DecodeOptions {
                color_detection: ColorDetection::Ml,
            },
            &mut resizer,
        )
        .unwrap();
        // OGSOV sees converted sRGB RGBA (never raw CMYK) — the verdict comes
        // from the model, and routing follows it. (The exact verdict on this
        // synthetic ramp depends on the embedded model; the deterministic
        // gray-routing guarantee is covered by the traditional-detection test.)
        let ColorDetectionOutcome::Ogsov {
            is_color,
            confidence,
        } = img.color_detection
        else {
            panic!(
                "expected an OGSOV verdict, got {}",
                img.color_detection.label()
            )
        };
        assert!(confidence >= 50, "decisive OGSOV verdict");
        assert_eq!(
            img.path,
            if is_color {
                ResizePath::Color
            } else {
                ResizePath::GrayFromColor
            },
            "routing follows the OGSOV verdict"
        );
    }

    #[test]
    fn color_cmyk_ml_detection_keeps_color() {
        if !ml_color_detection_available() {
            return;
        }
        let bytes = fixture!("cmyk_icc.jpg");
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            bytes,
            32,
            DecodeOptions {
                color_detection: ColorDetection::Ml,
            },
            &mut resizer,
        )
        .unwrap();
        // OGSOV sees the converted sRGB RGBA; the strongly colored patch grid
        // stays on the color path.
        let ColorDetectionOutcome::Ogsov {
            is_color,
            confidence,
        } = img.color_detection
        else {
            panic!(
                "expected an OGSOV verdict, got {}",
                img.color_detection.label()
            )
        };
        assert!(is_color, "color page stays color");
        assert!(confidence >= 50, "high-confidence color verdict");
        assert!(!img.gray);
        assert_eq!(img.path, ResizePath::Color);
    }

    #[test]
    fn detection_off_keeps_converted_cmyk_on_color_path() {
        let bytes = fixture!("cmyk_icc.jpg");
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            bytes,
            32,
            DecodeOptions {
                color_detection: ColorDetection::Off,
            },
            &mut resizer,
        )
        .unwrap();
        assert_eq!(img.color_detection, ColorDetectionOutcome::Off);
        assert!(!img.gray);
        assert_eq!(img.path, ResizePath::Color);
    }

    #[test]
    fn cmyka_jxl_is_transparent_and_skips_detection() {
        let bytes = fixture!("cmyka.jxl");
        let mut resizer = Resizer::new();
        let img = decode_and_downscale_with_options(
            bytes,
            32,
            DecodeOptions {
                color_detection: ColorDetection::Traditional,
            },
            &mut resizer,
        )
        .unwrap();
        // CMYKA converts to RGBA with alpha; detection is skipped and the color
        // resize path preserves the (premultiplied) alpha.
        assert_eq!(img.color_detection, ColorDetectionOutcome::Transparent);
        assert!(!img.gray);
        assert_eq!(img.path, ResizePath::Color);
        assert_eq!(
            img.pixels.len(),
            (img.w * img.h * 4) as usize,
            "alpha retained through the color path"
        );
    }
}

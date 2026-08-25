//! Windows-only implementation of the Explorer thumbnail provider.
//!
//! The provider implements `IThumbnailProvider` (produce a thumbnail) and
//! `IInitializeWithStream` (feed it the file bytes through Shell process
//! isolation — the provider deliberately never takes a file path). `Initialize`
//! copies the `IStream` to a unique temp file, `GetThumbnail` runs the shared
//! engine cover extraction + decode and returns an `HBITMAP`, and the temp file
//! is cleaned up when the object is released. Every cross-COM-boundary failure
//! returns an HRESULT; nothing here panics.

use std::io::Write as _;
use std::mem::size_of;
use std::path::PathBuf;
use std::ptr::null_mut;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, Ordering};

use fast_image_resize::Resizer;
use windows::Win32::Foundation::{
    CLASS_E_CLASSNOTAVAILABLE, CLASS_E_NOAGGREGATION, E_FAIL, E_INVALIDARG, E_NOINTERFACE, S_FALSE,
    S_OK,
};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, CreateDIBSection, DIB_RGB_COLORS, HBITMAP,
};
use windows::Win32::System::Com::{IClassFactory, IClassFactory_Impl, IStream, STREAM_SEEK_SET};
use windows::Win32::UI::Shell::PropertiesSystem::{
    IInitializeWithStream, IInitializeWithStream_Impl,
};
use windows::Win32::UI::Shell::{
    IThumbnailProvider, IThumbnailProvider_Impl, WTS_ALPHATYPE, WTSAT_ARGB, WTSAT_RGB,
};
use windows::core::{BOOL, ComObject, GUID, Interface, Ref, implement};

use yosh_engine::cover::cover_bytes;
use yosh_engine::decode::{DecodedImage, DecodedPage, decode_page};
use yosh_engine::meta;

/// The provider's class ID. Registered in the installer under `HKCU\Software\
/// Classes\CLSID\{...}` and bound to the comic extensions' `ShellEx` key. Any
/// GUID is valid; keep it stable so an uninstall/reinstall doesn't orphan cache
/// entries.
const PROVIDER_CLSID: GUID = GUID::from_u128(0xd3a1ee1d_fe0a_4772_902f_1fb50f3f9222);

/// Outstanding provider objects and `LockServer` pins, for `DllCanUnloadNow`.
static OBJECT_COUNT: AtomicI32 = AtomicI32::new(0);
static LOCK_COUNT: AtomicI32 = AtomicI32::new(0);

/// The thumbnail provider. Store only the temp-file path; everything else is
/// re-derived per `GetThumbnail` call.
#[implement(IThumbnailProvider, IInitializeWithStream)]
struct ThumbnailProvider {
    temp_path: Mutex<Option<PathBuf>>,
}

impl ThumbnailProvider {
    fn new() -> Self {
        Self {
            temp_path: Mutex::new(None),
        }
    }
}

impl Drop for ThumbnailProvider {
    fn drop(&mut self) {
        OBJECT_COUNT.fetch_sub(1, Ordering::SeqCst);
        if let Ok(mut p) = self.temp_path.lock()
            && let Some(path) = p.take()
        {
            let _ = std::fs::remove_file(path);
        }
    }
}

// ---- IThumbnailProvider ----

#[allow(non_snake_case)]
impl IThumbnailProvider_Impl for ThumbnailProvider_Impl {
    /// Produce the requested-size thumbnail. The decode/bitmap path is wrapped
    /// in `catch_unwind` so a panicking decoder can never unwind across the
    /// `extern "system"` COM boundary (which would abort the host process).
    fn GetThumbnail(
        &self,
        cx: u32,
        phbmp: *mut HBITMAP,
        pdwalpha: *mut WTS_ALPHATYPE,
    ) -> windows::core::Result<()> {
        if phbmp.is_null() || pdwalpha.is_null() {
            return Err(E_INVALIDARG.into());
        }

        let path = match self.temp_path.lock() {
            Ok(g) => g.clone(),
            Err(_) => None,
        };
        let Some(path) = path else {
            unsafe {
                *phbmp = HBITMAP::default();
                *pdwalpha = WTSAT_UNKNOWN;
            }
            return Err(E_FAIL.into());
        };

        let built =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build_thumbnail(&path, cx)));
        match built {
            Ok(Ok((hbmp, has_alpha))) => {
                unsafe {
                    *phbmp = hbmp;
                    *pdwalpha = if has_alpha { WTSAT_ARGB } else { WTSAT_RGB };
                }
                Ok(())
            }
            Ok(Err(_)) | Err(_) => {
                unsafe {
                    *phbmp = HBITMAP::default();
                    *pdwalpha = WTSAT_UNKNOWN;
                }
                Err(E_FAIL.into())
            }
        }
    }
}

// ---- IInitializeWithStream ----

#[allow(non_snake_case)]
impl IInitializeWithStream_Impl for ThumbnailProvider_Impl {
    /// Copy the stream to a unique temp file so the engine's path-based cover
    /// extraction (ZIP/RAR/7z all reopen the archive) can process it. This
    /// deliberately keeps Shell process isolation: we never hold a file path.
    fn Initialize(&self, pstream: Ref<IStream>, _grfmode: u32) -> windows::core::Result<()> {
        if pstream.is_null() {
            return Err(E_INVALIDARG.into());
        }
        let stream = pstream.as_ref().ok_or(E_INVALIDARG)?;
        let path = match read_stream_to_temp(stream) {
            Ok(p) => p,
            Err(_) => return Err(E_FAIL.into()),
        };
        *self.temp_path.lock().map_err(|_| E_FAIL)? = Some(path);
        Ok(())
    }
}

// ---- Class factory ----

/// A minimal `IClassFactory` that hands out [`ThumbnailProvider`] instances.
#[implement(IClassFactory)]
struct ThumbnailFactory;

#[allow(non_snake_case)]
impl IClassFactory_Impl for ThumbnailFactory_Impl {
    fn CreateInstance(
        &self,
        punkouter: Ref<windows::core::IUnknown>,
        riid: *const GUID,
        ppvobject: *mut *mut std::ffi::c_void,
    ) -> windows::core::Result<()> {
        if !punkouter.is_null() {
            return Err(CLASS_E_NOAGGREGATION.into());
        }
        if riid.is_null() || ppvobject.is_null() {
            return Err(E_INVALIDARG.into());
        }
        OBJECT_COUNT.fetch_add(1, Ordering::SeqCst);
        let object = ComObject::new(ThumbnailProvider::new());
        let guid = unsafe { &*riid };
        unsafe {
            if *guid == windows::core::IUnknown::IID {
                let p: windows::core::IUnknown = object.to_interface();
                *ppvobject = p.into_raw();
                return Ok(());
            }
            if *guid == <IThumbnailProvider as Interface>::IID {
                let p: IThumbnailProvider = object.to_interface();
                *ppvobject = p.into_raw();
                return Ok(());
            }
            if *guid == <IInitializeWithStream as Interface>::IID {
                let p: IInitializeWithStream = object.to_interface();
                *ppvobject = p.into_raw();
                return Ok(());
            }
            Err(E_NOINTERFACE.into())
        }
    }

    fn LockServer(&self, flock: BOOL) -> windows::core::Result<()> {
        if flock.as_bool() {
            LOCK_COUNT.fetch_add(1, Ordering::SeqCst);
        } else {
            LOCK_COUNT.fetch_sub(1, Ordering::SeqCst);
        }
        Ok(())
    }
}

// ---- DLL entry points ----

/// COM class factory entry point. Only serves `PROVIDER_CLSID`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DllGetClassObject(
    rclsid: *const GUID,
    riid: *const GUID,
    ppv: *mut *mut std::ffi::c_void,
) -> i32 {
    unsafe {
        if rclsid.is_null() || riid.is_null() || ppv.is_null() {
            return E_INVALIDARG.0;
        }
        if *rclsid != PROVIDER_CLSID {
            return CLASS_E_CLASSNOTAVAILABLE.0;
        }
        let object = ComObject::new(ThumbnailFactory);
        let factory: IClassFactory = object.to_interface();
        let guid = &*riid;
        // IClassFactory's vtable begins with IUnknown, so the same raw pointer
        // satisfies both IClassFactory and IUnknown requests.
        if *guid == <IClassFactory as Interface>::IID || *guid == windows::core::IUnknown::IID {
            *ppv = factory.into_raw();
            return S_OK.0;
        }
        E_NOINTERFACE.0
    }
}

/// Whether the DLL may be unloaded: only when no objects are alive and no
/// `LockServer` reference is held.
#[unsafe(no_mangle)]
pub extern "system" fn DllCanUnloadNow() -> i32 {
    if OBJECT_COUNT.load(Ordering::SeqCst) == 0 && LOCK_COUNT.load(Ordering::SeqCst) == 0 {
        S_OK.0
    } else {
        S_FALSE.0
    }
}

// ---- Helpers ----

/// The actual thumbnail work: extract the cover, decode it to fit `cx`, convert
/// to premultiplied BGRA, and wrap it in an owned `HBITMAP`. Returns whether
/// the image carries non-opaque alpha.
fn build_thumbnail(
    path: &std::path::Path,
    cx: u32,
) -> Result<(HBITMAP, bool), windows::core::Error> {
    let cover = cover_bytes(path).map_err(|_| E_FAIL)?;
    let target_h = thumbnail_target_h(&cover.bytes, cx);
    let mut resizer = Resizer::new();
    let page = decode_page(&cover.bytes, target_h, true, &mut resizer).map_err(|_| E_FAIL)?;
    let img = match page {
        DecodedPage::Still(i) => Some(i),
        DecodedPage::Animated(mut f) => f.drain(..).next().map(|(i, _)| i),
        DecodedPage::Layered(mut l) => l.drain(..).next(),
    };
    let img = img.ok_or(E_FAIL)?;

    let (bgra, has_alpha) = to_bgra(&img);
    let hbmp = create_dib_section(img.w, img.h, &bgra)?;
    Ok((hbmp, has_alpha))
}

/// Compute the decode target height so the **largest** dimension fits `cx`,
/// preserving aspect and never upscaling past the source. Thumbnails use LQ
/// decode, so a small (well below native) target is exactly what we want.
fn thumbnail_target_h(bytes: &[u8], cx: u32) -> u32 {
    let cx = cx.max(1);
    let (src_w, src_h, _) = meta::probe(bytes);
    if src_w == 0 || src_h == 0 {
        return cx;
    }
    let max_dim = src_w.max(src_h);
    let scale = (cx as f64 / max_dim as f64).min(1.0);
    ((src_h as f64 * scale).round() as u32).max(1)
}

/// Convert a decoded image to premultiplied BGRA, reporting whether any pixel is
/// not fully opaque (which drives the `WTS_ALPHATYPE` we return).
fn to_bgra(img: &DecodedImage) -> (Vec<u8>, bool) {
    let n = (img.w * img.h) as usize;
    let mut out = Vec::with_capacity(n * 4);
    if img.gray {
        for &g in &img.pixels {
            out.extend_from_slice(&[g, g, g, 255]);
        }
        return (out, false);
    }
    let mut has_alpha = false;
    for px in img.pixels.as_chunks::<4>().0 {
        let (r, g, b, a) = (px[0] as u32, px[1] as u32, px[2] as u32, px[3] as u32);
        if a < 255 {
            has_alpha = true;
        }
        // Premultiply in gamma space (matching the engine's own convention).
        let pr = ((r * a + 127) / 255) as u8;
        let pg = ((g * a + 127) / 255) as u8;
        let pb = ((b * a + 127) / 255) as u8;
        out.extend_from_slice(&[pb, pg, pr, a as u8]);
    }
    (out, has_alpha)
}

/// Allocate a 32-bpp top-down DIB section and copy `bgra` into it. The returned
/// `HBITMAP` is owned by the caller (Explorer), which releases it.
fn create_dib_section(w: u32, h: u32, bgra: &[u8]) -> Result<HBITMAP, windows::core::Error> {
    let mut bi = BITMAPINFO::default();
    bi.bmiHeader.biSize = size_of::<BITMAPINFOHEADER>() as u32;
    bi.bmiHeader.biWidth = w as i32;
    bi.bmiHeader.biHeight = -(h as i32); // top-down: rows in memory order
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB.0;
    let mut bits = null_mut();
    let hbmp = unsafe { CreateDIBSection(None, &bi, DIB_RGB_COLORS, &mut bits, None, 0)? };
    unsafe {
        if !bits.is_null() && !bgra.is_empty() {
            std::ptr::copy_nonoverlapping(bgra.as_ptr(), bits as *mut u8, bgra.len());
        }
    }
    Ok(hbmp)
}

/// Monotonic counter for temp file names. Independent of [`OBJECT_COUNT`] so the
/// names are unique even when the provider is created directly (as in tests)
/// rather than through the class factory.
static TEMP_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Read a stream into a unique temp file named with the archive's real extension
/// (the engine's `cover_bytes` dispatches on extension), returning its path. The
/// file is cleaned up by the provider's `Drop`.
fn read_stream_to_temp(pstream: &IStream) -> Result<PathBuf, windows::core::Error> {
    // Peek the first bytes to pick the archive extension. Explorer only invokes
    // us for .cbz/.cbr/.cb7, and each has unmistakable magic; a file with none is
    // left as .cbz so `cover_bytes` falls back to a typed error.
    let mut magic = [0u8; 8];
    unsafe {
        pstream.Seek(0, STREAM_SEEK_SET, None)?;
        let mut got = 0u32;
        let _ = pstream.Read(
            magic.as_mut_ptr() as *mut std::ffi::c_void,
            magic.len() as u32,
            Some(&mut got),
        );
        pstream.Seek(0, STREAM_SEEK_SET, None)?;
    }
    let ext = detect_archive_ext(&magic);

    let seq = TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut path = std::env::temp_dir();
    path.push(format!("yosh-{}-{seq}.{ext}", std::process::id()));

    let mut file = std::fs::File::create(&path).map_err(|_| E_FAIL)?;
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let mut read = 0u32;
        let hr = unsafe {
            pstream.Read(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                buf.len() as u32,
                Some(&mut read),
            )
        };
        if hr.is_err() {
            let _ = std::fs::remove_file(&path);
            return Err(E_FAIL.into());
        }
        if read == 0 {
            break;
        }
        if file.write_all(&buf[..read as usize]).is_err() {
            let _ = std::fs::remove_file(&path);
            return Err(E_FAIL.into());
        }
        if (read as usize) < buf.len() {
            break;
        }
    }
    Ok(path)
}

#[inline]
fn detect_archive_ext(prefix: &[u8]) -> &'static str {
    if prefix.starts_with(b"PK") {
        "cbz"
    } else if prefix.starts_with(b"Rar!\x1a\x07") {
        "cbr"
    } else if prefix.starts_with(&[0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C]) {
        "cb7"
    } else {
        "cbz"
    }
}

// `WTSAT_UNKNOWN` is the "no alpha info" return the interface expects on
// failure; keep it in scope for the failure paths above.
const WTSAT_UNKNOWN: WTS_ALPHATYPE = WTS_ALPHATYPE(0);

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use windows::Win32::Foundation::HGLOBAL;
    use windows::Win32::Graphics::Gdi::{BITMAP, GetObjectA, HGDIOBJ};
    use windows::Win32::System::Com::StructuredStorage::CreateStreamOnHGlobal;
    use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};

    /// Build a `.cbz` containing a single named image entry.
    fn write_cbz(tag: &str, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        use zip::write::SimpleFileOptions;
        let path =
            std::env::temp_dir().join(format!("yosh_thumb_{}_{tag}.cbz", std::process::id()));
        let f = std::fs::File::create(&path).unwrap();
        let mut w = zip::ZipWriter::new(f);
        w.start_file(name, SimpleFileOptions::default()).unwrap();
        w.write_all(bytes).unwrap();
        w.finish().unwrap();
        path
    }

    /// Encode a solid-colour `w × h` PNG.
    fn make_png(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let img =
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(w, h, image::Rgba(rgba)));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn provider_init_and_get_thumbnail_returns_scaled_bitmap() {
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }

        // 400×200 solid image → a 256×128 thumbnail (2:1 kept).
        let png = make_png(400, 200, [0, 128, 255, 255]);
        let cbz = write_cbz("cover", "cover.png", &png);

        let object = ComObject::new(ThumbnailProvider::new());
        let provider: IThumbnailProvider = object.to_interface();
        let init: IInitializeWithStream = object.to_interface();

        // Treat the CBZ bytes as an IStream (what Explorer hands us).
        let bytes = std::fs::read(&cbz).unwrap();
        let stream = unsafe { CreateStreamOnHGlobal(HGLOBAL::default(), true) }.unwrap();
        unsafe {
            let mut written = 0u32;
            stream
                .Write(
                    bytes.as_ptr() as *const c_void,
                    bytes.len() as u32,
                    Some(&mut written),
                )
                .ok()
                .unwrap();
            stream.Seek(0, STREAM_SEEK_SET, None).unwrap();
        }

        // Initialize with the stream, then request a 256px thumbnail.
        unsafe { init.Initialize(&stream, 0) }.unwrap();

        let mut hbmp = HBITMAP::default();
        let mut alpha = WTSAT_UNKNOWN;
        unsafe { provider.GetThumbnail(256, &mut hbmp, &mut alpha) }.unwrap();

        assert!(!hbmp.0.is_null(), "provider returned a null bitmap");
        assert_eq!(alpha, WTSAT_RGB, "opaque cover should be WTSAT_RGB");

        let mut bm: BITMAP = unsafe { std::mem::zeroed() };
        let got = unsafe {
            GetObjectA(
                HGDIOBJ(hbmp.0),
                std::mem::size_of::<BITMAP>() as i32,
                Some(&mut bm as *mut _ as *mut c_void),
            )
        };
        assert!(got != 0, "GetObjectA failed");
        assert_eq!(bm.bmWidth, 256);
        assert_eq!(bm.bmHeight, 128);

        // Clean up: drop the COM interfaces (releasing the object → temp file
        // removed) and delete the fixture.
        drop(provider);
        drop(init);
        drop(object);
        let _ = std::fs::remove_file(&cbz);
        unsafe {
            CoUninitialize();
        }
    }

    #[test]
    fn provider_uses_cover_priority_and_reports_failure_for_garbage() {
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }

        // A CBZ whose first image is not `cover`, plus a real `cover.png`.
        let png = make_png(40, 20, [255, 0, 0, 255]);
        let cbz = write_cbz("priority", "cover.png", &png);

        let object = ComObject::new(ThumbnailProvider::new());
        let provider: IThumbnailProvider = object.to_interface();
        let init: IInitializeWithStream = object.to_interface();

        let bytes = std::fs::read(&cbz).unwrap();
        let stream = unsafe { CreateStreamOnHGlobal(HGLOBAL::default(), true) }.unwrap();
        unsafe {
            let mut written = 0u32;
            stream
                .Write(
                    bytes.as_ptr() as *const c_void,
                    bytes.len() as u32,
                    Some(&mut written),
                )
                .ok()
                .unwrap();
            stream.Seek(0, STREAM_SEEK_SET, None).unwrap();
        }
        unsafe { init.Initialize(&stream, 0) }.unwrap();

        let mut hbmp = HBITMAP::default();
        let mut alpha = WTSAT_UNKNOWN;
        unsafe { provider.GetThumbnail(256, &mut hbmp, &mut alpha) }.unwrap();
        assert!(!hbmp.0.is_null());
        let mut bm: BITMAP = unsafe { std::mem::zeroed() };
        let _ = unsafe {
            GetObjectA(
                HGDIOBJ(hbmp.0),
                std::mem::size_of::<BITMAP>() as i32,
                Some(&mut bm as *mut _ as *mut c_void),
            )
        };
        // 40×20 never upscales past source, so the thumbnail is the native size.
        assert_eq!(bm.bmWidth, 40);
        assert_eq!(bm.bmHeight, 20);

        drop(provider);
        drop(init);
        drop(object);
        let _ = std::fs::remove_file(&cbz);

        // A corrupt archive must fail (E_FAIL) without crashing, leaving a null
        // bitmap + unknown alpha for Explorer to fall back to its icon.
        let corrupt = write_cbz("corrupt", "cover.png", b"not-an-image");
        let object2 = ComObject::new(ThumbnailProvider::new());
        let provider2: IThumbnailProvider = object2.to_interface();
        let init2: IInitializeWithStream = object2.to_interface();
        let bytes2 = std::fs::read(&corrupt).unwrap();
        let stream2 = unsafe { CreateStreamOnHGlobal(HGLOBAL::default(), true) }.unwrap();
        unsafe {
            let mut written = 0u32;
            stream2
                .Write(
                    bytes2.as_ptr() as *const c_void,
                    bytes2.len() as u32,
                    Some(&mut written),
                )
                .ok()
                .unwrap();
            stream2.Seek(0, STREAM_SEEK_SET, None).unwrap();
        }
        unsafe { init2.Initialize(&stream2, 0) }.unwrap();
        let mut hbmp2 = HBITMAP::default();
        let mut alpha2 = WTSAT_UNKNOWN;
        let hr = unsafe { provider2.GetThumbnail(256, &mut hbmp2, &mut alpha2) };
        assert!(hr.is_err());
        assert!(hbmp2.0.is_null(), "failure must leave a null bitmap");

        drop(provider2);
        drop(init2);
        drop(object2);
        let _ = std::fs::remove_file(&corrupt);
        unsafe {
            CoUninitialize();
        }
    }
}

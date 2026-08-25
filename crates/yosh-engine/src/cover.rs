//! Archive cover extraction, shared by the library view and the Windows
//! Explorer thumbnail provider.
//!
//! This is a **focused, one-entry** API, deliberately separate from the full
//! [`crate::source::PageSource`] backends. Reading a cover must not spin up a
//! whole `DecodePool`/`Reader` nor extract every page: we list image entries,
//! pick the cover (an entry whose filename stem is exactly `cover`,
//! case-insensitive, else the first naturally-sorted image), and read **only
//! that one entry's bytes** — skipping the rest of the archive.
//!
//! Only comic archives are served: `.cbz`, `.cbr`, `.cb7`. Plain `.zip`,
//! `.rar`, `.7z`, folders and standalone images are rejected so Explorer's
//! thumbnail provider leaves them alone.
//!
//! Every failure is a typed [`CoverError`] so callers (the library grid, the
//! thumbnail provider's `GetThumbnail`) can map it to the right fallback
//! instead of probing by string matching. Exploration runs this code while
//! browsing files, so it must never panic — only return errors.

use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::source::{PageSource, ZipSource, is_image_name};

/// Ceiling on an archive's on-disk size. Explorer asks us to make a thumbnail
/// for any `.cbz/.cbr/.cb7`, so a hostile or pathological file must be rejected
/// cheaply — `metadata().len()` is a stat, not a read. Comics never approach
/// this; it exists to bail out of an absurd claim without touching the archive.
pub const MAX_ARCHIVE_BYTES: u64 = 8 << 30; // 8 GiB

/// Ceiling on one chosen entry's decoded bytes. The existing ZIP prealloc caps
/// at 256 MiB; mirror that here so a corrupt header that claims a huge size
/// can't make us allocate it before reading actual data.
pub const MAX_ENTRY_BYTES: usize = 256 << 20; // 256 MiB

/// Why cover extraction failed, typed so callers can map to an HRESULT or a UI
/// fallback rather than string-matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverError {
    /// The path isn't a supported comic archive (extension isn't cbz/cbr/cb7).
    Unsupported,
    /// I/O failure opening or reading the archive (missing file, permission, …).
    Io(String),
    /// The archive has no image entries, so there is no cover to show.
    Empty,
    /// The archive (or its chosen entry) is encrypted / password-protected.
    Encrypted,
    /// The archive is corrupt or truncated.
    Corrupt,
    /// The archive or chosen entry exceeds a size bound.
    TooLarge,
}

impl std::fmt::Display for CoverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => write!(f, "unsupported comic archive"),
            Self::Io(e) => write!(f, "archive I/O error: {e}"),
            Self::Empty => write!(f, "archive has no images"),
            Self::Encrypted => write!(f, "archive is encrypted"),
            Self::Corrupt => write!(f, "archive is corrupt"),
            Self::TooLarge => write!(f, "archive or entry too large"),
        }
    }
}

impl std::error::Error for CoverError {}

impl From<io::Error> for CoverError {
    fn from(e: io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// A selected cover: the archive entry's name and its encoded image bytes.
/// The bytes are *encoded* (PNG/JPEG/…) — decoding to pixels is the caller's
/// job, so the library and the thumbnail provider share this one extraction
/// path and decode once to their own target size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cover {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// The comic archive kinds we serve thumbnails for. Deliberately omits plain
/// `.zip`/`.rar`/`.7z` — Explorer should not get a custom thumbnail for those.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArchiveKind {
    Cbz,
    Cbr,
    Cb7,
}

fn archive_kind(path: &Path) -> Option<ArchiveKind> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("cbz") => Some(ArchiveKind::Cbz),
        Some("cbr") => Some(ArchiveKind::Cbr),
        Some("cb7") => Some(ArchiveKind::Cb7),
        _ => None,
    }
}

/// Is an archive entry name the *exact* cover? The filename stem must be
/// exactly `cover`, case-insensitive: `Cover.png`, `cover.jpg`, `COVER.webp`.
/// `cover-art.png` and `my-cover.jpg` are **not** covers.
fn is_cover_name(name: &str) -> bool {
    Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("cover"))
}

/// Index of the cover entry within a natural-sorted name list: the first entry
/// whose filename stem is exactly `cover` (case-insensitive), else the first
/// (sorted) entry. Returns `None` only for an empty list.
pub fn select_cover_index(names: &[String]) -> Option<usize> {
    names
        .iter()
        .position(|n| is_cover_name(n))
        .or_else(|| (!names.is_empty()).then_some(0))
}

/// Cheap pre-checks shared by every entry point: the path must be a file under
/// the size bound. Called after `archive_kind`, so extension is already known.
fn check_input(path: &Path) -> Result<(), CoverError> {
    let meta = std::fs::metadata(path)?;
    if !meta.is_file() {
        return Err(CoverError::Unsupported);
    }
    if meta.len() > MAX_ARCHIVE_BYTES {
        return Err(CoverError::TooLarge);
    }
    Ok(())
}

/// List a comic archive's image entry names, natural-sorted by name. 
///
/// - CBZ names are legacy-codepage decoded exactly as [`ZipSource`] does (so a
///   Shift-JIS/GBK archive lists real characters, not CP437 mojibake).
/// - CBR / CB7 names are the archive's stored entry paths.
///
/// Returns [`CoverError::Empty`] if the archive holds no images. This is the
/// first step of the shared `cover` pipeline; the DLL and library both use it.
pub fn list_image_entries(path: &Path) -> Result<Vec<String>, CoverError> {
    check_input(path)?;
    match archive_kind(path) {
        Some(ArchiveKind::Cbz) => {
            let src = ZipSource::new(path).map_err(|_| CoverError::Corrupt)?;
            let names: Vec<String> = (0..src.len()).map(|i| src.name(i).to_string()).collect();
            if names.is_empty() {
                return Err(CoverError::Empty);
            }
            Ok(names)
        }
        Some(ArchiveKind::Cbr) => list_cbr_entries_gate(path),
        Some(ArchiveKind::Cb7) => list_cb7_entries(path),
        None => Err(CoverError::Unsupported),
    }
}

/// Extract the cover bytes for a comic archive (`.cbz`/`.cbr`/`.cb7`).
///
/// Selects the cover via [`select_cover_index`] then reads **only that entry**,
/// stopping the archive walk as soon as it lands — never extracting the whole
/// archive. Passes through the typed [`CoverError`] on any failure.
pub fn cover_bytes(path: &Path) -> Result<Cover, CoverError> {
    check_input(path)?;
    match archive_kind(path) {
        Some(ArchiveKind::Cbz) => read_cbz_cover(path),
        Some(ArchiveKind::Cbr) => read_cbr_cover_gate(path),
        Some(ArchiveKind::Cb7) => read_cb7_cover(path),
        None => Err(CoverError::Unsupported),
    }
}

/// CBZ/CBZ cover: random-access, so we can open the source, pick the cover by
/// name, and read just that entry.
fn read_cbz_cover(path: &Path) -> Result<Cover, CoverError> {
    let src = ZipSource::new(path).map_err(|_| CoverError::Corrupt)?;
    let names: Vec<String> = (0..src.len()).map(|i| src.name(i).to_string()).collect();
    let idx = select_cover_index(&names).ok_or(CoverError::Empty)?;
    let name = names[idx].clone();
    let bytes = src.read_page(idx).map_err(|_| CoverError::Corrupt)?;
    let bytes = Arc::try_unwrap(bytes).unwrap_or_else(|a| (*a).clone());
    if bytes.len() > MAX_ENTRY_BYTES {
        return Err(CoverError::TooLarge);
    }
    Ok(Cover { name, bytes })
}

/// CB7 cover: the chosen entry lives in one packed stream. Decode only the
/// block that contains it, consuming the entries before it so the block stream
/// stays aligned, then stop as soon as we have the cover.
fn read_cb7_cover(path: &Path) -> Result<Cover, CoverError> {
    use sevenz_rust2::{Archive, BlockDecoder, Password};

    let names = list_cb7_entries(path)?;
    let idx = select_cover_index(&names).ok_or(CoverError::Empty)?;
    let cover_name = names[idx].clone();

    let archive = Archive::open(path).map_err(|_| CoverError::Corrupt)?;
    let file_idx = archive
        .files
        .iter()
        .position(|f| f.name() == cover_name)
        .ok_or(CoverError::Corrupt)?;
    let Some(block) = archive.stream_map.file_block_index[file_idx] else {
        // A zero-length file outside any packed stream: nothing to decode.
        return Err(CoverError::Corrupt);
    };
    let mut file = std::fs::File::open(path)?;
    let password = Password::empty();
    let mut fi = archive.stream_map.block_first_file_index[block];
    let dec = BlockDecoder::new(1, block, &archive, &password, &mut file);
    let mut wanted: Option<Result<Vec<u8>, CoverError>> = None;
    let walked = dec.for_each_entries(&mut |entry, rd| {
        let this = fi;
        fi += 1;
        if this == file_idx {
            if entry.size as usize > MAX_ENTRY_BYTES {
                wanted = Some(Err(CoverError::TooLarge));
                return Ok(false);
            }
            let mut buf = Vec::with_capacity(entry.size as usize);
            io::Read::read_to_end(rd, &mut buf)?;
            if buf.len() > MAX_ENTRY_BYTES {
                wanted = Some(Err(CoverError::TooLarge));
            } else {
                wanted = Some(Ok(buf));
            }
            return Ok(false);
        }
        io::copy(rd, &mut io::sink())?;
        Ok(true)
    });
    walked.map_err(|_| CoverError::Corrupt)?;
    match wanted {
        Some(Ok(bytes)) => Ok(Cover {
            name: cover_name,
            bytes,
        }),
        Some(Err(e)) => Err(e),
        None => Err(CoverError::Corrupt),
    }
}

fn list_cb7_entries(path: &Path) -> Result<Vec<String>, CoverError> {
    let archive = sevenz_rust2::Archive::open(path).map_err(|_| CoverError::Corrupt)?;
    let mut names: Vec<String> = archive
        .files
        .iter()
        .filter(|e| !e.is_directory() && is_image_name(e.name()))
        .map(|e| e.name().to_string())
        .collect();
    if names.is_empty() {
        return Err(CoverError::Empty);
    }
    names.sort_by(|a, b| natord::compare(a, b));
    Ok(names)
}

// RAR has no random access (except `skip()` on a non-solid archive), so we list
// the image names first, then run a single front-to-back processing pass that
// **stops** as soon as the chosen cover entry is read. Non-image / non-cover
// entries are skipped; on a non-solid archive a skip is a pure seek, so finding
// a `cover.jpg` near the end costs a seek, not a decompression.

#[cfg(feature = "rar")]
fn list_cbr_entries_gate(path: &Path) -> Result<Vec<String>, CoverError> {
    list_cbr_entries(path)
}
#[cfg(not(feature = "rar"))]
fn list_cbr_entries_gate(_path: &Path) -> Result<Vec<String>, CoverError> {
    Err(CoverError::Unsupported)
}

#[cfg(feature = "rar")]
fn read_cbr_cover_gate(path: &Path) -> Result<Cover, CoverError> {
    read_cbr_cover(path)
}
#[cfg(not(feature = "rar"))]
fn read_cbr_cover_gate(_path: &Path) -> Result<Cover, CoverError> {
    Err(CoverError::Unsupported)
}

#[cfg(feature = "rar")]
fn list_cbr_entries(path: &Path) -> Result<Vec<String>, CoverError> {
    use unrar::Archive;

    let mut names: Vec<String> = Vec::new();
    let listing = Archive::new(path)
        .open_for_listing()
        .map_err(|_| CoverError::Corrupt)?;
    // `listing` is an iterator over entries; it is consumed here.
    for entry in listing {
        let entry = entry.map_err(|_| CoverError::Corrupt)?;
        let name = entry.filename.to_string_lossy().into_owned();
        if is_image_name(&name) {
            names.push(name);
        }
    }
    if names.is_empty() {
        return Err(CoverError::Empty);
    }
    names.sort_by(|a, b| natord::compare(a, b));
    Ok(names)
}

#[cfg(feature = "rar")]
fn read_cbr_cover(path: &Path) -> Result<Cover, CoverError> {
    use unrar::Archive;

    let names = list_cbr_entries(path)?;
    let idx = select_cover_index(&names).ok_or(CoverError::Empty)?;
    let cover_name = names[idx].clone();

    let mut cursor = Archive::new(path)
        .open_for_processing()
        .map_err(|_| CoverError::Corrupt)?;
    loop {
        let header = match cursor.read_header() {
            Ok(Some(h)) => h,
            Ok(None) => return Err(CoverError::Corrupt),
            Err(_) => return Err(CoverError::Corrupt),
        };
        let name = header.entry().filename.to_string_lossy().into_owned();
        if name == cover_name {
            let (bytes, _next) = header.read().map_err(|e| {
                let msg = e.to_string().to_ascii_lowercase();
                if msg.contains("password") || msg.contains("encrypted") {
                    CoverError::Encrypted
                } else {
                    CoverError::Corrupt
                }
            })?;
            if bytes.len() > MAX_ENTRY_BYTES {
                return Err(CoverError::TooLarge);
            }
            return Ok(Cover { name, bytes });
        }
        // Not the cover: skip past it. On a non-solid archive this is a pure
        // fseek; on a solid one it decompresses-but-discards, which is the
        // minimum needed to reach an entry later in the same stream.
        cursor = header.skip().map_err(|_| CoverError::Corrupt)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::path::PathBuf;

    fn tmp(tag: &str, ext: &str) -> PathBuf {
        std::env::temp_dir().join(format!("yosh_cover_{}_{tag}.{ext}", std::process::id()))
    }

    /// Build a `.cbz` from `(name, bytes)` entries plus optional directories.
    fn write_zip(tag: &str, files: &[(&str, &[u8])]) -> PathBuf {
        use zip::write::SimpleFileOptions;
        let path = tmp(tag, "cbz");
        let f = std::fs::File::create(&path).unwrap();
        let mut w = zip::ZipWriter::new(f);
        let opts = SimpleFileOptions::default();
        for (name, bytes) in files {
            w.start_file(*name, opts).unwrap();
            w.write_all(bytes).unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// Build a non-solid `.cb7` from `(name, bytes)` entries.
    fn write_7z(tag: &str, files: &[(&str, &[u8])]) -> PathBuf {
        use sevenz_rust2::ArchiveEntry;
        use std::io::Cursor;
        let path = tmp(tag, "cb7");
        let mut w = sevenz_rust2::ArchiveWriter::create(&path).unwrap();
        for (name, bytes) in files {
            w.push_archive_entry(
                ArchiveEntry::new_file(name),
                Some(Cursor::new(bytes.to_vec())),
            )
            .unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// Build a minimal stored (no compression) RAR4 `.cbr` from one or more
    /// image files. Header CRC is the low 16 bits of CRC-32 over the header
    /// bytes from the type byte to the end of the filename — validated against
    /// the unrar crate's own fixtures.
    fn write_rar4(tag: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let path = tmp(tag, "cbr");
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&[0x52, 0x61, 0x72, 0x21, 0x1A, 0x07, 0x00]); // marker
        // Main archive header (type 0x73).
        let main_body: Vec<u8> = vec![0x73, 0x00, 0x00, 13, 0, 0, 0, 0, 0, 0, 0];
        let main_crc = crc32fast::hash(&main_body) as u16;
        out.extend_from_slice(&main_crc.to_le_bytes());
        out.extend_from_slice(&main_body);
        // One file header per entry (type 0x74), method 0x30 = store.
        for (name, data) in files {
            let nb = name.as_bytes();
            let nsz = nb.len() as u16;
            let file_crc = crc32fast::hash(data) as u32;
            let hsize: u16 = 32 + nsz;
            let mut body: Vec<u8> = Vec::with_capacity(hsize as usize);
            body.push(0x74); // HEAD_TYPE
            body.extend_from_slice(&0u16.to_le_bytes()); // HEAD_FLAGS
            body.extend_from_slice(&hsize.to_le_bytes()); // HEAD_SIZE
            body.extend_from_slice(&(data.len() as u32).to_le_bytes()); // PACK_SIZE
            body.extend_from_slice(&(data.len() as u32).to_le_bytes()); // UNP_SIZE
            body.push(2); // HOST_OS = Windows
            body.extend_from_slice(&file_crc.to_le_bytes()); // FILE_CRC
            body.extend_from_slice(&0u32.to_le_bytes()); // FTIME
            body.push(20); // UNP_VER
            body.push(0x30); // METHOD = store
            body.extend_from_slice(&nsz.to_le_bytes()); // NAME_SIZE
            body.extend_from_slice(&0x20u32.to_le_bytes()); // ATTR = archive
            body.extend_from_slice(nb); // FILE_NAME
            // `body` omits the 2-byte HEAD_CRC that precedes it; HEAD_SIZE counts it.
            assert_eq!(body.len() as u16 + 2, hsize);
            let crc = crc32fast::hash(&body) as u16;
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&body);
            out.extend_from_slice(data);
        }
        // End-of-archive block (type 0x7b): HEAD_TYPE + HEAD_FLAGS + HEAD_SIZE.
        let end_body: Vec<u8> = vec![0x7b, 0x00, 0x40, 7, 0];
        let end_crc = crc32fast::hash(&end_body) as u16;
        out.extend_from_slice(&end_crc.to_le_bytes());
        out.extend_from_slice(&end_body);
        std::fs::write(&path, &out).unwrap();
        path
    }

    #[test]
    fn exact_cover_beats_first_page() {
        // 01/02 sorted before cover, but the exact `cover` stem wins.
        let names: Vec<String> = ["01.jpg", "02.jpg", "cover.jpg"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(select_cover_index(&names), Some(2));
    }

    #[test]
    fn cover_matching_is_case_insensitive() {
        for name in ["cover.jpg", "Cover.png", "COVER.webp"] {
            let names = vec![name.to_string(), "01.jpg".to_string()];
            assert_eq!(select_cover_index(&names), Some(0), "{name}");
        }
    }

    #[test]
    fn near_cover_names_are_not_covers() {
        // `cover-art.png`, `my-cover.jpg`, `cover2.jpg`, `a_cover.png` must NOT
        // be treated as an exact `cover` stem.
        for name in [
            "cover-art.png",
            "my-cover.jpg",
            "cover2.jpg",
            "a_cover.png",
            "the-cover.png",
            "covers.jpg",
        ] {
            assert!(!is_cover_name(name), "{name} should not be a cover");
        }
        // Natural-sorted: "01.jpg" (0) sorts before "cover-art.png" (c), so the
        // first image wins because the near-cover name doesn't match the stem.
        assert_eq!(
            select_cover_index(&["01.jpg".to_string(), "cover-art.png".to_string()]),
            Some(0)
        );
    }

    #[test]
    fn nested_path_cover_matches() {
        let names: Vec<String> = vec![
            "pages/01.jpg".to_string(),
            "pages/Cover.png".to_string(),
        ];
        assert_eq!(select_cover_index(&names), Some(1));
    }

    #[test]
    fn natural_order_fallback() {
        // No exact cover → the first (naturally-sorted) image. The list is
        // already sorted: "1.png" < "2.png" < "10.png".
        let names: Vec<String> = ["1.png", "2.png", "10.png"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(select_cover_index(&names), Some(0)); // "1.png"
    }

    #[test]
    fn empty_list_has_no_cover() {
        assert_eq!(select_cover_index(&[]), None);
    }

    #[test]
    fn cover_survives_path_separators_and_mixed_case_ext() {
        // A stem of "cover" with any image extension, under a directory.
        let names: Vec<String> = vec![
            "ch1/page-001.PNG".to_string(),
            "ch1/Cover.PNG".to_string(),
            "ch1/cover.JpG".to_string(),
        ];
        // First exact-cover match wins.
        assert_eq!(select_cover_index(&names), Some(1));
    }

    #[test]
    fn cbz_lists_image_entries_naturally_sorted() {
        let path = write_zip(
            "cbz_list",
            &[("10.png", b"TEN"), ("2.png", b"TWO"), ("1.png", b"ONE")],
        );
        let names = list_image_entries(&path).unwrap();
        assert_eq!(names, vec!["1.png", "2.png", "10.png"]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cbz_cover_priority_over_first_page() {
        let path = write_zip(
            "cbz_cover",
            &[("01.jpg", b"FIRST"), ("cover.jpg", b"COVER")],
        );
        let cover = cover_bytes(&path).unwrap();
        assert_eq!(cover.name, "cover.jpg");
        assert_eq!(cover.bytes, b"COVER");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cbz_cover_falls_back_to_first_image() {
        let path = write_zip(
            "cbz_fallback",
            &[("02.jpg", b"TWO"), ("01.jpg", b"ONE")],
        );
        let cover = cover_bytes(&path).unwrap();
        assert_eq!(cover.name, "01.jpg"); // natural first
        assert_eq!(cover.bytes, b"ONE");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cbz_archive_with_no_images_errors() {
        // A valid zip holding only non-images.
        let path = write_zip("cbz_noimg", &[("notes.txt", b"hello")]);
        assert_eq!(cover_bytes(&path), Err(CoverError::Empty));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cb7_cover_extracts_just_the_selected_entry() {
        let path = write_7z(
            "cb7_cover",
            &[("01.jpg", b"PAGE_ONE"), ("cover.jpg", b"THE_COVER")],
        );
        let names = list_image_entries(&path).unwrap();
        assert_eq!(names, vec!["01.jpg", "cover.jpg"]);
        let cover = cover_bytes(&path).unwrap();
        assert_eq!(cover.name, "cover.jpg");
        assert_eq!(cover.bytes, b"THE_COVER");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cb7_cover_falls_back_to_first_image() {
        let path = write_7z(
            "cb7_fallback",
            &[("02.jpg", b"TWO"), ("01.jpg", b"ONE")],
        );
        let cover = cover_bytes(&path).unwrap();
        assert_eq!(cover.name, "01.jpg");
        assert_eq!(cover.bytes, b"ONE");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cb7_empty_archive_errors() {
        let path = write_7z("cb7_noimg", &[("readme.txt", b"info")]);
        assert_eq!(cover_bytes(&path), Err(CoverError::Empty));
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(feature = "rar")]
    #[test]
    fn cbr_cover_extracts_selected_entry() {
        let path = write_rar4(
            "cbr_cover",
            &[("01.jpg", b"PAGE_ONE"), ("cover.jpg", b"THE_COVER")],
        );
        let names = list_image_entries(&path).unwrap();
        assert_eq!(names, vec!["01.jpg", "cover.jpg"]);
        let cover = cover_bytes(&path).unwrap();
        assert_eq!(cover.name, "cover.jpg");
        assert_eq!(cover.bytes, b"THE_COVER");
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(feature = "rar")]
    #[test]
    fn cbr_cover_falls_back_to_first_image() {
        let path = write_rar4(
            "cbr_fallback",
            &[("02.jpg", b"TWO"), ("01.jpg", b"ONE")],
        );
        let cover = cover_bytes(&path).unwrap();
        assert_eq!(cover.name, "01.jpg");
        assert_eq!(cover.bytes, b"ONE");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn plain_archives_and_images_are_rejected() {
        // `.zip` is explicitly not served to Explorer.
        let p = write_zip("zip_ext", &[("cover.jpg", b"X")]);
        let renamed = p.with_extension("zip");
        std::fs::rename(&p, &renamed).unwrap();
        assert_eq!(cover_bytes(&renamed), Err(CoverError::Unsupported));
        let _ = std::fs::remove_file(&renamed);

        // A standalone image path is unsupported.
        let img = tmp("standalone", "jpg");
        std::fs::write(&img, b"\xff\xd8\xff").unwrap();
        assert_eq!(cover_bytes(&img), Err(CoverError::Unsupported));
        let _ = std::fs::remove_file(&img);

        // A missing file reports an I/O error, not a panic.
        assert!(matches!(
            cover_bytes(Path::new("definitely-not-here.cbz")),
            Err(CoverError::Io(_))
        ));
    }
}

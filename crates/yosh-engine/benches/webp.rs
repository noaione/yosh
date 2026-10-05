//! Head-to-head WebP decode benchmark: [`wpd`] vs the `image` crate's pure-Rust
//! WebP backend ([`image_webp`]).
//!
//! Both decoders are driven by the *same* harness over the *same* files and
//! produce the same natural output (RGB when the source has no alpha, RGBA when
//! it does). Two groups are run:
//!
//! - `decode/native` — like-for-like: each decoder at its natural output format.
//! - `decode/rgba`   — yosh's actual pipeline format (always RGBA8). image-webp
//!   has no RGBA output for an RGB source, so this group includes the RGB→RGBA
//!   expansion the app would have to do itself — the extra cost of using
//!   image-webp behind yosh's RGBA-only decode contract.
//!
//! Throughput is the compressed-input rate (MiB/s); criterion's reported `time`
//! is per whole batch, so per-image ms = time ÷ file count (printed below).
//!
//! ```sh
//! cargo bench -p yosh-engine --bench webp
//! YOSH_WEBP_BENCH_LIMIT=0 cargo bench -p yosh-engine --bench webp   # all files
//! YOSH_WEBP_BENCH_LIMIT=8 cargo bench -p yosh-engine --bench webp   # quick
//! ```
//!
//! Environment:
//! - `YOSH_WEBP_BENCH_DIR`    corpus directory (default `<workspace>/target/bench-webp`)
//! - `YOSH_WEBP_BENCH_LIMIT`  max files to decode (default 24; `0` = all)
//! - `YOSH_WEBP_BENCH_VERIFY` set to `0` to skip the one-shot pixel-equality check

use std::hint::black_box;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use wpd::api::Decoder as WpdDecoder;
use wpd::image::Format as WpdFormat;

/// Files decoded per iteration by default. Decoding a 12 Mpx page is ~10–30 ms,
/// so 24 keeps one batch iteration in the hundreds-of-ms range where criterion's
/// statistics stay stable without the full 375 MiB corpus per iteration.
const DEFAULT_LIMIT: usize = 24;

/// One corpus file plus the metadata both decoders need (probed once, outside
/// the timed loop, so neither pays for it inside the benchmark).
struct BenchFile {
    name: String,
    bytes: Vec<u8>,
    has_alpha: bool,
    pixels: u64,
}

fn bench_dir() -> PathBuf {
    match std::env::var_os("YOSH_WEBP_BENCH_DIR") {
        Some(dir) => PathBuf::from(dir),
        // This bench lives in `crates/yosh-engine`; the workspace root is two up.
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/bench-webp"),
    }
}

fn limit() -> usize {
    std::env::var("YOSH_WEBP_BENCH_LIMIT")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_LIMIT)
}

fn verify_enabled() -> bool {
    !matches!(
        std::env::var("YOSH_WEBP_BENCH_VERIFY").as_deref(),
        Ok("0") | Ok("false")
    )
}

/// Load up to `limit` `*.webp` files (0 = all), name-sorted for reproducibility.
fn load_files() -> Vec<BenchFile> {
    let dir = bench_dir();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("webp"))
        })
        .collect();
    paths.sort();
    let limit = limit();
    if limit > 0 {
        paths.truncate(limit);
    }
    assert!(!paths.is_empty(), "no .webp files in {}", dir.display());

    let files: Vec<BenchFile> = paths
        .into_iter()
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
            let info = wpd::api::info(&bytes).unwrap_or_else(|e| panic!("wpd info {name}: {e}"));
            BenchFile {
                name,
                has_alpha: info.has_alpha,
                pixels: u64::from(info.width.max(0) as u32) * u64::from(info.height.max(0) as u32),
                bytes,
            }
        })
        .collect();

    let total_bytes: u64 = files.iter().map(|f| f.bytes.len() as u64).sum();
    let total_px: u64 = files.iter().map(|f| f.pixels).sum();
    eprintln!(
        "webp bench corpus: {} files, {:.1} MiB, {:.1} Mpx  (per-image ms = time / {})",
        files.len(),
        total_bytes as f64 / (1024.0 * 1024.0),
        total_px as f64 / 1e6,
        files.len(),
    );
    files
}

/// Reusable decode buffers so the timed loop never allocates on the hot path.
#[derive(Default)]
struct Decoders {
    out: Vec<u8>,
    scratch: Vec<u8>,
}

impl Decoders {
    fn new() -> Self {
        Self::default()
    }

    /// `wpd` → the source's natural format (RGB, or RGBA when it has alpha).
    fn wpd(&mut self, f: &BenchFile) -> usize {
        let format = if f.has_alpha {
            WpdFormat::Rgba
        } else {
            WpdFormat::Rgb
        };
        let mut d = WpdDecoder::new();
        d.set_format(format).expect("wpd set_format");
        d.set_options(wpd::options::Options::default())
            .expect("wpd set_options");
        d.open(&f.bytes).expect("wpd open");

        self.out.clear();
        if let Some(pic) = d.next_frame().expect("wpd next_frame") {
            let bpp = if f.has_alpha { 4 } else { 3 };
            self.out
                .reserve(pic.width() as usize * pic.height() as usize * bpp);
            for y in 0..pic.rows(0) {
                self.out.extend_from_slice(pic.row(0, y));
            }
        }
        black_box(&self.out);
        self.out.len()
    }

    /// `wpd` → RGBA8 always (yosh's pipeline format).
    fn wpd_rgba(&mut self, f: &BenchFile) -> usize {
        let mut d = WpdDecoder::new();
        d.set_format(WpdFormat::Rgba).expect("wpd set_format");
        d.set_options(wpd::options::Options::default())
            .expect("wpd set_options");
        d.open(&f.bytes).expect("wpd open");

        self.out.clear();
        if let Some(pic) = d.next_frame().expect("wpd next_frame") {
            self.out
                .reserve(pic.width() as usize * pic.height() as usize * 4);
            for y in 0..pic.rows(0) {
                self.out.extend_from_slice(pic.row(0, y));
            }
        }
        black_box(&self.out);
        self.out.len()
    }

    /// `image-webp` → its natural format (RGB, or RGBA when it has alpha).
    fn image_webp(&mut self, f: &BenchFile) -> usize {
        let mut d = image_webp::WebPDecoder::new(Cursor::new(&f.bytes)).expect("image-webp new");
        let size = d.output_buffer_size().expect("image-webp output size");
        self.out.clear();
        self.out.resize(size, 0);
        d.read_image(&mut self.out).expect("image-webp read_image");
        black_box(&self.out);
        self.out.len()
    }

    /// `image-webp` → RGBA8, expanding an RGB source (the extra pass yosh would
    /// have to add itself, since its pipeline is RGBA-only).
    fn image_webp_rgba(&mut self, f: &BenchFile) -> usize {
        let mut d = image_webp::WebPDecoder::new(Cursor::new(&f.bytes)).expect("image-webp new");
        let (w, h) = d.dimensions();
        let n = w as usize * h as usize;

        self.out.clear();
        if d.has_alpha() {
            self.out.resize(n * 4, 0);
            d.read_image(&mut self.out).expect("image-webp read_image");
        } else {
            self.scratch.clear();
            self.scratch.resize(n * 3, 0);
            d.read_image(&mut self.scratch)
                .expect("image-webp read_image");
            self.out.reserve(n * 4);
            for px in self.scratch.chunks_exact(3) {
                self.out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        black_box(&self.out);
        self.out.len()
    }
}

type DecodeFn = fn(&mut Decoders, &BenchFile) -> usize;

fn bench_batch(
    c: &mut Criterion,
    group_name: &str,
    files: &[BenchFile],
    decoders: &[(&str, DecodeFn)],
) {
    let total_bytes: u64 = files.iter().map(|f| f.bytes.len() as u64).sum();
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Bytes(total_bytes));
    group.sample_size(10);

    for (label, decode) in decoders {
        group.bench_function(*label, |b| {
            let mut d = Decoders::new();
            b.iter(|| {
                for f in files {
                    decode(&mut d, black_box(f));
                }
                black_box(&d.out);
            });
        });
    }
    group.finish();
}

/// Decode every file with both decoders (native RGB/RGBA) and report whether the
/// pixels agree. This proves the two paths are doing equivalent work — a fair
/// comparison — and would otherwise be hidden inside a pure timing loop.
fn verify(files: &[BenchFile]) {
    let mut d = Decoders::new();
    let mut identical = 0usize;
    let mut worst = 0i32;

    for f in files {
        d.wpd(f);
        let wpd_out = d.out.clone();
        d.image_webp(f);
        if wpd_out == d.out {
            identical += 1;
        } else {
            let diff = wpd_out
                .iter()
                .zip(&d.out)
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap_or(0);
            worst = worst.max(diff);
            eprintln!(
                "  verify: {} differs (wpd {} B, image-webp {} B, max Δ {diff})",
                f.name,
                wpd_out.len(),
                d.out.len()
            );
        }
    }
    eprintln!(
        "verify: {identical}/{} files byte-identical (worst channel Δ {worst})",
        files.len()
    );
}

fn main_config() -> Criterion {
    let output = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion");
    Criterion::default()
        .output_directory(&output)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
}

fn bench_native(c: &mut Criterion) {
    let files = load_files();
    if verify_enabled() {
        verify(&files);
    }
    bench_batch(
        c,
        "decode/native",
        &files,
        &[("wpd", Decoders::wpd), ("image-webp", Decoders::image_webp)],
    );
}

fn bench_rgba(c: &mut Criterion) {
    let files = load_files();
    bench_batch(
        c,
        "decode/rgba",
        &files,
        &[
            ("wpd", Decoders::wpd_rgba),
            ("image-webp", Decoders::image_webp_rgba),
        ],
    );
}

criterion_group! {
    name = benches;
    config = main_config();
    targets = bench_native, bench_rgba
}
criterion_main!(benches);

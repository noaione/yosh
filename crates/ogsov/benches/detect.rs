use std::collections::HashSet;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use image::GenericImageView;
use ogsov::detect::Ogsov;

struct Canvas {
    name: String,
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

fn main_config() -> Criterion {
    let output = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion");
    Criterion::default()
        .output_directory(&output)
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
}

fn model() -> &'static Ogsov {
    ogsov::embedded_model().unwrap_or_else(|| {
        panic!(
            "OGSOV weights missing; set OGSOV_WEIGHTS_PATH or add crates/ogsov/ogsov_weights.npz"
        )
    })
}

fn configured_sizes() -> Vec<usize> {
    let raw = std::env::var("OGSOV_BENCH_SIZES").unwrap_or_else(|_| "256,1024".to_owned());
    let sizes: Vec<_> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|&s| s > 4)
        .collect();
    assert!(
        !sizes.is_empty(),
        "OGSOV_BENCH_SIZES must contain comma-separated positive integers"
    );
    sizes
}

fn solid_gray(size: usize) -> Canvas {
    Canvas {
        name: "solid_gray".into(),
        width: size,
        height: size,
        rgb: [127, 127, 127].repeat(size * size),
    }
}

fn near_gray(size: usize) -> Canvas {
    let mut rgb = Vec::with_capacity(size * size * 3);
    for y in 0..size {
        for x in 0..size {
            let g = ((x * 17 + y * 31) & 0xff) as u8;
            rgb.extend_from_slice(&[g, g.saturating_add(1), g]);
        }
    }
    Canvas {
        name: "near_gray".into(),
        width: size,
        height: size,
        rgb,
    }
}

fn color_gradient(size: usize) -> Canvas {
    let mut rgb = Vec::with_capacity(size * size * 3);
    let den = size.saturating_sub(1).max(1);
    for y in 0..size {
        for x in 0..size {
            rgb.extend_from_slice(&[
                (x * 255 / den) as u8,
                (y * 255 / den) as u8,
                ((x + y) * 255 / (den * 2)) as u8,
            ]);
        }
    }
    Canvas {
        name: "color_gradient".into(),
        width: size,
        height: size,
        rgb,
    }
}

fn manga_with_spot_color(size: usize) -> Canvas {
    let mut rgb = vec![255; size * size * 3];
    let line_step = (size / 32).max(2);
    for y in (0..size).step_by(line_step) {
        for x in 0..size {
            let i = (y * size + x) * 3;
            rgb[i..i + 3].fill(24);
        }
    }
    for y in size / 3..size * 2 / 3 {
        for x in size / 3..size * 2 / 3 {
            let i = (y * size + x) * 3;
            rgb[i..i + 3].copy_from_slice(&[220, 45, 70]);
        }
    }
    Canvas {
        name: "manga_spot_color".into(),
        width: size,
        height: size,
        rgb,
    }
}

fn color_noise(size: usize) -> Canvas {
    let mut state = 0x4d59_5df4_d0f3_3173u64;
    let mut rgb = Vec::with_capacity(size * size * 3);
    for _ in 0..size * size * 3 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        rgb.push(state as u8);
    }
    Canvas {
        name: "color_noise".into(),
        width: size,
        height: size,
        rgb,
    }
}

fn synthetic_canvases(size: usize) -> Vec<Canvas> {
    vec![
        solid_gray(size),
        near_gray(size),
        color_gradient(size),
        manga_with_spot_color(size),
        color_noise(size),
    ]
}

fn bench_synthetic(c: &mut Criterion) {
    let model = model();
    let mut group = c.benchmark_group("synthetic/detect");
    for size in configured_sizes() {
        for canvas in synthetic_canvases(size) {
            group.throughput(Throughput::Elements((canvas.width * canvas.height) as u64));
            let id = BenchmarkId::new(
                canvas.name.clone(),
                format!("{}x{}", canvas.width, canvas.height),
            );
            group.bench_with_input(id, &canvas, |b, canvas| {
                b.iter(|| {
                    black_box(model.detect(black_box(&canvas.rgb), canvas.width, canvas.height))
                });
            });
        }
    }
    group.finish();
}

fn configured_image_roots() -> Vec<PathBuf> {
    std::env::var_os("OGSOV_BENCH_IMAGES")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default()
}

fn collect_files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        out.push(path.to_owned());
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        eprintln!("Skipping unreadable benchmark path: {}", path.display());
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

fn load_images() -> Vec<Canvas> {
    let mut files = Vec::new();
    for root in configured_image_roots() {
        collect_files(&root, &mut files);
    }
    files.sort();

    let mut names = HashSet::new();
    let mut images = Vec::new();
    for path in files {
        let Ok(img) = image::open(&path) else {
            continue;
        };
        let (width, height) = img.dimensions();
        if width.saturating_mul(height) <= 17 {
            eprintln!("Skipping tiny image: {}", path.display());
            continue;
        }
        let base = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
            .to_owned();
        let mut name = base.clone();
        let mut suffix = 2;
        while !names.insert(name.clone()) {
            name = format!("{base}_{suffix}");
            suffix += 1;
        }
        images.push(Canvas {
            name,
            width: width as usize,
            height: height as usize,
            rgb: img.into_rgb8().into_raw(),
        });
    }
    images
}

fn bench_images(c: &mut Criterion) {
    let images = load_images();
    if images.is_empty() {
        eprintln!("No real images configured; set OGSOV_BENCH_IMAGES to files/directories");
        return;
    }

    let model = model();
    let mut group = c.benchmark_group("images/detect");
    for image in images {
        group.throughput(Throughput::Elements((image.width * image.height) as u64));
        let id = BenchmarkId::new(
            image.name.clone(),
            format!("{}x{}", image.width, image.height),
        );
        group.bench_with_input(id, &image, |b, image| {
            b.iter(|| black_box(model.detect(black_box(&image.rgb), image.width, image.height)));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = main_config();
    targets = bench_synthetic, bench_images
}
criterion_main!(benches);

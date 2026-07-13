# OGSOV benchmarks

Benchmark measures `Ogsov::detect` only. Image decode and synthetic-canvas creation happen before timed loops.

Weights must exist at build time:

```powershell
$env:OGSOV_WEIGHTS_PATH = '/path/to/weights/ogsov.npz'
```

or,

```bash
set OGSOV_WEIGHTS_PATH=/path/to/weights/ogsov.npz
```

Run default deterministic canvases at 256x256 and 1024x1024:

```powershell
cargo bench -p ogsov --bench detect
```

Add custom sizes:

```powershell
$env:OGSOV_BENCH_SIZES = '256,1024,2048'
cargo bench -p ogsov --bench detect -- synthetic
```

or,
```bash
OGSOV_BENCH_SIZES=256,1024,2048 cargo bench -p ogsov --bench detect -- synthetic
```

Benchmark multiple real files, directories, or both. Directories scan recursively. PowerShell joins paths with `;`:

```powershell
$env:OGSOV_BENCH_IMAGES = 'I:/Manga/samples;I:/Manga/page.png'
cargo bench -p ogsov --bench detect -- images
```

or,
```bash
OGSOV_BENCH_IMAGES="I:/Manga/samples;I:/Manga/page.png" cargo bench -p ogsov --bench detect -- images
```

Useful Criterion options:

```powershell
cargo bench -p ogsov --bench detect -- --quick
cargo bench -p ogsov --bench detect -- --save-baseline before
cargo bench -p ogsov --bench detect -- --baseline before
```

Synthetic cases:

- `solid_gray`: exact grayscale fast pre-check.
- `near_gray`: full ML path with one-channel deviation.
- `color_gradient`: smooth full-color canvas.
- `manga_spot_color`: mostly monochrome line art plus color region.
- `color_noise`: deterministic high-entropy canvas.

Output reports latency and pixel throughput. Baseline data lands under `target/criterion/`.

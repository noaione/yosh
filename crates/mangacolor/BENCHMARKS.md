# mangacolor — validation and benchmarks

The CPU port of `vs-mangacolor`'s two models, embedded in yosh.  Everything
here is a development aid; the app only needs the crate.

## Weights

The build script converts the source weights into the two embedded assets.  It
looks for them, in order, via an environment variable, in the crate directory,
then in the workspace root:

| input | env var | file |
| --- | --- | --- |
| OGSOV MLPs + shared lookup | `OGSOV_WEIGHTS_PATH` | `ogsov_weights.npz` |
| OGSOX heads | `OGSOX_WEIGHTS_PATH` | `ogsox_classifier_optimized.onnx` |
| optional OGSOX lookup cross-check | `OGSOX_LOOKUP_PATH` | `ogsox_lookup_table.npz` |

Missing weights are a supported state: the crate builds without a model and
`ogsov_available()` / `ogsox_available()` report `false`.  A malformed file
that *is* present fails the build with a message naming the tensor.

CI (the test, release and artifact workflows) does the same thing as the
`vs-mangacolor` CI:
they download one zip holding `ogsov.npz`, `ogsox_classifier_optimized.onnx`
and `ogsox_lookup_table.npz` from the `MANGACOLOR_WEIGHTS_URL` secret
(`OGSOV_WEIGHTS_URL` is accepted as a fallback), unpack it by basename into
this crate's directory, and let the build script generate the assets.

The generated assets are byte-identical to `vs-mangacolor`'s
`assets/{ogsov,ogsox}.bin` (verified by `sha256sum` when the port landed).

## Parity with the references

`examples/parity.rs` classifies a raw interleaved RGB file:

```sh
cargo run --release -p mangacolor --example parity -- page.rgb 1025 777
```

The Python references in `../vs-mangacolor/tests/references/` take the same
pixels in BGR.  On the synthetic fixtures used when the port landed (near-gray,
spot colour, gradient, noise, and a 1025x777 gradient that exercises OGSOX
tiling):

* labels matched on every fixture for both models;
* OGSOV confidence agreed with the reference within float32 accumulation noise
  (≤ 1e-5 absolute);
* OGSOX's signed score agreed with both the full ONNX graph and the
  fast-feature reference within 1e-5 (the port follows the graph's own
  eight-lane reduction, so it lands on one or the other depending on the tile).

## Benchmarks

```sh
MANGACOLOR_BENCH_SIZES=1024,2048 cargo bench -p mangacolor --bench detect
MANGACOLOR_BENCH_IMAGES="I:/Manga/samples;I:/Manga/page.png" \
  cargo bench -p mangacolor --bench detect -- images
```

The benchmark measures `detect_ogsov` / `detect_ogsox`: the one-pass
de-interleave into reusable planes (NEON structure loads on aarch64) plus the
model's own kernels.  That is the cost yosh pays per decoded page.

### Port vs the superseded `ogsov` crate

Apple M4, 2048x2048 synthetic canvases, release profile, criterion means:

| canvas | old `ogsov` (planar input) | this port (RGBA-ready) | speed-up |
| --- | ---: | ---: | ---: |
| solid gray | 1.13 ms | 0.45 ms | 2.5x |
| near gray | 13.85 ms | 8.56 ms | 1.6x |
| color gradient | 13.77 ms | 8.14 ms | 1.7x |
| manga spot colour | 10.74 ms | 6.94 ms | 1.5x |
| color noise | 12.43 ms | 5.01 ms | 2.5x |

The old crate had no SIMD per-pixel kernels and took interleaved RGB directly;
the port adds the NEON/AVX2/AVX-512 mask test, the vectorised dense kernels,
the bounded-histogram order statistics, and a NEON de-interleave (so the new
numbers include work the old ones did not).

OGSOX on the same canvases is 1.6–2.1 ms (512² tiling, five attention heads).

The per-pixel SIMD kernels are selected at compile time, as in
`vs-mangacolor`: on aarch64 NEON is part of the baseline, while on x86-64 the
AVX2/AVX-512 kernels exist only when the build enables those target features
(`RUSTFLAGS="-C target-feature=+avx2"`).  yosh's release CI does not enable
them (a single binary must run on pre-AVX2 CPUs), so x86 builds take the scalar
kernels — still faster than the old crate because of the algorithmic changes,
but runtime CPU dispatch would be the way to get the vector paths there too.


## Tests

```sh
cargo test -p mangacolor
```

The crate's tests cover: the SIMD/scalar differential checks (per-pixel scans,
dense kernels, de-interleave), the histogram-reuse invariants, the ONNX-derived
tile geometry, and the ported reference tests.

//! Converts the checked-out model weights into the flat assets this crate embeds.
//!
//! The crate never requires the weights: this script always writes
//! `$OUT_DIR/ogsov.bin`, `$OUT_DIR/ogsox.bin` and `$OUT_DIR/models_gen.rs`, and
//! a missing input simply produces an empty asset plus a `*_EMBEDDED = false`
//! constant.  An input that *is* present but malformed aborts the build with a
//! message naming the file and the tensor.  No external crate is used to read
//! the ONNX file: the protobuf wire format is small enough to walk here.
//!
//! Inputs are resolved in this order:
//!
//! | input          | env var              | crate directory                   | workspace root                    |
//! |----------------|----------------------|-----------------------------------|-----------------------------------|
//! | OGSOV weights  | `OGSOV_WEIGHTS_PATH` | `ogsov_weights.npz`               | `ogsov_weights.npz`               |
//! | OGSOX weights  | `OGSOX_WEIGHTS_PATH` | `ogsox_classifier_optimized.onnx` | `ogsox_classifier_optimized.onnx` |
//! | OGSOX lookup   | `OGSOX_LOOKUP_PATH`  | `ogsox_lookup_table.npz`          | `ogsox_lookup_table.npz`          |
//!
//! Relative env paths resolve against the crate directory.  The workspace root
//! is the crate's grandparent (`crates/mangacolor` -> the workspace).  The
//! lookup npz is optional; when it is present it must agree with the OGSOV
//! npz's `mask_lookup`, otherwise the OGSOV table is used for both assets.
//!
//! Asset layout, little-endian, as decoded by `src/weights.rs`:
//!
//! * `OGSV` v1: magic, `u32` version, `u32` min_spread, then five classifiers in
//!   layer order 0, 3, 6 of (w1 512x32, b1 512, w2 512x512, b2 512, w3 1x512,
//!   b3 1) as `f32`, then the run-length lookup.
//! * `OGSX` v2: the same header, then five heads of the fifteen tensors below,
//!   then the same run-length lookup.
//!
//! The lookup is `u32` run count followed by that many `u32` run lengths,
//! alternating false/true and summing to exactly `256^3`.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Build-script errors are plain strings: they end up inside an actionable panic.
type Error = String;
type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// OGSOV: five MLPs over a 32-value colourfulness histogram
// ---------------------------------------------------------------------------

const OGSOV_MAGIC: &[u8; 4] = b"OGSV";
const OGSOV_VERSION: u32 = 1;
const OGSOV_CLASSIFIERS: usize = 5;
const OGSOV_INPUT: usize = 32;
const OGSOV_HIDDEN: usize = 512;
/// `(npz layer prefix, rows, columns)` of the three dense layers, in asset order.
const OGSOV_LAYERS: [(&str, usize, usize); 3] = [
    ("0", OGSOV_HIDDEN, OGSOV_INPUT),
    ("3", OGSOV_HIDDEN, OGSOV_HIDDEN),
    ("6", 1, OGSOV_HIDDEN),
];
const OGSOV_PARAMETERS_PER_CLASSIFIER: usize = OGSOV_HIDDEN * OGSOV_INPUT
    + OGSOV_HIDDEN
    + OGSOV_HIDDEN * OGSOV_HIDDEN
    + OGSOV_HIDDEN
    + OGSOV_HIDDEN
    + 1;

// ---------------------------------------------------------------------------
// OGSOX: five attention heads over nine per-tile features
// ---------------------------------------------------------------------------

const OGSOX_MAGIC: &[u8; 4] = b"OGSX";
const OGSOX_VERSION: u32 = 2;
const OGSOX_HEADS: usize = 5;
const OGSOX_GATE: u64 = 256;
const OGSOX_HALF: u64 = 128;
const OGSOX_EMBED: u64 = 16;
const OGSOX_ATTENTION: u64 = 64;
const OGSOX_LOGITS: u64 = 2;
const OGSOX_FEATURES: u64 = 9;

/// The ONNX `TensorProto.data_type` of a float32 tensor.
const ONNX_FLOAT: u64 = 1;

/// The heads' tensors in the order the asset stores them, with asset shapes.
///
/// `Wa1` is `(16, 64)` input-major because the reference graph feeds it to a
/// `MatMul`; every other weight is output-major because the graph uses
/// `Gemm(transB=1)`.  `Wa2` is stored as `(64, 1)` in the graph and reaches the
/// asset as its column 0, which for that shape is the whole tensor.
const OGSOX_TENSORS: [(&str, &[u64]); 15] = [
    ("W1", &[OGSOX_GATE, OGSOX_FEATURES]),
    ("b1", &[OGSOX_GATE]),
    ("ln1g", &[OGSOX_HALF]),
    ("ln1b", &[OGSOX_HALF]),
    ("W2", &[OGSOX_EMBED, OGSOX_HALF]),
    ("b2", &[OGSOX_EMBED]),
    ("ln2g", &[OGSOX_EMBED]),
    ("ln2b", &[OGSOX_EMBED]),
    ("Wa1", &[OGSOX_EMBED, OGSOX_ATTENTION]),
    ("ba1", &[OGSOX_ATTENTION]),
    ("Wa2", &[OGSOX_ATTENTION]),
    ("Wc1", &[OGSOX_ATTENTION, OGSOX_EMBED]),
    ("bc1", &[OGSOX_ATTENTION]),
    ("Wc2", &[OGSOX_LOGITS, OGSOX_ATTENTION]),
    ("bc2", &[OGSOX_LOGITS]),
];

/// Floats one head occupies, matching `weights.rs::OGSOX_HEAD_FLOATS` (7282).
const OGSOX_HEAD_FLOATS: u64 = OGSOX_GATE * OGSOX_FEATURES
    + OGSOX_GATE
    + OGSOX_HALF
    + OGSOX_HALF
    + OGSOX_EMBED * OGSOX_HALF
    + OGSOX_EMBED
    + OGSOX_EMBED
    + OGSOX_EMBED
    + OGSOX_EMBED * OGSOX_ATTENTION
    + OGSOX_ATTENTION
    + OGSOX_ATTENTION
    + OGSOX_ATTENTION * OGSOX_EMBED
    + OGSOX_ATTENTION
    + OGSOX_LOGITS * OGSOX_ATTENTION
    + OGSOX_LOGITS;

/// Entries in the flattened `256^3` `(b, g, r)` lookup table.
const MASK_ELEMS: usize = 256 * 256 * 256;

fn main() {
    for variable in [
        "OGSOV_WEIGHTS_PATH",
        "OGSOX_WEIGHTS_PATH",
        "OGSOX_LOOKUP_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"));
    // `crates/mangacolor` -> the workspace root.
    let workspace_root = manifest_dir
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.clone());
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));

    let ogsov_candidates = candidates(
        "OGSOV_WEIGHTS_PATH",
        &manifest_dir,
        &workspace_root,
        "ogsov_weights.npz",
    );
    let ogsox_candidates = candidates(
        "OGSOX_WEIGHTS_PATH",
        &manifest_dir,
        &workspace_root,
        "ogsox_classifier_optimized.onnx",
    );
    let lookup_candidates = candidates(
        "OGSOX_LOOKUP_PATH",
        &manifest_dir,
        &workspace_root,
        "ogsox_lookup_table.npz",
    );

    let ogsov_path = pick(&ogsov_candidates);
    let ogsox_path = pick(&ogsox_candidates);
    let lookup_path = pick(&lookup_candidates);
    watch(ogsov_path.as_deref());
    watch(ogsox_path.as_deref());
    watch(lookup_path.as_deref());

    // --- read whatever is present ------------------------------------------
    let ogsov = ogsov_path.as_deref().map(|path| {
        (
            path,
            read_ogsov(path).unwrap_or_else(|error| {
                panic!(
                    "failed to convert the OGSOV weights at {}: {error}\n\
                     fix or replace that file, or unset OGSOV_WEIGHTS_PATH / remove it to build \
                     without the embedded OGSOV model",
                    path.display()
                )
            }),
        )
    });

    let lookup = lookup_path.as_deref().map(|path| {
        (
            path,
            read_lookup(path).unwrap_or_else(|error| {
                panic!(
                    "failed to read the OGSOX lookup table at {}: {error}\n\
                     fix or replace that file, or unset OGSOX_LOOKUP_PATH / remove it to fall \
                     back to the OGSOV mask_lookup",
                    path.display()
                )
            }),
        )
    });

    if let (Some((ogsov_at, source)), Some((lookup_at, table))) = (ogsov.as_ref(), lookup.as_ref())
        && source.mask != *table
    {
        panic!(
            "the OGSOX lookup table at {} and the OGSOV mask_lookup at {} are different; both \
             assets must carry the same table — point OGSOX_LOOKUP_PATH at the matching npz or \
             remove it",
            lookup_at.display(),
            ogsov_at.display()
        );
    }

    // OGSOX uses its own lookup npz when present, else the OGSOV table.
    let ogsox_mask: Option<&[bool]> = lookup
        .as_ref()
        .map(|(_, table)| table.as_slice())
        .or_else(|| ogsov.as_ref().map(|(_, source)| source.mask.as_slice()));

    // --- verdict -----------------------------------------------------------
    let ogsov_ok = ogsov.is_some();
    let ogsox_ok = ogsox_path.is_some();
    if !ogsov_ok {
        println!(
            "cargo:warning=OGSOV weights not found (searched {}); building without the embedded \
             OGSOV model",
            describe(&ogsov_candidates)
        );
    }
    if !ogsox_ok {
        println!(
            "cargo:warning=OGSOX weights not found (searched {}); building without the embedded \
             OGSOX model",
            describe(&ogsox_candidates)
        );
    }

    // --- write the assets --------------------------------------------------
    let ogsov_bin = match ogsov.as_ref() {
        Some((path, source)) => {
            let lookup = Lookup::new(&source.mask).unwrap_or_else(|error| {
                panic!(
                    "cannot encode the OGSOV lookup table in {}: {error}",
                    path.display()
                )
            });
            let mut bytes = Vec::with_capacity(
                OGSOV_MAGIC.len() + 8 + source.parameters.len() + 4 + lookup.runs.len() * 4,
            );
            bytes.extend_from_slice(OGSOV_MAGIC);
            push_u32(&mut bytes, OGSOV_VERSION);
            push_u32(&mut bytes, lookup.min_spread);
            bytes.extend_from_slice(&source.parameters);
            lookup.append_to(&mut bytes);
            println!(
                "cargo:warning=embedded OGSV v1 ({} bytes) from {}",
                bytes.len(),
                path.display()
            );
            bytes
        }
        None => Vec::new(),
    };

    let ogsox_bin = match (ogsox_path.as_deref(), ogsox_mask) {
        (Some(path), Some(mask)) => {
            let initializers = parse_onnx(path).unwrap_or_else(|error| {
                panic!(
                    "failed to convert the OGSOX weights at {}: {error}\n\
                     fix or replace that file, or unset OGSOX_WEIGHTS_PATH / remove it to build \
                     without the embedded OGSOX model",
                    path.display()
                )
            });
            let lookup = Lookup::new(mask).unwrap_or_else(|error| {
                panic!(
                    "cannot encode the OGSOX lookup table used for {}: {error}",
                    path.display()
                )
            });
            let mut bytes = Vec::with_capacity(
                OGSOX_MAGIC.len()
                    + 8
                    + (OGSOX_HEADS as u64 * OGSOX_HEAD_FLOATS * 4) as usize
                    + 4
                    + lookup.runs.len() * 4,
            );
            bytes.extend_from_slice(OGSOX_MAGIC);
            push_u32(&mut bytes, OGSOX_VERSION);
            push_u32(&mut bytes, lookup.min_spread);
            for head in 0..OGSOX_HEADS {
                append_head(&initializers, head, &mut bytes).unwrap_or_else(|error| {
                    panic!(
                        "failed to convert the OGSOX weights at {}: {error}",
                        path.display()
                    )
                });
            }
            lookup.append_to(&mut bytes);
            println!(
                "cargo:warning=embedded OGSX v2 ({} bytes) from {}",
                bytes.len(),
                path.display()
            );
            bytes
        }
        (Some(path), None) => {
            println!(
                "cargo:warning=OGSOX weights at {} need a lookup table, but neither \
                 ogsox_lookup_table.npz nor the OGSOV npz was found; building without the \
                 embedded OGSOX model",
                path.display()
            );
            Vec::new()
        }
        (None, _) => Vec::new(),
    };

    write_file(&out_dir.join("ogsov.bin"), &ogsov_bin);
    write_file(&out_dir.join("ogsox.bin"), &ogsox_bin);
    let generated = format!(
        "/// Whether the OGSOV weights were embedded at build time.\n\
         pub const OGSOV_EMBEDDED: bool = {ogsov};\n\
         \n\
         /// Whether the OGSOX weights were embedded at build time.\n\
         pub const OGSOX_EMBEDDED: bool = {ogsox};\n",
        ogsov = !ogsov_bin.is_empty(),
        ogsox = !ogsox_bin.is_empty(),
    );
    write_file(&out_dir.join("models_gen.rs"), generated.as_bytes());
}

// ---------------------------------------------------------------------------
// input discovery
// ---------------------------------------------------------------------------

/// Candidate paths for one input, highest priority first.
///
/// A relative env var resolves against the crate directory, matching the old
/// `crates/ogsov` converter.  When the env var is unset the crate directory is
/// tried before the workspace root.
fn candidates(
    variable: &str,
    manifest_dir: &Path,
    workspace_root: &Path,
    file_name: &str,
) -> Vec<PathBuf> {
    match env::var_os(variable) {
        Some(value) => {
            let path = PathBuf::from(value);
            vec![if path.is_absolute() {
                path
            } else {
                manifest_dir.join(path)
            }]
        }
        None => vec![manifest_dir.join(file_name), workspace_root.join(file_name)],
    }
}

/// The first candidate that exists.
fn pick(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|path| path.is_file()).cloned()
}

/// Ask Cargo to re-run when an input we actually used changes.
///
/// Only existing paths are declared: Cargo re-runs the script on *every* build
/// for a `rerun-if-changed` path that does not exist, which would re-convert the
/// present weights each time.  When nothing is found at all no directive is
/// emitted, so Cargo falls back to watching the package and dropping a weights
/// file next to `build.rs` still triggers a rebuild; a file dropped at the
/// workspace root needs `touch build.rs` (or `cargo clean -p mangacolor`) to be
/// noticed.
fn watch(path: Option<&Path>) {
    if let Some(path) = path {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn describe(candidates: &[PathBuf]) -> String {
    candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// .npz reading (zip + npyz, same style as crates/ogsov/build.rs)
// ---------------------------------------------------------------------------

/// An open `.npz`: a zip of `<name>.npy` members.
struct Npz {
    path: PathBuf,
    archive: zip::ZipArchive<std::io::BufReader<fs::File>>,
}

impl Npz {
    fn open(path: &Path) -> Result<Self> {
        let file = fs::File::open(path)
            .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        let archive = zip::ZipArchive::new(std::io::BufReader::new(file)).map_err(|error| {
            format!("{} is not a readable .npz archive: {error}", path.display())
        })?;
        Ok(Npz {
            path: path.to_path_buf(),
            archive,
        })
    }

    fn member(&mut self, name: &str) -> Result<npyz::NpyFile<std::io::Cursor<Vec<u8>>>> {
        let path = self.path.clone();
        let mut member = self
            .archive
            .by_name(&format!("{name}.npy"))
            .map_err(|error| format!("{}: {name}.npy is missing: {error}", path.display()))?;
        let mut buffer = Vec::with_capacity(member.size() as usize);
        member
            .read_to_end(&mut buffer)
            .map_err(|error| format!("{}: cannot read {name}.npy: {error}", path.display()))?;
        npyz::NpyFile::new(std::io::Cursor::new(buffer)).map_err(|error| {
            format!(
                "{}: {name}.npy has an unreadable header: {error}",
                path.display()
            )
        })
    }

    /// Read and flatten the shared `(256, 256, 256)` boolean `mask_lookup`.
    fn read_mask(&mut self) -> Result<Vec<bool>> {
        let path = self.path.clone();
        let array = self.member("mask_lookup")?;
        if array.shape() != &[256u64, 256, 256][..] {
            return Err(format!(
                "{}: mask_lookup has shape {:?}, expected [256, 256, 256]",
                path.display(),
                array.shape()
            ));
        }
        if array.order() != npyz::Order::C {
            return Err(format!(
                "{}: mask_lookup is stored in Fortran order, expected C order",
                path.display()
            ));
        }
        // The stored dtype may be bool ('|b1') or a signed/unsigned byte type.
        match array.dtype() {
            npyz::DType::Plain(ty) if ty.to_string().ends_with("b1") => array
                .into_vec::<bool>()
                .map_err(|error| format!("{}: cannot read mask_lookup: {error}", path.display())),
            npyz::DType::Plain(ty) if ty.to_string().ends_with("u1") => array
                .into_vec::<u8>()
                .map_err(|error| format!("{}: cannot read mask_lookup: {error}", path.display()))
                .map(|values| values.into_iter().map(|value| value != 0).collect()),
            npyz::DType::Plain(ty) if ty.to_string().ends_with("i1") => array
                .into_vec::<i8>()
                .map_err(|error| format!("{}: cannot read mask_lookup: {error}", path.display()))
                .map(|values| values.into_iter().map(|value| value != 0).collect()),
            other => Err(format!(
                "{}: mask_lookup has dtype {other:?}, expected bool, uint8 or int8",
                path.display()
            )),
        }
    }

    /// Append one named float array as little-endian `f32`, narrowing `f64`.
    fn append_f32s(&mut self, name: &str, shape: &[u64], out: &mut Vec<u8>) -> Result<()> {
        let path = self.path.clone();
        let array = self.member(name)?;
        if array.shape() != shape {
            return Err(format!(
                "{}: {name} has shape {:?}, expected {shape:?}",
                path.display(),
                array.shape()
            ));
        }
        if array.order() != npyz::Order::C {
            return Err(format!(
                "{}: {name} is stored in Fortran order, expected C order",
                path.display()
            ));
        }
        let values: Vec<f32> = match array.dtype() {
            npyz::DType::Plain(ty) if ty.to_string().ends_with("f4") => array
                .into_vec::<f32>()
                .map_err(|error| format!("{}: cannot read {name}: {error}", path.display()))?,
            npyz::DType::Plain(ty) if ty.to_string().ends_with("f8") => array
                .into_vec::<f64>()
                .map_err(|error| format!("{}: cannot read {name}: {error}", path.display()))?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            other => {
                return Err(format!(
                    "{}: {name} has dtype {other:?}, expected float32 or float64",
                    path.display()
                ));
            }
        };
        out.reserve(values.len() * 4);
        for (index, value) in values.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "{}: {name} parameter {index} is not finite ({value})",
                    path.display()
                ));
            }
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

/// The OGSOV npz's two payloads: classifier parameters and the shared table.
struct OgsovSource {
    /// Little-endian `f32` bytes of the five classifiers, in asset order.
    parameters: Vec<u8>,
    /// `mask_lookup`, flattened C-order.
    mask: Vec<bool>,
}

fn read_ogsov(path: &Path) -> Result<OgsovSource> {
    let mut npz = Npz::open(path)?;
    let mask = npz.read_mask()?;
    let mut parameters =
        Vec::with_capacity(OGSOV_CLASSIFIERS * OGSOV_PARAMETERS_PER_CLASSIFIER * 4);
    for classifier in 0..OGSOV_CLASSIFIERS {
        for &(layer, rows, columns) in &OGSOV_LAYERS {
            npz.append_f32s(
                &format!("{classifier}.{layer}.weight"),
                &[rows as u64, columns as u64],
                &mut parameters,
            )?;
            npz.append_f32s(
                &format!("{classifier}.{layer}.bias"),
                &[rows as u64],
                &mut parameters,
            )?;
        }
    }
    Ok(OgsovSource { parameters, mask })
}

fn read_lookup(path: &Path) -> Result<Vec<bool>> {
    Npz::open(path)?.read_mask()
}

// ---------------------------------------------------------------------------
// lookup encoding
// ---------------------------------------------------------------------------

/// The run-length encoded lookup payload plus the header's `min_spread`.
struct Lookup {
    runs: Vec<u32>,
    min_spread: u32,
}

impl Lookup {
    fn new(mask: &[bool]) -> Result<Self> {
        if mask.len() != MASK_ELEMS {
            return Err(format!(
                "the lookup table has {} entries, expected 256^3 = {MASK_ELEMS}",
                mask.len()
            ));
        }
        // Runs alternate false/true, so they always start with a false run; a
        // table that starts true gets an empty one prepended.
        let mut runs: Vec<u32> = Vec::new();
        if mask[0] {
            runs.push(0);
        }
        let mut run_value = mask[0];
        let mut run_length = 0u32;
        for &value in mask {
            if value == run_value {
                run_length += 1;
            } else {
                runs.push(run_length);
                run_value = value;
                run_length = 1;
            }
        }
        runs.push(run_length);
        verify_runs(mask, &runs)?;
        Ok(Lookup {
            runs,
            min_spread: min_spread(mask),
        })
    }

    fn append_to(&self, out: &mut Vec<u8>) {
        push_u32(out, self.runs.len() as u32);
        for &run in &self.runs {
            push_u32(out, run);
        }
    }
}

/// Prove the encoding is lossless and covers the whole table before writing it.
fn verify_runs(mask: &[bool], runs: &[u32]) -> Result<()> {
    let total: u64 = runs.iter().map(|&run| u64::from(run)).sum();
    if total != MASK_ELEMS as u64 {
        return Err(format!(
            "the lookup runs cover {total} entries, expected {MASK_ELEMS}"
        ));
    }
    let mut decoded = vec![false; MASK_ELEMS];
    let mut position = 0usize;
    for (index, &run) in runs.iter().enumerate() {
        let end = position + run as usize;
        if index % 2 == 1 {
            decoded[position..end].fill(true);
        }
        position = end;
    }
    if decoded != mask {
        return Err("the lookup run-length round trip is not lossless".to_string());
    }
    Ok(())
}

/// Smallest `max(b, g, r) - min(b, g, r)` over the true entries, else 256.
fn min_spread(mask: &[bool]) -> u32 {
    let mut best = 256;
    for blue in 0..256u32 {
        for green in 0..256u32 {
            let high = blue.max(green);
            let low = blue.min(green);
            let base = ((blue << 16) | (green << 8)) as usize;
            for red in 0..256u32 {
                if mask[base + red as usize] {
                    let spread = high.max(red) - low.min(red);
                    if spread < best {
                        best = spread;
                        if best == 0 {
                            return 0;
                        }
                    }
                }
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// minimal ONNX reader (protobuf wire format only)
// ---------------------------------------------------------------------------

/// One `TensorProto` from the model graph's initializer list.
struct Tensor {
    name: String,
    dims: Vec<u64>,
    data_type: u64,
    raw_data: Vec<u8>,
}

#[derive(Clone, Copy)]
enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

struct Field<'a> {
    number: u64,
    value: Value<'a>,
}

/// Bounds-checked protobuf reader; fixed-width fields are skipped.
struct Cursor<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, position: 0 }
    }

    fn done(&self) -> bool {
        self.position >= self.data.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            if shift >= 64 {
                return Err("protobuf varint is longer than 64 bits".to_string());
            }
            let byte = *self
                .data
                .get(self.position)
                .ok_or_else(|| "protobuf input ends inside a varint".to_string())?;
            self.position += 1;
            let bits = u64::from(byte & 0x7f);
            if shift == 63 && bits > 1 {
                return Err("protobuf varint overflows 64 bits".to_string());
            }
            value |= bits << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
    }

    fn length_delimited(&mut self) -> Result<&'a [u8]> {
        let length = usize::try_from(self.varint()?)
            .map_err(|_| "protobuf length does not fit in usize".to_string())?;
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| "protobuf length overflows usize".to_string())?;
        if end > self.data.len() {
            return Err("protobuf field runs past the end of the input".to_string());
        }
        let slice = &self.data[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn skip(&mut self, count: usize) -> Result<()> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| "protobuf field overflows usize".to_string())?;
        if end > self.data.len() {
            return Err("protobuf field runs past the end of the input".to_string());
        }
        self.position = end;
        Ok(())
    }
}

/// Walk one message's fields, keeping varint and length-delimited payloads.
fn fields(data: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut cursor = Cursor::new(data);
    let mut fields = Vec::new();
    while !cursor.done() {
        let key = cursor.varint()?;
        let number = key >> 3;
        if number == 0 {
            return Err("protobuf field number 0 is invalid".to_string());
        }
        match key & 7 {
            0 => {
                let value = cursor.varint()?;
                fields.push(Field {
                    number,
                    value: Value::Varint(value),
                });
            }
            1 => cursor.skip(8)?,
            2 => {
                let value = cursor.length_delimited()?;
                fields.push(Field {
                    number,
                    value: Value::Bytes(value),
                });
            }
            3 | 4 => {
                return Err(format!(
                    "protobuf groups (wire type {}) are not supported",
                    key & 7
                ));
            }
            5 => cursor.skip(4)?,
            other => return Err(format!("unknown protobuf wire type {other}")),
        }
    }
    Ok(fields)
}

fn first_bytes<'a>(fields: &[Field<'a>], number: u64) -> Option<&'a [u8]> {
    fields.iter().find_map(|field| match field.value {
        Value::Bytes(bytes) if field.number == number => Some(bytes),
        _ => None,
    })
}

fn packed_varints(data: &[u8]) -> Result<Vec<u64>> {
    let mut cursor = Cursor::new(data);
    let mut values = Vec::new();
    while !cursor.done() {
        values.push(cursor.varint()?);
    }
    Ok(values)
}

fn parse_onnx(path: &Path) -> Result<HashMap<String, Tensor>> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let model = fields(&bytes)
        .map_err(|error| format!("{} is not a protobuf ONNX model: {error}", path.display()))?;
    let graph = first_bytes(&model, 7)
        .ok_or_else(|| format!("{} has no model graph (ModelProto field 7)", path.display()))?;
    let graph = fields(graph)
        .map_err(|error| format!("{}: the model graph is malformed: {error}", path.display()))?;
    let mut initializers = HashMap::new();
    for field in &graph {
        if field.number != 5 {
            continue;
        }
        let Value::Bytes(tensor) = field.value else {
            continue;
        };
        let tensor = parse_tensor(tensor, path)?;
        initializers.insert(tensor.name.clone(), tensor);
    }
    if initializers.is_empty() {
        return Err(format!(
            "{}: the model graph has no initializers (GraphProto field 5)",
            path.display()
        ));
    }
    Ok(initializers)
}

fn parse_tensor(data: &[u8], path: &Path) -> Result<Tensor> {
    let mut name = None;
    let mut dims = Vec::new();
    let mut data_type = None;
    let mut raw_data = None;
    for field in fields(data).map_err(|error| {
        format!(
            "{}: an initializer is not a valid TensorProto: {error}",
            path.display()
        )
    })? {
        match (field.number, field.value) {
            (1, Value::Varint(dim)) => dims.push(dim),
            (1, Value::Bytes(packed)) => dims.extend(packed_varints(packed).map_err(|error| {
                format!(
                    "{}: an initializer's packed dims are malformed: {error}",
                    path.display()
                )
            })?),
            (2, Value::Varint(kind)) => data_type = Some(kind),
            (8, Value::Bytes(bytes)) => name = Some(String::from_utf8_lossy(bytes).into_owned()),
            (9, Value::Bytes(bytes)) => raw_data = Some(bytes.to_vec()),
            _ => {}
        }
    }
    let name = name.ok_or_else(|| format!("{}: an initializer has no name", path.display()))?;
    let data_type = data_type
        .ok_or_else(|| format!("{}: initializer {name} has no data type", path.display()))?;
    let raw_data = raw_data.ok_or_else(|| {
        format!(
            "{}: initializer {name} is not stored in raw_data",
            path.display()
        )
    })?;
    Ok(Tensor {
        name,
        dims,
        data_type,
        raw_data,
    })
}

fn asset_shape(name: &str) -> Result<&'static [u64]> {
    OGSOX_TENSORS
        .iter()
        .find_map(|&(tensor, shape)| (tensor == name).then_some(shape))
        .ok_or_else(|| format!("internal: no asset shape for OGSOX tensor {name}"))
}

/// Append one head's fifteen tensors, in asset order, as little-endian `f32`.
fn append_head(
    initializers: &HashMap<String, Tensor>,
    head: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let value = [93usize, 128, 162, 196, 230][head];
    let base = 1 + 15 * head;
    // `(asset name, ONNX initializer, arrives as a (n, 1) column)`.
    let sources: [(&str, String, bool); 15] = [
        ("W1", format!("lifted_tensor_{base}"), false),
        ("b1", format!("lifted_tensor_{}", base + 1), false),
        ("ln1g", format!("lifted_tensor_{}", base + 2), false),
        ("ln1b", format!("lifted_tensor_{}", base + 3), false),
        ("W2", format!("lifted_tensor_{}", base + 4), false),
        ("b2", format!("lifted_tensor_{}", base + 5), false),
        ("ln2g", format!("lifted_tensor_{}", base + 6), false),
        ("ln2b", format!("lifted_tensor_{}", base + 7), false),
        ("Wa1", format!("val_{value}"), false),
        ("ba1", format!("lifted_tensor_{}", base + 9), false),
        // `Wa2` comes from a MatMul operand shaped (64, 1); its column 0 is the
        // whole tensor, so the stored order already matches the asset.
        ("Wa2", format!("val_{}", value + 2), true),
        ("Wc1", format!("lifted_tensor_{}", base + 11), false),
        ("bc1", format!("lifted_tensor_{}", base + 12), false),
        ("Wc2", format!("lifted_tensor_{}", base + 13), false),
        ("bc2", format!("lifted_tensor_{}", base + 14), false),
    ];

    for (name, tensor_name, column) in sources {
        let tensor = initializers.get(&tensor_name).ok_or_else(|| {
            format!("the graph has no initializer {tensor_name}, needed for head {head}'s {name}")
        })?;
        let shape = asset_shape(name)?;
        let source_shape: Vec<u64> = if column {
            let mut shape = shape.to_vec();
            shape.push(1);
            shape
        } else {
            shape.to_vec()
        };
        if tensor.dims != source_shape {
            return Err(format!(
                "head {head} tensor {name} ({tensor_name}) has shape {:?}, expected {source_shape:?}",
                tensor.dims
            ));
        }
        if tensor.data_type != ONNX_FLOAT {
            return Err(format!(
                "head {head} tensor {name} ({tensor_name}) has ONNX data type {}, expected \
                 {ONNX_FLOAT} (float32)",
                tensor.data_type
            ));
        }
        let expected_bytes = source_shape.iter().product::<u64>() * 4;
        if tensor.raw_data.len() as u64 != expected_bytes {
            return Err(format!(
                "head {head} tensor {name} ({tensor_name}) has {} raw bytes, expected \
                 {expected_bytes}",
                tensor.raw_data.len()
            ));
        }
        out.reserve(expected_bytes as usize);
        for (index, chunk) in tensor.raw_data.as_chunks::<4>().0.iter().enumerate() {
            let value = f32::from_le_bytes(*chunk);
            if !value.is_finite() {
                return Err(format!(
                    "head {head} tensor {name} ({tensor_name}) parameter {index} is not finite \
                     ({value})"
                ));
            }
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_file(path: &Path, contents: &[u8]) {
    fs::write(path, contents)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", path.display()));
}

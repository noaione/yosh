//! Converts the OGSOV .npz weights into a flat binary at build time.
//!
//! Weight resolution order:
//!   1. `OGSOV_WEIGHTS_PATH` env var (relative paths resolved against the
//!      crate manifest dir)
//!   2. `ogsov_weights.npz` in the crate root
//!
//! If found, `$OUT_DIR/ogsov.bin` gets the converted weights and the
//! generated constant `OGSOV_EMBEDDED` is `true`. If not, an empty file is
//! written and the constant is `false`, so the crate always compiles.
//!
//! Binary layout (little-endian):
//!   [0 .. 2_097_152)  bit-packed mask_lookup (MSB-first, C-order over
//!                     (256,256,256); bit index = b*65536 + g*256 + r)
//!   then for classifier 0..5, layers (0, 3, 6):
//!       weight (out, in) f32 row-major, then bias (out,) f32

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const MASK_ELEMS: usize = 256 * 256 * 256;
const LAYER_SHAPES: [(&str, usize, usize); 3] = [("0", 512, 32), ("3", 512, 512), ("6", 1, 512)];

fn main() {
    println!("cargo:rerun-if-env-changed=OGSOV_WEIGHTS_PATH");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let npz_path = match env::var_os("OGSOV_WEIGHTS_PATH") {
        Some(p) => {
            let p = PathBuf::from(p);
            if p.is_absolute() { p } else { manifest_dir.join(p) }
        }
        None => manifest_dir.join("ogsov_weights.npz"),
    };
    // NOTE: if the file doesn't exist yet, cargo re-runs this script every
    // build until it appears. That's cheap and means dropping the npz in
    // place "just works" without a `cargo clean`.
    println!("cargo:rerun-if-changed={}", npz_path.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bin_path = out_dir.join("ogsov.bin");
    let gen_path = out_dir.join("ogsov_gen.rs");

    let embedded = if npz_path.exists() {
        println!(
            "cargo:warning=converting OGSOV weights at {}",
            npz_path.display()
        );
        match convert(&npz_path, &bin_path) {
            Ok(()) => true,
            Err(e) => panic!("failed to convert OGSOV weights at {}: {e}", npz_path.display()),
        }
    } else {
        println!(
            "cargo:warning=OGSOV weights not found at {} — building without embedded model",
            npz_path.display()
        );
        fs::write(&bin_path, []).unwrap();
        false
    };

    fs::write(
        &gen_path,
        format!("/// Whether OGSOV weights were embedded at build time.\npub const OGSOV_EMBEDDED: bool = {embedded};\n"),
    )
    .unwrap();
}

type Zip = zip::ZipArchive<std::io::BufReader<fs::File>>;

fn open_member(npz: &mut Zip, name: &str) -> Result<npyz::NpyFile<std::io::Cursor<Vec<u8>>>, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut member = npz
        .by_name(&format!("{name}.npy"))
        .map_err(|e| format!("{name} missing from npz: {e}"))?;
    let mut buf = Vec::with_capacity(member.size() as usize);
    member.read_to_end(&mut buf)?;
    Ok(npyz::NpyFile::new(std::io::Cursor::new(buf))?)
}

fn convert(npz_path: &Path, bin_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut npz: Zip = zip::ZipArchive::new(std::io::BufReader::new(fs::File::open(npz_path)?))?;
    let mut out: Vec<u8> = Vec::with_capacity(MASK_ELEMS / 8 + 5 * 280_065 * 4);

    // --- mask_lookup -> bit-packed, MSB-first (np.packbits convention) ---
    let mask = read_mask(&mut npz)?;
    if mask.len() != MASK_ELEMS {
        return Err(format!("mask_lookup has {} elements, expected 256^3", mask.len()).into());
    }
    for chunk in mask.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            if bit {
                byte |= 1 << (7 - i);
            }
        }
        out.push(byte);
    }

    // --- classifier weights ---
    for clf in 0..5 {
        for (layer, rows, cols) in LAYER_SHAPES {
            append_f32s(&mut npz, &format!("{clf}.{layer}.weight"), &[rows as u64, cols as u64], &mut out)?;
            append_f32s(&mut npz, &format!("{clf}.{layer}.bias"), &[rows as u64], &mut out)?;
        }
    }

    fs::write(bin_path, &out)?;
    Ok(())
}

fn read_mask(npz: &mut Zip) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
    let arr = open_member(npz, "mask_lookup")?;
    if arr.shape() != [256, 256, 256] {
        return Err(format!("mask_lookup shape {:?}, expected [256,256,256]", arr.shape()).into());
    }
    if arr.order() != npyz::Order::C {
        return Err("mask_lookup is not C-order".into());
    }
    // Stored dtype may be bool ('|b1') or an integer type — handle both.
    match arr.dtype() {
        npyz::DType::Plain(ty) if ty.to_string().ends_with("b1") => Ok(arr.into_vec::<bool>()?),
        npyz::DType::Plain(ty) if ty.to_string().ends_with("u1") || ty.to_string().ends_with("i1") => {
            Ok(arr.into_vec::<u8>()?.into_iter().map(|v| v != 0).collect())
        }
        other => Err(format!("unsupported mask_lookup dtype: {other:?}").into()),
    }
}

fn append_f32s(
    npz: &mut Zip,
    name: &str,
    shape: &[u64],
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let arr = open_member(npz, name)?;
    if arr.shape() != shape {
        return Err(format!("{name}: shape {:?}, expected {shape:?}", arr.shape()).into());
    }
    if arr.order() != npyz::Order::C {
        return Err(format!("{name} is not C-order").into());
    }
    // Accept f32 or f64 (cast down).
    let values: Vec<f32> = match arr.dtype() {
        npyz::DType::Plain(ty) if ty.to_string().ends_with("f4") => arr.into_vec::<f32>()?,
        npyz::DType::Plain(ty) if ty.to_string().ends_with("f8") => {
            arr.into_vec::<f64>()?.into_iter().map(|v| v as f32).collect()
        }
        other => return Err(format!("{name}: unsupported dtype {other:?}").into()),
    };
    out.reserve(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

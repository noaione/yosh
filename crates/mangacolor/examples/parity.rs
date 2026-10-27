//! Parity harness: classify a raw interleaved RGB file and print the answers.
//!
//! Usage: `cargo run -p mangacolor --example parity -- <file.rgb> <width> <height>`
//!
//! The Python references in `../vs-mangacolor/tests/references/` consume BGR;
//! pass the same pixels to both (this side expects RGB, the reference BGR).

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("raw file path");
    let width: usize = args.next().expect("width").parse().expect("width");
    let height: usize = args.next().expect("height").parse().expect("height");
    let rgb = std::fs::read(&path).expect("read");
    assert_eq!(rgb.len(), width * height * 3, "file size does not match");

    let ogsov = mangacolor::detect_ogsov(&rgb, width, height, 3).expect("ogsov");
    println!(
        "ogsov is_color={} score={:.12} confidence={:.12} fast_gray={}",
        ogsov.is_color, ogsov.score, ogsov.confidence, ogsov.fast_gray
    );
    let ogsox = mangacolor::detect_ogsox(&rgb, width, height, 3).expect("ogsox");
    println!(
        "ogsox is_color={} score={:.12} confidence={:.12} fast_gray={}",
        ogsox.is_color, ogsox.score, ogsox.confidence, ogsox.fast_gray
    );
}

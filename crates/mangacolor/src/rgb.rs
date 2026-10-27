//! Borrowed 8-bit plane views with explicit strides, plus the interleaved
//! adapter.
//!
//! Decoders may hand back a row stride wider than the row, and the detection
//! core must never read past a row's width.  A [`PlaneView`] is a borrowed
//! window over one contiguous plane; [`RgbView`] groups the three planes the
//! models consume.  Nothing here owns a frame, so the same views feed tests and
//! hosts alike.
//!
//! Interleaved RGB/RGBA input (what yosh's page decoders produce) has no
//! contiguous row per channel.  [`PlanarScratch`] de-interleaves it once into
//! reused planes and hands back an [`RgbView`], so the models run on exactly
//! one layout.

use crate::error::{Error, Result};

/// One borrowed 8-bit plane.
#[derive(Debug, Clone, Copy)]
pub struct PlaneView<'a> {
    data: &'a [u8],
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> PlaneView<'a> {
    /// Wrap a plane, checking that every addressed byte is inside `data`.
    ///
    /// `stride` is the distance between the start of two rows, which may be
    /// larger than `width`; the bytes in that gap are never read.
    pub fn new(data: &'a [u8], stride: usize, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::input(format!(
                "plane dimensions must be positive, got {width}x{height}"
            )));
        }
        if stride < width {
            return Err(Error::input(format!(
                "plane stride {stride} is narrower than its width {width}"
            )));
        }
        // Only the bytes actually addressed are required: the last row needs
        // its width, not a whole further stride.
        let last_row =
            (height - 1)
                .checked_mul(stride)
                .ok_or_else(|| Error::DimensionOverflow {
                    detail: format!("plane stride {stride} x height {height} overflows usize"),
                })?;
        let needed = last_row
            .checked_add(width)
            .ok_or_else(|| Error::DimensionOverflow {
                detail: format!("plane row offset {last_row} + width {width} overflows usize"),
            })?;
        if data.len() < needed {
            return Err(Error::input(format!(
                "plane needs {needed} bytes for stride {stride}, width {width} and height \
                 {height}, but only {} were provided",
                data.len()
            )));
        }
        Ok(PlaneView {
            data,
            stride,
            width,
            height,
        })
    }

    /// The plane's visible width in pixels.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The plane's visible height in pixels.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The distance between two rows in bytes.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The visible bytes of one row; the padding beyond `width` is excluded.
    #[inline]
    pub fn row(&self, y: usize) -> &'a [u8] {
        debug_assert!(y < self.height);
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// One pixel, with a bounds check in debug builds.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        debug_assert!(x < self.width && y < self.height);
        self.data[y * self.stride + x]
    }
}

/// The three colour planes of an 8-bit RGB frame, in R, G, B order.
#[derive(Debug, Clone, Copy)]
pub struct RgbView<'a> {
    /// The red plane.
    pub red: PlaneView<'a>,
    /// The green plane.
    pub green: PlaneView<'a>,
    /// The blue plane.
    pub blue: PlaneView<'a>,
}

impl<'a> RgbView<'a> {
    /// Group three planes, rejecting mismatched dimensions.
    pub fn new(red: PlaneView<'a>, green: PlaneView<'a>, blue: PlaneView<'a>) -> Result<Self> {
        let (w, h) = (red.width(), red.height());
        for (name, plane) in [("green", green), ("blue", blue)] {
            if plane.width() != w || plane.height() != h {
                return Err(Error::input(format!(
                    "plane {name} is {}x{} but the red plane is {w}x{h}",
                    plane.width(),
                    plane.height()
                )));
            }
        }
        Ok(RgbView { red, green, blue })
    }

    /// Visible width in pixels.
    #[inline]
    pub fn width(&self) -> usize {
        self.red.width()
    }

    /// Visible height in pixels.
    #[inline]
    pub fn height(&self) -> usize {
        self.red.height()
    }

    /// Total visible pixels, checked for overflow.
    pub fn pixel_count(&self) -> Result<u64> {
        let width = self.width() as u64;
        let height = self.height() as u64;
        width
            .checked_mul(height)
            .ok_or_else(|| Error::DimensionOverflow {
                detail: format!("{width}x{height} pixels overflows u64"),
            })
    }
}

/// Owned planar buffers for interleaved input.
///
/// The models read three separate channel rows; a decoder that produces
/// interleaved RGB or RGBA (which is what yosh's page decoders return) can
/// adopt a frame here. The pixels are copied into these three planes once,
/// the buffers are reused for the next frame, and every kernel then runs on
/// exactly the layout it was ported against.
#[derive(Debug, Default)]
pub struct PlanarScratch {
    red: Vec<u8>,
    green: Vec<u8>,
    blue: Vec<u8>,
    width: usize,
    height: usize,
    /// Whether every pixel adopted so far has `r == g == b`.
    gray: bool,
}

impl PlanarScratch {
    /// Empty buffers.
    pub fn new() -> Self {
        Self::default()
    }

    /// De-interleave `pixels` (3 or 4 bytes per pixel, row-major) into the
    /// three planes.  Read them back with [`Self::view`], and ask
    /// [`Self::is_gray`] about the frame that was just adopted.
    ///
    /// The alpha byte of an RGBA frame is ignored: every model consumes only
    /// red, green and blue. A caller decides whether a transparent frame is
    /// eligible for detection at all.
    pub fn adopt(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Result<()> {
        if channels != 3 && channels != 4 {
            return Err(Error::input(format!(
                "interleaved pixels must have 3 or 4 channels, got {channels}"
            )));
        }
        if width == 0 || height == 0 {
            return Err(Error::input(format!(
                "frame dimensions must be positive, got {width}x{height}"
            )));
        }
        let count = width
            .checked_mul(height)
            .ok_or_else(|| Error::DimensionOverflow {
                detail: format!("{width}x{height} pixels overflows usize"),
            })?;
        let needed = count
            .checked_mul(channels)
            .ok_or_else(|| Error::DimensionOverflow {
                detail: format!("{count} pixels x {channels} channels overflows usize"),
            })?;
        if pixels.len() != needed {
            return Err(Error::input(format!(
                "a {width}x{height} frame of {channels} channels needs {needed} bytes, but {} were provided",
                pixels.len()
            )));
        }

        self.width = width;
        self.height = height;
        self.red.resize(count, 0);
        self.green.resize(count, 0);
        self.blue.resize(count, 0);
        // The gray question is answered for free by the split: every pixel is
        // already in registers, so a host can skip the model entirely for an
        // exact-gray frame (the answer is certain, and the model would only
        // re-derive it).
        self.gray = split_channels(
            pixels,
            channels,
            count,
            &mut self.red,
            &mut self.green,
            &mut self.blue,
        );
        Ok(())
    }

    /// A view over the planes as they currently stand.
    pub fn view(&self) -> Result<RgbView<'_>> {
        RgbView::new(
            PlaneView::new(&self.red, self.width, self.width, self.height)?,
            PlaneView::new(&self.green, self.width, self.width, self.height)?,
            PlaneView::new(&self.blue, self.width, self.width, self.height)?,
        )
    }

    /// Whether every pixel of the frame most recently adopted is exactly gray
    /// (`r == g == b`).
    ///
    /// Such a frame is certainly not a colour page, so a caller can answer
    /// without running a model; see `OgsovPrediction::fast_gray`.
    pub fn is_gray(&self) -> bool {
        self.gray
    }

    /// Visible width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Visible height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Bytes the three planes hold.
    pub fn resident_bytes(&self) -> usize {
        self.red.capacity() + self.green.capacity() + self.blue.capacity()
    }
}

/// Split interleaved pixels into the three planes, reporting whether every
/// pixel was exactly gray (`r == g == b`).
///
/// The channel split is the one pass every caller pays, so aarch64 does it with
/// the `vld3q`/`vld4q` structure loads (sixteen pixels per instruction, and the
/// gray comparison rides along in vector registers); elsewhere the portable
/// loop below is the definition.
fn split_channels(
    pixels: &[u8],
    channels: usize,
    count: usize,
    red: &mut [u8],
    green: &mut [u8],
    blue: &mut [u8],
) -> bool {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline, and the caller checked that
    // `pixels` holds `count * channels` bytes while the planes hold `count`.
    return unsafe { neon::split(pixels, channels, count, red, green, blue) };
    #[cfg(not(target_arch = "aarch64"))]
    scalar::split(pixels, channels, count, red, green, blue)
}

/// The portable split, which every other path is measured against.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
mod scalar {
    pub(super) fn split(
        pixels: &[u8],
        channels: usize,
        count: usize,
        red: &mut [u8],
        green: &mut [u8],
        blue: &mut [u8],
    ) -> bool {
        let mut gray = true;
        for (index, pixel) in pixels.chunks_exact(channels).take(count).enumerate() {
            let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
            red[index] = r;
            green[index] = g;
            blue[index] = b;
            gray &= r == g && g == b;
        }
        gray
    }
}

/// The aarch64 split: structure loads de-interleave sixteen pixels at a time.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Whether one vector's pixels are all exactly gray.
    ///
    /// # Safety
    ///
    /// The three vectors are live NEON registers (the type is a plain value
    /// here, so no pointer is dereferenced).
    #[inline]
    unsafe fn all_gray(red: uint8x16_t, green: uint8x16_t, blue: uint8x16_t) -> bool {
        unsafe {
            let matches = vandq_u8(vceqq_u8(red, green), vceqq_u8(green, blue));
            vminvq_u8(matches) == 0xFF
        }
    }

    /// Split the interleaved pixels; see the module caller.
    ///
    /// # Safety
    ///
    /// `pixels` holds `count * channels` bytes and each plane holds `count`.
    pub(super) unsafe fn split(
        pixels: &[u8],
        channels: usize,
        count: usize,
        red: &mut [u8],
        green: &mut [u8],
        blue: &mut [u8],
    ) -> bool {
        unsafe {
            let mut index = 0usize;
            let mut gray = true;
            if channels == 4 {
                while index + 16 <= count {
                    // Forty-eight structure loads read sixty-four bytes: exactly
                    // sixteen RGBA pixels.
                    let block = vld4q_u8(pixels.as_ptr().add(index * 4));
                    vst1q_u8(red.as_mut_ptr().add(index), block.0);
                    vst1q_u8(green.as_mut_ptr().add(index), block.1);
                    vst1q_u8(blue.as_mut_ptr().add(index), block.2);
                    gray &= all_gray(block.0, block.1, block.2);
                    index += 16;
                }
            } else {
                while index + 16 <= count {
                    // Forty-eight bytes: exactly sixteen RGB pixels.
                    let block = vld3q_u8(pixels.as_ptr().add(index * 3));
                    vst1q_u8(red.as_mut_ptr().add(index), block.0);
                    vst1q_u8(green.as_mut_ptr().add(index), block.1);
                    vst1q_u8(blue.as_mut_ptr().add(index), block.2);
                    gray &= all_gray(block.0, block.1, block.2);
                    index += 16;
                }
            }
            // A tail shorter than one vector goes one pixel at a time.
            while index < count {
                let base = index * channels;
                let (r, g, b) = (
                    *pixels.get_unchecked(base),
                    *pixels.get_unchecked(base + 1),
                    *pixels.get_unchecked(base + 2),
                );
                *red.get_unchecked_mut(index) = r;
                *green.get_unchecked_mut(index) = g;
                *blue.get_unchecked_mut(index) = b;
                gray &= r == g && g == b;
                index += 1;
            }
            gray
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three tiny planar planes laid out contiguously, as the corpus stores
    /// them: two pixels of red (1, 4), green (2, 5) and blue (3, 6).
    const PLANES: [u8; 6] = [1, 4, 2, 5, 3, 6];

    fn view<'a>(data: &'a [u8; 6]) -> RgbView<'a> {
        RgbView::new(
            PlaneView::new(&data[0..2], 2, 2, 1).expect("red"),
            PlaneView::new(&data[2..4], 2, 2, 1).expect("green"),
            PlaneView::new(&data[4..6], 2, 2, 1).expect("blue"),
        )
        .expect("rgb view")
    }

    #[test]
    fn planar_views_read_the_right_channels() {
        let view = view(&PLANES);
        assert_eq!(view.red.get(0, 0), 1);
        assert_eq!(view.green.get(0, 0), 2);
        assert_eq!(view.blue.get(0, 0), 3);
        assert_eq!(view.red.get(1, 0), 4);
        assert_eq!(view.green.get(1, 0), 5);
        assert_eq!(view.blue.get(1, 0), 6);
        assert_eq!(view.pixel_count().expect("count"), 2);
    }

    #[test]
    fn a_short_plane_is_rejected() {
        let data = [0u8; 16];
        let error = PlaneView::new(&data, 8, 8, 3).expect_err("must reject");
        assert!(matches!(error, Error::UnsupportedInput { .. }));
    }

    #[test]
    fn a_narrow_stride_is_rejected() {
        let data = [0u8; 16];
        let error = PlaneView::new(&data, 2, 4, 2).expect_err("must reject");
        assert!(matches!(error, Error::UnsupportedInput { .. }));
    }

    #[test]
    fn mismatched_planes_are_rejected() {
        let data = [0u8; 16];
        let plane = PlaneView::new(&data, 4, 4, 4).expect("plane");
        let short = PlaneView::new(&data, 4, 2, 4).expect("plane");
        let error = RgbView::new(plane, short, plane).expect_err("must reject");
        assert!(matches!(error, Error::UnsupportedInput { .. }));
    }

    #[test]
    fn padded_rows_are_not_read() {
        let mut data = vec![0u8; 3 * 4];
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = index as u8;
        }
        let plane = PlaneView::new(&data, 4, 2, 3).expect("plane");
        assert_eq!(plane.row(0), &[0, 1]);
        assert_eq!(plane.row(1), &[4, 5]);
        assert_eq!(plane.row(2), &[8, 9]);
    }

    #[test]
    fn the_last_row_needs_only_its_width() {
        // Two rows of four, strided to six: 6 + 4 = 10 bytes suffice.
        let data = [0u8; 10];
        assert!(PlaneView::new(&data, 6, 4, 2).is_ok());
        let error = PlaneView::new(&data, 6, 5, 2).expect_err("must reject");
        assert!(matches!(error, Error::UnsupportedInput { .. }));
    }

    #[test]
    fn the_neon_split_matches_the_scalar_one() {
        #[cfg(target_arch = "aarch64")]
        {
            for channels in [3usize, 4] {
                for (count, seed) in [
                    (0usize, 1u64),
                    (1, 2),
                    (15, 3),
                    (16, 4),
                    (17, 5),
                    (31, 6),
                    (32, 7),
                    (33, 8),
                    (100, 9),
                    (257, 10),
                ] {
                    let mut state = seed;
                    let mut pixels = Vec::with_capacity(count * channels);
                    for index in 0..count {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let r = state as u8;
                        let g = state.rotate_left(17) as u8;
                        // Every third pixel is exact gray, so both paths see one.
                        let b = if index % 3 == 0 {
                            r
                        } else {
                            state.rotate_left(41) as u8
                        };
                        let g = if index % 3 == 0 { r } else { g };
                        pixels.extend_from_slice(&[r, g, b]);
                        if channels == 4 {
                            pixels.push(255);
                        }
                    }
                    let mut want_red = vec![0u8; count];
                    let mut want_green = vec![0u8; count];
                    let mut want_blue = vec![0u8; count];
                    let mut got_red = vec![0u8; count];
                    let mut got_green = vec![0u8; count];
                    let mut got_blue = vec![0u8; count];
                    let want = super::scalar::split(
                        &pixels,
                        channels,
                        count,
                        &mut want_red,
                        &mut want_green,
                        &mut want_blue,
                    );
                    // SAFETY: the buffers hold exactly `count` pixels x `channels`.
                    let got = unsafe {
                        super::neon::split(
                            &pixels,
                            channels,
                            count,
                            &mut got_red,
                            &mut got_green,
                            &mut got_blue,
                        )
                    };
                    assert_eq!(got, want, "{channels} channels, {count} pixels");
                    assert_eq!(
                        got_red, want_red,
                        "red, {channels} channels, {count} pixels"
                    );
                    assert_eq!(
                        got_green, want_green,
                        "green, {channels} channels, {count} pixels"
                    );
                    assert_eq!(
                        got_blue, want_blue,
                        "blue, {channels} channels, {count} pixels"
                    );
                }
            }
        }
    }

    #[test]
    fn adopting_an_exact_gray_frame_is_detected() {
        let mut planes = PlanarScratch::new();
        let rgba: Vec<u8> = (0..4 * 17)
            .flat_map(|index| {
                let value = (index * 13) as u8;
                [value, value, value, 255]
            })
            .collect();
        planes.adopt(&rgba, 17, 4, 4).expect("gray frame");
        assert!(planes.is_gray());
        planes
            .adopt(&[1, 2, 3, 255], 1, 1, 4)
            .expect("colour frame");
        assert!(!planes.is_gray());
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        let data = [0u8; 4];
        assert!(PlaneView::new(&data, 4, 0, 1).is_err());
        assert!(PlaneView::new(&data, 4, 4, 0).is_err());
    }
}

#!/usr/bin/env python3
# ruff: file-ignore[os-path-dirname, os-path-join, os-path-abspath, builtin-open, os-remove, os-path-exists]
"""Generate the CMYK test fixtures under this directory.

Requirements:
  - Pillow (with ImageCms / lcms2 bindings) for the JPEG fixtures + references
  - the libjxl `cjxl` CLI on PATH for the JXL fixtures
  - gcc + libjpeg-turbo headers/libs for the YCCK JPEG fixture (see _ycck_enc.c)

Fixture provenance:
  - USWebCoatedSWOP.icc is Adobe's classic v2 CMYK output profile (the ICC's
    standard reference CMYK profile, freely redistributable). This copy came
    from the Windows color directory (C:\\Windows\\System32\\spool\\drivers\\color).
  - cmyk_ycck.jpg is written by _ycck_enc.c (libjpeg-turbo, JCS_YCCK + Adobe
    APP14 transform=2) from CMYK patches converted to YCbCr+K here.
  - All images are synthetic patch grids generated here; no copyrighted content.

Regenerate everything with:  python make_fixtures.py
"""
import os
import shutil
import subprocess

from PIL import Image, ImageCms

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, "USWebCoatedSWOP.icc")
W = 64
H = 64

# 4x4 grid of CMYK patches (conventional CMYK: 0 = no ink, 255 = full ink).
PATCHES = [
    (0, 0, 0, 0),         # white
    (0, 0, 0, 255),       # black
    (255, 0, 0, 0),       # cyan
    (0, 255, 0, 0),       # magenta
    (0, 0, 255, 0),       # yellow
    (255, 255, 0, 0),     # red-ish (M+Y)
    (255, 0, 255, 0),     # green-ish (C+Y)
    (0, 255, 255, 0),     # blue-ish (C+M)
    (64, 48, 48, 0),      # neutral-ish light gray (C=M=Y, no K)
    (128, 96, 96, 0),     # mid gray
    (192, 144, 144, 0),   # dark gray
    (32, 32, 32, 160),    # gray with black generation
    (255, 192, 64, 32),   # warm tone
    (64, 128, 192, 48),   # cool tone
    (16, 240, 32, 0),     # saturated mix
    (200, 200, 200, 40),  # light neutral with K
]

# A near-neutral page (SWOP's K-only axis is its gray axis) for the OGSOV
# gray-routing tests: an 8-step black-generation ramp.
NEUTRAL = [(0, 0, 0, k) for k in (0, 32, 64, 96, 128, 160, 192, 224)]


def patch_grid(patches, w=W, h=H):
    cols = 4
    # rows = (len(patches) + cols - 1) // cols
    out = []
    for y in range(h):
        for x in range(w):
            i = (y * cols // h) * cols + (x * cols // w)
            out.append(patches[min(i, len(patches) - 1)])
    return out


def write_pam(path, cmyk_pixels, alpha=None):
    """Write a PAM file for cjxl.

    libjxl's PNM parser has no CMYK tuple type: CMYK is modeled as three
    color channels plus a "Black" extra channel (and "Alpha" for CMYKA), so
    those TUPLTYPE lines are what cjxl accepts. The samples are stored
    INVERTED (0 = full ink): that is the JPEG XL internal convention — the
    decoder flips them back to conventional CMYK (0 = no ink) before the CMS.
    Alpha is stored normally.
    """
    depth = 5 if alpha is not None else 4
    tupl = b"TUPLTYPE Black\n" + (b"TUPLTYPE Alpha\n" if alpha is not None else b"")
    with open(path, "wb") as f:
        f.write(b"P7\n")
        f.write(f"WIDTH {W}\n".encode())
        f.write(f"HEIGHT {H}\n".encode())
        f.write(f"DEPTH {depth}\n".encode())
        f.write(b"MAXVAL 255\n")
        f.write(tupl)
        f.write(b"ENDHDR\n")
        for i, (c, m, y, k) in enumerate(cmyk_pixels):
            f.write(bytes((255 - c, 255 - m, 255 - y, 255 - k)))
            if alpha is not None:
                f.write(bytes((alpha[i],)))


def jpeg_cmyk(path, cmyk_pixels, icc=True):
    """CMYK JPEG via Pillow (optionally with the embedded SWOP profile)."""
    img = Image.new("CMYK", (W, H))
    img.putdata([(c, m, y, k) for (c, m, y, k) in cmyk_pixels])
    kwargs = {}
    if icc:
        kwargs["icc_profile"] = open(PROFILE, "rb").read()
    img.save(path, "JPEG", quality=95, subsampling=0, **kwargs)
    return img


def jpeg_ycck(path):
    """True Adobe YCCK JPEG via _ycck_enc.c (libjpeg-turbo, APP14 transform=2)."""
    c_src = os.path.join(HERE, "_ycck_enc.c")
    exe = os.path.join(HERE, "_ycck_enc.exe")
    if not os.path.exists(exe):
        if not shutil.which("gcc"):
            print("!! gcc not found — skipping the YCCK fixture")
            return
        subprocess.run(
            ["gcc", "-O2", "-I", os.environ.get("LIBJPEG_INC", ""), c_src,
             "-L", os.environ.get("LIBJPEG_LIB", ""), "-ljpeg", "-o", exe],
            check=True,
        )

    # CMYK patches -> RGB (generic model) -> YCbCr; K is stored inverted
    # (255 - ink), the Adobe YCCK convention the APP14 transform=2 marker
    # promises (0 = 100% ink).
    def cmyk_to_rgb(c, m, y, k):
        return ((255 - c) * (255 - k) / 255 / 255 * 255,
                (255 - m) * (255 - k) / 255 / 255 * 255,
                (255 - y) * (255 - k) / 255 / 255 * 255)

    def rgb_to_ycbcr(r, g, b):
        return (0.299 * r + 0.587 * g + 0.114 * b,
                128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
                128 + 0.5 * r - 0.418688 * g - 0.081312 * b)

    def cl(v):
        return max(0, min(255, round(v)))

    raw = bytearray()
    for c, m, y, k in patch_grid(PATCHES):
        r, g, b = cmyk_to_rgb(c, m, y, k)
        yv, cb, cr = rgb_to_ycbcr(r, g, b)
        raw += bytes((cl(yv), cl(cb), cl(cr), 255 - k))
    with open(os.path.join(HERE, "_ycck.raw"), "wb") as f:
        f.write(bytes(raw))
    subprocess.run([exe, path, "ycck"], stdin=open(os.path.join(HERE, "_ycck.raw"), "rb"), check=True)
    os.remove(os.path.join(HERE, "_ycck.raw"))

    # US-convention CMYK JPEG (no Adobe marker, conventional ink samples) via
    # the same encoder in `cmyk` mode — libjpeg always writes an APP14 "Adobe"
    # marker for JCS_CMYK, so strip that segment afterwards (marker surgery is
    # safe: APP14 is a standalone segment between SOI and SOS).
    raw = bytearray()
    for c, m, y, k in patch_grid(PATCHES):
        raw += bytes((c, m, y, k))
    us_path = os.path.join(HERE, "cmyk_us.jpg")
    with open(os.path.join(HERE, "_cmyk_us.raw"), "wb") as f:
        f.write(bytes(raw))
    subprocess.run(
        [exe, us_path, "cmyk"],
        stdin=open(os.path.join(HERE, "_cmyk_us.raw"), "rb"),
        check=True,
    )
    os.remove(os.path.join(HERE, "_cmyk_us.raw"))
    data = open(us_path, "rb").read()
    out = bytearray(data[:2])
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            raise SystemExit("cmyk_us.jpg: lost marker sync")
        m = data[i + 1]
        if m == 0xDA:  # SOS starts entropy data — copy the rest verbatim.
            out += data[i:]
            break
        seg = 2 + ((data[i + 2] << 8) | data[i + 3])
        if not (m == 0xEE and data[i + 4 : i + 9] == b"Adobe"):
            out += data[i : i + seg]
        i += seg
    open(us_path, "wb").write(bytes(out))


def main():
    grid = patch_grid(PATCHES)
    neutral = patch_grid(NEUTRAL)

    # Untagged + SWOP-tagged CMYK JPEG.
    jpeg_cmyk(os.path.join(HERE, "cmyk_untagged.jpg"), grid, icc=False)
    jpeg_cmyk(os.path.join(HERE, "cmyk_icc.jpg"), grid, icc=True)
    jpeg_cmyk(os.path.join(HERE, "cmyk_neutral.jpg"), neutral, icc=True)
    jpeg_ycck(os.path.join(HERE, "cmyk_ycck.jpg"))

    # Plain-format source files for the JXL fixtures.
    with open(os.path.join(HERE, "_gray.pgm"), "wb") as f:
        f.write(b"P5\n64 64\n255\n")
        f.write(bytes([128]) * (W * H))
    with open(os.path.join(HERE, "_graya.pam"), "wb") as f:
        f.write(b"P7\nWIDTH 64\nHEIGHT 64\nDEPTH 2\nMAXVAL 255\nTUPLTYPE GRAYSCALE_ALPHA\nENDHDR\n")
        for i in range(W * H):
            f.write(bytes((128, 255 if i % 2 else 0)))
    with open(os.path.join(HERE, "_rgb.ppm"), "wb") as f:
        f.write(b"P6\n64 64\n255\n")
        for i in range(W * H):
            f.write(bytes((0, 0, 255)))
    with open(os.path.join(HERE, "_rgba.pam"), "wb") as f:
        f.write(b"P7\nWIDTH 64\nHEIGHT 64\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n")
        for i in range(W * H):
            f.write(bytes((255, 0, 0, 255 if i % 2 else 0)))

    # CMYK / CMYKA JXL via cjxl (PAM in, SWOP profile attached).
    write_pam(os.path.join(HERE, "_cmyk.pam"), grid)
    write_pam(
        os.path.join(HERE, "_cmyka.pam"),
        grid,
        alpha=[0 if i % 16 == 0 else 255 for i in range(W * H)],
    )
    subprocess.run(
        [
            "cjxl",
            os.path.join(HERE, "_cmyk.pam"),
            os.path.join(HERE, "cmyk.jxl"),
            "-x",
            "icc_pathname=" + PROFILE,
            "-d",
            "0",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cjxl",
            os.path.join(HERE, "_cmyka.pam"),
            os.path.join(HERE, "cmyka.jxl"),
            "-x",
            "icc_pathname=" + PROFILE,
            "-d",
            "0",
        ],
        check=True,
    )
    os.remove(os.path.join(HERE, "_cmyk.pam"))
    os.remove(os.path.join(HERE, "_cmyka.pam"))

    # Plain-format JXL fixtures (gray/graya/rgb/rgba) for the unchanged-path
    # tests. Lossy (-d 1) keeps them tiny; flat patches stay visually exact.
    subprocess.run(
        ["cjxl", os.path.join(HERE, "_gray.pgm"), os.path.join(HERE, "gray.jxl"), "-d", "1"],
        check=True,
    )
    subprocess.run(
        ["cjxl", os.path.join(HERE, "_graya.pam"), os.path.join(HERE, "graya.jxl"), "-d", "1"],
        check=True,
    )
    subprocess.run(
        ["cjxl", os.path.join(HERE, "_rgb.ppm"), os.path.join(HERE, "rgb.jxl"), "-d", "1"],
        check=True,
    )
    subprocess.run(
        ["cjxl", os.path.join(HERE, "_rgba.pam"), os.path.join(HERE, "rgba.jxl"), "-d", "1"],
        check=True,
    )
    os.remove(os.path.join(HERE, "_gray.pgm"))
    os.remove(os.path.join(HERE, "_graya.pam"))
    os.remove(os.path.join(HERE, "_rgb.ppm"))
    os.remove(os.path.join(HERE, "_rgba.pam"))

    # Print lcms2 (Pillow ImageCms) reference conversions of the SWOP-tagged
    # patches to sRGB — the expected values for the qcms tests (tolerance used
    # because qcms and lcms differ in interpolation).
    profile = ImageCms.ImageCmsProfile(PROFILE)
    rgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    print("=== lcms2 CMYK->sRGB references (USWebCoatedSWOP) ===")
    img = Image.new("CMYK", (W, H))
    img.putdata([tuple(q) for q in grid])
    transform = ImageCms.buildTransform(
        profile, rgb_profile, "CMYK", "RGB",
        renderingIntent=ImageCms.Intent.PERCEPTUAL,
    )
    conv = ImageCms.applyTransform(img, transform).convert("RGB")  # type: ignore
    cols = 4
    for i in range(len(PATCHES)):
        x = (i % cols) * (W // cols) + (W // cols) // 2
        y = (i // cols) * (H // cols) + (H // cols) // 2
        print(f"  {i:2d}  cmyk={PATCHES[i]!s:24s}  srgb={conv.getpixel((x, y))}")


if __name__ == "__main__":
    main()

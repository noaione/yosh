/* One-off fixture generator for 4-component CMYK JPEGs from raw 64x64x4
 * samples on stdin, built with libjpeg-turbo:
 *   gcc -O2 -I<inc> ycck_enc.c -L<lib> -ljpeg
 * Usage: ycck_enc OUT.jpg MODE < raw
 *   MODE = ycck: Adobe YCCK (APP14 transform=2). Samples are YCbCr + K, with
 *          K ALREADY INVERTED (255 - ink) per the Adobe convention the marker
 *          promises (0 = 100% ink); YCbCr is stored as-is.
 *   MODE = cmyk: US-convention CMYK (no Adobe marker, 0 = no ink). Samples
 *          are conventional CMYK ink; libjpeg stores them as-is.
 * Only needed because no readily available CLI writes these files. */
#include <stdio.h>
#include <string.h>
#include <jpeglib.h>
#include <jerror.h>

#define W 64
#define H 64

int main(int argc, char** argv) {
    if (argc != 3 || (strcmp(argv[2], "ycck") != 0 && strcmp(argv[2], "cmyk") != 0)) {
        fprintf(stderr, "usage: ycck_enc OUT.jpg ycck|cmyk\n");
        return 1;
    }
    int ycck = strcmp(argv[2], "ycck") == 0;
    unsigned char raw[W * H * 4];
    unsigned char rows[W * 4];
    if (fread(raw, 1, sizeof(raw), stdin) != sizeof(raw)) {
        fprintf(stderr, "need %zu bytes\n", sizeof(raw));
        return 1;
    }
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    FILE* out = fopen(argv[1], "wb");
    if (!out) { perror("fopen"); return 1; }
    jpeg_stdio_dest(&cinfo, out);
    cinfo.image_width = W;
    cinfo.image_height = H;
    cinfo.input_components = 4;
    cinfo.in_color_space = ycck ? JCS_YCCK : JCS_CMYK;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 95, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    if (ycck) {
        /* Adobe APP14 marker with transform=2 (YCCK). */
        unsigned char adobe[12] = { 'A','d','o','b','e', 0,100, 0,0, 0,0, 2 };
        jpeg_write_marker(&cinfo, JPEG_APP0 + 14, adobe, 12);
    }
    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned y = cinfo.next_scanline;
        for (unsigned x = 0; x < W; x++) {
            rows[x * 4 + 0] = raw[(y * W + x) * 4 + 0];
            rows[x * 4 + 1] = raw[(y * W + x) * 4 + 1];
            rows[x * 4 + 2] = raw[(y * W + x) * 4 + 2];
            rows[x * 4 + 3] = raw[(y * W + x) * 4 + 3];
        }
        JSAMPROW row = rows;
        jpeg_write_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(out);
    return 0;
}

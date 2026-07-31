/* compare_dequant.c — Dequantize a tensor using ggml and print first values.
 * Compile:
 *   gcc -O2 -o compare_dequant compare_dequant.c \
 *     -I/home/ksante/.local/lib/python3.12/site-packages/include \
 *     -L/home/ksante/.local/lib/python3.12/site-packages/lib \
 *     -l:libggml-base.so.0 -lm -Wl,-rpath,/home/ksante/.local/lib/python3.12/site-packages/lib
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ggml.h"
#include "gguf.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <tensor_name>\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    const char *tname = argv[2];

    /* Open GGUF */
    struct gguf_init_params gparams = { .no_alloc = true, .ctx = NULL };
    struct gguf_context *ctx = gguf_init_from_file(path, gparams);
    if (!ctx) { fprintf(stderr, "gguf_init failed\n"); return 1; }

    int n_tensors = gguf_get_n_tensors(ctx);
    fprintf(stderr, "n_tensors=%d\n", n_tensors);

    /* Find tensor */
    int tidx = gguf_find_tensor(ctx, tname);
    if (tidx < 0) { fprintf(stderr, "tensor '%s' not found\n", tname); return 1; }

    size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tidx);
    enum ggml_type type = gguf_get_tensor_type(ctx, tidx);
    int64_t ne[4] = {1,1,1,1};
    int n_dims = gguf_get_tensor_n_dims(ctx, tidx);
    for (int i = 0; i < n_dims; i++) ne[i] = gguf_get_tensor_ne(ctx, tidx)[i];

    int64_t n_elems = ne[0] * ne[1] * ne[2] * ne[3];
    fprintf(stderr, "tensor: %s, type=%d, ne=[%ld,%ld,%ld,%ld], n_elems=%ld, offset=%zu\n",
            tname, type, ne[0], ne[1], ne[2], ne[3], n_elems, (size_t)offset);

    /* Read raw bytes */
    size_t nbytes;
    if (ggml_type_size(type) > 0 && !ggml_is_quantized(type)) {
        nbytes = n_elems * ggml_type_size(type);
    } else {
        int block_elems = (type >= GGML_TYPE_Q4_0 && type <= GGML_TYPE_Q8_K) ? 
            (type >= GGML_TYPE_Q2_K ? 256 : 32) : 256;
        int n_blocks = (n_elems + block_elems - 1) / block_elems;
        nbytes = n_blocks * ggml_type_size(type);
    }
    fprintf(stderr, "nbytes=%zu\n", nbytes);

    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, offset, SEEK_SET);
    void *raw = malloc(nbytes);
    fread(raw, 1, nbytes, f);
    fclose(f);

    /* Dequantize using ggml */
    float *f32 = (float *)malloc(n_elems * sizeof(float));
    const struct ggml_type_traits *tt = ggml_get_type_traits(type);
    if (tt && tt->to_float) {
        tt->to_float(raw, f32, n_elems);
    } else {
        fprintf(stderr, "no to_float for type %d\n", type);
        return 1;
    }

    /* Print first 10 values */
    fprintf(stderr, "First 10 dequantized values:\n");
    for (int i = 0; i < 10 && i < n_elems; i++) {
        fprintf(stderr, "  [%d] = %.10f\n", i, f32[i]);
    }

    /* Compute norm */
    double norm = 0;
    for (int64_t i = 0; i < n_elems; i++) norm += (double)f32[i] * f32[i];
    fprintf(stderr, "Norm: %.6f\n", sqrt(norm));

    /* Write f32 to stdout */
    fwrite(f32, sizeof(float), n_elems, stdout);

    free(f32); free(raw);
    gguf_free(ctx);
    return 0;
}

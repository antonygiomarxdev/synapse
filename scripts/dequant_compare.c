/* dequant_compare.c — Use gguf+ggml to dequantize V weight */
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static const char * MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b";

int main() {
    struct gguf_init_params gparams = { .no_alloc = true, .ctx = NULL };
    struct gguf_context * ctx = gguf_init_from_file(MODEL_PATH, gparams);
    if (!ctx) { fprintf(stderr, "gguf_init failed\n"); return 1; }

    int64_t tidx = gguf_find_tensor(ctx, "blk.0.attn_v.weight");
    if (tidx < 0) { fprintf(stderr, "Tensor not found\n"); return 1; }

    enum ggml_type type = gguf_get_tensor_type(ctx, tidx);
    const int64_t * ne = gguf_get_tensor_ne(ctx, tidx);
    size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tidx);
    size_t tsize = gguf_get_tensor_size(ctx, tidx);

    int64_t n = 1;
    int n_dims = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (ne[i] > 1) { n *= ne[i]; n_dims = i + 1; }
    }

    fprintf(stderr, "Type: %d, n_dims: %d\n", type, n_dims);
    for (int i = 0; i < n_dims; i++) fprintf(stderr, "  ne[%d] = %ld\n", i, (long)ne[i]);
    fprintf(stderr, "n_elems: %ld, offset: %zu, tsize: %zu\n", (long)n, offset, tsize);

    // Read raw bytes
    FILE * f = fopen(MODEL_PATH, "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, offset, SEEK_SET);
    std::vector<uint8_t> raw(tsize);
    fread(raw.data(), 1, tsize, f);
    fclose(f);

    // Dequantize using ggml type traits
    const struct ggml_type_traits * tt = ggml_get_type_traits(type);
    if (!tt || !tt->to_float) {
        fprintf(stderr, "No dequant function for type %d\n", type);
        return 1;
    }

    std::vector<float> data(n);
    tt->to_float(raw.data(), data.data(), n);

    fprintf(stderr, "First 10 values:\n");
    for (int i = 0; i < 10; i++) {
        fprintf(stderr, "  [%d] = %.10f\n", i, data[i]);
    }

    double norm = 0;
    for (int64_t i = 0; i < n; i++) norm += (double)data[i] * data[i];
    fprintf(stderr, "Norm: %.6f\n", sqrt(norm));

    // Save to file
    FILE * out = fopen("/tmp/llama_v_weight.bin", "wb");
    fwrite(data.data(), sizeof(float), n, out);
    fclose(out);
    fprintf(stderr, "Saved to /tmp/llama_v_weight.bin\n");

    gguf_free(ctx);
    return 0;
}

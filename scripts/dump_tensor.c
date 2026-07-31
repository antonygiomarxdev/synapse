/* dump_tensor.c — Use llama.cpp C API to dump dequantized tensor values.
 * Compile: gcc -O2 -o dump_tensor dump_tensor.c -L/home/ksante/.local/lib/python3.12/site-packages/lib -lllama -I/home/ksante/.local/lib/python3.12/site-packages/include
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <tensor_name>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *tensor_name = argv[2];

    /* Init backend */
    llama_backend_init();

    /* Load model */
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  /* CPU only */
    
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    /* Get tensor */
    struct ggml_tensor *tensor = llama_model_get_tensor(model, tensor_name);
    if (!tensor) {
        fprintf(stderr, "Tensor '%s' not found\n", tensor_name);
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "Tensor: %s\n", tensor_name);
    fprintf(stderr, "Type: %d\n", tensor->type);
    fprintf(stderr, "Dims: %d\n", ggml_n_dims(tensor));
    for (int i = 0; i < ggml_n_dims(tensor); i++) {
        fprintf(stderr, "  ne[%d] = %ld\n", i, tensor->ne[i]);
    }

    /* Number of elements */
    int64_t n_elems = ggml_nelements(tensor);
    fprintf(stderr, "Elements: %ld\n", n_elems);

    /* Allocate buffer for f32 values */
    float *f32_data = (float *)malloc(n_elems * sizeof(float));
    if (!f32_data) {
        fprintf(stderr, "Failed to allocate\n");
        llama_model_free(model);
        return 1;
    }

    /* Convert to f32 */
    enum ggml_type type = tensor->type;
    size_t type_size = ggml_type_size(type);
    void *data = ggml_get_data(tensor);
    
    /* Use ggml's built-in dequantization */
    const struct ggml_type_traits *traits = ggml_get_type_traits(type);
    if (traits && traits->to_float) {
        traits->to_float(data, f32_data, n_elems);
    } else {
        fprintf(stderr, "No dequantization function for type %d\n", type);
        free(f32_data);
        llama_model_free(model);
        return 1;
    }

    /* Print first 10 values */
    fprintf(stderr, "First 10 values:\n");
    for (int i = 0; i < 10 && i < n_elems; i++) {
        fprintf(stderr, "  [%d] = %.10f\n", i, f32_data[i]);
    }

    /* Print norm */
    double norm = 0.0;
    for (int64_t i = 0; i < n_elems; i++) {
        norm += (double)f32_data[i] * f32_data[i];
    }
    norm = sqrt(norm);
    fprintf(stderr, "Norm: %.6f\n", norm);

    /* Print all values to stdout (binary f32) */
    fwrite(f32_data, sizeof(float), n_elems, stdout);

    free(f32_data);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

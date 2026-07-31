/* dump_hidden_states.cpp — Read hidden states from modified llama.cpp */
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static const char * MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b";

extern ggml_tensor * g_debug_hidden[32];
extern ggml_tensor * g_debug_embd;
extern ggml_tensor * g_debug_attn_norm;
extern ggml_tensor * g_debug_ffn_inp;
extern ggml_tensor * g_debug_ffn_norm;
extern ggml_tensor * g_debug_v_proj;
extern ggml_tensor * g_debug_attn_out_pre;

static void print_tensor(const char * label, ggml_tensor * t, int n = 5) {
    if (!t) { fprintf(stderr, "%s: NULL\n", label); return; }
    int64_t ne = ggml_nelements(t);
    if (ne < n) { fprintf(stderr, "%s: too small (%ld)\n", label, (long)ne); return; }

    std::vector<float> data(n);
    if (t->buffer) {
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
    } else if (t->data) {
        memcpy(data.data(), t->data, n * sizeof(float));
    }

    int64_t norm_n = ne < 1536 ? ne : 1536;
    std::vector<float> norm_data(norm_n);
    if (t->buffer) {
        ggml_backend_tensor_get(t, norm_data.data(), 0, norm_n * sizeof(float));
    } else if (t->data) {
        memcpy(norm_data.data(), t->data, norm_n * sizeof(float));
    }
    double norm = 0;
    for (int64_t i = 0; i < norm_n; i++) norm += (double)norm_data[i] * norm_data[i];
    norm = sqrt(norm);

    fprintf(stderr, "%s: [%.6f, %.6f, %.6f, %.6f, %.6f] norm=%.4f\n",
            label, data[0], data[1], data[2], data[3], data[4], norm);
}

int main(int argc, char ** argv) {
    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    struct llama_model * model = llama_model_load_from_file(MODEL_PATH, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1;
    cparams.n_batch = 1;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    llama_token tokens[] = {49};
    struct llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = 49;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;

    int ret = llama_decode(ctx, batch);
    if (ret != 0) { fprintf(stderr, "llama_decode failed: %d\n", ret); return 1; }

    // Logits
    float * logits = llama_get_logits_ith(ctx, 0);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    float mean = 0, max_val = -1e9;
    for (int i = 0; i < n_vocab; i++) { mean += logits[i]; if (logits[i] > max_val) max_val = logits[i]; }
    mean /= n_vocab;
    fprintf(stderr, "LOGITS: mean=%.4f max=%.2f\n", mean, max_val);

    // Print all debug tensors
    fprintf(stderr, "\n=== Layer 0 intermediate values ===\n");
    print_tensor("embd", g_debug_embd);
    print_tensor("attn_norm", g_debug_attn_norm);
    print_tensor("v_proj", g_debug_v_proj);
    print_tensor("attn_out_pre", g_debug_attn_out_pre);
    print_tensor("ffn_inp", g_debug_ffn_inp);
    print_tensor("ffn_norm", g_debug_ffn_norm);
    print_tensor("L00_final", g_debug_hidden[0]);

    // Print first few layers
    fprintf(stderr, "\n=== Per-layer final hidden states ===\n");
    for (int il = 0; il < 5; il++) {
        char label[32];
        snprintf(label, sizeof(label), "L%02d", il);
        print_tensor(label, g_debug_hidden[il]);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

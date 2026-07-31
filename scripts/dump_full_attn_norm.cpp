/* dump_full_attn_norm.cpp — Dump full 1536-element attn_norm from llama.cpp */
#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static const char * MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b";

extern ggml_tensor * g_debug_hidden[32];
extern ggml_tensor * g_debug_embd;
extern ggml_tensor * g_debug_attn_norm;
extern ggml_tensor * g_debug_v_proj;
extern ggml_tensor * g_debug_attn_out_pre;

int main() {
    llama_backend_init();
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(MODEL_PATH, mparams);
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1; cparams.n_batch = 1;
    struct llama_context * ctx = llama_init_from_model(model, cparams);

    struct llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1; batch.token[0] = 49; batch.pos[0] = 0;
    batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
    llama_decode(ctx, batch);

    // Dump full tensors
    auto dump = [](const char * name, ggml_tensor * t) {
        if (!t) return;
        int64_t n = ggml_nelements(t);
        std::vector<float> data(n);
        if (t->buffer) ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
        else if (t->data) memcpy(data.data(), t->data, n * sizeof(float));
        
        FILE * f = fopen(("/tmp/llama_" + std::string(name) + ".bin").c_str(), "wb");
        fwrite(data.data(), sizeof(float), n, f);
        fclose(f);
        
        double norm = 0;
        for (int64_t i = 0; i < n; i++) norm += (double)data[i] * data[i];
        fprintf(stderr, "%s: n=%ld norm=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                name, (long)n, sqrt(norm), data[0], data[1], data[2], data[3], data[4]);
    };

    dump("embd", g_debug_embd);
    dump("attn_norm", g_debug_attn_norm);
    dump("v_proj", g_debug_v_proj);
    dump("attn_out_pre", g_debug_attn_out_pre);
    dump("L00", g_debug_hidden[0]);

    // Logits
    float * logits = llama_get_logits_ith(ctx, 0);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * f = fopen("/tmp/llama_logits.bin", "wb");
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);
    fprintf(stderr, "logits: n=%d\n", n_vocab);

    llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
}

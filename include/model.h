#pragma once
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "tensor.h"
#include "tokenizer.h"
#include "paged_kv.h"

struct GenerationConfig { 
    int max_new_tokens = 50; 
    float repetition_penalty = 1.1f; 
};

struct LlamaLayerPaged {
    int dim, hidden_dim, n_heads, n_kv_heads, head_dim;
    int total_blocks, block_size;
    HalfTensor w_rms1, w_rms2, Wq, Wk, Wv, Wo, k_block_pool, v_block_pool;
    QuantizedTensor W1, W2, W3;

    LlamaLayerPaged(MemPool& pool, int d, int hd, int nh, int nkv, int tb, int bs);
    void load(FILE* f);
    void forward_decode_paged(MemPool& scratch, HalfTensor& x, int pos, int* d_block_table);
};

struct Llama2Paged {
    int vocab=32000, dim=2048, hidden=5632, layers=22, heads=32, kv_heads=4;
    int total_blocks=1024, block_size=16; 
    HalfTensor tok_embed, norm_w, lm_head;
    std::vector<LlamaLayerPaged*> transformer;
    LlamaTokenizer tokenizer;
    cublasHandle_t handle; 
    PagedKVManager kv_manager;

    Llama2Paged(MemPool& pool);
    void load_weights(const char* path);
    void chat(MemPool& scratch, const std::vector<int>& prompt_ids, GenerationConfig cfg);
};
#include "model.h"
#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <chrono>

void read_ht(FILE* f, HalfTensor& t) { size_t b = t.numel * 2; half* buf = (half*)malloc(b); fread(buf, 2, t.numel, f); cudaMemcpy(t.data, buf, b, cudaMemcpyHostToDevice); free(buf); }
void read_qt(FILE* f, QuantizedTensor& t) { size_t b1 = t.rows * t.cols; int8_t* buf1 = (int8_t*)malloc(b1); fread(buf1, 1, b1, f); cudaMemcpy(t.data, buf1, b1, cudaMemcpyHostToDevice); free(buf1); size_t b2 = t.rows * 2; half* buf2 = (half*)malloc(b2); fread(buf2, 2, t.rows, f); cudaMemcpy(t.scales, buf2, b2, cudaMemcpyHostToDevice); free(buf2); }

LlamaLayerPaged::LlamaLayerPaged(MemPool& pool, int d, int hd, int nh, int nkv, int tb, int bs) : dim(d), hidden_dim(hd), n_heads(nh), n_kv_heads(nkv), head_dim(d/nh), total_blocks(tb), block_size(bs), w_rms1(pool, {d}), w_rms2(pool, {d}), Wq(pool, {d, d}), Wk(pool, {nkv*(d/nh), d}), Wv(pool, {nkv*(d/nh), d}), Wo(pool, {d, d}), W1(pool, hd, d), W2(pool, hd, d), W3(pool, d, hd), k_block_pool(pool, {tb, bs, nkv*(d/nh)}), v_block_pool(pool, {tb, bs, nkv*(d/nh)}) {}

void LlamaLayerPaged::load(FILE* f) { read_ht(f, w_rms1); read_ht(f, Wq); read_ht(f, Wk); read_ht(f, Wv); read_ht(f, Wo); read_ht(f, w_rms2); read_qt(f, W1); read_qt(f, W2); read_qt(f, W3); }

void LlamaLayerPaged::forward_decode_paged(MemPool& scratch, HalfTensor& x, int pos, int* d_block_table) {
    int hd = head_dim; 
    HalfTensor xn(scratch, {1, dim}), Q(scratch, {1, dim}), K(scratch, {1, n_kv_heads * hd}), V(scratch, {1, n_kv_heads * hd}); 
    Tensor S(scratch, {n_heads, 4096}), P(scratch, {n_heads, 4096}); 
    HalfTensor A(scratch, {1, dim}), AO(scratch, {1, dim}), fn(scratch, {1, dim}), gate(scratch, {1, hidden_dim}), up(scratch, {1, hidden_dim}), swi(scratch, {1, hidden_dim}), F(scratch, {1, dim});
    
    k_half_rmsnorm(x.data, w_rms1.data, xn.data, 1, dim, 1e-5f);
    k_half_gemv(xn.data, Wq.data, Q.data, dim, dim); 
    k_half_gemv(xn.data, Wk.data, K.data, dim, n_kv_heads * hd); 
    k_half_gemv(xn.data, Wv.data, V.data, dim, n_kv_heads * hd);
    
    int* d_pos = scratch.alloc<int>(1);
    cudaMemcpyAsync(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, 0);
    
    k_half_llama_rope_graph(Q.data, n_heads, hd, d_pos); 
    k_half_llama_rope_graph(K.data, n_kv_heads, hd, d_pos);
    
    k_paged_kv_write(K.data, V.data, k_block_pool.data, v_block_pool.data, d_block_table, pos, block_size, n_kv_heads * hd);
    k_paged_mha_scores(Q.data, k_block_pool.data, S.data, d_block_table, pos, block_size, n_heads, n_kv_heads, hd);
    k_row_softmax_graph(S.data, P.data, n_heads, d_pos);
    k_paged_mha_weighted_sum(P.data, v_block_pool.data, A.data, d_block_table, pos, block_size, n_heads, n_kv_heads, hd);
    
    k_half_gemv(A.data, Wo.data, AO.data, dim, dim);
    k_half_fused_add_rmsnorm(x.data, AO.data, w_rms2.data, fn.data, 1, dim, 1e-5f);
    k_int8_gemv(fn.data, W1.data, W1.scales, gate.data, dim, hidden_dim); 
    k_int8_gemv(fn.data, W2.data, W2.scales, up.data, dim, hidden_dim);
    k_half_swiglu(gate.data, up.data, swi.data, hidden_dim);
    k_int8_gemv(swi.data, W3.data, W3.scales, F.data, hidden_dim, dim);
    k_half_add(x.data, F.data, x.data, dim);
}

Llama2Paged::Llama2Paged(MemPool& pool) : tok_embed(pool, {vocab, dim}), norm_w(pool, {dim}), lm_head(pool, {vocab, dim}), kv_manager(total_blocks, block_size) {
    for (int i=0; i<layers; i++) transformer.push_back(new LlamaLayerPaged(pool, dim, hidden, heads, kv_heads, total_blocks, block_size)); 
    tokenizer.load("tokenizer.bin");
}

void Llama2Paged::load_weights(const char* path) { 
    FILE* f = fopen(path, "rb"); 
    read_ht(f, tok_embed); 
    for (int i=0; i<layers; i++) transformer[i]->load(f); 
    read_ht(f, norm_w); read_ht(f, lm_head); 
    fclose(f); 
}

void Llama2Paged::chat(MemPool& scratch, const std::vector<int>& prompt_ids, GenerationConfig cfg) {
    kv_manager.allocate_sequence(1);
    int pos = 0;
    std::vector<int> past;
    printf("\n[TinyLlama INT8 + Paged Attention]: "); fflush(stdout);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int token = prompt_ids[0];
    int tokens_generated = 0;
    std::vector<float> h_logits(vocab);

    while (pos < prompt_ids.size() - 1) {
        kv_manager.append_token(1);
        scratch.reset();
        int num_blocks = kv_manager.active_sequences[1].blocks.size();
        int* d_block_table = scratch.alloc<int>(num_blocks);
        cudaMemcpyAsync(d_block_table, kv_manager.active_sequences[1].blocks.data(), num_blocks * sizeof(int), cudaMemcpyHostToDevice, 0);
        
        HalfTensor dx(scratch, {1, dim});
        int* d_token = scratch.alloc<int>(1);
        cudaMemcpyAsync(d_token, &token, sizeof(int), cudaMemcpyHostToDevice, 0);
        k_half_embedding_lookup(d_token, tok_embed.data, dx.data, 1, dim);
        
        for (int l=0; l<layers; l++) transformer[l]->forward_decode_paged(scratch, dx, pos, d_block_table);
        
        printf("%s", tokenizer.decode(token).c_str()); fflush(stdout);
        past.push_back(token);
        pos++;
        token = prompt_ids[pos];
    }

    for (int i=0; i<cfg.max_new_tokens; i++) {
        kv_manager.append_token(1);
        scratch.reset();
        
        int num_blocks = kv_manager.active_sequences[1].blocks.size();
        int* d_block_table = scratch.alloc<int>(num_blocks);
        cudaMemcpyAsync(d_block_table, kv_manager.active_sequences[1].blocks.data(), num_blocks * sizeof(int), cudaMemcpyHostToDevice, 0);
        
        HalfTensor dx(scratch, {1, dim});
        int* d_token = scratch.alloc<int>(1);
        cudaMemcpyAsync(d_token, &token, sizeof(int), cudaMemcpyHostToDevice, 0);
        k_half_embedding_lookup(d_token, tok_embed.data, dx.data, 1, dim);
        
        for (int l=0; l<layers; l++) transformer[l]->forward_decode_paged(scratch, dx, pos, d_block_table);
        
        HalfTensor dfn(scratch, {1, dim}), dlogits16(scratch, {1, vocab});
        k_half_rmsnorm(dx.data, norm_w.data, dfn.data, 1, dim, 1e-5f);
        k_half_gemv(dfn.data, lm_head.data, dlogits16.data, dim, vocab);
        Tensor logits32(scratch, {1, vocab}); k_half_to_float(dlogits16.data, logits32.data, vocab);
        
        cudaStreamSynchronize(0);
        cudaMemcpy(h_logits.data(), logits32.data, vocab * sizeof(float), cudaMemcpyDeviceToHost);
        for (int p_id : past) h_logits[p_id] = h_logits[p_id] > 0 ? h_logits[p_id] / cfg.repetition_penalty : h_logits[p_id] * cfg.repetition_penalty;
        
        int best_id = 0; float best_val = -1e9f;
        for (int v = 0; v < vocab; v++) if (h_logits[v] > best_val) { best_val = h_logits[v]; best_id = v; }
        token = best_id;
        
        if (token <= 2) break; 
        past.push_back(token);
        printf("%s", tokenizer.decode(token).c_str()); fflush(stdout); 
        pos++; tokens_generated++;
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t2 - t1;
    if (tokens_generated > 0) printf("\n\n[Decode Speed: %.2f tok/s]\n", tokens_generated / dt.count());
    kv_manager.free_sequence(1);
}
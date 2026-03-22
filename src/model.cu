#include "model.h"
#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <chrono>

void read_ht(FILE* f, HalfTensor& t) { size_t b = t.numel * 2; half* buf = (half*)malloc(b); fread(buf, 2, t.numel, f); cudaMemcpy(t.data, buf, b, cudaMemcpyHostToDevice); free(buf); }
void read_qt(FILE* f, QuantizedTensor& t) { size_t b1 = t.rows * t.cols; int8_t* buf1 = (int8_t*)malloc(b1); fread(buf1, 1, b1, f); cudaMemcpy(t.data, buf1, b1, cudaMemcpyHostToDevice); free(buf1); size_t b2 = t.rows * 2; half* buf2 = (half*)malloc(b2); fread(buf2, 2, t.rows, f); cudaMemcpy(t.scales, buf2, b2, cudaMemcpyHostToDevice); free(buf2); }

LlamaLayerPaged::LlamaLayerPaged(MemPool& pool, int d, int hd, int nh, int nkv, int tb, int bs) 
    : dim(d), hidden_dim(hd), n_heads(nh), n_kv_heads(nkv), head_dim(d/nh), total_blocks(tb), block_size(bs), 
      w_rms1(pool, {d}), w_rms2(pool, {d}), Wq(pool, {d, d}), Wk(pool, {nkv*(d/nh), d}), Wv(pool, {nkv*(d/nh), d}), Wo(pool, {d, d}), 
      W1(pool, hd, d), W2(pool, hd, d), W3(pool, d, hd), k_block_pool(pool, {tb, bs, nkv*(d/nh)}), v_block_pool(pool, {tb, bs, nkv*(d/nh)}) {}

void LlamaLayerPaged::load(FILE* f) { read_ht(f, w_rms1); read_ht(f, Wq); read_ht(f, Wk); read_ht(f, Wv); read_ht(f, Wo); read_ht(f, w_rms2); read_qt(f, W1); read_qt(f, W2); read_qt(f, W3); }

void LlamaLayerPaged::forward_decode_batched(MemPool& scratch, cublasHandle_t handle, HalfTensor& x, int* d_pos, int* d_block_table, int batch_size, int max_blocks_per_seq) {
    int hd = head_dim; 
    HalfTensor xn(scratch, {batch_size, dim}), Q(scratch, {batch_size, dim}), K(scratch, {batch_size, n_kv_heads * hd}), V(scratch, {batch_size, n_kv_heads * hd}); 
    Tensor S(scratch, {batch_size, n_heads, 4096}), P(scratch, {batch_size, n_heads, 4096}); 
    HalfTensor A(scratch, {batch_size, dim}), AO(scratch, {batch_size, dim}), fn(scratch, {batch_size, dim}), gate(scratch, {batch_size, hidden_dim}), up(scratch, {batch_size, hidden_dim}), swi(scratch, {batch_size, hidden_dim}), F(scratch, {batch_size, dim});
    
    k_half_rmsnorm(x.data, w_rms1.data, xn.data, batch_size, dim, 1e-5f);
    k_half_linear(handle, xn.data, Wq.data, Q.data, batch_size, dim, dim); 
    k_half_linear(handle, xn.data, Wk.data, K.data, batch_size, dim, n_kv_heads * hd); 
    k_half_linear(handle, xn.data, Wv.data, V.data, batch_size, dim, n_kv_heads * hd);
    
    k_batched_llama_rope(Q.data, d_pos, n_heads, hd, batch_size); 
    k_batched_llama_rope(K.data, d_pos, n_kv_heads, hd, batch_size);
    
    k_batched_paged_kv_write(K.data, V.data, k_block_pool.data, v_block_pool.data, d_block_table, d_pos, block_size, n_kv_heads * hd, max_blocks_per_seq, batch_size);
    k_batched_paged_mha_scores(Q.data, k_block_pool.data, S.data, d_block_table, d_pos, block_size, n_heads, n_kv_heads, hd, max_blocks_per_seq, batch_size);
    k_batched_row_softmax(S.data, P.data, n_heads, d_pos, batch_size);
    k_batched_paged_mha_sum(P.data, v_block_pool.data, A.data, d_block_table, d_pos, block_size, n_heads, n_kv_heads, hd, max_blocks_per_seq, batch_size);
    
    k_half_linear(handle, A.data, Wo.data, AO.data, batch_size, dim, dim);
    k_half_fused_add_rmsnorm(x.data, AO.data, w_rms2.data, fn.data, batch_size, dim, 1e-5f);
    
    for(int b=0; b<batch_size; b++) {
        k_int8_gemv(fn.data + b*dim, W1.data, W1.scales, gate.data + b*hidden_dim, dim, hidden_dim); 
        k_int8_gemv(fn.data + b*dim, W2.data, W2.scales, up.data + b*hidden_dim, dim, hidden_dim);
    }
    k_half_swiglu(gate.data, up.data, swi.data, batch_size * hidden_dim);
    for(int b=0; b<batch_size; b++) {
        k_int8_gemv(swi.data + b*hidden_dim, W3.data, W3.scales, F.data + b*dim, hidden_dim, dim);
    }
    k_half_add(x.data, F.data, x.data, batch_size * dim);
}

Llama2Paged::Llama2Paged(MemPool& pool) : tok_embed(pool, {vocab, dim}), norm_w(pool, {dim}), lm_head(pool, {vocab, dim}), kv_manager(total_blocks, block_size) {
    cublasCreate(&handle); 
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

void Llama2Paged::chat_batched(MemPool& scratch, const std::vector<std::vector<int>>& prompts, GenerationConfig cfg) {
    int batch_size = prompts.size();
    std::vector<int> positions(batch_size, 0);
    std::vector<int> tokens(batch_size, 0);
    std::vector<std::vector<int>> pasts(batch_size);
    std::vector<bool> active(batch_size, true);
    std::vector<std::string> generated_texts(batch_size, "");

    int max_prompt_len = 0;
    for (int b = 0; b < batch_size; b++) {
        kv_manager.allocate_sequence(b);
        tokens[b] = prompts[b][0]; 
        pasts[b].push_back(tokens[b]);
        if (prompts[b].size() > max_prompt_len) max_prompt_len = prompts[b].size();
    }

    int max_blocks_per_seq = max_seq / block_size;
    std::vector<int> flat_block_table(batch_size * max_blocks_per_seq, 0);

    auto t1 = std::chrono::high_resolution_clock::now();
    int tokens_generated = 0;

    for (int step = 0; step < max_prompt_len - 1 + cfg.max_new_tokens; step++) {
        int active_count = 0;
        for(bool a : active) if(a) active_count++;
        if(active_count == 0) break;

        for (int b = 0; b < batch_size; b++) {
            if(active[b]) {
                kv_manager.append_token(b);
                auto& seq_blocks = kv_manager.active_sequences[b].blocks;
                for(int j=0; j<seq_blocks.size(); j++) flat_block_table[b * max_blocks_per_seq + j] = seq_blocks[j];
            }
        }

        scratch.reset();
        int* d_block_table = scratch.alloc<int>(batch_size * max_blocks_per_seq);
        cudaMemcpyAsync(d_block_table, flat_block_table.data(), batch_size * max_blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice, 0);

        int* d_pos = scratch.alloc<int>(batch_size);
        cudaMemcpyAsync(d_pos, positions.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, 0);

        int* d_tokens = scratch.alloc<int>(batch_size);
        cudaMemcpyAsync(d_tokens, tokens.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, 0);

        HalfTensor dx(scratch, {batch_size, dim});
        k_half_embedding_lookup(d_tokens, tok_embed.data, dx.data, batch_size, dim);

        for (int l=0; l<layers; l++) {
            transformer[l]->forward_decode_batched(scratch, handle, dx, d_pos, d_block_table, batch_size, max_blocks_per_seq);
        }

        HalfTensor dfn(scratch, {batch_size, dim}), dlogits16(scratch, {batch_size, vocab});
        k_half_rmsnorm(dx.data, norm_w.data, dfn.data, batch_size, dim, 1e-5f);
        k_half_linear(handle, dfn.data, lm_head.data, dlogits16.data, batch_size, dim, vocab);

        Tensor logits32(scratch, {batch_size, vocab});
        k_half_to_float(dlogits16.data, logits32.data, batch_size * vocab);
        
        std::vector<float> h_logits(batch_size * vocab);
        cudaStreamSynchronize(0);
        cudaMemcpy(h_logits.data(), logits32.data, batch_size * vocab * sizeof(float), cudaMemcpyDeviceToHost);

        for (int b = 0; b < batch_size; b++) {
            if(!active[b]) continue;
            
            int next_token = 0;
            
            if (step < prompts[b].size() - 1) {
                next_token = prompts[b][step + 1];
            } else {
                for (int p_id : pasts[b]) {
                    float val = h_logits[b * vocab + p_id];
                    h_logits[b * vocab + p_id] = val > 0 ? val / cfg.repetition_penalty : val * cfg.repetition_penalty;
                }

                int best_id = 0; float best_val = -1e9f;
                for (int v = 0; v < vocab; v++) {
                    if (h_logits[b * vocab + v] > best_val) { best_val = h_logits[b * vocab + v]; best_id = v; }
                }
                next_token = best_id;
                
                if (next_token <= 2) { active[b] = false; continue; }
                tokens_generated++;
                generated_texts[b] += tokenizer.decode(next_token);
            }

            tokens[b] = next_token;
            pasts[b].push_back(next_token);
            positions[b]++;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t2 - t1;

    for (int b = 0; b < batch_size; b++) {
        printf("\n=== USER %d ===\n", b+1);
        for(int id : prompts[b]) printf("%s", tokenizer.decode(id).c_str());
        printf("%s\n", generated_texts[b].c_str());
        kv_manager.free_sequence(b);
    }
    if (tokens_generated > 0) printf("\n[Batched Decode Speed: %.2f tok/s]\n", tokens_generated / dt.count());
}
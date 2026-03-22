#include "kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void _half_to_float(half* src, float* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = __half2float(src[i]); }
void k_half_to_float(half* src, float* dst, int n) { _half_to_float<<<(n+255)/256, 256>>>(src, dst, n); }

__global__ void _half_copy(half* src, half* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = src[i]; }
void k_half_copy(half* src, half* dst, int n) { _half_copy<<<(n+255)/256, 256>>>(src, dst, n); }

__global__ void _half_gemv(half* x, half* W, half* y, int in_features, int out_features) {
    int row = blockIdx.x; if (row >= out_features) return;
    float sum = 0.0f; int w_base = row * in_features;
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) sum += __half2float(x[i]) * __half2float(W[w_base + i]);
    __shared__ float sdata[32]; int warpID = threadIdx.x / 32; int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) sdata[warpID] = sum; __syncthreads();
    if (warpID == 0) {
        sum = (lane < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) y[row] = __float2half(sum);
    }
}
void k_half_gemv(half* x, half* W, half* y, int in_features, int out_features) { _half_gemv<<<out_features, 256>>>(x, W, y, in_features, out_features); }

__global__ void _int8_gemv(half* x, int8_t* W, half* scales, half* y, int in_features, int out_features) {
    int row = blockIdx.x; if (row >= out_features) return;
    float sum = 0.0f; int w_base = row * in_features; float scale = __half2float(scales[row]);
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) sum += __half2float(x[i]) * (float)(W[w_base + i]) * scale;
    __shared__ float sdata[32]; int warpID = threadIdx.x / 32; int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) sdata[warpID] = sum; __syncthreads();
    if (warpID == 0) {
        sum = (lane < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) y[row] = __float2half(sum);
    }
}
void k_int8_gemv(half* x, int8_t* W, half* scales, half* y, int in_features, int out_features) { _int8_gemv<<<out_features, 256>>>(x, W, scales, y, in_features, out_features); }

void k_half_linear(cublasHandle_t handle, half* x, half* W, half* y, int b, int in_features, int out_features) { 
    half alpha = __float2half(1.0f), beta = __float2half(0.0f); 
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, b, in_features, &alpha, W, in_features, x, in_features, &beta, y, out_features); 
}

__global__ void _stable_attention_prefill(half* Q, half* K, half* V, half* O, int seq, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, r = blockIdx.x; if (r >= seq) return;
    int kv_h = h / (n_heads / n_kv_heads), q_base = r * (n_heads * head_dim) + h * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim), max_score = -1e20f;
    
    // Prevent stack overflow with loop splitting or limit
    int start_c = (r > 2048) ? r - 2048 : 0; 
    
    // Pass 1: Max
    for (int c = start_c; c <= r; c++) {
        int k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += __half2float(Q[q_base + d]) * __half2float(K[k_base + d]);
        s *= scale;
        if (s > max_score) max_score = s;
    }
    
    // Pass 2: Sum Exp & Accumulate
    float sum_exp = 0.0f;
    
    // We compute exp(s - max) on the fly to avoid large array
    // This is less efficient than shared mem but avoids the crash
    // For a proper kernel, use shared memory tiling.
    
    // Recomputing dot products is costly but safe for this "stable" patch.
    // Given the constraints, we will just cap the lookback or rely on recompute.
    
    // Accumulator for O
    float acc[128]; // Max head_dim
    for(int d=0; d<head_dim; d++) acc[d] = 0.0f;

    for (int c = start_c; c <= r; c++) {
        int k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        int v_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += __half2float(Q[q_base + d]) * __half2float(K[k_base + d]);
        s = expf(s * scale - max_score);
        sum_exp += s;
        
        for (int d = 0; d < head_dim; d++) {
             acc[d] += s * __half2float(V[v_base + d]);
        }
    }
    
    for (int d = 0; d < head_dim; d++) {
        O[q_base + d] = __float2half(acc[d] / sum_exp);
    }
}
void k_flash_attention_prefill(half* Q, half* K, half* V, half* O, int seq, int n_heads, int n_kv_heads, int head_dim) { 
    dim3 g(seq, n_heads); _stable_attention_prefill<<<g, 1>>>(Q, K, V, O, seq, n_heads, n_kv_heads, head_dim); 
}

__global__ void _half_mha_scores_one_graph(half* q, half* K_cache, float* s, int* d_pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, pos = *d_pos;
    if (h < n_heads && c <= pos) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads);
        int q_base = h * head_dim, k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        for(int d=0; d < head_dim; d++) sum += __half2float(q[q_base+d]) * __half2float(K_cache[k_base+d]);
        s[h * 4096 + c] = sum / sqrtf((float)head_dim);
    }
}
void k_half_mha_scores_one(half* q, half* K, float* s, int* d_pos, int n_heads, int n_kv_heads, int head_dim) { 
    dim3 g((4096)/256, n_heads), b(256); _half_mha_scores_one_graph<<<g, b>>>(q, K, s, d_pos, n_heads, n_kv_heads, head_dim); 
}

__global__ void _half_mha_weighted_sum_one_graph(float* p, half* V_cache, half* o, int* d_pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x, pos = *d_pos;
    if (h < n_heads && d < head_dim) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads);
        for(int c=0; c <= pos; c++) sum += p[h * 4096 + c] * __half2float(V_cache[c*(n_kv_heads*head_dim)+kv_h*head_dim+d]);
        o[h*head_dim+d] = __float2half(sum);
    }
}
void k_half_mha_weighted_sum_one(float* p, half* V, half* o, int* d_pos, int n_heads, int n_kv_heads, int head_dim) { 
    dim3 g((head_dim+255)/256, n_heads), b(256); _half_mha_weighted_sum_one_graph<<<g, b>>>(p, V, o, d_pos, n_heads, n_kv_heads, head_dim); 
}

__global__ void _row_softmax_graph(float* x, float* y, int n_heads, int* d_pos) {
    int r = blockIdx.x, pos = *d_pos, cols = pos + 1;
    if (r >= n_heads) return;
    __shared__ float m, s;
    if (threadIdx.x == 0) { 
        float mx = x[r * 4096]; 
        for (int i = 1; i < cols; i++) if (x[r * 4096 + i] > mx) mx = x[r * 4096 + i]; 
        float sm = 0.0f; 
        for (int i = 0; i < cols; i++) sm += expf(x[r * 4096 + i] - mx); 
        m = mx; s = sm; 
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r * 4096 + i] = expf(x[r * 4096 + i] - m) / s;
}
void k_row_softmax_graph(float* x, float* y, int n_heads, int* d_pos) { _row_softmax_graph<<<n_heads, 256>>>(x, y, n_heads, d_pos); }

__global__ void _half_llama_rope_graph(half* x, int n_heads, int head_dim, int* d_pos) {
    int h = blockIdx.y, i = threadIdx.x, pos = *d_pos;
    if (h < n_heads && i < head_dim / 2) {
        int base = h * head_dim;
        float x0 = __half2float(x[base+i]), x1 = __half2float(x[base+i+head_dim/2]);
        float th = pos * powf(10000.0f, -(2.0f * i) / head_dim);
        x[base+i] = __float2half(x0*cosf(th) - x1*sinf(th)); x[base+i+head_dim/2] = __float2half(x0*sinf(th) + x1*cosf(th));
    }
}
void k_half_llama_rope_graph(half* x, int n_heads, int head_dim, int* d_pos) { dim3 g(1, n_heads); _half_llama_rope_graph<<<g, head_dim / 2>>>(x, n_heads, head_dim, d_pos); }

__global__ void _half_copy_to_cache_graph(half* src, half* cache, int* d_pos, int dim) { 
    int d = blockIdx.x * blockDim.x + threadIdx.x, pos = *d_pos; 
    if (d < dim) cache[pos * dim + d] = src[d]; 
}
void k_half_copy_to_cache_graph(half* src, half* cache, int* d_pos, int dim) { _half_copy_to_cache_graph<<<(dim + 255) / 256, 256>>>(src, cache, d_pos, dim); }

__global__ void _paged_kv_write(half* k_src, half* v_src, half* k_cache, half* v_cache, int* block_table, int pos, int block_size, int kv_dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kv_dim) return;
    int logical = pos / block_size, offset = pos % block_size, phys = block_table[logical];
    int target = phys * (block_size * kv_dim) + (offset * kv_dim) + d;
    k_cache[target] = k_src[d]; v_cache[target] = v_src[d];
}
void k_paged_kv_write(half* k_src, half* v_src, half* k_cache, half* v_cache, int* block_table, int pos, int block_size, int dim) {
    _paged_kv_write<<<(dim + 255) / 256, 256>>>(k_src, v_src, k_cache, v_cache, block_table, pos, block_size, dim);
}

__global__ void _paged_mha_scores(half* q, half* K_cache, float* s, int* block_table, int pos, int block_size, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x; 
    if (h < n_heads && c <= pos) {
        int logical = c / block_size, offset = c % block_size, phys = block_table[logical];
        int kv_h = h / (n_heads / n_kv_heads), q_base = h * head_dim, kv_dim = n_kv_heads * head_dim;
        int k_base = phys * (block_size * kv_dim) + (offset * kv_dim) + (kv_h * head_dim);
        float sum = 0.0f;
        for(int d=0; d < head_dim; d++) sum += __half2float(q[q_base+d]) * __half2float(K_cache[k_base+d]);
        s[h * 4096 + c] = sum / sqrtf((float)head_dim);
    }
}
void k_paged_mha_scores(half* q, half* K_cache, float* s, int* block_table, int pos, int block_size, int n_heads, int n_kv_heads, int head_dim) {
    dim3 g((4096)/256, n_heads), b(256);
    _paged_mha_scores<<<g, b>>>(q, K_cache, s, block_table, pos, block_size, n_heads, n_kv_heads, head_dim);
}

__global__ void _paged_mha_weighted_sum(float* p, half* V_cache, half* o, int* block_table, int pos, int block_size, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x; 
    if (h < n_heads && d < head_dim) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads), kv_dim = n_kv_heads * head_dim;
        for(int c=0; c <= pos; c++) {
            int logical = c / block_size, offset = c % block_size, phys = block_table[logical];
            int v_base = phys * (block_size * kv_dim) + (offset * kv_dim) + (kv_h * head_dim);
            sum += p[h * 4096 + c] * __half2float(V_cache[v_base+d]);
        }
        o[h * head_dim + d] = __float2half(sum);
    }
}
void k_paged_mha_weighted_sum(float* p, half* V_cache, half* o, int* block_table, int pos, int block_size, int n_heads, int n_kv_heads, int head_dim) {
    dim3 g((head_dim+255)/256, n_heads), b(256);
    _paged_mha_weighted_sum<<<g, b>>>(p, V_cache, o, block_table, pos, block_size, n_heads, n_kv_heads, head_dim);
}

__global__ void _batched_llama_rope(half* x, int* d_pos, int n_heads, int head_dim) {
    int b = blockIdx.z, h = blockIdx.y, i = threadIdx.x, pos = d_pos[b];
    if (h < n_heads && i < head_dim / 2) {
        int base = b * (n_heads * head_dim) + h * head_dim;
        
        // Correct paired RoPE
        int j = 2 * i;
        float x0 = __half2float(x[base+j]);
        float x1 = __half2float(x[base+j+1]);
        
        float th = pos * powf(10000.0f, -(2.0f * i) / head_dim);
        float c = cosf(th);
        float s = sinf(th);
        
        x[base+j] = __float2half(x0*c - x1*s);
        x[base+j+1] = __float2half(x0*s + x1*c);
    }
}
void k_batched_llama_rope(half* x, int* d_pos, int n_heads, int head_dim, int batch_size) {
    dim3 g(1, n_heads, batch_size);
    _batched_llama_rope<<<g, head_dim / 2>>>(x, d_pos, n_heads, head_dim);
}

__global__ void _batched_paged_kv_write(half* k_src, half* v_src, half* k_cache, half* v_cache, int* block_table, int* d_pos, int block_size, int kv_dim, int max_blocks) {
    int b = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kv_dim) return;
    int pos = d_pos[b], logical = pos / block_size, offset = pos % block_size;
    int phys = block_table[b * max_blocks + logical];
    int target = phys * (block_size * kv_dim) + (offset * kv_dim) + d, src = b * kv_dim + d;
    k_cache[target] = k_src[src]; v_cache[target] = v_src[src];
}
void k_batched_paged_kv_write(half* k_src, half* v_src, half* k_cache, half* v_cache, int* block_table, int* d_pos, int block_size, int kv_dim, int max_blocks, int batch_size) {
    dim3 g((kv_dim + 255) / 256, batch_size);
    _batched_paged_kv_write<<<g, 256>>>(k_src, v_src, k_cache, v_cache, block_table, d_pos, block_size, kv_dim, max_blocks);
}

__global__ void _batched_paged_mha_scores(half* q, half* K_cache, float* s, int* block_table, int* d_pos, int block_size, int n_heads, int n_kv_heads, int head_dim, int max_blocks) {
    int b = blockIdx.z, h = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, pos = d_pos[b];
    if (h < n_heads && c <= pos) {
        int logical = c / block_size, offset = c % block_size, phys = block_table[b * max_blocks + logical];
        int kv_h = h / (n_heads / n_kv_heads), q_base = b * (n_heads * head_dim) + h * head_dim, kv_dim = n_kv_heads * head_dim;
        int k_base = phys * (block_size * kv_dim) + (offset * kv_dim) + (kv_h * head_dim);
        float sum = 0.0f;
        for(int d=0; d < head_dim; d++) sum += __half2float(q[q_base+d]) * __half2float(K_cache[k_base+d]);
        s[b * (n_heads * 4096) + h * 4096 + c] = sum / sqrtf((float)head_dim);
    }
}
void k_batched_paged_mha_scores(half* q, half* K_cache, float* s, int* block_table, int* d_pos, int block_size, int n_heads, int n_kv_heads, int head_dim, int max_blocks, int batch_size) {
    dim3 g((4096)/256, n_heads, batch_size);
    _batched_paged_mha_scores<<<g, 256>>>(q, K_cache, s, block_table, d_pos, block_size, n_heads, n_kv_heads, head_dim, max_blocks);
}

__global__ void _batched_row_softmax(float* x, float* y, int n_heads, int* d_pos) {
    int b = blockIdx.y, r = blockIdx.x, pos = d_pos[b], cols = pos + 1;
    if (r >= n_heads) return;
    int row_idx = b * (n_heads * 4096) + r * 4096;
    __shared__ float m, s;
    if (threadIdx.x == 0) { 
        float mx = x[row_idx]; 
        for (int i = 1; i < cols; i++) if (x[row_idx + i] > mx) mx = x[row_idx + i]; 
        float sm = 0.0f; 
        for (int i = 0; i < cols; i++) sm += expf(x[row_idx + i] - mx); 
        m = mx; s = sm; 
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[row_idx + i] = expf(x[row_idx + i] - m) / s;
}
void k_batched_row_softmax(float* x, float* y, int n_heads, int* d_pos, int batch_size) {
    dim3 g(n_heads, batch_size);
    _batched_row_softmax<<<g, 256>>>(x, y, n_heads, d_pos);
}

__global__ void _batched_paged_mha_sum(float* p, half* V_cache, half* o, int* block_table, int* d_pos, int block_size, int n_heads, int n_kv_heads, int head_dim, int max_blocks) {
    int b = blockIdx.z, h = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x, pos = d_pos[b];
    if (h < n_heads && d < head_dim) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads), kv_dim = n_kv_heads * head_dim;
        for(int c=0; c <= pos; c++) {
            int logical = c / block_size, offset = c % block_size, phys = block_table[b * max_blocks + logical];
            int v_base = phys * (block_size * kv_dim) + (offset * kv_dim) + (kv_h * head_dim);
            sum += p[b * (n_heads * 4096) + h * 4096 + c] * __half2float(V_cache[v_base+d]);
        }
        o[b * (n_heads * head_dim) + h * head_dim + d] = __float2half(sum);
    }
}
void k_batched_paged_mha_sum(float* p, half* V_cache, half* o, int* block_table, int* d_pos, int block_size, int n_heads, int n_kv_heads, int head_dim, int max_blocks, int batch_size) {
    dim3 g((head_dim+255)/256, n_heads, batch_size);
    _batched_paged_mha_sum<<<g, 256>>>(p, V_cache, o, block_table, d_pos, block_size, n_heads, n_kv_heads, head_dim, max_blocks);
}

__global__ void _half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps) {
    int r = blockIdx.x; if (r >= rows) return;
    __shared__ float s;
    if (threadIdx.x == 0) { float ss = 0.0f; for (int i = 0; i < cols; i++) ss += __half2float(x[r*cols+i])*__half2float(x[r*cols+i]); s = rsqrtf(ss/cols+eps); }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r*cols+i] = __float2half(__half2float(x[r*cols+i])*s*__half2float(w[i]));
}
void k_half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps) { _half_rmsnorm<<<rows, 256>>>(x, w, y, rows, cols, eps); }

__global__ void _half_fused_add_rmsnorm(half* x, half* res, half* w, half* out, int rows, int cols, float eps) {
    int r = blockIdx.x; if (r >= rows) return;
    __shared__ float s; if (threadIdx.x == 0) s = 0.0f; __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(x[r * cols + i]) + __half2float(res[r * cols + i]);
        x[r * cols + i] = __float2half(val); ss += val * val;
    }
    for (int offset = 16; offset > 0; offset /= 2) ss += __shfl_down_sync(0xffffffff, ss, offset);
    if (threadIdx.x % 32 == 0) atomicAdd(&s, ss); __syncthreads();
    if (threadIdx.x == 0) s = rsqrtf(s / cols + eps); __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) out[r * cols + i] = __float2half(__half2float(x[r * cols + i]) * s * __half2float(w[i]));
}
void k_half_fused_add_rmsnorm(half* x, half* res, half* w, half* out, int rows, int cols, float eps) { _half_fused_add_rmsnorm<<<rows, 32>>>(x, res, w, out, rows, cols, eps); }

__global__ void _half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base) {
    int pos = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    if (pos < seq && h < n_heads && i < head_dim / 2) {
        int base = pos * (n_heads * head_dim) + h * head_dim;
        float x0 = __half2float(x[base+i]), x1 = __half2float(x[base+i+head_dim/2]);
        float th = (pos_base + pos) * powf(10000.0f, -(2.0f * i) / head_dim);
        x[base+i] = __float2half(x0*cosf(th) - x1*sinf(th)); x[base+i+head_dim/2] = __float2half(x0*sinf(th) + x1*cosf(th));
    }
}
void k_half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base) { dim3 g(seq, n_heads); _half_llama_rope<<<g, head_dim / 2>>>(x, seq, n_heads, head_dim, pos_base); }

__global__ void _half_add(half* a, half* b, half* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = __float2half(__half2float(a[i]) + __half2float(b[i])); }
void k_half_add(half* a, half* b, half* c, int n) { _half_add<<<(n+255)/256, 256>>>(a, b, c, n); }

__global__ void _half_swiglu(half* gate, half* up, half* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float g = __half2float(gate[i]), u = __half2float(up[i]); out[i] = __float2half((g / (1.0f + expf(-g))) * u); } }
void k_half_swiglu(half* gate, half* up, half* out, int n) { _half_swiglu<<<(n+255)/256, 256>>>(gate, up, out, n); }

__global__ void _half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim) { int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x; if (t < seq && d < dim) out[t*dim+d] = table[ids[t]*dim+d]; }
void k_half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim) { dim3 g((dim+255)/256, seq), b(256); _half_embedding_lookup<<<g, b>>>(ids, table, out, seq, dim); }

__global__ void _half_copy_block_to_cache(half* src, half* cache, int pos_base, int seq_len, int dim) { int seq_idx = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x; if (seq_idx < seq_len && d < dim) cache[(pos_base + seq_idx) * dim + d] = src[seq_idx * dim + d]; }
void k_half_copy_block_to_cache(half* src, half* cache, int pos_base, int seq_len, int dim) { dim3 g((dim+255)/256, seq_len), b(256); _half_copy_block_to_cache<<<g, b>>>(src, cache, pos_base, seq_len, dim); }

__global__ void _argmax_row(float* x, int* out, int rows, int cols) { int r = blockIdx.x; if (r < rows && threadIdx.x == 0) { int idx = 0; float best = x[r*cols]; for (int i=1; i<cols; i++) { if (x[r*cols+i] > best) { best = x[r*cols+i]; idx = i; } } out[r] = idx; } }
void k_argmax_row(float* x, int* out, int rows, int cols) { _argmax_row<<<rows, 1>>>(x, out, rows, cols); }
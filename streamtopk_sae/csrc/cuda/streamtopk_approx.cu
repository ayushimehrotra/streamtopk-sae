/*
 * CUDA approximate streaming top-k SAE forward pass.
 *
 * Algorithm:
 *   For each batch row i:
 *     For each latent tile t:
 *       Compute scores for the tile (T latents).
 *       Find top-c within the tile (block-wide).
 *       Write c (value, index) pairs to candidate_buffer[i, t*c:(t+1)*c].
 *   Python wrapper then calls torch.topk on the candidate buffer.
 *
 * Candidate buffer: (B, num_tiles, c) in global memory.
 * c restricted to templated values: 16, 32, 64, 128.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cuda/topk_buffer.cuh"

// Type-safe float conversion helpers
__device__ __forceinline__ float scalar_to_float(float v)              { return v; }
__device__ __forceinline__ float scalar_to_float(__half v)             { return __half2float(v); }
__device__ __forceinline__ float scalar_to_float(__nv_bfloat16 v)      { return __bfloat162float(v); }

// BLOCK_THREADS_APPROX=64, T_APPROX=128: LATENTS_PER_THREAD=2 (same as exact kernel).
// Smem budget for fp32, C=128: x(256B)+w(32KB)+red(64KB)=96.25KB < 99KB optin limit.
static constexpr int BLOCK_THREADS_APPROX = 64;
static constexpr int T_APPROX             = 128; // latent tile size
static constexpr int D_TILE_APPROX        = 64;
static constexpr int LATENTS_PER_THREAD_APPROX = T_APPROX / BLOCK_THREADS_APPROX;
// Same padding as exact kernel: eliminates 32-way bank conflicts for bf16/fp16.
static constexpr int W_SMEM_PAD_APPROX    = 2;
static constexpr int W_SMEM_STRIDE_APPROX = D_TILE_APPROX + W_SMEM_PAD_APPROX;  // 66

// Kernel: fills candidate_vals and candidate_idxs with top-c per tile per row.
// M12: 2D grid — one block per (row, F-tile), matching the exact kernel's approach.
// blockIdx.x = row, blockIdx.y = F-tile index.
// Eliminates the serial tile loop that left most SMs idle for small B.
template<typename scalar_t, int C>
__global__ void streamtopk_approx_kernel(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ W_enc,
    const float*    __restrict__ b_enc,
    float*          __restrict__ cand_vals,   // (B, num_tiles, C)
    int32_t*        __restrict__ cand_idxs,   // (B, num_tiles, C)
    int64_t B, int64_t d, int64_t F, int64_t num_tiles)
{
    extern __shared__ char smem_raw[];
    scalar_t* x_smem   = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* w_smem   = x_smem + D_TILE_APPROX;
    float*    red_vals = reinterpret_cast<float*>(w_smem + T_APPROX * W_SMEM_STRIDE_APPROX);
    int32_t*  red_idxs = reinterpret_cast<int32_t*>(red_vals + BLOCK_THREADS_APPROX * C);

    int row    = blockIdx.x;
    int tile_t = blockIdx.y;   // F-tile index from 2D grid
    if (row >= (int)B) return;
    int tid = threadIdx.x;

    int f_start = tile_t * T_APPROX;
    int f_end   = min(f_start + T_APPROX, (int)F);
    int tile_f  = f_end - f_start;

    TopKBuffer<C> buf;
    buf.init();

    float partial[LATENTS_PER_THREAD_APPROX];
    #pragma unroll
    for (int li = 0; li < LATENTS_PER_THREAD_APPROX; ++li) partial[li] = 0.0f;

    int num_tiles_D = ((int)d + D_TILE_APPROX - 1) / D_TILE_APPROX;

    for (int dt = 0; dt < num_tiles_D; ++dt) {
        int d_start = dt * D_TILE_APPROX;
        int d_end   = min(d_start + D_TILE_APPROX, (int)d);
        int tile_d  = d_end - d_start;

        if (tid < tile_d) x_smem[tid] = X[row * d + d_start + tid];

        if (tile_d == D_TILE_APPROX) {
            int total_w = tile_f * D_TILE_APPROX;
            for (int idx = tid; idx < total_w; idx += BLOCK_THREADS_APPROX) {
                int fi = idx >> 6;
                int di = idx & 63;
                w_smem[fi * W_SMEM_STRIDE_APPROX + di] = W_enc[(f_start + fi) * d + d_start + di];
            }
        } else {
            for (int idx = tid; idx < tile_f * tile_d; idx += BLOCK_THREADS_APPROX) {
                int fi = idx / tile_d;
                int di = idx % tile_d;
                w_smem[fi * W_SMEM_STRIDE_APPROX + di] = W_enc[(f_start + fi) * d + d_start + di];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int li = 0; li < LATENTS_PER_THREAD_APPROX; ++li) {
            int fi = tid + li * BLOCK_THREADS_APPROX;
            if (fi < tile_f) {
                float acc = 0.0f;
                const scalar_t* wrow = w_smem + fi * W_SMEM_STRIDE_APPROX;
                if (tile_d == D_TILE_APPROX) {
                    #pragma unroll
                    for (int di = 0; di < D_TILE_APPROX; ++di)
                        acc += scalar_to_float(wrow[di]) * scalar_to_float(x_smem[di]);
                } else {
                    for (int di = 0; di < tile_d; ++di)
                        acc += scalar_to_float(wrow[di]) * scalar_to_float(x_smem[di]);
                }
                partial[li] += acc;
            }
        }
        __syncthreads();
    }

    float thr = buf.threshold();
    #pragma unroll
    for (int li = 0; li < LATENTS_PER_THREAD_APPROX; ++li) {
        int fi = tid + li * BLOCK_THREADS_APPROX;
        if (fi < tile_f) {
            float score = partial[li] + b_enc[f_start + fi];
            if (score > thr) {
                buf.try_insert(score, f_start + fi);
                thr = buf.threshold();
            }
        }
    }

    #pragma unroll
    for (int ci = 0; ci < C; ++ci) {
        red_vals[tid * C + ci] = buf.values[ci];
        red_idxs[tid * C + ci] = buf.indices[ci];
    }
    __syncthreads();

    for (int stride = BLOCK_THREADS_APPROX / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            float*   my_vals  = red_vals + tid * C;
            int32_t* my_idxs  = red_idxs + tid * C;
            float*   oth_vals = red_vals + (tid + stride) * C;
            int32_t* oth_idxs = red_idxs + (tid + stride) * C;
            for (int ci = 0; ci < C; ++ci) {
                float ov     = oth_vals[ci];
                float my_min = my_vals[C - 1];
                if (ov > my_min) {
                    my_vals[C - 1]  = ov;
                    my_idxs[C - 1]  = oth_idxs[ci];
                    for (int p = C - 1; p > 0 && my_vals[p] > my_vals[p-1]; --p) {
                        float tv = my_vals[p]; my_vals[p] = my_vals[p-1]; my_vals[p-1] = tv;
                        int   ti = my_idxs[p]; my_idxs[p] = my_idxs[p-1]; my_idxs[p-1] = ti;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t offset = ((int64_t)row * num_tiles + tile_t) * C;
        for (int ci = 0; ci < C; ++ci) {
            cand_vals[offset + ci] = red_vals[ci];
            cand_idxs[offset + ci] = red_idxs[ci];
        }
    }
}

static size_t smem_bytes_approx(int C, bool is_half) {
    size_t scalar_size = is_half ? 2 : 4;
    size_t x_smem   = D_TILE_APPROX * scalar_size;
    size_t w_smem   = T_APPROX * W_SMEM_STRIDE_APPROX * scalar_size;
    size_t red_vals = BLOCK_THREADS_APPROX * C * sizeof(float);
    size_t red_idxs = BLOCK_THREADS_APPROX * C * sizeof(int32_t);
    return x_smem + w_smem + red_vals + red_idxs;
}

// Request extended shared memory for a kernel function pointer.
template<typename KernelFn>
static void set_max_smem_approx(KernelFn fn, size_t smem) {
    int max_smem = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if ((int)smem <= max_smem) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
}

template<int C>
static void launch_approx(
    torch::Tensor& X, torch::Tensor& W_enc, torch::Tensor& b_enc,
    torch::Tensor& cand_vals, torch::Tensor& cand_idxs,
    int64_t B, int64_t d, int64_t F, int64_t num_tiles)
{
    dim3 grid((int)B, (int)num_tiles);   // 2D grid: fill all SMs for small B
    dim3 block(BLOCK_THREADS_APPROX);
    auto dtype = X.scalar_type();
    bool is_half = (dtype != torch::kFloat32);
    size_t smem = smem_bytes_approx(C, is_half);

    if (dtype == torch::kBFloat16) {
        set_max_smem_approx(streamtopk_approx_kernel<__nv_bfloat16, C>, smem);
        streamtopk_approx_kernel<__nv_bfloat16, C><<<grid, block, smem>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(W_enc.data_ptr()),
            b_enc.data_ptr<float>(),
            cand_vals.data_ptr<float>(),
            cand_idxs.data_ptr<int32_t>(),
            B, d, F, num_tiles);
    } else if (dtype == torch::kFloat16) {
        set_max_smem_approx(streamtopk_approx_kernel<__half, C>, smem);
        streamtopk_approx_kernel<__half, C><<<grid, block, smem>>>(
            reinterpret_cast<const __half*>(X.data_ptr()),
            reinterpret_cast<const __half*>(W_enc.data_ptr()),
            b_enc.data_ptr<float>(),
            cand_vals.data_ptr<float>(),
            cand_idxs.data_ptr<int32_t>(),
            B, d, F, num_tiles);
    } else {
        set_max_smem_approx(streamtopk_approx_kernel<float, C>, smem);
        streamtopk_approx_kernel<float, C><<<grid, block, smem>>>(
            X.data_ptr<float>(),
            W_enc.data_ptr<float>(),
            b_enc.data_ptr<float>(),
            cand_vals.data_ptr<float>(),
            cand_idxs.data_ptr<int32_t>(),
            B, d, F, num_tiles);
    }
    C10_CUDA_CHECK(cudaGetLastError());
}

// Returns: candidate_vals (B, num_tiles*c), candidate_idxs (B, num_tiles*c)
// Python picks top-k from these.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
streamtopk_cuda_approx_forward(
    torch::Tensor X,
    torch::Tensor W_enc,
    torch::Tensor b_enc,
    int64_t k,
    int64_t c,
    torch::optional<torch::Tensor> candidate_buffer)
{
    TORCH_CHECK(X.is_contiguous(),     "X must be contiguous");
    TORCH_CHECK(W_enc.is_contiguous(), "W_enc must be contiguous");
    TORCH_CHECK(b_enc.is_contiguous(), "b_enc must be contiguous");
    TORCH_CHECK(X.is_cuda(),           "X must be on CUDA");
    TORCH_CHECK(b_enc.dtype() == torch::kFloat32, "b_enc must be fp32");

    auto dtype = X.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "X dtype must be fp32, fp16, or bf16");
    TORCH_CHECK(W_enc.scalar_type() == dtype, "W_enc dtype must match X");
    TORCH_CHECK(k >= 1 && k <= c, "k must be <= c");

    int64_t B = X.size(0);
    int64_t d = X.size(1);
    int64_t F = W_enc.size(0);
    TORCH_CHECK(W_enc.size(1) == d, "W_enc d-dim mismatch");
    TORCH_CHECK(b_enc.size(0) == F, "b_enc F-dim mismatch");

    int64_t num_tiles = (F + T_APPROX - 1) / T_APPROX;

    // Allocate or reuse candidate buffer
    torch::Tensor cand_vals, cand_idxs;
    if (candidate_buffer.has_value()) {
        auto& cb = candidate_buffer.value();
        TORCH_CHECK(cb.size(0) == B && cb.size(1) == num_tiles && cb.size(2) == c,
                    "candidate_buffer shape mismatch");
        cand_vals = cb;
        cand_idxs = torch::empty({B, num_tiles, c}, X.options().dtype(torch::kInt32));
    } else {
        cand_vals = torch::empty({B, num_tiles, c}, X.options().dtype(torch::kFloat32));
        cand_idxs = torch::empty({B, num_tiles, c}, X.options().dtype(torch::kInt32));
    }

    switch (c) {
        case 16:  launch_approx<16> (X, W_enc, b_enc, cand_vals, cand_idxs, B, d, F, num_tiles); break;
        case 32:  launch_approx<32> (X, W_enc, b_enc, cand_vals, cand_idxs, B, d, F, num_tiles); break;
        case 64:  launch_approx<64> (X, W_enc, b_enc, cand_vals, cand_idxs, B, d, F, num_tiles); break;
        case 128: launch_approx<128>(X, W_enc, b_enc, cand_vals, cand_idxs, B, d, F, num_tiles); break;
        default:
            TORCH_CHECK(false, "streamtopk_cuda_approx: unsupported c=", c,
                        ". Supported: 16,32,64,128.");
    }

    // Flatten candidates and return for Python-side topk
    auto flat_vals = cand_vals.view({B, num_tiles * c});
    auto flat_idxs = cand_idxs.view({B, num_tiles * c});

    return {flat_vals, flat_idxs, cand_vals, cand_idxs};
}

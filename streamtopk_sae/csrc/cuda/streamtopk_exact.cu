/*
 * CUDA exact streaming top-k SAE forward pass.
 *
 * Two-pass implementation (M11/M12):
 *
 *   Pass 1 (tile kernel): Grid (B, num_tiles_F) — one block per (row, F-tile).
 *     Each block accumulates dot products for T latents across all d-tiles,
 *     runs a block-wide top-K reduction, and writes K candidates to a temp
 *     buffer of shape (B, num_tiles_F, K).
 *
 *   Pass 2 (merge kernel): Grid (B,) — each block reads num_tiles_F × K
 *     candidates and reduces them to the final top-K output.
 *
 * M12 additions:
 *   - wmma tensor-core path for bf16 + K≤16 + B multiple of 16 (3.26× speedup).
 *   - 2D grid for cuda_approx (see streamtopk_approx.cu).
 *
 * Tile kernel uses single-buffer smem (same as M11): ~48 KB for bf16/K=32,
 * allowing 2 blocks/SM on sm_86. Double-buffering was tested (M12 draft) but
 * raised smem to ~80 KB → 1 block/SM → 0.55–0.68× regression for K=32.
 *
 * Smem budget: D_TILE=64, T=256 (bf16/fp16), T=128 (fp32).
 *   bf16/K=32: 128B + 32KB + 16KB = ~48KB (2 blocks/SM ✓)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "cuda/topk_buffer.cuh"

using namespace nvcuda::wmma;

static constexpr int BLOCK_THREADS = 64;
static constexpr int D_TILE        = 64;
static constexpr int D_MAX         = 4096;
static constexpr int T_HALF  = 256;
static constexpr int T_FLOAT = 128;
// Pad each w_smem row by 2 half-precision elements (4 bytes).
// Row stride becomes 66 elements; bank stride = 66/2 = 33 (odd) → zero 32-bank
// conflicts for bf16/fp16, and a large reduction for fp32.
static constexpr int W_SMEM_PAD    = 2;
static constexpr int W_SMEM_STRIDE = D_TILE + W_SMEM_PAD;   // 66

__device__ __forceinline__ float to_f32(float v)         { return v; }
__device__ __forceinline__ float to_f32(__half v)        { return __half2float(v); }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

// ---------------------------------------------------------------------------
// Pass 1: tile kernel (single-buffer).
// blockIdx.x = row, blockIdx.y = F-tile index.
//
// Smem layout: x_smem[D_TILE] | w_smem[T*D_TILE] | red_vals[BT*K] | red_idxs[BT*K]
// For bf16, T=256, K=32: 128B + 32KB + 16KB ≈ 48KB → 2 blocks/SM on sm_86 ✓
// ---------------------------------------------------------------------------
template<typename scalar_t, int K, int T>
__global__ void streamtopk_exact_tile_kernel(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ W_enc,
    const float*    __restrict__ b_enc,
    float*          __restrict__ tile_vals,
    int32_t*        __restrict__ tile_idxs,
    int64_t B, int64_t d, int64_t F, int64_t num_tiles_F)
{
    constexpr int LPT = T / BLOCK_THREADS;

    extern __shared__ char smem_raw[];
    scalar_t* x_smem   = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* w_smem   = x_smem + D_TILE;
    float*    red_vals = reinterpret_cast<float*>(w_smem + T * W_SMEM_STRIDE);
    int32_t*  red_idxs = reinterpret_cast<int32_t*>(red_vals + BLOCK_THREADS * K);

    int row    = blockIdx.x;
    int tile_t = blockIdx.y;
    if (row >= (int)B) return;
    int tid = threadIdx.x;

    int f_start = tile_t * T;
    int f_end   = min(f_start + T, (int)F);
    int tile_f  = f_end - f_start;

    TopKBuffer<K> buf;
    buf.init();

    const int num_tiles_D = ((int)d + D_TILE - 1) / D_TILE;

    float partial[LPT];
    #pragma unroll
    for (int li = 0; li < LPT; ++li) partial[li] = 0.0f;

    for (int dt = 0; dt < num_tiles_D; ++dt) {
        int d_start = dt * D_TILE;
        int d_end   = min(d_start + D_TILE, (int)d);
        int tile_d  = d_end - d_start;

        if (tid < tile_d)
            x_smem[tid] = X[row * d + d_start + tid];

        if (tile_d == D_TILE) {
            int total_w = tile_f * D_TILE;
            for (int idx = tid; idx < total_w; idx += BLOCK_THREADS) {
                int fi = idx >> 6;
                int di = idx & 63;
                w_smem[fi * W_SMEM_STRIDE + di] = W_enc[(f_start + fi) * d + d_start + di];
            }
        } else {
            for (int idx = tid; idx < tile_f * tile_d; idx += BLOCK_THREADS) {
                int fi = idx / tile_d;
                int di = idx % tile_d;
                w_smem[fi * W_SMEM_STRIDE + di] = W_enc[(f_start + fi) * d + d_start + di];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int li = 0; li < LPT; ++li) {
            int fi = tid + li * BLOCK_THREADS;
            if (fi < tile_f) {
                float acc = 0.0f;
                const scalar_t* wrow = w_smem + fi * W_SMEM_STRIDE;
                if (tile_d == D_TILE) {
                    #pragma unroll
                    for (int di = 0; di < D_TILE; ++di)
                        acc += to_f32(wrow[di]) * to_f32(x_smem[di]);
                } else {
                    for (int di = 0; di < tile_d; ++di)
                        acc += to_f32(wrow[di]) * to_f32(x_smem[di]);
                }
                partial[li] += acc;
            }
        }
        __syncthreads();
    }

    float thr = buf.threshold();
    #pragma unroll
    for (int li = 0; li < LPT; ++li) {
        int fi = tid + li * BLOCK_THREADS;
        if (fi < tile_f) {
            float score = partial[li] + b_enc[f_start + fi];
            if (score > thr) {
                buf.try_insert(score, f_start + fi);
                thr = buf.threshold();
            }
        }
    }

    // Block-wide reduction
    #pragma unroll
    for (int ki = 0; ki < K; ++ki) {
        red_vals[tid * K + ki] = buf.values[ki];
        red_idxs[tid * K + ki] = buf.indices[ki];
    }
    __syncthreads();

    for (int stride = BLOCK_THREADS / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            float*   my_vals  = red_vals + tid * K;
            int32_t* my_idxs  = red_idxs + tid * K;
            float*   oth_vals = red_vals + (tid + stride) * K;
            int32_t* oth_idxs = red_idxs + (tid + stride) * K;
            for (int ki = 0; ki < K; ++ki) {
                float ov     = oth_vals[ki];
                float my_min = my_vals[K - 1];
                if (ov > my_min) {
                    my_vals[K - 1]  = ov;
                    my_idxs[K - 1]  = oth_idxs[ki];
                    for (int p = K - 1; p > 0 && my_vals[p] > my_vals[p-1]; --p) {
                        float tv = my_vals[p]; my_vals[p] = my_vals[p-1]; my_vals[p-1] = tv;
                        int   ti = my_idxs[p]; my_idxs[p] = my_idxs[p-1]; my_idxs[p-1] = ti;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t off = ((int64_t)row * num_tiles_F + tile_t) * K;
        for (int ki = 0; ki < K; ++ki) {
            tile_vals[off + ki] = red_vals[ki];
            tile_idxs[off + ki] = red_idxs[ki];
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 1 (wmma variant): processes WARP_ROWS=16 batch rows per block using
// tensor core wmma 16×16×16 for the inner dot product.
//
// Layout: grid (ceil(B/16), num_tiles_F).
// Each warp computes one C[16latents × 16batchrows] fragment per d-sub-tile.
// With 2 warps (BLOCK_THREADS=64), warp 0 handles latent-groups 0,2,4,...
// and warp 1 handles 1,3,5,...  sharing the same x_smem and w_smem.
//
// Active when: dtype==bf16, K<=16, B is a multiple of 16.
// For K=16: 16 TopKBuffers × 32 regs = 512 regs/thread — within 1 block/SM limit.
// Falls back to scalar tile kernel otherwise (see launch_exact).
// ---------------------------------------------------------------------------
static constexpr int WARP_ROWS  = 16;   // batch rows processed per block
static constexpr int WMMA_M     = 16;
static constexpr int WMMA_N     = 16;
static constexpr int WMMA_K     = 16;
static constexpr int WARP_SIZE  = 32;

template<int K, int T>
__global__ void streamtopk_exact_tile_wmma_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W_enc,
    const float*         __restrict__ b_enc,
    float*               __restrict__ tile_vals,
    int32_t*             __restrict__ tile_idxs,
    int64_t B, int64_t d, int64_t F, int64_t num_tiles_F)
{
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;   // 2
    // Each warp handles T/(WMMA_M*WARPS_PER_BLOCK) latent-groups per d-tile.
    // T=256, WMMA_M=16, WARPS=2 → 256/16/2 = 8 latent-groups per warp per d-tile.
    constexpr int LAT_GROUPS_PER_WARP = T / (WMMA_M * WARPS_PER_BLOCK);

    extern __shared__ char smem_raw[];
    // Layout: x_smem[WARP_ROWS * D_TILE] | w_smem[T * D_TILE] | red[BT*K*8]
    __nv_bfloat16* x_smem   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* w_smem   = x_smem + WARP_ROWS * D_TILE;
    float*         red_vals = reinterpret_cast<float*>(w_smem + T * W_SMEM_STRIDE);
    int32_t*       red_idxs = reinterpret_cast<int32_t*>(red_vals + BLOCK_THREADS * K);

    int base_row = blockIdx.x * WARP_ROWS;
    int tile_t   = blockIdx.y;
    if (base_row >= (int)B) return;
    int tid      = threadIdx.x;
    int warp_id  = tid / WARP_SIZE;
    int lane     = tid % WARP_SIZE;

    int f_start = tile_t * T;
    int f_end   = min(f_start + T, (int)F);
    int tile_f  = f_end - f_start;

    // Per-thread top-K buffers for each of the WARP_ROWS batch rows.
    TopKBuffer<K> buf[WARP_ROWS];
    #pragma unroll
    for (int r = 0; r < WARP_ROWS; ++r) buf[r].init();

    // Accumulator fragments: one per latent-group this warp owns, over all d.
    // After all d-tiles these hold the full scores[WMMA_M latents × WMMA_N batchrows].
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[LAT_GROUPS_PER_WARP];
    #pragma unroll
    for (int g = 0; g < LAT_GROUPS_PER_WARP; ++g)
        fill_fragment(acc_frag[g], 0.0f);

    const int num_tiles_D = ((int)d + D_TILE - 1) / D_TILE;

    for (int dt = 0; dt < num_tiles_D; ++dt) {
        int d_start = dt * D_TILE;
        int d_end   = min(d_start + D_TILE, (int)d);
        int tile_d  = d_end - d_start;

        // Load x_smem[WARP_ROWS × tile_d]: all threads cooperate.
        // Thread tid loads element (tid/D_TILE, tid%D_TILE) when D_TILE==64.
        if (tile_d == D_TILE) {
            // 64 threads × WARP_ROWS=16 elements each → loop
            for (int r = 0; r < WARP_ROWS; ++r) {
                int row_r = base_row + r;
                if (row_r < (int)B && tid < D_TILE)
                    x_smem[r * D_TILE + tid] = X[row_r * d + d_start + tid];
            }
        } else {
            for (int r = 0; r < WARP_ROWS; ++r) {
                int row_r = base_row + r;
                if (row_r < (int)B && tid < tile_d)
                    x_smem[r * D_TILE + tid] = X[row_r * d + d_start + tid];
            }
        }

        // Load w_smem[tile_f × D_TILE]
        if (tile_d == D_TILE) {
            int total_w = tile_f * D_TILE;
            for (int idx = tid; idx < total_w; idx += BLOCK_THREADS) {
                int fi = idx >> 6;
                int di = idx & 63;
                w_smem[fi * W_SMEM_STRIDE + di] = W_enc[(f_start + fi) * d + d_start + di];
            }
        } else {
            for (int idx = tid; idx < tile_f * tile_d; idx += BLOCK_THREADS) {
                int fi = idx / tile_d;
                int di = idx % tile_d;
                w_smem[fi * W_SMEM_STRIDE + di] = W_enc[(f_start + fi) * d + d_start + di];
            }
        }
        __syncthreads();

        // wmma: iterate over WMMA_K=16 sub-tiles within D_TILE=64 (4 sub-tiles).
        // Each warp accumulates LAT_GROUPS_PER_WARP=8 acc_frags over d.
        int num_dk = tile_d / WMMA_K;  // 64/16=4 for full tile
        for (int dk = 0; dk < num_dk; ++dk) {
            // B-fragment: X[WARP_ROWS × WMMA_K] in col_major
            // x_smem layout: [r][di], stride=D_TILE; col_major B means B[k][n]=x_smem[dk*16+k + n*D_TILE]
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
            load_matrix_sync(b_frag, x_smem + dk * WMMA_K, D_TILE);

            #pragma unroll
            for (int g = 0; g < LAT_GROUPS_PER_WARP; ++g) {
                // A-fragment: W_enc[WMMA_M latents × WMMA_K d-sub], row_major
                int lat_group = warp_id + g * WARPS_PER_BLOCK;
                if (lat_group * WMMA_M < tile_f) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                    load_matrix_sync(a_frag,
                        w_smem + lat_group * WMMA_M * W_SMEM_STRIDE + dk * WMMA_K,
                        W_SMEM_STRIDE);
                    mma_sync(acc_frag[g], a_frag, b_frag, acc_frag[g]);
                }
            }
        }
        __syncthreads();
    }

    // Extract scores from accumulator fragments and update TopKBuffers.
    // Each warp stores its acc_frags to smem so all threads can access them.
    // Reuse red_vals as temp storage (BT*K*8 >= WARPS*LAT_GROUPS*WMMA_M*WMMA_N*4).
    float* score_tmp = red_vals;  // temporary reuse before reduction writes

    #pragma unroll
    for (int g = 0; g < LAT_GROUPS_PER_WARP; ++g) {
        int lat_group = warp_id + g * WARPS_PER_BLOCK;
        int lat_base  = lat_group * WMMA_M;
        if (lat_base < tile_f) {
            // Store fragment to smem: layout [WMMA_M × WMMA_N]
            float* dst = score_tmp + (warp_id * LAT_GROUPS_PER_WARP + g) * WMMA_M * WMMA_N;
            store_matrix_sync(dst, acc_frag[g], WMMA_N, mem_row_major);
        }
    }
    __syncthreads();

    // Each thread reads scores for its assigned latents from score_tmp and updates buffers.
    // tid in [0,64): thread tid handles latent (tid % 32) within a warp's group,
    // but we simplify: iterate over all latent groups and batch rows.
    for (int g = 0; g < LAT_GROUPS_PER_WARP * WARPS_PER_BLOCK; ++g) {
        int owning_warp = g % WARPS_PER_BLOCK;
        int local_g     = g / WARPS_PER_BLOCK;
        int lat_base    = (owning_warp + local_g * WARPS_PER_BLOCK) * WMMA_M;
        if (lat_base >= tile_f) continue;

        float* scores_g = score_tmp + (owning_warp * LAT_GROUPS_PER_WARP + local_g) * WMMA_M * WMMA_N;

        // Each thread updates TopKBuffers for its "row stripe"
        int rows_per_thread = (WARP_ROWS + BLOCK_THREADS - 1) / BLOCK_THREADS;
        for (int rr = 0; rr < rows_per_thread; ++rr) {
            int r = tid * rows_per_thread + rr;
            if (r >= WARP_ROWS || base_row + r >= (int)B) continue;
            float thr = buf[r].threshold();
            for (int li = 0; li < WMMA_M; ++li) {
                int fi = f_start + lat_base + li;
                if (lat_base + li < tile_f) {
                    float score = scores_g[li * WMMA_N + r] + b_enc[fi];
                    if (score > thr) {
                        buf[r].try_insert(score, fi);
                        thr = buf[r].threshold();
                    }
                }
            }
        }
    }
    __syncthreads();

    // Block-wide reduction and output: one TopKBuffer<K> per batch row.
    // Write all rows' buffers to red_vals/red_idxs stacked as [WARP_ROWS][BT][K].
    // This is large; only do the merge for rows that exist.
    for (int r = 0; r < WARP_ROWS; ++r) {
        int row_r = base_row + r;
        if (row_r >= (int)B) break;

        // Write this row's per-thread buffer
        #pragma unroll
        for (int ki = 0; ki < K; ++ki) {
            red_vals[tid * K + ki] = buf[r].values[ki];
            red_idxs[tid * K + ki] = buf[r].indices[ki];
        }
        __syncthreads();

        for (int stride = BLOCK_THREADS / 2; stride >= 1; stride >>= 1) {
            if (tid < stride) {
                float*   my_vals  = red_vals + tid * K;
                int32_t* my_idxs  = red_idxs + tid * K;
                float*   oth_vals = red_vals + (tid + stride) * K;
                int32_t* oth_idxs = red_idxs + (tid + stride) * K;
                for (int ki = 0; ki < K; ++ki) {
                    float ov = oth_vals[ki];
                    if (ov > my_vals[K - 1]) {
                        my_vals[K - 1] = ov;
                        my_idxs[K - 1] = oth_idxs[ki];
                        for (int p = K-1; p > 0 && my_vals[p] > my_vals[p-1]; --p) {
                            float tv = my_vals[p]; my_vals[p] = my_vals[p-1]; my_vals[p-1] = tv;
                            int   ti = my_idxs[p]; my_idxs[p] = my_idxs[p-1]; my_idxs[p-1] = ti;
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            int64_t off = ((int64_t)row_r * num_tiles_F + tile_t) * K;
            for (int ki = 0; ki < K; ++ki) {
                tile_vals[off + ki] = red_vals[ki];
                tile_idxs[off + ki] = red_idxs[ki];
            }
        }
        __syncthreads();
    }
}

// smem for wmma tile kernel
template<int T_VAL>
static size_t smem_bytes_wmma(int K) {
    constexpr int WARPS = BLOCK_THREADS / WARP_SIZE;
    constexpr int LATS  = T_VAL / (WMMA_M * WARPS);
    size_t x_bytes   = WARP_ROWS * D_TILE * 2;
    size_t w_bytes   = (size_t)T_VAL * W_SMEM_STRIDE * 2;
    size_t score_tmp = (size_t)WARPS * LATS * WMMA_M * WMMA_N * 4;
    size_t red_bytes = (size_t)BLOCK_THREADS * K * (sizeof(float) + sizeof(int32_t));
    return x_bytes + w_bytes + std::max(score_tmp, red_bytes);
}

// ---------------------------------------------------------------------------
// Pass 2: merge kernel.
// blockIdx.x = row.  Reads num_tiles_F * K candidates and reduces to top-K.
// ---------------------------------------------------------------------------
template<int K>
__global__ void streamtopk_exact_merge_kernel(
    const float*    __restrict__ tile_vals,   // (B, num_tiles_F, K)
    const int32_t*  __restrict__ tile_idxs,   // (B, num_tiles_F, K)
    float*          __restrict__ V_out,
    int32_t*        __restrict__ I_out,
    int64_t B, int64_t num_tiles_F, int64_t k)
{
    extern __shared__ char smem_raw[];
    float*   red_vals = reinterpret_cast<float*>(smem_raw);
    int32_t* red_idxs = reinterpret_cast<int32_t*>(red_vals + BLOCK_THREADS * K);

    int row = blockIdx.x;
    if (row >= (int)B) return;
    int tid = threadIdx.x;

    TopKBuffer<K> buf;
    buf.init();

    int64_t total_cands = num_tiles_F * K;
    const float*   rv = tile_vals + row * total_cands;
    const int32_t* ri = tile_idxs + row * total_cands;

    float thr = buf.threshold();
    for (int64_t ci = tid; ci < total_cands; ci += BLOCK_THREADS) {
        float v = rv[ci];
        if (v > thr) {
            buf.try_insert(v, ri[ci]);
            thr = buf.threshold();
        }
    }

    // Block-wide reduction
    #pragma unroll
    for (int ki = 0; ki < K; ++ki) {
        red_vals[tid * K + ki] = buf.values[ki];
        red_idxs[tid * K + ki] = buf.indices[ki];
    }
    __syncthreads();

    for (int stride = BLOCK_THREADS / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            float*   my_vals  = red_vals + tid * K;
            int32_t* my_idxs  = red_idxs + tid * K;
            float*   oth_vals = red_vals + (tid + stride) * K;
            int32_t* oth_idxs = red_idxs + (tid + stride) * K;
            for (int ki = 0; ki < K; ++ki) {
                float ov     = oth_vals[ki];
                float my_min = my_vals[K - 1];
                if (ov > my_min) {
                    my_vals[K - 1]  = ov;
                    my_idxs[K - 1]  = oth_idxs[ki];
                    for (int p = K - 1; p > 0 && my_vals[p] > my_vals[p-1]; --p) {
                        float tv = my_vals[p]; my_vals[p] = my_vals[p-1]; my_vals[p-1] = tv;
                        int   ti = my_idxs[p]; my_idxs[p] = my_idxs[p-1]; my_idxs[p-1] = ti;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float*   vi = V_out + row * k;
        int32_t* ii = I_out + row * k;
        for (int ki = 0; ki < (int)k; ++ki) {
            vi[ki] = red_vals[ki];
            ii[ki] = red_idxs[ki];
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 1 (single-pass fallback): original M10 kernel — grid (B,).
// Used when B >= num_SMs so the GPU is already well-utilized and temp-buffer
// overhead from the two-pass approach would be wasteful.
// ---------------------------------------------------------------------------
template<typename scalar_t, int K, int T>
__global__ void streamtopk_exact_single_kernel(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ W_enc,
    const float*    __restrict__ b_enc,
    float*          __restrict__ V_out,
    int32_t*        __restrict__ I_out,
    int64_t B, int64_t d, int64_t F, int64_t k)
{
    constexpr int LPT = T / BLOCK_THREADS;

    extern __shared__ char smem_raw[];
    scalar_t* x_smem   = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* w_smem   = x_smem + D_TILE;
    float*    red_vals = reinterpret_cast<float*>(w_smem + T * W_SMEM_STRIDE);
    int32_t*  red_idxs = reinterpret_cast<int32_t*>(red_vals + BLOCK_THREADS * K);

    int row = blockIdx.x;
    if (row >= (int)B) return;
    int tid = threadIdx.x;

    TopKBuffer<K> buf;
    buf.init();

    const int num_tiles_F = ((int)F + T - 1) / T;
    const int num_tiles_D = ((int)d + D_TILE - 1) / D_TILE;

    float partial[LPT];

    for (int t = 0; t < num_tiles_F; ++t) {
        int f_start = t * T;
        int f_end   = min(f_start + T, (int)F);
        int tile_f  = f_end - f_start;

        #pragma unroll
        for (int li = 0; li < LPT; ++li) partial[li] = 0.0f;

        for (int dt = 0; dt < num_tiles_D; ++dt) {
            int d_start = dt * D_TILE;
            int d_end   = min(d_start + D_TILE, (int)d);
            int tile_d  = d_end - d_start;

            if (tid < tile_d)
                x_smem[tid] = X[row * d + d_start + tid];

            if (tile_d == D_TILE) {
                int total_w = tile_f * D_TILE;
                for (int idx = tid; idx < total_w; idx += BLOCK_THREADS) {
                    int fi = idx >> 6;
                    int di = idx & 63;
                    w_smem[fi * W_SMEM_STRIDE + di] = W_enc[(f_start + fi) * d + d_start + di];
                }
            } else {
                for (int idx = tid; idx < tile_f * tile_d; idx += BLOCK_THREADS) {
                    int fi = idx / tile_d;
                    int di = idx % tile_d;
                    w_smem[fi * W_SMEM_STRIDE + di] = W_enc[(f_start + fi) * d + d_start + di];
                }
            }
            __syncthreads();

            #pragma unroll
            for (int li = 0; li < LPT; ++li) {
                int fi = tid + li * BLOCK_THREADS;
                if (fi < tile_f) {
                    float acc = 0.0f;
                    const scalar_t* wrow = w_smem + fi * W_SMEM_STRIDE;
                    if (tile_d == D_TILE) {
                        #pragma unroll
                        for (int di = 0; di < D_TILE; ++di)
                            acc += to_f32(wrow[di]) * to_f32(x_smem[di]);
                    } else {
                        for (int di = 0; di < tile_d; ++di)
                            acc += to_f32(wrow[di]) * to_f32(x_smem[di]);
                    }
                    partial[li] += acc;
                }
            }
            __syncthreads();
        }

        float thr = buf.threshold();
        #pragma unroll
        for (int li = 0; li < LPT; ++li) {
            int fi = tid + li * BLOCK_THREADS;
            if (fi < tile_f) {
                float score = partial[li] + b_enc[f_start + fi];
                if (score > thr) {
                    buf.try_insert(score, f_start + fi);
                    thr = buf.threshold();
                }
            }
        }
    }

    #pragma unroll
    for (int ki = 0; ki < K; ++ki) {
        red_vals[tid * K + ki] = buf.values[ki];
        red_idxs[tid * K + ki] = buf.indices[ki];
    }
    __syncthreads();

    for (int stride = BLOCK_THREADS / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            float*   my_vals  = red_vals + tid * K;
            int32_t* my_idxs  = red_idxs + tid * K;
            float*   oth_vals = red_vals + (tid + stride) * K;
            int32_t* oth_idxs = red_idxs + (tid + stride) * K;
            for (int ki = 0; ki < K; ++ki) {
                float ov     = oth_vals[ki];
                float my_min = my_vals[K - 1];
                if (ov > my_min) {
                    my_vals[K - 1]  = ov;
                    my_idxs[K - 1]  = oth_idxs[ki];
                    for (int p = K - 1; p > 0 && my_vals[p] > my_vals[p-1]; --p) {
                        float tv = my_vals[p]; my_vals[p] = my_vals[p-1]; my_vals[p-1] = tv;
                        int   ti = my_idxs[p]; my_idxs[p] = my_idxs[p-1]; my_idxs[p-1] = ti;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float*   vi = V_out + row * k;
        int32_t* ii = I_out + row * k;
        for (int ki = 0; ki < (int)k; ++ki) {
            vi[ki] = red_vals[ki];
            ii[ki] = red_idxs[ki];
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static size_t smem_bytes_tile(int K, int T, bool is_half) {
    size_t ss = is_half ? 2 : 4;
    return D_TILE * ss + (size_t)T * W_SMEM_STRIDE * ss +
           (size_t)BLOCK_THREADS * K * (sizeof(float) + sizeof(int32_t));
}

static size_t smem_bytes_merge(int K) {
    return (size_t)BLOCK_THREADS * K * (sizeof(float) + sizeof(int32_t));
}

template<typename KernelFn>
static void set_max_smem(KernelFn fn, size_t smem) {
    int max_smem = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if ((int)smem <= max_smem)
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
}

template<int K>
static void launch_exact(
    torch::Tensor& X, torch::Tensor& W_enc, torch::Tensor& b_enc,
    torch::Tensor& V, torch::Tensor& I,
    int64_t B, int64_t d, int64_t F, int64_t k)
{
    auto dtype   = X.scalar_type();
    bool is_half = (dtype != torch::kFloat32);
    int  T       = is_half ? T_HALF : T_FLOAT;
    int64_t num_tiles_F = (F + T - 1) / T;
    dim3 block(BLOCK_THREADS);

    // Two-pass (tile + merge) when B < num_SMs: the 2D grid fills idle SMs.
    // Single-pass (M10) when B >= num_SMs: GPU already saturated, temp-buffer
    // overhead from the two-pass approach would outweigh the benefit.
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    bool use_two_pass = (B < num_sms);

    // wmma path: bf16, K<=16, B a multiple of WARP_ROWS=16
    bool use_wmma = (dtype == torch::kBFloat16) && (K <= 16) && (B % WARP_ROWS == 0);

    if (use_two_pass) {
        auto tile_vals = torch::empty({B, num_tiles_F, K},
                                      X.options().dtype(torch::kFloat32));
        auto tile_idxs = torch::empty({B, num_tiles_F, K},
                                      X.options().dtype(torch::kInt32));

        if (use_wmma) {
            // wmma grid: (B/WARP_ROWS, num_tiles_F)
            dim3 grid_wmma((int)(B / WARP_ROWS), (int)num_tiles_F);
            size_t smem_w = smem_bytes_wmma<T_HALF>(K);
            set_max_smem(streamtopk_exact_tile_wmma_kernel<K, T_HALF>, smem_w);
            streamtopk_exact_tile_wmma_kernel<K, T_HALF><<<grid_wmma, block, smem_w>>>(
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(W_enc.data_ptr()),
                b_enc.data_ptr<float>(),
                tile_vals.data_ptr<float>(), tile_idxs.data_ptr<int32_t>(),
                B, d, F, num_tiles_F);
            C10_CUDA_CHECK(cudaGetLastError());

            dim3 grid2((int)B);
            size_t smem_m = smem_bytes_merge(K);
            set_max_smem(streamtopk_exact_merge_kernel<K>, smem_m);
            streamtopk_exact_merge_kernel<K><<<grid2, block, smem_m>>>(
                tile_vals.data_ptr<float>(), tile_idxs.data_ptr<int32_t>(),
                V.data_ptr<float>(), I.data_ptr<int32_t>(),
                B, num_tiles_F, k);
            C10_CUDA_CHECK(cudaGetLastError());
            return;
        }

        dim3 grid1((int)B, (int)num_tiles_F);

        if (dtype == torch::kBFloat16) {
            size_t smem = smem_bytes_tile(K, T_HALF, true);
            set_max_smem(streamtopk_exact_tile_kernel<__nv_bfloat16, K, T_HALF>, smem);
            streamtopk_exact_tile_kernel<__nv_bfloat16, K, T_HALF><<<grid1, block, smem>>>(
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(W_enc.data_ptr()),
                b_enc.data_ptr<float>(),
                tile_vals.data_ptr<float>(), tile_idxs.data_ptr<int32_t>(),
                B, d, F, num_tiles_F);
        } else if (dtype == torch::kFloat16) {
            size_t smem = smem_bytes_tile(K, T_HALF, true);
            set_max_smem(streamtopk_exact_tile_kernel<__half, K, T_HALF>, smem);
            streamtopk_exact_tile_kernel<__half, K, T_HALF><<<grid1, block, smem>>>(
                reinterpret_cast<const __half*>(X.data_ptr()),
                reinterpret_cast<const __half*>(W_enc.data_ptr()),
                b_enc.data_ptr<float>(),
                tile_vals.data_ptr<float>(), tile_idxs.data_ptr<int32_t>(),
                B, d, F, num_tiles_F);
        } else {
            size_t smem = smem_bytes_tile(K, T_FLOAT, false);
            set_max_smem(streamtopk_exact_tile_kernel<float, K, T_FLOAT>, smem);
            streamtopk_exact_tile_kernel<float, K, T_FLOAT><<<grid1, block, smem>>>(
                X.data_ptr<float>(), W_enc.data_ptr<float>(), b_enc.data_ptr<float>(),
                tile_vals.data_ptr<float>(), tile_idxs.data_ptr<int32_t>(),
                B, d, F, num_tiles_F);
        }
        C10_CUDA_CHECK(cudaGetLastError());

        dim3 grid2((int)B);
        size_t smem_m = smem_bytes_merge(K);
        set_max_smem(streamtopk_exact_merge_kernel<K>, smem_m);
        streamtopk_exact_merge_kernel<K><<<grid2, block, smem_m>>>(
            tile_vals.data_ptr<float>(), tile_idxs.data_ptr<int32_t>(),
            V.data_ptr<float>(), I.data_ptr<int32_t>(),
            B, num_tiles_F, k);
        C10_CUDA_CHECK(cudaGetLastError());

    } else {
        // Single-pass: grid (B,), one block per row
        dim3 grid1((int)B);
        if (dtype == torch::kBFloat16) {
            size_t smem = smem_bytes_tile(K, T_HALF, true);
            set_max_smem(streamtopk_exact_single_kernel<__nv_bfloat16, K, T_HALF>, smem);
            streamtopk_exact_single_kernel<__nv_bfloat16, K, T_HALF><<<grid1, block, smem>>>(
                reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(W_enc.data_ptr()),
                b_enc.data_ptr<float>(), V.data_ptr<float>(), I.data_ptr<int32_t>(),
                B, d, F, k);
        } else if (dtype == torch::kFloat16) {
            size_t smem = smem_bytes_tile(K, T_HALF, true);
            set_max_smem(streamtopk_exact_single_kernel<__half, K, T_HALF>, smem);
            streamtopk_exact_single_kernel<__half, K, T_HALF><<<grid1, block, smem>>>(
                reinterpret_cast<const __half*>(X.data_ptr()),
                reinterpret_cast<const __half*>(W_enc.data_ptr()),
                b_enc.data_ptr<float>(), V.data_ptr<float>(), I.data_ptr<int32_t>(),
                B, d, F, k);
        } else {
            size_t smem = smem_bytes_tile(K, T_FLOAT, false);
            set_max_smem(streamtopk_exact_single_kernel<float, K, T_FLOAT>, smem);
            streamtopk_exact_single_kernel<float, K, T_FLOAT><<<grid1, block, smem>>>(
                X.data_ptr<float>(), W_enc.data_ptr<float>(), b_enc.data_ptr<float>(),
                V.data_ptr<float>(), I.data_ptr<int32_t>(),
                B, d, F, k);
        }
        C10_CUDA_CHECK(cudaGetLastError());
    }
}

// ---------------------------------------------------------------------------
// Public entry point (API unchanged)
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> streamtopk_cuda_exact_forward(
    torch::Tensor X,
    torch::Tensor W_enc,
    torch::Tensor b_enc,
    int64_t k)
{
    TORCH_CHECK(X.is_contiguous(),     "X must be contiguous");
    TORCH_CHECK(W_enc.is_contiguous(), "W_enc must be contiguous");
    TORCH_CHECK(b_enc.is_contiguous(), "b_enc must be contiguous");
    TORCH_CHECK(X.is_cuda(),           "X must be on CUDA");
    TORCH_CHECK(W_enc.is_cuda(),       "W_enc must be on CUDA");
    TORCH_CHECK(b_enc.is_cuda(),       "b_enc must be on CUDA");
    TORCH_CHECK(b_enc.dtype() == torch::kFloat32, "b_enc must be fp32");

    auto dtype = X.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "X dtype must be fp32, fp16, or bf16");
    TORCH_CHECK(W_enc.scalar_type() == dtype, "W_enc dtype must match X");

    int64_t B = X.size(0);
    int64_t d = X.size(1);
    int64_t F = W_enc.size(0);
    TORCH_CHECK(W_enc.size(1) == d, "W_enc d-dim mismatch");
    TORCH_CHECK(b_enc.size(0) == F, "b_enc F-dim mismatch");
    TORCH_CHECK(k >= 1 && k <= F,   "k out of range");
    TORCH_CHECK(d <= D_MAX,         "d exceeds D_MAX=", D_MAX);

    auto V = torch::empty({B, k}, X.options().dtype(torch::kFloat32));
    auto I = torch::empty({B, k}, X.options().dtype(torch::kInt32));

    switch (k) {
        case 1:   launch_exact<1>  (X, W_enc, b_enc, V, I, B, d, F, k); break;
        case 2:   launch_exact<2>  (X, W_enc, b_enc, V, I, B, d, F, k); break;
        case 4:   launch_exact<4>  (X, W_enc, b_enc, V, I, B, d, F, k); break;
        case 8:   launch_exact<8>  (X, W_enc, b_enc, V, I, B, d, F, k); break;
        case 16:  launch_exact<16> (X, W_enc, b_enc, V, I, B, d, F, k); break;
        case 32:  launch_exact<32> (X, W_enc, b_enc, V, I, B, d, F, k); break;
        case 64:  launch_exact<64> (X, W_enc, b_enc, V, I, B, d, F, k); break;
        case 128: launch_exact<128>(X, W_enc, b_enc, V, I, B, d, F, k); break;
        default:
            TORCH_CHECK(false, "streamtopk_cuda_exact: unsupported k=", k,
                        ". Supported: 1,2,4,8,16,32,64,128.");
    }

    return {V, I};
}

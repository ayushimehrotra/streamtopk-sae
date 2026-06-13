/*
 * CPU streaming top-k SAE forward pass.
 * Tiles over F to avoid materializing the full (B, F) score matrix.
 * Single-threaded; supports fp32 only (fp16/bf16 are upcast in Python wrapper).
 */

#include <torch/extension.h>
#include <cstring>
#include <algorithm>
#include "topk_buffer.hpp"

static constexpr int T_CPU = 512; // tile size along F

// Process a single row using a fixed-K top-k buffer.
template<int K>
static void process_row(
    int64_t i,
    const float* __restrict__ X,        // (B, d)
    const float* __restrict__ W_enc,    // (F, d)
    const float* __restrict__ b_enc,    // (F,)
    float* __restrict__ V,              // (B, k) output values
    int32_t* __restrict__ I,            // (B, k) output indices
    int64_t k,
    int64_t F,
    int64_t d)
{
    TopKBuffer<K> buf;
    buf.init();

    const float* xi = X + i * d;

    // Tile over F dimension
    for (int64_t f0 = 0; f0 < F; f0 += T_CPU) {
        int64_t f_end = std::min(f0 + T_CPU, F);
        int64_t tile  = f_end - f0;

        // Compute partial scores for this tile
        float scores[T_CPU];
        for (int64_t j = 0; j < tile; ++j) {
            const float* wj = W_enc + (f0 + j) * d;
            float s = b_enc[f0 + j];
            for (int64_t dd = 0; dd < d; ++dd) {
                s += xi[dd] * wj[dd];
            }
            scores[j] = s;
        }

        // Merge tile scores into top-k buffer
        float thr = buf.threshold();
        for (int64_t j = 0; j < tile; ++j) {
            if (scores[j] > thr) {
                buf.try_insert(scores[j], (int)(f0 + j));
                thr = buf.threshold();
            }
        }
    }

    // Write outputs (not sorted by value, as per spec)
    float*   vi = V + i * k;
    int32_t* ii = I + i * k;
    if constexpr (K > 0) {
        for (int64_t ki = 0; ki < k; ++ki) {
            vi[ki] = buf.values[ki];
            ii[ki] = (int32_t)buf.indices[ki];
        }
    }
}

// Dispatcher: template over common k values
template<int K>
static void dispatch_rows(
    const float* X, const float* W_enc, const float* b_enc,
    float* V, int32_t* I,
    int64_t B, int64_t F, int64_t d, int64_t k)
{
    for (int64_t i = 0; i < B; ++i) {
        process_row<K>(i, X, W_enc, b_enc, V, I, k, F, d);
    }
}

std::tuple<torch::Tensor, torch::Tensor> streamtopk_cpu_forward(
    torch::Tensor X,
    torch::Tensor W_enc,
    torch::Tensor b_enc,
    int64_t k)
{
    TORCH_CHECK(X.is_contiguous(),     "X must be contiguous");
    TORCH_CHECK(W_enc.is_contiguous(), "W_enc must be contiguous");
    TORCH_CHECK(b_enc.is_contiguous(), "b_enc must be contiguous");
    TORCH_CHECK(X.dtype() == torch::kFloat32,     "X must be fp32");
    TORCH_CHECK(W_enc.dtype() == torch::kFloat32, "W_enc must be fp32");
    TORCH_CHECK(b_enc.dtype() == torch::kFloat32, "b_enc must be fp32");

    int64_t B = X.size(0);
    int64_t d = X.size(1);
    int64_t F = W_enc.size(0);
    TORCH_CHECK(W_enc.size(1) == d,   "W_enc d-dim mismatch");
    TORCH_CHECK(b_enc.size(0) == F,   "b_enc F-dim mismatch");
    TORCH_CHECK(k >= 1 && k <= F,     "k out of range");

    auto V = torch::empty({B, k}, torch::dtype(torch::kFloat32));
    auto I = torch::empty({B, k}, torch::dtype(torch::kInt32));

    const float* Xp     = X.data_ptr<float>();
    const float* Wp     = W_enc.data_ptr<float>();
    const float* bp     = b_enc.data_ptr<float>();
    float*       Vp     = V.data_ptr<float>();
    int32_t*     Ip     = I.data_ptr<int32_t>();

    switch (k) {
        case 1:   dispatch_rows<1>  (Xp, Wp, bp, Vp, Ip, B, F, d, k); break;
        case 2:   dispatch_rows<2>  (Xp, Wp, bp, Vp, Ip, B, F, d, k); break;
        case 4:   dispatch_rows<4>  (Xp, Wp, bp, Vp, Ip, B, F, d, k); break;
        case 8:   dispatch_rows<8>  (Xp, Wp, bp, Vp, Ip, B, F, d, k); break;
        case 16:  dispatch_rows<16> (Xp, Wp, bp, Vp, Ip, B, F, d, k); break;
        case 32:  dispatch_rows<32> (Xp, Wp, bp, Vp, Ip, B, F, d, k); break;
        case 64:  dispatch_rows<64> (Xp, Wp, bp, Vp, Ip, B, F, d, k); break;
        case 128: dispatch_rows<128>(Xp, Wp, bp, Vp, Ip, B, F, d, k); break;
        default:
            TORCH_CHECK(false, "streamtopk_cpu: unsupported k=", k,
                        ". Supported: 1,2,4,8,16,32,64,128.");
    }

    return {V, I};
}

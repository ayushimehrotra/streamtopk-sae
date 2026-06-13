#include <torch/extension.h>

// CPU forward declaration
std::tuple<torch::Tensor, torch::Tensor> streamtopk_cpu_forward(
    torch::Tensor X,
    torch::Tensor W_enc,
    torch::Tensor b_enc,
    int64_t k);

#ifdef WITH_CUDA
// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor> streamtopk_cuda_exact_forward(
    torch::Tensor X,
    torch::Tensor W_enc,
    torch::Tensor b_enc,
    int64_t k);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
streamtopk_cuda_approx_forward(
    torch::Tensor X,
    torch::Tensor W_enc,
    torch::Tensor b_enc,
    int64_t k,
    int64_t c,
    torch::optional<torch::Tensor> candidate_buffer);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("streamtopk_cpu_forward", &streamtopk_cpu_forward,
          "StreamTopK CPU forward pass (fp32 only, OpenMP parallel over batch)");

#ifdef WITH_CUDA
    m.def("streamtopk_cuda_exact_forward", &streamtopk_cuda_exact_forward,
          "StreamTopK CUDA exact forward pass (fp16/bf16/fp32)");
    m.def("streamtopk_cuda_approx_forward", &streamtopk_cuda_approx_forward,
          "StreamTopK CUDA approximate forward pass (block-candidate selection)");
#else
    m.def("streamtopk_cuda_exact_forward",
          [](torch::Tensor, torch::Tensor, torch::Tensor, int64_t) {
              TORCH_CHECK(false, "streamtopk_sae was built without CUDA support.");
              return std::make_tuple(torch::Tensor(), torch::Tensor());
          });
    m.def("streamtopk_cuda_approx_forward",
          [](torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t,
             torch::optional<torch::Tensor>) {
              TORCH_CHECK(false, "streamtopk_sae was built without CUDA support.");
              return std::make_tuple(torch::Tensor(), torch::Tensor(),
                                     torch::Tensor(), torch::Tensor());
          });
#endif
}

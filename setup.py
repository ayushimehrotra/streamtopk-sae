import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

CUDA_AVAILABLE = torch.cuda.is_available()

base_dir = os.path.dirname(os.path.abspath(__file__))
csrc = os.path.join(base_dir, "streamtopk_sae", "csrc")

extra_compile_args_cpu = {
    "cxx": ["-O3", "-march=native", "-ffast-math"],
}

if CUDA_AVAILABLE:
    sources = [
        os.path.join(csrc, "bindings.cpp"),
        os.path.join(csrc, "cpu", "streamtopk_cpu.cpp"),
        os.path.join(csrc, "cuda", "streamtopk_exact.cu"),
        os.path.join(csrc, "cuda", "streamtopk_approx.cu"),
    ]
    ext = CUDAExtension(
        name="streamtopk_sae_native",
        sources=sources,
        include_dirs=[csrc],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": ["-O3", "-march=native", "-ffast-math"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_89,code=sm_89",
            ],
        },
    )
else:
    sources = [
        os.path.join(csrc, "bindings.cpp"),
        os.path.join(csrc, "cpu", "streamtopk_cpu.cpp"),
    ]
    ext = CppExtension(
        name="streamtopk_sae_native",
        sources=sources,
        include_dirs=[csrc],
        extra_compile_args=extra_compile_args_cpu,
    )

setup(
    name="streamtopk-sae",
    version="0.1.0",
    packages=["streamtopk_sae"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)

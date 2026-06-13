# StreamTopK-SAE: Claude Code Implementation Spec

This spec implements a CS 179 final project. The deliverables are fixed. Do not add scope. If you finish early, harden tests and polish what exists.

## Deliverables

1. CPU C++ forward pass for exact row-wise TopK SAE encoding.
2. CUDA forward pass for exact row-wise TopK SAE encoding.
3. CUDA forward pass for approximate block-candidate TopK SAE encoding.
4. Eager PyTorch, `torch.compile` (max-autotune), and self-written Triton baselines.
5. Correctness tests across the benchmark grid, including tie analysis.
6. Runtime, peak memory, and achieved bandwidth benchmarks across `B, d, F, k, dtype` on the target GPU.
7. Empirical Recall@k analysis for the approximate kernel as a function of `c`.

## Repo layout

```
streamtopk-sae/
├── README.md
├── setup.py
├── pyproject.toml
├── requirements.txt
├── streamtopk_sae/
│   ├── __init__.py
│   ├── baselines.py
│   ├── reference.py
│   ├── triton_kernels.py
│   ├── synthetic.py
│   ├── csrc/
│   │   ├── cuda/
│   │   │   ├── streamtopk_exact.cu
│   │   │   ├── streamtopk_approx.cu
│   │   │   └── topk_buffer.cuh
│   │   ├── cpu/
│   │   │   ├── streamtopk_cpu.cpp
│   │   │   └── topk_buffer.hpp
│   │   └── bindings.cpp
│   ├── ops.py
│   └── utils.py
├── tests/
│   ├── test_correctness.py
│   ├── test_recall.py
│   ├── test_synthetic.py
│   └── test_smoke.py
├── bench/
│   ├── grid.py
│   ├── run_runtime.py
│   ├── run_memory.py
│   ├── run_bandwidth.py
│   ├── run_recall.py
│   ├── run_max_F.py
│   └── report.py
└── results/
    └── .gitkeep
```

## Build system

PyTorch C++/CUDA extension via `setup.py`. Build command: `pip install -e .`.

Both CPU and CUDA sources compile into the same extension module `streamtopk_sae_native`. Use `CppExtension` for CPU-only builds and `CUDAExtension` when CUDA is available. Detect via `torch.cuda.is_available()` at setup time. If CUDA is not available, build CPU only; the CUDA bindings stub out with a clear error.

nvcc flags: `-O3 --use_fast_math -lineinfo`. Target compute capabilities 8.0 (A100), 8.6 (A5000, RTX 3090), 8.9 (RTX 4090).

CPU flags: `-O3 -fopenmp -march=native -ffast-math`. Pass `-fopenmp` to both compile and link steps. Use `setuptools.command.build_ext` to inject OpenMP correctly on Linux.

Python deps: `torch>=2.3`, `triton>=2.3`, `numpy`, `scipy`, `pytest`, `pandas`, `matplotlib`, `tqdm`. No other deps.

## Tensor shape conventions

Lock these on day one. Do not change them.

- `X` input activations: `(B, d)`, contiguous, row-major.
- `W_enc` encoder weights: `(F, d)`, contiguous, row-major.
- `b_enc` encoder bias: `(F,)`.
- Output values: `(B, k)`, fp32 regardless of input dtype.
- Output indices: `(B, k)`, int32, values in `[0, F)`.

Supported input dtypes: fp16, bf16, fp32. Accumulation always fp32.

Output values are NOT sorted within each row. Tests check set equality on indices, not order equality.

## Synthetic preactivation generator (`streamtopk_sae/synthetic.py`)

The point: generate `X, W_enc, b_enc` such that `Z = X @ W_enc.T + b_enc` has statistics resembling real TopK SAE preactivations. Random gaussian inputs do not have these properties and will give misleading recall numbers for the approximate kernel.

### Target statistical properties

A realistic preactivation matrix should have:

1. **Heavy-tailed positive tail.** Most entries near zero or negative, a small fraction of large positive entries per row. The top-k for typical SAE configs sits well above the bulk.
2. **Variable per-feature firing rate.** Long-tail distribution of "how often does feature j land in the top-k across rows." Some features fire on 30%+ of rows, others <0.1%.
3. **Within-row correlation between high-scoring features.** Features that co-fire share dictionary directions. Synthetic generator must induce this so top-k indices are clustered, not uniformly random.
4. **Bias structure.** A small fraction of features have positive bias and tend to fire often.

### Generator design

```python
def generate_realistic_preacts(B, F, d, n_themes=64, k_effective=64,
                                heavy_tail_alpha=2.5, seed=0,
                                dtype=torch.bfloat16):
    """
    Returns X, W_enc, b_enc with shapes (B, d), (F, d), (F,).
    Z = X @ W_enc.T + b_enc has statistics approximating real TopK SAE
    preactivations.
    """
```

Implementation sketch:

1. Generate `n_themes` "theme" vectors in d-dim, unit-normalized.
2. Assign each of F features to a small set of themes (Zipf-distributed assignment). Each feature's encoder direction is the sum of its themes plus small isotropic noise, then normalized.
3. For each input row, sample sparse theme weights (most zero, a few large, Pareto-distributed with shape `heavy_tail_alpha`). The row's `x` is the linear combination of theme vectors with those weights.
4. `b_enc`: small fraction of features with positive bias (Pareto-tailed magnitudes), rest near zero.

Verify in `test_synthetic.py`:
- Top-k preactivations per row positive in >95% of rows for default config.
- Per-feature firing-rate distribution long-tailed; top 1% of features fire on >10x the median rate.
- Top-k indices NOT uniform across F (entropy < 0.9 * log F).
- k-th-largest preactivation well-separated from (k+1)-th in >80% of rows.

Provide a CLI `python -m streamtopk_sae.synthetic --diagnostics` that prints these statistics. The agent must run this and confirm the statistics are in expected ranges before declaring the generator done.

### Use in tests and benchmarks

By default all tests and benchmarks use the synthetic generator. Add an env var `STREAMTOPK_DATA=gaussian` to fall back to plain gaussian for sanity. This makes it easy to compare and see whether kernel behavior depends on data distribution.

## Reference implementation (`reference.py`)

```python
def reference_topk_sae(X, W_enc, b_enc, k):
    preacts = X @ W_enc.T + b_enc
    values, indices = torch.topk(preacts.float(), k, dim=-1)
    return values, indices.to(torch.int32)
```

Cast preacts to fp32 before `torch.topk`. This is the ground truth.

## Baselines (`baselines.py`)

Three implementations:

1. `baseline_eager(X, W_enc, b_enc, k)`: reference on GPU with input dtype matmul, fp32 topk. Weak baseline.
2. `baseline_compiled(X, W_enc, b_enc, k)`: same wrapped in `torch.compile(mode="max-autotune", fullgraph=True)`. Cache compiled function module-level. Provide `warmup_compiled(B, d, F, k, dtype)`.
3. `baseline_triton(X, W_enc, b_enc, k)`: see next section.

All three identical signatures and output shapes.

## Triton baseline (`triton_kernels.py`)

Hand-written Triton kernel doing dense matmul + row-wise topk as two separate Triton kernels in a Python function. This is NOT fused; it isolates the fusion speedup from the Triton-vs-cuBLAS effect.

- Kernel 1: dense bf16 matmul with fp32 accumulation, output `(B, F)` fp32. Autotune over `(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)` covering at minimum `(64,128,32), (128,128,32), (128,256,32), (64,256,64)`.
- Kernel 2: row-wise top-k in Triton for `k <= 128`. For `k > 128` fall back to `torch.topk` on the fp32 score tensor and log a warning.

Document the k limit in the docstring.

## CPU C++ implementation (`csrc/cpu/streamtopk_cpu.cpp`)

Uses the SAME algorithmic structure as the CUDA exact kernel: tile over F, maintain per-row top-k buffer, never materialize the full `(B, F)` score matrix. This makes CPU-vs-CUDA comparison apples-to-apples and gives a meaningful "fused streaming on CPU" baseline.

### Parallelism

OpenMP parallel-for over the batch dimension:

```cpp
#pragma omp parallel for schedule(static)
for (int64_t i = 0; i < B; ++i) {
    process_row(i, X, W_enc, b_enc, V, I, k, F, d);
}
```

Within `process_row`, the top-k buffer lives on the stack. The tile loop over F is sequential within each thread.

### Tile structure

Process F in tiles of `T_CPU` (default 512). For each tile, compute the `T_CPU` scores via a small matmul, merge into the running top-k buffer using the same insertion-sort or min-heap logic as the GPU version.

**Default inner-loop strategy: plain C++ with `#pragma omp simd` on the innermost loop.** Trust the compiler with `-march=native -ffast-math` to auto-vectorize. This is simpler than vendoring Eigen and good enough for a class baseline.

If autovec doesn't fire and performance is unacceptable, document the regression and fall back to explicit AVX2 intrinsics for the dot product. Do NOT vendor Eigen as a first move; it's a yak-shave.

### Numerical precision

CPU implementation supports fp32 only. fp16 and bf16 on CPU are complicated (no native arithmetic on most CPUs) and not worth it for a class project. Python wrapper upcasts inputs to fp32 if needed and documents this clearly.

### Top-k buffer (`topk_buffer.hpp`)

```cpp
template<int K>
struct TopKBuffer {
    float values[K];
    int   indices[K];
    
    inline void init();
    inline void try_insert(float v, int idx);
    inline float threshold() const;
    inline void merge(const TopKBuffer<K>& other);
};
```

Insertion-sort for K <= 64, min-heap for larger. The `threshold()` early-rejection optimization is critical for performance.

### Python binding

```cpp
std::tuple<torch::Tensor, torch::Tensor> streamtopk_cpu_forward(
    torch::Tensor X,        // (B, d), fp32
    torch::Tensor W_enc,    // (F, d), fp32
    torch::Tensor b_enc,    // (F,), fp32
    int64_t k
);
```

Validate: contiguous, fp32, correct shapes, k templated.

## CUDA exact kernel (`csrc/cuda/streamtopk_exact.cu`)

Main GPU deliverable. Single `__global__` kernel.

### Launch configuration

- Grid: 1D with `B` blocks. One block per batch row.
- Block: 1D with `BLOCK_THREADS = 128` threads (parameter, sweep later).

### Algorithm

```
for batch row i (= blockIdx.x):
    load x_i (shape d) into shared memory cooperatively
    initialize per-thread local top-k buffer with (-inf, -1)
    for each latent tile t = 0..ceil(F/T)-1:
        cooperatively compute partial scores for this tile:
            scores[j] = dot(x_i, W_enc[t*T + j, :]) + b_enc[t*T + j]
        each thread merges its assigned scores into its local top-k buffer
    parallel reduce per-thread buffers into a block-wide top-k of size k
    write V[i, :], I[i, :]
```

### Critical design parameters

- `T` (tile size along F): default 1024, multiple of `BLOCK_THREADS`. Each thread handles `T / BLOCK_THREADS` latents per tile.
- `BLOCK_THREADS`: default 128.
- `k`: templated for 16, 32, 64, 128. For other `k`, dispatch to the closest larger template, ignore extra slots.

### Shared memory and the d-tiling issue

For `d = 4096`, `T = 1024`, bf16, a W_enc tile is 8 MB. Does NOT fit in shared memory (~100 KB max on A5000). You MUST also tile the d dimension.

Process d in chunks of `D_TILE` (default 64). Inner loop accumulates partial dot products in registers across d-chunks before adding the bias and attempting top-k insertion:

```cuda
for (int t = 0; t < num_tiles_F; t++) {
    // Initialize partial sums in registers (one per latent this thread owns)
    for (int dt = 0; dt < num_tiles_D; dt++) {
        // Cooperatively load x_smem[dt*D_TILE : (dt+1)*D_TILE]
        // Cooperatively load w_smem[T][D_TILE]
        __syncthreads();
        // Each thread updates partial sums for its latents
        __syncthreads();
    }
    // Add bias, attempt top-k insertion
}
```

Sizes for `d=4096, T=1024, D_TILE=64`: x_smem 256B, w_smem 128 KB bf16. Still exceeds A5000's 100 KB. Either reduce T to 512 (w_smem 64 KB, fits) or reduce D_TILE. Document chosen defaults and explain constraints. Verify shared memory usage at compile time using `cudaFuncSetAttribute` or compute in the launcher.

### Numerical precision

- Input dtypes: fp16, bf16, fp32.
- Accumulation: fp32 always.
- Top-k comparisons: fp32.
- Output values: fp32.

Use `__bfloat162float`, `__half2float` for casts.

### Per-thread top-k buffer (`csrc/cuda/topk_buffer.cuh`)

Header-only template, mirrors CPU version:

```cpp
template<int K>
struct TopKBuffer {
    float values[K];
    int   indices[K];
    
    __device__ __forceinline__ void init();
    __device__ __forceinline__ void try_insert(float v, int idx);
    __device__ __forceinline__ float threshold() const;
    __device__ __forceinline__ void merge(const TopKBuffer<K>& other);
};
```

Insertion-sort (default, K <= 64) and min-heap (K > 64) variants selectable via template parameter. The `threshold()` early-rejection is critical because on real data most scores fall below it.

### Block-wide reduction

After all threads have local top-k buffers:

1. Write all buffers to shared memory: `(BLOCK_THREADS, K)`.
2. `log2(BLOCK_THREADS)` rounds of pairwise merging via two-pointer merge of sorted sequences, O(K) per merge.
3. Thread 0 writes final top-k to global memory.

### Python binding

```cpp
std::tuple<torch::Tensor, torch::Tensor> streamtopk_cuda_exact_forward(
    torch::Tensor X,
    torch::Tensor W_enc,
    torch::Tensor b_enc,
    int64_t k
);
```

Validate: contiguous, on CUDA, correct dtypes/shapes, k templated, d <= D_MAX (default 4096).

## CUDA approximate kernel (`csrc/cuda/streamtopk_approx.cu`)

Same overall structure as exact, but instead of merging each tile's scores into a running top-k, keep the top-c candidates from each tile in a candidate buffer, then do a final global top-k over the union.

### Algorithm

```
for batch row i:
    load x_i
    for each latent tile t:
        compute scores for this tile
        find top-c scores within this tile (block-wide)
        write c (value, index) pairs to candidate_buffer[i, t*c : (t+1)*c]
    after all tiles, do final top-k over row's full candidate buffer
    write V[i, :], I[i, :]
```

### Candidate buffer

Per-row buffer in **global memory**, shape `(B, num_tiles, c)`. Allocated once in the Python wrapper, not per-call.

For B=1024, F=2^20, T=1024, c=64: 512 MB. Acceptable on A5000. Wrapper accepts optional preallocated buffer; otherwise allocates.

### In-tile top-c

Block-wide top-k with K=c. Reuse `TopKBuffer<C>`. `c` restricted to templated values (16, 32, 64, 128).

### Final global top-k

For v1, call `torch.topk` from Python on the candidate buffer. The point of the approximate kernel is to reduce the initial score matrix, not to fuse the final selection. Optional v2 can fuse it later if time permits.

### Recall metric

`Recall@k = |hat_S ∩ S*| / k` averaged over batch dimension. Set intersection on indices. Tied boundary value that lands in `S*` but not `hat_S` counts as a miss, no partial credit.

### Python binding

```cpp
std::tuple<torch::Tensor, torch::Tensor> streamtopk_cuda_approx_forward(
    torch::Tensor X,
    torch::Tensor W_enc,
    torch::Tensor b_enc,
    int64_t k,
    int64_t c,
    torch::optional<torch::Tensor> candidate_buffer = c10::nullopt
);
```

## Tests (`tests/`)

Pytest. Parametrize over dtype, shapes, k.

### Correctness (`tests/test_correctness.py`)

For each combination of:
- dtype: fp16, bf16, fp32 (CPU runs fp32 only)
- shapes: small `(B=4, d=64, F=256, k=8)`, medium `(B=32, d=512, F=4096, k=32)`, large `(B=128, d=2048, F=131072, k=64)` (skip large where memory doesn't fit)
- k: 16, 32, 64
- impl: cpu, cuda_exact, baseline_eager, baseline_compiled, baseline_triton

Steps:
1. Generate inputs via synthetic generator with seed.
2. Compute reference.
3. Compute kernel output.
4. Assert sorted values match reference within dtype-appropriate tolerance.
5. Assert index set match modulo boundary ties.

### Tie analysis

Construct inputs with deliberate ties at the top-k boundary (k-th and (k+1)-th values exactly equal). For each row classify as:
- All indices match (no boundary tie).
- Indices differ but boundary value tied (acceptable).
- Indices differ and boundary value not tied (FAIL).

Fail count must be zero.

### Edge cases

- `k = F`, `k = 1`.
- `F` not a multiple of tile size T.
- `d` at maximum supported value.
- All-zero `X`.
- `b_enc` with single extremely large entry (always in top-k).
- CPU: `B` large enough to exercise OpenMP scheduling (1024+ rows).

### Recall (`tests/test_recall.py`)

For each `c` in `[k, 2k, 4k, T/4, T/2, T]` and a small grid:
- Compute exact top-k.
- Compute approximate top-k.
- Compute Recall@k per row.
- Assert mean recall monotonically non-decreasing as `c` increases (small noise tolerance).
- Assert `c = T` gives recall = 1.0 exactly.

### Synthetic (`tests/test_synthetic.py`)

Verify generator's statistical properties:
- Top-k positive in >95% of rows.
- Per-feature firing rate long-tailed.
- Top-k indices not uniform (entropy < 0.9 * log F).
- k-th-largest well-separated from (k+1)-th in >80% of rows.

## Benchmark grid (`bench/grid.py`)

```python
BENCH_GRID = {
    "B": [32, 128, 512, 2048],
    "d": [768, 2048, 4096],
    "F": [2**14, 2**16, 2**18, 2**20, 2**22],
    "k": [16, 32, 64, 128],
    "dtype": ["fp16", "bf16", "fp32"],
}
```

720 cells. Many OOM on dense baseline at large F; scripts must catch OOM gracefully.

```python
BENCH_GRID_CPU = {
    "B": [32, 128, 512],
    "d": [768, 2048],
    "F": [2**14, 2**16, 2**18],
    "k": [16, 32, 64],
    "dtype": ["fp32"],
}
```

## Benchmark scripts

All scripts support `--dry-run` mode running one cell as smoke test.

### Runtime (`bench/run_runtime.py`)

```
python -m bench.run_runtime --impl cpu,cuda_exact,cuda_approx,eager,compiled,triton \
    --warmup 10 --iters 100 --out results/runtime.csv
```

Per cell:
1. Allocate inputs via synthetic generator.
2. Warmup `warmup` iterations.
3. Time `iters` iterations with `torch.cuda.Event` (or `std::chrono` for CPU).
4. Record median, p10, p90, min, max in microseconds.
5. Free inputs, clear cache.

Log device info per row. Shuffle cell order across iterations.

### Memory (`bench/run_memory.py`)

GPU impls:
1. `torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()`.
2. Allocate inputs (baseline_mem).
3. Call impl once.
4. Record `torch.cuda.max_memory_allocated()` as peak_mem.
5. Working memory = `peak_mem - baseline_mem - output_size`.

CPU impl: `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` before/after, fresh subprocess per measurement for clean isolation.

### Bandwidth (`bench/run_bandwidth.py`)

For each impl on representative cells:
1. Compute bytes moved (lower bound): `bytes_X + bytes_W + bytes_bias + bytes_output`.
2. Time the kernel.
3. Achieved bandwidth = `bytes_moved / runtime`.
4. Arithmetic intensity = `flops / bytes_moved`.
5. Roofline placement vs peak HBM bandwidth and peak FLOPs.

### Recall (`bench/run_recall.py`)

```python
RECALL_GRID = {
    "B": [128, 512],
    "d": [2048, 4096],
    "F": [2**16, 2**18, 2**20],
    "k": [32, 64],
    "c_multiplier": [1, 2, 4, 8, 16],
    "dtype": ["bf16"],
}
```

Per cell:
1. Compute exact top-k.
2. Compute approximate top-k.
3. Compute Recall@k per row.
4. Record mean, std, min, p10, p50, p90.

### Max feasible F (`bench/run_max_F.py`)

For each (impl, B, d, k, dtype):
1. Binary search F in powers of 2 from 2^14 to 2^24.
2. Try one forward at each F.
3. OOM → decrease; success → increase.
4. Record max F that fits.

### Report (`bench/report.py`)

Reads all CSVs, produces:
1. Runtime table (best impl per cell, speedup vs eager).
2. Memory table (peak working memory per impl, ratio vs dense).
3. Roofline plot.
4. Recall plot (mean recall vs c/k, one curve per (F, k)).
5. Pareto plot (runtime vs recall for approximate kernel, colored by c).
6. Max-F table per impl.

Plots as PDF, tables as Markdown and CSV.

## Milestone-based progression

Work through milestones in order. Do NOT advance to a milestone until the prior one's gate is met. Run each gate command explicitly and verify output before moving on. If a gate fails, fix it before adding more code.

### M0: Repo scaffold
- All directory structure exists.
- `setup.py`, `pyproject.toml`, `requirements.txt` populated.
- `pip install -e .` succeeds and produces an importable `streamtopk_sae` module.
- **Gate**: `python -c "import streamtopk_sae"` exits 0.

### M1: Reference and synthetic generator
- `reference.py` with the dense matmul + topk path.
- `synthetic.py` with `generate_realistic_preacts` and a `--diagnostics` CLI.
- `test_synthetic.py` passes all statistical checks.
- **Gate**: `pytest tests/test_synthetic.py -x` exits 0 AND `python -m streamtopk_sae.synthetic --diagnostics` shows expected statistics (top-k positivity, firing-rate long-tail, index non-uniformity, boundary separation).

### M2: Eager and compiled baselines
- `baselines.py` with eager and compiled wrappers.
- `test_correctness.py` runs for eager and compiled against reference on small grid.
- **Gate**: `pytest tests/test_correctness.py -x -k "small and (eager or compiled)"` exits 0.

### M3: Triton baseline
- `triton_kernels.py` with matmul + topk.
- Correctness against reference on small and medium shapes.
- **Gate**: `pytest tests/test_correctness.py -x -k "triton"` exits 0.

### M4: CPU implementation
- C++ source with OpenMP parallel-for over batch.
- pybind11 binding wired into `setup.py`.
- Correctness against reference on small and medium shapes (fp32 only).
- Confirm OpenMP is actually using multiple threads: run with `OMP_NUM_THREADS=1` and `OMP_NUM_THREADS=8`, observe different runtimes on a 1024+ row workload.
- **Gate**: `pytest tests/test_correctness.py -x -k "cpu"` exits 0 AND the OpenMP scaling check shows the 8-thread run is faster than the 1-thread run.

### M5: CUDA exact kernel, correctness only
- `streamtopk_exact.cu` with the tiled streaming algorithm.
- D-tiling implemented so shared memory fits within device limits.
- All dtypes (fp16, bf16, fp32) supported.
- Correctness against reference on small, medium, and large shapes.
- Tie analysis test passes.
- **Gate**: `pytest tests/test_correctness.py -x -k "cuda_exact"` exits 0 including the tie test.

### M6: CUDA approximate kernel
- `streamtopk_approx.cu` with block-candidate selection.
- Recall test passes (monotonicity, `c = T` gives recall 1.0).
- **Gate**: `pytest tests/test_recall.py -x` exits 0.

### M7: All benchmark scripts smoke
- All `bench/run_*.py` scripts have `--dry-run`.
- Each dry-run completes without error.
- **Gate**: For each script in `bench/run_*.py`, `python -m bench.<script> --dry-run` exits 0.

### M8: Full benchmark runs
- Run runtime, memory, bandwidth, recall, max-F benchmarks on the target GPU.
- CSVs in `results/` with expected schema.
- **Gate**: All five output CSVs exist and have data rows. Spot-check: kernel beats eager on at least one cell, and max-F is larger for fused kernel than dense baseline.

### M9: Report
- `bench/report.py` produces all tables and plots.
- README documents reproduction from a fresh clone.
- **Gate**: `python -m bench.report` produces output files without error. README reproduction instructions tested.

### M10: Optimization pass (only if M9 complete)
- Tune `T`, `BLOCK_THREADS`, `D_TILE`, top-k buffer variant for exact kernel based on M8 results.
- Re-run benchmarks. Document changes.
- **Gate**: Updated CSVs show measurable improvement on at least medium-and-large cells without regressing correctness.

## Definition of done

The class project is complete when:

1. `pip install -e .` succeeds on a Linux box with CUDA 12 and a recent PyTorch.
2. `pytest tests/` passes.
3. All `bench/run_*.py` scripts have working `--dry-run` modes.
4. The full benchmark suite has been run on the target GPU and CSVs are in `results/`.
5. `python -m bench.report` produces tables and plots.
6. README documents reproduction from a fresh clone.

Nothing beyond this is part of the class deliverable. Do not add backward passes, real SAE library integration (sae_lens, sparsify), probabilistic recall bounds, or comparisons to external GPU top-k libraries (RadiK, Dr.Top-k).

## Constraints on the agent

- After each milestone, run the gate command and report the output. Do not advance until the gate passes.
- Do not change tensor shape conventions or function signatures partway through.
- Every numerical claim in the final report must be backed by a CSV the script can regenerate.
- All randomness uses seeded PRNG. Default seed: 0. Tests must be deterministic.
- When in doubt about a design choice, choose the simpler option, add a TODO comment, document the choice in the file's docstring.
- Do not silently degrade behavior. If a code path can't handle some input, raise a clear error at the Python boundary.
- Use `pytest -x` during development to fail fast.
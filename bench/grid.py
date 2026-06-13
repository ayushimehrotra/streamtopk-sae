"""
Benchmark grid definitions.
"""

BENCH_GRID = {
    "B":     [32, 128, 512, 2048],
    "d":     [768, 2048, 4096],
    "F":     [2**14, 2**16, 2**18, 2**20, 2**22],
    "k":     [16, 32, 64, 128],
    "dtype": ["fp16", "bf16", "fp32"],
}

BENCH_GRID_CPU = {
    "B":     [32, 128, 512],
    "d":     [768, 2048],
    "F":     [2**14, 2**16, 2**18],
    "k":     [16, 32, 64],
    "dtype": ["fp32"],
}

# Smaller grid for recall benchmarks
RECALL_GRID = {
    "B":            [128, 512],
    "d":            [2048, 4096],
    "F":            [2**16, 2**18, 2**20],
    "k":            [32, 64],
    "c_multiplier": [1, 2, 4, 8, 16],
    "dtype":        ["bf16"],
}


def iter_grid(grid):
    """Yield all combinations from a grid dict as list of dicts."""
    import itertools
    keys   = list(grid.keys())
    values = list(grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def dry_run_cell(grid):
    """Return the first cell of a grid (for --dry-run mode)."""
    return next(iter_grid(grid))

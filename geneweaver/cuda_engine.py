"""Numba CUDA mismatch-counting kernel for GeneWeaver."""

import numpy as np
from numba import cuda


@cuda.jit
def mismatch_kernel(guides, sites, results):
    idx = cuda.grid(1)

    if idx < sites.shape[0]:
        mismatches = 0
        for position in range(guides.shape[1]):
            if guides[idx, position] != sites[idx, position]:
                mismatches += 1
        results[idx] = mismatches


def count_mismatches(guides, sites):
    guides = np.asarray(guides, dtype=np.uint8)
    sites = np.asarray(sites, dtype=np.uint8)

    if guides.ndim != 2 or sites.ndim != 2:
        raise ValueError("Guides and sites must be 2D arrays.")
    if guides.shape != sites.shape:
        raise ValueError("Guides and sites must have the same shape.")
    if guides.shape[1] == 0:
        raise ValueError("DNA sequences cannot be empty.")
    if guides.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    results = np.zeros(guides.shape[0], dtype=np.int32)
    d_guides = cuda.to_device(guides)
    d_sites = cuda.to_device(sites)
    d_results = cuda.to_device(results)

    threads_per_block = 128
    blocks_per_grid = (guides.shape[0] + threads_per_block - 1) // threads_per_block

    mismatch_kernel[blocks_per_grid, threads_per_block](
        d_guides, d_sites, d_results
    )

    return d_results.copy_to_host()

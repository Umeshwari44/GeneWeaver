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
    """Count mismatches between corresponding encoded DNA sequences."""

    guides = np.asarray(guides)
    sites = np.asarray(sites)

    if guides.ndim != 2 or sites.ndim != 2:
        raise ValueError("Guides and sites must be 2D arrays.")

    if guides.shape != sites.shape:
        raise ValueError("Guides and sites must have the same shape.")

    if guides.shape[1] == 0:
        raise ValueError("DNA sequences cannot be empty.")

    if not np.issubdtype(guides.dtype, np.integer):
        raise ValueError("DNA encodings must contain integers.")

    if not np.issubdtype(sites.dtype, np.integer):
        raise ValueError("DNA encodings must contain integers.")

    if np.any((guides < 0) | (guides > 4)):
        raise ValueError("Guide DNA encodings must be between 0 and 4.")

    if np.any((sites < 0) | (sites > 4)):
        raise ValueError("Site DNA encodings must be between 0 and 4.")

    guides = guides.astype(np.uint8)
    sites = sites.astype(np.uint8)

    if guides.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    results = np.zeros(guides.shape[0], dtype=np.int32)

    d_guides = cuda.to_device(guides)
    d_sites = cuda.to_device(sites)
    d_results = cuda.to_device(results)

    threads_per_block = 128
    blocks_per_grid = (
        guides.shape[0] + threads_per_block - 1
    ) // threads_per_block

    mismatch_kernel[blocks_per_grid, threads_per_block](
        d_guides, d_sites, d_results
    )

    return d_results.copy_to_host()


@cuda.jit
def mismatch_matrix_kernel(guides, sites, results):
    """Compare every guide against every candidate DNA site."""

    idx = cuda.grid(1)
    total = guides.shape[0] * sites.shape[0]

    if idx < total:
        guide_idx = idx // sites.shape[0]
        site_idx = idx % sites.shape[0]

        mismatches = 0

        for position in range(guides.shape[1]):
            if guides[guide_idx, position] != sites[site_idx, position]:
                mismatches += 1

        results[idx] = mismatches


def count_mismatches_matrix(guides, sites):
    """Count mismatches between every guide and every candidate site."""

    guides = np.asarray(guides)
    sites = np.asarray(sites)

    if guides.ndim != 2 or sites.ndim != 2:
        raise ValueError("Guides and sites must be 2D arrays.")

    if guides.shape[1] != sites.shape[1]:
        raise ValueError("Guides and sites must have the same sequence length.")

    if guides.shape[1] == 0:
        raise ValueError("DNA sequences cannot be empty.")

    if not np.issubdtype(guides.dtype, np.integer):
        raise ValueError("DNA encodings must contain integers.")

    if not np.issubdtype(sites.dtype, np.integer):
        raise ValueError("DNA encodings must contain integers.")

    if np.any((guides < 0) | (guides > 4)):
        raise ValueError("Guide DNA encodings must be between 0 and 4.")

    if np.any((sites < 0) | (sites > 4)):
        raise ValueError("Site DNA encodings must be between 0 and 4.")

    guides = guides.astype(np.uint8)
    sites = sites.astype(np.uint8)

    if guides.shape[0] == 0 or sites.shape[0] == 0:
        return np.zeros((guides.shape[0], sites.shape[0]), dtype=np.int32)

    results = np.zeros(guides.shape[0] * sites.shape[0], dtype=np.int32)

    d_guides = cuda.to_device(guides)
    d_sites = cuda.to_device(sites)
    d_results = cuda.to_device(results)

    threads_per_block = 128
    blocks_per_grid = (results.size + threads_per_block - 1) // threads_per_block

    mismatch_matrix_kernel[blocks_per_grid, threads_per_block](
        d_guides, d_sites, d_results
    )

    return d_results.copy_to_host().reshape(
        guides.shape[0], sites.shape[0]
    )

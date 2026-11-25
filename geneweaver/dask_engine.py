"""Dask-based distributed genome processing for GeneWeaver."""

import dask
import numpy as np
from dask import delayed
from dask.distributed import Client

from geneweaver.biopython_pipeline import read_genome_fasta, chunk_sequence
from geneweaver.encoding import encode


def _encode_chunk(sequence):
    """Encode one DNA chunk as a NumPy array."""
    return encode(sequence)


def process_genome_distributed(
    file_path,
    chunk_size=10000,
    n_workers=2
):
    """Read a FASTA genome and encode its chunks using Dask workers."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if n_workers <= 0:
        raise ValueError("Number of workers must be greater than zero.")

    genome = read_genome_fasta(file_path)

    tasks = []
    chunk_locations = []

    for name, sequence in genome.items():
        chunks = chunk_sequence(sequence, chunk_size)

        for chunk_index, chunk in enumerate(chunks):
            tasks.append(delayed(_encode_chunk)(chunk))
            chunk_locations.append((name, chunk_index))

    processed = {
        name: [None] * len(chunk_sequence(sequence, chunk_size))
        for name, sequence in genome.items()
    }

    with Client(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None
    ) as client:
        results = dask.compute(*tasks, scheduler=client)

    for location, result in zip(chunk_locations, results):
        name, chunk_index = location
        processed[name][chunk_index] = np.asarray(result)

    return processed

"""Compare CPU and Dask genome encoding performance."""

import os
import random
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from geneweaver.biopython_pipeline import process_genome_as_arrays
from geneweaver.dask_engine import process_genome_distributed

GENOME_LENGTH = 250_000
CHUNK_SIZE = 10_000
N_WORKERS = 2
REPEATS = 3


def create_test_fasta(file_path):
    """Generate a reproducible synthetic FASTA genome."""
    rng = random.Random(42)

    with open(file_path, "w") as fasta:
        for contig_number in range(2):
            sequence = "".join(
                rng.choices("ACGT", k=GENOME_LENGTH)
            )
            fasta.write(f">chr{contig_number + 1}\n")

            for start in range(0, len(sequence), 80):
                fasta.write(sequence[start:start + 80] + "\n")


def measure(function):
    """Return the result and execution time."""
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    return result, elapsed


def verify_results(cpu_result, dask_result):
    """Verify that both pipelines produce identical encoded chunks."""
    assert cpu_result.keys() == dask_result.keys()

    for contig in cpu_result:
        assert len(cpu_result[contig]) == len(dask_result[contig])

        for cpu_chunk, dask_chunk in zip(
            cpu_result[contig], dask_result[contig]
        ):
            assert np.array_equal(cpu_chunk, dask_chunk)

    return True


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        fasta_file = os.path.join(temp_dir, "benchmark.fa")
        create_test_fasta(fasta_file)

        cpu_times = []
        dask_times = []

        for run in range(REPEATS):
            cpu_result, cpu_elapsed = measure(
                lambda: process_genome_as_arrays(
                    fasta_file, CHUNK_SIZE
                )
            )

            dask_result, dask_elapsed = measure(
                lambda: process_genome_distributed(
                    fasta_file, CHUNK_SIZE, N_WORKERS
                )
            )

            verify_results(cpu_result, dask_result)

            cpu_times.append(cpu_elapsed)
            dask_times.append(dask_elapsed)

            print(
                f"Run {run + 1}: "
                f"CPU={cpu_elapsed:.4f}s, "
                f"Dask={dask_elapsed:.4f}s"
            )

    cpu_median = statistics.median(cpu_times)
    dask_median = statistics.median(dask_times)
    speedup = cpu_median / dask_median if dask_median else 0

    output = [
        "# GeneWeaver: CPU vs Dask Benchmark",
        "",
        "## Configuration",
        f"- Contigs: 2",
        f"- Total genome size: {GENOME_LENGTH * 2:,} bp",
        f"- Chunk size: {CHUNK_SIZE:,} bp",
        f"- Dask workers: {N_WORKERS}",
        f"- Repeated runs: {REPEATS}",
        "- Execution environment: Local machine",
        "",
        "## Results",
        "",
        "| Measurement | CPU | Dask |",
        "|---|---:|---:|",
    ]

    for index, (cpu_time, dask_time) in enumerate(
        zip(cpu_times, dask_times), start=1
    ):
        output.append(
            f"| Run {index} | {cpu_time:.4f}s | {dask_time:.4f}s |"
        )

    output.extend([
        f"| Median | {cpu_median:.4f}s | {dask_median:.4f}s |",
        "",
        f"- CPU/Dask time ratio: {speedup:.2f}x",
        "- Output verification: Passed for every run",
        "- Note: Dask uses local workers; this is not a multi-machine benchmark.",
        "- Dask startup and scheduling overhead are included in its timings.",
        "",
    ])

    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(
        results_dir, "dask_benchmark.md"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(output))

    print("\n".join(output))
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

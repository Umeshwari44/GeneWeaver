from pathlib import Path
import sys
import tempfile
import time
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geneweaver.biopython_pipeline import (
    read_genome_fasta,
    process_genome_as_arrays,
)
from geneweaver.encoding import encode


def previous_process(file_path, chunk_size=10000):
    genome = read_genome_fasta(file_path)
    processed = {}

    for name, sequence in genome.items():
        encoded = encode(sequence)
        processed[name] = [
            encoded[i:i + chunk_size]
            for i in range(0, len(encoded), chunk_size)
        ]

    return processed


def benchmark(function, file_path, runs=5):
    timings = []

    for _ in range(runs):
        start = time.perf_counter()
        result = function(file_path)
        timings.append(time.perf_counter() - start)

    return result, statistics.median(timings)


with tempfile.TemporaryDirectory() as temp_dir:
    fasta_file = Path(temp_dir) / "benchmark.fa"
    sequence = "ACGT" * 125000

    fasta_file.write_text(
        f">chr1\n{sequence}\n>chr2\n{sequence}\n"
    )

    old_result, old_time = benchmark(previous_process, fasta_file)
    new_result, new_time = benchmark(process_genome_as_arrays, fasta_file)

    for name in old_result:
        for old_chunk, new_chunk in zip(old_result[name], new_result[name]):
            assert (old_chunk == new_chunk).all()

    print("Correctness check: PASSED")
    print(f"Previous implementation median: {old_time:.6f} seconds")
    print(f"Optimized implementation median: {new_time:.6f} seconds")
    print(f"Time difference: {new_time - old_time:.6f} seconds")

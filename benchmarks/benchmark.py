"""Benchmark: pure-Python brute force vs GeneWeaver engine (CPU, and GPU if present).

Usage:  python benchmarks/benchmark.py            -> writes results/benchmark.md
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geneweaver.backend import get_backend  # noqa: E402
from geneweaver.engine import OffTargetSearcher  # noqa: E402
from geneweaver.naive import naive_search  # noqa: E402
from geneweaver.synthetic import generate_dataset  # noqa: E402

N_GUIDES = 20
MAX_MM = 4


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def run_engine(contigs, guides, backend):
    s = OffTargetSearcher(guides, max_mismatches=MAX_MM, backend=backend)
    hits = []
    for n, q in contigs.items():
        hits += s.search_sequence(n, q)
    return hits


def main():
    cpu = get_backend(prefer_gpu=False)
    gpu = get_backend(prefer_gpu=True)
    has_gpu = gpu.is_gpu
    rows = []
    # (genome size, run the naive baseline too?)
    for size, do_naive in [(100_000, True), (1_000_000, False), (10_000_000, False), (50_000_000, False)]:
        contigs, guides, _ = generate_dataset(genome_len=size, n_contigs=2, n_guides=N_GUIDES, seed=3)
        row = {"size": size}
        if do_naive:
            ref, row["naive"] = timed(lambda: naive_search(contigs, guides, "NGG", MAX_MM))
        hits, row["cpu"] = timed(lambda: run_engine(contigs, guides, cpu))
        if do_naive:
            got = {(h.guide_id, h.contig, h.start, h.strand, h.mismatches) for h in hits}
            assert got == ref, "engine disagrees with brute force!"
        if has_gpu:
            run_engine(contigs, guides, gpu)  # warm-up (CUDA context, kernel compile)
            _, row["gpu"] = timed(lambda: run_engine(contigs, guides, gpu))
        row["hits"] = len(hits)
        rows.append(row)
        print(row, flush=True)

    lines = [f"# GeneWeaver benchmark\n",
             f"- guides: {N_GUIDES}, PAM: NGG, max mismatches: {MAX_MM}, both strands",
             f"- CPU backend: {cpu.name}",
             f"- GPU backend: {gpu.name if has_gpu else 'not available on this machine (no CuPy / CUDA device)'}\n",
             "| Genome size (bp) | Pure-Python brute force (s) | GeneWeaver CPU (s) | GeneWeaver GPU (s) | Sites found |",
             "|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['size']:,} | {r['naive']:.2f} | " if "naive" in r else f"| {r['size']:,} | not run | ")
        lines[-1] += f"{r['cpu']:.2f} | " + (f"{r['gpu']:.2f}" if "gpu" in r else "n/a") + f" | {r['hits']} |"
    lines.append("\nBrute-force and engine outputs were verified identical on the 100,000 bp run.")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

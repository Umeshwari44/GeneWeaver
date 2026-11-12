"""Command-line interface:  python -m geneweaver <command> ..."""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from .backend import get_backend
from .engine import OffTargetSearcher
from .io import read_guides, write_fasta, write_hits_csv
from .scoring import summarize_guides
from .synthetic import generate_dataset, recall


def _print_summary(rows, max_mm):
    cols = ["guide_id", "total_sites"] + [f"{m}mm" for m in range(max_mm + 1)] + ["specificity"]
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


def cmd_info(args):
    b = get_backend(not args.cpu)
    print(f"GeneWeaver {__version__}")
    print(f"Compute backend: {b.name}")
    return 0


def cmd_search(args):
    backend = get_backend(not args.cpu)
    guides = read_guides(args.guides)
    print(f"Backend: {backend.name} | guides: {len(guides)} | PAM: {args.pam} | "
          f"max mismatches: {args.max_mismatches}")
    searcher = OffTargetSearcher(guides, pam=args.pam, max_mismatches=args.max_mismatches,
                                 backend=backend, dtype="float16" if args.fp16 else "float32")
    t0 = time.perf_counter()
    hits = searcher.search_fasta(args.genome, progress=print)
    dt = time.perf_counter() - t0
    write_hits_csv(args.out, hits)
    print(f"\nFound {len(hits)} sites in {dt:.2f}s -> {args.out}\n")
    _print_summary(summarize_guides(hits, searcher.guide_ids, args.max_mismatches), args.max_mismatches)
    return 0


def cmd_generate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    contigs, guides, planted = generate_dataset(
        genome_len=args.length, n_contigs=args.contigs, n_guides=args.guides, seed=args.seed)
    write_fasta(os.path.join(args.out_dir, "sample_genome.fa"), contigs)
    with open(os.path.join(args.out_dir, "sample_guides.txt"), "w") as fh:
        fh.write("# name  20-nt guide sequence (5'->3', without PAM)\n")
        for k, v in guides.items():
            fh.write(f"{k}\t{v}\n")
    with open(os.path.join(args.out_dir, "planted_truth.tsv"), "w") as fh:
        fh.write("guide_id\tcontig\tstart\tstrand\tmismatches\tmismatch_positions\n")
        for p in planted:
            fh.write(f"{p.guide_id}\t{p.contig}\t{p.start}\t{p.strand}\t{p.mismatches}\t"
                     f"{';'.join(map(str, p.mm_positions)) or '-'}\n")
    print(f"Wrote sample_genome.fa, sample_guides.txt, planted_truth.tsv to {args.out_dir}/")
    return 0


def cmd_demo(args):
    backend = get_backend(not args.cpu)
    contigs, guides, planted = generate_dataset(genome_len=args.length, seed=args.seed)
    print(f"Backend: {backend.name}")
    print(f"Synthetic genome: {sum(map(len, contigs.values())):,} bp, {len(contigs)} contigs, "
          f"{len(guides)} guides, {len(planted)} planted sites")
    searcher = OffTargetSearcher(guides, backend=backend)
    t0 = time.perf_counter()
    hits = []
    for name, seq in contigs.items():
        hits.extend(searcher.search_sequence(name, seq))
    dt = time.perf_counter() - t0
    found, total, missing = recall(planted, hits)
    print(f"Search took {dt:.3f}s and returned {len(hits)} sites")
    print(f"Planted-site recall: {found}/{total}" + ("  (all recovered)" if not missing else f"  MISSING: {missing}"))
    print()
    _print_summary(summarize_guides(hits, searcher.guide_ids, 4), 4)
    return 0 if not missing else 1


def build_parser():
    p = argparse.ArgumentParser(prog="geneweaver", description="GeneWeaver: GPU-accelerated CRISPR off-target alignment engine")
    p.add_argument("--version", action="version", version=f"geneweaver {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="show which compute backend (GPU/CPU) will be used")
    s.add_argument("--cpu", action="store_true", help="force CPU")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("search", help="find on/off-target sites of guides in a genome FASTA")
    s.add_argument("--genome", required=True, help="genome FASTA (.fa/.fasta, optionally .gz)")
    s.add_argument("--guides", required=True, help="guide list file (name<TAB>20-nt sequence per line, or FASTA)")
    s.add_argument("--pam", default="NGG", help="PAM, IUPAC letters allowed (default NGG for SpCas9)")
    s.add_argument("--max-mismatches", type=int, default=4)
    s.add_argument("--out", default="results/hits.csv")
    s.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    s.add_argument("--fp16", action="store_true", help="use float16 matmul (faster on GPU tensor cores)")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("generate-data", help="create a synthetic genome with planted sites (for testing)")
    s.add_argument("--out-dir", default="data")
    s.add_argument("--length", type=int, default=200_000)
    s.add_argument("--contigs", type=int, default=2)
    s.add_argument("--guides", type=int, default=5)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("demo", help="end-to-end demo + recall check on synthetic data")
    s.add_argument("--length", type=int, default=200_000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

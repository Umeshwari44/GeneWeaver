# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

GeneWeaver finds every place in a genome where a CRISPR guide RNA could bind and cut
(its **on-target** and **off-target** sites), allowing mismatches, on both DNA strands,
and scores how risky each site is. The heavy part of the search is turned into a
single matrix multiplication, so it runs on a **GPU** (via CuPy) when one is available
and falls back automatically to NumPy on CPU.

> Infotact Solutions - Python Development internship, Project 2
> (Ref. ITS/PY/1445)

## Features

- Scans both strands of every contig in a (multi-)FASTA genome (`.fa`, `.fasta`, `.gz`)
- Configurable PAM with IUPAC letters (default `NGG` for SpCas9; try `NAG`, `NNGRRT`, ...)
- Up to N mismatches (default 4); `N` bases in the genome count as mismatches
- Reports coordinates, strand, mismatch positions, PAM, and the MIT off-target score
- Per-guide specificity summary
- GPU acceleration with automatic CPU fallback
- Verified against a brute-force reference implementation; 21 automated tests

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # numpy (+ pytest)
pip install -e .                                       # optional: gives the `geneweaver` command
```

**Optional GPU support** (needs an NVIDIA GPU + CUDA). Pick the CuPy wheel matching your CUDA:

```bash
pip install cupy-cuda12x     # CUDA 12.x   (use cupy-cuda11x for CUDA 11.x)
python -m geneweaver info    # should print: Compute backend: GPU (CuPy) - <your GPU>
```

No GPU on your laptop? Use a free **Google Colab** GPU runtime
(Runtime -> Change runtime type -> T4 GPU), upload the folder, and run the same commands.

## Quick start

```bash
python -m geneweaver demo                                   # synthetic data + recall check
python -m geneweaver generate-data --out-dir data           # sample genome, guides, ground truth
python -m geneweaver search \
    --genome data/sample_genome.fa \
    --guides data/sample_guides.txt \
    --max-mismatches 4 --pam NGG --out results/hits.csv
```

Guide file format (one per line, 20 nt, no PAM; `#` = comment):

```
guide1    AGCGCTATGGGATTGTAAAT
guide2    GACGTCAGTACGTTAGCCAT
```

Output `results/hits.csv` columns: `guide_id, guide_seq, contig, start, end, strand,
mismatches, mismatch_positions, site_seq, pam, score` (mismatched bases appear in
lowercase in `site_seq`; positions count from the 5' end of the guide).

Python API:

```python
from geneweaver import OffTargetSearcher
searcher = OffTargetSearcher({"g1": "AGCGCTATGGGATTGTAAAT"}, pam="NGG", max_mismatches=4)
hits = searcher.search_fasta("genome.fa")
```

## Tests and benchmark

```bash
python -m pytest -q                      # 21 tests
python benchmarks/benchmark.py           # writes results/benchmark.md
```

## Project layout

```
geneweaver/
  backend.py     GPU (CuPy) / CPU (NumPy) selection
  encoding.py    DNA <-> integers, reverse complement, PAM masks
  engine.py      the alignment engine (PAM scan + one-hot matmul)
  scoring.py     MIT off-target score and guide specificity
  io.py          FASTA / guide reading, CSV writing
  naive.py       slow reference implementation (for tests & benchmark)
  synthetic.py   synthetic genome with planted sites (ground truth)
  cli.py         command-line interface
tests/           pytest suite
benchmarks/      benchmark script
data/            sample genome, guides and planted ground truth
results/         example outputs (hits.csv, benchmark.md)
docs/EXPLANATION.md   full explanation of the project, code, and how to present it
```

## How it works (30 seconds)

1. Find every position followed by a valid PAM (vectorised).
2. Collect the 20 bases before each PAM into a matrix and one-hot encode it (n x 80).
3. Multiply by the one-hot guides (80 x G): each entry is the number of matching bases.
4. mismatches = 20 - matches; keep entries <= the limit.
5. Reverse-strand sites: repeat on the reverse complement and map coordinates back.

## Limitations

- Substitution mismatches only (no DNA/RNA bulges / indels).
- The sample genome is synthetic; for real work download a genome (e.g. human GRCh38)
  as FASTA. Whole-contig sequences are held in memory (largest human chromosome ~250 MB).
- GPU path is implemented with CuPy but has not been benchmarked on a physical GPU in
  the development environment; CPU results are measured and verified.
- The MIT score weights come from Hsu et al. 2013; newer models (CFD, CRISPRoff) exist.

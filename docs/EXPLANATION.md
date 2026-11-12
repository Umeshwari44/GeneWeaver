# GeneWeaver: Full Explanation

## 0. A note on "Assignment 1"

Only the Assignment 2 letter was shared with me, so I could not see your first
assignment. This repository is built purely from the Assignment 2 letter
("GeneWeaver": GPU-Accelerated CRISPR Alignment Engine). If Assignment 1 defined
specific requirements, file names, or a dataset, tell me and I will adapt the code.

## 1. The biology in plain words

- **DNA** is a long string over the letters A, C, G, T.
- **CRISPR-Cas9** is a molecular "scissors" that cuts DNA at a chosen place. You steer it
  with a **guide RNA**: a 20-letter sequence. Cas9 cuts where the DNA matches the guide
  **and** is immediately followed by a short signal called the **PAM** (for the common
  SpCas9 enzyme the PAM is `NGG`, where N = any base).
- Problem: Cas9 sometimes also cuts places that *almost* match the guide (a few letters
  differ = **mismatches**). These are **off-target** sites and can damage genes by mistake.
- So before using a guide, scientists search the whole genome for every on-target and
  near-match site. A human genome has 3 billion letters, so this must be fast. That is
  the job of this project: **alignment** (matching with mismatches) made fast with a **GPU**.

DNA has two strands. The second strand is the *reverse complement* of the first
(reverse the string, swap A<->T, C<->G). A guide can bind on either, so we search both.

## 2. The algorithm

Naive approach: for every position in the genome, compare 20 letters with every guide.
That is `genome_length x guides x 20` comparisons in slow Python loops.

GeneWeaver's approach:

1. **Encode** DNA as numbers: A=0, C=1, G=2, T=3, N/other=4. (`encoding.py`)
2. **PAM scan.** A position is only a candidate if the 3 bases after the 20-mer match
   `NGG`. Done for the whole chromosome at once with boolean array operations. This
   removes ~7/8 of all positions immediately. (`engine.py`, step 1)
3. **Gather.** Build a matrix `windows` of shape `(n_sites, 20)`: the 20 bases in front
   of each PAM.
4. **One-hot encode.** Each base becomes 4 numbers, e.g. G -> `[0,0,1,0]`, so each
   window becomes a row of 80 zeros/ones; each guide becomes a column of an
   `(80, n_guides)` matrix. An `N` becomes all zeros (so it never matches).
5. **Matrix multiplication.** `windows_onehot @ guides_onehot` gives, for every
   (site, guide) pair, the **number of matching letters** (each matching position adds 1).
   `mismatches = 20 - matches`. One matmul compares *all* sites with *all* guides.
6. **Filter** to `mismatches <= max_mismatches`, then compute coordinates, mismatch
   positions and the score for the few survivors.
7. **Reverse strand:** repeat on the reverse complement and convert coordinates back.

**Why GPU?** Step 5 is a big dense matrix product, the operation GPUs are built for
(thousands of cores, tensor cores). The code uses `xp = cupy` or `numpy` through the
same API, so one code path serves both. If CuPy or a CUDA GPU is missing, it silently
uses NumPy.

**Memory control:** sites are processed in chunks (`chunk_sites`, `matrix_budget`) so
the `(sites x guides)` result matrix never gets too large.

## 3. Scoring

For each hit, the **MIT off-target score** (Hsu et al. 2013) is computed from the
mismatch positions: mismatches near the PAM (positions 13-20, the "seed") are weighted
much more heavily because Cas9 is less tolerant there. Perfect match = 100, and the
score drops with more mismatches and with mismatches close together near the PAM.
The per-guide **specificity** = `10000 / (100 + sum of off-target scores)`; 100 means
the guide has no off-targets within the mismatch limit.

## 4. File-by-file

| File | What it does |
|---|---|
| `geneweaver/backend.py` | Chooses CuPy (GPU) or NumPy (CPU); `to_host` copies GPU arrays back |
| `geneweaver/encoding.py` | DNA<->numbers, reverse complement, IUPAC PAM masks |
| `geneweaver/engine.py` | `OffTargetSearcher`: the PAM scan + matmul engine; `Hit` record |
| `geneweaver/scoring.py` | MIT score, guide specificity, per-guide summary |
| `geneweaver/io.py` | Read FASTA and guide files; write the hits CSV |
| `geneweaver/naive.py` | Slow brute-force version used to prove the engine is correct |
| `geneweaver/synthetic.py` | Builds a random genome and plants sites with known mismatches |
| `geneweaver/cli.py` | `info`, `search`, `generate-data`, `demo` commands |
| `tests/test_core.py` | 21 tests (encoding, PAM, scoring, both strands, N bases, vs brute force, I/O, CLI) |
| `benchmarks/benchmark.py` | Timing comparison; writes `results/benchmark.md` |

## 5. Results measured in development (CPU only)

- 21/21 tests pass. On random genomes the engine returns **exactly** the same set of
  sites as the brute-force reference (guide, contig, start, strand, mismatches).
- Demo: 25/25 planted sites (5 guides x mismatch levels 0-4, both strands) recovered,
  with correct mismatch positions.
- Timing (20 guides, NGG, up to 4 mismatches, both strands, CPU/NumPy):

| Genome size | Brute-force Python | GeneWeaver (CPU) |
|---:|---:|---:|
| 100,000 bp | 0.30 s | 0.01 s |
| 1,000,000 bp | not run | 0.05 s |
| 10,000,000 bp | not run | 0.55 s |
| 50,000,000 bp | not run | 2.53 s |

Even on CPU the vectorised engine is ~50x faster than loops on the 100 kb case.
**GPU timings are not included because no GPU was available where I built this.**
When you run `python benchmarks/benchmark.py` on a GPU machine (e.g. Colab T4) it
automatically adds a GPU column. Put those real numbers in your report.

## 6. Running it, step by step

```bash
cd geneweaver
pip install -r requirements.txt
python -m pytest -q                      # should say 21 passed
python -m geneweaver demo                # quick proof it works
python -m geneweaver search --genome data/sample_genome.fa \
        --guides data/sample_guides.txt --out results/hits.csv
python benchmarks/benchmark.py
```

To use a **real genome** (e.g. a bacterial genome or one human chromosome), download
a FASTA from NCBI/Ensembl/UCSC and pass it to `--genome`. Use real guides as 20-letter
lines in a text file.

## 7. Putting it on GitHub (the "repository")

```bash
cd geneweaver
git init
git add .
git commit -m "GeneWeaver: GPU-accelerated CRISPR alignment engine"
# create an empty repo named geneweaver on github.com, then:
git branch -M main
git remote add origin https://github.com/<your-username>/geneweaver.git
git push -u origin main
```

## 8. Mapping to the deliverables in your letter

| Required | Where it is |
|---|---|
| Source code | `geneweaver/` package |
| Project documentation | `README.md`, this file |
| Project report | Write from sections 1-5 of this file (intro, method, results, limitations); I can draft it |
| Presentation (PPT) | Slides: problem -> CRISPR off-targets -> algorithm diagram -> results/benchmark -> demo -> limitations; I can build it |
| Model evaluation results | `results/benchmark.md`, test output, recall 25/25 |
| Final demonstration | Run `python -m geneweaver demo` and the `search` command live |
| Periodic progress updates / version control | Use git commits (section 7) |
| Originality | Write your own words in the report and be ready to explain every file |

## 9. Likely questions from your mentor

- **Why one-hot and matmul?** It turns "count matching letters for all pairs" into a
  single dense matrix product, which GPUs run extremely fast, and avoids Python loops.
- **Why filter by PAM first?** Cas9 needs the PAM, and it discards ~7/8 of positions cheaply.
- **How do you handle the reverse strand?** Search the reverse complement and map the
  coordinates back (`start = n - i - 20 + 1`).
- **How do you know it's correct?** Tests compare against a brute-force implementation
  on random genomes, and planted sites with known mismatches are all recovered.
- **What happens without a GPU?** `backend.py` falls back to NumPy on CPU.
- **What about `N` in the genome?** One-hot of `N` is all zeros, so it counts as a mismatch.
- **Limitations?** No insertions/deletions (bulges); MIT score is an older model;
  whole chromosome must fit in memory; GPU path not benchmarked on real hardware here.
- **Possible improvements:** bulge support, CFD scoring, streaming huge chromosomes,
  multi-GPU, a seed-index (FM-index) to prune candidates, a web/REST front end.

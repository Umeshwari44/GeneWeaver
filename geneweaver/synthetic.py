"""Synthetic genome generator with planted off-target sites (ground truth).

We plant, for every guide, one site for each mismatch level requested, on random
contigs / strands, each followed by a valid NGG PAM. Because we know exactly what
was planted, we can measure the engine's recall.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from .encoding import reverse_complement


@dataclass
class Planted:
    guide_id: str
    contig: str
    start: int       # 1-based forward coordinate of protospacer start
    strand: str
    mismatches: int
    mm_positions: Tuple[int, ...]

    def key(self):
        return (self.guide_id, self.contig, self.start, self.strand)


def generate_dataset(
    genome_len: int = 200_000,
    n_contigs: int = 2,
    n_guides: int = 5,
    mismatch_profile: Tuple[int, ...] = (0, 1, 2, 3, 4),
    seed: int = 7,
) -> Tuple[Dict[str, str], Dict[str, str], List[Planted]]:
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    per_contig = genome_len // n_contigs
    letters = np.frombuffer(b"ACGT", dtype=np.uint8)
    contigs = {
        f"chr{i + 1}": bytearray(letters[rng.integers(0, 4, per_contig)].tobytes())
        for i in range(n_contigs)
    }
    guides = {
        f"guide{i + 1}": letters[rng.integers(0, 4, 20)].tobytes().decode()
        for i in range(n_guides)
    }

    used: Dict[str, List[Tuple[int, int]]] = {c: [] for c in contigs}
    planted: List[Planted] = []
    for gid, g in guides.items():
        for n_mm in mismatch_profile:
            # choose mismatch positions and substitute bases
            positions = sorted(pyrng.sample(range(20), n_mm))
            site = list(g)
            for p in positions:
                site[p] = pyrng.choice([b for b in "ACGT" if b != g[p]])
            unit = "".join(site) + pyrng.choice("ACGT") + "GG"   # 20 nt + NGG
            strand = pyrng.choice("+-")
            for _ in range(1000):
                contig = pyrng.choice(list(contigs))
                p0 = pyrng.randrange(50, len(contigs[contig]) - 50 - 23)
                if all(p0 + 23 + 30 < a or p0 > b + 30 for a, b in used[contig]):
                    break
            else:
                raise RuntimeError("Could not place site - genome too small")
            used[contig].append((p0, p0 + 22))
            if strand == "+":
                contigs[contig][p0:p0 + 23] = unit.encode()
                start = p0 + 1
            else:
                contigs[contig][p0:p0 + 23] = reverse_complement(unit).encode()
                start = p0 + 3 + 1   # protospacer sits after the 3-nt reverse-complemented PAM
            planted.append(Planted(gid, contig, start, strand, n_mm, tuple(p + 1 for p in positions)))
    return {k: v.decode() for k, v in contigs.items()}, guides, planted


def recall(planted: List[Planted], hits) -> Tuple[int, int, Set]:
    found = {(h.guide_id, h.contig, h.start, h.strand) for h in hits}
    missing = {p.key() for p in planted} - found
    return len(planted) - len(missing), len(planted), missing

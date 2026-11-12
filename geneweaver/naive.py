"""Slow, obviously-correct reference implementation (pure Python loops).

Used by the tests to verify the fast engine, and by the benchmark as a baseline.
Returns a set of (guide_id, contig, start, strand, n_mismatches) tuples.
"""
from __future__ import annotations

from typing import Dict, Set, Tuple

from .encoding import IUPAC, reverse_complement


def naive_search(contigs: Dict[str, str], guides: Dict[str, str],
                 pam: str = "NGG", max_mismatches: int = 4) -> Set[Tuple[str, str, int, str, int]]:
    pam = pam.upper()
    allowed = [set(IUPAC[c]) for c in pam]
    p = len(pam)
    results = set()
    for contig, seq in contigs.items():
        seq = seq.upper()
        n = len(seq)
        for strand, s in (("+", seq), ("-", reverse_complement(seq))):
            for i in range(n - 20 - p + 1):
                if any(s[i + 20 + k] not in allowed[k] for k in range(p)):
                    continue
                site = s[i:i + 20]
                for gid, g in guides.items():
                    mm = sum(1 for a, b in zip(site, g) if a != b)
                    if mm <= max_mismatches:
                        start = i + 1 if strand == "+" else n - i - 20 + 1
                        results.add((gid, contig, start, strand, mm))
    return results

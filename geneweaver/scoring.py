"""Off-target scoring.

Implements the MIT off-target score (Hsu et al., Nature Biotechnology 2013) as used
by common tools such as CRISPOR, plus the aggregate guide specificity score.

Mismatches near the PAM (the 3' end of the 20-nt guide, seed region) are tolerated
much less than mismatches at the 5' end, so they get larger weights.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

# Position 1 = 5' end of guide ... position 20 = next to the PAM.
MIT_WEIGHTS = [
    0.0, 0.0, 0.014, 0.0, 0.0, 0.395, 0.317, 0.0, 0.389, 0.079,
    0.445, 0.508, 0.613, 0.851, 0.732, 0.828, 0.615, 0.804, 0.685, 0.583,
]


def mit_off_target_score(mm_positions: Sequence[int]) -> float:
    """Score in [0, 100] for one site. `mm_positions` are 1-based (1 = 5' end).
    100 = perfect match; lower = less likely to be cleaved."""
    n = len(mm_positions)
    if n == 0:
        return 100.0
    pos = sorted(p - 1 for p in mm_positions)
    prod = 1.0
    for p in pos:
        prod *= 1.0 - MIT_WEIGHTS[p]
    if n > 1:
        mean_dist = (pos[-1] - pos[0]) / (n - 1)
    else:
        mean_dist = 19.0
    dist_factor = 1.0 / (((19.0 - mean_dist) / 19.0) * 4.0 + 1.0)
    return 100.0 * prod * dist_factor / (n * n)


def guide_specificity(off_target_scores: Iterable[float]) -> float:
    """Aggregate specificity in (0, 100]: 100 means no off-targets at all.
    `off_target_scores` must exclude the perfect-match (on-target) site(s)."""
    return 10000.0 / (100.0 + sum(off_target_scores))


def summarize_guides(hits, guide_ids: List[str], max_mismatches: int) -> List[Dict]:
    """One summary row per guide: hit counts per mismatch level + specificity."""
    by_guide: Dict[str, list] = {g: [] for g in guide_ids}
    for h in hits:
        by_guide[h.guide_id].append(h)
    rows = []
    for g in guide_ids:
        hs = by_guide[g]
        counts = Counter(h.mismatches for h in hs)
        off = [h.score for h in hs if h.mismatches > 0]
        row = {"guide_id": g, "total_sites": len(hs)}
        for m in range(max_mismatches + 1):
            row[f"{m}mm"] = counts.get(m, 0)
        row["specificity"] = round(guide_specificity(off), 2)
        rows.append(row)
    return rows

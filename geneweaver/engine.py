"""GeneWeaver alignment engine: GPU-accelerated CRISPR off-target search.

Pipeline for each contig and each strand:
  1. PAM scan      - find every position followed by a valid PAM (e.g. NGG), fully
                     vectorised (one boolean mask per PAM letter).
  2. Gather        - pull the 20-nt protospacer in front of each PAM into an
                     (n_sites x 20) integer matrix.
  3. One-hot       - turn that into an (n_sites x 80) 0/1 matrix.
  4. Matrix product- (n_sites x 80) @ (80 x n_guides) = number of MATCHING bases for
                     every (site, guide) pair at once. mismatches = 20 - matches.
                     This is a dense matmul, which is exactly what GPUs are best at.
  5. Filter        - keep pairs with <= max_mismatches and build Hit records.

Reverse-strand sites are found by running the same procedure on the reverse
complement of the contig and mapping coordinates back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backend import Backend, get_backend
from .encoding import decode, encode, pam_masks, reverse_complement_encoded
from .io import read_fasta
from .scoring import mit_off_target_score

GUIDE_LEN = 20


@dataclass
class Hit:
    guide_id: str
    guide_seq: str
    contig: str
    start: int            # 1-based, forward-strand coordinate of protospacer start
    end: int              # 1-based inclusive
    strand: str           # '+' or '-'
    mismatches: int
    mm_positions: Tuple[int, ...]   # 1-based, 1 = 5' end of guide, 20 = PAM-proximal
    site_seq: str         # target as read 5'->3' in guide orientation; mismatches lowercase
    pam: str
    score: float          # MIT off-target score (100 = perfect match)


class OffTargetSearcher:
    def __init__(
        self,
        guides: Dict[str, str],
        pam: str = "NGG",
        max_mismatches: int = 4,
        backend: Optional[Backend] = None,
        dtype: str = "float32",
        chunk_sites: int = 200_000,
        matrix_budget: int = 40_000_000,
    ):
        if not guides:
            raise ValueError("At least one guide is required")
        for gid, seq in guides.items():
            if len(seq) != GUIDE_LEN or set(seq.upper()) - set("ACGT"):
                raise ValueError(f"Guide {gid!r} must be {GUIDE_LEN} nt of A/C/G/T")
        if not 0 <= max_mismatches <= GUIDE_LEN:
            raise ValueError("max_mismatches must be between 0 and 20")

        self.b = backend or get_backend()
        self.xp = self.b.xp
        self.guide_ids = list(guides)
        self.guide_seqs = [guides[g].upper() for g in self.guide_ids]
        self.pam = pam.upper()
        self.pam_len = len(self.pam)
        self.max_mm = max_mismatches
        self.dtype = dtype
        self.chunk_sites = chunk_sites
        self.matrix_budget = matrix_budget

        self._pam_masks = [self.xp.asarray(m) for m in pam_masks(self.pam)]
        self._offsets = self.xp.arange(GUIDE_LEN)
        self._bases = self.xp.arange(4, dtype=self.xp.uint8)

        # Guides as one-hot matrix of shape (80, n_guides)
        enc = np.stack([encode(s) for s in self.guide_seqs])             # (G, 20)
        onehot = (enc[:, :, None] == np.arange(4)).reshape(len(enc), -1)  # (G, 80)
        self._guide_onehot = self.xp.asarray(onehot.T.astype(dtype))     # (80, G)

    # ------------------------------------------------------------------ public
    def search_sequence(self, name: str, seq: str) -> List[Hit]:
        enc = encode(seq)
        hits = self._scan_strand(name, enc, "+")
        hits += self._scan_strand(name, reverse_complement_encoded(enc), "-")
        return hits

    def search_fasta(self, path: str, progress=None) -> List[Hit]:
        hits: List[Hit] = []
        for name, seq in read_fasta(path):
            if progress:
                progress(f"  scanning {name} ({len(seq):,} bp)")
            hits.extend(self.search_sequence(name, seq))
        hits.sort(key=lambda h: (self.guide_ids.index(h.guide_id), h.mismatches, h.contig, h.start))
        return hits

    # ---------------------------------------------------------------- internal
    def _scan_strand(self, contig: str, enc: np.ndarray, strand: str) -> List[Hit]:
        xp = self.xp
        n = len(enc)
        span = GUIDE_LEN + self.pam_len
        if n < span:
            return []
        n_starts = n - span + 1
        enc_d = xp.asarray(enc)

        # 1. PAM scan: start i is a candidate if enc[i+20+k] matches PAM letter k.
        ok = xp.ones(n_starts, dtype=bool)
        for k, mask in enumerate(self._pam_masks):
            ok &= mask[enc_d[GUIDE_LEN + k: GUIDE_LEN + k + n_starts]]
        starts = xp.nonzero(ok)[0]
        n_guides = len(self.guide_ids)
        step = max(1024, min(self.chunk_sites, self.matrix_budget // n_guides))

        hits: List[Hit] = []
        for s in range(0, len(starts), step):
            st = starts[s: s + step]
            # 2. gather protospacers  (S, 20)
            windows = enc_d[st[:, None] + self._offsets[None, :]]
            # 3. one-hot (S, 80). Genome 'N' (code 4) -> all zeros -> counts as mismatch.
            onehot = (windows[:, :, None] == self._bases[None, None, :]).reshape(len(st), -1)
            # 4. matches for every (site, guide) pair via one matrix product
            matches = onehot.astype(self.dtype) @ self._guide_onehot        # (S, G)
            mism = GUIDE_LEN - matches
            # 5. filter
            site_idx, guide_idx = xp.nonzero(mism <= self.max_mm + 0.5)
            if len(site_idx) == 0:
                continue
            pos = self.b.to_host(st[site_idx])
            gidx = self.b.to_host(guide_idx)
            hits.extend(self._make_hits(contig, enc, strand, pos, gidx))
        return hits

    def _make_hits(self, contig, enc_host, strand, positions, guide_indices) -> List[Hit]:
        n = len(enc_host)
        out = []
        for i, g in zip(positions.tolist(), guide_indices.tolist()):
            guide = self.guide_seqs[g]
            site = decode(enc_host[i: i + GUIDE_LEN])
            mm_pos = tuple(k + 1 for k in range(GUIDE_LEN) if site[k] != guide[k])
            marked = "".join(c.lower() if site[k] != guide[k] else c for k, c in enumerate(site))
            if strand == "+":
                start, end = i + 1, i + GUIDE_LEN
            else:  # position i on the reverse complement maps back to the forward strand
                start, end = n - i - GUIDE_LEN + 1, n - i
            out.append(Hit(
                guide_id=self.guide_ids[g], guide_seq=guide, contig=contig,
                start=start, end=end, strand=strand, mismatches=len(mm_pos),
                mm_positions=mm_pos, site_seq=marked,
                pam=decode(enc_host[i + GUIDE_LEN: i + GUIDE_LEN + self.pam_len]),
                score=mit_off_target_score(mm_pos),
            ))
        return out

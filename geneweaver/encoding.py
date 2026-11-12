"""DNA <-> integer encoding helpers.

A=0, C=1, G=2, T=3, anything else (N, IUPAC codes, gaps) = 4.
With this order the complement of base b (b < 4) is simply 3 - b.
"""
from __future__ import annotations

import numpy as np

_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i

_DECODE = np.frombuffer(b"ACGTN", dtype=np.uint8)

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}

_COMP_STR = str.maketrans("ACGTacgtN", "TGCAtgcaN")


def encode(seq: str) -> np.ndarray:
    """String -> uint8 array (A,C,G,T -> 0..3 ; everything else -> 4)."""
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def decode(arr: np.ndarray) -> str:
    return _DECODE[np.asarray(arr)].tobytes().decode("ascii")


def reverse_complement_encoded(arr: np.ndarray) -> np.ndarray:
    comp = np.where(arr < 4, 3 - arr, 4).astype(np.uint8)
    return comp[::-1].copy()


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMP_STR)[::-1]


def pam_masks(pam: str) -> np.ndarray:
    """Boolean lookup table, shape (len(pam), 5): masks[k, base] is True if the
    encoded base matches the k-th PAM symbol. Genome 'N' (code 4) never matches."""
    pam = pam.upper()
    masks = np.zeros((len(pam), 5), dtype=bool)
    for k, symbol in enumerate(pam):
        if symbol not in IUPAC:
            raise ValueError(f"Invalid PAM symbol {symbol!r} in {pam!r}")
        for base in IUPAC[symbol]:
            masks[k, "ACGT".index(base)] = True
    return masks

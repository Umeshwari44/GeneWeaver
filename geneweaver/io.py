"""FASTA / guide-list input and CSV output."""
from __future__ import annotations

import csv
import gzip
from typing import Dict, Iterator, Tuple

HIT_COLUMNS = [
    "guide_id", "guide_seq", "contig", "start", "end", "strand",
    "mismatches", "mismatch_positions", "site_seq", "pam", "score",
]


def _open(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (name, sequence) for every record. Name = first word of the header."""
    name, chunks = None, []
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                if name is None:
                    raise ValueError(f"{path}: sequence data before first '>' header")
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def read_guides(path: str, guide_len: int = 20) -> Dict[str, str]:
    """Read guides from a text file. Accepted line formats:
        ACGT...(20 nt)
        name<whitespace or comma>ACGT...
    or a FASTA file (header = name). Lines starting with '#' are ignored."""
    guides: Dict[str, str] = {}
    with _open(path) as fh:
        text = fh.read()
    if text.lstrip().startswith(">"):
        records = []
        name, chunks = None, []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
        if name is not None:
            records.append((name, "".join(chunks)))
    else:
        records = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            records.append((parts[0], parts[1]) if len(parts) >= 2 else (f"guide{len(records)+1}", parts[0]))
    for name, seq in records:
        seq = seq.upper().replace("U", "T")
        if len(seq) != guide_len or set(seq) - set("ACGT"):
            raise ValueError(f"Guide {name!r} must be exactly {guide_len} nt of A/C/G/T, got {seq!r}")
        if name in guides:
            raise ValueError(f"Duplicate guide name {name!r}")
        guides[name] = seq
    if not guides:
        raise ValueError(f"No guides found in {path}")
    return guides


def write_fasta(path: str, contigs: Dict[str, str], width: int = 80) -> None:
    with open(path, "w") as fh:
        for name, seq in contigs.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")


def write_hits_csv(path: str, hits) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HIT_COLUMNS)
        for h in hits:
            w.writerow([
                h.guide_id, h.guide_seq, h.contig, h.start, h.end, h.strand,
                h.mismatches, ";".join(map(str, h.mm_positions)) or "-",
                h.site_seq, h.pam, f"{h.score:.3f}",
            ])

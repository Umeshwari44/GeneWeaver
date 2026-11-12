import numpy as np
import pytest

from geneweaver.backend import get_backend
from geneweaver.encoding import (decode, encode, pam_masks, reverse_complement,
                                 reverse_complement_encoded)
from geneweaver.engine import OffTargetSearcher
from geneweaver.io import read_fasta, read_guides, write_fasta, write_hits_csv
from geneweaver.naive import naive_search
from geneweaver.scoring import guide_specificity, mit_off_target_score
from geneweaver.synthetic import generate_dataset, recall

CPU = get_backend(prefer_gpu=False)


# ---------------------------------------------------------------- encoding
def test_encode_decode_roundtrip():
    assert decode(encode("ACGTNacgt")) == "ACGTNACGT"


def test_reverse_complement():
    assert reverse_complement("AACG") == "CGTT"
    assert decode(reverse_complement_encoded(encode("AACG"))) == "CGTT"


def test_pam_masks_ngg():
    m = pam_masks("NGG")
    assert m.shape == (3, 5)
    assert m[0, :4].all() and not m[0, 4]        # N matches ACGT but not genome-N
    assert list(m[1]) == [False, False, True, False, False]


def test_invalid_pam():
    with pytest.raises(ValueError):
        pam_masks("NXG")


# ----------------------------------------------------------------- scoring
def test_mit_score_perfect_and_monotonic():
    assert mit_off_target_score([]) == 100.0
    one_seed = mit_off_target_score([20])      # seed mismatch -> strongly reduced
    one_distal = mit_off_target_score([1])     # distal mismatch -> still high
    assert one_distal == 100.0                 # weight 0 at position 1
    assert one_seed < 50
    assert mit_off_target_score([20, 19]) < one_seed


def test_specificity():
    assert guide_specificity([]) == 100.0
    assert guide_specificity([100.0]) == 50.0


# ------------------------------------------------------------------ engine
def test_exact_match_both_strands():
    guide = "GACGTCAGTACGTTAGCCAT"
    fwd = "TTT" + guide + "AGG" + "CCC"
    rev = "TTT" + reverse_complement(guide + "TGG") + "CCC"
    s = OffTargetSearcher({"g": guide}, max_mismatches=0, backend=CPU)
    h = s.search_sequence("c", fwd)
    assert [(x.start, x.strand, x.mismatches, x.pam) for x in h] == [(4, "+", 0, "AGG")]
    h = s.search_sequence("c", rev)
    assert [(x.start, x.strand, x.mismatches, x.pam) for x in h] == [(7, "-", 0, "TGG")]


def test_no_pam_no_hit():
    guide = "GACGTCAGTACGTTAGCCAT"
    s = OffTargetSearcher({"g": guide}, max_mismatches=4, backend=CPU)
    assert s.search_sequence("c", "TTT" + guide + "ATC" + "CCC") == []


def test_mismatch_reporting():
    guide = "GACGTCAGTACGTTAGCCAT"
    site = "CACGTCAGTACGTTAGCCAA"      # mismatches at positions 1 and 20
    s = OffTargetSearcher({"g": guide}, max_mismatches=4, backend=CPU)
    h = s.search_sequence("c", site + "CGG")
    assert len(h) == 1
    assert h[0].mm_positions == (1, 20)
    assert h[0].site_seq == "cACGTCAGTACGTTAGCCAa"


def test_n_in_genome_counts_as_mismatch():
    guide = "GACGTCAGTACGTTAGCCAT"
    site = "GACGTCAGTACGTTAGCCAN"
    s = OffTargetSearcher({"g": guide}, max_mismatches=1, backend=CPU)
    h = s.search_sequence("c", site + "AGG")
    assert len(h) == 1 and h[0].mm_positions == (20,)
    assert OffTargetSearcher({"g": guide}, max_mismatches=0, backend=CPU).search_sequence("c", site + "AGG") == []


def test_short_sequence_ok():
    s = OffTargetSearcher({"g": "A" * 20}, backend=CPU)
    assert s.search_sequence("c", "ACGT") == []


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_bruteforce_on_random_genomes(seed):
    contigs, guides, _ = generate_dataset(genome_len=20_000, n_contigs=2, n_guides=4, seed=seed)
    s = OffTargetSearcher(guides, max_mismatches=4, backend=CPU)
    hits = []
    for n, q in contigs.items():
        hits += s.search_sequence(n, q)
    got = {(h.guide_id, h.contig, h.start, h.strand, h.mismatches) for h in hits}
    assert got == naive_search(contigs, guides, "NGG", 4)


def test_alternative_pam_nag():
    contigs, guides, _ = generate_dataset(genome_len=10_000, n_contigs=1, n_guides=3, seed=5)
    s = OffTargetSearcher(guides, pam="NAG", max_mismatches=6, backend=CPU)
    hits = s.search_sequence("chr1", contigs["chr1"])
    got = {(h.guide_id, h.contig, h.start, h.strand, h.mismatches) for h in hits}
    assert got == naive_search({"chr1": contigs["chr1"]}, guides, "NAG", 6)


def test_chunking_does_not_change_results():
    contigs, guides, _ = generate_dataset(genome_len=20_000, n_contigs=1, n_guides=3, seed=9)
    a = OffTargetSearcher(guides, backend=CPU, chunk_sites=1024)
    b = OffTargetSearcher(guides, backend=CPU, chunk_sites=10_000_000)
    ha = {(h.guide_id, h.start, h.strand) for h in a.search_sequence("c", contigs["chr1"])}
    hb = {(h.guide_id, h.start, h.strand) for h in b.search_sequence("c", contigs["chr1"])}
    assert ha == hb


def test_recall_of_planted_sites():
    contigs, guides, planted = generate_dataset(genome_len=100_000, seed=11)
    s = OffTargetSearcher(guides, max_mismatches=4, backend=CPU)
    hits = []
    for n, q in contigs.items():
        hits += s.search_sequence(n, q)
    found, total, missing = recall(planted, hits)
    assert not missing and found == total == 25
    # mismatch counts and positions must also agree with ground truth
    by_key = {(h.guide_id, h.contig, h.start, h.strand): h for h in hits}
    for p in planted:
        assert by_key[p.key()].mm_positions == p.mm_positions


def test_validation_errors():
    with pytest.raises(ValueError):
        OffTargetSearcher({"g": "ACGT"}, backend=CPU)
    with pytest.raises(ValueError):
        OffTargetSearcher({}, backend=CPU)


# ---------------------------------------------------------------------- io
def test_fasta_and_guides_io(tmp_path):
    write_fasta(str(tmp_path / "g.fa"), {"a": "ACGT" * 50, "b": "TTTT"})
    assert dict(read_fasta(str(tmp_path / "g.fa"))) == {"a": "ACGT" * 50, "b": "TTTT"}
    gp = tmp_path / "guides.txt"
    gp.write_text("# comment\ng1\tGACGTCAGTACGTTAGCCAT\nGACGTCAGTACGTTAGCCAA\n")
    g = read_guides(str(gp))
    assert g == {"g1": "GACGTCAGTACGTTAGCCAT", "guide2": "GACGTCAGTACGTTAGCCAA"}
    gp.write_text("g1\tACGT\n")
    with pytest.raises(ValueError):
        read_guides(str(gp))


def test_write_hits_csv(tmp_path):
    guide = "GACGTCAGTACGTTAGCCAT"
    hits = OffTargetSearcher({"g": guide}, backend=CPU).search_sequence("c", guide + "AGG")
    out = tmp_path / "h.csv"
    write_hits_csv(str(out), hits)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("guide_id,guide_seq,contig")
    assert len(lines) == 2


# --------------------------------------------------------------------- cli
def test_cli_end_to_end(tmp_path, capsys):
    from geneweaver.cli import main
    d = str(tmp_path)
    assert main(["generate-data", "--out-dir", d, "--length", "60000"]) == 0
    out = str(tmp_path / "hits.csv")
    assert main(["search", "--genome", d + "/sample_genome.fa", "--guides", d + "/sample_guides.txt",
                 "--out", out, "--cpu"]) == 0
    assert (tmp_path / "hits.csv").exists()
    assert main(["demo", "--length", "50000", "--cpu"]) == 0
    assert "all recovered" in capsys.readouterr().out
    assert main(["search", "--genome", "nope.fa", "--guides", d + "/sample_guides.txt", "--cpu"]) == 2

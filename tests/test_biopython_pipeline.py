from geneweaver.biopython_pipeline import (
    clean_sequence,
    chunk_sequence,
    read_genome_fasta,
    process_genome,
)


def test_clean_sequence():
    assert clean_sequence("acgt acgt") == "ACGTACGT"


def test_chunk_sequence():
    chunks = chunk_sequence("ACGTACGT", chunk_size=4)
    assert chunks == ["ACGT", "ACGT"]


def test_chunk_sequence_with_remainder():
    chunks = chunk_sequence("ACGTACGTA", chunk_size=4)
    assert chunks == ["ACGT", "ACGT", "A"]


def test_invalid_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_sequence("ACGT", chunk_size=0)


def test_read_genome_fasta(tmp_path):
    fasta_file = tmp_path / "test.fa"
    fasta_file.write_text(">chr1\nacgtacgt\n")

    genome = read_genome_fasta(fasta_file)

    assert genome == {"chr1": "ACGTACGT"}


def test_process_genome(tmp_path):
    fasta_file = tmp_path / "test.fa"
    fasta_file.write_text(">chr1\nACGTACGT\n")

    result = process_genome(fasta_file, chunk_size=4)

    assert result == {"chr1": ["ACGT", "ACGT"]}
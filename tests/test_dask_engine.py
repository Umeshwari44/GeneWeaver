import numpy as np
import pytest

from geneweaver.dask_engine import process_genome_distributed


def test_process_genome_distributed(tmp_path):
    fasta_file = tmp_path / "test.fa"
    fasta_file.write_text(">chr1\nACGTACGT\n")

    result = process_genome_distributed(
        fasta_file,
        chunk_size=4,
        n_workers=1
    )

    assert len(result["chr1"]) == 2
    assert all(isinstance(chunk, np.ndarray) for chunk in result["chr1"])
    assert result["chr1"][0].tolist() == [0, 1, 2, 3]
    assert result["chr1"][1].tolist() == [0, 1, 2, 3]


def test_process_genome_distributed_multiple_contigs(tmp_path):
    fasta_file = tmp_path / "test.fa"
    fasta_file.write_text(">chr1\nACGT\n>chr2\nTTAA\n")

    result = process_genome_distributed(
        fasta_file,
        chunk_size=2,
        n_workers=1
    )

    assert result["chr1"][0].tolist() == [0, 1]
    assert result["chr1"][1].tolist() == [2, 3]
    assert result["chr2"][0].tolist() == [3, 3]
    assert result["chr2"][1].tolist() == [0, 0]


def test_distributed_invalid_chunk_size(tmp_path):
    fasta_file = tmp_path / "test.fa"
    fasta_file.write_text(">chr1\nACGT\n")

    with pytest.raises(ValueError):
        process_genome_distributed(fasta_file, chunk_size=0)


def test_distributed_invalid_worker_count(tmp_path):
    fasta_file = tmp_path / "test.fa"
    fasta_file.write_text(">chr1\nACGT\n")

    with pytest.raises(ValueError):
        process_genome_distributed(fasta_file, n_workers=0)


def test_distributed_unknown_dna_bases(tmp_path):
    fasta_file = tmp_path / "test.fa"
    fasta_file.write_text(">chr1\nACNT\n")

    result = process_genome_distributed(
        fasta_file,
        chunk_size=4,
        n_workers=1
    )

    assert result["chr1"][0].tolist() == [0, 1, 4, 3]


def test_distributed_chunk_with_remainder(tmp_path):
    fasta_file = tmp_path / "test.fa"
    fasta_file.write_text(">chr1\nACGTACG\n")

    result = process_genome_distributed(
        fasta_file,
        chunk_size=4,
        n_workers=1
    )

    assert len(result["chr1"]) == 2
    assert result["chr1"][0].tolist() == [0, 1, 2, 3]
    assert result["chr1"][1].tolist() == [0, 1, 2]

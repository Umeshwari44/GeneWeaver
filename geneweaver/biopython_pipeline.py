"""BioPython-based FASTA processing for GeneWeaver."""

from Bio import SeqIO
from geneweaver.encoding import encode


def clean_sequence(sequence):
    """Convert DNA sequence to uppercase and remove whitespace."""
    return str(sequence).upper().replace(" ", "").replace("\n", "")


def read_genome_fasta(file_path):
    """Read genome records from a FASTA file using BioPython."""
    genome = {}

    for record in SeqIO.parse(file_path, "fasta"):
        name = record.id
        sequence = clean_sequence(record.seq)

        if not sequence:
            raise ValueError(f"Empty sequence found: {name}")

        genome[name] = sequence

    if not genome:
        raise ValueError("No FASTA records found.")

    return genome


def chunk_sequence(sequence, chunk_size=10000):
    """Split a DNA sequence into smaller chunks."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    return [
        sequence[i:i + chunk_size]
        for i in range(0, len(sequence), chunk_size)
    ]


def process_genome(file_path, chunk_size=10000):
    """Read, clean, and chunk all genome records."""
    genome = read_genome_fasta(file_path)

    processed = {}

    for name, sequence in genome.items():
        processed[name] = chunk_sequence(sequence, chunk_size)

    return processed


def process_genome_as_arrays(file_path, chunk_size=10000):
    """Read and encode FASTA records individually to reduce memory overhead."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    processed = {}

    for record in SeqIO.parse(file_path, "fasta"):
        name = record.id
        sequence = clean_sequence(record.seq)

        if not sequence:
            raise ValueError(f"Empty sequence found: {name}")

        encoded_sequence = encode(sequence)

        processed[name] = [
            encoded_sequence[i:i + chunk_size]
            for i in range(0, len(encoded_sequence), chunk_size)
        ]

    if not processed:
        raise ValueError("No FASTA records found.")

    return processed

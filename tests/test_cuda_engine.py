import numpy as np
import pytest

from geneweaver.cuda_engine import count_mismatches


def test_count_mismatches():
    guides = np.array(
        [[0, 1, 2, 3], [0, 0, 0, 0]],
        dtype=np.uint8
    )
    sites = np.array(
        [[0, 1, 2, 3], [1, 1, 0, 2]],
        dtype=np.uint8
    )

    result = count_mismatches(guides, sites)

    np.testing.assert_array_equal(result, [0, 3])


def test_multiple_guide_site_pairs():
    guides = np.array(
        [[0, 1, 2, 3], [0, 0, 0, 0], [1, 1, 1, 1]],
        dtype=np.uint8
    )
    sites = np.array(
        [[0, 1, 2, 3], [1, 1, 0, 2], [1, 1, 1, 0]],
        dtype=np.uint8
    )

    result = count_mismatches(guides, sites)

    np.testing.assert_array_equal(result, [0, 3, 1])


def test_empty_batch():
    guides = np.empty((0, 4), dtype=np.uint8)
    sites = np.empty((0, 4), dtype=np.uint8)

    result = count_mismatches(guides, sites)

    assert result.size == 0
    assert result.dtype == np.int32


def test_invalid_dimensions():
    guides = np.array([0, 1, 2, 3], dtype=np.uint8)
    sites = np.array([[0, 1, 2, 3]], dtype=np.uint8)

    with pytest.raises(ValueError, match="2D arrays"):
        count_mismatches(guides, sites)


def test_different_shapes():
    guides = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    sites = np.array([[0, 1, 2]], dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        count_mismatches(guides, sites)


def test_empty_sequences():
    guides = np.empty((2, 0), dtype=np.uint8)
    sites = np.empty((2, 0), dtype=np.uint8)

    with pytest.raises(ValueError, match="cannot be empty"):
        count_mismatches(guides, sites)

def test_negative_dna_encoding():
    guides = np.array([[-1, 1, 2, 3]])
    sites = np.array([[0, 1, 2, 3]])

    with pytest.raises(ValueError, match="between 0 and 4"):
        count_mismatches(guides, sites)


def test_encoding_above_four():
    guides = np.array([[0, 1, 2, 5]])
    sites = np.array([[0, 1, 2, 3]])

    with pytest.raises(ValueError, match="between 0 and 4"):
        count_mismatches(guides, sites)


def test_non_integer_encoding():
    guides = np.array([[0.0, 1.0, 2.0, 3.0]])
    sites = np.array([[0, 1, 2, 3]])

    with pytest.raises(ValueError, match="must contain integers"):
        count_mismatches(guides, sites)

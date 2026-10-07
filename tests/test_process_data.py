import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyphoplacecellanalysis.PhoPositionalData.process_data import bin_edges_to_midpoints

def test_bin_edges_to_midpoints():
    # Normal array with integers
    edges_int = np.array([1, 2, 3, 4])
    expected_int = np.array([1.5, 2.5, 3.5])
    assert_array_equal(bin_edges_to_midpoints(edges_int), expected_int)

    # Normal array with floats
    edges_float = np.array([1.0, 2.5, 4.0])
    expected_float = np.array([1.75, 3.25])
    assert_array_equal(bin_edges_to_midpoints(edges_float), expected_float)

    # Negative numbers
    edges_neg = np.array([-2.0, 2.0])
    expected_neg = np.array([0.0])
    assert_array_equal(bin_edges_to_midpoints(edges_neg), expected_neg)

    # Single element array
    edges_single = np.array([1.0])
    expected_single = np.array([])
    assert_array_equal(bin_edges_to_midpoints(edges_single), expected_single)

    # Empty array
    edges_empty = np.array([])
    expected_empty = np.array([])
    assert_array_equal(bin_edges_to_midpoints(edges_empty), expected_empty)

import pyvista as pv
import numpy as np
from pyphocorehelpers.indexing_helpers import interleave_elements

def interlieve_points(start_points, end_points):
    return interleave_elements(start_points, end_points)

# Plot current segment as spline:
def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


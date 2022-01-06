import pyvista as pv
import numpy as np


def interlieve_points(start_points, end_points):
    assert np.shape(start_points) == np.shape(end_points), f"start_points and end_points must be the same shape. np.shape(start_points): {np.shape(start_points)}, np.shape(end_points): {np.shape(end_points)}"
    all_points_shape = (np.shape(start_points)[0] * 2, np.shape(start_points)[1]) # it's double the length of the start_points
    all_points = np.zeros(all_points_shape)
    all_points[np.arange(0, all_points_shape[0], 2), :] = start_points # fill the even elements
    all_points[np.arange(1, all_points_shape[0], 2), :] = end_points # fill the odd elements
    assert np.shape(all_points)[0] == (np.shape(start_points)[0] * 2), f"newly created all_points is not of corrrect size! np.shape(all_points): {np.shape(all_points)}"
    return all_points

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


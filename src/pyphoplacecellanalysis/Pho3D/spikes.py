import pyvista as pv
import numpy as np

def draw_line_spike(p, x, y, start_z, end_z, **args):
    assert np.shape(x) == np.shape(y), f"x and y must be the same shape. np.shape(x): {np.shape(x)}, np.shape(y): {np.shape(y)}"
    if np.isscalar(start_z):
        # constant value z positions
        start_z = np.full_like(x, start_z)
        
    if np.isscalar(end_z):
        # constant value z positions
        end_z = np.full_like(x, end_z)
        
    assert np.shape(x) == np.shape(start_z), f"x and start_z must be the same shape. np.shape(x): {np.shape(x)}, np.shape(start_z): {np.shape(start_z)}"
    assert np.shape(end_z) == np.shape(start_z), f"end_z and start_z must be the same shape. np.shape(end_z): {np.shape(end_z)}, np.shape(start_z): {np.shape(start_z)}"
    
    points = np.array(zip(x, y, start_z))
    return points

    ## PyVista lines object are composed of [2, start_point_idx, end_point_idx]
    # lines = np.hstack(([2, 0, 1],
    #                [2, 1, 2]))
        
        

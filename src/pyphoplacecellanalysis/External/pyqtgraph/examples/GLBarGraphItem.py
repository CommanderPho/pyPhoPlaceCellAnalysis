"""
This example demonstrates the use of GLBarGraphItem.

"""

import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl


def build_grid_items(data):
    """ builds the pos, size inputs required for gl.GLBarGraphItem(pos, size) from the data. """
    # xbins, ybins
    data_shape = np.shape(data)
    
    # regular grid of starting positions
    pos = np.mgrid[0:data_shape[0], 0:data_shape[1], 0:1].reshape(3,data_shape[0],data_shape[1]).transpose(1,2,0)
    
    # fixed widths, random heights
    size = np.empty((data_shape[0],data_shape[1], 3)) # 3 components for (x, y, z)
    size[...,0:2] = 1.0 # fixed item width components (0.4)
    # size[...,2] = np.random.normal(size=(10,10)) # random z-height
    size[...,2] = data
    return pos, size











app = pg.mkQApp("GLBarGraphItem Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLBarGraphItem')
w.setCameraPosition(distance=40)

gx = gl.GLGridItem()
gx.rotate(90, 0, 1, 0)
gx.translate(-10, 0, 10)
w.addItem(gx)
gy = gl.GLGridItem()
gy.rotate(90, 1, 0, 0)
gy.translate(0, -10, 10)
w.addItem(gy)
gz = gl.GLGridItem()
gz.translate(0, 0, 0)
w.addItem(gz)

test_data = np.abs(np.random.normal(size=(25,25)))
pos, size = build_grid_items(test_data)
bg = gl.GLBarGraphItem(pos, size)
w.addItem(bg)

if __name__ == '__main__':
    pg.exec()

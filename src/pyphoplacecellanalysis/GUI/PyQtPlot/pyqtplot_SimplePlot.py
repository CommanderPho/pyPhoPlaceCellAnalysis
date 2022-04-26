import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np

def plot_simple_graph(y=np.random.normal(size=100)):
    app = pg.mkQApp("Plotting Example")
    #mw = QtGui.QMainWindow()
    #mw.resize(800,800)
    
    win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
    win.resize(1000,600)
    win.setWindowTitle('pyqtgraph example: Plotting')
    
    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    
    p1 = win.addPlot(title="Basic array plotting", y=y)
    
    return [p1], win, app


if __name__ == '__main__':
    # win, app = plot_simple_graph()
    pg.exec()
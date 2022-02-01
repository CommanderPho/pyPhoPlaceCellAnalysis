# -*- coding: utf-8 -*-
"""
This example demonstrates the use of RemoteGraphicsView to improve performance in
applications with heavy load. It works by starting a second process to handle 
all graphics rendering, thus freeing up the main process to do its work.

In this example, the update() function is very expensive and is called frequently.
After update() generates a new set of data, it can either plot directly to a local
plot (bottom) or remotely via a RemoteGraphicsView (top), allowing speed comparison
between the two cases. IF you have a multi-core CPU, it should be obvious that the 
remote case is much faster.
"""


from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.widgets.RemoteGraphicsView
import numpy as np


# pg.setConfigOptions(antialias=True)  ## this will be expensive for the local plot


def build_remote_graphics_view(app):
    view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
    view.pg.setConfigOptions(antialias=True)  ## prettier plots at no cost to the main process! 
    view.setWindowTitle('pyqtgraph example: RemoteSpeedTest')

    app.aboutToQuit.connect(view.close)

    ## Create a PlotItem in the remote process that will be displayed locally
    rplt = view.pg.PlotItem()
    rplt._setProxyOptions(deferGetattr=True)  ## speeds up access to rplt.plot
    view.setCentralItem(rplt)

    # a slow update function:
    def update():
        global rpltfunc
        # update the data (slow):
        data = np.random.normal(size=(10000,50)).sum(axis=1)
        data += 5 * np.sin(np.linspace(0, 10, data.shape[0]))
        
        # Update the remote plot:
        rplt.plot(data, clear=True, _callSync='off')  ## We do not expect a return value.
                                                        ## By turning off callSync, we tell
                                                        ## the proxy that it does not need to 
                                                        ## wait for a reply from the remote
                                                        ## process.


            
            
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)
 
    return view, rplt

 
if __name__ == '__main__':
   pg.exec()

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
import pyphoplacecellanalysis.External.pyqtgraph.console
import numpy as np

from pyphoplacecellanalysis.External.pyqtgraph.dockarea import *


def plot_dockAreaWidget(title='PhoDockAreaWidgetApp'):
    app = pg.mkQApp(title)
    win = QtGui.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1000,500)
    win.setWindowTitle(f'PhoDockAreaWidgetApp: pyqtgraph dockarea: {title}')

    ## Create docks, place them into the window one at a time.
    ## Note that size arguments are only a suggestion; docks will still have to
    ## fill the entire dock area and obey the limits of their internal widgets.
    d1 = Dock("Dock1", size=(1, 1))     ## give this dock the minimum possible size
    d2 = Dock("Dock2 - Console", size=(500,300), closable=True)
    d3 = Dock("Dock3", size=(500,400))
    d4 = Dock("Dock4 (tabbed) - Plot", size=(500,200))
    d5 = Dock("Dock5 - Image", size=(500,200))
    d6 = Dock("Dock6 (tabbed) - Plot", size=(500,200))
    area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
    area.addDock(d2, 'right')     ## place d2 at right edge of dock area
    area.addDock(d3, 'bottom', d1)## place d3 at bottom edge of d1
    area.addDock(d4, 'right')     ## place d4 at right edge of dock area
    area.addDock(d5, 'left', d1)  ## place d5 at left edge of d1
    area.addDock(d6, 'top', d4)   ## place d5 at top edge of d4

    ## Test ability to move docks programatically after they have been placed
    area.moveDock(d4, 'top', d2)     ## move d4 to top edge of d2
    area.moveDock(d6, 'above', d4)   ## move d6 to stack on top of d4
    area.moveDock(d5, 'top', d2)     ## move d5 to top edge of d2


    ## Add widgets into each dock

    ## first dock gets save/restore buttons
    w1 = pg.LayoutWidget()
    label = QtGui.QLabel(""" -- DockArea Example -- 
    This window has 6 Dock widgets in it. Each dock can be dragged
    by its title bar to occupy a different space within the window 
    but note that one dock has its title bar hidden). Additionally,
    the borders between docks may be dragged to resize. Docks that are dragged on top
    of one another are stacked in a tabbed layout. Double-click a dock title
    bar to place it in its own window.
    """)
    saveBtn = QtGui.QPushButton('Save dock state')
    restoreBtn = QtGui.QPushButton('Restore dock state')
    restoreBtn.setEnabled(False)
    w1.addWidget(label, row=0, col=0)
    w1.addWidget(saveBtn, row=1, col=0)
    w1.addWidget(restoreBtn, row=2, col=0)
    d1.addWidget(w1)
    state = None
    def save():
        global state
        state = area.saveState()
        restoreBtn.setEnabled(True)
    def load():
        global state
        area.restoreState(state)
    saveBtn.clicked.connect(save)
    restoreBtn.clicked.connect(load)


    w2 = pg.console.ConsoleWidget()
    d2.addWidget(w2)

    ## Hide title bar on dock 3
    d3.hideTitleBar()
    w3 = pg.PlotWidget(title="Plot inside dock with no title bar")
    w3.plot(np.random.normal(size=100))
    d3.addWidget(w3)

    w4 = pg.PlotWidget(title="Dock 4 plot")
    w4.plot(np.random.normal(size=100))
    d4.addWidget(w4)

    w5 = pg.ImageView()
    w5.setImage(np.random.normal(size=(100,100)))
    d5.addWidget(w5)

    w6 = pg.PlotWidget(title="Dock 6 plot")
    w6.plot(np.random.normal(size=100))
    d6.addWidget(w6)

    win.show()

    return win, app



# app = pg.mkQApp("DockArea Example")
# win = QtGui.QMainWindow()
# area = DockArea()
# win.setCentralWidget(area)
# win.resize(1000,500)
# win.setWindowTitle('pyqtgraph example: dockarea')

## Create docks, place them into the window one at a time.
## Note that size arguments are only a suggestion; docks will still have to
## fill the entire dock area and obey the limits of their internal widgets.
# d1 = Dock("Dock1", size=(1, 1))     ## give this dock the minimum possible size
# d2 = Dock("Dock2 - Console", size=(500,300), closable=True)
# d3 = Dock("Dock3", size=(500,400))
# d4 = Dock("Dock4 (tabbed) - Plot", size=(500,200))
# d5 = Dock("Dock5 - Image", size=(500,200))
# d6 = Dock("Dock6 (tabbed) - Plot", size=(500,200))
# area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
# area.addDock(d2, 'right')     ## place d2 at right edge of dock area
# area.addDock(d3, 'bottom', d1)## place d3 at bottom edge of d1
# area.addDock(d4, 'right')     ## place d4 at right edge of dock area
# area.addDock(d5, 'left', d1)  ## place d5 at left edge of d1
# area.addDock(d6, 'top', d4)   ## place d5 at top edge of d4

# ## Test ability to move docks programatically after they have been placed
# area.moveDock(d4, 'top', d2)     ## move d4 to top edge of d2
# area.moveDock(d6, 'above', d4)   ## move d6 to stack on top of d4
# area.moveDock(d5, 'top', d2)     ## move d5 to top edge of d2




if __name__ == '__main__':
    win, app = plot_dockAreaWidget()
    pg.exec()

"""
Example beeswarm / bar chart
"""

import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg

win = pg.plot()
win.setWindowTitle('pyqtgraph example: beeswarm')

data = np.random.normal(size=(4,20))
data[0] += 5
data[1] += 7
data[2] += 5
data[3] = 10 + data[3] * 2

## Make bar graph
#bar = pg.BarGraphItem(x=range(4), height=data.mean(axis=1), width=0.5, brush=0.4)
#win.addItem(bar)

## add scatter plots on top
for i in range(4):
    xvals = pg.pseudoScatter(data[i], spacing=0.4, bidir=True) * 0.2
    win.plot(x=xvals+i, y=data[i], pen=None, symbol='o', symbolBrush=pg.intColor(i,6,maxValue=128))

## Make error bars
err = pg.ErrorBarItem(x=np.arange(4), y=data.mean(axis=1), height=data.std(axis=1), beam=0.5, pen={'color':'w', 'width':2})
win.addItem(err)


if __name__ == '__main__':
    pg.exec()

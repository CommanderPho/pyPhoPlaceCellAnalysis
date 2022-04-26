import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg

plt = pg.plot(np.random.normal(size=100), title="Simplest possible plotting example")

if __name__ == '__main__':
    pg.exec()

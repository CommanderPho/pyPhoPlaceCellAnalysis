"""
ConsoleWidget is used to allow execution of user-supplied python commands
in an application. It also includes a command history and functionality for trapping
and inspecting stack traces.

"""

import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph.console

app = pg.mkQApp()

## build an initial namespace for console commands to be executed in (this is optional;
## the user can always import these modules manually)
namespace = {'pg': pg, 'np': np}

## initial text to display in the console
text = """
This is an interactive python console. The numpy and pyqtgraph modules have already been imported 
as 'np' and 'pg'. 

Go, play.
"""
c = pg.console.ConsoleWidget(namespace=namespace, text=text)
c.show()
c.setWindowTitle('pyqtgraph example: ConsoleWidget')

if __name__ == '__main__':
    pg.exec()

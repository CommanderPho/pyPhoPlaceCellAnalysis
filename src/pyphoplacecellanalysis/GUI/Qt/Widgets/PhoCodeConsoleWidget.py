# PhoCodeConsoleWidget

from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore
from pyphoplacecellanalysis.External.pyqtgraph.console import ConsoleWidget


class PhoCodeConsoleWidget(ConsoleWidget):
    """
    Widget displaying console output and accepting command input.
    Implements:
        
      - eval python expressions / exec python statements
      - storable history of commands
      - exception handling allowing commands to be interpreted in the context of any level in the exception stack frame
    
    Why not just use python in an interactive shell (or ipython) ? There are a few reasons:
       
      - pyside does not yet allow Qt event processing and interactive shell at the same time
      - on some systems, typing in the console _blocks_ the qt event loop until the user presses enter. This can
        be baffling and frustrating to users since it would appear the program has frozen.
      - some terminals (eg windows cmd.exe) have notoriously unfriendly interfaces
      - ability to add extra features like exception stack introspection
      - ability to have multiple interactive prompts, including for spawned sub-processes
    """
    
    def __init__(self, parent=None, namespace=None, historyFile=None, text=None, editor=None, **kwargs):
        super(PhoCodeConsoleWidget, self).__init__(parent=parent, namespace=namespace, historyFile=historyFile, text=text, editor=editor)

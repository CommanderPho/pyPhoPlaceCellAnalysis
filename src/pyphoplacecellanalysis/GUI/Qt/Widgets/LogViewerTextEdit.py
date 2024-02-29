import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore, QtWidgets
# from PyQt5.QtWidgets import QTextEdit, QApplication
# from PyQt5.QtCore import QTimer

__all__ = ['LogViewer']


class LogViewer(QtWidgets.QTextEdit):
    """ 
    from pyphoplacecellanalysis.GUI.Qt.Widgets.LogViewerTextEdit import LogViewer
    
    """
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        
        # Create a QTimer instance
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setInterval(200)  # Update every 200 milliseconds
        self._update_timer.timeout.connect(self.flush_log)
        self._log_cache = []

    def write_to_log(self, message):
        self._log_cache.append(message)
        if not self._update_timer.isActive():
            self._update_timer.start()
    
    def flush_log(self):
        # Append all cached messages to the QTextEdit
        for message in self._log_cache:
            self.append(message)
        
        # Clear the log cache
        self._log_cache = []
        
        # Automatically scroll to the bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

        # Stop the timer until more data arrives
        self._update_timer.stop()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    viewer = LogViewer()
    viewer.show()
    
    # Simulate logging some messages
    viewer.write_to_log('First log entry.')
    viewer.write_to_log('Second log entry.')

    app.exec_()


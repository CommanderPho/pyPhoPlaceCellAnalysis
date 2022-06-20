import traceback, sys
from qtpy import QtCore, QtGui, QtWidgets


""" 
Usage Example:


    import sys
    import time
    import importlib
    from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
    import pyphoplacecellanalysis.External.pyqtgraph as pg
    from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.TestBackgroundWorker import WorkerSignals, Worker

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, *args, **kwargs):
            super(MainWindow, self).__init__(*args, **kwargs)

            self.counter = 0

            layout = QtWidgets.QVBoxLayout()

            self.l = QtWidgets.QLabel("Start")
            b = QtWidgets.QPushButton("DANGER!")
            b.pressed.connect(self.oh_no)

            layout.addWidget(self.l)
            layout.addWidget(b)

            w = QtWidgets.QWidget()
            w.setLayout(layout)

            self.setCentralWidget(w)

            self.show()

            self.threadpool = QtCore.QThreadPool()
            print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

            self.timer = QtCore.QTimer()
            self.timer.setInterval(1000)
            self.timer.timeout.connect(self.recurring_timer)
            self.timer.start()

        def progress_fn(self, n):
            print("%d%% done" % n)

        def execute_this_fn(self, progress_callback):
            for n in range(0, 5):
                time.sleep(1)
                progress_callback.emit(n*100/4)

            return "Done."

        def print_output(self, s):
            print(s)

        def thread_complete(self):
            print("THREAD COMPLETE!")

        def oh_no(self):
            # Pass the function to execute
            worker = Worker(self.execute_this_fn) # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.print_output)
            worker.signals.finished.connect(self.thread_complete)
            worker.signals.progress.connect(self.progress_fn)

            # Execute
            self.threadpool.start(worker)


        def recurring_timer(self):
            self.counter +=1
            self.l.setText("Counter: %d" % self.counter)
            

    app = pg.mkQApp('PyQt_MultiThreading_Testing')
    # app = QtWidgets.QApplication([])
    window = MainWindow()
    # app.exec_()
    window

"""

# class WorkerThread(QtCore.QThread):
#     """ 
    
#     """
#     update_progress = Signal()
    
    
    
#     def run(self):
#         """ Operates on run """
#         pass
    
# QtCore.pyqtSignal
# QtGui.PYQT_SIGNAL
    


class WorkerSignals(QtCore.QObject):
    """docstring for WorkerSignals."""
    
    finished = QtCore.Signal()
    progress = QtCore.Signal(int)
    error = QtCore.Signal(tuple)
    result = QtCore.Signal(object)
    
    
class Worker(QtCore.QRunnable):
    """ Worker runnable (thread)
    
    """
    def __init__(self, fn, *args, **kwargs) -> None:
        super(Worker, self).__init__()
    
        # Store constructor arguments:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        
        # add the callback to our kwargs:
        self.kwargs['progress_callback'] = self.signals.progress
        
        
    @QtCore.Slot()
    def run(self):
        """ 
        
        """
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result) # Return the result
        finally:
            self.signals.finished.emit() # Done
        
        
        
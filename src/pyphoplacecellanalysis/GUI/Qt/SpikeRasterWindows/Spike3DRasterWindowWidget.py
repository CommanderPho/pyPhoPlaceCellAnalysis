import numpy as np

from qtpy import QtCore, QtWidgets
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowBase import Ui_RootWidget # Generated file from .ui

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster


class Spike3DRasterWindowWidget(QtWidgets.QWidget):
    """ A main raster window loaded from a Qt .ui file. 
    
    Usage:
    
    from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget

    spike_raster_window = Spike3DRasterWindowWidget(curr_spikes_df)
    
    """
    
    enable_window_close_confirmation = False
    # Application/Window Configuration Options:
    applicationName = 'Spike3DRasterWindow'
    windowName = 'Spike3DRasterWindow'
    
    
    # TODO: add signals here:
    
    
    @property
    def spike_raster_plt_2d(self):
        """The spike_raster_plt_2d property."""
        return self.ui.spike_raster_plt_2d
    
    @property
    def spike_raster_plt_3d(self):
        """The spike_raster_plt_2d property."""
        return self.ui.spike_raster_plt_3d
    
    
    
    def __init__(self, curr_spikes_df, core_app_name='UnifiedSpikeRasterApp', window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_RootWidget()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        self.initUI(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order)
        
        # Update the windows once before showing the UI:
        self.spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
        
        self.show() # Show the GUI

    def initUI(self, curr_spikes_df, core_app_name='UnifiedSpikeRasterApp', window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None):
        # 
        
        self.ui.spike_raster_plt_3d = Spike3DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, parent=None)
        # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
        self.ui.spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
        
        # Add the plotter widgets to the UI:
        self.ui.v_layout = QtWidgets.QVBoxLayout()
        self.ui.v_layout.setContentsMargins(0,0,0,0)
        self.ui.v_layout.addWidget(self.ui.spike_raster_plt_3d)
        self.ui.mainSpike3DRasterWidget.setLayout(self.ui.v_layout)
        
        self.ui.v_layout_secondary = QtWidgets.QVBoxLayout()
        self.ui.v_layout_secondary.setContentsMargins(0,0,0,0)
        self.ui.v_layout_secondary.addWidget(self.ui.spike_raster_plt_2d)
        self.ui.secondarySpikeRasterControlWidget.setLayout(self.ui.v_layout_secondary)
        
        self.connect_plotter_time_windows()
        
        # self.spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
        # self.spike_3d_to_2d_window_connection = self.spike_raster_plt_2d.window_scrolled.connect(self.spike_raster_plt_3d.spikes_window.update_window_start_end)
        # self.spike_raster_plt_3d.disable_render_window_controls()
        # spike_raster_plt_3d.setWindowTitle('3D Raster with 2D Control Window')
        # self.spike_raster_plt_3d.setWindowTitle('Main 3D Raster Window')
        
    def connect_plotter_time_windows(self):
         self.spike_3d_to_2d_window_connection = self.spike_raster_plt_2d.window_scrolled.connect(self.spike_raster_plt_3d.spikes_window.update_window_start_end)
         
        
                
    def __str__(self):
         return
     
     
    
    ###################################
    #### EVENT HANDLERS
    ##################################

    
    def closeEvent(self, event):
        """closeEvent(self, event): pyqt default event, doesn't have to be registered. Called when the widget will close.
        """
        if self.enable_window_close_confirmation:
            reply = QtWidgets.QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        else:
            reply = QtWidgets.QMessageBox.Yes
            
        if reply == QtWidgets.QMessageBox.Yes:
            self.onClose() # ensure onClose() is called
            event.accept()
            print('Window closed')
        else:
            event.ignore()
   

     
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    testWidget = Spike3DRasterWindowWidget()
    # testWidget.show()
    sys.exit(app.exec_())


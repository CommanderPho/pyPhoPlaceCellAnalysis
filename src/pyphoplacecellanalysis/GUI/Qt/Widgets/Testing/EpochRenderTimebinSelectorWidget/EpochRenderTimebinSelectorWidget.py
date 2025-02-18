# EpochRenderTimebinSelectorWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\Testing\EpochRenderTimebinSelectorWidget\EpochRenderTimebinSelectorWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
import numpy as np
from attrs import define, field, Factory
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QSlider
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'EpochRenderTimebinSelectorWidget.ui')

class EpochTimebinningIndexingDatasource:
    # def get_num_epochs(self) -> int:
    #     """ returns the number of time_bins for the specified epoch index """
    #     raise NotImplementedError(f'implementor must override')
    #     if self._epoch_index_to_time_bins_provider is None:
    #         n_epochs: int = 5
    #         epochs = np.arange(n_epochs)
    #         return epochs
    #     else:
    #         return self._epoch_index_to_time_bins_provider.get_num_epochs()
        
    def get_epochs(self) -> NDArray:
        """ returns the number of time_bins for the specified epoch index """
        raise NotImplementedError(f'implementor must override')
        # if self._epoch_index_to_time_bins_provider is None:
        #     n_epochs: int = 12
        #     epochs = np.arange(n_epochs)
        #     return epochs
        # else:
        #     return self._epoch_index_to_time_bins_provider.get_num_epochs()
        
    def get_num_epochs(self) -> int:
        """ returns the number of time_bins for the specified epoch index """
        return len(self.get_epochs())

    def get_time_bins_for_epoch_index(self, an_epoch_idx: int) -> NDArray:
        """ returns the number of time_bins for the specified epoch index """
        raise NotImplementedError(f'implementor must override')
        if self._epoch_index_to_time_bins_provider is None:
            n_time_bins: int = 5
            time_bins = np.arange(n_time_bins)
            return time_bins
        else:
            return self._epoch_index_to_time_bins_provider.get_time_bins_for_epoch_index(an_epoch_idx=an_epoch_idx)
        


@define(slots=False, eq=False)
class ConcreteEpochTimebinningIndexingDatasource(EpochTimebinningIndexingDatasource):
    epoch_time_bins_lists: List[List[int]] = field()
    
    @classmethod
    def init_with_random_values(cls, n_random_epochs: int=10, max_bins_per_epoch:int=12):
        """Initialize with random epoch time bin lists."""
        epoch_time_bins_lists = [list(np.arange(np.random.randint(1, max_bins_per_epoch))) for _ in range(n_random_epochs)]
        # epoch_time_bins_lists = [list(np.random.randint(0, 100, size=np.random.randint(1, max_bins_per_epoch))) for _ in range(n_random_epochs)]
        return cls(epoch_time_bins_lists=epoch_time_bins_lists)

    def get_epochs(self) -> NDArray:
        """Returns the list of epoch indices."""
        return np.arange(len(self.epoch_time_bins_lists))

    def get_time_bins_for_epoch_index(self, an_epoch_idx: int) -> NDArray:
        """Returns the time bins for a given epoch index."""
        return np.array(self.epoch_time_bins_lists[an_epoch_idx])


class EpochRenderTimebinSelectorWidget(QWidget):
    """ 
    
    .ui.horizontalSlider_EpochIndex
    .ui.horizontalSlider_TimeBinIndex
    
    .ui.spinBox_EpochIndex
    .ui.spinBox_TimeBinIndex
    
    from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.EpochRenderTimebinSelectorWidget.EpochRenderTimebinSelectorWidget import EpochRenderTimebinSelectorWidget
    
    
    """
    sigEpochIndexChanged = pyqtSignal(int)
    sigTimeBinIndexChanged = pyqtSignal(int)
    
    

    @property
    def epoch_index_to_time_bins_provider(self) -> Optional[EpochTimebinningIndexingDatasource]:
        """The epoch_index_to_time_bins_provider property."""
        return self._epoch_index_to_time_bins_provider


    @property
    def slider_EpochIndex(self) -> QSlider:
        """The slider_EpochIndex property."""
        return self.ui.horizontalSlider_EpochIndex
    
    @property
    def slider_TimeBinIndex(self) -> QSlider:
        """The slider_EpochIndex property."""
        return self.ui.horizontalSlider_TimeBinIndex


    def __init__(self, epoch_ds: Optional[EpochTimebinningIndexingDatasource]=None, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self._epoch_index_to_time_bins_provider = epoch_ds

        self.curr_epoch_indicies = np.arange(12)
        self.curr_epoch_index = 0
        
        self.curr_time_bin_indicies = np.arange(5)
        self.curr_time_bin_index = 0
        
        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        self.ui.horizontalSlider_EpochIndex.valueChanged.connect(self.on_update_slider_epoch_idx)
        self.ui.horizontalSlider_TimeBinIndex.valueChanged.connect(self.on_update_slider_time_bin_idx)
        
        # self.ui.horizontalSlider_TimeBinIndex.set
        # self.curr_time_bin_indicies
        # self.slider_EpochIndex.setTickInterval(1)
        # self.slider_EpochIndex.setRange(self.curr_time_bin_indicies)
        # self.slider_EpochIndex.setValue(0)
        
        self.update_ui()


    # ==================================================================================================================== #
    # EpochTimebinningIndexingDatasource Conformances                                                                      #
    # ==================================================================================================================== #
    def get_epochs(self) -> NDArray:
        """ returns the number of time_bins for the specified epoch index """
        if self._epoch_index_to_time_bins_provider is None:
            n_epochs: int = 12
            epochs = np.arange(n_epochs)
            return epochs
        else:
            return self._epoch_index_to_time_bins_provider.get_epochs()
        
    def get_num_epochs(self) -> int:
        """ returns the number of time_bins for the specified epoch index """
        return len(self.get_epochs())
        

    def get_time_bins_for_epoch_index(self, an_epoch_idx: int) -> NDArray:
        """ returns the number of time_bins for the specified epoch index """
        if self._epoch_index_to_time_bins_provider is None:
            n_time_bins: int = 5
            time_bins = np.arange(n_time_bins)
            return time_bins
        else:
            return self._epoch_index_to_time_bins_provider.get_time_bins_for_epoch_index(an_epoch_idx=an_epoch_idx)
        

    def update_ui(self):
        """ uses
        self.curr_epoch_indicies, self.curr_epoch_index
        self.curr_time_bin_indicies, self.curr_time_bin_index
        """
        self.curr_epoch_indicies = np.arange(self.get_num_epochs())
        self.curr_epoch_index = 0 
        
        self.curr_time_bin_indicies = self.get_time_bins_for_epoch_index(an_epoch_idx=self.curr_epoch_index)
        self.curr_time_bin_index = 0

        self.slider_EpochIndex.blockSignals(True)
        self.slider_TimeBinIndex.blockSignals(True)
        
        self.slider_EpochIndex.setTickInterval(1)
        self.slider_EpochIndex.setRange(self.curr_epoch_indicies[0], self.curr_epoch_indicies[-1])
        self.slider_EpochIndex.setValue(self.curr_epoch_index)
        

        self.slider_TimeBinIndex.setTickInterval(1)
        self.slider_TimeBinIndex.setRange(self.curr_time_bin_indicies[0], self.curr_time_bin_indicies[-1])
        self.slider_TimeBinIndex.setValue(self.curr_time_bin_index)
        
        self.slider_TimeBinIndex.blockSignals(False)        
        self.slider_EpochIndex.blockSignals(False)

    def perform_programmatic_slider_epoch_update(self, value):
        """ called to programmatically update the epoch_idx slider. """
        # if (self.slider_epoch is not None):
        #     print(f'updating slider_epoch index to : {int(value)}')
        #     self.slider_epoch.GetRepresentation().SetValue(int(value)) # set to 0
        #     self.on_update_slider_epoch_idx(value=int(value))
        #     print(f'\tdone.')
        raise NotImplementedError()
    

    def on_update_slider_epoch_idx(self, value: int):
        """ called when the epoch_idx slider changes. 
        """
        print(f'EpochRenderTimebinSelectorWidget.on_update_slider_epoch_idx(value: {value})')
        self.curr_epoch_index = int(value) ## Update `curr_epoch_idx`
        self.curr_time_bin_indicies = self.get_time_bins_for_epoch_index(an_epoch_idx=self.curr_epoch_index)
        self.curr_time_bin_index = 0

        self.slider_TimeBinIndex.blockSignals(True)
        self.slider_TimeBinIndex.setRange(self.curr_time_bin_indicies[0], self.curr_time_bin_indicies[-1])
        self.slider_TimeBinIndex.setValue(self.curr_time_bin_index)
        self.slider_TimeBinIndex.blockSignals(False)

        # self.update_ui()
        
        self.sigEpochIndexChanged.emit(self.curr_time_bin_index)
        
        # if not self.enable_plot_all_time_bins_in_epoch_mode:
        #     self.curr_time_bin_index = 0 # change to 0
        # else:
        #     ## otherwise default to a range
        #     self.curr_time_bin_index = np.arange(self.curr_n_time_bins)

        # self.update_ui() # called to update the dependent time_bin slider

        # if not self.enable_plot_all_time_bins_in_epoch_mode:
        #     self.perform_update_plot_single_epoch_time_bin(self.curr_time_bin_index)
        # else:
        #     ## otherwise default to a range
        #     self.perform_update_plot_epoch_time_bin_range(self.curr_time_bin_index)

        # ## shouldn't be here:
        # # update_plot_fn = self.data_dict.get('plot_3d_binned_bars[55.63197815967686]', {}).get('update_plot_fn', None)
        # update_plot_fn = self.data_dict.get('plot_3d_stem_points_P_x_given_n', {}).get('update_plot_fn', None)
        # if update_plot_fn is not None:
        #     update_plot_fn(self.curr_time_bin_index)


    def on_update_slider_time_bin_idx(self, value: int):
        """ called when the epoch_idx slider changes. 
        """
        print(f'EpochRenderTimebinSelectorWidget.on_update_slider_time_bin_idx(value: {value})')
        # # print(f'.on_update_slider_epoch(value: {value})')
        self.curr_time_bin_index = int(value) ## Update `curr_time_bin_index`
        self.sigTimeBinIndexChanged.emit(self.curr_time_bin_index)
        
        # if not self.enable_plot_all_time_bins_in_epoch_mode:
        #     self.curr_time_bin_index = 0 # change to 0
        # else:
        #     ## otherwise default to a range
        #     self.curr_time_bin_index = np.arange(self.curr_n_time_bins)

        # self.update_ui() # called to update the dependent time_bin slider

        # if not self.enable_plot_all_time_bins_in_epoch_mode:
        #     self.perform_update_plot_single_epoch_time_bin(self.curr_time_bin_index)
        # else:
        #     ## otherwise default to a range
        #     self.perform_update_plot_epoch_time_bin_range(self.curr_time_bin_index)

        # ## shouldn't be here:
        # # update_plot_fn = self.data_dict.get('plot_3d_binned_bars[55.63197815967686]', {}).get('update_plot_fn', None)
        # update_plot_fn = self.data_dict.get('plot_3d_stem_points_P_x_given_n', {}).get('update_plot_fn', None)
        # if update_plot_fn is not None:
        #     update_plot_fn(self.curr_time_bin_index)
            



## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    a_ds: ConcreteEpochTimebinningIndexingDatasource = ConcreteEpochTimebinningIndexingDatasource.init_with_random_values(n_random_epochs=43, max_bins_per_epoch=12)
    widget = EpochRenderTimebinSelectorWidget(epoch_ds=a_ds)
    widget.show()
    sys.exit(app.exec_())

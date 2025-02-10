# PipelineComputationWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\PipelineComputationWidget\PipelineComputationWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
from copy import deepcopy
import sys
import os

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# 
from pyphoplacecellanalysis.GUI.Qt.Mixins.PipelineOwningMixin import PipelineOwningMixin
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'PipelineComputationWidget.ui')

class PipelineComputationWidget(PipelineOwningMixin, QWidget):
    """ 
    
    mainComputationFunctionsContainer
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.Qt.Widgets.PipelineComputationWidget.PipelineComputationWidget import PipelineComputationWidget
    
    """
    # @property
    # def gridLayout_MainContent(self) -> QGridLayout:
    #     """The gridLayout_MainContent property."""
    #     return self.ui.gridLayout_MainContent
    

    def __init__(self, parent=None, owning_pipeline=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        ## Set member properties:
        self._owning_pipeline = owning_pipeline

        self.params = VisualizationParameters(name='PipelineComputationWidget')
        # self.plots_data = RenderPlotsData(name='PipelineComputationWidget')
        # self.plots = RenderPlots(name='PipelineComputationWidget')
        # self.ui = PhoUIContainer(name='PipelineComputationWidget')
        self.ui.connections = PhoUIContainer(name='PipelineComputationWidget')
        

        ## Process curr_active_pipeline
        any_recent, epoch_latest, self.params.epoch_each, (global_latest, self.params.global_comp) = self._owning_pipeline.get_computation_times(debug_print=False)
        self.params.all_validators = self._owning_pipeline.get_merged_computation_function_validators()
        global_only = {k: v for k, v in self.params.all_validators.items() if v.is_global}
        non_global_only = {k: v for k, v in self.params.all_validators.items() if not v.is_global}
        non_global_map = {v.computation_fn_name: v.short_name for k, v in non_global_only.items() if not v.short_name.startswith('_DEP')}
        global_map = {v.computation_fn_name: v.short_name for k, v in global_only.items() if not v.short_name.startswith('_DEP')}
        self.params.epoch_each = {epoch: {non_global_map.get(fn, fn): t for fn, t in comps.items()} for epoch, comps in self.params.epoch_each.items()}
        self.params.global_comp = {global_map.get(fn, fn): t for fn, t in self.params.global_comp.items()}
        

        self.params.filtered_epoch_column_names = deepcopy(list(self._owning_pipeline.filtered_epochs.keys()))
        self.params.num_filtered_epoch_columns = len(self.params.filtered_epoch_column_names)
        

        # recompute_date = datetime.datetime(2024, 4, 1, 0, 0, 0)
        # self.params.epochs_recompute = [epoch for epoch, t in epoch_latest.items() if t < recompute_date]
        # self.params.epoch_each_recompute = {epoch: {comp: t for comp, t in comps.items() if t < recompute_date} for epoch, comps in self.params.epoch_each.items()}
        



        self.initUI()
        self.show() # Show the GUI


    def _initUI_build_local_epoch_results_tree(self):
        """ 
        Uses: self.params.epoch_each, self.params.filtered_epoch_column_names, self.params.num_filtered_epoch_columns
        """
        tree_EpochLocalResults = QTreeWidget()
        tree_EpochLocalResults.setColumnCount(self.params.num_filtered_epoch_columns+1)
        tree_EpochLocalResults.setHeaderLabels(["Epoch/Computation", *self.params.filtered_epoch_column_names])
        for epoch, comps in self.params.epoch_each.items():
            epoch_item = QTreeWidgetItem([str(epoch), ""])
            for comp, t in comps.items():
                epoch_item.addChild(QTreeWidgetItem([str(comp), str(t)]))
            tree_EpochLocalResults.addTopLevelItem(epoch_item)
        tree_EpochLocalResults.expandAll()
        return tree_EpochLocalResults
    

    def initUI(self):
        """ build 
        """
        cw = QWidget()
        layout = QVBoxLayout(cw)


        # layout.addWidget(QLabel(f"Any Most Recent Computation: {self.params.any_recent}"))
        # tbl1 = QTableWidget()
        # tbl1.setColumnCount(2)
        # tbl1.setHorizontalHeaderLabels(["Epoch", "Latest Time"])
        # tbl1.setRowCount(len(self.params.epoch_latest))
        # for i, (epoch, t) in enumerate(self.params.epoch_latest.items()):
        #     tbl1.setItem(i, 0, QTableWidgetItem(str(epoch)))
        # tbl1.setItem(i, 1, QTableWidgetItem(str(t)))
        # layout.addWidget(QLabel("Epoch Latest Computations"))
        # layout.addWidget(tbl1)
        # tbl1.resizeColumnsToContents()
        
        layout.addWidget(QLabel("Epoch Each Result Computations"))
        tree_EpochLocalResults = self._initUI_build_local_epoch_results_tree()
        layout.addWidget(tree_EpochLocalResults)


        tbl_global_computations = QTableWidget()
        tbl_global_computations.setColumnCount(2)
        tbl_global_computations.setHorizontalHeaderLabels(["Global Computation", "Completion Time"])
        tbl_global_computations.setRowCount(len(self.params.global_comp))
        for i, (comp, t) in enumerate(self.params.global_comp.items()):
            tbl_global_computations.setItem(i, 0, QTableWidgetItem(str(comp)))
        tbl_global_computations.setItem(i, 1, QTableWidgetItem(str(t)))
        layout.addWidget(QLabel("Global Computations"))
        layout.addWidget(tbl_global_computations)
        tbl_global_computations.resizeColumnsToContents()
        
        # tbl3 = QTableWidget()
        # tbl3.setColumnCount(2)
        # tbl3.setHorizontalHeaderLabels(["Epoch", "Needs Recompute"])
        # tbl3.setRowCount(len(self.params.epochs_recompute))
        # for i, epoch in enumerate(self.params.epochs_recompute):
        #     tbl3.setItem(i, 0, QTableWidgetItem(str(epoch)))
        # tbl3.setItem(i, 1, QTableWidgetItem("Yes"))
        # layout.addWidget(QLabel("Epochs Needing Recompute"))
        # layout.addWidget(tbl3)
        # tbl3.resizeColumnsToContents()
        # tree2 = QTreeWidget()
        # tree2.setColumnCount(2)
        # tree2.setHeaderLabels(["Epoch/Computation", "Needs Recompute"])
        # for epoch, comps in self.params.epoch_each_recompute.items():
        #     epoch_item = QTreeWidgetItem([str(epoch), ""])
        #     for comp, t in comps.items():
        #         epoch_item.addChild(QTreeWidgetItem([str(comp), "Yes"]))
        #     tree2.addTopLevelItem(epoch_item)
        # tree2.expandAll()
        # layout.addWidget(QLabel("Epoch Each Result Needing Recompute"))
        # layout.addWidget(tree2)

        ## add to the main layout widget:
        # self.gridLayout_MainContent.addWidget(cw, row=1, column=0)
        self.ui.gridLayout_MainContent.addWidget(cw) # row=1, column=0

            

    def updateUi(self):
        # Update UI for children controls:
        # self.ui.contextSelectorWidget.updateUi()
        # if self.owning_pipeline is not None:
        #     self._programmatically_add_display_function_buttons()
        # self.updateButtonsEnabled(False) # disable all buttons to start
        # self.ui.contextSelectorWidget.sigContextChanged.connect(self.on_context_changed)
        pass
    


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = PipelineComputationWidget()
    widget.show()
    sys.exit(app.exec_())

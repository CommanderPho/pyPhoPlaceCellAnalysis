# PipelineComputationWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\PipelineComputationWidget\PipelineComputationWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
from copy import deepcopy
import numpy as np
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
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import TableSizingHelpers

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'PipelineComputationWidget.ui')




class PipelineComputationWidget(PipelineOwningMixin, QWidget):
    """ 
    
    mainComputationFunctionsContainer
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.Qt.Widgets.PipelineComputationWidget.PipelineComputationWidget import PipelineComputationWidget

        win = PipelineComputationWidget(owning_pipeline=curr_active_pipeline)
        win.show()

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
        
        self.params.table_stylesheet = "QHeaderView::section { background-color:  rgb(61, 61, 61); }"
        # self.params.dt_format_string = "%Y-%m-%d %H:%M:%S"
        self.params.dt_format_string = "%m-%d %H:%M"

        ## Process curr_active_pipeline
        any_recent, epoch_latest, self.params.epoch_each, (global_latest, self.params.global_comp) = self._owning_pipeline.get_computation_times(debug_print=False)
        self.params.all_validators = self._owning_pipeline.get_merged_computation_function_validators()
        self.params.global_only = {k: v for k, v in self.params.all_validators.items() if v.is_global}
        self.params.non_global_only = {k: v for k, v in self.params.all_validators.items() if not v.is_global}
        self.params.non_global_map = {v.computation_fn_name: v.short_name for k, v in self.params.non_global_only.items() if not v.short_name.startswith('_DEP')}
        self.params.global_map = {v.computation_fn_name: v.short_name for k, v in self.params.global_only.items() if not v.short_name.startswith('_DEP')}
        self.params.epoch_each = {epoch: {self.params.non_global_map.get(fn, fn): t for fn, t in comps.items()} for epoch, comps in self.params.epoch_each.items()}
        self.params.global_comp = {self.params.global_map.get(fn, fn): t for fn, t in self.params.global_comp.items()}
        

        self.params.filtered_epoch_column_names = deepcopy(list(self._owning_pipeline.filtered_epochs.keys()))
        self.params.num_filtered_epoch_columns = len(self.params.filtered_epoch_column_names)
        

        

        # recompute_date = datetime.datetime(2024, 4, 1, 0, 0, 0)
        # self.params.epochs_recompute = [epoch for epoch, t in epoch_latest.items() if t < recompute_date]
        # self.params.epoch_each_recompute = {epoch: {comp: t for comp, t in comps.items() if t < recompute_date} for epoch, comps in self.params.epoch_each.items()}
        
        # unique_comp_names_set = set([])
        unique_comp_names_list = []
        for an_epoch, a_results_dict in self.params.epoch_each.items():
            for k,v in a_results_dict.items():
                curr_comp_name: str = self.params.non_global_map.get(k, k)
                # unique_comp_names_set.add(curr_comp_name)
                if curr_comp_name not in unique_comp_names_list:
                    unique_comp_names_list.append(curr_comp_name) ## preserving order
                
        self.params.required_unique_comp_names_list = deepcopy(unique_comp_names_list)
        # self.params.required_num_computation_rows = np.max([len(a_results_dict) for an_epoch, a_results_dict in self.params.epoch_each.items()])
        self.params.required_num_computation_rows = len(self.params.required_unique_comp_names_list)

        self.initUI()
        self.show() # Show the GUI


    # def _initUI_build_local_epoch_results_tree(self):
    #     """ 
    #     Uses: self.params.epoch_each, self.params.filtered_epoch_column_names, self.params.num_filtered_epoch_columns
    #     """
    #     tree_EpochLocalResults = QTreeWidget()
    #     tree_EpochLocalResults.setColumnCount(self.params.num_filtered_epoch_columns+1)
    #     tree_EpochLocalResults.setHeaderLabels(["Epoch/Computation", *self.params.filtered_epoch_column_names])
    #     for epoch, comps in self.params.epoch_each.items():
    #         epoch_item = QTreeWidgetItem([str(epoch), ""])
    #         for comp, t in comps.items():
    #             epoch_item.addChild(QTreeWidgetItem([str(comp), str(t)]))
    #         tree_EpochLocalResults.addTopLevelItem(epoch_item)
    #     tree_EpochLocalResults.expandAll()
    #     return tree_EpochLocalResults
    

    def _initUI_build_local_epoch_results_table(self):
        """ 
        Uses: self.params.epoch_each, self.params.filtered_epoch_column_names, self.params.num_filtered_epoch_columns, self.params.required_num_computation_rows
        """
        tbl_EpochLocalResults = QTableWidget()
        tbl_EpochLocalResults.setStyleSheet(self.params.table_stylesheet)
        ## row-header style:
        tbl_EpochLocalResults.setColumnCount(self.params.num_filtered_epoch_columns)
        tbl_EpochLocalResults.setHorizontalHeaderLabels(self.params.filtered_epoch_column_names)
        # tbl_EpochLocalResults.setColumnCount(self.params.num_filtered_epoch_columns+1)
        # tbl_EpochLocalResults.setHorizontalHeaderLabels(["Epoch/Computation", *self.params.filtered_epoch_column_names])
        tbl_EpochLocalResults.setRowCount(self.params.required_num_computation_rows)
        # for i, (comp, t) in enumerate(global_comp.items()):
        
        for comp_name_row, comp_name in enumerate(self.params.required_unique_comp_names_list):
            # model.setHeaderData(row, Qt.Vertical, label) 
            tbl_EpochLocalResults.setVerticalHeaderItem(comp_name_row, QTableWidgetItem(str(comp_name)))  # Vertical header (row headers)
            # tbl_EpochLocalResults.setItem(comp_name_row, 0, QTableWidgetItem(str(comp_name))) ## set the comp_name column
            ## for each epoch, set the time
            for epoch_col_idx, (epoch_name, comps_t_dict) in enumerate(self.params.epoch_each.items()):            
                ## find this particular computation's datetime
                # active_col_idx: int = (epoch_col_idx+1)
                active_col_idx: int = epoch_col_idx # header-style
                curr_comp_dt = comps_t_dict.get(comp_name, None)
                curr_comp_formatted_dt: str = curr_comp_dt.strftime(self.params.dt_format_string)
                tbl_EpochLocalResults.setItem(comp_name_row, active_col_idx, QTableWidgetItem(curr_comp_formatted_dt)) # the (+1) holds space for the computation name

        tbl_EpochLocalResults.resizeColumnsToContents()
        # tbl_EpochLocalResults.verticalHeader().setVisible(False)
        total_required_table_height: int = TableSizingHelpers.determine_required_table_height(tbl_EpochLocalResults)
        tbl_EpochLocalResults.setMinimumHeight(total_required_table_height)  # Set the required height
        tbl_EpochLocalResults.setMaximumHeight(total_required_table_height)  # Prevent scrolling
        
        return tbl_EpochLocalResults
    
    

    def initUI(self):
        """ build 
        """
        self.ui.mainContentWidget = QWidget()
        self.ui.mainContentVBoxLayout = QVBoxLayout(self.ui.mainContentWidget)


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
        
        self.ui.mainContentVBoxLayout.addWidget(QLabel("Epoch Each Result Computations"))
        # self.ui.tree_EpochLocalResults = self._initUI_build_local_epoch_results_tree()
        # layout.addWidget(self.ui.tree_EpochLocalResults)
        
        self.ui.tbl_EpochLocalResults = self._initUI_build_local_epoch_results_table()
        self.ui.mainContentVBoxLayout.addWidget(self.ui.tbl_EpochLocalResults)


        ## Build Global Computation Progress Widget:
        self.ui.mainContentVBoxLayout.addWidget(QLabel("Global Computations"))
        self.ui.tbl_global_computations = QTableWidget()
        self.ui.tbl_global_computations.setColumnCount(1)
        self.ui.tbl_global_computations.setHorizontalHeaderLabels(["Completion Time"])
        # self.ui.tbl_global_computations.setColumnCount(2)
        # self.ui.tbl_global_computations.setHorizontalHeaderLabels(["Global Computation", "Completion Time"])
        self.ui.tbl_global_computations.setRowCount(len(self.params.global_comp))
        for i, (comp_name, dt) in enumerate(self.params.global_comp.items()):
            curr_comp_formatted_dt: str = dt.strftime(self.params.dt_format_string)
            # self.ui.tbl_global_computations.setItem(i, 0, QTableWidgetItem(str(comp_name)))
            # self.ui.tbl_global_computations.setItem(i, 1, QTableWidgetItem(curr_comp_formatted_dt))
            self.ui.tbl_global_computations.setVerticalHeaderItem(i, QTableWidgetItem(str(comp_name)))  # Vertical header (row headers)
            self.ui.tbl_global_computations.setItem(i, 0, QTableWidgetItem(curr_comp_formatted_dt))
            
        
        self.ui.mainContentVBoxLayout.addWidget(self.ui.tbl_global_computations)
        self.ui.tbl_global_computations.resizeColumnsToContents()
        self.ui.tbl_global_computations.setStyleSheet(self.params.table_stylesheet)
        # self.ui.tbl_global_computations.verticalHeader().setVisible(False)
        total_required_table_height: int = TableSizingHelpers.determine_required_table_height(self.ui.tbl_global_computations)
        self.ui.tbl_global_computations.setMinimumHeight(total_required_table_height)  # Set the required height
        self.ui.tbl_global_computations.setMaximumHeight(total_required_table_height)  # Prevent scrolling
        
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
        self.ui.gridLayout_MainContent.addWidget(self.ui.mainContentWidget) # row=1, column=0

            

    def updateUi(self):
        # Update UI for children controls:
        # self.ui.contextSelectorWidget.updateUi()
        # if self.owning_pipeline is not None:
        #     self._programmatically_add_display_function_buttons()
        # self.updateButtonsEnabled(False) # disable all buttons to start
        # self.ui.contextSelectorWidget.sigContextChanged.connect(self.on_context_changed)
        

        ## Update the local epoch results tree:
        

        pass
    


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = PipelineComputationWidget()
    widget.show()
    sys.exit(app.exec_())

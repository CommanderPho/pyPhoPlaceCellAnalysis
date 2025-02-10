# PipelineComputationWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\PipelineComputationWidget\PipelineComputationWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
from copy import deepcopy
from typing import Optional
import numpy as np
import sys
import os

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QMenu, QAction, QTreeWidget, QTreeWidgetItem
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


class TableContextMenuProviderDelegate:
    """ implementors provide context menus for clicked table cells
    
    """
    def get_context_menu(self, target_table, row_index: Optional[int]=None, column_index: Optional[int]=None, is_row_header: bool=False, is_column_header: bool=False) -> QMenu:
        raise NotImplementedError(f'Implementors must override and provide this function!')
    



class CustomTableWidget(QTableWidget):
    """ provides specific context menus based on whether the user right-clicked in the row/col headers, a cell, or elsewhere 
    """
    def __init__(self, rows: Optional[int]=None, columns: Optional[int]=None, context_menu_delegate: Optional[TableContextMenuProviderDelegate]=None, parent: Optional[QWidget] = None):
        if (rows is not None) or (columns is not None):
            super().__init__(rows, columns, parent=parent)
        else:
            super().__init__(parent=parent)
            
        self._debug_print = False
        self._context_menu_delegate = context_menu_delegate

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_cell_menu)

        self.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self.show_column_header_menu)
        
        self.verticalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.verticalHeader().customContextMenuRequested.connect(self.show_row_header_menu)
        
        

    def show_cell_menu(self, pos):
        # Get the cell that was clicked
        index = self.indexAt(pos)
        if not index.isValid():
            return  # Ignore clicks outside valid cells

        row, col = index.row(), index.column()
        # print(f'show_cell_menu(pos: {pos}):\n\trow: {row}, col: {col}')
        
        menu: Optional[QMenu] = None
        if self._context_menu_delegate is not None:
            # print(f'\t has context_menu_delegate!')
            menu = self._context_menu_delegate.get_context_menu(target_table=self, row_index=row, column_index=col, is_row_header=False, is_column_header=False)
        else:
            print(f'CustomTableWidget: has no context menu delegate!')
            pass

        if menu is None: 
            ## still has no menu, make default:
            menu = QMenu(self)
            action_edit = QAction(f"Edit Cell ({row}, {col})", self)
            action_delete = QAction(f"Delete Cell ({row}, {col})", self)
            action_info = QAction(f"Cell Info ({row}, {col})", self)

            action_edit.triggered.connect(lambda: self.editItem(self.item(row, col)))
            action_delete.triggered.connect(lambda: self.setItem(row, col, None))
            action_info.triggered.connect(lambda: print(f"Cell ({row}, {col}): {self.item(row, col).text() if self.item(row, col) else 'Empty'}"))

            menu.addAction(action_edit)
            menu.addAction(action_delete)
            menu.addAction(action_info)

        ## cell `menu.exec_(...)` to display the menu:
        menu.exec_(self.viewport().mapToGlobal(pos))
        


    def show_row_header_menu(self, pos):
        # Convert local position to global position
        header = self.verticalHeader()
        global_pos = header.mapToGlobal(pos)

        # Determine which column was clicked
        row = header.logicalIndexAt(pos.y())

        if row >= 0:  # Ensure it's a valid column
            if self._debug_print:
                print(f'show_row_header_menu(pos: {pos}):\n\trow: {row}')
            menu: Optional[QMenu] = None
            if self._context_menu_delegate is not None:
                if self._debug_print:
                    print(f'\t has context_menu_delegate!')
                menu = self._context_menu_delegate.get_context_menu(target_table=self, row_index=row, column_index=None, is_row_header=True, is_column_header=False)
            else:
                print(f'CustomTableWidget: has no context menu delegate!')
                pass
            if menu is None: 
                ## still has no menu, make default:
                menu = QMenu(self)

                action_sort = QAction(f"Sort Row {row}", self)
                action_hide = QAction(f"Hide Row {row}", self)
                action_resize = QAction(f"Resize Row {row}", self)

                action_sort.triggered.connect(lambda: print(f"Sorting row {row}"))
                action_hide.triggered.connect(lambda: self.setRowHidden(row, True))
                action_resize.triggered.connect(lambda: header.resizeSection(row, 100))

                menu.addAction(action_sort)
                menu.addAction(action_hide)
                menu.addAction(action_resize)

            ## cell `menu.exec_(...)` to display the menu:
            menu.exec_(global_pos)
            

    def show_column_header_menu(self, pos):
        # Convert local position to global position
        header = self.horizontalHeader()
        global_pos = header.mapToGlobal(pos)

        # Determine which column was clicked
        col = header.logicalIndexAt(pos.x())

        if col >= 0:  # Ensure it's a valid column
            if self._debug_print:
                print(f'show_column_header_menu(pos: {pos}):\n\tcol: {col}')
            menu: Optional[QMenu] = None
            if self._context_menu_delegate is not None:
                if self._debug_print:
                    print(f'\t has context_menu_delegate!')
                menu = self._context_menu_delegate.get_context_menu(target_table=self, row_index=None, column_index=col, is_row_header=False, is_column_header=True)
            else:
                print(f'CustomTableWidget: has no context menu delegate!')
                pass 
               
            if menu is None: 
                ## still has no menu, make default:
                menu = QMenu(self)
                action_sort = QAction(f"Sort Column {col}", self)
                action_hide = QAction(f"Hide Column {col}", self)
                action_resize = QAction(f"Resize Column {col}", self)

                action_sort.triggered.connect(lambda: print(f"Sorting column {col}"))
                action_hide.triggered.connect(lambda: self.setColumnHidden(col, True))
                action_resize.triggered.connect(lambda: header.resizeSection(col, 100))

                menu.addAction(action_sort)
                menu.addAction(action_hide)
                menu.addAction(action_resize)

            menu.exec_(global_pos)
            



class PipelineComputationWidget(TableContextMenuProviderDelegate, PipelineOwningMixin, QWidget):
    """ A widget that contains two tables, the first displaying the computation completion times for the local computations, and the second for the global computations
    
    
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

        self.params = VisualizationParameters(name='PipelineComputationWidget', debug_print=False)
        # self.plots_data = RenderPlotsData(name='PipelineComputationWidget')
        # self.plots = RenderPlots(name='PipelineComputationWidget')
        # self.ui = PhoUIContainer(name='PipelineComputationWidget')
        self.ui.connections = PhoUIContainer(name='PipelineComputationWidget')
        
        self.params.table_stylesheet = "QHeaderView::section { background-color:  rgb(61, 61, 61); }"
        # self.params.dt_format_string = "%Y-%m-%d %H:%M:%S"
        self.params.dt_format_string = "%m-%d %H:%M"

        # ## Process curr_active_pipeline
        self.rebuild_data_from_pipeline()
        
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
        """ Updates the local computations table, creating it if needed
        
        Uses: self.params.epoch_each, self.params.filtered_epoch_column_names, self.params.num_filtered_epoch_columns, self.params.required_num_computation_rows
        """
        was_table_created: bool = False
        if (not hasattr(self.ui, 'tbl_EpochLocalResults')) or (self.ui.tbl_EpochLocalResults is None):        
            tbl_EpochLocalResults = CustomTableWidget(context_menu_delegate=self)
            tbl_EpochLocalResults.setStyleSheet(self.params.table_stylesheet)
            was_table_created = True
        else:
            ## has valid extant table
            tbl_EpochLocalResults = self.ui.tbl_EpochLocalResults
            

        
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
        

        if was_table_created:
            ## new table, created, must assign it to self.ui:
            self.ui.tbl_EpochLocalResults = tbl_EpochLocalResults
            

        return tbl_EpochLocalResults
    
    
    def _initUI_build_global_computations_results_table(self):
        """ Updates the global computations table, creating it if needed
        
        Uses: self.params.epoch_each, self.params.filtered_epoch_column_names, self.params.num_filtered_epoch_columns, self.params.required_num_computation_rows
        """
        was_table_created: bool = False
        if (not hasattr(self.ui, 'tbl_global_computations')) or (self.ui.tbl_global_computations is None):        
            tbl_global_computations = CustomTableWidget(context_menu_delegate=self)
            tbl_global_computations.setStyleSheet(self.params.table_stylesheet)
            was_table_created = True
        else:
            ## has valid extant table
            tbl_global_computations = self.ui.tbl_global_computations
            

        ## row-header style:
        tbl_global_computations.setColumnCount(1)
        tbl_global_computations.setHorizontalHeaderLabels(["Completion Time"])
        # tbl_global_computations.setColumnCount(2)
        # tbl_global_computations.setHorizontalHeaderLabels(["Global Computation", "Completion Time"])
        tbl_global_computations.setRowCount(len(self.params.global_comp))
        for i, (comp_name, dt) in enumerate(self.params.global_comp.items()):
            curr_comp_formatted_dt: str = dt.strftime(self.params.dt_format_string)
            # tbl_global_computations.setItem(i, 0, QTableWidgetItem(str(comp_name)))
            # tbl_global_computations.setItem(i, 1, QTableWidgetItem(curr_comp_formatted_dt))
            tbl_global_computations.setVerticalHeaderItem(i, QTableWidgetItem(str(comp_name)))  # Vertical header (row headers)
            tbl_global_computations.setItem(i, 0, QTableWidgetItem(curr_comp_formatted_dt))
            
        
        self.ui.mainContentVBoxLayout.addWidget(tbl_global_computations)
        tbl_global_computations.resizeColumnsToContents()
        tbl_global_computations.setStyleSheet(self.params.table_stylesheet)
        # tbl_global_computations.verticalHeader().setVisible(False)
        total_required_table_height: int = TableSizingHelpers.determine_required_table_height(tbl_global_computations)
        tbl_global_computations.setMinimumHeight(total_required_table_height)  # Set the required height
        tbl_global_computations.setMaximumHeight(total_required_table_height)  # Prevent scrolling

        if was_table_created:
            ## new table, created, must assign it to self.ui:
            self.ui.tbl_global_computations = tbl_global_computations
            

        return tbl_global_computations
    

    def initUI(self):
        """ build 
        """
        self.ui.mainContentWidget = QWidget()
        self.ui.mainContentVBoxLayout = QVBoxLayout(self.ui.mainContentWidget)


        # layout.addWidget(QLabel(f"Any Most Recent Computation: {self.params.any_recent}"))
        # tbl1 = CustomTableWidget()
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
        
        ## Build Local Epoch Computation Progress Table Widget:
        self.ui.tbl_EpochLocalResults = self._initUI_build_local_epoch_results_table()
        self.ui.mainContentVBoxLayout.addWidget(self.ui.tbl_EpochLocalResults)


        ## Build Global Computation Progress Table Widget:
        self.ui.mainContentVBoxLayout.addWidget(QLabel("Global Computations"))
        self.ui.tbl_global_computations = self._initUI_build_global_computations_results_table()
        self.ui.mainContentVBoxLayout.addWidget(self.ui.tbl_global_computations)
        

        # tbl3 = CustomTableWidget()
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


    def rebuild_data_from_pipeline(self):
        """ uses `self._owning_pipeline` to update all self.params.* variables used in creating/updating tables
        """
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

        ## global computations:
        self.params.required_unique_global_comp_names_list = list(self.params.global_comp.keys())
        self.params.required_num_global_computation_rows = len(self.params.required_unique_global_comp_names_list)
        
            

    def updateUi(self):
        # Update UI for children controls:
        # self.ui.contextSelectorWidget.updateUi()
        # if self.owning_pipeline is not None:
        #     self._programmatically_add_display_function_buttons()
        # self.updateButtonsEnabled(False) # disable all buttons to start
        # self.ui.contextSelectorWidget.sigContextChanged.connect(self.on_context_changed)
        self.rebuild_data_from_pipeline()

        ## Update the local epoch results tree:
        self.ui.tbl_EpochLocalResults = self._initUI_build_local_epoch_results_table()
        self.ui.tbl_global_computations = self._initUI_build_global_computations_results_table()

        pass
    

    # ==================================================================================================================== #
    # Computation/Pipeline Action Functions                                                                                #
    # ==================================================================================================================== #
    
    def _perform_run_compute_function(self, curr_compute_fcn, debug_print=False):
        # custom_args = {} # TODO
        # custom_args = self.active_figure_format_config or {}
        # if debug_print:
        #     print(f'custom_args: {custom_args}')
            

        curr_active_pipeline = self._owning_pipeline
        curr_active_pipeline.reload_default_computation_functions()
        # newly_computed_values = curr_active_pipeline.batch_extended_computations(include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
        #                                                     force_recompute=force_recompute_global, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        # needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = curr_active_pipeline.batch_evaluate_required_computations(include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
        #                                                     force_recompute=force_recompute_global, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        # print(f'Post-load global computations: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')


        # custom_kwargs = {'num_shuffles': 5, 'skip_laps': False, 'minimum_inclusion_fr_Hz':2.0, 'included_qclu_values':[1,2,4,5,6,7]}
        # computation_kwargs_list = [custom_kwargs]
        
        computation_kwargs_list = None
        compute_outputs = curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=[curr_compute_fcn], computation_kwargs_list=computation_kwargs_list, 
                                                        enabled_filter_names=None, fail_on_exception=True, debug_print=False)



        # 'optional_kwargs'
        # print(f'_perform_run_display_function(curr_display_fcn: {curr_display_fcn}): context: {self.ui.contextSelectorWidget.current_selected_context}')
        # compute_outputs = self.owning_pipeline.display(curr_compute_fcn, self.ui.contextSelectorWidget.current_selected_context, **custom_args)
        return compute_outputs
    

    # ==================================================================================================================== #
    # TableContextMenuProviderDelegate Implementations                                                                     #
    # ==================================================================================================================== #
    def get_context_menu(self, target_table, row_index: Optional[int]=None, column_index: Optional[int]=None, is_row_header: bool=False, is_column_header: bool=False) -> QMenu:
        is_global_comp_table: bool = False
        if target_table == self.ui.tbl_global_computations:
            is_global_comp_table = True
        else:
            assert target_table == self.ui.tbl_EpochLocalResults, f"target_table equals neither known table! target_table: {target_table}"
            ## local table
            
        if self.params.debug_print:
            print(f'.get_context_menu(is_global_comp_table: {is_global_comp_table}, row_index: {row_index}, column_index: {column_index}, is_row_header: {is_row_header}, is_column_header: {is_column_header})')
        
        menu = QMenu(target_table)
        

        if is_global_comp_table:
            ## Global Computations Table
            # Disable column header
            if is_column_header:
                return menu ## disallow by returning an empty menu
            if not is_row_header:
                return menu ## disallow by returning an empty menu
            ## only respond to row-header clicks            
            ## get the name of the global computation to recompute
            curr_global_comp_fn_name: str = self.params.required_unique_global_comp_names_list[row_index]
            action_recompute = QAction(f"Recompute `{curr_global_comp_fn_name}`", target_table)
            # action_recompute.triggered.connect(lambda: print(f"Recomputing Global Computation `{curr_global_comp_fn_name}` - {row_index}"); self._perform_run_compute_function(curr_compute_fcn=curr_global_comp_fn_name);)
            action_recompute.triggered.connect(lambda: self._perform_run_compute_function(curr_compute_fcn=curr_global_comp_fn_name))
            menu.addAction(action_recompute)

        else:
            ## Local Computations Table
            if is_column_header:
                ## column header -- trigger epoch all comps recomputations
                assert column_index is not None
                curr_epoch_name: str = self.params.filtered_epoch_column_names[column_index]
                action_recompute = QAction(f"Recompute ALL computations for epoch '{curr_epoch_name}'.", target_table)
                action_recompute.triggered.connect(lambda: print(f"Recomputing ALL Local Computations - for specific epoch '{curr_epoch_name}'"))
                menu.addAction(action_recompute)
                return menu
                            
            if is_row_header:
                ## row header -- triger recomputation of a specific function for all epochs
                assert row_index is not None
                curr_comp_fn_name: str = self.params.required_unique_comp_names_list[row_index]
                action_recompute = QAction(f"Recompute `{curr_comp_fn_name}` for ALL epochs.", target_table)
                action_recompute.triggered.connect(lambda: print(f"Recomputing Local Computation `{curr_comp_fn_name}` - for ALL epochs"))
                menu.addAction(action_recompute)
                return menu

            if (not is_column_header) and (not is_row_header):
                ## regular cell -- triger recomputation for a specific computation function for a specific epoch
                curr_epoch_name: str = self.params.filtered_epoch_column_names[column_index]
                curr_comp_fn_name: str = self.params.required_unique_comp_names_list[row_index]
                action_recompute = QAction(f"Recompute `{curr_comp_fn_name}` for epoch '{curr_epoch_name}'.", target_table)
                action_recompute.triggered.connect(lambda: print(f"Recomputing Local Computation `{curr_comp_fn_name}` - for specific epoch '{curr_epoch_name}'"))
                menu.addAction(action_recompute)
                return menu

        return menu 
        # raise NotImplementedError(f'Implementors must override and provide this function!')
    


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = PipelineComputationWidget()
    widget.show()
    sys.exit(app.exec_())

from copy import deepcopy
import sys, datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem

class PipelineComputationStatusStandalone(QMainWindow):
    """
    
    
    from pyphoplacecellanalysis.GUI.Qt.Widgets.PipelineComputationWidget.PipelineComputationWindowStandalone import PipelineComputationStatusStandalone
    
    
    """
    def __init__(self, curr_active_pipeline):
        super().__init__()
        self.setWindowTitle("Computation Status")
        
        ## Process curr_active_pipeline
        any_recent, epoch_latest, epoch_each, (global_latest, global_comp) = curr_active_pipeline.get_computation_times(debug_print=False)
        all_validators = curr_active_pipeline.get_merged_computation_function_validators()
        global_only = {k: v for k, v in all_validators.items() if v.is_global}
        non_global_only = {k: v for k, v in all_validators.items() if not v.is_global}
        non_global_map = {v.computation_fn_name: v.short_name for k, v in non_global_only.items() if not v.short_name.startswith('_DEP')}
        global_map = {v.computation_fn_name: v.short_name for k, v in global_only.items() if not v.short_name.startswith('_DEP')}
        epoch_each = {epoch: {non_global_map.get(fn, fn): t for fn, t in comps.items()} for epoch, comps in epoch_each.items()}
        global_comp = {global_map.get(fn, fn): t for fn, t in global_comp.items()}
        recompute_date = datetime.datetime(2024, 4, 1, 0, 0, 0)
        epochs_recompute = [epoch for epoch, t in epoch_latest.items() if t < recompute_date]
        epoch_each_recompute = {epoch: {comp: t for comp, t in comps.items() if t < recompute_date} for epoch, comps in epoch_each.items()}
        


        cw = QWidget()
        layout = QVBoxLayout(cw)
        layout.addWidget(QLabel(f"Any Most Recent Computation: {any_recent}"))
        tbl1 = QTableWidget()
        tbl1.setColumnCount(2)
        tbl1.setHorizontalHeaderLabels(["Epoch", "Latest Time"])
        tbl1.setRowCount(len(epoch_latest))
        for i, (epoch, t) in enumerate(epoch_latest.items()):
            tbl1.setItem(i, 0, QTableWidgetItem(str(epoch)))
        tbl1.setItem(i, 1, QTableWidgetItem(str(t)))
        layout.addWidget(QLabel("Epoch Latest Computations"))
        layout.addWidget(tbl1)
        tbl1.resizeColumnsToContents()
        

        
        layout.addWidget(QLabel("Epoch Each Result Computations"))
        tree_EpochLocalResults = self._initUI_build_local_epoch_results_tree(curr_active_pipeline)
        layout.addWidget(tree_EpochLocalResults)
        


        tbl2 = QTableWidget()
        tbl2.setColumnCount(2)
        tbl2.setHorizontalHeaderLabels(["Global Computation", "Completion Time"])
        tbl2.setRowCount(len(global_comp))
        for i, (comp, t) in enumerate(global_comp.items()):
            tbl2.setItem(i, 0, QTableWidgetItem(str(comp)))
        tbl2.setItem(i, 1, QTableWidgetItem(str(t)))
        layout.addWidget(QLabel("Global Computations"))
        layout.addWidget(tbl2)
        tbl2.resizeColumnsToContents()
        tbl3 = QTableWidget()
        tbl3.setColumnCount(2)
        tbl3.setHorizontalHeaderLabels(["Epoch", "Needs Recompute"])
        tbl3.setRowCount(len(epochs_recompute))
        for i, epoch in enumerate(epochs_recompute):
            tbl3.setItem(i, 0, QTableWidgetItem(str(epoch)))
        tbl3.setItem(i, 1, QTableWidgetItem("Yes"))
        layout.addWidget(QLabel("Epochs Needing Recompute"))
        layout.addWidget(tbl3)
        tbl3.resizeColumnsToContents()
        tree2 = QTreeWidget()
        tree2.setColumnCount(2)
        tree2.setHeaderLabels(["Epoch/Computation", "Needs Recompute"])
        for epoch, comps in epoch_each_recompute.items():
            epoch_item = QTreeWidgetItem([str(epoch), ""])
            for comp, t in comps.items():
                epoch_item.addChild(QTreeWidgetItem([str(comp), "Yes"]))
            tree2.addTopLevelItem(epoch_item)
        tree2.expandAll()
        layout.addWidget(QLabel("Epoch Each Result Needing Recompute"))
        layout.addWidget(tree2)
        

        self.setCentralWidget(cw)
        

    def _initUI_build_local_epoch_results_tree(self, curr_active_pipeline):
        ## Use `curr_active_pipeline.filtered_epochs`
        filtered_epoch_column_names = deepcopy(list(curr_active_pipeline.filtered_epochs.keys()))
        num_filtered_epoch_columns: int = len(filtered_epoch_column_names)

        
        tree_EpochLocalResults = QTreeWidget()
        tree_EpochLocalResults.setColumnCount(num_filtered_epoch_columns+1)
        tree_EpochLocalResults.setHeaderLabels(["Epoch/Computation", *filtered_epoch_column_names])
        for epoch, comps in epoch_each.items():
            epoch_item = QTreeWidgetItem([str(epoch), ""])
            for comp, t in comps.items():
                epoch_item.addChild(QTreeWidgetItem([str(comp), str(t)]))
            tree_EpochLocalResults.addTopLevelItem(epoch_item)
        tree_EpochLocalResults.expandAll()
        return tree_EpochLocalResults
    



class DummyValidator:
    def __init__(self, is_global, short_name, computation_fn_name): 
        self.is_global = is_global
        self.short_name = short_name
        self.computation_fn_name = computation_fn_name
class DummyPipeline:
    def get_computation_times(self, debug_print=False):
        any_recent = datetime.datetime(2024, 5, 1, 12, 0, 0)
        epoch_latest = {"epoch1": datetime.datetime(2024, 3, 30, 10, 0, 0), "epoch2": datetime.datetime(2024, 4, 2, 15, 30, 0)}
        epoch_each = {"epoch1": {"_compA": datetime.datetime(2024, 3, 29, 9, 0, 0), "_compB": datetime.datetime(2024, 3, 29, 10, 0, 0)}, "epoch2": {"_compA": datetime.datetime(2024, 4, 2, 15, 0, 0), "_compB": datetime.datetime(2024, 4, 2, 15, 30, 0)}}
        global_latest = datetime.datetime(2024, 4, 2, 16, 0, 0)
        global_comp = {"_globalCompA": datetime.datetime(2024, 4, 2, 16, 0, 0), "_globalCompB": datetime.datetime(2024, 4, 2, 16, 5, 0)}
        return any_recent, epoch_latest, epoch_each, (global_latest, global_comp)
    def get_merged_computation_function_validators(self):
        return {"_compA": DummyValidator(False, "compA", "_compA"), "_compB": DummyValidator(False, "compB", "_compB"), "_globalCompA": DummyValidator(True, "globalCompA", "_globalCompA"), "_globalCompB": DummyValidator(True, "globalCompB", "_globalCompB")}

if __name__=="__main__":
    curr_active_pipeline = DummyPipeline()
    app = QApplication(sys.argv)
        
    win = PipelineComputationStatusStandalone(curr_active_pipeline)
    win.show()
    sys.exit(app.exec_())
    
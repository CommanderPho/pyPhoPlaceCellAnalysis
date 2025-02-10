import sys, datetime; from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem

class MainWindow(QMainWindow):
    def __init__(self, any_recent, epoch_latest, epoch_each, global_latest, global_comp, epoch_recompute, epoch_each_recompute):
        super().__init__()
        self.setWindowTitle("Computation Status")
        cw = QWidget(); lay = QVBoxLayout(cw)
        lay.addWidget(QLabel(f"Any Most Recent Computation: {any_recent}"))
        tbl1 = QTableWidget(); tbl1.setColumnCount(2); tbl1.setHorizontalHeaderLabels(["Epoch", "Latest Time"]); tbl1.setRowCount(len(epoch_latest))
        for i, (e, t) in enumerate(epoch_latest.items()): tbl1.setItem(i, 0, QTableWidgetItem(str(e))); tbl1.setItem(i, 1, QTableWidgetItem(str(t)))
        lay.addWidget(QLabel("Epoch Latest Computations")); lay.addWidget(tbl1); tbl1.resizeColumnsToContents()
        tree = QTreeWidget(); tree.setColumnCount(2); tree.setHeaderLabels(["Epoch/Computation", "Completion Time"])
        for e, comp_dict in epoch_each.items():
            epoch_item = QTreeWidgetItem([str(e), ""]); 
            [epoch_item.addChild(QTreeWidgetItem([str(comp), str(t)])) for comp, t in comp_dict.items()]
            tree.addTopLevelItem(epoch_item)
        tree.expandAll(); lay.addWidget(QLabel("Epoch Each Result Computations")); lay.addWidget(tree)
        tbl2 = QTableWidget(); tbl2.setColumnCount(2); tbl2.setHorizontalHeaderLabels(["Global Computation", "Completion Time"]); tbl2.setRowCount(len(global_comp))
        for i, (comp, t) in enumerate(global_comp.items()): tbl2.setItem(i, 0, QTableWidgetItem(str(comp))); tbl2.setItem(i, 1, QTableWidgetItem(str(t)))
        lay.addWidget(QLabel("Global Computations")); lay.addWidget(tbl2); tbl2.resizeColumnsToContents()
        tbl3 = QTableWidget(); tbl3.setColumnCount(2); tbl3.setHorizontalHeaderLabels(["Epoch", "Needs Recompute"]); tbl3.setRowCount(len(epoch_recompute))
        for i, e in enumerate(epoch_recompute): tbl3.setItem(i, 0, QTableWidgetItem(str(e))); tbl3.setItem(i, 1, QTableWidgetItem("Yes"))
        lay.addWidget(QLabel("Epochs Needing Recompute")); lay.addWidget(tbl3); tbl3.resizeColumnsToContents()
        tree2 = QTreeWidget(); tree2.setColumnCount(2); tree2.setHeaderLabels(["Epoch/Computation", "Needs Recompute"])
        for e, comp_dict in epoch_each_recompute.items():
            epoch_item = QTreeWidgetItem([str(e), ""]); 
            [epoch_item.addChild(QTreeWidgetItem([str(comp), "Yes"])) for comp, t in comp_dict.items()]
            tree2.addTopLevelItem(epoch_item)
        tree2.expandAll(); lay.addWidget(QLabel("Epoch Each Result Needing Recompute")); lay.addWidget(tree2)
        self.setCentralWidget(cw)
        
if __name__=="__main__":
    any_recent = datetime.datetime(2024, 5, 1, 12, 0, 0)
    epoch_latest = {"epoch1": datetime.datetime(2024, 3, 30, 10, 0, 0), "epoch2": datetime.datetime(2024, 4, 2, 15, 30, 0)}
    epoch_each = {"epoch1": {"compA": datetime.datetime(2024, 3, 29, 9, 0, 0), "compB": datetime.datetime(2024, 3, 29, 10, 0, 0)}, "epoch2": {"compA": datetime.datetime(2024, 4, 2, 15, 0, 0), "compB": datetime.datetime(2024, 4, 2, 15, 30, 0)}}
    global_latest = datetime.datetime(2024, 4, 2, 16, 0, 0)
    global_comp = {"globalCompA": datetime.datetime(2024, 4, 2, 16, 0, 0), "globalCompB": datetime.datetime(2024, 4, 2, 16, 5, 0)}
    recompute_date = datetime.datetime(2024, 4, 1, 0, 0, 0)
    epoch_recompute = [e for e, t in epoch_latest.items() if t < recompute_date]
    epoch_each_recompute = {e: {comp: t for comp, t in comp_dict.items() if t < recompute_date} for e, comp_dict in epoch_each.items()}
    app = QApplication(sys.argv)
    win = MainWindow(any_recent, epoch_latest, epoch_each, global_latest, global_comp, epoch_recompute, epoch_each_recompute)
    win.show()
    sys.exit(app.exec_())

"""
Build GUI parameter tree widget from Parameters object

"""
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# Custom Param Types:
from pyphoplacecellanalysis.External.pyqtgraph.parametertree import Parameter, ParameterTree
from pyphoplacecellanalysis.GUI.PyQtPlot.Params.SaveRestoreStateParamHelpers import default_parameters_save_restore_state_button_children, add_save_restore_btn_functionality # for adding save/restore buttons

## For qdarkstyle theming support:
import qdarkstyle
# app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
# app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

# For BreezeStylesheets support:
from qtpy import QtWidgets
from qtpy.QtCore import QFile, QTextStream
import pyphoplacecellanalysis.External.breeze_style_sheets.breeze_resources
# # set stylesheet:
# stylesheet_qss_file = QFile(":/dark/stylesheet.qss")
# stylesheet_qss_file.open(QFile.ReadOnly | QFile.Text)
# stylesheet_data_stream = QTextStream(stylesheet_qss_file)
# # app.setStyleSheet(stylesheet_data_stream.readAll())



app = pg.mkQApp("Parameter Tree Filter Options")
# app.setStyleSheet(stylesheet_data_stream.readAll())
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5()) # QDarkStyle version






## Main Create function
def create_parameter_tree_widget(parameters: Parameter, debug_print=False):
    """ Creates an actual ParameterTree widget (the GUI)
    
    """
    ## Create two ParameterTree widgets, both accessing the same data
    param_tree = ParameterTree()
    param_tree.setParameters(parameters, showTop=False)
    param_tree.setWindowTitle('pyqtgraph example: Parameter Tree')
    
    layout_win = pg.LayoutWidget()
    # Add widgets:
    layout_win.addWidget(param_tree)
    # layout.addWidget(QtGui.QLabel("These are two views of the same data. They should always display the same values."), 0, 0, 1, 2)
    # layout.addWidget(t, 1, 0, 1, 1)
    layout_win.show()
    layout_win.resize(800,900)

    ## test save/restore
    state = parameters.saveState()
    parameters.restoreState(state)
    compareState = parameters.saveState()
    assert pg.eq(compareState, state)
    return layout_win, param_tree



if __name__ == '__main__':
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ParameterTreeWidget import create_parameter_tree_widget
    # win, param_tree = create_pipeline_filter_parameter_tree()
    win, param_tree = create_parameter_tree_widget()
    pg.exec()

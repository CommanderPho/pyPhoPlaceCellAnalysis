"""
Simple use of DataTreeWidget to display a structure of nested dicts, lists, and arrays

Example:
    d = {
        'active_sess_config':curr_active_pipeline.active_sess_config.__dict__,
        'active_configs':curr_active_pipeline.active_configs,
        'active_session_computation_configs':active_session_computation_configs[0].__dict__
    }
    # d = {
    #     'active_two_step_decoder': active_two_step_decoder,
    #     'active_extended_stats': active_extended_stats
    # }
    # d = {
    #     'active_session_computation_configs':active_session_computation_configs,
    #     'active_two_step_decoder': active_two_step_decoder,
    #     'active_extended_stats': active_extended_stats
    # }

    import pyphoplacecellanalysis.External.pyqtgraph as pg
    from pyphoplacecellanalysis.GUI.PyQtPlot.Examples.pyqtplot_DataTreeWidget import CustomFormattingDataTreeWidget, plot_dataTreeWidget
    tree, app = plot_dataTreeWidget(data=d, title='PhoOutputDataTreeApp')
    pg.exec() # required in an empty notebook to get the window to show instead of just launching and locking up

"""

# required to enable non-blocking interaction:
# from PyQt5.Qt import QApplication
# # start qt event loop
# _instance = QApplication.instance()
# if not _instance:
#     _instance = QApplication([])
# app = _instance

import traceback
import types
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from collections import OrderedDict

## For custom parameters:
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from neuropy.utils.dynamic_container import DynamicContainer

try:
    import metaarray  # noqa
    HAVE_METAARRAY = True
except:
    HAVE_METAARRAY = False
    
from pyphocorehelpers.print_helpers import debug_dump_object_member_shapes

from pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs import InteractivePlaceCellConfig

__all__ = ['CustomFormattingDataTreeWidget']

class CustomFormattingDataTreeWidget(pg.DataTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays)
    """
    def __init__(self, parent=None, data=None):
        pg.DataTreeWidget.__init__(self, parent=parent, data=data)

    def custom_parse_data_format(self, data):
        """ called by self.parse(data) to get custom info for custom widget types. Should return None if the data isn't specially handled so the default formatting can be applied by the superclass. """
        # print('CustomFormattingDataTreeWidget.custom_parse_data_format(...) called!')
        # defaults for all objects
        typeStr = type(data).__name__
        if typeStr == 'instance':
            typeStr += ": " + data.__class__.__name__
        widget = None
        desc = ""
        childs = {}
        
        # custom type-specific changes
        if isinstance(data, InteractivePlaceCellConfig):
            desc = "PhoCustomFormatting applied! InteractivePlaceCellConfig"
            childs = OrderedDict(enumerate(data.__dict__)) ## Convert to an OrderedDict
            # childs = data.__dict__ ## Convert to a regular __dict__
            # childs = {} # returns no children
            
        elif isinstance(data, (DynamicParameters, DynamicContainer)):
            desc = "PhoCustomFormatting applied! DynamicParameters or DynamicContainer"
            # childs = OrderedDict(enumerate(data.to_dict())) ## Convert to an OrderedDict
            childs = OrderedDict(data.to_dict()) ## Convert to an OrderedDict

        else:
            raise NotImplementedError
        
        return typeStr, desc, childs, widget


    def parse(self, data):
        """ 
        Given any python object, return:
          * type
          * a short string representation
          * childs: a dict of sub-objects to be parsed
          * optional widget to display as sub-node
          
        Note that effectively only OrderedDictionary objects are displayed in the table, and nearly everything else is converted into an OrderedDict and then parsed again (recurrsively).
        
        """

        try:
            typeStr, desc, childs, widget = self.custom_parse_data_format(data)
            # print('handled by custom formatter')
            
        
        except NotImplementedError:
            # this type isn't specially handled by the custom formatter, use the defaults from the parent class:
            # print('not handled by custom formatter')
            # TODO: in the future actually call super, but for now just re-implement:
        
            # defaults for all objects
            typeStr = type(data).__name__
            if typeStr == 'instance':
                typeStr += ": " + data.__class__.__name__
            widget = None
            desc = ""
            childs = {}
            
            # type-specific changes
            if isinstance(data, dict):
                desc = "length=%d" % len(data)
                if isinstance(data, OrderedDict):
                    childs = data
                else:
                    try:
                        childs = OrderedDict(sorted(data.items()))
                    except TypeError: # if sorting falls
                        childs = OrderedDict(data.items())
                        
            elif isinstance(data, (list, tuple)):
                desc = "length=%d" % len(data)
                childs = OrderedDict(enumerate(data)) # for list-like objects, enumerate their indices/items as an OrderedDict so recurrsion will work.
            elif HAVE_METAARRAY and (hasattr(data, 'implements') and data.implements('MetaArray')):
                childs = OrderedDict([
                    ('data', data.view(np.ndarray)),
                    ('meta', data.infoCopy())
                ])
            elif isinstance(data, np.ndarray):
                desc = "shape=%s dtype=%s" % (data.shape, data.dtype)
                table = pg.TableWidget()
                table.setData(data)
                table.setMaximumHeight(200)
                widget = table
            elif isinstance(data, types.TracebackType):  ## convert traceback to a list of strings
                frames = list(map(str.strip, traceback.format_list(traceback.extract_tb(data))))
                widget = QtWidgets.QPlainTextEdit('\n'.join(frames))
                widget.setMaximumHeight(200)
                widget.setReadOnly(True)
            else:
                desc = str(data)

        # return no matter what
        return typeStr, desc, childs, widget

        
        




def plot_dataTreeWidget(data, title='PhoOutputDataTreeApp'):
    """ 

    from pyphoplacecellanalysis.GUI.PyQtPlot.Examples.pyqtplot_DataTreeWidget import plot_dataTreeWidget

    """
    app = pg.mkQApp(title)
    tree = CustomFormattingDataTreeWidget(data=data)
    tree.show()
    tree.setWindowTitle(f'PhoOutputDataTreeApp: pyqtgraph DataTreeWidget: {title}')
    tree.resize(800,600)
    return tree, app



if __name__ == '__main__':
    d = {
        'active_sess_config':curr_active_pipeline.active_sess_config.__dict__,
        'active_configs':curr_active_pipeline.active_configs,
        'active_session_computation_configs':active_session_computation_configs[0].__dict__
    }
    # d = {
    #     'active_two_step_decoder': active_two_step_decoder,
    #     'active_extended_stats': active_extended_stats
    # }
    # d = {
    #     'active_session_computation_configs':active_session_computation_configs,
    #     'active_two_step_decoder': active_two_step_decoder,
    #     'active_extended_stats': active_extended_stats
    # }

    from pyphoplacecellanalysis.GUI.PyQtPlot.Examples.pyqtplot_DataTreeWidget import plot_dataTreeWidget
    tree, app = plot_dataTreeWidget(data=d, title='PhoOutputDataTreeApp')
    
    pg.exec()

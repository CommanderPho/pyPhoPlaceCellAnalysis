from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph as pg
import numpy as np

# matplotlib:
# import matplotlib.pyplot as plt


from pyphoplacecellanalysis.General.Pipeline.Stages.Display import DefaultDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.Ratemaps import DefaultRatemapDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import DefaultDecoderDisplayFunctions

from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.AssociatedOutputWidgetNodeMixin import AssociatedAppNodeMixin, AssociatedOutputWidgetNodeMixin


class PipelineDisplayNode(AssociatedOutputWidgetNodeMixin, AssociatedAppNodeMixin, CtrlNode):
    """Displays active pipeline.
        TODO: allow the user to select which display function will be used, and optionally pass any function-specific parameters by adding additional inputs.
            - Probably should have a plaintext input like the arbitrary python exec example node to allow typing the function.
            - Ideally would have an option to spawn the output widget in a new window or to add it to the main window.
    """
    nodeName = "PipelineDisplayNode"
    # uiTemplate = [
    #     ('filter_function', 'combo', {'values': ['test1', 'test2', 'custom...'], 'index': 0}),
    #     # ('sigma',  'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}),
    #     # ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5, 'minStep': 0.01, 'bounds': [0.0, None]}),
    # ]
    def __init__(self, name):
        # Initialize the associated app
        self.app = None
        # Initialize the associated view
        self.view = None
        self.on_remove_function = None
        ## Define the input / output terminals available on this node
        terminals = {
            'active_data_mode': dict(io='in'),
            'active_session_computation_configs': dict(io='in'),
            'active_session_filter_configurations': dict(io='in'),
            'active_pipeline': dict(io='in'),
            'display_outputs': dict(io='out'),            
        }
        CtrlNode.__init__(self, name, terminals=terminals)
        
    def process(self, active_data_mode=None, active_session_computation_configs=None, active_session_filter_configurations=None, active_pipeline=None, display=True):
        
        if (active_pipeline is None) or (not display):
            return {'display_outputs': None}

        active_config_name = 'maze1'
        
        # print(f'plt.isinteractive(): {plt.isinteractive()}')
              
        # plt.plot(np.arange(9))
        # plt.show()
        
        # with plt.ion():
        #     plt.plot(np.arange(9))
    
        # not shown immediately:    
        # with plt.ioff():
        #     plt.plot(np.arange(9))
        # plt.show()
        # display_outputs = {
        #     'fig':plt.gcf() 
        # }
        
        if (self.view is None):
            return {'display_outputs': None}
        else:
            subplot = self.view.getFigure().add_subplot(111)
            subplot.plot(np.arange(9), np.full((9,), 15))
            display_outputs = {
            'subplot':subplot 
            }
            self.view.draw()
        
        # display_outputs = active_pipeline.display(DefaultDecoderDisplayFunctions._display_two_step_decoder_prediction_error_2D, active_config_name, variable_name='p_x_given_n') # works!
        # if (self.app is not None) and (self.view is not None):
        #     app, parent_root_widget, root_render_widget = active_pipeline.display(DefaultRatemapDisplayFunctions._display_placemaps_pyqtplot_2D, active_config_name, 
        #                                                                             app=self.app, parent_root_widget=self.view, root_render_widget=None)
            
        #     # root_render_widget is added to parent_root_widget if it's needed, which currently it is every frame.
            
        #     # parent_root_widget.show()
        #     display_outputs = {
        #     'app':app, 'parent_root_widget':parent_root_widget, 'root_render_widget':root_render_widget   
        #     }
        # else:
        #     display_outputs = None
        #     raise
            
        return {'display_outputs': display_outputs}


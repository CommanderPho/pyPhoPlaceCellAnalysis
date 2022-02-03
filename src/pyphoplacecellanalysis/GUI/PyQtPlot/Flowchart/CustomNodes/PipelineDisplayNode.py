import sys
import importlib
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph as pg
import numpy as np

# matplotlib:
# import matplotlib.pyplot as plt
# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core
    importlib.reload(core)
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print("neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.")
    from neuropy import core

from neuropy.plotting.ratemaps import enumTuningMap2DPlotVariables

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
        enable_saving_to_disk = False
        
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
            # test plot
            active_fig = self.view.getFigure()
            active_fig.clf()
            self.view.draw()
            
            # subplot = self.view.getFigure().add_subplot(111)
            # subplot.plot(np.arange(9), np.full((9,), 15))
            
            # active_fig_num = None
            active_fig_num = 1
            # active_fig_num = active_fig.number
                        
            # active_fig_num = self.view.getFigure() # pass the figure itself as the fignum
            # print(f'active_fig_num: {active_fig_num}')
            
            # curr_kdiba_pipeline.display(DefaultDisplayFunctions._display_2d_placefield_result_plot_ratemaps_2D, filter_name, enable_spike_overlay=False, plot_variable=enumTuningMap2DPlotVariables.FIRING_MAPS, fignum=0, max_screen_figure_size=(None, 1868), debug_print=False, enable_saving_to_disk=enable_saving_to_disk) # works!
            
            active_pf_2D_figures = active_pipeline.display(DefaultDisplayFunctions._display_2d_placefield_result_plot_ratemaps_2D, active_config_name, enable_spike_overlay=False, plot_variable=enumTuningMap2DPlotVariables.TUNING_MAPS, fignum=active_fig_num, fig=active_fig, max_screen_figure_size=(None, 1868), debug_print=False, enable_saving_to_disk=enable_saving_to_disk)

            post_plot_active_fig = active_pf_2D_figures[0]
            
            # active_fig_num = post_plot_active_fig.number() # pass the figure itself as the fignum
            print(f'active_fig_num: {active_fig_num}')
            
            # active_fig.add_subfigure(post_plot_active_fig)
            
            # display_outputs = {'subplot':subplot}
            display_outputs = {'fig':active_fig, 'fig_num':active_fig_num}
            
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


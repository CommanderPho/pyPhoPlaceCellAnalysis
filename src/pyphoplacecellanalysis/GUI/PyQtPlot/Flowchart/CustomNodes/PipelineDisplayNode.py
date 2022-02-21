import sys
import importlib
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode, PlottingCtrlNode
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
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.CtrlNodeMixins import KeysListAccessingMixin


class PipelineDisplayNode(AssociatedOutputWidgetNodeMixin, AssociatedAppNodeMixin, KeysListAccessingMixin, PlottingCtrlNode):
    """Displays active pipeline.
        TODO: allow the user to select which display function will be used, and optionally pass any function-specific parameters by adding additional inputs.
            - Probably should have a plaintext input like the arbitrary python exec example node to allow typing the function.
            - Ideally would have an option to spawn the output widget in a new window or to add it to the main window.
    """
    nodeName = "PipelineDisplayNode"
    uiTemplate = [
        ('display_function', 'combo', {'values': [], 'index': 0}),
        ('computed_result', 'combo', {'values': [], 'index': 0}),
        ('display', 'action'),
        # ('sigma',  'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}),
        # ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5, 'minStep': 0.01, 'bounds': [0.0, None]}),
    ]
    def __init__(self, name):
        # Initialize the associated app
        self.app = None
        # Initialize the associated view
        self.view = None
        self.on_remove_function = None
        ## Define the input / output terminals available on this node
        terminals = {
            'mode': dict(io='in'),
            'computation_configs': dict(io='in'),
            'filter_configs': dict(io='in'),
            'pipeline': dict(io='in'),
            'display_outputs': dict(io='out'),            
        }
        PlottingCtrlNode.__init__(self, name, terminals=terminals)
        
        # Set up the combo boxes:
        # self.display_function_keys = []
        # self.computed_result_keys = []
        self.combo_box_keys_dict = {'display_function':[], 'computed_result':[]}

        
        # Setup the display button:
        self.ctrls['display'].setText('Display')
        def click():
            self.ctrls['display'].processing("Hold on..")
            # time.sleep(2.0)
            
            # Not sure whether to call self.changed() (from CtrlNode) or self.update() from its parent class.
            # self.update() 
            self.changed() # should trigger re-computation in a blocking manner.
            
            # global fail
            # fail = not fail
            
            fail = False
            if fail:
                self.ctrls['display'].failure(message="FAIL.", tip="There was a failure. Get over it.")
            else:
                self.ctrls['display'].success(message="Bueno!")
                
                
        self.ctrls['display'].clicked.connect(click)
        
        
    def process(self, mode=None, computation_configs=None, filter_configs=None, pipeline=None, display=True):
        # Get the list of available display functions:
        all_display_functions_list = pipeline.registered_display_function_names
        """
            ['_display_1d_placefield_validations',
            '_display_2d_placefield_result_plot_ratemaps_2D',
            '_display_2d_placefield_result_plot_raw',
            '_display_3d_image_plotter',
            '_display_3d_interactive_custom_data_explorer',
            '_display_3d_interactive_spike_and_behavior_browser',
            '_display_3d_interactive_tuning_curves_plotter',
            '_display_normal',
            '_display_placemaps_pyqtplot_2D',
            '_display_decoder_result',
            '_display_plot_most_likely_position_comparisons',
            '_display_two_step_decoder_prediction_error_2D',
            '_display_two_step_decoder_prediction_error_animated_2D']
        """
        self.updateKeys('display_function', all_display_functions_list)
        
        
        if (pipeline is None) or (not display):
            return {'display_outputs': None}

        # Update the list of available results:
        all_computation_results_keys = list(pipeline.computation_results.keys()) # ['maze1', 'maze2']
        self.updateKeys('computed_result', all_computation_results_keys)
        
        
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
            
            active_pf_2D_figures = pipeline.display(DefaultDisplayFunctions._display_2d_placefield_result_plot_ratemaps_2D, active_config_name, enable_spike_overlay=False, plot_variable=enumTuningMap2DPlotVariables.TUNING_MAPS, fignum=active_fig_num, fig=active_fig, max_screen_figure_size=(None, 1868), debug_print=False, enable_saving_to_disk=enable_saving_to_disk)

            post_plot_active_fig = active_pf_2D_figures[0]
            
            # active_fig_num = post_plot_active_fig.number() # pass the figure itself as the fignum
            print(f'active_fig_num: {active_fig_num}')
            
            # active_fig.add_subfigure(post_plot_active_fig)
            
            # display_outputs = {'subplot':subplot}
            display_outputs = {'fig':active_fig, 'fig_num':active_fig_num}
            
            self.view.draw()
        
        # display_outputs = pipeline.display(DefaultDecoderDisplayFunctions._display_two_step_decoder_prediction_error_2D, active_config_name, variable_name='p_x_given_n') # works!
        # if (self.app is not None) and (self.view is not None):
        #     app, parent_root_widget, root_render_widget = pipeline.display(DefaultRatemapDisplayFunctions._display_placemaps_pyqtplot_2D, active_config_name, 
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

        
    def updateKeys(self, ctrl_name, data):
        keys = PipelineDisplayNode.get_keys_list(data)
        
        for c in self.ctrls.values():
            c.blockSignals(True)

        for c in [self.ctrls[ctrl_name]]:
            cur = str(c.currentText())
            c.clear()
            for k in keys:
                c.addItem(k)
                if k == cur:
                    c.setCurrentIndex(c.count()-1)
        # for c in [self.ctrls['color'], self.ctrls['border']]:
        #     c.setArgList(keys)
        for c in self.ctrls.values():
            c.blockSignals(False)
        # Update the self.keys value:
        self.combo_box_keys_dict[ctrl_name] = keys
       
        
    def saveState(self):
        state = PlottingCtrlNode.saveState(self)        
        # return {'display_function_keys': self.display_function_keys, 'computed_result_keys': self.computed_result_keys, 'ctrls': state}
        return {'combo_box_keys_dict': self.combo_box_keys_dict, 'ctrls': state}
        
    

        
    def restoreState(self, state):
        combo_box_keys_dict = state['combo_box_keys_dict'] 
        self.updateKeys('display_function', combo_box_keys_dict['display_function'])
        self.updateKeys('computed_result', combo_box_keys_dict['computed_result'])
        # self.updateKeys('display_function', state['display_function_keys'])
        # self.updateKeys('computed_result', state['computed_result_keys'])
        PlottingCtrlNode.restoreState(self, state['ctrls'])

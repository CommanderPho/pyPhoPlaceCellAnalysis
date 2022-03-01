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

from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.AssociatedOutputWidgetNodeMixin import AddRemoveActionNodeMixin, AssociatedAppNodeMixin, AssociatedOutputWidgetNodeMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.CtrlNodeMixins import KeysListAccessingMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Mixins.DisplayNodeViewHelpers import DisplayMatplotlibWidgetMixin



class PipelineDisplayNode(DisplayMatplotlibWidgetMixin, AssociatedOutputWidgetNodeMixin, AddRemoveActionNodeMixin, AssociatedAppNodeMixin, KeysListAccessingMixin, PlottingCtrlNode):
    """Displays active pipeline.
        TODO: allow the user to select which display function will be used, and optionally pass any function-specific parameters by adding additional inputs.
            - Probably should have a plaintext input like the arbitrary python exec example node to allow typing the function.
            - Ideally would have an option to spawn the output widget in a new window or to add it to the main window.
    """
    nodeName = "PipelineDisplayNode"
    uiTemplate = [
        ('display_function', 'combo', {'values': [], 'index': 0}),
        ('computed_result', 'combo', {'values': [], 'index': 0}),
        ('rebuild_widgets', 'action'),
        ('display', 'action'),
    ]
    # TODO: currently hardcoded:
    plotter_widget_fcns = ['_display_3d_image_plotter', '_display_3d_interactive_custom_data_explorer','_display_3d_interactive_spike_and_behavior_browser','_display_3d_interactive_tuning_curves_plotter']
    
    def __init__(self, name, on_add_function=None, on_remove_function=None):
        # Initialize the associated app
        self.app = None
        # Initialize the associated view
        self._display_results = dict()
        
        # self.setView() # initializes self._view, self._owned_parent_container, self.on_add_function, and self._on_remove_function to None
        self._owned_parent_container = None
        self._view = None
        self._on_add_function = on_add_function
        self._on_remove_function = on_remove_function
        
        # self.view = None
        # self.on_remove_function = None
        

        ## Define the input / output terminals available on this node
        terminals = {
            'computation_configs': dict(io='in'),
            'pipeline': dict(io='in'),
            'display_outputs': dict(io='out'),
            'display_results': dict(io='out'),             
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
            self.update() 
            # self.changed() # should trigger re-computation in a blocking manner.
            
            # global fail
            # fail = not fail
            
            fail = False
            if fail:
                self.ctrls['display'].failure(message="FAIL.", tip="There was a failure. Get over it.")
            else:
                self.ctrls['display'].success(message="Bueno!")

        self.ctrls['display'].clicked.connect(click)

    
        # Setup the rebuild_widgets button:
        self.ctrls['rebuild_widgets'].setText('Rebuild')
        def click_rebuild_widgets():
            self.ctrls['rebuild_widgets'].processing("Hold on..")
            # time.sleep(2.0)
            
            # Not sure whether to call self.changed() (from CtrlNode) or self.update() from its parent class.
            self.update() 
            # self.changed() # should trigger re-computation in a blocking manner.
            
            # global fail
            # fail = not fail
            
            fail = False
            if fail:
                self.ctrls['rebuild_widgets'].failure(message="FAIL.", tip="There was a failure. Get over it.")
            else:
                self.ctrls['rebuild_widgets'].success(message="Bueno!")
                
        self.ctrls['rebuild_widgets'].clicked.connect(click_rebuild_widgets)
        
        
        
    
    @property
    def selected_display_function_name(self):
        """The selected_display_function_name property."""
        return str(self.ctrls['display_function'].currentText())
    
    @property
    def selected_computed_result_name(self):
        """The selected_display_function_name property."""
        return str(self.ctrls['computed_result'].currentText())
    
    
    @property
    def display_results(self):
        """The display_results property."""
        return self._display_results
    @display_results.setter
    def display_results(self, value):
        self._display_results = value
        
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
        
        # Get current setup from GUI:
        active_config_name = self.selected_computed_result_name
        curr_display_fcn = pipeline.registered_display_function_dict.get(self.selected_display_function_name, None)
        
        # if self.selected_display_function_name in pipeline.registered_display_function_dict:
        if curr_display_fcn is not None:
            # if there's a valid selected display function
            # print(f'curr_display_fcn: {self.selected_display_function_name}')
            # active_pf_2D_figures = pipeline.display(curr_display_fcn, active_config_name, enable_spike_overlay=False, plot_variable=enumTuningMap2DPlotVariables.TUNING_MAPS, fignum=active_fig_num, fig=active_fig, max_screen_figure_size=(None, 1868), debug_print=False, enable_saving_to_disk=enable_saving_to_disk)
            
            is_plotter_widget_fcn = (self.selected_display_function_name in PipelineDisplayNode.plotter_widget_fcns)
            is_matplotlib_widget_fcn = (self.selected_display_function_name not in self.plotter_widget_fcns)
            if is_plotter_widget_fcn:
                pass
            elif is_matplotlib_widget_fcn:
                # raise
                self.display_results['kwargs'] = self.display_widget() # provided by DisplayMatplotlibWidgetMixin. Returns a dict like {'fignum':active_fig_num, 'fig':active_fig}
            else:
                raise
                pass 
            
            # curr_kdiba_pipeline.display(DefaultDisplayFunctions._display_2d_placefield_result_plot_ratemaps_2D, filter_name, enable_spike_overlay=False, plot_variable=enumTuningMap2DPlotVariables.FIRING_MAPS, fignum=0, max_screen_figure_size=(None, 1868), debug_print=False, enable_saving_to_disk=enable_saving_to_disk) # works!
            
            if self.display_results is not None:
                custom_args = self.display_results.get('kwargs', {})
            else:
                custom_args = {} # no custom args, just pass empty dictionary

            display_outputs = pipeline.display(curr_display_fcn, active_config_name, **custom_args)

            if isinstance(display_outputs, dict):
                # For 3D pyvista display functions:     'pActiveInteractivePlaceSpikesPlotter', etc.
                # self.display_results = dict()
                self.display_results['outputs'] = display_outputs
                # Search for extant_plotter to reuse in the future calls:
                active_plotter = display_outputs.get('plotter', None)
                # BackgroundPlotter, MultiPlotter
                self.display_results['kwargs'] = {'extant_plotter':active_plotter}
            elif isinstance(display_outputs, list):
                # 2d functions typically
                self.display_results['outputs'] = display_outputs # set the 'outputs' key to the list
                # self.display_results['kwargs'] = {}
                # self.display_results['kwargs'] = {'fignum':active_fig_num, 'fig':active_fig} # could do, but it wouldn't work for 2d functions that didn't accept either of thse parameters.
                # Here there will be an issue with neededing to clear the old kwargs when switching from 3d to 2d mode. Will encounter 'extant_plotter' arg being passed into 2D functions.
            else:
                raise
            
            # Old style:
            # active_pf_2D_figures = pipeline.display(DefaultDisplayFunctions._display_2d_placefield_result_plot_ratemaps_2D, active_config_name, enable_spike_overlay=False, plot_variable=enumTuningMap2DPlotVariables.TUNING_MAPS, fignum=active_fig_num, fig=active_fig, max_screen_figure_size=(None, 1868), debug_print=False, enable_saving_to_disk=enable_saving_to_disk)

            # post_plot_active_fig = active_pf_2D_figures[0]
            
            # active_fig_num = post_plot_active_fig.number() # pass the figure itself as the fignum
            # print(f'active_fig_num: {active_fig_num}')
            
            # active_fig.add_subfigure(post_plot_active_fig)
            
        else:
            # curr_display_fcn is None, meaning all display_outputs should be properly closed.
            if self.display_results is not None:
                active_fig = self.display_results.get('kwargs',{}).get('fig', None) # get the active figure for matplotlib style plots.
                if active_fig is not None:
                    active_fig.close()

            active_fig = None
            active_fig_num = None
                    
            display_outputs = []
            # TODO: properly close all the figures and such if there are some open: do 'self.on_deselect_display_fcn(...)' stuff
            # self.display_results.clear()
            

        
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
            
        return {'display_outputs': display_outputs, 'display_results': self.display_results}

        
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

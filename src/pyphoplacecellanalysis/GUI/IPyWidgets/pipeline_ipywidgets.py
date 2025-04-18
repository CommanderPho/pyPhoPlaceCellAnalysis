from copy import deepcopy
import sys
from enum import Enum
from pyphocorehelpers.DataStructure.enum_helpers import OrderedEnum
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from attrs import define, field, Factory

import ipywidgets as widgets
from IPython.display import display
import matplotlib

from pathlib import Path

# from silx.gui.dialog.ImageFileDialog import ImageFileDialog
# import silx.io

import pandas as pd
from pyphocorehelpers.gui.Jupyter.JupyterButtonRowWidget import build_fn_bound_buttons, JupyterButtonRowWidget, JupyterButtonColumnWidget
from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager
from pyphocorehelpers.Filesystem.path_helpers import open_file_with_system_default
from pyphocorehelpers.assertion_helpers import Assert


from pyphocorehelpers.exception_helpers import CapturedException
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore, QtWidgets
# from pyphoplacecellanalysis.External.pyqtgraph.parametertree.parameterTypes.file import popupFilePicker
from pyphoplacecellanalysis.External.pyqtgraph.widgets.FileDialog import FileDialog
from pyphocorehelpers.gui.Jupyter.simple_widgets import fullwidth_path_widget, create_file_browser


from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme # used in perform_pipeline_save


# AbstractDataFileDialog

def saveFile(on_save_file_callback, caption:str="Save as..", startDir=None, suggestedFileName='test.h5', filter="H5py File (*.h5)", default_suffix:str="h5"):
    """Save this Custom Eval Node to a .pEval file

    fileDialog = saveFile(lambda fileName: print(f'_on_save_file(fileName: {fileName})'), caption="Save pickle as..", startDir=None, suggestedFileName='test.pkl', filter="Pickle File (*.pkl)", default_suffix="pkl")

    fileDialog = saveFile(lambda fileName: print(f'_on_save_file(fileName: {fileName})'), caption="Save HDF5 file as..", startDir=None, suggestedFileName='test.h5', filter="H5py File (*.h5)", default_suffix="h5")

    """
    if startDir is None:
        startDir = '.'
    if suggestedFileName is not None:
        startFile = Path(startDir).joinpath(suggestedFileName).resolve()
    else:
        startFile = Path(startDir).resolve()
    fileDialog: FileDialog = FileDialog(None, caption, str(startFile), filter)
    # fileDialog.setDefaultFilename(suggestedFileName)
    fileDialog.setDefaultSuffix(default_suffix)
    fileDialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave) 
    fileDialog.show()
    fileDialog.fileSelected.connect(on_save_file_callback)
    fileDialog.exec_() # open modally
    return fileDialog


def openDialogAtHome():
    """
    
    """
    from silx.gui import qt
    from silx.gui.dialog.DataFileDialog import DataFileDialog
    
    # Clear the dialog
    path = qt.QDir.homePath()
    # dialog = self.createDialog()

    dialog = DataFileDialog()
    
    # Execute the dialog as modal
    result = dialog.exec()
    if result:
        print("Selection:")
        print(dialog.selectedFile())
        print(dialog.selectedUrl())
    else:
        print("Nothing selected")

    dialog.setDirectory(path)

    # # Execute the dialog as modal
    # result = dialog.exec()
    # self.printResult(dialog, result)
    return dialog, result


def try_save_pickle_as(original_file_path, file_confirmed_callback):
    original_file_path: Path = Path(original_file_path).resolve()
    def _perform_try_save_pickle_as(fileName):
        print(f'_perform_try_save_pickle_as(fileName: {fileName}, original_file_path: {original_file_path})')
        # TODO: perform the copy here:
        return file_confirmed_callback(fileName)

    fileDialog = saveFile(_perform_try_save_pickle_as, caption="Save pickle as..", startDir=original_file_path.parent, suggestedFileName=f'{original_file_path.stem}_bak.pkl', filter="Pickle File (*.pkl)", default_suffix="pkl")
    # dialog = saveFile(lambda fileName: print(f'_on_save_file(fileName: {fileName})'), caption="Save HDF5 file as..", startDir=None, suggestedFileName='test.h5', filter="H5py File (*.h5)", default_suffix="h5")
    return



@metadata_attributes(short_name=None, tags=['enum', 'phases'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-06 22:56', related_items=[])
class CustomProcessingPhases(OrderedEnum):
    """ These phases keep track of groups of computations to run.

    from pyphoplacecellanalysis.GUI.IPyWidgets.pipeline_ipywidgets import PipelineJupyterHelpers, CustomProcessingPhases

    CustomProcessingPhases.clean_run
    CustomProcessingPhases.continued_run
    CustomProcessingPhases.final_run
    
    selector.value


    """
    # Enum members with an additional `_order` attribute for ordering
    clean_run = ("clean_run", 0)
    continued_run = ("continued_run", 1)
    final_run = ("final_run", 2)

    def __init__(self, value, order):
        self._value_ = value
        self._order = order
    
    def get_run_configuration(self) -> Dict:
        ## Different run configurations:
        _out_run_config = {}
        if self.value == CustomProcessingPhases.clean_run.value:
            clean_run = dict(saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE, should_force_reload_all=True, should_freeze_pipeline_updates=False)
            _out_run_config = clean_run
        elif self.value == CustomProcessingPhases.continued_run.value:
            continued_run = dict(saving_mode=PipelineSavingScheme.SKIP_SAVING, should_force_reload_all=False, should_freeze_pipeline_updates=False)
            _out_run_config = continued_run
        elif self.value == CustomProcessingPhases.final_run.value:
            final_run = dict(saving_mode=PipelineSavingScheme.SKIP_SAVING, should_force_reload_all=False, should_freeze_pipeline_updates=False) # use export_session_h5_file_completion_function instead of enable_hdf5_output=True
            _out_run_config = final_run
        else: 
            raise NotImplementedError
        
        return _out_run_config
    

    @classmethod
    def get_extended_computations_include_includelist_phase_dict(cls) -> Dict[str, "CustomProcessingPhases"]:
        return{'lap_direction_determination':CustomProcessingPhases.clean_run, 'pf_computation':CustomProcessingPhases.clean_run, 'pfdt_computation':CustomProcessingPhases.clean_run,
                                        'position_decoding':CustomProcessingPhases.clean_run, 'position_decoding_two_step':CustomProcessingPhases.final_run, 
                                        'firing_rate_trends':CustomProcessingPhases.continued_run,
            # 'pf_dt_sequential_surprise':CustomProcessingPhases.final_run,  # commented out 2024-11-05
            'extended_stats':CustomProcessingPhases.continued_run,
            'long_short_decoding_analyses':CustomProcessingPhases.continued_run, 'jonathan_firing_rate_analysis':CustomProcessingPhases.continued_run, 'long_short_fr_indicies_analyses':CustomProcessingPhases.continued_run, 'short_long_pf_overlap_analyses':CustomProcessingPhases.final_run, 'long_short_post_decoding':CustomProcessingPhases.continued_run, 
            # 'ratemap_peaks_prominence2d':CustomProcessingPhases.final_run, # commented out 2024-11-05
            'long_short_inst_spike_rate_groups':CustomProcessingPhases.continued_run,
            'long_short_endcap_analysis':CustomProcessingPhases.continued_run,
            # 'spike_burst_detection':CustomProcessingPhases.continued_run,
            'split_to_directional_laps':CustomProcessingPhases.clean_run,
            'merged_directional_placefields':CustomProcessingPhases.continued_run,
            'rank_order_shuffle_analysis':CustomProcessingPhases.final_run,
            'directional_train_test_split':CustomProcessingPhases.final_run,
            'directional_decoders_decode_continuous':CustomProcessingPhases.continued_run,
            'directional_decoders_evaluate_epochs':CustomProcessingPhases.continued_run,
            'directional_decoders_epoch_heuristic_scoring':CustomProcessingPhases.continued_run,
            'extended_pf_peak_information':CustomProcessingPhases.final_run,
            'perform_wcorr_shuffle_analysis':CustomProcessingPhases.final_run,
            'non_PBE_epochs_results':CustomProcessingPhases.continued_run, # #TODO 2025-02-18 20:15: - [ ] Added to compute the new non_PBE results
            'generalized_specific_epochs_decoding':CustomProcessingPhases.continued_run, # 2025-04-15 00:25 added
        }


@metadata_attributes(short_name=None, tags=['jupyter'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-06 22:56', related_items=['CustomProcessingPhases'])
class PipelineJupyterHelpers:
    """ 

    Usage:    
        from pyphocorehelpers.gui.Jupyter.ipython_widget_helpers import EnumSelectorWidgets
        from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme # used in perform_pipeline_save
        from pyphoplacecellanalysis.GUI.IPyWidgets.pipeline_ipywidgets import PipelineJupyterHelpers, CustomProcessingPhases

        def update_global_variable(var_name, value):
            globals()[var_name] = value
            
        saving_mode = PipelineSavingScheme.SKIP_SAVING
        force_reload = False
        selector = PipelineJupyterHelpers._build_pipeline_custom_processing_mode_selector_widget(debug_print=True, update_global_variable_fn=update_global_variable)
        print(f'force_reload: {force_reload}, saving_mode: {saving_mode}')
        force_reload
        saving_mode

    """

    @classmethod
    def perform_pipeline_save(cls, curr_active_pipeline):
        if not curr_active_pipeline.updated_since_last_pickle:
            # _bak_saving_mode = saving_mode
            print(f'WARN: pipeline does not seem to have been updated since last pickle. Saving anyway.')
        try:
            saving_mode = PipelineSavingScheme.TEMP_THEN_OVERWRITE
            print(f'saving pipeline...')
            curr_active_pipeline.save_pipeline(saving_mode=saving_mode)
            print(f'saving global_computation_results...')
            curr_active_pipeline.save_global_computation_results()
            print(f'saving complete.')
        except Exception as e:
            ## TODO: catch/log saving error and indicate that it isn't saved.
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f'ERROR RE-SAVING PIPELINE after update. error: {e}')
        finally:
            # saving_mode = _bak_saving_mode
            print(f'done!')
            


    @classmethod
    def pipeline_computation_mode_widget(cls, curr_active_pipeline):
        """ not fully implemented. Works when it's in a notebook but needs to be refactored. """
        import ipywidgets as widgets
        from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme

        saving_mode = PipelineSavingScheme.SKIP_SAVING
        force_reload = False

        saving_mode_dropdown = widgets.Dropdown(
            options=[('Read if possible', PipelineSavingScheme.SKIP_SAVING), ('Temp then write', PipelineSavingScheme.TEMP_THEN_OVERWRITE), ('Overwrite in place', PipelineSavingScheme.OVERWRITE_IN_PLACE)],
            description='Mode:'
        )

        force_reload_checkbox = widgets.Checkbox(
            value=False,
            description='Force reload'
        )

        def update_variables(change):
            global saving_mode, force_reload
            saving_mode = saving_mode_dropdown.value
            force_reload = force_reload_checkbox.value

        saving_mode_dropdown.observe(update_variables, 'value')
        force_reload_checkbox.observe(update_variables, 'value')

        return widgets.VBox([saving_mode_dropdown, force_reload_checkbox])
                


    @classmethod
    def _build_pipeline_custom_processing_mode_selector_widget(cls, update_global_variable_fn=None, active_post_on_value_change_fn=None, enable_full_view: bool = True, debug_print=False):
        """ Renders an interactive jupyter widget control to select the current pipeline mode:

        Usage:        
            from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme # used in perform_pipeline_save
            from pyphoplacecellanalysis.GUI.IPyWidgets.pipeline_ipywidgets import PipelineJupyterHelpers, CustomProcessingPhases

            def update_global_variable(var_name, value):
                globals()[var_name] = value
                
            selector = PipelineJupyterHelpers._build_pipeline_custom_processing_mode_selector_widget(update_global_variable_fn=update_global_variable, debug_print=False, enable_full_view=True)


        """
        from pyphocorehelpers.gui.Jupyter.ipython_widget_helpers import EnumSelectorWidgets
        
        if update_global_variable_fn is None:
            def update_global_variable(var_name, value):
                if debug_print:
                    print(f"update_global_variable(var_name: '{var_name}', value: {value})") # for `create_enum_selector`
                globals()[var_name] = value
                if debug_print:
                    post_val = globals()[var_name]
                    print(f'\tglobals()[var_name]): {post_val}')
            update_global_variable_fn = update_global_variable
            
        assert update_global_variable_fn is not None

        # target_enum_class = Color
        # target_enum_class = PipelineSavingScheme
        target_enum_class = CustomProcessingPhases

        # Create and display the selector
        # selector = EnumSelectorWidgets.create_enum_selector(target_enum_class, defer_display=True)
        selector = EnumSelectorWidgets.create_enum_toggle_buttons(target_enum_class, defer_display=True)
        
        if enable_full_view:       
             
            output_label = widgets.Label(value=f"Empty")
            
            if active_post_on_value_change_fn is None:
                def post_on_value_change_fn(new_val):
                    """ 
                    # Read if possible:
                    saving_mode = PipelineSavingScheme.SKIP_SAVING
                    force_reload = False

                    # 
                    # # Force write:
                    # saving_mode = PipelineSavingScheme.TEMP_THEN_OVERWRITE
                    # saving_mode = PipelineSavingScheme.OVERWRITE_IN_PLACE
                    # force_reload = True

                    ## TODO: if loading is not possible, we need to change the `saving_mode` so that the new results are properly saved.

                    """
                    new_run_config = new_val.get_run_configuration()
                    # if debug_print:
                    #     print(f"Selected: {new_val}") # for `create_enum_selector`
                    #     print(new_run_config)
                    force_reload: bool = new_run_config['should_force_reload_all']
                    saving_mode: PipelineSavingScheme = new_run_config['saving_mode']                                
                    # force_reload = globals()['force_reload']
                    # saving_mode = globals()['saving_mode']
                    output_label.value = f'force_reload: {force_reload}, saving_mode: {saving_mode}'
                ## END def post_on_value_change_fn(new_val)...

                active_post_on_value_change_fn = post_on_value_change_fn

            # output_view = widgets.Output()
            # widgets.interactive_output(
            # widgets.widget_output
            
            # selector = PipelineJupyterHelpers._build_pipeline_custom_processing_mode_selector_widget(update_global_variable_fn=update_global_variable, post_on_value_change_fn=post_on_value_change_fn, debug_print=False)
            vbox = widgets.VBox([selector, output_label])
            display(vbox)
        else:
            display(selector)
        
        # Access selected value
        def on_value_change(change):
            """ 
            # Read if possible:
            saving_mode = PipelineSavingScheme.SKIP_SAVING
            force_reload = False

            # 
            # # Force write:
            # saving_mode = PipelineSavingScheme.TEMP_THEN_OVERWRITE
            # saving_mode = PipelineSavingScheme.OVERWRITE_IN_PLACE
            # force_reload = True

            ## TODO: if loading is not possible, we need to change the `saving_mode` so that the new results are properly saved.

            """
            # new_val = change['new']
            new_val = target_enum_class[change['new']] #
            new_run_config = new_val.get_run_configuration()
            if debug_print:
                print(f"Selected: {new_val}") # for `create_enum_selector`
                print(new_run_config)
            
            # should_force_reload_all: bool = new_run_config['should_force_reload_all']
            # saving_mode: PipelineSavingScheme = new_run_config['saving_mode']
            
            # Update the global variable
            update_global_variable_fn('force_reload', new_run_config['should_force_reload_all'])
            update_global_variable_fn('saving_mode', new_run_config['saving_mode'])

            if active_post_on_value_change_fn is not None:
                active_post_on_value_change_fn(new_val)
                
        ## END def on_value_change(change)...	

        selector.observe(on_value_change, names='value')
        
        # if enable_full_view:       
             
        #     output_label = widgets.Label(value=f"Empty")
            
        #     def post_on_value_change_fn(new_val):
        #         """ 
        #         # Read if possible:
        #         saving_mode = PipelineSavingScheme.SKIP_SAVING
        #         force_reload = False

        #         # 
        #         # # Force write:
        #         # saving_mode = PipelineSavingScheme.TEMP_THEN_OVERWRITE
        #         # saving_mode = PipelineSavingScheme.OVERWRITE_IN_PLACE
        #         # force_reload = True

        #         ## TODO: if loading is not possible, we need to change the `saving_mode` so that the new results are properly saved.

        #         """
        #         new_run_config = new_val.get_run_configuration()
        #         # if debug_print:
        #         #     print(f"Selected: {new_val}") # for `create_enum_selector`
        #         #     print(new_run_config)
        #         force_reload: bool = new_run_config['should_force_reload_all']
        #         saving_mode: PipelineSavingScheme = new_run_config['saving_mode']                                
        #         # force_reload = globals()['force_reload']
        #         # saving_mode = globals()['saving_mode']
        #         output_label.value = f'force_reload: {force_reload}, saving_mode: {saving_mode}'
                

        #     # output_view = widgets.Output()
        #     # widgets.interactive_output(
        #     # widgets.widget_output
            
        #     # selector = PipelineJupyterHelpers._build_pipeline_custom_processing_mode_selector_widget(update_global_variable_fn=update_global_variable, post_on_value_change_fn=post_on_value_change_fn, debug_print=False)
        #     vbox = widgets.VBox([selector, output_label])
        #     return vbox
        # else:
        #     return selector
        return selector, on_value_change


def interactive_pipeline_files(curr_active_pipeline, defer_display:bool=False) -> JupyterButtonRowWidget:
    """	Displays a row of four buttons relating to the curr_active_pipeline that reveal the Output Folder, global pickle, pipeline pickle, and .h5 export path in the system file explorer.

    from pyphoplacecellanalysis.GUI.IPyWidgets.pipeline_ipywidgets import interactive_pipeline_files
    button_executor = interactive_pipeline_files(curr_active_pipeline)

    """
    

    # Define the set of buttons:
    button_defns = [("Output Folder", lambda _: reveal_in_system_file_manager(curr_active_pipeline.get_output_path())),
            ("global pickle", lambda _: reveal_in_system_file_manager(curr_active_pipeline.global_computation_results_pickle_path)),
            ("pipeline pickle", lambda _: reveal_in_system_file_manager(curr_active_pipeline.pickle_path)),
            (".h5 export", lambda _: reveal_in_system_file_manager(curr_active_pipeline.h5_export_path)),
            ("TEST - Dialog", lambda _: try_save_pickle_as(curr_active_pipeline.global_computation_results_pickle_path)),
            ("Save Pipeline", lambda _: PipelineJupyterHelpers.perform_pipeline_save(curr_active_pipeline)),
            # ("ViTables .h5 export", lambda _: reveal_in_system_file_manager(curr_active_pipeline.h5_export_path))
        ]
        
    outman = curr_active_pipeline.get_output_manager()
    figure_output_path = outman.get_figure_save_file_path(curr_active_pipeline.get_session_context(), make_folder_if_needed=False)
    if figure_output_path.exists():
        button_defns.append(("Figure Export Folder", lambda _: reveal_in_system_file_manager(figure_output_path)))

    # Create and display the button
    button_executor = JupyterButtonRowWidget.init_from_button_defns(button_defns=button_defns, defer_display=True)
    # updating_button_executor = JupyterButtonRowWidget.init_from_button_defns(button_defns=updating_button_defns, defer_display=True)

    # combined_button_executor = widgets.VBox((widgets.HBox(button_executor.button_list), widgets.HBox(updating_button_executor.button_list)))
    # combined_button_executor = widgets.VBox((button_executor.button_list, updating_button_executor.button_list))
    # return display(combined_button_executor)


    # ## New Method, need to convert
    # btn_layout = widgets.Layout(width='auto', height='40px') #set width and height
    # default_kwargs = dict(display='flex', flex_flow='column', align_items='stretch', layout=btn_layout)

    # _out_row = JupyterButtonRowWidget.init_from_button_defns(button_defns=[
    # 	("Documentation Folder", lambda _: reveal_in_system_file_manager(self.doc_output_parent_folder), default_kwargs),
    # 	("Generated Documentation", lambda _: self.reveal_output_files_in_system_file_manager(), default_kwargs),
    # 	])

    # _out_row_html = JupyterButtonRowWidget.init_from_button_defns(button_defns=[
    # 		("Open generated .html Documentation", lambda _: open_file_with_system_default(str(self.output_html_file.resolve())), default_kwargs),
    # 		("Reveal Generated .html Documentation", lambda _: reveal_in_system_file_manager(self.output_html_file), default_kwargs),
    # 	])

    # _out_row_md = JupyterButtonRowWidget.init_from_button_defns(button_defns=[
    # 		("Open generated .md Documentation", lambda _: open_file_with_system_default(str(self.output_md_file.resolve())), default_kwargs),
    # 		("Reveal Generated .md Documentation", lambda _: reveal_in_system_file_manager(self.output_md_file), default_kwargs),
    # 	])

    # return widgets.VBox([_out_row.root_widget,
    # 	_out_row_html.root_widget,
    # 	_out_row_md.root_widget,
    # ])



    return button_executor





# def interactive_pipeline_figure_widget(curr_active_pipeline):
# 	import matplotlib
# 	# configure backend here
# 	matplotlib.use('Qt5Agg')
# 	# backend_qt5agg
# 	# %matplotlib qt5
# 	# %matplotlib qt
# 	# matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!

# 	import matplotlib as mpl
# 	import matplotlib.pyplot as plt
# 	_bak_rcParams = mpl.rcParams.copy()
# 	# mpl.rcParams['toolbar'] = 'None' # disable toolbars

# 	# Showing
# 	restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
# 	from pyphoplacecellanalysis.External.pyqtgraph import QtWidgets, QtCore, QtGui
# 	from pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.LauncherWidget.LauncherWidget import LauncherWidget

# 	widget = LauncherWidget()
# 	treeWidget = widget.mainTreeWidget # QTreeWidget
# 	widget.build_for_pipeline(curr_active_pipeline=curr_active_pipeline)
# 	widget.show()

import panel as pn
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline, PipelineSavingScheme # used in perform_pipeline_save

@define(slots=False)
class PipelinePickleFileSelectorWidget:
    """ Allows the user to interactively choose between multiple pickles in the session directory to load
    
    
    Usage:
    
    from pyphoplacecellanalysis.GUI.IPyWidgets.pipeline_ipywidgets import PipelinePickleFileSelectorWidget
    widget = PipelinePickleFileSelectorWidget(directory=basedir)

    # Display the widget
    widget.local_file_browser_widget.servable()
    widget.global_file_browser_widget.servable()

    # OUTPUTS: widget, widget.active_local_pkl, widget.active_global_pkl

    """
    directory: Path = field()
    selected_local_pkl_files: List[Path] = field(default=Factory(list))
    selected_global_pkl_files: List[Path] = field(default=Factory(list))
    # Add new fields for the callback functions
    on_load_callback: Optional[Callable] = field(default=None)
    on_save_callback: Optional[Callable] = field(default=None)
    on_compute_callback: Optional[Callable] = field(default=None)
    on_get_global_variable_callback: Optional[Callable] = field(default=None)
    on_update_global_variable_callback: Optional[Callable] = field(default=None)
    
    local_file_browser_widget = field(init=False)
    global_file_browser_widget = field(init=False)
    layout: pn.Column = field(init=False)
    debug_print: bool = field(default=False)


    @property
    def active_local_pkl(self) -> Optional[Path]:
        """The active_local_pkl property."""
        if len(self.selected_local_pkl_files) < 1:
            return None
        else:
            return Path(self.selected_local_pkl_files[0])
        
    @property
    def active_global_pkl(self) -> Optional[Path]:
        """The active_local_pkl property."""
        if len(self.selected_global_pkl_files) < 1:
            return None
        else:
            return Path(self.selected_global_pkl_files[0])


    @property
    def is_load_button_disabled(self) -> bool:
        """The is_load_button_disabled property."""
        return (self.on_load_callback is None) or (self.on_get_global_variable_callback is None) or (self.on_update_global_variable_callback is None) or (self.active_local_pkl is None)


    @property
    def is_save_button_disabled(self) -> bool:
        """The is_save_button_enabled property."""
        return (self.on_save_callback is None) or (self.on_get_global_variable_callback is None)
    
    @property 
    def is_compute_button_disabled(self) -> bool:
        """The is_compute_button_disabled property."""
        return (self.on_compute_callback is None) or (self.on_get_global_variable_callback is None)




    @property
    def is_local_file_names_list_empty(self) -> bool:
        """The discovered local filenames."""
        return not ((len(self.local_file_browser_widget._data) > 0) and ('File Name' in self.local_file_browser_widget._data))


    @property
    def is_global_file_names_list_empty(self) -> bool:
        """The discovered global filenames."""
        return not ((len(self.global_file_browser_widget._data) > 0) and ('File Name' in self.global_file_browser_widget._data))



    @property
    def active_local_file_names_list(self) -> List[str]:
        """The discovered local filenames."""
        if (len(self.local_file_browser_widget._data) > 0) and ('File Name' in self.local_file_browser_widget._data):
            return self.local_file_browser_widget._data['File Name'].tolist()
        else:
            return []
        
    @property
    def active_global_file_names_list(self) -> List[str]:
        """The discovered global filenames."""
        if (len(self.global_file_browser_widget._data) > 0) and ('File Name' in self.global_file_browser_widget._data):
            return self.global_file_browser_widget._data['File Name'].tolist()
        else:
            return []

    def try_extract_custom_suffix(self) -> Optional[str]:
        """ uses the local pkl first 
        
        custom_suffix: str = widget.try_extract_custom_suffix()
        
        """
        if self.active_local_pkl is None:
            return None
        else:
            proposed_load_pkl_path = self.active_local_pkl.resolve()
            ## infer the `custom_suffix`
            basename: str = proposed_load_pkl_path.stem # 'loadedSessPickle_withNormalComputedReplays-qclu_[1, 2]-frateThresh_5.0'
            # custom_suffix
            pickle_basename_part: str = 'loadedSessPickle' 
            custom_suffix: str = basename.removeprefix(pickle_basename_part).removesuffix(pickle_basename_part) # '_withNormalComputedReplays-qclu_[1, 2]-frateThresh_5.0'
            if self.debug_print:
                print(f'custom_suffix: "{custom_suffix}"')
            return custom_suffix
    

    def on_selected_local_sess_pkl_files_changed(self, selected_df: pd.DataFrame):
        """ captures: file_table, on_selected_files_changed
        """
        if self.debug_print:
            print(f"on_selected_local_sess_pkl_files_changed(selected_df: {selected_df})")
        full_paths = selected_df['File Path'].to_list()
        if self.debug_print:
            print(f'\tfull_paths: {full_paths}')
        self.selected_local_pkl_files = full_paths
        self._update_load_save_button_disabled_state()
        

    def on_selected_global_computation_result_pkl_files_changed(self, selected_df: pd.DataFrame):
        """ captures: file_table, on_selected_files_changed
        """
        if self.debug_print:
            print(f"on_selected_global_computation_result_pkl_files_changed(selected_df: {selected_df})")
        full_paths = selected_df['File Path'].to_list()
        if self.debug_print:
            print(f'\tfull_paths: {full_paths}')
        self.selected_global_pkl_files = full_paths
        self._update_load_save_button_disabled_state()


    def __attrs_post_init__(self):
        # Create the file browser widget
        # file_browser_widget = create_file_browser(directory, patterns, page_size=10, widget_height=400, on_selected_files_changed_fn=on_selected_files_changed)
        self.local_file_browser_widget = create_file_browser(self.directory, ['*loadedSessPickle*.pkl'], page_size=10, widget_height=400, selectable=1, on_selected_files_changed_fn=self.on_selected_local_sess_pkl_files_changed)
        self.global_file_browser_widget = create_file_browser(self.directory, ['output/*global_computation_results*.pkl'], page_size=10, widget_height=400, selectable=1, on_selected_files_changed_fn=self.on_selected_global_computation_result_pkl_files_changed)

        # Create Load/Save buttons
        # self.load_button = widgets.Button(description="Load")
        # self.save_button = widgets.Button(description="Save")

        self.load_button = pn.widgets.Button(name='Load', button_type='primary')
        self.save_button = pn.widgets.Button(name='Save', button_type='success')
        self.compute_button = pn.widgets.Button(name='Compute', button_type='warning')  # Add compute button
        
        self.load_button.disabled = self.is_load_button_disabled
        self.save_button.disabled = self.is_save_button_disabled
        self.compute_button.disabled = self.is_compute_button_disabled  # Add compute button state
        
        self.load_button.on_click(self._handle_load_click)
        self.save_button.on_click(self._handle_save_click)
        self.compute_button.on_click(self._handle_compute_click)  # Add compute handler
        
        # Update layout to include compute button
        self.layout = pn.Column(self.local_file_browser_widget, self.global_file_browser_widget, 
                            pn.Row(self.save_button, self.load_button, self.compute_button))
        

    def _update_load_save_button_disabled_state(self):
        """ updates the .disabled property for the two action buttons """
        self.load_button.disabled = self.is_load_button_disabled
        self.save_button.disabled = self.is_save_button_disabled
        self.compute_button.disabled = self.is_compute_button_disabled


    def _handle_load_click(self, event):
        print(f'\t._handle_load_click(event: {event})')
        if self.on_load_callback is not None:
            # self.on_load_callback(self.active_local_pkl, self.active_global_pkl)
            self.on_load_callback()

    def _handle_save_click(self, event):
        print(f'\t._handle_save_click(event: {event})')
        if self.on_save_callback is not None:
            # self.on_save_callback(self.active_local_pkl, self.active_global_pkl)
            self.on_save_callback()
            
    def _handle_compute_click(self, event):
        print(f'\t._handle_compute_click(event: {event})')
        if self.on_compute_callback is not None:
            self.on_compute_callback()


    def servable(self, **kwargs):
        return self.layout.servable(**kwargs)


    ## Main load/save functions
    @function_attributes(short_name=None, tags=['working', 'ui', 'interactive'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-17 20:34', related_items=['PipelinePickleFileSelectorWidget', 'on_load_global'])
    def on_load_local(self, global_data_root_parent_path: Path, active_data_mode_name: str, basedir: Path, saving_mode, force_reload: bool):
        """ Loads custom pipeline pickles that were saved out via `custom_save_filepaths['pipeline_pkl'] = curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE, active_pickle_filename=custom_save_filenames['pipeline_pkl'])`

            # INPUTS: global_data_root_parent_path, active_data_mode_name, basedir, saving_mode, force_reload, custom_save_filenames
            custom_suffix: str = '_withNewKamranExportedReplays'

            custom_suffix: str = '_withNewComputedReplays'
            custom_suffix: str = '_withNewComputedReplays-qclu_[1, 2]-frateThresh_5.0'

            custom_save_filenames = {
                'pipeline_pkl':f'loadedSessPickle{custom_suffix}.pkl',
                'global_computation_pkl':f"global_computation_results{custom_suffix}.pkl",
                'pipeline_h5':f'pipeline{custom_suffix}.h5',
            }
            print(f'custom_save_filenames: {custom_save_filenames}')
            custom_save_filepaths = {k:v for k, v in custom_save_filenames.items()}

            # ==================================================================================================================== #
            # PIPELINE LOADING                                                                                                     #
            # ==================================================================================================================== #
            # load the custom saved outputs
            active_pickle_filename = custom_save_filenames['pipeline_pkl'] # 'loadedSessPickle_withParameters.pkl'
            print(f'active_pickle_filename: "{active_pickle_filename}"')
            # assert active_pickle_filename.exists()
            active_session_h5_filename = custom_save_filenames['pipeline_h5'] # 'pipeline_withParameters.h5'
            print(f'active_session_h5_filename: "{active_session_h5_filename}"')

            ==================================================================================================================== #
            Load Pipeline                                                                                                        #
            ==================================================================================================================== #
            # DO NOT allow recompute if the file doesn't exist!!
            Computing loaded session pickle file results : "W:/Data/KDIBA/gor01/two/2006-6-07_16-40-19/loadedSessPickle_withNewComputedReplays.pkl"... done.
            Failure loading W:\Data\KDIBA\gor01\two\2006-6-07_16-40-19\loadedSessPickle_withNewComputedReplays.pkl.
            proposed_load_pkl_path = basedir.joinpath(active_pickle_filename).resolve()




        Usage:

            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import on_load_local, on_load_global


            curr_active_pipeline, custom_suffix, proposed_load_pkl_path = on_load_local(active_session_pickle_file_widget=active_session_pickle_file_widget, global_data_root_parent_path=global_data_root_parent_path, active_data_mode_name=active_data_mode_name, basedir=basedir, saving_mode=saving_mode, force_reload=force_reload) 
            curr_active_pipeline = on_load_global(active_session_pickle_file_widget=active_session_pickle_file_widget, curr_active_pipeline=curr_active_pipeline, basedir=basedir, extended_computations_include_includelist=extended_computations_include_includelist, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist,
                                                skip_global_load=False, force_reload=False, override_global_computation_results_pickle_path=active_session_pickle_file_widget.active_global_pkl)

                                                

        """
        ## INPUTS: widget.active_global_pkl, widget.active_global_pkl
        # from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows
        from pyphoplacecellanalysis.General.Batch.runBatch import BatchSessionCompletionHandler # for `post_compute_validate(...
        from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline # get_neuron_identities
        from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_load_session
        from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme # used in perform_pipeline_save
        from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows

        proposed_load_pkl_path = self.active_local_pkl.resolve()
        Assert.path_exists(proposed_load_pkl_path)
        
        custom_suffix: str = self.try_extract_custom_suffix()
        print(f'custom_suffix: "{custom_suffix}"')

        ## OUTPUTS: custom_suffix, proposed_load_pkl_path, (override_global_computation_results_pickle_path, skip_global_load)


        ## INPUTS: proposed_load_pkl_path
        assert proposed_load_pkl_path.exists(), f"for a saved custom the file must exist!"

        epoch_name_includelist=None
        active_computation_functions_name_includelist=['lap_direction_determination', 'pf_computation','firing_rate_trends', 'position_decoding']

        with set_posix_windows():
            curr_active_pipeline: NeuropyPipeline = batch_load_session(global_data_root_parent_path, active_data_mode_name, basedir, epoch_name_includelist=epoch_name_includelist,
                                                    computation_functions_name_includelist=active_computation_functions_name_includelist,
                                                    saving_mode=saving_mode, force_reload=force_reload,
                                                    skip_extended_batch_computations=True, debug_print=False, fail_on_exception=True, active_pickle_filename=proposed_load_pkl_path) # , active_pickle_filename = 'loadedSessPickle_withParameters.pkl'

        ## Post Compute Validate 2023-05-16:
        was_updated = BatchSessionCompletionHandler.post_compute_validate(curr_active_pipeline) ## TODO: need to potentially re-save if was_updated. This will fail because constained versions not ran yet.
        if was_updated:
            print(f'was_updated: {was_updated}')
            try:
                if saving_mode == PipelineSavingScheme.SKIP_SAVING:
                    print(f'WARNING: PipelineSavingScheme.SKIP_SAVING but need to save post_compute_validate changes!!')
                else:
                    curr_active_pipeline.save_pipeline(saving_mode=saving_mode)
            except Exception as e:
                ## TODO: catch/log saving error and indicate that it isn't saved.
                exception_info = sys.exc_info()
                e = CapturedException(e, exception_info)
                print(f'ERROR RE-SAVING PIPELINE after update. error: {e}')

        print(f'Pipeline loaded from custom pickle!!')
        ## OUTPUT: curr_active_pipeline
        print(f'''# ==================================================================================================================== #
        # on_load_local -- COMPLETE -- 
        # ==================================================================================================================== #''')
        return curr_active_pipeline, custom_suffix, proposed_load_pkl_path


    @function_attributes(short_name=None, tags=['working', 'ui', 'interactive', 'load', 'global'], input_requires=[], output_provides=[], uses=['batch_evaluate_required_computations', 'curr_active_pipeline.load_pickled_global_computation_results'], used_by=[], creation_date='2025-01-17 17:04', related_items=['PipelinePickleFileSelectorWidget', 'on_load_local'])
    def on_load_global(self, curr_active_pipeline, basedir: Path, extended_computations_include_includelist: List[str], force_recompute_override_computations_includelist: List[str]=[],
                        skip_global_load: bool = True, saving_mode: PipelineSavingScheme=PipelineSavingScheme.SKIP_SAVING, force_reload: bool = False, override_global_computation_results_pickle_path: Path=None):
        """

        curr_active_pipeline = on_load_global(active_session_pickle_file_widget, curr_active_pipeline, extended_computations_include_includelist: List[str], skip_global_load: bool = True, force_reload: bool = False, override_global_computation_results_pickle_path: Path=None)


        """
        # from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
        from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_extended_computations, batch_evaluate_required_computations
        
        # from pyphoplacecellanalysis.GUI.IPyWidgets.pipeline_ipywidgets import PipelineJupyterHelpers, CustomProcessingPhases
        from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows

        # ==================================================================================================================== #
        # Global computations loading:                                                                                            #
        # ==================================================================================================================== #
        # Loads saved global computations that were saved out via: `custom_save_filepaths['global_computation_pkl'] = curr_active_pipeline.save_global_computation_results(override_global_pickle_filename=custom_save_filenames['global_computation_pkl'])`
        ## INPUTS: custom_save_filenames
        ## INPUTS: curr_active_pipeline, override_global_computation_results_pickle_path, extended_computations_include_includelist

        if self.active_global_pkl is None:
            skip_global_load = True
            override_global_computation_results_pickle_path = None
        else:
            skip_global_load = False
            override_global_computation_results_pickle_path = self.active_global_pkl.resolve()
            Assert.path_exists(override_global_computation_results_pickle_path)
            override_global_computation_results_pickle_path


        # override_global_computation_results_pickle_path = None
        # override_global_computation_results_pickle_path = custom_save_filenames['global_computation_pkl']
        print(f'override_global_computation_results_pickle_path: "{override_global_computation_results_pickle_path}"')

        # Pre-load ___________________________________________________________________________________________________________ #
        force_recompute_global = force_reload
        needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=force_recompute_global, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        print(f'Pre-load global computations: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')
        # valid_computed_results_output_list

        # Try Unpickling Global Computations to update pipeline ______________________________________________________________ #
        if (not force_reload) and (not skip_global_load): # not just force_reload, needs to recompute whenever the computation fails.
            try:
                # INPUTS: override_global_computation_results_pickle_path
                with set_posix_windows():
                    sucessfully_updated_keys, successfully_loaded_keys = curr_active_pipeline.load_pickled_global_computation_results(override_global_computation_results_pickle_path=override_global_computation_results_pickle_path,
                                                                                                    allow_overwrite_existing=True, allow_overwrite_existing_allow_keys=extended_computations_include_includelist, ) # is new
                    print(f'sucessfully_updated_keys: {sucessfully_updated_keys}\nsuccessfully_loaded_keys: {successfully_loaded_keys}')
                    did_any_paths_change: bool = curr_active_pipeline.post_load_fixup_sess_basedirs(updated_session_basepath=deepcopy(basedir)) ## use INPUT: basedir

            except FileNotFoundError as e:
                exception_info = sys.exc_info()
                e = CapturedException(e, exception_info)
                print(f'cannot load global results because pickle file does not exist! Maybe it has never been created? {e}')
            except Exception as e:
                exception_info = sys.exc_info()
                e = CapturedException(e, exception_info)
                print(f'Unhandled exception: cannot load global results: {e}')
                raise

        # Post-Load __________________________________________________________________________________________________________ #
        force_recompute_global = force_reload
        needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=force_recompute_global, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        print(f'Post-load global computations: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')

        ## fixup missing paths
        # self.basepath: WindowsPath('/nfs/turbo/umms-kdiba/KDIBA/gor01/one/2006-6-09_1-22-43')

        ## INPUTS: basedir
        did_any_paths_change: bool = curr_active_pipeline.post_load_fixup_sess_basedirs(updated_session_basepath=deepcopy(basedir)) ## use INPUT: basedir

        # Compute ____________________________________________________________________________________________________________ #
        curr_active_pipeline.reload_default_computation_functions()
        force_recompute_global = force_reload
        # force_recompute_global = True
        newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=force_recompute_global, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        if (len(newly_computed_values) > 0):
            print(f'newly_computed_values: {newly_computed_values}.')
            if (saving_mode.value != 'skip_saving'):
                print(f'Saving global results...')
                try:
                    # curr_active_pipeline.global_computation_results.persist_time = datetime.now()
                    # Try to write out the global computation function results:
                    curr_active_pipeline.save_global_computation_results()
                except Exception as e:
                    exception_info = sys.exc_info()
                    e = CapturedException(e, exception_info)
                    print(f'\n\n!!WARNING!!: saving the global results threw the exception: {e}')
                    print(f'\tthe global results are currently unsaved! proceed with caution and save as soon as you can!\n\n\n')
            else:
                print(f'\n\n!!WARNING!!: changes to global results have been made but they will not be saved since saving_mode.value == "skip_saving"')
                print(f'\tthe global results are currently unsaved! proceed with caution and save as soon as you can!\n\n\n')
        else:
            print(f'no changes in global results.')

        # Post-compute _______________________________________________________________________________________________________ #
        # Post-hoc verification that the computations worked and that the validators reflect that. The list should be empty now.
        needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=False, force_recompute_override_computations_includelist=[], debug_print=True)
        print(f'Post-compute validation: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')


        # Post-Load __________________________________________________________________________________________________________ #
        force_recompute_global = force_reload
        needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=force_recompute_global, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        print(f'Post-load global computations: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')


        print(f'''# ==================================================================================================================== #
        # on_load_global -- COMPLETE -- 
        # ==================================================================================================================== #''')
        return curr_active_pipeline


    def _build_load_save_callbacks(self, global_data_root_parent_path: Path, active_data_mode_name: str, basedir: Path, saving_mode, force_reload: bool,
                                extended_computations_include_includelist: List[str], force_recompute_override_computations_includelist: Optional[List[str]]=None):
        """ Called to provide the widget with everything needed to actually load the pipeline, which is required before the "Load" button is enabled.
        
        
        _subfn_load, _subfn_save, _subfn_compute = active_session_pickle_file_widget._build_load_save_callbacks(global_data_root_parent_path=global_data_root_parent_path, active_data_mode_name=active_data_mode_name, basedir=basedir, saving_mode=saving_mode, force_reload=force_reload,
															 extended_computations_include_includelist=extended_computations_include_includelist, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist)

        """
        from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_extended_computations, batch_evaluate_required_computations
        
        def _subfn_load():
            """ captures: everything in calling context!
            Modifies in workspace: ['curr_active_pipeline', 'custom_suffix', 'proposed_load_pkl_path']
            """
            curr_active_pipeline, custom_suffix, proposed_load_pkl_path = self.on_load_local(global_data_root_parent_path=global_data_root_parent_path, active_data_mode_name=active_data_mode_name, basedir=basedir, saving_mode=saving_mode, force_reload=force_reload)
            curr_active_pipeline = self.on_load_global(curr_active_pipeline=curr_active_pipeline, basedir=basedir, extended_computations_include_includelist=extended_computations_include_includelist, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist,
                                        skip_global_load=False, force_reload=False, override_global_computation_results_pickle_path=self.active_global_pkl)
            
            update_global_variable_fn = self.on_update_global_variable_callback
            assert update_global_variable_fn is not None
            # Update the global variable
            update_global_variable_fn('curr_active_pipeline', curr_active_pipeline)
            update_global_variable_fn('custom_suffix', custom_suffix)
            update_global_variable_fn('proposed_load_pkl_path', proposed_load_pkl_path)
            

        def _subfn_save():
            """ captures: everything in calling context! """
            get_global_variable_fn = self.on_get_global_variable_callback
            assert get_global_variable_fn is not None
            # Get the pipeline
            curr_active_pipeline = get_global_variable_fn('curr_active_pipeline')
            assert curr_active_pipeline.pickle_path is not None, f"curr_active_pipeline.pickle_path is None! Must be set before knowing where to save to!"
            curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE, override_pickle_path=curr_active_pipeline.pickle_path, active_pickle_filename=curr_active_pipeline.pickle_path.name) #active_pickle_filename=
            assert curr_active_pipeline.global_computation_results_pickle_path is not None, f"curr_active_pipeline.global_computation_results_pickle_path is None! Must be set before knowing where to save to!"
            curr_active_pipeline.save_global_computation_results(override_global_pickle_path=curr_active_pipeline.global_computation_results_pickle_path)
            

        def _subfn_compute():
            """Performs computations in clean_run mode"""
            get_global_variable_fn = self.on_get_global_variable_callback
            assert get_global_variable_fn is not None
            curr_active_pipeline = get_global_variable_fn('curr_active_pipeline')
            
            # Reload computation functions and perform computations
            curr_active_pipeline.reload_default_computation_functions()
            newly_computed_values = batch_extended_computations(curr_active_pipeline, 
                                                            include_includelist=extended_computations_include_includelist,
                                                            include_global_functions=True,
                                                            fail_on_exception=False,
                                                            progress_print=True,
                                                            force_recompute=True,
                                                            force_recompute_override_computations_includelist=force_recompute_override_computations_includelist,
                                                            debug_print=False)
            
            if len(newly_computed_values) > 0:
                print(f'newly_computed_values: {newly_computed_values}')
                _subfn_save() ## call _subfn_save() to attempt to save out the results
                # curr_active_pipeline.save_global_computation_results()
                
        # ==================================================================================================================== #
        # BEGIN MAIN FUNCTION BODY                                                                                             #
        # ==================================================================================================================== #

        self.on_load_callback = _subfn_load
        self.on_save_callback = _subfn_save
        self.on_compute_callback = _subfn_compute
    
        ## Update button enable states:        
        self._update_load_save_button_disabled_state()
        
        return _subfn_load, _subfn_save, _subfn_compute
    


    @function_attributes(short_name=None, tags=['matching', 'pipeline', 'file'], input_requires=[], output_provides=[], uses=[], used_by=['try_select_first_valid_files'], creation_date='2025-02-11 02:40', related_items=[])
    def try_determine_matching_global_file(self, local_file_name: str) -> Tuple[str, Optional[int]]:
        """ try to find corresponding global file: for the specified local_file_name 
        """
        selected_context_suffix_only: str = local_file_name.removeprefix('loadedSessPickle') # '_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_4.0'
        ## try to find corresponding global file:
        corresponding_global_file_name: str = f'global_computation_results{selected_context_suffix_only}.pkl'
        corresponding_global_section_index = None
        try:
            corresponding_global_section_index = self.active_global_file_names_list.index(corresponding_global_file_name)
            if corresponding_global_section_index == -1:
                corresponding_global_section_index = None
            # else:
            #     # return corresponding_global_section_index
            #     pass
        except ValueError as e:
            # index not found
            corresponding_global_section_index = None
            # return None
        except Exception as e:
            raise e
        # corresponding_global_section_indicies = [self.active_global_file_names_list.index(corresponding_global_file_name)]
        return corresponding_global_file_name, corresponding_global_section_index
    
    @function_attributes(short_name=None, tags=['select-first', 'startup', 'select', 'gui'], input_requires=[], output_provides=[], uses=['try_determine_matching_global_file'], used_by=[], creation_date='2025-02-11 02:40', related_items=[])
    def try_select_first_valid_files(self) -> bool:
        """
        try selecting the first
        """
        if len(self.active_local_file_names_list) < 1:
            print(f'have 0 local files, cannot select first.')
            return False
        
        ## otherwise it's safe to set the selection
        self.local_file_browser_widget.selection = [0]
        # self.global_file_browser_widget.selection = [0]
        first_selected_local_path: Path = Path(self.selected_local_pkl_files[0])
        selected_local_file_name: str = first_selected_local_path.stem # 'loadedSessPickle_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_4.0'
        ## try to find corresponding global file:
        corresponding_global_file_name, corresponding_global_section_index = self.try_determine_matching_global_file(local_file_name=selected_local_file_name)
        if corresponding_global_section_index is None:
            # failed to find
            self.global_file_browser_widget.selection = [] ## clear global selection
            return False # failed
        else:
            corresponding_global_section_indicies = [corresponding_global_section_index] # single list

        ## set selection to corresponding
        self.global_file_browser_widget.selection = corresponding_global_section_indicies
        return True



@function_attributes(short_name=None, tags=['ipywidgets', 'ipython', 'jupyterwidgets', 'interactive'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-09-26 12:01', related_items=[])
def interactive_pipeline_widget(curr_active_pipeline):
    """ 
    Usage:
        import ipywidgets as widgets
        from IPython.display import display
        from pyphoplacecellanalysis.GUI.IPyWidgets.pipeline_ipywidgets import interactive_pipeline_widget, fullwidth_path_widget, interactive_pipeline_files


        _out_widget = interactive_pipeline_widget(curr_active_pipeline=curr_active_pipeline)
        display(_out_widget)

    """
    from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update # needed for toggle_figure_displaying_function
    
    session_path = str(curr_active_pipeline.get_output_path())
    _session_path_widget = fullwidth_path_widget(a_path=session_path, file_name_label="session path:")
    _button_executor = interactive_pipeline_files(curr_active_pipeline, defer_display=True)

    updating_button_defns = [("Reload display functions...", lambda _: curr_active_pipeline.reload_default_display_functions()),
            ("Reload computation functions...", lambda _: curr_active_pipeline.reload_default_computation_functions()),
            # ("pipeline pickle", lambda _: reveal_in_system_file_manager(curr_active_pipeline.pickle_path)),
            # ("Try .h5 export", lambda _: curr_active_pipeline.export_pipeline_to_h5('output/2023_09_19-pipeline_test.h5') reveal_in_system_file_manager(curr_active_pipeline.h5_export_path)),
            # ("TEST - Dialog", lambda _: try_save_pickle_as(curr_active_pipeline.global_computation_results_pickle_path)),
            # ("ViTables .h5 export", lambda _: reveal_in_system_file_manager(curr_active_pipeline.h5_export_path))
        ]
    updating_button_executor = JupyterButtonRowWidget.init_from_button_defns(button_defns=updating_button_defns, defer_display=True)
    # combined_button_executor = widgets.VBox((widgets.HBox(button_executor.button_list), widgets.HBox(updating_button_executor.button_list)))

    
    def toggle_figure_displaying_function(change):
        """ toggles between showing matplotlib figures or rendering them in background 
        captures: matplotlib_configuration_update
        """
        if change['new']:
            print("Figure windows enabled")
            # Showing
            # configure backend here
            matplotlib.use('Qt5Agg')
            # backend_qt5agg
            # %matplotlib qt5
            # %matplotlib qt
            # matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            _bak_rcParams = mpl.rcParams.copy()
            _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
        else:
            print("Figure windows disabled (file only)")
            # Showing
            _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=False, backend='AGG')

    figure_display_toggle_button = widgets.ToggleButton(description="Figures Displaying", value=True)
    figure_display_toggle_button.observe(toggle_figure_displaying_function, names='value')
    

    # _out_widget = widgets.VBox([_session_path_widget, _button_executor.root_widget])
    _out_widget = widgets.VBox([_session_path_widget, _button_executor.root_widget, updating_button_executor.root_widget, figure_display_toggle_button])

    return _out_widget        
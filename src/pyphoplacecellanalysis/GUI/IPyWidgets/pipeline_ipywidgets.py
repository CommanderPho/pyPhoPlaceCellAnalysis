import sys
import ipywidgets as widgets
from IPython.display import display
import matplotlib

from pathlib import Path

# from silx.gui.dialog.ImageFileDialog import ImageFileDialog
# import silx.io

from pyphocorehelpers.gui.Jupyter.JupyterButtonRowWidget import build_fn_bound_buttons, JupyterButtonRowWidget, JupyterButtonColumnWidget
from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager
from pyphocorehelpers.Filesystem.path_helpers import open_file_with_system_default


from pyphocorehelpers.exception_helpers import CapturedException
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore, QtWidgets
# from pyphoplacecellanalysis.External.pyqtgraph.parametertree.parameterTypes.file import popupFilePicker
from pyphoplacecellanalysis.External.pyqtgraph.widgets.FileDialog import FileDialog
from pyphocorehelpers.gui.Jupyter.simple_widgets import fullwidth_path_widget


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



class PipelineJupyterHelpers:
    

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
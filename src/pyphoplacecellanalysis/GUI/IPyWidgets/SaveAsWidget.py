from copy import deepcopy
import sys
from enum import Enum
import panel as pn
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


from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline, PipelineSavingScheme # used in perform_pipeline_save


@define(slots=False)
class PipelineBackupWidget:
    """Allows saving backups of the currently loaded pipeline with custom filenames.
    Includes progress indication and detailed error feedback.
    
    Usage:
	    from pyphoplacecellanalysis.GUI.IPyWidgets.SaveAsWidget import PipelineBackupWidget
        backup_widget = PipelineBackupWidget(curr_active_pipeline)
        backup_widget.servable()
    """
    curr_active_pipeline: NeuropyPipeline = field()
    on_get_global_variable_callback: Optional[Callable] = field(default=None)
    debug_print: bool = field(default=False)
    
    filename_input: pn.widgets.TextInput = field(init=False)
    save_button: pn.widgets.Button = field(init=False)
    progress: pn.widgets.Progress = field(init=False)
    status_indicator: pn.pane.Markdown = field(init=False)
    layout: pn.Column = field(init=False)

    def __attrs_post_init__(self):
        curr_pipeline_path = Path(self.curr_active_pipeline.pickle_path)
        suggested_backup = f"{curr_pipeline_path.stem}_backup{curr_pipeline_path.suffix}"
        
        self.filename_input = pn.widgets.TextInput(
            name='Backup Filename',
            value=suggested_backup,
            placeholder='Enter backup filename...'
        )
        
        self.save_button = pn.widgets.Button(
            name='Save Backup',
            button_type='success',
            icon='save'
        )
        self.save_button.on_click(self._handle_save_backup)
        
        # Add progress bar and status indicator
        self.progress = pn.widgets.Progress(
            name='Save Progress',
            value=0,
            bar_color='primary',
            visible=False
        )
        
        self.status_indicator = pn.pane.Markdown("")
        
        self.layout = pn.Column(
            pn.pane.Markdown("### Pipeline Backup"),
            self.filename_input,
            self.save_button,
            self.progress,
            self.status_indicator
        )

    def _update_status(self, message: str, is_error: bool = False, is_success: bool = False):
        """Updates the status display with appropriate styling"""
        if is_error:
            self.status_indicator.object = f'<div style="color: red;">❌ {message}</div>'
        elif is_success:
            self.status_indicator.object = f'<div style="color: green;">✓ {message}</div>'
        else:
            self.status_indicator.object = message

    def _handle_save_backup(self, event):
        """Handles the backup save operation with progress tracking and error handling"""
        if self.debug_print:
            print(f'_handle_save_backup(event: {event})')
            
        backup_filename = self.filename_input.value
        self.progress.visible = True
        self.progress.value = 0
        self._update_status("Starting backup...")
        
        try:
            # Setup paths
            curr_path = Path(self.curr_active_pipeline.pickle_path)
            backup_path = curr_path.parent.joinpath(backup_filename)
            global_backup_filename = f"global_computation_results_{Path(backup_filename).stem}.pkl"
            global_backup_path = curr_path.parent.joinpath('output', global_backup_filename)
            
            # Ensure output directory exists
            global_backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.progress.value = 25
            self._update_status("Saving pipeline...")
            
            # Save pipeline with safeSaveData
            try:
                self.curr_active_pipeline.save_pipeline(
                    saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE,
                    override_pickle_path=backup_path
                )
                self.progress.value = 50
                self._update_status("Pipeline saved, saving global results...")
                
                # Save global results
                self.curr_active_pipeline.save_global_computation_results(
                    override_global_pickle_path=global_backup_path
                )
                self.progress.value = 100
                
                success_message = f'Successfully saved pipeline backup:<br/>Pipeline: {backup_path}<br/>Global Results: {global_backup_path}'
                self._update_status(success_message, is_success=True)
                
            except Exception as e:
                exception_info = sys.exc_info()
                err = CapturedException(e, exception_info)
                error_message = f'Error during save:<br/>{str(err)}'
                self._update_status(error_message, is_error=True)
                raise
                
        finally:
            # Hide progress after 2 seconds
            def hide_progress():
                self.progress.visible = False
            pn.state.onload(lambda: pn.state.add_timeout(2000, hide_progress))

    def servable(self, **kwargs):
        """Display the widget"""
        return self.layout.servable(**kwargs)

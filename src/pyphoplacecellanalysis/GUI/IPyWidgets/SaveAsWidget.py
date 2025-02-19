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
    curr_active_pipeline: NeuropyPipeline = field()
    on_get_global_variable_callback: Optional[Callable] = field(default=None)
    debug_print: bool = field(default=False)
    
    prefix_input: pn.widgets.TextInput = field(init=False)
    filename_input: pn.widgets.TextInput = field(init=False)
    suffix_input: pn.widgets.TextInput = field(init=False)
    filename_preview: pn.pane.Markdown = field(init=False)
    save_button: pn.widgets.Button = field(init=False)
    progress: pn.widgets.Progress = field(init=False)
    status_indicator: pn.pane.Markdown = field(init=False)
    layout: pn.Column = field(init=False)

    def __attrs_post_init__(self):
        curr_pipeline_path = Path(self.curr_active_pipeline.pickle_path)
        suggested_backup = f"{curr_pipeline_path.stem}_backup{curr_pipeline_path.suffix}"
        
        self.prefix_input = pn.widgets.TextInput(
            name='User Prefix',
            value='',
            placeholder='Optional prefix...',
            width=150
        )
        
        self.filename_input = pn.widgets.TextInput(
            name='Backup Filename',
            value=suggested_backup,
            placeholder='Enter backup filename...',
            width=800
        )
        
        self.suffix_input = pn.widgets.TextInput(
            name='User Suffix',
            value='',
            placeholder='Optional suffix...',
            width=150
        )
        
        # Create a row for the filename inputs
        filename_row = pn.Row(
            self.prefix_input,
            self.filename_input,
            self.suffix_input
        )
        
        # Add filename preview
        self.filename_preview = pn.pane.Markdown("", styles={
            'background': '#2a2a2a',  # Dark background
            'color': '#ffffff',       # White text
            'padding': '8px',
            'border-radius': '4px',   # Rounded corners
            'font-family': 'monospace'
        })
        self._update_filename_preview()
        
        # Add watchers for input changes
        self.prefix_input.param.watch(self._update_filename_preview, 'value')
        self.filename_input.param.watch(self._update_filename_preview, 'value')
        self.suffix_input.param.watch(self._update_filename_preview, 'value')
        
        # Rest of the widget setup remains the same
        self.save_button = pn.widgets.Button(
            name='Save Backup',
            button_type='success',
            icon='save'
        )
        self.save_button.on_click(self._handle_save_backup)
        
        self.progress = pn.widgets.Progress(
            name='Save Progress',
            value=0,
            bar_color='primary',
            visible=False
        )
        
        self.status_indicator = pn.pane.Markdown("")
        
        self.layout = pn.Column(
            pn.pane.Markdown("### Pipeline Backup"),
            filename_row,
            self.filename_preview,
            self.save_button,
            self.progress,
            self.status_indicator
        )

    def _update_filename_preview(self, *events):
        """Updates the preview of the final filename"""
        full_filename = self._get_full_backup_filename()
        self.filename_preview.object = f"<b>Final filename:</b> {full_filename}"

    def _get_full_backup_filename(self) -> str:
        """Combines prefix, filename, and suffix to create the full backup filename"""
        prefix = self.prefix_input.value.strip()
        main_name = self.filename_input.value.strip()
        suffix = self.suffix_input.value.strip()
        
        # Split the main filename into stem and extension
        path = Path(main_name)
        stem = path.stem
        ext = path.suffix
        
        # Combine parts with underscores if they exist
        full_stem = '_'.join(filter(None, [prefix, stem, suffix]))
        return full_stem + ext


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
            
        self.progress.visible = True
        self.progress.value = 0
        self._update_status("Starting backup...")
        
        try:
            # Use the pipeline's built-in custom save function
            user_prefix = self.prefix_input.value.strip()
            user_suffix = self.suffix_input.value.strip()
            
            self.progress.value = 25
            self._update_status("Saving pipeline with custom modifiers...")
            
            # Get the custom filenames and save
            user_custom_modified_filenames, did_save_success = self.curr_active_pipeline.try_save_pipeline_with_custom_user_modifiers(
                user_prefix=user_prefix,
                user_suffix=user_suffix,
                is_dryrun=False
            )
            
            self.progress.value = 100
            
            if did_save_success:
                success_message = f'Successfully saved pipeline backup:<br/>Files: {user_custom_modified_filenames}'
                self._update_status(success_message, is_success=True)
            else:
                self._update_status("Save operation failed", is_error=True)
                
        except Exception as e:
            exception_info = sys.exc_info()
            err = CapturedException(e, exception_info)
            error_message = f'Error during save:<br/>{str(err)}'
            self._update_status(error_message, is_error=True)
            raise
            
        finally:
            def hide_progress():
                self.progress.visible = False
            pn.state.onload(lambda: pn.state.add_timeout(2000, hide_progress))


    def servable(self, **kwargs):
        """Display the widget"""
        return self.layout.servable(**kwargs)
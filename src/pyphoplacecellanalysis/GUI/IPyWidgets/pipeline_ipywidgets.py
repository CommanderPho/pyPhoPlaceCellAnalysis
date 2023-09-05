import ipywidgets as widgets
from IPython.display import display
from pyphocorehelpers.gui.Jupyter.JupyterButtonRowWidget import JupyterButtonRowWidget
from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager


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
			# ("ViTables .h5 export", lambda _: reveal_in_system_file_manager(curr_active_pipeline.h5_export_path))
		]

	# Create and display the button
	button_executor = JupyterButtonRowWidget(button_defns=button_defns, defer_display=defer_display)
	return button_executor


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
		
	outman = curr_active_pipeline.get_output_manager()
	figure_output_path = outman.get_figure_save_file_path(curr_active_pipeline.get_session_context(), make_folder_if_needed=False)
	if figure_output_path.exists():
		button_defns.append(("Figure Export Folder", lambda _: reveal_in_system_file_manager(figure_output_path)))

	# Create and display the button
	button_executor = JupyterButtonRowWidget(button_defns=button_defns, defer_display=defer_display)
	return button_executor


def fullwidth_path_widget(a_path):
	left_label = widgets.Label("session path:", layout=widgets.Layout(width='auto'))
	right_label = widgets.Label(a_path, layout=widgets.Layout(width='auto', flex='1'))
	hbox = widgets.HBox([left_label, right_label], layout=widgets.Layout(display='flex'))
	return hbox


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





def interactive_pipeline_widget(curr_active_pipeline):
	""" 
	Usage:
		import ipywidgets as widgets
		from IPython.display import display
		from pyphoplacecellanalysis.GUI.IPyWidgets.pipeline_ipywidgets import interactive_pipeline_widget, fullwidth_path_widget, interactive_pipeline_files


		_out_widget = interactive_pipeline_widget(curr_active_pipeline=curr_active_pipeline)
		display(_out_widget)

	"""
	session_path = str(curr_active_pipeline.get_output_path())
	_session_path_widget = fullwidth_path_widget(a_path=session_path)
	_button_executor = interactive_pipeline_files(curr_active_pipeline, defer_display=True)
	_out_widget = widgets.VBox([_session_path_widget, _button_executor.root_widget])
	return _out_widget
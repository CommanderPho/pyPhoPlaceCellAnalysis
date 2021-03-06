# CreateNewConnectedWidgetMenuMixin
from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.BaseMenuProviderMixin import BaseMenuCommand
# GuiResources_rc


class CreateNewConnectedWidgetMenuMixin(object):
    """Adds a dynamically generated menubar to a QMainWindow for the purpose of connecting various separate windows
    
    Requirements:
        Implementor must be a QWidget class with:
            .window() property
            
    Main Functions:
        try_add_create_new_connected_widget_menu(...)
        try_remove_create_new_connected_widget_menu(...)
    
    Example:
        from pyphoplacecellanalysis.GUI.Qt.MainWindowWrapper import PhoBaseMainWindow

        curr_content_widget = spike_raster_window.window()
        curr_window = PhoBaseMainWindow(content_widget=curr_content_widget)
        menuCreateNewConnectedWidget, actions_dict = build_menu(curr_window)
        
        from pyphoplacecellanalysis.GUI.Qt.Mixins.CreateNewConnectedWidgetMenuMixin import CreateNewConnectedWidgetMenuMixin
        curr_window, menuCreateNewConnectedWidget, actions_dict = CreateNewConnectedWidgetMenuMixin.try_add_create_new_connected_widget_menu(spike_raster_window)

    
    """
    def build_create_new_connected_widget_menu(self):
        return CreateNewConnectedWidgetMenuMixin.try_add_create_new_connected_widget_menu(self)


    def remove_create_new_connected_widget_menu(self):
        """ Works to remove the menu created with menuCreateNewConnectedWidget, actions_dict = build_menu(curr_window) """
        return CreateNewConnectedWidgetMenuMixin.try_remove_create_new_connected_widget_menu(self)


    @classmethod
    def try_add_create_new_connected_widget_menu(cls, a_content_widget, curr_active_pipeline, active_config_name, display_output):
        """ 
        Usage:
            curr_window, menuCreateNewConnected, actions_dict = _try_add_create_new_connected_widget_menu(spike_raster_window)
        """
        curr_window = PhoMenuHelper.try_get_menu_window(a_content_widget)
        menuCreateNewConnected, actions_dict = cls._build_create_new_connected_widget_menu(curr_window)
        # Build and attach the menu commands by passing the references needed to perform the actions:
        cls.try_attach_action_commands(a_content_widget, curr_active_pipeline, active_config_name, display_output) # Attaches the commands
        return curr_window, menuCreateNewConnected, actions_dict

    @classmethod
    def try_remove_create_new_connected_widget_menu(cls, a_content_widget):
        """ Works to remove the menu created with menuCreateNewConnectedWidget, actions_dict = build_menu(curr_window) """
        curr_window = a_content_widget.window()
        curr_actions_dict = curr_window.ui.createNewConnectedWidgetMenuActionsDict
        curr_menubar = curr_window.menuBar()
        if curr_window.ui.menuCreateNewConnectedWidget is not None:
            # remove the menu's children:
            curr_window.ui.menuCreateNewConnectedWidget.clear() # remove children items

        # curr_menubar.removeAction(curr_actions_dict['actionMenuCreateNewConnectedWidget'])
        curr_menubar.removeAction(curr_actions_dict['actionMenuCreateNewConnectedWidget'])
        # curr_menubar.removeAction(curr_window.ui.actionMenuCreateNewConnectedWidget)
        
        # curr_window.ui.actionMenuCreateNewConnectedWidget.deleteLater()
        curr_window.ui.actionMenuCreateNewConnectedWidget = None # Null out the action:
        curr_window.ui.menuCreateNewConnectedWidget = None # Null out the reference to the menu item itself
        curr_window.ui.createNewConnectedWidgetMenuActionsDict = {} # Empty the dict of actions
            
    @classmethod
    def try_attach_action_commands(cls, spike_raster_window, curr_active_pipeline, active_config_name, display_output):
        ## Register the actual commands for each action:
        curr_window = PhoMenuHelper.try_get_menu_window(spike_raster_window)
        curr_window.ui.createNewConnectedWidgetMenuActionsDict['actionNewConnected3DRaster_PyQtGraph'].triggered.connect(CreateNewPyQtGraphPlotterCommand(spike_raster_window))
        curr_window.ui.createNewConnectedWidgetMenuActionsDict['actionNewConnected3DRaster_Vedo'].triggered.connect(CreateNewVedoPlotterCommand(spike_raster_window))
        curr_window.ui.createNewConnectedWidgetMenuActionsDict['actionNewConnectedDataExplorer_ipc'].triggered.connect(CreateNewDataExplorer_ipc_PlotterCommand(spike_raster_window, curr_active_pipeline, active_config_name, display_output))
        curr_window.ui.createNewConnectedWidgetMenuActionsDict['actionNewConnectedDataExplorer_ipspikes'].triggered.connect(CreateNewDataExplorer_ipspikes_PlotterCommand(spike_raster_window, curr_active_pipeline, active_config_name, display_output))

            
    @classmethod
    def _build_create_new_connected_widget_menu(cls, a_main_window, debug_print=False):
        a_main_window.ui.menubar = a_main_window.menuBar()
        found_extant_menu = a_main_window.ui.menubar.findChild(QtWidgets.QMenu, "menuCreateNewConnectedWidget") #"menuCreateNewConnectedWidget"
        if found_extant_menu is not None:
            if debug_print:
                print(f'existing create new connected widget menu found. Returning without creating.')
            return a_main_window.ui.menuCreateNewConnectedWidget, a_main_window.ui.createNewConnectedWidgetMenuActionsDict
        else:
            PhoMenuHelper.set_menu_default_stylesheet(a_main_window.ui.menubar) # Sets the default menu stylesheet
            ## Only creates the QActions now, no QMenus:
            # Define dictionary for actions:
            a_main_window.ui.createNewConnectedWidgetMenuActionsDict = {}            
            
            curr_action_key = PhoMenuHelper.add_action_item(a_main_window, "New Connected 3D Raster (PyQtGraph)", name="actionNewConnected3DRaster_PyQtGraph", tooltip="Create a new PyQtGraph 3D Raster plotter and connect it to this window", icon_path=":/Icons/Icons/SpikeRaster3DIcon.ico", actions_dict=a_main_window.ui.createNewConnectedWidgetMenuActionsDict)
            
            curr_action_key = PhoMenuHelper.add_action_item(a_main_window, "New Connected 3D Raster (Vedo)", name="actionNewConnected3DRaster_Vedo", tooltip="Create a new Vedo 3D Raster plotter and connect it to this window", icon_path=":/Icons/Icons/SpikeRaster3D_VedoIcon.ico", actions_dict=a_main_window.ui.createNewConnectedWidgetMenuActionsDict)
        
            curr_action_key = PhoMenuHelper.add_action_item(a_main_window, "New Connected Tuning Curves Explorer (ipcDataExplorer)", name="actionNewConnectedDataExplorer_ipc", tooltip="Create a new 3D Interactive Tuning Curve Data Explorer Plotter and connect it to this window", icon_path=":/Icons/Icons/InteractivePlaceCellDataExplorerIconWithLabel.ico", actions_dict=a_main_window.ui.createNewConnectedWidgetMenuActionsDict)
            
            curr_action_key = PhoMenuHelper.add_action_item(a_main_window, "New Connected Spikes+Behavior Explorer (ipspikesDataExplorer)", name="actionNewConnectedDataExplorer_ipspikes", tooltip="Create a new 3D Interactive Spike and Behavior Plotter and connect it to this window", icon_path=":/Icons/Icons/TuningMapDataExplorerIconWithLabel.ico", actions_dict=a_main_window.ui.createNewConnectedWidgetMenuActionsDict)
            
            
            ## Now Create the Menus for each QAction:
            
            # menuCreateNewConnectedWidget = menubar.addMenu('&Connections')
            a_main_window.ui.menuCreateNewConnectedWidget = QtWidgets.QMenu(a_main_window.ui.menubar) # A QMenu
            a_main_window.ui.actionMenuCreateNewConnectedWidget = a_main_window.ui.menubar.addMenu(a_main_window.ui.menuCreateNewConnectedWidget) # Used to remove the menu, a QAction
            # a_main_window.ui.menuCreateNewConnectedWidget.setTearOffEnabled(True)
            a_main_window.ui.menuCreateNewConnectedWidget.setObjectName("menuCreateNewConnectedWidget")
            a_main_window.ui.menuCreateNewConnectedWidget.setTitle("Create Connected Widget")

            ## Add the actions to the QMenu item:
            a_main_window.ui.menuCreateNewConnectedWidget.addActions(a_main_window.ui.createNewConnectedWidgetMenuActionsDict.values())
            
            ## TODO: is this even needed? I think it's done to remove it, but can't I just use a_main_window.ui.actionMenuCreateNewConnectedWidget directly?
            a_main_window.ui.createNewConnectedWidgetMenuActionsDict['actionMenuCreateNewConnectedWidget'] = a_main_window.ui.actionMenuCreateNewConnectedWidget            
            
            return a_main_window.ui.menuCreateNewConnectedWidget, a_main_window.ui.createNewConnectedWidgetMenuActionsDict



## Actions to be executed to create new plotters:
class CreateNewPyQtGraphPlotterCommand(BaseMenuCommand):
    """
    A command to create a plotter as needed
    """
    def __init__(self, spike_raster_window) -> None:
        super(CreateNewPyQtGraphPlotterCommand, self).__init__()
        self._spike_raster_window = spike_raster_window
        
    def execute(self, filename: str) -> None:
        """ Implicitly captures spike_raster_window """
        test_independent_pyqtgraph_raster_widget = self._spike_raster_window.create_new_connected_widget(type_of_3d_plotter='pyqtgraph')
        test_independent_pyqtgraph_raster_widget.show()
        # print(f"hiding {filename}")
        # self._hidden_files.append(filename)

        
class CreateNewVedoPlotterCommand(BaseMenuCommand):
    """
    A command to create a plotter as needed
    """
    def __init__(self, spike_raster_window) -> None:
        super(CreateNewVedoPlotterCommand, self).__init__()
        self._spike_raster_window = spike_raster_window
        
    def execute(self, filename: str) -> None:
        """ Implicitly captures spike_raster_window """
        test_independent_vedo_raster_widget = self._spike_raster_window.create_new_connected_widget(type_of_3d_plotter='vedo')
        test_independent_vedo_raster_widget.show()
        # global_connected_widgets['test_independent_vedo_raster_widget'] = test_independent_vedo_raster_widget

        
## These DataExplorers can't be created from spike_raster_window alone because it only holds a spike_df and not a full session object. Thus need to capture:
""" 
curr_active_pipeline, active_config_name, display_output, and spike_raster_window

TODO: NOTE that I run into an issue here, as the menus can't be setup properly from a _display_* function because by defn display functions only receieve computation_results and optional arguments, and never a full pipeline reference (they're instead performed on a pipeline itself. As such they can't get the 'curr_active_pipeline' to pass in to build these commands.

"""
class CreateNewDataExplorer_ipc_PlotterCommand(BaseMenuCommand):
    def __init__(self, spike_raster_window, curr_active_pipeline, active_config_name, display_output={}) -> None:
        super(CreateNewDataExplorer_ipc_PlotterCommand, self).__init__()
        self._spike_raster_window = spike_raster_window
        self._curr_active_pipeline = curr_active_pipeline
        self._active_config_name = active_config_name
        self._display_output = display_output
        
    def execute(self, filename: str) -> None:
        pActiveTuningCurvesPlotter = None
        # display_output = {}
        self._display_output = self._display_output | self._curr_active_pipeline.display('_display_3d_interactive_tuning_curves_plotter', self._active_config_name, extant_plotter=self._display_output.get('pActiveTuningCurvesPlotter', None), panel_controls_mode='Qt') # Works now!
        ipcDataExplorer = self._display_output['ipcDataExplorer']
        self._display_output['pActiveTuningCurvesPlotter'] = self._display_output.pop('plotter') # rename the key from the generic "plotter" to "pActiveSpikesBehaviorPlotter" to avoid collisions with others
        pActiveTuningCurvesPlotter = self._display_output['pActiveTuningCurvesPlotter']
        root_dockAreaWindow, placefieldControlsContainerWidget, pf_widgets = self._display_output['pane'] # for Qt mode:

class CreateNewDataExplorer_ipspikes_PlotterCommand(BaseMenuCommand):
    def __init__(self, spike_raster_window, curr_active_pipeline, active_config_name, display_output={}) -> None:
        super(CreateNewDataExplorer_ipspikes_PlotterCommand, self).__init__()
        self._spike_raster_window = spike_raster_window
        self._curr_active_pipeline = curr_active_pipeline
        self._active_config_name = active_config_name
        self._display_output = display_output
        
    def execute(self, filename: str) -> None:
        pActiveSpikesBehaviorPlotter = None
        # display_output = {}
        self._display_output = self._display_output | self._curr_active_pipeline.display('_display_3d_interactive_spike_and_behavior_browser', self._active_config_name, extant_plotter=self._display_output.get('pActiveSpikesBehaviorPlotter', None)) # Works now!
        ipspikesDataExplorer = self._display_output['ipspikesDataExplorer']
        self._display_output['pActiveSpikesBehaviorPlotter'] = self._display_output.pop('plotter') # rename the key from the generic "plotter" to "pActiveSpikesBehaviorPlotter" to avoid collisions with others
        pActiveSpikesBehaviorPlotter = self._display_output['pActiveSpikesBehaviorPlotter']
        ## Sync ipspikesDataExplorer to raster window:
        extra_interactive_spike_behavior_browser_sync_connection = self._spike_raster_window.connect_additional_controlled_plotter(controlled_plt=ipspikesDataExplorer)
        # test_independent_vedo_raster_widget.show()

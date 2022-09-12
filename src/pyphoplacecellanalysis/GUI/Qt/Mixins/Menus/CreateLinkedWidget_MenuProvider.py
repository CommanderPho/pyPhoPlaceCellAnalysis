from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.Mixins.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.BaseMenuProviderMixin import BaseMenuProviderMixin

from pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus.LocalMenus_AddRenderable import LocalMenus_AddRenderable

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.helpers import build_combined_time_synchronized_plotters_window, build_connected_time_synchronized_occupancy_plotter, build_connected_time_synchronized_placefields_plotter, build_connected_time_synchronized_decoder_plotter

    
from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.BaseMenuProviderMixin import BaseMenuCommand # for commands
    

class CreateLinkedWidget_MenuProvider(BaseMenuProviderMixin):
    """ 
    
    .ui.menus.global_window_menus.debug_menu_provider.top_level_menu
    
    .ui.menus.global_window_menus.debug.actions_dict
    .ui.menus.global_window_menus.debug.actions_dict
    
    .ui.menus.global_window_menus.actionCombineTimeSynchronizedPlotterWindow
    
    
    """
    top_level_menu_name = 'actionMenuCreateLinkedWidget'
    
    @property
    def activeMenuReference(self):
        """The reference to the top-level PhoUIContainer for this menu where references are stored to the ui elements and their actions."""
        return self.root_window.ui.menus.global_window_menus.create_linked_widget
    @activeMenuReference.setter
    def activeMenuReference(self, value):
        self.root_window.ui.menus.global_window_menus.create_linked_widget = value
        
    @property
    def CreateLinkedWidget_MenuProvider_actionsDict(self):
        return self.activeMenuReference.actions_dict
    @CreateLinkedWidget_MenuProvider_actionsDict.setter
    def CreateLinkedWidget_MenuProvider_actionsDict(self, value):
        self.activeMenuReference.actions_dict = value


    def __init__(self, render_widget: QtWidgets.QWidget, parent=None, **kwargs):
        """ the __init__ form allows adding menus to extant widgets without modifying their class to inherit from this mixin """
        super(CreateLinkedWidget_MenuProvider, self).__init__(render_widget=render_widget, parent=parent, **kwargs)
        # Setup member variables:
        
        
    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_init(self):
        """ perform any parameters setting/checking during init """
        BaseMenuProviderMixin.BaseMenuProviderMixin_on_init(self)
        
        assert self.has_root_window, "No root window!"
        # Define the core object
        self.activeMenuReference = PhoUIContainer.init_from_dict({'top_level_menu': None, 'actions_dict': {}})
    
    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    
    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_buildUI(self, **kwargs):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc)
        
        build QMenus from UI
        
        self.ui.menuCreate_Paired_Widget
        self.ui.actionTimeSynchronizedOccupancyPlotter
        self.ui.actionTimeSynchronizedPlacefieldsPlotter
        self.ui.actionTimeSynchronizedDecoderPlotter
        
        self.ui.actionCombineTimeSynchronizedPlotterWindow
        
        
        
        """
        spike_raster_window = kwargs.get('spike_raster_window', None)
        active_pf_2D_dt = kwargs.get('active_pf_2D_dt', None)
        display_output = kwargs.get('display_output', None)
        
        ## Add menu to the main menu bar:
        curr_window = self.root_window
        curr_menubar = self.root_menu_bar
        
        create_linked_widget = LocalMenus_AddRenderable.perform_build_manual_paired_Widget_menu(action_parent=curr_window, menu_parent=curr_menubar)
        self.activeMenuReference = create_linked_widget
        
        curr_actions_dict = self.CreateLinkedWidget_MenuProvider_actionsDict
        
        # print(f'activeMenuReference.top_level_menu: {self.activeMenuReference.top_level_menu}, activeMenuReference.actions_dict: {self.activeMenuReference.actions_dict}')
        
        self.activeMenuReference.top_level_menu.setObjectName("menuCreateLinkedWidget")
        
        # Manual Setup:
        # curr_actions_dict['actionTimeSynchronizedOccupancyPlotter'] = widget.ui.actionTimeSynchronizedOccupancyPlotter
        # curr_actions_dict['actionTimeSynchronizedPlacefieldsPlotter'] = widget.ui.actionTimeSynchronizedPlacefieldsPlotter
        # curr_actions_dict['actionTimeSynchronizedDecoderPlotter'] = widget.ui.actionTimeSynchronizedDecoderPlotter
        # curr_actions_dict['actionCombineTimeSynchronizedPlotterWindow'] = widget.ui.actionCombineTimeSynchronizedPlotterWindow
        
        curr_actions_dict['actionTimeSynchronizedOccupancyPlotter'].triggered.connect(CreateNewTimeSynchronizedPlotterCommand(spike_raster_window, active_pf_2D_dt, plotter_type='occupancy', display_output=display_output))
        curr_actions_dict['actionTimeSynchronizedPlacefieldsPlotter'].triggered.connect(CreateNewTimeSynchronizedPlotterCommand(spike_raster_window, active_pf_2D_dt, plotter_type='placefields', display_output=display_output))
        curr_actions_dict['actionTimeSynchronizedDecoderPlotter'].triggered.connect(CreateNewTimeSynchronizedPlotterCommand(spike_raster_window, active_pf_2D_dt, plotter_type='decoder', display_output=display_output))
        curr_actions_dict['actionCombineTimeSynchronizedPlotterWindow'].triggered.connect(CreateNewTimeSynchronizedCombinedPlotterCommand(spike_raster_window, active_pf_2D_dt, display_output))
        
        curr_window.ui.actionMenuCreateLinkedWidget = curr_menubar.addMenu(self.activeMenuReference.top_level_menu)  # add it to the menubar

        # Save references in the curr_window
        self.activeMenuReference.actions_dict['actionMenuCreateLinkedWidget'] = curr_window.ui.actionMenuCreateLinkedWidget 
        
        
        # Debugging Reference
        # self.activeMenuReference.all_refs = active_2d_plot_renderable_menus
        
        # self.activeMenuReference.all_refs = widget
        
        # widget.close()
        # self.activeMenuReference.all_refs = None
    
        return self.activeMenuReference.top_level_menu, self.activeMenuReference.actions_dict
        
        # return self.activeMenuReference.top_level_menu, self.activeMenuReference.actions_dict, self.activeMenuReference.all_refs
    
        

    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        ## Remove Debug Menu:
        curr_window = self.root_window
        curr_menubar = self.root_menu_bar
        curr_actions_dict = self.CreateLinkedWidget_MenuProvider_actionsDict

        curr_menubar.removeAction(curr_actions_dict[self.top_level_menu_name])
        curr_window.ui.actionMenuCreateLinkedWidget = None
        
        # curr_window.ui.menus.global_window_menus.debug.actions_dict = {} # Empty the dict of actions
        self.CreateLinkedWidget_MenuProvider_actionsDict = {}

    @QtCore.Slot()
    def CreateLinkedWidget_MenuProvider_on_menus_update(self):
        """ called to update menus dynamically. Only needed if the menu items themselves change dynamically.
        """
        pass
    
    
    
# build_combined_time_synchronized_plotters_window, build_connected_time_synchronized_occupancy_plotter, build_connected_time_synchronized_placefields_plotter, build_connected_time_synchronized_decoder_plotter
    
## Actions to be executed to create new plotters:

        
class CreateNewTimeSynchronizedPlotterCommand(BaseMenuCommand):
    """ build_combined_time_synchronized_plotters_window
    A command to create a plotter as needed
    """
    def __init__(self, spike_raster_window, active_pf_2D_dt, plotter_type='occupancy', display_output={}) -> None:
        super(CreateNewTimeSynchronizedPlotterCommand, self).__init__()
        self._spike_raster_window = spike_raster_window
        self._active_pf_2D_dt = active_pf_2D_dt
        self._display_output = display_output
        self._plotter_type = plotter_type
        
        
    def execute(self, filename: str) -> None:
        """ Implicitly captures spike_raster_window """
        print(f'CreateNewTimeSynchronizedPlotterCommand(): {self._plotter_type} callback')
        
        if self._plotter_type == 'occupancy':
            _out_sync_tuple = build_connected_time_synchronized_occupancy_plotter(active_pf_2D_dt=self._active_pf_2D_dt, sync_driver=self._spike_raster_window)
        elif self._plotter_type == 'placefields':
            _out_sync_tuple = build_connected_time_synchronized_placefields_plotter(active_pf_2D_dt=self._active_pf_2D_dt, sync_driver=self._spike_raster_window)
        elif self._plotter_type == 'decoder':
            _out_sync_tuple = build_connected_time_synchronized_decoder_plotter(active_pf_2D_dt=self._active_pf_2D_dt, sync_driver=self._spike_raster_window)
        else:
            raise NotImplementedError
        
        _out_display_key = f'synchronizedPlotter_{self._plotter_type}'
        _out_synchronized_plotter, _out_sync_connection = _out_sync_tuple
        print(f'_out_display_key: {_out_display_key}')
        self._display_output[_out_display_key] = _out_sync_tuple
        
    
class CreateNewTimeSynchronizedCombinedPlotterCommand(BaseMenuCommand):
    """ build_combined_time_synchronized_plotters_window
    A command to create a plotter as needed
    """
    def __init__(self, spike_raster_window, active_pf_2D_dt, display_output={}) -> None:
        super(CreateNewTimeSynchronizedCombinedPlotterCommand, self).__init__()
        self._spike_raster_window = spike_raster_window
        self._active_pf_2D_dt = active_pf_2D_dt
        self._display_output = display_output
        
        
    def execute(self, filename: str) -> None:
        """ Implicitly captures spike_raster_window """
        _out_synchronized_plotter = build_combined_time_synchronized_plotters_window(active_pf_2D_dt=self._active_pf_2D_dt, controlling_widget=self._spike_raster_window, create_new_controlling_widget=False)
        
        self._display_output['comboSynchronizedPlotter'] = _out_synchronized_plotter
        # (controlling_widget, curr_sync_occupancy_plotter, curr_placefields_plotter), root_dockAreaWindow, app = _out_synchronized_plotter
        
from qtpy import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons

from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Menus.BaseMenuProviderMixin import BaseMenuProviderMixin

from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable import LocalMenus_AddRenderable

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.helpers import build_combined_time_synchronized_plotters_window, build_connected_time_synchronized_occupancy_plotter, build_connected_time_synchronized_placefields_plotter, build_connected_time_synchronized_decoder_plotter
    
from pyphoplacecellanalysis.GUI.Qt.Menus.BaseMenuProviderMixin import BaseMenuCommand # for commands
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import AddNewDecodedPosition_MatplotlibPlotCommand, AddNewLongShortDecodedEpochSlices_MatplotlibPlotCommand # for add matplotlib plot action


    

class DockedWidgets_MenuProvider(BaseMenuProviderMixin):
    """ The most up-to-date (as of 2022-09-29) method of building custom menu items. 
    
    Conceptually, DockedWidgets are widgets that initially are docked within the main SpikeRaster2D window (usually as dockItems) and are synchronized with the scrolling of the time window.
     
    #TODO 2024-12-18 13:57: - [ ] NOTE: compared to `LocalMenus_AddRenderable` this class is not easy to call the menus programmatically :[
    
    ## 2024-12-18 - Right:
        _docked_menu_provider: DockedWidgets_MenuProvider = spike_raster_window.main_menu_window.ui.menus.global_window_menus.docked_widgets.menu_provider_obj
        _docked_menu_provider

    ## Wrong?
    .ui.menus.global_window_menus.debug_menu_provider.top_level_menu
    
    .ui.menus.global_window_menus.docked_widgets.actions_dict
    .ui.menus.global_window_menus.docked_widgets.actions_dict
    
    .ui.menus.global_window_menus.actionCombineTimeSynchronizedPlotterWindow
    
    
    Can be used in two forms:
        1. Via inherting the desired Window widget class from this as a mixin
        2. Via initializing via the __init__(...) method
    
    """
    action_name = 'DockedWidgets'
    top_level_menu_name = 'actionMenuDockedWidgets'
    
    @property
    def activeMenuReference(self):
        """The reference to the top-level PhoUIContainer for this menu where references are stored to the ui elements and their actions."""
        return self.root_window.ui.menus.global_window_menus.docked_widgets
    @activeMenuReference.setter
    def activeMenuReference(self, value):
        self.root_window.ui.menus.global_window_menus.docked_widgets = value
        
    @property
    def DockedWidgets_MenuProvider_actionsDict(self):
        return self.activeMenuReference.actions_dict
    @DockedWidgets_MenuProvider_actionsDict.setter
    def DockedWidgets_MenuProvider_actionsDict(self, value):
        self.activeMenuReference.actions_dict = value


    def __init__(self, render_widget: QtWidgets.QWidget, parent=None, **kwargs):
        """ the __init__ form allows adding menus to extant widgets without modifying their class to inherit from this mixin """
        super(DockedWidgets_MenuProvider, self).__init__(render_widget=render_widget, parent=parent, **kwargs)
        # Setup member variables:
        self.DockedWidgets_MenuProvider_on_init()
        self.DockedWidgets_MenuProvider_on_setup()
        
        
    @pyqtExceptionPrintingSlot()
    def DockedWidgets_MenuProvider_on_init(self):
        """ perform any parameters setting/checking during init """
        BaseMenuProviderMixin.BaseMenuProviderMixin_on_init(self)
        
        assert self.has_root_window, "No root window!"
        # Define the core object
        self.activeMenuReference = PhoUIContainer.init_from_dict({'top_level_menu': None, 'actions_dict': {}, 'menu_provider_obj': None}) # Can we just set 'self' here?
    
    @pyqtExceptionPrintingSlot()
    def DockedWidgets_MenuProvider_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    def DockedWidgets_MenuProvider_on_buildUI(self, **kwargs):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc)
        
        build QMenus from UI
        
        self.ui.menuCreate_Paired_Widget
        self.ui.actionTimeSynchronizedOccupancyPlotter
        self.ui.actionTimeSynchronizedPlacefieldsPlotter
        self.ui.actionTimeSynchronizedDecoderPlotter
        
        self.ui.actionCombineTimeSynchronizedPlotterWindow
        
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand, AddNewPseudo2DDecodedEpochs_MatplotlibPlotCommand, AddNewDecodedEpochMarginal_MatplotlibPlotCommand
        
        spike_raster_window = kwargs.get('spike_raster_window', None)
        # active_pf_2D_dt = kwargs.get('active_pf_2D_dt', None)
        curr_active_pipeline = kwargs.get('owning_pipeline_reference', None)
        active_config_name = kwargs.get('active_config_name', None)
        active_context = kwargs.get('context', None)
        display_output = kwargs.get('display_output', None)
        
        ## Add menu to the main menu bar:
        curr_window = self.root_window
        curr_menubar = self.root_menu_bar
        curr_actions_dict = self.DockedWidgets_MenuProvider_actionsDict
        
        ## Only creates the QActions now, no QMenus:
        addSubmenuActionKeys = []
        curr_action_key = PhoMenuHelper.add_action_item(self.root_window, "Matplotlib View", name="actionNewDockedMatplotlibView", tooltip="", icon_path=":/Graphics/Icons/graphics/ic_multiline_chart_48px.png", actions_dict=curr_actions_dict)
        addSubmenuActionKeys.append(curr_action_key)
        curr_action_key = PhoMenuHelper.add_action_item(self.root_window, "Context Nested Docks", name="actionNewDockedContextNested", tooltip="", icon_path=":/Graphics/Icons/graphics/ic_multiline_chart_48px.png", actions_dict=curr_actions_dict)
        addSubmenuActionKeys.append(curr_action_key)
        curr_action_key = PhoMenuHelper.add_action_item(self.root_window, "Long Short Decoded Epochs in Matplotlib Views", name="actionLongShortDecodedEpochsDockedMatplotlibView", tooltip="", icon_path=":/Graphics/Icons/graphics/ic_multiline_chart_48px.png", actions_dict=curr_actions_dict)
        addSubmenuActionKeys.append(curr_action_key)
        curr_action_key = PhoMenuHelper.add_action_item(self.root_window, "Directional Decoded Epochs in Matplotlib Views", name="actionDirectionalDecodedEpochsDockedMatplotlibView", tooltip="", icon_path=":/Graphics/Icons/graphics/ic_multiline_chart_48px.png", actions_dict=curr_actions_dict)
        addSubmenuActionKeys.append(curr_action_key)
        curr_action_key = PhoMenuHelper.add_action_item(self.root_window, "Pseudo2D Continuous Decoded Epochs in Matplotlib Views", name="actionPseudo2DDecodedEpochsDockedMatplotlibView", tooltip="", icon_path=":/Graphics/Icons/graphics/ic_multiline_chart_48px.png", actions_dict=curr_actions_dict)
        addSubmenuActionKeys.append(curr_action_key)
        curr_action_key = PhoMenuHelper.add_action_item(self.root_window, "Pseudo2D Continuous Decoded Marginals in Matplotlib Views", name="actionContinuousPseudo2DDecodedMarginalsDockedMatplotlibView", tooltip="", icon_path=":/Graphics/Icons/graphics/ic_multiline_chart_48px.png", actions_dict=curr_actions_dict)
        addSubmenuActionKeys.append(curr_action_key)
        curr_action_key = PhoMenuHelper.add_action_item(self.root_window, "Custom...", name="actionNewDockedCustom", tooltip="", actions_dict=curr_actions_dict)
        addSubmenuActionKeys.append(curr_action_key)
        
        # ==================================================================================================================== #
        # Connect the relevent actions to each action:
        
        curr_actions_dict['actionNewDockedMatplotlibView'].triggered.connect(AddNewDecodedPosition_MatplotlibPlotCommand(spike_raster_window, curr_active_pipeline, active_config_name, display_output))
        curr_actions_dict['actionNewDockedContextNested'].triggered.connect(CreateNewContextNestedDocksCommand(spike_raster_window, curr_active_pipeline, active_config_name=active_config_name, active_context=active_context, display_output=display_output))
        curr_actions_dict['actionLongShortDecodedEpochsDockedMatplotlibView'].triggered.connect(AddNewLongShortDecodedEpochSlices_MatplotlibPlotCommand(spike_raster_window, curr_active_pipeline, active_config_name=active_config_name, active_context=active_context, display_output=display_output))
        curr_actions_dict['actionDirectionalDecodedEpochsDockedMatplotlibView'].triggered.connect(AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand(spike_raster_window, curr_active_pipeline, active_config_name=active_config_name, active_context=active_context, display_output=display_output))
        curr_actions_dict['actionPseudo2DDecodedEpochsDockedMatplotlibView'].triggered.connect(AddNewPseudo2DDecodedEpochs_MatplotlibPlotCommand(spike_raster_window, curr_active_pipeline, active_config_name=active_config_name, active_context=active_context, display_output=display_output))
        curr_actions_dict['actionContinuousPseudo2DDecodedMarginalsDockedMatplotlibView'].triggered.connect(AddNewDecodedEpochMarginal_MatplotlibPlotCommand(spike_raster_window, curr_active_pipeline, active_config_name=active_config_name, active_context=active_context, display_output=display_output))
        
        # curr_actions_dict['actionNewDockedCustom'].triggered.connect(CreateNewTimeSynchronizedPlotterCommand(spike_raster_window, active_pf_2D_dt, plotter_type='decoder', active_context=active_context, display_output=display_output))
                

        # ==================================================================================================================== #
        # Now Create the Menus for each QAction:
        
        # curr_window.ui.actionMenuDockedWidgets = curr_menubar.addMenu(self.activeMenuReference.top_level_menu)  # add it to the menubar
        
        an_action_key, self.activeMenuReference.top_level_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Docked Widgets", name=self.top_level_menu_name, parent_menu=self.root_menu_bar, menu_actions_dict=curr_actions_dict)
        self.activeMenuReference.top_level_menu.setObjectName("menuDockedWidgets")
        an_action_key, self.activeMenuReference.add_docked_widget_menu = PhoMenuHelper.add_menu(a_main_window=self.root_window, text="Add Docked Widget", name='actionAddDockedWidget', parent_menu=self.activeMenuReference.top_level_menu, menu_actions_dict=curr_actions_dict)
        
        ## Add the actions to the QMenu item:
        # self.activeMenuReference.add_docked_widget_menu.addActions(curr_actions_dict.values())
        ## Add the addSubmenuActions to the Add submenu:
        self.activeMenuReference.add_docked_widget_menu.addActions([curr_actions_dict[a_key] for a_key in addSubmenuActionKeys])

        # Save references in the curr_window
        self.activeMenuReference.actions_dict['actionMenuDockedWidgets'] = curr_window.ui.actionMenuDockedWidgets
        # self.activeMenuReference.actions_dict['actionAddDockedWidget'] = curr_window.ui.actionAddDockedWidget # is this needed?        
        return self.activeMenuReference.top_level_menu, self.activeMenuReference.actions_dict

    @pyqtExceptionPrintingSlot()
    def DockedWidgets_MenuProvider_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        ## Remove Debug Menu:
        curr_window = self.root_window
        curr_menubar = self.root_menu_bar
        curr_actions_dict = self.DockedWidgets_MenuProvider_actionsDict

        curr_menubar.removeAction(curr_actions_dict[self.top_level_menu_name])
        curr_window.ui.actionMenuDockedWidgets = None
        
        self.DockedWidgets_MenuProvider_actionsDict = {}

    @pyqtExceptionPrintingSlot()
    def DockedWidgets_MenuProvider_on_menus_update(self):
        """ called to update menus dynamically. Only needed if the menu items themselves change dynamically.
        """
        pass
    
    
# ## Actions to be executed to create new plotters:
        
class CreateNewContextNestedDocksCommand(BaseMenuCommand):
    """ build_combined_time_synchronized_plotters_window
    A command to create a plotter as needed
    """
    def __init__(self, spike_raster_window, active_pipeline, active_config_name=None, active_context=None, display_output={}) -> None:
        super(CreateNewContextNestedDocksCommand, self).__init__()
        self._spike_raster_window = spike_raster_window
        self._active_pipeline = active_pipeline
        self._active_config_name = active_config_name
        self._context = active_context
        self._display_output = display_output
        
    def execute(self, *args, **kwargs) -> None:
        """ Implicitly captures spike_raster_window """
        print(f'menu execute(): {self}')
        pass # TODO:
        # _out_synchronized_plotter = build_combined_time_synchronized_plotters_window(active_pf_2D_dt=self._active_pf_2D_dt, controlling_widget=self._spike_raster_window.spike_raster_plt_2d, context=self._context, create_new_controlling_widget=False)
        # self._display_output['comboSynchronizedPlotter'] = _out_synchronized_plotter
        # (controlling_widget, curr_sync_occupancy_plotter, curr_placefields_plotter), root_dockAreaWindow, app = _out_synchronized_plotter
        
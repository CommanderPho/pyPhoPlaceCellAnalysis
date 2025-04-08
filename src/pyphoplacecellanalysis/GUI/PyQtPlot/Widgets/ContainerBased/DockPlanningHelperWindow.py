from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Callable, Union
from attrs import define, field, Factory
import nptyping as ND
from nptyping import NDArray
import numpy as np
import pandas as pd
from pathlib import Path
import io
from contextlib import redirect_stdout # used by DocumentationFilePrinter to capture print output

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.connections_container import ConnectionsContainer

import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum

from pyphoplacecellanalysis.GUI.Qt.Widgets.ScrollBarWithSpinBox.ScrollBarWithSpinBox import ScrollBarWithSpinBox
from pyphoplacecellanalysis.GUI.Qt.Widgets.LogViewerTextEdit import LogViewer
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.Resources.icon_helpers import try_get_icon

from pyphocorehelpers.gui.Qt.pandas_model import SimplePandasModel, create_tabbed_table_widget
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomCyclicColorsDockDisplayConfig, CustomDockDisplayConfig, DockDisplayColors, get_utility_dock_colors
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import DynamicDockDisplayAreaContentMixin, DynamicDockDisplayAreaOwningMixin

__all__ = ['DockPlanningHelperWindow']


# ==================================================================================================================== #
# Helper functions                                                                                                     #
# ==================================================================================================================== #

@define(slots=False, eq=False)
class DockPlanningHelperWindow(DynamicDockDisplayAreaOwningMixin):
    """ DockPlanningHelperWindow displays several testing widgets
    

    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.DockPlanningHelperWindow import DockPlanningHelperWindow
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget

    _out = DockPlanningHelperWindow.init_dock_area_builder(global_spikes_df, active_epochs_dfe, track_templates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)


    Updating Display Epoch:
        The `self.on_update_epoch_IDX(an_epoch_idx=0)` can be used to control which Epoch is displayed, and is synchronized across all four sorts.

    """
    plots: RenderPlots = field(init=False)
    plots_data: RenderPlotsData = field(init=False, repr=False)
    ui: PhoUIContainer = field(init=False, repr=False)
    params: VisualizationParameters = field(init=False, repr=keys_only_repr)

    # ==================================================================================================================== #
    # DynamicDockDisplayAreaOwningMixin Conformances                                                                       #
    # ==================================================================================================================== #
    @property 
    def dock_manager_widget(self) -> DynamicDockDisplayAreaContentMixin:
        """Must be implemented by subclasses to return the widget that manages the docks"""
        return self.ui.root_dockAreaWindow ## returns the PhoDockAreaContainingWindow
    


    # Plot Convenience Accessors _________________________________________________________________________________________ #
    # @property
    # def seperate_new_sorted_rasters_dict(self) -> Dict[str, NewSimpleRaster]:
    #     return self.plots_data.seperate_new_sorted_rasters_dict


    @property
    def dock_helper_widgets_container(self) -> PhoUIContainer:
        return self.ui.dock_helper_widgets

    @property
    def dock_helper_widgets(self) -> Dict[str, DockPlanningHelperWidget]:
        return self.ui.dock_helper_widgets.dock_helper_widgets
    @dock_helper_widgets.setter
    def dock_helper_widgets(self, value: Dict):
        self.ui.dock_helper_widgets.dock_helper_widgets = value

    @property
    def dock_widgets(self) -> Dict:
        """The dock_widgets property."""
        return self.plots.dock_widgets
    @dock_widgets.setter
    def dock_widgets(self, value: Dict):
        self.plots.dock_widgets = value

    @property
    def dock_configs(self) -> Dict[str, CustomDockDisplayConfig]:
        return self.ui.dock_configs
    @dock_configs.setter
    def dock_configs(self, value: Dict[str, CustomDockDisplayConfig]):
        self.ui.dock_configs = value

    @property
    def root_dockAreaWindow(self) -> PhoDockAreaContainingWindow:
        return self.ui.root_dockAreaWindow
    

    @classmethod
    def init_dock_area_builder(cls, n_dock_planning_helper_widgets:int=0, dock_add_locations=None, **param_kwargs):
        """
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps) # .trimmed_to_non_overlapping()
        global_laps_epochs_df = global_laps.to_dataframe()

        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors

        _obj = cls()

        name:str = 'DockPlanningHelperWindow'
        # Embedding in docks:
        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho DockPlanningHelperWindow')
        # icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico")
        # if icon is not None:
        #     root_dockAreaWindow.setWindowIcon(icon)

        _dock_helper_widgets_dict = {}
        _out_dock_widgets = {}
        dock_configs = {k:CustomCyclicColorsDockDisplayConfig(showCloseButton=False) for k, v in _dock_helper_widgets_dict.items()} # named_color_scheme=DisplayColorsEnum.Laps.get_LR_dock_colors
        

        # ## Build the utility controls at the bottom:
        # utility_controls_ui_dict, ctrls_dock_widgets_dict = _obj._build_utility_controls(root_dockAreaWindow)
        # _out_dock_widgets = ctrls_dock_widgets_dict | _out_dock_widgets
        
        
        # # Top Info Bar: ______________________________________________________________________________________________________ #
        # ## Add two labels in the top row that show the Long/Short column values:
        # long_short_info_layout = pg.LayoutWidget()
        # long_short_info_layout.setObjectName('layoutLongShortInfo')

        # long_info_label = long_short_info_layout.addLabel(text='LONG', row=0, col=0)
        # long_info_label.setObjectName('lblLongInfo')
        # # long_info_label.setAlignment(pg.QtCore.Qt.AlignCenter)
        # long_info_label.setAlignment(pg.QtCore.Qt.AlignLeft)

        # short_info_label = long_short_info_layout.addLabel(text='SHORT', row=0, col=1)
        # short_info_label.setObjectName('lblShortInfo')
        # # short_info_label.setAlignment(pg.QtCore.Qt.AlignCenter)
        # short_info_label.setAlignment(pg.QtCore.Qt.AlignRight)

        # _out_dock_widgets['LongShortColumnsInfo_dock'] = root_dockAreaWindow.add_display_dock(identifier='LongShortColumnsInfo_dock', widget=long_short_info_layout, dockSize=(600,60), dockAddLocationOpts=['top'], display_config=CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False, corner_radius='0px'))
        # _out_dock_widgets['LongShortColumnsInfo_dock'][1].hideTitleBar() # hide the dock title bar

        # # Add the widgets to the .ui:
        # long_short_info_layout = long_short_info_layout
        # long_info_label = long_info_label
        # short_info_label = short_info_label
        # info_labels_widgets_dict = dict(long_short_info_layout=long_short_info_layout, long_info_label=long_info_label, short_info_label=short_info_label)

        root_dockAreaWindow.resize(600, 900)

        ## Build final .plots and .plots_data:
        _obj.plots = RenderPlots(name=name, root_dockAreaWindow=root_dockAreaWindow, dock_widgets=_out_dock_widgets, text_items_dict=None) # , ctrl_widgets={'slider': slider}
        _obj.plots_data = RenderPlotsData(name=name, 
                                            # **{k:v for k, v in _obj.plots_data.to_dict().items() if k not in ['name']},
                                            )
        dock_helper_widgets = PhoUIContainer(name=f'{name}.dock_helper_widgets', dock_helper_widgets=_dock_helper_widgets_dict)
        _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, dock_helper_widgets=dock_helper_widgets, dock_configs=dock_configs, controlled_references=None, connections=ConnectionsContainer()) # , **utility_controls_ui_dict, **info_labels_widgets_dict
        _obj.params = VisualizationParameters(name=name, use_plaintext_title=False, **param_kwargs)

        _obj.register_internal_callbacks()


        # ==================================================================================================================== #
        # Improved method:                                                                                                     #
        # ==================================================================================================================== #
        _obj._add_N_random_dock_planning_helper_widgets(n_dock_planning_helper_widgets=3)

        return _obj


    def _add_N_random_dock_planning_helper_widgets(self, n_dock_planning_helper_widgets: int):
        """ adds N new planning helper docked widgets to the window
        """
        for i in np.arange(n_dock_planning_helper_widgets):
            dock_id_str: str = f'dock[{i}]'
            a_dock_helper_widget, a_dock_config, a_dock_widget = self.perform_create_new_dock_widget(dock_id_str=dock_id_str, active_dock_add_location='bottom', dockSize=(300,600), autoOrientation=False)            



    def _build_utility_controls(self, root_dockAreaWindow):
        """ Build the utility controls at the bottom """
        return {}, {}


    def register_internal_callbacks(self):
        """ registers all internally-owned callback functions. """
        # self.on_idx_changed_callback_function_dict['update_plot_titles_with_stats'] = self.update_plot_titles_with_stats
        for i, (dock_id_str, a_dock_helper_widget) in enumerate(self.ui.dock_helper_widgets.dock_helper_widgets.items()):
            # dock_id_str: str = f'dock[{i}]'
            extant_connection = self.ui.connections.pop(dock_id_str, None)
            if extant_connection is not None:
                a_dock_helper_widget.sigCreateNewDock.disconnect() ## disconnect
                
            ## create a new one:
            _a_conn = a_dock_helper_widget.sigCreateNewDock.connect(self.on_click_create_new_dock)
            self.ui.connections[dock_id_str] = {'sigCreateNewDock': _a_conn} # create a new connection
            
        
        

    @function_attributes(short_name=None, tags=['create'], input_requires=[], output_provides=[], uses=[], used_by=['.on_click_create_new_dock'], creation_date='2025-03-25 15:33', related_items=[])
    def perform_create_new_dock_widget(self, dock_id_str:str=None, active_dock_add_location:Union[str, Tuple[str, str]]='bottom', dockSize=None, autoOrientation=False):
        """ Creaets a new DockPlanningHelper widget
        
        a_dock_helper_widget, a_dock_config, a_dock_widget = _out.perform_create_new_dock_widget()
        
        """
        if dock_id_str is None:
            num_helper_widgets: int = len(self.dock_helper_widgets)
            next_helper_widget_idx: int = num_helper_widgets
            dock_id_str: str = f'dock[{next_helper_widget_idx}]'
            
        if active_dock_add_location is None:
            active_dock_add_location:str = 'bottom'
            
        # if isinstance(active_dock_add_location, str):
        #     # try to convert to tuple:
        #     _split_active_dock_add_location = active_dock_add_location.split(', ', maxsplit=2)
        #     if len(_split_active_dock_add_location) == 2:
        #         active_dock_add_location = tuple(_split_active_dock_add_location) # use a tuple
        #         print(f'active_dock_add_location: {active_dock_add_location}')
        #         assert len(active_dock_add_location) == 2
                        
        if dockSize is None:
            dockSize = (300,600)
            
        extant_dock_helper_widget = self.dock_helper_widgets.get(dock_id_str, None)
        extant_config = self.dock_configs.get(dock_id_str, None)
        extant_dock_widget = self.dock_widgets.get(dock_id_str, None)
        assert extant_dock_helper_widget is None
        assert extant_config is None
        assert extant_dock_widget is None
        
        ## make new widget:
        a_dock_helper_widget = DockPlanningHelperWidget(dock_title=dock_id_str, dock_id=dock_id_str, defer_show=True)
        
        ## connections:
        an_extant_dock_helper_widget_connections_dict = self.ui.connections.get(dock_id_str, None)
        assert an_extant_dock_helper_widget_connections_dict is None
        _a_conn = a_dock_helper_widget.sigCreateNewDock.connect(self.on_click_create_new_dock)
        self.ui.connections[dock_id_str] = {'sigCreateNewDock': _a_conn, 'sigDockConfigChanged': None}

        ## Updates: self.dock_helper_widgets, self.dock_configs, self.dock_widgets
        self.dock_helper_widgets[dock_id_str] = a_dock_helper_widget
        self.dock_configs[dock_id_str] = CustomDockDisplayConfig(showCloseButton=False)
        self.dock_widgets[dock_id_str] = self.root_dockAreaWindow.add_display_dock(identifier=dock_id_str, widget=a_dock_helper_widget, dockSize=dockSize, dockAddLocationOpts=active_dock_add_location, display_config=self.dock_configs[dock_id_str], autoOrientation=autoOrientation)
        
        a_widget_fg_color, a_widget_bg_color, a_widget_border_color = self.dock_configs[dock_id_str].get_colors(None, is_dim=False)
        print(f'a_widget_color: {a_widget_bg_color}')
        a_dock_helper_widget.color = a_widget_bg_color
        _a_new_conn = a_dock_helper_widget.sigDockConfigChanged.connect(self.on_update_dock_config)
        self.ui.connections[dock_id_str]['sigDockConfigChanged'] = _a_new_conn

        # embedded_child_widget, dDisplayItem = self.dock_widgets[dock_id_str]

        return self.dock_helper_widgets[dock_id_str], self.dock_configs[dock_id_str], self.dock_widgets[dock_id_str]



    @function_attributes(short_name=None, tags=['dock'], input_requires=[], output_provides=[], uses=['self.perform_create_new_dock_widget'], used_by=[], creation_date='2025-03-25 15:33', related_items=[])
    def on_click_create_new_dock(self, child_widget: DockPlanningHelperWidget, relative_location: str):
        """ called with the child_widget to create a new relative widget
        
        """
        # [self.embedding_dock_item, 'bottom']
        print(f'DockPlanningHelperWindow.on_click_create_new_dock(...)')
        # print(f'\t')
        print(f'\trelative_location: {relative_location}')
        is_add_containing_parent_item: bool = False
        
        if child_widget is None:
            print(f'\t child_widget is None!')
        else:
            print(f'\tchild_widget.identifier: {child_widget.identifier}')
            log_string = child_widget.rebuild_output()
            print(f'\tchild_widget: {log_string}') # TypeError: __str__ returned non-string (type NoneType)
            new_dock_config = child_widget.rebuild_config()
            if isinstance(relative_location, str):
                # try to convert to tuple:
                _split_active_dock_add_location = relative_location.split(', ', maxsplit=2)
                if len(_split_active_dock_add_location) == 2:
                    rel_loc, rel_dock_id = _split_active_dock_add_location
                    rel_dock_item = self.root_dockAreaWindow.find_display_dock(rel_dock_id)
                    assert rel_dock_item is not None, f"rel_dock_id: '{rel_dock_id}' does not exist. relative_location: {relative_location}"
                    # relative_location_tuple = tuple(_split_active_dock_add_location) # use a tuple
                    relative_location_tuple = (rel_loc, rel_dock_item) # use a tuple
                    print(f'relative_location_tuple: {relative_location_tuple}')
                    assert len(relative_location_tuple) == 2
                    new_dock_config.setdefault('dockAddLocationOpts', relative_location_tuple)
                    if rel_loc == 'containing':
                        ## Special handling
                        print(f'found special "containing" case!')
                        is_add_containing_parent_item = True
                    else:
                        is_add_containing_parent_item = False
                        pass                                            
                else:
                    # not parsable 
                    new_dock_config.setdefault('dockAddLocationOpts', (relative_location, ))
            else:
                    raise NotImplementedError(f'relative_location: {relative_location}')

            # new_dock_config.setdefault('dockAddLocationOpts', (relative_location, ))
            new_dock_config['widget'] = child_widget
            print(f'\t creating new child widget with config: {new_dock_config}\n')
            if is_add_containing_parent_item:
                ## Special handling
                print(f'\thandling special "containing" case!')
                
                ## get the child_widget's Dock
                
                # dDisplayItem, nested_dynamic_docked_widget_container = self.dock_manager_widget.build_wrapping_nested_dock_area([child_widget], dock_group_name='test_group_A')
                dDisplayItem, nested_dynamic_docked_widget_container = self.dock_manager_widget.build_wrapping_nested_dock_area([child_widget.embedding_dock_item], dock_group_name='test_group_A')
                # nested_dock_items[dock_group_name] = dDisplayItem # Dock
                # nested_dynamic_docked_widget_container_widgets[dock_group_name] = nested_dynamic_docked_widget_container # nested_dynamic_docked_widget_container
                print(f'dDisplayItem: {dDisplayItem}, nested_dynamic_docked_widget_container: {nested_dynamic_docked_widget_container}')
                pass
            else:
                a_dock_helper_widget, a_dock_config, a_dock_widget = self.perform_create_new_dock_widget(active_dock_add_location=new_dock_config.get('dockAddLocationOpts', None), dockSize=new_dock_config.get('dockSize', None), autoOrientation=new_dock_config.get('autoOrientation', None))
        
            print(f'\t done.')

        
        # print(f'DockPlanningHelperWindow.on_click_create_new_dock(child_widget: {child_widget or 'None'}, relative_location: "{relative_location or None}")')
        # self.action_create_new_dock.emit(self.embedding_dock_item, 'bottom')
        # self.action_create_new_dock.emit(self, 'bottom')


    def on_update_dock_config(self, child_widget: DockPlanningHelperWidget):
        """ called with the child_widget when the color or other property changes
        
        """
        # [self.embedding_dock_item, 'bottom']
        print(f'DockPlanningHelperWindow.on_update_dock_config(...)')
        # print(f'\t')
        
        if child_widget is None:
            print(f'\t child_widget is None!')
        else:
            print(f'\tchild_widget.identifier: {child_widget.identifier}')
            log_string = child_widget.rebuild_output()
            print(f'\tchild_widget: {log_string}') # TypeError: __str__ returned non-string (type NoneType)
            new_dock_config: Dict = child_widget.rebuild_config()
            print(f'\tnew_dock_config: {new_dock_config}')
            # if isinstance(relative_location, str):
            #     # try to convert to tuple:
            #     _split_active_dock_add_location = relative_location.split(', ', maxsplit=2)
            #     if len(_split_active_dock_add_location) == 2:
            #         rel_loc, rel_dock_id = _split_active_dock_add_location
            #         rel_dock_item = self.root_dockAreaWindow.find_display_dock(rel_dock_id)
            #         assert rel_dock_item is not None, f"rel_dock_id: '{rel_dock_id}' does not exist. relative_location: {relative_location}"
            #         # relative_location_tuple = tuple(_split_active_dock_add_location) # use a tuple
            #         relative_location_tuple = (rel_loc, rel_dock_item) # use a tuple
            #         print(f'relative_location_tuple: {relative_location_tuple}')
            #         assert len(relative_location_tuple) == 2
            #         new_dock_config.setdefault('dockAddLocationOpts', relative_location_tuple)
            #     else:
            #         # not parsable 
            #         new_dock_config.setdefault('dockAddLocationOpts', (relative_location, ))
            # else:
            #         raise NotImplementedError(f'relative_location: {relative_location}')


            a_dock_id = child_widget.identifier
            print(f'child_widget.identifier: "{child_widget.identifier}"')
            
            assert a_dock_id in self.dock_helper_widgets
            assert a_dock_id in self.dock_configs
            assert a_dock_id in self.dock_widgets
            print(f'\tchild widget found!')
            a_widget = self.dock_helper_widgets[a_dock_id]
            a_config = self.dock_configs[a_dock_id]
            a_helper_widget, a_dock = self.dock_widgets[a_dock_id]
            # print(a_config)
            a_hex_bg_color: str = a_widget.color.name(pg.QtGui.QColor.HexRgb)
            print(f'a_hex_bg_color: {a_hex_bg_color}')
            # a_config.custom_get_colors_callback
            # a_config.custom_get_colors_callback = CustomDockDisplayConfig.build_custom_get_colors_fn(bg_color='#44aa44', border_color='#339933')
            # a_config.custom_get_colors_callback = CustomDockDisplayConfig.build_custom_get_colors_fn(bg_color=a_hex_bg_color, border_color=a_hex_bg_color)
            a_config.custom_get_colors_callback = None
            a_config.custom_get_colors_dict = {False: DockDisplayColors(fg_color='#fff', bg_color=a_hex_bg_color, border_color=a_hex_bg_color),
                        True: DockDisplayColors(fg_color='#aaa', bg_color=a_hex_bg_color, border_color=a_hex_bg_color),
                }
            
            # a_config
            a_dock.updateStyle()
            



            # for a_dock_id, a_widget in child_widget.dock_helper_widgets.items():
            # for a_dock_id, a_widget in self.dock_helper_widgets.items():
                
                # # print(a_widget.color)
                # a_config = child_widget.dock_configs[a_dock_id]
                # a_helper_widget, a_dock = child_widget.dock_widgets[a_dock_id]
                # # print(a_config)
                # a_hex_bg_color: str = a_widget.color.name(pg.QtGui.QColor.HexRgb)
                # print(f'a_hex_bg_color: {a_hex_bg_color}')
                # # a_config.custom_get_colors_callback
                # # a_config.custom_get_colors_callback = CustomDockDisplayConfig.build_custom_get_colors_fn(bg_color='#44aa44', border_color='#339933')
                # # a_config.custom_get_colors_callback = CustomDockDisplayConfig.build_custom_get_colors_fn(bg_color=a_hex_bg_color, border_color=a_hex_bg_color)
                # a_config.custom_get_colors_callback = None
                # a_config.custom_get_colors_dict = {False: DockDisplayColors(fg_color='#fff', bg_color=a_hex_bg_color, border_color=a_hex_bg_color),
                #             True: DockDisplayColors(fg_color='#aaa', bg_color=a_hex_bg_color, border_color=a_hex_bg_color),
                #     }
                
                # # a_config
                # a_dock.updateStyle()
                

            # new_dock_config.setdefault('dockAddLocationOpts', (relative_location, ))
            # new_dock_config['widget'] = child_widget
            print(f'\t creating new child widget with config: {new_dock_config}\n')
            # a_dock_helper_widget, a_dock_config, a_dock_widget = self.perform_create_new_dock_widget(active_dock_add_location=new_dock_config.get('dockAddLocationOpts', None), dockSize=new_dock_config.get('dockSize', None), autoOrientation=new_dock_config.get('autoOrientation', None))
            print(f'\t done.')

        
        # print(f'DockPlanningHelperWindow.on_click_create_new_dock(child_widget: {child_widget or 'None'}, relative_location: "{relative_location or None}")')
        # self.action_create_new_dock.emit(self.embedding_dock_item, 'bottom')
        # self.action_create_new_dock.emit(self, 'bottom')



    # ==================================================================================================================== #
    # Other Functions                                                                                                      #
    # ==================================================================================================================== #

    def write_to_log(self, log_messages):
        """ logs text to the text widget at the bottom """
        self.ui.logTextEdit.write_to_log(log_messages)
        # self.ui.logTextEdit.append(log_messages)
        # # Automatically scroll to the bottom
        # self.ui.logTextEdit.verticalScrollBar().setValue(
        #     self.ui.logTextEdit.verticalScrollBar().maximum()
        # )


    def setWindowTitle(self, title: str):
        """ updates the window's title """
        self.ui.root_dockAreaWindow.setWindowTitle(title)


    def set_top_info_bar_visibility(self, is_visible=False):
        """Hides/Shows the top info bar dock """
        LongShortColumnsInfo_dock_layout, LongShortColumnsInfo_dock_Dock = self.plots.dock_widgets['LongShortColumnsInfo_dock']
        # LongShortColumnsInfo_dock_layout.hide() # No use
        # _out_ripple_rasters.ui.long_short_info_layout.hide() # No use
        LongShortColumnsInfo_dock_Dock.setVisible(is_visible)

    def set_bottom_controls_visibility(self, is_visible=False):
        """Hides/Shows the top info bar dock """
        found_dock_layout, found_dock_Dock = self.plots.dock_widgets['bottom_controls']
        # LongShortColumnsInfo_dock_layout.hide() # No use
        # _out_ripple_rasters.ui.long_short_info_layout.hide() # No use
        found_dock_Dock.setVisible(is_visible)







    # ==================================================================================================================== #
    # Core Component Building Classmethods                                                                                 #
    # ==================================================================================================================== #

    # @classmethod
    # def _build_internal_raster_plots(cls, spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, track_templates: TrackTemplates, debug_print=True, defer_show=True, **kwargs):
    #     """ 2023-11-30 **DO EM ALL SEPERATELY**

    #     _out_data, _out_plots = _build_internal_raster_plots(spikes_df, active_epochs_df, track_templates, debug_print=True)

    #     History:
    #         Called `_post_modern_debug_plot_directional_template_rasters`


    #     Uses:
    #         paired_separately_sort_neurons

    #     """
    #     from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`
    #     from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays
    #     from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
    #     from pyphocorehelpers.gui.Qt.color_helpers import QColor, build_adjusted_color
    #     from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons
    #     from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot, NewSimpleRaster

    #     ## spikes_df: get the spikes to plot
    #     # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
    #     # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
    #     # track_templates.shared_LR_aclus_only_neuron_IDs

    #     figure_name: str = kwargs.pop('figure_name', 'rasters debugger')

    #     decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

    #     neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
    #     # _unit_qcolors_map, unit_colors_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
    #     unit_colors_map, _unit_colors_ndarray_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
    #     # `unit_colors_map` is main colors output

    #     included_neuron_ids = np.array(list(unit_colors_map.keys())) # one list for all decoders
    #     n_neurons = len(included_neuron_ids)

    #     print(f'included_neuron_ids: {included_neuron_ids}, n_neurons: {n_neurons}')

    #     # included_neuron_ids = np.sort(np.union1d(track_templates.shared_RL_aclus_only_neuron_IDs, track_templates.shared_LR_aclus_only_neuron_IDs))
    #     # n_neurons = len(included_neuron_ids)

    #     # Get only the spikes for the shared_aclus:
    #     spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(included_neuron_ids)
    #     # spikes_df = spikes_df.spikes.adding_lap_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX')
    #     spikes_df = spikes_df.spikes.adding_epochs_identity_column(active_epochs_df, epoch_id_key_name='new_epoch_IDX', epoch_label_column_name='label') # , override_time_variable_name='t_seconds'
    #     # spikes_df = spikes_df[spikes_df['ripple_id'] != -1]
    #     spikes_df = spikes_df[(spikes_df['new_epoch_IDX'] != -1)] # ['lap', 'maze_relative_lap', 'maze_id']
    #     spikes_df, neuron_id_to_new_IDX_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards


    #     # CORRECT: Even: RL, Odd: LR
    #     RL_neuron_ids = track_templates.shared_RL_aclus_only_neuron_IDs.copy() # (69, )
    #     LR_neuron_ids = track_templates.shared_LR_aclus_only_neuron_IDs.copy() # (64, )

    #     included_any_context_neuron_ids_dict = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], (LR_neuron_ids, RL_neuron_ids, LR_neuron_ids, RL_neuron_ids)))

    #     # INDIVIDUAL SORTING for each raster:
    #     # sortable_values_list_dict = {k:deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for k, a_decoder in decoders_dict.items()} # tuning_curve peak location
    #     sortable_values_list_dict = {k:deepcopy(a_decoder.pf.peak_tuning_curve_center_of_masses) for k, a_decoder in decoders_dict.items()} # tuning_curve CoM location
    #     sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, (unsorted_original_neuron_IDs_lists, unsorted_neuron_IDs_lists, unsorted_sortable_values_lists, unsorted_unit_colors_map) = paired_separately_sort_neurons(decoders_dict, included_any_context_neuron_ids_dict, sortable_values_list_dict=sortable_values_list_dict)

    #     _out_data = RenderPlotsData(name=figure_name, spikes_df=spikes_df, unit_sort_orders_dict=None, included_any_context_neuron_ids_dict=included_any_context_neuron_ids_dict,
    #                                 sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None,
    #                                 unsorted_original_neuron_IDs_lists=deepcopy(unsorted_original_neuron_IDs_lists), unsorted_neuron_IDs_lists=deepcopy(unsorted_neuron_IDs_lists), unsorted_sortable_values_lists=deepcopy(unsorted_sortable_values_lists), unsorted_unit_colors_map=deepcopy(unsorted_unit_colors_map))
    #     _out_plots = RenderPlots(name=figure_name, rasters_display_outputs=None)

    #     # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`
    #     _out_data.sorted_neuron_IDs_lists = sorted_neuron_IDs_lists
    #     _out_data.sort_helper_neuron_id_to_neuron_colors_dicts = sort_helper_neuron_id_to_neuron_colors_dicts
    #     _out_data.sort_helper_neuron_id_to_sort_IDX_dicts = sort_helper_neuron_id_to_sort_IDX_dicts
    #     _out_data.unit_sort_orders_dict = {} # empty array

    #     ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
    #     _out_plots.rasters = {}
    #     _out_plots.rasters_display_outputs = {}
    #     for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
    #         title_str = f'{a_decoder_name}'

    #         # an_included_unsorted_neuron_ids = deepcopy(included_any_context_neuron_ids_dict[a_decoder_name])
    #         an_included_unsorted_neuron_ids = deepcopy(unsorted_neuron_IDs_lists[i])
    #         a_sorted_neuron_ids = deepcopy(sorted_neuron_IDs_lists[i])

    #         unit_sort_order, desired_sort_arr = find_desired_sort_indicies(an_included_unsorted_neuron_ids, a_sorted_neuron_ids)
    #         print(f'unit_sort_order: {unit_sort_order}\ndesired_sort_arr: {desired_sort_arr}')
    #         _out_data.unit_sort_orders_dict[a_decoder_name] = deepcopy(unit_sort_order)

    #         # Get only the spikes for the shared_aclus:
    #         a_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(an_included_unsorted_neuron_ids)
    #         a_spikes_df, neuron_id_to_new_IDX_map = a_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards

    #         _out_plots.rasters_display_outputs[a_decoder_name] = new_plot_raster_plot(a_spikes_df, an_included_unsorted_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=deepcopy(unsorted_unit_colors_map), scatter_plot_kwargs=None, scatter_app_name=f'pho_directional_laps_rasters_{title_str}', defer_show=defer_show, active_context=None)
    #         # an_app, a_win, a_plots, a_plots_data, an_on_update_active_epoch, an_on_update_active_scatterplot_kwargs = _out_plots.rasters_display_outputs[a_decoder_name]


    #     return _out_data, _out_plots


    # ==================================================================================================================== #
    # Registering Output Signals                                                                                           #
    # ==================================================================================================================== #
    def a_debug_callback_fn(self, an_idx: int, an_epoch=None):
        out = io.StringIO()

        curr_epoch_label = self.lookup_label_from_index(an_idx)

        with redirect_stdout(out):
            print(f'=====================================================================================\n\tactive_epoch_IDX: {an_idx} :::', end='\t')

            ## Registered printing functions are called here, and anything they print is written to the textarea at the bottom of the widget.


            print(f'______________________________________________________________________________________________________________________\n')

        self.write_to_log(str(out.getvalue()))


    def update_plot_titles_with_stats(self, an_idx: int):
        """ Updates the titles of each of the four rasters with the appropriate spearman rho value.
        captures: rank_order_results_debug_values || active_epochs_df, formatted_title_strings_dict


        Usages:
            self.params.enable_show_spearman
            self.params.enable_show_pearson
            self.params.enable_show_Z_values

            self.active_epoch_result_df


        """
        from pyphocorehelpers.print_helpers import generate_html_string # used for `plot_long_short_surprise_difference_plot`
        
        is_laps: bool = self.params.is_laps
        use_plaintext_title: bool = self.params.use_plaintext_title
        if not use_plaintext_title:
            formatted_title_strings_dict = DisplayColorsEnum.get_pyqtgraph_formatted_title_dict()

        # curr_epoch_label = a_plotter.lookup_label_from_index(an_idx)
        # ripple_combined_epoch_stats_df = a_plotter.rank_order_results.ripple_combined_epoch_stats_df
        # curr_new_results_df = ripple_combined_epoch_stats_df[ripple_combined_epoch_stats_df.index == curr_epoch_label]

        curr_new_results_df = self.active_epoch_result_df
        for a_decoder_name, a_root_plot in self.plots.root_plots.items():
            # a_real_value = rank_order_results_debug_values[a_decoder_name][0][an_idx]
            a_std_column_name: str = self.decoder_name_to_column_name_prefix_map[a_decoder_name]

            if (curr_new_results_df is not None):
                all_column_names = curr_new_results_df.filter(regex=f'^{a_std_column_name}').columns.tolist()
                active_column_names = []
                # print(active_column_names)
                if self.params.enable_show_spearman:
                    active_column_names = [col for col in all_column_names if col.endswith("_spearman")]
                    if self.params.enable_show_Z_values:
                        active_column_names += [col for col in all_column_names if col.endswith("_spearman_Z")]


                if self.params.enable_show_pearson:
                    active_column_names += [col for col in all_column_names if col.endswith("_pearson")]
                    if self.params.enable_show_Z_values:
                        active_column_names += [col for col in all_column_names if col.endswith("_pearson_Z")]


                active_column_values = curr_new_results_df[active_column_names]
                active_values_dict = active_column_values.iloc[0].to_dict() # {'LR_Long_spearman': -0.34965034965034975, 'LR_Long_pearson': -0.5736588716389961, 'LR_Long_spearman_Z': -0.865774983083525, 'LR_Long_pearson_Z': -1.4243571733839517}
                active_raw_col_val_dict = {k.replace(f'{a_std_column_name}_', ''):v for k,v in active_values_dict.items()} # remove the "LR_Long" prefix so it's just the variable names
            else:
                ## No RankOrderResults
                print(f'WARN: No RankOrderResults')
                active_raw_col_val_dict = {}
                
            active_formatted_col_val_list = [':'.join([generate_html_string(str(k), color='grey', bold=False), generate_html_string(f'{v:0.3f}', color='white', bold=True)]) for k,v in active_raw_col_val_dict.items()]
            final_values_string: str = '; '.join(active_formatted_col_val_list)

            if use_plaintext_title:
                title_str = generate_html_string(f"{a_std_column_name}: {final_values_string}")
            else:
                # Color formatted title:
                a_formatted_title_string_prefix: str = formatted_title_strings_dict[a_std_column_name]
                title_str = generate_html_string(f"{a_formatted_title_string_prefix}: {final_values_string}")

            a_root_plot.setTitle(title=title_str)


   


## Adding callbacks to `DockPlanningHelperWindow` when the slider changes:


# # ==================================================================================================================== #
# # CALLBACKS:                                                                                                           #
# # ==================================================================================================================== #

## Start Qt event loop
if __name__ == '__main__':
    from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock, DockDisplayConfig
    from pyphoplacecellanalysis.External.pyqtgraph.dockarea.DockArea import DockArea
    from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.DockPlanningHelperWindow import DockPlanningHelperWindow

    app = pg.mkQApp("DockPlanningHelperWindow Example")
    widget: DockPlanningHelperWindow = DockPlanningHelperWindow.init_dock_area_builder(n_dock_planning_helper_widgets=4)
    a_dock_helper_widget, a_dock_config, a_dock_widget = widget.perform_create_new_dock_widget()
    
    # widget.show()
    pg.exec()

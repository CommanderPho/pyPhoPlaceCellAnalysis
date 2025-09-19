import operator
import weakref
from collections import OrderedDict

import copy
from typing import Optional, Dict, List, Tuple, Callable
from attrs import define, field, Factory
from neuropy.core.user_annotations import function_attributes, metadata_attributes
import numpy as np
import pandas as pd
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets
# from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsWidget import GraphicsWidget

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_scrollable_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_with_nested_viewbox_ui
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_pyqtgraph_epoch_indicator_regions
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum


__all__ = ['EpochsEditor']

class CustomViewBox(pg.ViewBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseEnabled(x=True, y=False)
        self.epochs_editor = None  # Will be set by EpochsEditor


    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            print("CustomViewBox.mouseClickEvent(event): Right-click detected!")
            pos = event.scenePos()
            if self.sceneBoundingRect().contains(pos):
                mouse_point = self.mapSceneToView(pos)
                if self.parent() is not None:
                    self.parent().show_context_menu(event.screenPos(), mouse_point) # AttributeError: 'NoneType' object has no attribute 'show_context_menu'
        else:
            super().mouseClickEvent(event)


@metadata_attributes(short_name=None, tags=['useful', 'gui', 'utility', 'epochs', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-17 19:25', related_items=[])
@define(slots=False)
class EpochsEditor:
    """ EpochsEditor to allow user modification of epoch intervals using PyQtGraph and multiple custom linear rect items, like for the laps

    It was used to display the laps for the entire session and allow the user to tweak their start/end times and then save the changes manually.
    
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.EpochsEditorItem import EpochsEditor # perform_plot_laps_diagnoser

        sess = global_session

        # pos_df = sess.compute_position_laps() # ensures the laps are computed if they need to be:
        position_obj = deepcopy(sess.position)
        position_obj.compute_higher_order_derivatives()
        pos_df = position_obj.compute_smoothed_position_info(N=20) ## Smooth the velocity curve to apply meaningful logic to it
        pos_df = position_obj.to_dataframe()
        # Drop rows with missing data in columns: 't', 'velocity_x_smooth' and 2 other columns. This occurs from smoothing
        pos_df = pos_df.dropna(subset=['t', 'x_smooth', 'velocity_x_smooth', 'acceleration_x_smooth']).reset_index(drop=True)
        curr_laps_df = sess.laps.to_dataframe()

        epochs_editor = EpochsEditor.init_laps_diagnoser(pos_df, curr_laps_df, include_velocity=True, include_accel=False)


    """
    pos_df: pd.DataFrame = field(repr=False) # disables super excessive dataframe printing
    curr_laps_df: pd.DataFrame = field()
    on_epoch_region_updated_callback: Optional[Callable] = field()
    on_epoch_region_selection_toggled_callback: Optional[Callable] = field()

    plots: RenderPlots = field(init=False)
    changed_laps_df: pd.DataFrame = field(init=False)

    loaded_track_limits: Optional[Dict] = field(default=None, metadata={'desc':'a dictionary of track limits to be plotted as horizontal lines over the laps'})


    _pos_variable_names = ('x_smooth', 'velocity_x_smooth', 'acceleration_x_smooth')
    # _pos_variable_names = ('x', 'velocity_x', 'acceleration_x')

    def connect_double_click_event(self):
        print("Connecting double-click event...")
        self.plots.viewboxes[0].scene().sigMouseClicked.connect(self.on_mouse_click)

    def on_mouse_click(self, event):
        print(f"EpochsEditor.on_mouse_click(...): Mouse click detected: {event}")
        if event.double():
            print("EpochsEditor.on_mouse_click(...): Double-click detected!")
            pos = event.scenePos()
            viewbox = self.plots.viewboxes[0]
            if viewbox.sceneBoundingRect().contains(pos):
                mouse_point = viewbox.mapSceneToView(pos)
                t_start = mouse_point.x()
                t_end = t_start + 1.0  # Default width of 1.0 units
                new_epoch_label = f'New Epoch {len(self.curr_laps_df) + 1}'
                new_epoch = {
                    'start': t_start,
                    'stop': t_end,
                    'lap_id': len(self.curr_laps_df) + 1,
                    'label': new_epoch_label,
                    'lap_dir': 1,
                    'is_LR_dir': True,
                    'lap_color': DisplayColorsEnum.Laps.LR,
                    'lap_accent_color': '#c4ff26de',
                    'is_included': True
                }
                print(f"\tAdding new epoch: {new_epoch}")
                self.curr_laps_df = self.curr_laps_df.append(new_epoch, ignore_index=True)
                self.add_epoch_region(new_epoch)


        elif event.button() == QtCore.Qt.RightButton:
            ## right-click
            print(f'EpochsEditor.on_mouse_click(event): right-click')

        elif event.button() == QtCore.Qt.MiddleButton:
            ## right-click
            print(f'EpochsEditor.on_mouse_click(event): middle-click')

        else:
            ## Non-double click
            # super().mouseClickEvent(event)
            print(f'EpochsEditor.on_mouse_click(event): unknown event. event: {event}')
            pass


    def show_context_menu(self, screen_pos, mouse_point):
        """ Generate the context menu items for the epoch rectangles

        NOT WORKING

        Adds: 
            - an option to change the direction for the epoch
            - an option to remove the epoch
        """
        print(f'.show_context_menu(screen_pos: {screen_pos}, mouse_point: {mouse_point})')
        context_menu = QtWidgets.QMenu()

        # Find the epoch region that was right-clicked
        clicked_epoch_region = None
        clicked_epoch_label = None

        for label, region in self.plots.lap_epoch_widgets.items():
            if region.getRegion()[0] <= mouse_point.x() <= region.getRegion()[1]:
                clicked_epoch_region = region
                clicked_epoch_label = label
                break

        if clicked_epoch_region is not None:
            # Add "Toggle lap direction" action
            toggle_direction_action = context_menu.addAction("Toggle lap direction")
            context_menu.addSeparator()

            # Add "Remove" action
            remove_action = context_menu.addAction("Remove")

            # Execute the context menu
            action = context_menu.exec_(screen_pos)

            if action == toggle_direction_action:
                self.toggle_lap_direction(clicked_epoch_region, clicked_epoch_label)
            elif action == remove_action:
                self.remove_epoch_at(mouse_point)


    def update_epoch_region_appearance(self, epoch_region, new_color, new_accent_color):
        """Update the visual appearance of an epoch region with new colors."""
        # Create new pen and brush with the updated colors
        new_pen = pg.mkPen(f'{new_color}d6', width=1.0)
        new_brush = pg.mkBrush(f"{new_color}42")
        new_hover_brush = pg.mkBrush(f"{new_color}a8")
        new_hover_pen = pg.mkPen(new_accent_color, width=2.5)

        # Update the region's appearance
        epoch_region.setBrush(new_brush)
        epoch_region.setPen(new_pen)
        epoch_region.setHoverBrush(new_hover_brush)
        epoch_region.setHoverPen(new_hover_pen)

        # Force a redraw
        epoch_region.update()


    def toggle_lap_direction(self, epoch_region, epoch_label):
        """Toggle the lap direction for the specified epoch and update its visual representation."""
        # Get the epoch index from the region's custom_bound_data
        epoch_index = epoch_region.custom_bound_data

        # Toggle the lap direction in the dataframe
        current_direction = self.curr_laps_df.loc[epoch_index, 'is_LR_dir']
        new_direction = not current_direction
        self.curr_laps_df.loc[epoch_index, 'is_LR_dir'] = new_direction

        # Update the lap_dir column as well (if it exists)
        if 'lap_dir' in self.curr_laps_df.columns:
            self.curr_laps_df.loc[epoch_index, 'lap_dir'] = 1 if new_direction else -1

        # Update the colors based on the new direction
        if new_direction:  # LR direction
            new_color = DisplayColorsEnum.Laps.LR
            new_accent_color = '#c4ff26de'
        else:  # RL direction
            new_color = DisplayColorsEnum.Laps.RL
            new_accent_color = '#6227ffde'

        # Update the dataframe with new colors
        self.curr_laps_df.loc[epoch_index, 'lap_color'] = new_color
        self.curr_laps_df.loc[epoch_index, 'lap_accent_color'] = new_accent_color

        # Update the visual representation of the epoch region
        self.update_epoch_region_appearance(epoch_region, new_color, new_accent_color)

        print(f"Toggled lap direction for {epoch_label}: {'LR' if new_direction else 'RL'}")


    def remove_epoch_at(self, mouse_point):
        for label, region in self.plots.lap_epoch_widgets.items():
            if region.getRegion()[0] <= mouse_point.x() <= region.getRegion()[1]:
                print(f"Removing epoch: {label}")
                region.setParentItem(None)  # Remove the region from the plot by setting its parent to None
                del self.plots.lap_epoch_widgets[label]
                del self.plots.lap_epoch_labels[label]
                self.curr_laps_df = self.curr_laps_df[self.curr_laps_df['label'] != label]
                break

    def _remove_all_epochs(self):
        """ removes all epoch graphics 
        """
        for label, region in self.plots.lap_epoch_widgets.items():
            print(f"Removing epoch: {label}")
            region.setParentItem(None)  # Remove the region from the plot by setting its parent to None
            del self.plots.lap_epoch_widgets[label]
            del self.plots.lap_epoch_labels[label]
            # self.curr_laps_df = self.curr_laps_df[self.curr_laps_df['label'] != label] # do not remove the data, just the graphics
        self.plots.lap_epoch_widgets = {}
        self.plots.lap_epoch_labels = {}
        



    def add_epoch_region(self, epoch):
        """ creates a new epoch region, a CustomLinearRegionItem """

        print(f"Creating epoch region for: {epoch}")
        v1 = self.plots.viewboxes[0]
        epoch_linear_region, epoch_region_label = build_pyqtgraph_epoch_indicator_regions(
            v1, t_start=epoch['start'], t_stop=epoch['stop'], epoch_label=epoch['label'], movable=True,
            **dict(pen=pg.mkPen(f'{epoch["lap_color"]}d6', width=1.0), brush=pg.mkBrush(f"{epoch['lap_color']}42"),
                   hoverBrush=pg.mkBrush(f"{epoch['lap_color']}a8"), hoverPen=pg.mkPen(epoch['lap_accent_color'], width=2.5)),
            custom_bound_data=epoch['lap_id']
        )
        self.plots.lap_epoch_widgets[epoch['label']] = epoch_linear_region # CustomLinearRegionItem
        self.plots.lap_epoch_labels[epoch['label']] = epoch_region_label
        epoch_linear_region.sigRegionChangeFinished.connect(self.on_epoch_region_updated)
        epoch_linear_region.sigClicked.connect(self.on_epoch_region_selection_toggled)
        epoch_linear_region.epochs_editor = self
        # epoch_linear_region.getContextMenus = 
        # Connect context menu signal
        epoch_linear_region.setContextMenuPolicy(pg.Qt.CustomContextMenu)
        epoch_linear_region.customContextMenuRequested.connect(self.show_context_menu)    


    def rebuild_epoch_regions(self):
        """ rebuilds the graphics when the `self.curr_laps_df` changes
        
        plots = RenderPlots('lap_debugger_plot', win=win,
            sub_layouts=sub_layouts, viewboxes=viewboxes, position_plots=position_plots,
            lap_epoch_widgets=lap_epoch_widgets, lap_epoch_labels=lap_epoch_labels,
            legend=legend, scatter_points={}, additional_items=additional_items)
            
        """
        self._remove_all_epochs()

        lap_epoch_widgets = {}
        lap_epoch_labels = {}
        
        curr_laps_df = self.curr_laps_df
        v1 = self.plots.viewboxes[0]
        
        for a_lap in curr_laps_df.itertuples():
            epoch_linear_region = lap_epoch_widgets.get(a_lap.label, None)
            if epoch_linear_region is None:
                ## Create a new one:
                # add alpha
                epoch_linear_region, epoch_region_label = build_pyqtgraph_epoch_indicator_regions(v1, t_start=a_lap.start, t_stop=a_lap.stop, epoch_label=a_lap.label, movable=True, removable=True,
                                                                                                   **dict(pen=pg.mkPen(f'{a_lap.lap_color}d6', width=1.0), brush=pg.mkBrush(f"{a_lap.lap_color}42"), hoverBrush=pg.mkBrush(f"{a_lap.lap_color}a8"), hoverPen=pg.mkPen(a_lap.lap_accent_color, width=2.5)), 
                                                                                                  custom_bound_data=a_lap.Index)                
                lap_epoch_widgets[a_lap.label] = epoch_linear_region
                lap_epoch_labels[a_lap.label] = epoch_region_label
                if self.on_epoch_region_updated is not None:
                    epoch_linear_region.sigRegionChangeFinished.connect(self.on_epoch_region_updated)
                
                if self.on_epoch_region_selection_toggled is not None:
                    epoch_linear_region.sigClicked.connect(self.on_epoch_region_selection_toggled)


                # Provide a callback to remove the ROI (and its children) when
                # "remove" is selected from the context menu.
                def remove():
                    self.plots.v1.removeItem(epoch_linear_region)
                epoch_linear_region.sigRemoveRequested.connect(remove)

                # epoch_linear_region.getContextMenus(
        ## END for a_lap in curr_laps_df.it...
        self.plots.lap_epoch_widgets = lap_epoch_widgets
        self.plots.lap_epoch_labels = lap_epoch_labels
        


    @classmethod
    def add_visualization_columns(cls, curr_laps_df: pd.DataFrame) -> pd.DataFrame:
        """ adds 'lap_color', 'lap_accent_color' columns to the laps_df for use in visualization 
        
        NOTE: These hardcoded colors were computed with:
        
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
        from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter, build_adjusted_color
        
        ColorFormatConverter.qColor_to_hexstring(build_adjusted_color(pg.mkColor(DisplayColorsEnum.Laps.odd), value_scale=1.4))
        ColorFormatConverter.qColor_to_hexstring(build_adjusted_color(pg.mkColor(DisplayColorsEnum.Laps.even), value_scale=1.4))
        
        """
        # Add 'lap_color' column to curr_laps_df
        if ('lap_color' not in curr_laps_df.columns) or ('lap_accent_color' not in curr_laps_df.columns):
            curr_laps_df['lap_color'] = DisplayColorsEnum.Laps.RL
            curr_laps_df['lap_accent_color'] = '#6227ffde'

            # lap_direction_df_key: str = 'lap_dir'
            lap_direction_df_key: str = 'is_LR_dir'
            assert lap_direction_df_key in curr_laps_df.columns, f"lap_direction_df_key ({lap_direction_df_key}) missing from curr_laps_df.columns: {list(curr_laps_df.columns)}"
            curr_laps_df.loc[(curr_laps_df['is_LR_dir']), 'lap_color'] = DisplayColorsEnum.Laps.LR
            curr_laps_df.loc[(curr_laps_df['is_LR_dir']), 'lap_accent_color'] = '#c4ff26de'

            # assert 'lap_dir' in curr_laps_df.columns, f"'lap_dir' missing from curr_laps_df.columns: {list(curr_laps_df.columns)}"
            # curr_laps_df.loc[(curr_laps_df['lap_dir'] > 0), 'lap_color'] = DisplayColorsEnum.Laps.LR
            # curr_laps_df.loc[(curr_laps_df['lap_dir'] > 0), 'lap_accent_color'] = '#c4ff26de'


        ## add the 'is_included' column
        if 'is_included' not in curr_laps_df.columns:
            curr_laps_df['is_included'] = True ## all are included by default

        # curr_laps_df['lap_color'] = DisplayColorsEnum.Laps.RL
        # curr_laps_df['lap_accent_color'] = '#6227ffde'
        
        # curr_laps_df.loc[(curr_laps_df['is_included']), 'lap_color'] = DisplayColorsEnum.Laps.LR
        # curr_laps_df.loc[(curr_laps_df['is_included']), 'lap_color'] = DisplayColorsEnum.Laps.LR

        return curr_laps_df


    # INIT METHODS _______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    @function_attributes(short_name=None, tags=['epochs'], input_requires=[], output_provides=[], uses=['build_pyqtgraph_epoch_indicator_regions'], used_by=[], creation_date='2025-09-19 08:43', related_items=[])
    @classmethod
    def perform_plot_laps_diagnoser(cls, pos_df: pd.DataFrame, curr_laps_df: pd.DataFrame, include_velocity=True, include_accel=True, on_epoch_region_updated_callback=None, on_epoch_region_selection_toggled_callback=None, loaded_track_limits=None, grid_bin_bounds=None, epoch_label_position=0.5, epoch_label_rotateAxis=(1,0), epoch_label_anchor=(1, 1), **kwargs):
        """
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.EpochsEditorItem import perform_plot_laps_diagnoser
            plots_obj = EpochsEditor.perform_plot_laps_diagnoser(pos_df, curr_laps_df, include_velocity=True, include_accel=True, on_epoch_region_updated_callback=_on_epoch_region_updated)

        """
        # custom_epoch_label_kwargs = dict(epoch_label_position=0.5, epoch_label_rotateAxis=(1,0), epoch_label_anchor=(1, 1))
        custom_epoch_label_kwargs = dict(epoch_label_position=epoch_label_position, epoch_label_rotateAxis=epoch_label_rotateAxis, epoch_label_anchor=epoch_label_anchor)
        
        # Add 'lap_color' column to curr_laps_df
        curr_laps_df = cls.add_visualization_columns(curr_laps_df=curr_laps_df)


        pos_variable_names = cls._pos_variable_names
        ## Begin Plotting
        win = pg.GraphicsLayoutWidget(show=True)
        win.setWindowTitle('Lap Overlap Debugger')
        win.resize(2000, 400)
        win.ci.setBorder((50, 50, 100))

        position_plots = []
        sub_layouts = []
        viewboxes = []
        additional_items = {}
        
        sub1 = win.addLayout()
        sub1.addLabel("Position (x)")
        # sub1.addLabel(axis='left', text='Recorded Position Data')
        # sub1.addLabel(axis='bottom', text='time')

        sub1.nextRow()
        v1 = CustomViewBox()
        
        track_lines = dict() 

        if grid_bin_bounds is not None:
            # Draws the grid_bin_bounds for x-axis thick horizontal lines ______________________________________________________________ #
            grid_bin_bounds_xlim_start, grid_bin_bounds_xlim_end = grid_bin_bounds[0] ## x-axis
            grid_bin_bounds_xlim_pen_style_dict = dict(color='#ca0000', width=1.0)
    
            track_lines['grid_bin_bounds_xlim_start'] = pg.InfiniteLine(pos=grid_bin_bounds_xlim_start, angle=0, movable=False, pen=pg.mkPen(**grid_bin_bounds_xlim_pen_style_dict), label='grid_bin_bounds_min')
            track_lines['grid_bin_bounds_xlim_end'] = pg.InfiniteLine(pos=grid_bin_bounds_xlim_end, angle=0, movable=False, pen=pg.mkPen(**grid_bin_bounds_xlim_pen_style_dict), label='grid_bin_bounds_max')

        if loaded_track_limits is not None:
            # Draws the track boundaries as dotted horizontal lines ______________________________________________________________ #
            long_xlim_start, long_xlim_end = loaded_track_limits['long_xlim']
            long_xlim_pen_style_dict = dict(color='#b6b6b6ff', width=0.5, style=pg.QtCore.Qt.DashLine)
    
            track_lines['long_xlim_start'] = pg.InfiniteLine(pos=long_xlim_start, angle=0, movable=False, pen=pg.mkPen(**long_xlim_pen_style_dict))
            track_lines['long_xlim_end'] = pg.InfiniteLine(pos=long_xlim_end, angle=0, movable=False, pen=pg.mkPen(**long_xlim_pen_style_dict))
                        
            short_xlim_start, short_xlim_end = loaded_track_limits['short_xlim']
            short_xlim_pen_style_dict = dict(color='#b6b6b6ff', width=0.5, style=pg.QtCore.Qt.DashLine)
    
            track_lines['short_xlim_start'] = pg.InfiniteLine(pos=short_xlim_start, angle=0, movable=False, pen=pg.mkPen(**short_xlim_pen_style_dict), label='loaded_track_limits')
            track_lines['short_xlim_end'] = pg.InfiniteLine(pos=short_xlim_end, angle=0, movable=False, pen=pg.mkPen(**short_xlim_pen_style_dict))
    

        ## Add all lines to the viewbox and store them in the additional_items dictionary
        for k, vline in track_lines.items():
            v1.addItem(vline)
            additional_items[f'{k}_line'] = vline
            



        # Draws the lap positions as a white line _____________________________________________________________________________________ #
        ax_pos = pg.PlotDataItem(x=pos_df.t.to_numpy(), y=pos_df[pos_variable_names[0]].to_numpy(), pen='white', name=pos_variable_names[0]) # Draws the laps as white lines


        v1.setMouseEnabled(x=True, y=False)
        # v1.enableAutoRange(x=False, y=True)
        # v1.setXRange(300, 450)
        # v1.setAutoVisible(x=False, y=True)
        v1.addItem(ax_pos)
        sub1.addItem(v1)

        # legend_size = (80,60) # fixed size legend
        legend_size = None # auto-sizing legend to contents
        legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
        legend.setParentItem(v1.graphicsItem())


        # ax_pos = win.plot(x=pos_df.t.to_numpy(), y=pos_df.x.to_numpy(), pen='white', name=f'pos_x') #  , symbol='o', symbolBrush=pg.intColor(0,6,maxValue=128), symbolBrush=pg.intColor(i,6,maxValue=128)
        legend.addItem(ax_pos, pos_variable_names[0])
        sub_layouts.append(sub1)
        viewboxes.append(v1)
        position_plots.append(ax_pos)
        
        # Emphasize the y=0 crossing by drawing a horizontal line at y=0
        vline1 = pg.InfiniteLine(pos=0, angle=0, movable=False, pen=pg.mkPen(color='#b6b6b6ff', width=1, style=pg.QtCore.Qt.DashLine))
        v1.addItem(vline1)
        additional_items['position_y_zero_line'] = vline1
    

        if include_velocity:
            sub1.nextRow()
            v2 = CustomViewBox()
            ax_velocity = pg.PlotDataItem(x=pos_df.t.to_numpy(), y=pos_df[pos_variable_names[1]].to_numpy(), pen='#21b9ffcb', name=pos_variable_names[1])
            
            v2.setMouseEnabled(x=True, y=False)
            v2.setXLink(v1)
            v2.addItem(ax_velocity)
            sub1.addItem(v2)
            legend.addItem(ax_velocity, pos_variable_names[1])
            position_plots.append(ax_velocity)
            viewboxes.append(v2)
            
            # Emphasize the y=0 crossing by drawing a horizontal line at y=0
            vline = pg.InfiniteLine(pos=0, angle=0, movable=False, pen=pg.mkPen(color='#b6b6b6ff', width=1, style=pg.QtCore.Qt.DashLine))
            v2.addItem(vline)
            additional_items['velocity_y_zero_line'] = vline

            # ax_velocity.showGrid(True, True)  # Show grid for reference


        # win.graphicsItem().setLabel(axis='left', text='Recorded Position Data')
        # win.graphicsItem().setLabel(axis='bottom', text='time')


        movable = kwargs.pop('movable', True)
        removable = kwargs.pop('removable', True)
        
        lap_epoch_widgets = {}
        lap_epoch_labels = {}

        for a_lap in curr_laps_df.itertuples():
            epoch_linear_region = lap_epoch_widgets.get(a_lap.label, None)
            if epoch_linear_region is None:
                ## Create a new one:
                # add alpha
                epoch_linear_region, epoch_region_label = build_pyqtgraph_epoch_indicator_regions(v1, t_start=a_lap.start, t_stop=a_lap.stop, epoch_label=a_lap.label, movable=movable, removable=removable, **dict(pen=pg.mkPen(f'{a_lap.lap_color}d6', width=1.0), brush=pg.mkBrush(f"{a_lap.lap_color}42"), hoverBrush=pg.mkBrush(f"{a_lap.lap_color}a8"), hoverPen=pg.mkPen(a_lap.lap_accent_color, width=2.5)), 
                                                                                                  custom_bound_data=a_lap.Index, **custom_epoch_label_kwargs, **kwargs)               
                lap_epoch_widgets[a_lap.label] = epoch_linear_region
                lap_epoch_labels[a_lap.label] = epoch_region_label
                if on_epoch_region_updated_callback is not None:
                    epoch_linear_region.sigRegionChangeFinished.connect(on_epoch_region_updated_callback)
                
                if on_epoch_region_selection_toggled_callback is not None:
                    epoch_linear_region.sigClicked.connect(on_epoch_region_selection_toggled_callback)


                # Provide a callback to remove the ROI (and its children) when
                # "remove" is selected from the context menu.
                def remove():
                    v1.removeItem(epoch_linear_region)
                epoch_linear_region.sigRemoveRequested.connect(remove)

                # epoch_linear_region.getContextMenus(
                

        
        plots = RenderPlots('lap_debugger_plot', win=win,
            sub_layouts=sub_layouts, viewboxes=viewboxes, position_plots=position_plots,
            lap_epoch_widgets=lap_epoch_widgets, lap_epoch_labels=lap_epoch_labels,
            legend=legend, scatter_points={}, additional_items=additional_items,
            on_epoch_region_updated_callback=on_epoch_region_updated_callback, on_epoch_region_selection_toggled_callback=on_epoch_region_selection_toggled_callback,
            )
        return plots



    @classmethod
    def init_laps_diagnoser(cls, pos_df: pd.DataFrame, curr_laps_df: pd.DataFrame, include_velocity=True, include_accel=True, on_epoch_region_updated_callback=None, on_epoch_region_selection_toggled_callback=None, loaded_track_limits=None, grid_bin_bounds=None, **kwargs):
        """ 

        Usage:
            sess = global_session

            # pos_df = sess.compute_position_laps() # ensures the laps are computed if they need to be:
            position_obj = deepcopy(sess.position)
            position_obj.compute_higher_order_derivatives()
            pos_df = position_obj.compute_smoothed_position_info(N=20) ## Smooth the velocity curve to apply meaningful logic to it
            pos_df = position_obj.to_dataframe()
            # Drop rows with missing data in columns: 't', 'velocity_x_smooth' and 2 other columns. This occurs from smoothing
            pos_df = pos_df.dropna(subset=['t', 'x_smooth', 'velocity_x_smooth', 'acceleration_x_smooth']).reset_index(drop=True)
            curr_laps_df = sess.laps.to_dataframe()

            epochs_editor = EpochsEditor.init_laps_diagnoser(pos_df, curr_laps_df, include_velocity=True, include_accel=False)
        """
        curr_laps_df = cls.add_visualization_columns(curr_laps_df=curr_laps_df)
        _obj = cls(pos_df=pos_df, curr_laps_df=curr_laps_df, on_epoch_region_updated_callback=on_epoch_region_updated_callback, on_epoch_region_selection_toggled_callback=on_epoch_region_selection_toggled_callback, loaded_track_limits=loaded_track_limits)
        _obj.changed_laps_df = _obj.curr_laps_df.iloc[:0,:].copy() # should be in attrs_post_init
        _obj.plots = cls.perform_plot_laps_diagnoser(pos_df, curr_laps_df, include_velocity=include_velocity, include_accel=include_accel, on_epoch_region_updated_callback=_obj.on_epoch_region_updated, on_epoch_region_selection_toggled_callback=_obj.on_epoch_region_selection_toggled, loaded_track_limits=loaded_track_limits, grid_bin_bounds=grid_bin_bounds, **kwargs)
        _obj.connect_double_click_event()  # Connect the double-click event after plots are initialized
        return _obj



    @classmethod
    def init_from_session(cls, sess, include_velocity=True, include_accel=True, on_epoch_region_updated_callback=None, on_epoch_region_selection_toggled_callback=None, grid_bin_bounds=None, **kwargs):
        """ initialize from a session object. Does not modify the session. """
        # pos_df = sess.compute_position_laps() # ensures the laps are computed if they need to be:
        position_obj = copy.deepcopy(sess.position)
        position_obj.compute_higher_order_derivatives()
        pos_df = position_obj.compute_smoothed_position_info(N=20) ## Smooth the velocity curve to apply meaningful logic to it
        pos_df = position_obj.to_dataframe()
        # Drop rows with missing data in columns: 't', 'velocity_x_smooth' and 2 other columns. This occurs from smoothing
        pos_df = pos_df.dropna(subset=['t', 'x_smooth', 'velocity_x_smooth', 'acceleration_x_smooth']).reset_index(drop=True)
        curr_laps_df = sess.laps.to_dataframe()
        loaded_track_limits = copy.deepcopy(sess.config.loaded_track_limits)
        loaded_track_limits['x_midpoint'] = sess.config.x_midpoint
        return cls.init_laps_diagnoser(pos_df=pos_df, curr_laps_df=curr_laps_df, include_velocity=include_velocity, include_accel=include_accel, on_epoch_region_updated_callback=on_epoch_region_updated_callback, on_epoch_region_selection_toggled_callback=on_epoch_region_selection_toggled_callback, loaded_track_limits=loaded_track_limits, grid_bin_bounds=grid_bin_bounds, **kwargs)
    


    # @QtCore.pyqtSlot(object)
    def on_epoch_region_updated(self, epoch_region_item):
        print(f'.on_epoch_region_updated(...): epoch_region_item: {epoch_region_item}, epoch_region_item.custom_bound_data: {epoch_region_item.custom_bound_data}')
        epoch_index = epoch_region_item.custom_bound_data
        
        prev_start, prev_stop = self.curr_laps_df.loc[epoch_index].start, self.curr_laps_df.loc[epoch_index].stop
        # curr_laps_df.loc[epoch_index]
        new_start, new_stop = epoch_region_item.getRegion()
        change_delta = ((prev_start- new_start), (prev_stop-new_stop))
        start_changed = np.logical_not(np.isclose(prev_start, new_start))
        stop_changed = np.logical_not(np.isclose(prev_stop, new_stop))
        either_changed = stop_changed or start_changed
        if either_changed:
            print(f'\teither_changed! change_delta: {change_delta}')
            self.changed_laps_df.loc[epoch_index, :] = self.curr_laps_df.loc[epoch_index] # copy the existing row
            self.changed_laps_df.loc[epoch_index, 'start'] =  new_start
            self.changed_laps_df.loc[epoch_index, 'stop'] =  new_stop
            self.changed_laps_df.loc[epoch_index, 'duration'] =  (new_stop - new_start)

        if self.on_epoch_region_updated_callback is not None:
            self.on_epoch_region_updated_callback(epoch_region_item) # pass through the change
            


    def on_epoch_region_selection_toggled(self, epoch_region_item):
        print(f'.on_epoch_region_selection_toggled(...): epoch_region_item: {epoch_region_item}, epoch_region_item.custom_bound_data: {epoch_region_item.custom_bound_data}')
        epoch_index = epoch_region_item.custom_bound_data       
        print(f'\tepoch_index: {epoch_index}')
         
        # prev_start, prev_stop = self.curr_laps_df.loc[epoch_index].start, self.curr_laps_df.loc[epoch_index].stop
        # # curr_laps_df.loc[epoch_index]
        # new_start, new_stop = epoch_region_item.getRegion()
        # change_delta = ((prev_start- new_start), (prev_stop-new_stop))
        # start_changed = np.logical_not(np.isclose(prev_start, new_start))
        # stop_changed = np.logical_not(np.isclose(prev_stop, new_stop))
        # either_changed = stop_changed or start_changed
        # if either_changed:
        #     print(f'\teither_changed! change_delta: {change_delta}')
        #     self.changed_laps_df.loc[epoch_index, :] = self.curr_laps_df.loc[epoch_index] # copy the existing row
        #     self.changed_laps_df.loc[epoch_index, 'start'] =  new_start
        #     self.changed_laps_df.loc[epoch_index, 'stop'] =  new_stop
        #     self.changed_laps_df.loc[epoch_index, 'duration'] =  (new_stop - new_start)

        ## Toggle the current item
        if 'is_included' not in self.curr_laps_df.columns:
            self.curr_laps_df['is_included'] = True

        ## toggle the current item:
        self.curr_laps_df.loc[epoch_index, 'is_included'] = (not self.curr_laps_df.loc[epoch_index, 'is_included']) ## toggle
        print(f"\tself.curr_laps_df.loc[epoch_index, 'is_included']: {self.curr_laps_df.loc[epoch_index, 'is_included']}")

        # if ('lap_color' not in curr_laps_df.columns) or ('lap_accent_color' not in curr_laps_df.columns):
        #     curr_laps_df['lap_color'] = DisplayColorsEnum.Laps.RL
        #     curr_laps_df['lap_accent_color'] = '#6227ffde'

        #     # lap_direction_df_key: str = 'lap_dir'
        #     lap_direction_df_key: str = 'is_LR_dir'
        #     assert lap_direction_df_key in curr_laps_df.columns, f"lap_direction_df_key ({lap_direction_df_key}) missing from curr_laps_df.columns: {list(curr_laps_df.columns)}"
        #     curr_laps_df.loc[(curr_laps_df['is_LR_dir']), 'lap_color'] = DisplayColorsEnum.Laps.LR
        #     curr_laps_df.loc[(curr_laps_df['is_LR_dir']), 'lap_accent_color'] = '#c4ff26de'

        #     # assert 'lap_dir' in curr_laps_df.columns, f"'lap_dir' missing from curr_laps_df.columns: {list(curr_laps_df.columns)}"
        #     # curr_laps_df.loc[(curr_laps_df['lap_dir'] > 0), 'lap_color'] = DisplayColorsEnum.Laps.LR
        #     # curr_laps_df.loc[(curr_laps_df['lap_dir'] > 0), 'lap_accent_color'] = '#c4ff26de'
            

        if self.on_epoch_region_selection_toggled_callback is not None:
            self.on_epoch_region_selection_toggled_callback(epoch_region_item) # pass through the change
            


    def get_user_labeled_epochs_df(self) -> pd.DataFrame:
        """ returns the complete epochs_df with user-labeled changes. returns all epochs, not just the ones modified. """
        ## Extract changes from `epochs_editor`
        user_labeled_laps_df: pd.DataFrame = copy.deepcopy(self.curr_laps_df)
        user_labeled_laps_df.loc[self.changed_laps_df.index, :] = self.changed_laps_df ## (get only the changed laps)?
        user_labeled_laps_df['duration'] = user_labeled_laps_df['stop'] - user_labeled_laps_df['start']
        # Select columns: 'lap_dir', 'start' and 4 other columns
        user_labeled_laps_df = user_labeled_laps_df.loc[:, ['lap_dir', 'start', 'stop', 'lap_id', 'label', 'duration', 'is_LR_dir']]
        ## sort any newly added epochs
        user_labeled_laps_df = user_labeled_laps_df.sort_values(by=['start', 'stop'], ascending=True, axis='index', ignore_index=True)
        
        return user_labeled_laps_df
    



    def set_user_labeled_epochs_df(self, user_labeled_laps_df: pd.DataFrame):
        """ returns the complete epochs_df with user-labeled changes. returns all epochs, not just the ones modified.
        """
        user_labeled_laps_df = self.add_visualization_columns(curr_laps_df=copy.deepcopy(user_labeled_laps_df))
        user_labeled_laps_df['duration'] = user_labeled_laps_df['stop'] - user_labeled_laps_df['start']
        # Select columns: 'lap_dir', 'start' and 4 other columns
        user_labeled_laps_df = user_labeled_laps_df.loc[:, ['lap_dir', 'start', 'stop', 'lap_id', 'label', 'duration', 'is_LR_dir']]
        ## sort any newly added epochs
        user_labeled_laps_df = user_labeled_laps_df.sort_values(by=['start', 'stop'], ascending=True, axis='index', ignore_index=True)
        
        self.curr_laps_df = user_labeled_laps_df ## Actualy perform the assignment
        self.changed_laps_df = self.curr_laps_df.iloc[:0,:].copy() # update changed laps
        self.rebuild_epoch_regions()
        print(f'laps updated.')
    


    def add_indicies_points_scatter(self, points_indicies, points_name_str: str = 'asc_crossing_midpoint_idxs', points_kwargs = dict(symbol='o', size=10, pen={'color': 'w', 'width': 1}), add_subplot_index=None):
        if add_subplot_index is None:
            add_subplot_index = np.arange(len(self.plots.viewboxes)) # include all

        pos_variable_names = self._pos_variable_names # ('x_smooth', 'velocity_x_smooth', 'acceleration_x_smooth')

        for i in add_subplot_index:
            active_var_name = pos_variable_names[i]
            scatter_plot_name_str: str = f'{points_name_str}_scatter_plot_{active_var_name}'
            
            if scatter_plot_name_str in self.plots.scatter_points:
                a_scatter_plot = self.plots.scatter_points[scatter_plot_name_str]
                spots = [{'pos': a_pos, 'data': an_idx} for an_idx, a_pos in zip(points_indicies, self.pos_df.loc[points_indicies, ['t', active_var_name]].to_numpy())]
                self.plots.scatter_points[scatter_plot_name_str].setData(spots)
            else:               
                self.plots.scatter_points[scatter_plot_name_str] = pg.ScatterPlotItem(name=points_name_str, pxMode=True, **points_kwargs, hoverable=True)
                self.plots.scatter_points[scatter_plot_name_str].setObjectName(scatter_plot_name_str) # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
                self.plots.scatter_points[scatter_plot_name_str].opts['useCache'] = True

                spots = [{'pos': a_pos, 'data': an_idx} for an_idx, a_pos in zip(points_indicies, self.pos_df.loc[points_indicies, ['t', active_var_name]].to_numpy())]
                self.plots.scatter_points[scatter_plot_name_str].addPoints(spots)

                v1 = self.plots.viewboxes[i]
                v1.addItem(self.plots.scatter_points[scatter_plot_name_str])
            
        ## add to all position plots:
        # active_viewboxes = [v1 for i, v1 in enumerate(self.plots.scatter_points.viewboxes) if i in add_subplot_index]
        # for v1 in active_viewboxes:
        #     v1.addItem(self.plots.scatter_points[scatter_plot_name_str])

        return self.plots.scatter_points[scatter_plot_name_str]



    def add_lap_split_points(self, lap_change_indicies):
        """ NOTE: lap specific
        
        lap_change_indicies = _subfn_perform_estimate_lap_splits_1D(pos_df, hardcoded_track_midpoint_x=None, debug_trace_epochs_editor=epochs_editor, debug_print=True) # allow smart midpoint determiniation
        
        """
        
        (desc_crossing_begining_idxs, desc_crossing_midpoint_idxs, desc_crossing_ending_idxs), (asc_crossing_begining_idxs, asc_crossing_midpoint_idxs, asc_crossing_ending_idxs), hardcoded_track_midpoint_x = lap_change_indicies

        # Add track position midpoint:
        v1 = self.plots.viewboxes[0]
        vline_track_midpoint = pg.InfiniteLine(pos=hardcoded_track_midpoint_x, angle=0, movable=False, pen=pg.mkPen(color='#ffe345be', width=2, style=pg.QtCore.Qt.DashLine), name='track_midpoint')
        v1.addItem(vline_track_midpoint)
        self.plots.additional_items['position_track_midpoint_line'] = vline_track_midpoint

        # midpoint_kwargs = dict(symbol='+', pen={'color': '#222222', 'width': 0.5}, brush='#ff73006e', size=20)
        # endpoint_beginning_kwargs = dict(symbol='arrow_right', pen={'color': '#222222', 'width': 0.5}, brush='#ffe0586e', size=40)
        # endpoint_ending_kwargs = dict(symbol='arrow_left', pen={'color': '#222222', 'width': 0.5}, brush='#ffe0586e', size=40)

        midpoint_kwargs = dict(symbol='o', pen={'color': '#222222', 'width': 0.5}, brush='#ff7300dc', size=10)
        endpoint_beginning_kwargs = dict(symbol='o', pen={'color': '#222222', 'width': 0.5}, brush='#ffe058dc', size=5)
        endpoint_ending_kwargs = dict(symbol='o', pen={'color': '#222222', 'width': 0.5}, brush='#ffe058dc', size=5)
        
        # desc_crossing_midpoint_idxs, asc_crossing_midpoint_idxs
        for a_name, an_idxs in zip(('desc_crossing_midpoint_idxs', 'asc_crossing_midpoint_idxs'), (desc_crossing_midpoint_idxs, asc_crossing_midpoint_idxs)):
            an_item = self.add_indicies_points_scatter(an_idxs, points_name_str=a_name, points_kwargs=midpoint_kwargs, add_subplot_index=[0,1])

        # desc_crossing_begining_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_ending_idxs
        # for a_name, an_idxs in zip(('desc_crossing_begining_idxs', 'desc_crossing_ending_idxs', 'asc_crossing_begining_idxs', 'asc_crossing_ending_idxs'), (desc_crossing_begining_idxs, desc_crossing_ending_idxs, asc_crossing_begining_idxs, asc_crossing_ending_idxs)):
        # 	an_item = epochs_editor.add_indicies_points_scatter(an_idxs, points_name_str=a_name, points_kwargs=endpoint_kwargs, add_subplot_index=[0,1])

        for a_name, an_idxs in zip(('desc_crossing_begining_idxs', 'asc_crossing_begining_idxs'), (desc_crossing_begining_idxs, asc_crossing_begining_idxs)):
            an_item = self.add_indicies_points_scatter(an_idxs, points_name_str=a_name, points_kwargs=endpoint_beginning_kwargs, add_subplot_index=[0,1])
            
        for a_name, an_idxs in zip(('desc_crossing_ending_idxs', 'asc_crossing_ending_idxs'), (desc_crossing_ending_idxs, asc_crossing_ending_idxs)):
            an_item = self.add_indicies_points_scatter(an_idxs, points_name_str=a_name, points_kwargs=endpoint_ending_kwargs, add_subplot_index=[0,1])


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Example data
    pos_data = {
        't': np.linspace(0, 100, 500),
        'x_smooth': np.sin(np.linspace(0, 10, 500)),
        'velocity_x_smooth': np.cos(np.linspace(0, 10, 500)),
        'acceleration_x_smooth': np.sin(np.linspace(0, 10, 500)) * np.cos(np.linspace(0, 10, 500))
    }
    pos_df = pd.DataFrame(pos_data)

    laps_data = {
        'start': [10, 30, 50, 70, 90],
        'stop': [20, 40, 60, 80, 100],
        'lap_id': [1, 2, 3, 4, 5],
        'label': ['Lap 1', 'Lap 2', 'Lap 3', 'Lap 4', 'Lap 5'],
        'lap_dir': [1, -1, 1, -1, 1],
        'is_LR_dir': [True, False, True, False, True]
    }
    curr_laps_df = pd.DataFrame(laps_data)

    epochs_editor = EpochsEditor.init_laps_diagnoser(pos_df, curr_laps_df, include_velocity=True, include_accel=True)
    epochs_editor.plots.win.show()  # Ensure the window is shown

    sys.exit(app.exec_())







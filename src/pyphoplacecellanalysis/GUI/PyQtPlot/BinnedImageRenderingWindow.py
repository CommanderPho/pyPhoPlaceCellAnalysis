import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import nptyping as ND
from nptyping import NDArray
from copy import deepcopy
import attrs
from attrs import define, field, Factory, asdict, astuple
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtWidgets, mkQApp, QtGui
from pyphoplacecellanalysis.External.pyqtgraph.colormap import ColorMap
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.NonUniformImage import NonUniformImage

from neuropy.utils.mixins.binning_helpers import get_bin_centers

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

# For scrollable BasicBinnedImageRenderingWindow
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, _perform_build_root_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_with_nested_viewbox_ui
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors

class BasicBinnedImageRenderingHelpers:
    @classmethod
    def _add_bin_ticks(cls, plot_item, xbins=None, ybins=None, grid_opacity:float=0.65, extra_space_for_labels=12):
        """ adds the ticks/grid for xbins and ybins to the plot_item """
        # show full frame, label tick marks at top and left sides, with some extra space for labels:
        plot_item.showAxes(True, showValues=(True, True, False, False), size=extra_space_for_labels)
        # define major tick marks and labels:
        if xbins is not None:
            xticks = [(idx, label) for idx, label in enumerate(xbins)]
            for side in ('top','bottom'):
                plot_item.getAxis(side).setStyle(showValues=False)
                plot_item.getAxis(side).setTicks((xticks, [])) # add list of major ticks; no minor ticks
        if ybins is not None:
            yticks = [(idx, label) for idx, label in enumerate(ybins)]
            for side in ('left','right'):
                plot_item.getAxis(side).setStyle(showValues=False)
                plot_item.getAxis(side).setTicks((yticks, [])) # add list of major ticks; no minor ticks
        plot_item.showGrid(x=True, y=True, alpha=grid_opacity)
        return plot_item

    @classmethod
    def _build_binned_imageItem(cls, plot_item: pg.PlotItem, params, xbins=None, ybins=None, matrix=None, name='avg_velocity', data_label='Avg Velocity', color_bar_mode=None, use_bin_index_axes:bool=True) -> Tuple[RenderPlots, RenderPlotsData]:
        """ Builds and wraps a new `pg.ImageItem` 
        
        color_bar_mode: options for the colorbar of each image
            ### curr_cbar_mode: 'each', 'one', None
        """
        assert color_bar_mode in ('each', 'one', None)
        
        local_plots_data = RenderPlotsData(name=name, matrix=None, matrix_min=None, matrix_max=None, xbins=None, ybins=None, x_min=None, x_max=None, y_min=None, y_max=None)
        local_plots_data.matrix = matrix.copy()
        local_plots_data.matrix_min = np.nanmin(matrix)
        local_plots_data.matrix_max = np.nanmax(matrix)
        
        n_xbins, n_ybins = np.shape(local_plots_data.matrix)

        if xbins is None:
            local_plots_data.xbins = np.arange(n_xbins)
        else:
            if (len(xbins) == (n_xbins + 1)):
                # have edges, get centers
                xbins = get_bin_centers(xbins)
                
            assert len(xbins) == n_xbins
            local_plots_data.xbins = deepcopy(xbins)

        if ybins is None:
            local_plots_data.ybins = np.arange(n_ybins)
        else:
            if (len(ybins) == (n_ybins + 1)):
                # have edges, get centers
                ybins = get_bin_centers(ybins)
        
            assert len(ybins) == n_ybins
            local_plots_data.ybins = deepcopy(ybins)


        if use_bin_index_axes or (local_plots_data.xbins is None):
            local_plots_data.x_min = 0
            local_plots_data.x_max = n_xbins
        else:
            local_plots_data.x_min = np.nanmin(local_plots_data.xbins)
            local_plots_data.x_max = np.nanmax(local_plots_data.xbins)

        if use_bin_index_axes or (local_plots_data.ybins is None):
            local_plots_data.y_min = 0
            local_plots_data.y_max = n_ybins
        else:
            local_plots_data.y_min = np.nanmin(local_plots_data.ybins)
            local_plots_data.y_max = np.nanmax(local_plots_data.ybins)
        
        # plotItem.invertY(True)           # orient y axis to run top-to-bottom
        
        local_plots = RenderPlots(name=name, imageItem=None, colorBarItem=None, matrixBoundaryRectItem=None)
        # Normal ImageItem():
        local_plots.imageItem = pg.ImageItem(matrix.T)
        plot_item.addItem(local_plots.imageItem)

        plot_item.disableAutoRange()

        plot_item.setAspectLocked(lock=True, ratio=1)

        view_padding: float = 0

        # Set up the view range
        plot_item.setXRange(local_plots_data.x_min, local_plots_data.x_max, padding=view_padding)
        plot_item.setYRange(local_plots_data.y_min, local_plots_data.y_max, padding=view_padding)

        # Disable auto range
        plot_item.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        plot_item.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)

        # Draw the boundary as a thick rectangle
        rect = QtCore.QRectF(local_plots_data.x_min, local_plots_data.y_min, (local_plots_data.x_max - local_plots_data.x_min), (local_plots_data.y_max - local_plots_data.y_min))
        matrix_boundary_pen_color: QtCore.QColor = pg.mkColor('#ffffff')
        matrix_boundary_pen_color.setAlphaF(0.7)
        pen = pg.mkPen(matrix_boundary_pen_color, width=4)  # Adjust the color and thickness as needed
        local_plots.matrixBoundaryRectItem = pg.QtGui.QGraphicsRectItem(rect)
        local_plots.matrixBoundaryRectItem.setPen(pen)
        plot_item.addItem(local_plots.matrixBoundaryRectItem)

        # Mask the outside area to be transparent
        plot_item.setClipToView(True)

        # Color Map:
        if hasattr(params, 'colorMap'):
            colorMap = params.colorMap
        else:
            colorMap = pg.colormap.get("viridis")      
            
        if color_bar_mode is None:
            local_plots.colorBarItem = None # no colorbar item
            ## Still need to setup the colormap on the image
            lut = colorMap.getLookupTable(0.0, 1.0)
            local_plots.imageItem.setLookupTable(lut)
            local_plots.imageItem.setLevels([local_plots_data.matrix_min, local_plots_data.matrix_max])
            
        else:
            if color_bar_mode == 'each':   
                # generate an adjustabled color bar
                local_plots.colorBarItem = pg.ColorBarItem(values=(0,1), colorMap=colorMap, label=data_label)
                # link color bar and color map to correlogram, and show it in plotItem:
                local_plots.colorBarItem.setImageItem(local_plots.imageItem, insert_in=plot_item)        
                # Set the colorbar to the range:
                local_plots.colorBarItem.setLevels(low=local_plots_data.matrix_min, high=local_plots_data.matrix_max)
            else:
                ## TODO: globally shared colorbar item:
                # local_plots.colorBarItem = self.params.shared_colorBarItem # shared colorbar item
                local_plots.colorBarItem = None # shared colorbar item
                
        return local_plots, local_plots_data


class SliderUpdatedBinnedImageRenderingMixin:
    """ implementor has data controlled by a slider
    
    ## Build the main widget:
    markov_order_idx: int = 5
    window_title = f'TransitionMatrix: For each decoder (markov order {markov_order_idx+1})'
    a_spec = [ # two rows: (2 columns, 1 column)
        ({'long_LR': binned_x_transition_matrix_higher_order_list_dict['long_LR'][markov_order_idx]}, {'short_LR': binned_x_transition_matrix_higher_order_list_dict['short_LR'][markov_order_idx]},), # single row (2 columns)
        ({'long_RL': binned_x_transition_matrix_higher_order_list_dict['long_RL'][markov_order_idx]}, {'short_RL': binned_x_transition_matrix_higher_order_list_dict['short_RL'][markov_order_idx]},), # single row (2 columns)
        # ({'diff':_diff}, )
    ]
    out_test_markov_test_compare_widget: BasicBinnedImageRenderingWidget = BasicBinnedImageRenderingWidget.init_from_data_spec(a_spec, window_title=window_title)
    out_test_markov_test_compare_widget._init_SliderUpdatedBinnedImageRenderingMixin(binned_x_transition_matrix_higher_order_list_dict=deepcopy(binned_x_transition_matrix_higher_order_list_dict),
                                                                                    n_powers=n_powers)

                                                                                     
    """


    @property
    def slider_curr_idx(self) -> int:
        """The slider_curr_idx property."""
        return self.params.curr_markov_order_idx
    @slider_curr_idx.setter
    def slider_curr_idx(self, value):
        self.params.curr_markov_order_idx = value

    @property
    def slider_max_idx(self) -> int:
        """The slider_curr_idx property."""
        return self.params.n_powers


    @property
    def slider_plot_data(self) -> Dict[str, List[NDArray]]:
        """The plot_names property."""
        # return self.plots_data.binned_x_transition_matrix_higher_order_list_dict
        return self.plots_data['SliderUpdatedBinnedImageRenderingMixin']


    def SliderUpdatedBinnedImageRenderingMixin_sliderValueChanged(self, new_val: int):
        """ captures: out_test_markov_test_compare_widget, binned_x_transition_matrix_higher_order_list_dict
        
        """
        print(f'SliderUpdatedBinnedImageRenderingMixin_sliderValueChanged valueChanged(new_val: {new_val})')
        markov_order_idx = int(round(new_val))
        # self.params.curr_markov_order_idx = markov_order_idx ## update the interval variable
        self.slider_curr_idx = markov_order_idx ## update the interval variable
        slider_plot_data = self.plots_data['SliderUpdatedBinnedImageRenderingMixin']
        
        window_title = f'TransitionMatrix: For each decoder (markov order {markov_order_idx+1})'
        updated_spec = [ # two rows: (2 columns, 1 column)
            ({'long_LR': slider_plot_data['long_LR'][markov_order_idx]}, {'short_LR': slider_plot_data['short_LR'][markov_order_idx]},), # single row (2 columns)
            ({'long_RL': slider_plot_data['long_RL'][markov_order_idx]}, {'short_RL': slider_plot_data['short_RL'][markov_order_idx]},), # single row (2 columns)
        ]
        self.update_from_data_spec(updated_spec, window_title=window_title)
        


    def SliderUpdatedBinnedImageRenderingMixin_build_utility_controls(self, root_dockAreaWindow):
        """ Build the utility controls at the bottom """
        
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from pyphoplacecellanalysis.GUI.Qt.Widgets.ScrollBarWithSpinBox.ScrollBarWithSpinBox import ScrollBarWithSpinBox
        
        ctrls_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)

        ctrls_widget = ScrollBarWithSpinBox()
        ctrls_widget.setObjectName("ctrls_widget")
        ctrls_widget.update_range(0, (self.params.n_powers-1))
        ctrls_widget.setValue(self.params.curr_markov_order_idx)
            
        def valueChanged(new_val: int):
            """ captures: out_test_markov_test_compare_widget, binned_x_transition_matrix_higher_order_list_dict
            
            """
            self.SliderUpdatedBinnedImageRenderingMixin_sliderValueChanged(new_val)
        

        ctrls_widget_connection = ctrls_widget.sigValueChanged.connect(valueChanged)
        ctrl_layout = pg.LayoutWidget()
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.layout.setSpacing(0)

        # ctrl_layout.setSp
        ctrl_layout.addWidget(ctrls_widget, row=1, rowspan=1, col=1, colspan=2)
        ctrl_widgets_dict = dict(ctrls_widget=ctrls_widget, ctrls_widget_connection=ctrls_widget_connection)

        # Set size policy to expand horizontally but minimize height
        size_policy = pg.QtGui.QSizePolicy(pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum)
        ctrl_layout.setSizePolicy(size_policy)
        # Set the maximum height
        ctrl_layout.setMaximumHeight(26)  # Replace 'max_height' with the desired maximum height in pixels

        # _out_dock_widgets['bottom_controls'] = root_dockAreaWindow.add_display_dock(identifier='bottom_controls', widget=ctrl_layout, dockSize=(600,200), dockAddLocationOpts=['bottom'], display_config=ctrls_dock_config)
        ctrls_dock_widgets_dict = {}
        ctrls_dock_widgets_dict['bottom_controls'] = root_dockAreaWindow.add_display_dock(identifier='bottom_controls', widget=ctrl_layout, dockSize=(600,42), dockAddLocationOpts=['bottom'], display_config=ctrls_dock_config)
        ctrls_dock_widgets_dict['bottom_controls'][1].hideTitleBar() # hide the dock title bar
        
        ## Update the UI directly:
        self.ui.ctrls_widget = ctrls_widget
        self.ui.on_valueChanged=valueChanged
        self.ui.ctrl_layout = ctrl_layout
        self.ui.ctrls_widget_connection = ctrls_widget_connection
        
        self.ui.ctrls_dock_widgets = PhoUIContainer(name='ui.ctrls_dock_widgets', **ctrls_dock_widgets_dict)

        # ui_dict = dict(ctrl_layout=ctrl_layout, **ctrl_widgets_dict, on_valueChanged=valueChanged)
        # return ui_dict, ctrls_dock_widgets_dict
        

    def _init_SliderUpdatedBinnedImageRenderingMixin(self, binned_x_transition_matrix_higher_order_list_dict, n_powers: int = 200):
        """
        Init
        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
        

        self.plots_data['SliderUpdatedBinnedImageRenderingMixin'] = RenderPlotsData(name='binned_x_transition_matrix_higher_order_list_dict', **binned_x_transition_matrix_higher_order_list_dict)
        self.params.n_powers = n_powers
        self.params.curr_markov_order_idx = 0

        # build a win of type PhoDockAreaContainingWindow
        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title=self.params.window_title, defer_show=True)

        self.ui.root_dockAreaWindow = root_dockAreaWindow
        self.ui.app = app
        
        # common_kwargs_dict = dict(dockSize=(final_desired_width, final_desired_height))

        display_config1 = CustomDockDisplayConfig(showCloseButton=False)
        _, dDisplayItem1 = root_dockAreaWindow.add_display_dock("TransitionMatricies", widget=self, dockAddLocationOpts=['left'], display_config=display_config1)
        # display_config2 = CustomDockDisplayConfig(showCloseButton=False)
        # _, dDisplayItem2 = root_dockAreaWindow.add_display_dock("Bottom", widget=None, dockAddLocationOpts=['right'], display_config=display_config2)

        self.SliderUpdatedBinnedImageRenderingMixin_build_utility_controls(root_dockAreaWindow=root_dockAreaWindow)

        self.ui.root_dockAreaWindow.show()
        

@metadata_attributes(short_name=None, tags=['binning', 'image', 'window', 'standalone', 'widget'], input_requires=[], output_provides=[], uses=['_perform_build_root_graphics_layout_widget_ui', 'LayoutScrollability'], used_by=[], creation_date='2023-10-19 02:28', related_items=[])
class BasicBinnedImageRenderingMixin:
    """ Renders a Matrix of binned data in the window.NonUniformImage and includes no histogram.
        NOTE: uses basic pg.ImageItem instead of pg.
        Observed to work well to display simple binned heatmaps/grids such as avg velocity across spatial bins, etc.    
        
        History:
            Based off of pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_Matrix.MatrixRenderingWindow
            2024-08-13 Factored out of `BasicBinnedImageRenderingWindow`
            
            
        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
            out = BasicBinnedImageRenderingWindow(active_eloy_analysis.avg_2D_speed_per_pos, active_pf_2D_dt.xbin_labels, active_pf_2D_dt.ybin_labels, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity', scrollability_mode=LayoutScrollability.SCROLLABLE)
            out.add_data(row=1, col=0, matrix=active_eloy_analysis.pf_overlapDensity_2D, xbins=active_pf_2D_dt.xbin_labels, ybins=active_pf_2D_dt.ybin_labels, name='pf_overlapDensity', title='pf overlapDensity metric', variable_label='pf overlapDensity')
            out.add_data(row=2, col=0, matrix=active_pf_2D.ratemap.occupancy, xbins=active_pf_2D.xbin, ybins=active_pf_2D.ybin, name='occupancy_seconds', title='Seconds Occupancy', variable_label='seconds')
            out.add_data(row=3, col=0, matrix=active_simpler_pf_densities_analysis.n_neurons_meeting_firing_critiera_by_position_bins_2D, xbins=active_pf_2D.xbin, ybins=active_pf_2D.ybin, name='n_neurons_meeting_firing_critiera_by_position_bins_2D', title='# neurons > 1Hz per Pos (X, Y)', variable_label='# neurons')

            
        NOTE:
            Label for `title` is too large, needs to changed to a smaller font


        from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
        out = BasicBinnedImageRenderingWindow(active_eloy_analysis.avg_2D_speed_per_pos, active_pf_2D_dt.xbin_labels, active_pf_2D_dt.ybin_labels, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity')


    """
    
    @property
    def plot_names(self) -> List[str]:
        """The plot_names property."""
        return [v for v in list(self.plots.keys()) if v not in ('name', 'context')]

    @property
    def graphics_layout(self) -> pg.GraphicsLayoutWidget:
        return self.ui.graphics_layout
    

    def update_columns_if_needed(self, new_num_columns: int):
        """ called to update where the label is positioned """
        if (new_num_columns > self.params.max_num_columns):
            self.params.max_num_columns = new_num_columns
            self.ui.graphics_layout.removeItem(self.ui.mainLabel)  # Remove the old item
            self.ui.graphics_layout.addItem(self.ui.mainLabel, row=0, col=1, colspan=self.params.max_num_columns)
            return True
        else:
            return False
        
    def build_formatted_title_string(self, title: str) -> str:
        return f"<span style = 'font-size : 12px;' >{title}</span>"
        
    # ==================================================================================================================== #
    # Data Add/Remove Methods                                                                                              #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['add'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-12 23:51', related_items=['remove_data'])
    def add_data(self, row=1, col=0, rowspan=1, colspan=1,
                 matrix=None, xbins=None, ybins=None, name='avg_velocity', title=None, variable_label=None, drop_below_threshold:float=None,
                 target_layout=None,
                 defer_column_update:bool=False, replace:bool=False):
        """ adds a new data subplot to the output
        
        if it exists and `replace=True` it will be removed and then re-inserted
        
        """
        if drop_below_threshold is None:
            drop_below_threshold: float = self.params.get('drop_below_threshold', 0.0000001) # try to get the default from params, otherwise fallback
                    

        if target_layout is None:
            target_layout = self.ui.graphics_layout
        
        # Format data:
        if (drop_below_threshold is not None) and (drop_below_threshold > 0):
            matrix = matrix.astype(float) # required because NaN isn't available in Integer dtype arrays (in case the matrix is of integer type, this prevents a ValueError)
            matrix[np.where(matrix < drop_below_threshold)] = np.nan # null out the occupancy


        # if title is None:
        #     title = f"{name}" # "Avg Velocity per Pos (X, Y)"
            
        if variable_label is None:
            variable_label = f"{name}"
            

        ## Check for existance:
        needs_create_new: bool = True
        extant_local_plots_data = self.plots_data.get(name, None)
        extant_local_plots = self.plots.get(name, None)
        
        if (extant_local_plots is not None) and (extant_local_plots_data is not None):
            # Local plot exists:
            if self.params.debug_print:
                print(f'local plot named "{name}" already exists!')
            needs_create_new = False ## Set needs_create_new to False
            newPlotItem = extant_local_plots['mainPlotItem'] # : pg.PlotItem
            local_plots = extant_local_plots
            local_plots_data = extant_local_plots_data
            
            if replace:
                if self.params.debug_print:
                    print(f'\t... but replace=True, so we will remove the old item and add a new one!')
                ## remove old objects
                self.remove_data(name=name)
                local_plots = None
                local_plots_data = None
                # newPlotItem.removeItem(
                needs_create_new = True
                


        if needs_create_new:
            newPlotItem: pg.PlotItem = target_layout.addPlot(title=title, row=(self.params.plot_row_offset + row), col=col, rowspan=rowspan, colspan=colspan) # add PlotItem to the main GraphicsLayoutWidget
            newPlotItem.setDefaultPadding(0.0)  # plot without padding data range
            

        ## Common formatting:    
        # Set the plot title:
        formatted_title = self.build_formatted_title_string(title=title)        
        newPlotItem.setTitle(formatted_title)
        
        newPlotItem.setDefaultPadding(0.0)  # plot without padding data range
        newPlotItem.setMouseEnabled(x=False, y=False)
        
        if needs_create_new:
            newPlotItem = BasicBinnedImageRenderingHelpers._add_bin_ticks(plot_item=newPlotItem, xbins=xbins, ybins=ybins, grid_opacity=self.params.grid_opacity)

        if needs_create_new:
            local_plots, local_plots_data = BasicBinnedImageRenderingHelpers._build_binned_imageItem(newPlotItem, params=self.params, xbins=xbins, ybins=ybins, matrix=matrix, name=name, data_label=variable_label, color_bar_mode=self.params.color_bar_mode)
            
        self.plots_data[name] = local_plots_data
        self.plots[name] = local_plots
        self.plots[name].mainPlotItem = newPlotItem


        if self.params.color_bar_mode == 'one':
            self.plots[name].colorBarItem = self.params.shared_colorBarItem # shared colorbar item
            self._update_global_shared_colorbaritem()
        
        if (needs_create_new and self.params.wants_crosshairs):
            self.add_crosshairs(newPlotItem, matrix, xbins=xbins, ybins=ybins, name=name)

        ## Scrollable-support:
        ## TODO: this assumes that provided `row` is the maximum row, or that we fill each column before filling rows
        # active_num_rows = row+1 # get the number of rows after adding the data (adding one to go from an index to a count)
        
        # end_row: int = (row + (rowspan-1))
        end_row: int = ((self.params.plot_row_offset + row) + (rowspan-1))
        active_num_rows = end_row + 1 # get the number of rows after adding the data (adding one to go from an index to a count)
        

        if self.params.scrollability_mode.is_scrollable:
            self.params.single_plot_fixed_height = 80.0
            self.params.all_plots_height = float(active_num_rows) * float(self.params.single_plot_fixed_height)
            assert target_layout == self.ui.graphics_layout, f"target_layout is not the root graphics_layout, so not sure how to constrain its height..."
            self.ui.graphics_layout.setFixedHeight(self.params.all_plots_height)
            # self.ui.graphics_layout.setMinimumHeight(self.params.all_plots_height)
            
        if (not defer_column_update):
            _did_update = self.update_columns_if_needed(new_num_columns=col)

    @function_attributes(short_name=None, tags=['remove'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-12 23:50', related_items=['add_data'])
    def remove_data(self, name: str, remove_connections:bool=True) -> bool:
        """ remove the data, but not the plot """
        extant_local_plots_data = self.plots_data.pop(name, None)
        extant_local_plots = self.plots.pop(name, None)
        if remove_connections:
            extant_local_connections = self.ui.connections.pop(name, None)
        else:
            extant_local_connections = None
            
        if (extant_local_plots is not None) and (extant_local_plots_data is not None):
            # Local plot exists:
            if self.params.debug_print:
                print(f'Removing data named "{name}"!')
            mainPlotItem = extant_local_plots.get('mainPlotItem', None)
            assert mainPlotItem is not None
            
            # mainPlotItem = extant_local_plots.pop('mainPlotItem', None)
            # old_imageItem = extant_local_plots.pop('imageItem', None)
            
            ## Disconnect crosshairs signal:
            if extant_local_connections is not None:
                extant_local_connections.disconnect()
                # extant_local_connections.delte

            # # remove old image item:
            # mainPlotItem.removeItem(old_imageItem)

            removed_items_keys = []
            for k, v in extant_local_plots.items():
                if (k != 'mainPlotItem'):
                    mainPlotItem.removeItem(v)
                    removed_items_keys.append(k)
            if self.params.debug_print:
                print(f'removed: {removed_items_keys}')
            
            # if self.params.color_bar_mode == 'one':
            #     self.plots[name].colorBarItem = self.params.shared_colorBarItem # shared colorbar item
            #     self._update_global_shared_colorbaritem()
                
            ## remove non-shared color bars
            # remove old keys after finishing:
            for k in removed_items_keys:
                extant_local_plots.pop('k', None)

            ## remove plot item:
            self.ui.graphics_layout.removeItem(mainPlotItem)
            # mainPlotItem.deleteLater()
            
            return True
                            
        else:
            print(f'WARN: Cannot remove, no local plot named "{name}" exists!')
            return False


    def _update_global_shared_colorbaritem(self):
        ## Add Global Colorbar for single colorbar mode:
        # Get all data for the purpose of computing global min/max:
        all_pf_plot_data = [self.plots_data[a_plot_name] for a_plot_name in self.plots_data.dynamically_added_attributes] # all plot items PlotItem
        all_pf_plot_data_mins = np.array([a_dataum.matrix_min for a_dataum in all_pf_plot_data])
        all_pf_plot_data_maxes = np.array([a_dataum.matrix_max for a_dataum in all_pf_plot_data])
        global_data_min = np.nanmin(all_pf_plot_data_mins)
        global_data_max = np.nanmax(all_pf_plot_data_maxes)

        all_pf_plot_items = [self.plots[a_plot_name].mainPlotItem for a_plot_name in self.plots.dynamically_added_attributes] # all plot items PlotItem
        all_pf_image_items = [self.plots[a_plot_name].imageItem for a_plot_name in self.plots.dynamically_added_attributes] # all plot items ImageItems

        # if hasattr(self.params, 'colorMap'):
        #     colorMap = self.params.colorMap
        # else:
        #     colorMap = pg.colormap.get("viridis")

        ## All same colorbar mode:
        # generate an adjustabled color bar
        # shared_colorBarItem = pg.ColorBarItem(values=(0,1), colorMap=colorMap, label='all_pf_2Ds')
        
        shared_colorBarItem = self.params.shared_colorBarItem # get the shared color bar item
        # link color bar and color map to correlogram, and show it in plotItem:
        # shared_colorBarItem
        shared_colorBarItem.setImageItem(all_pf_image_items, insert_in=all_pf_plot_items[0]) # pass a list of ImageItems, insert the color bar after the last plot  , insert_in=all_pf_plot_items[-1]
        # Update the colorbar to the range:
        shared_colorBarItem.setLevels(low=global_data_min, high=global_data_max)


    def add_crosshairs(self, plot_item, matrix, name, xbins=None, ybins=None):
        """ adds crosshairs that allow the user to hover a bin and have the label dynamically display the bin (x, y) and value."""
        vLine = pg.InfiniteLine(angle=90, movable=False)
        hLine = pg.InfiniteLine(angle=0, movable=False)
        
        self.plots[name]['crosshairs_vLine'] = vLine
        self.plots[name]['crosshairs_hLine'] = hLine

        plot_item.addItem(vLine, ignoreBounds=True)
        plot_item.addItem(hLine, ignoreBounds=True)
        vb = plot_item.vb

        should_force_discrete_to_bins: bool = self.params.get('crosshairs_discrete', True)

        def mouseMoved(evt):
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if plot_item.sceneBoundingRect().contains(pos):
                mousePoint = vb.mapSceneToView(pos)

                if should_force_discrete_to_bins:
                    x_point = float(int(round(mousePoint.x())))
                    y_point = float(int(round(mousePoint.y())))

                    x_point = x_point + 0.5 # Snap point to center. Does not affect indexing because it truncates
                    y_point = y_point + 0.5 
                else:
                    x_point = mousePoint.x()
                    y_point = mousePoint.y()
                
                # Note that int(...) truncates towards zero (floor effect)
                index_x = int(x_point)
                index_y = int(y_point)
                
                matrix_shape = np.shape(matrix)
                # is_valid_x_index = (index_x > 0 and index_x < matrix_shape[0])
                # is_valid_y_index = (index_y > 0 and index_y < matrix_shape[1])
                is_valid_x_index = (index_x >= 0 and index_x < matrix_shape[0])
                is_valid_y_index = (index_y >= 0 and index_y < matrix_shape[1])
                
                if is_valid_x_index and is_valid_y_index:
                    if should_force_discrete_to_bins:
                        if (xbins is not None) and (ybins is not None):
                            # Display special xbins/ybins if we have em
                            bin_x = xbins[index_x]
                            bin_y = ybins[index_y]
                            value_str = "<span style='font-size: 12pt'>(x[%d]=%0.3f, y[%d]=%0.3f), <span style='color: green'>value=%0.3f</span>" % (index_x, bin_x, index_y, bin_y, matrix[index_x][index_y])
                        else:
                            value_str = "<span style='font-size: 12pt'>(x=%d, y=%d), <span style='color: green'>value=%0.3f</span>" % (index_x, index_y, matrix[index_x][index_y])

                        self.ui.mainLabel.setText(value_str)
                    else:
                        self.ui.mainLabel.setText("<span style='font-size: 12pt'>(x=%0.1f, y=%0.1f), <span style='color: green'>value=%0.3f</span>" % (index_x, index_y, matrix[index_x][index_y]))

                ## Move the lines:
                vLine.setPos(x_point)
                hLine.setPos(y_point)

        self.ui.connections[name] = pg.SignalProxy(plot_item.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

    def export_all_plots(self, curr_active_pipeline):
        """ Exports each subplot individually to file using `curr_active_pipeline.output_figure(final_context, a_plot.getViewBox())`
        
        NOTE: creates many outputs, one for each subplot
        
        curr_active_pipeline is only used to call 
        
            curr_active_pipeline.output_figure(final_context, a_plot.getViewBox())
        
            
        You can obtain outputs like:
        
        out_figs_dict = self.export_all_plots(curr_active_pipeline)
        out_figs_paths = [v[0][0] for v in list(out_figs_dict.values())] # [0] is for first figure (there's only one), [0] is for the .png version or [1] for the .svg version
        out_figs_paths

        """
        out_figs_dict = {}
        
        for a_name in self.plot_names:
            # Adjust the size of the text for the item by passing formatted text
            a_plot: pg.PlotItem = self.plots[a_name].mainPlotItem # PlotItem 
            # if (a_plot is not None) and (not isinstance(a_plot, str)):
            # a_plot.setTitle(f"<span style = 'font-size : 12px;' >{a_name}</span>")
            # a_plo
            # active_context , epochs='replays', decoder='long_results_obj'	
            final_context = curr_active_pipeline.build_display_context_for_session(display_fn_name='directional_merged_pfs', subplot=a_name)
            out_figs_dict[a_name] = curr_active_pipeline.output_figure(final_context, a_plot.getViewBox())
            
        return out_figs_dict
    

    def _init_UI_buildMainLabel(self):
        """Add Label for debugging:
        """
        self.ui.mainLabel = pg.LabelItem(justify='right')
        self.ui.graphics_layout.addItem(self.ui.mainLabel, row=0, col=0, rowspan=1, colspan=self.params.max_num_columns) # last column
        self.params.plot_row_offset = self.params.plot_row_offset + 1

    def _init_UI_buildColorBarItem(self, label='all_pf_2Ds'):
        """Add color_bar:
        """
        if self.params.color_bar_mode == 'one':
            # Single shared color_bar between all items:
            self.params.shared_colorBarItem = pg.ColorBarItem(values=(0,1), colorMap=self.params.colorMap, label=label)
        else:
            self.params.shared_colorBarItem = None
            
    def init_UI(self):
        raise NotImplementedError('Inheritors must override!')
            

    @classmethod
    def init_from_data_spec(cls, a_spec: List[List[Dict]], window_title=None, scrollability_mode=LayoutScrollability.NON_SCROLLABLE, grid_opacity=0.4, drop_below_threshold=1e-12, color_bar_mode=None, **_shared_kwargs) -> "BasicBinnedImageRenderingMixin":
        """ adds all plots

        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow

        Example 1:
            ## Allow inline spec
            a_spec = [
                ({'measured':_measured}, {'markov_ideal':_markov_ideal}, {'diff':_diff}) # single row (3 columns)
            ]
            out_test_markov_test_compare = BasicBinnedImageRenderingWindow.init_from_data_spec(a_spec, xbins=test_pos_bins, ybins=test_pos_bins)

        Example 2:            
            window_title = 'Transform Matrix: Measured v. Markov Ideal'
            a_spec = [ # two rows: (2 columns, 1 column)
                ({'measured':_measured}, {'markov_ideal':_markov_ideal},), # single row (2 columns)
                ({'diff':_diff}, )
            ]
            out_test_markov_test_compare = BasicBinnedImageRenderingWindow.init_from_data_spec(a_spec, window_title=window_title, xbins=test_pos_bins, ybins=test_pos_bins)


        """
        # _default_window_kwargs = dict(color_bar_mode=None, wants_crosshairs=True, scrollability_mode=LayoutScrollability.SCROLLABLE, grid_opacity=0.65, defer_show=False) ## KWARGS specific to window, not data spec        
        # curr_window_kwargs = dict(window_title='Test', scrollability_mode=LayoutScrollability.NON_SCROLLABLE, grid_opacity=0.4, drop_below_threshold=1e-12)
        curr_window_kwargs = dict(window_title=window_title, scrollability_mode=scrollability_mode, grid_opacity=grid_opacity, drop_below_threshold=drop_below_threshold, color_bar_mode=color_bar_mode)

        # _shared_kwargs = dict(xbins=test_pos_bins, ybins=test_pos_bins)
        n_rows = len(a_spec)
        n_cols_per_row = [len(a_row) for a_row in a_spec] # each row can have different number of columns
        max_n_columns_per_row = np.max(n_cols_per_row)
        curr_window_kwargs['max_num_columns'] = max_n_columns_per_row
        curr_window_kwargs['max_num_rows'] = n_rows
        
        needs_built_window_title: bool = False
        if window_title is None:
            needs_built_window_title: bool = True
            window_title = f"BasicBinnedImageRenderingMixin[rows: {n_rows}, n_cols: {max_n_columns_per_row}]"
            subplot_titles = []
            
        out_binned_window = None

        _built_add_data_kwargs = []
        for row_idx, a_row in enumerate(a_spec):
            for col_idx, a_row_col in enumerate(a_row):
                # first key is always the name
                _curr_identifier: str = list(a_row_col.keys())[0]
                _curr_data = a_row_col.pop(_curr_identifier)
                if needs_built_window_title:
                    subplot_titles.append(_curr_identifier)
                _curr_build_kwargs = dict(row=row_idx, col=col_idx, name=_curr_identifier, title=_curr_identifier, variable_label=_curr_identifier, matrix=_curr_data, **_shared_kwargs)
                _built_add_data_kwargs.append(_curr_build_kwargs)

                if out_binned_window is None:
                    ## create new instance
                    _curr_initialization_kwargs = deepcopy(_curr_build_kwargs)
                    _curr_initialization_kwargs['matrix'] = None
                    
                    # out_binned_window = cls(_curr_build_kwargs.pop('matrix'), **_curr_build_kwargs, **curr_window_kwargs)
                    out_binned_window = cls(**_curr_initialization_kwargs, **curr_window_kwargs) ## create the window, but don't set its data yet
                    
                                        
                # already have a window, use .add_data	
                out_binned_window.add_data(**_curr_build_kwargs, defer_column_update=True) # defer_column_update=True to prevent columns updating each time. We call `update_columns_if_needed` when done

        # row=0, col=1, 
        # _built_add_data_kwargs
        if needs_built_window_title:
            if len(subplot_titles) > 0:
                window_title = window_title + ': ' + ', '.join(subplot_titles)
            out_binned_window.setWindowTitle(window_title)

        out_binned_window.update_columns_if_needed(new_num_columns=max_n_columns_per_row)
        
        ## OUTPUTS: out_test_markov_test_compare, _built_add_data_kwargs
        return out_binned_window
    

    def update_from_data_spec(self, a_spec: List[List[Dict]], window_title=None, **_shared_kwargs):
        """ adds all plots

        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow

        Example 1:
            ## Allow inline spec
            a_spec = [
                ({'measured':_measured}, {'markov_ideal':_markov_ideal}, {'diff':_diff}) # single row (3 columns)
            ]
            out_test_markov_test_compare = BasicBinnedImageRenderingWindow.init_from_data_spec(a_spec, xbins=test_pos_bins, ybins=test_pos_bins)

        Example 2:            
            window_title = 'Transform Matrix: Measured v. Markov Ideal'
            a_spec = [ # two rows: (2 columns, 1 column)
                ({'measured':_measured}, {'markov_ideal':_markov_ideal},), # single row (2 columns)
                ({'diff':_diff}, )
            ]
            out_test_markov_test_compare = BasicBinnedImageRenderingWindow.init_from_data_spec(a_spec, window_title=window_title, xbins=test_pos_bins, ybins=test_pos_bins)


        """
        # _shared_kwargs = dict(xbins=test_pos_bins, ybins=test_pos_bins)
        n_rows = len(a_spec)
        n_cols_per_row = [len(a_row) for a_row in a_spec] # each row can have different number of columns
        max_n_columns_per_row = np.max(n_cols_per_row)
        
        _built_add_data_kwargs = []
        for row_idx, a_row in enumerate(a_spec):
            for col_idx, a_row_col in enumerate(a_row):
                # first key is always the name
                _curr_identifier: str = list(a_row_col.keys())[0]
                _curr_data = a_row_col.pop(_curr_identifier)
                _curr_build_kwargs = dict(row=row_idx, col=col_idx, name=_curr_identifier, title=_curr_identifier, variable_label=_curr_identifier, matrix=_curr_data, **_shared_kwargs)
                _built_add_data_kwargs.append(_curr_build_kwargs)

                # already have a window, use .add_data	
                self.add_data(**_curr_build_kwargs, defer_column_update=True, replace=True) # defer_column_update=True to prevent columns updating each time. We call `update_columns_if_needed` when done

        # row=0, col=1, 
        # _built_add_data_kwargs
        if window_title is not None:
            self.setWindowTitle(window_title)

        self.update_columns_if_needed(new_num_columns=max_n_columns_per_row)
        
        ## OUTPUTS: out_test_markov_test_compare, _built_add_data_kwargs
        # return self



class BasicBinnedImageRenderingWidget(QtWidgets.QWidget, BasicBinnedImageRenderingMixin, SliderUpdatedBinnedImageRenderingMixin):
    """ 
    
    from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWidget
    
    
    """
    def __init__(self, matrix=None, xbins=None, ybins=None, name='avg_velocity', title=None, variable_label=None,
                 drop_below_threshold: float=0.0000001, color_map='viridis', color_bar_mode=None, wants_crosshairs=True, scrollability_mode=LayoutScrollability.SCROLLABLE, grid_opacity:float=0.65, crosshairs_discrete:bool=True, defer_show=False, **kwargs):
        row = kwargs.pop('row', 0)
        col = kwargs.pop('col', 0)
        window_title: str = kwargs.pop('window_title', title)
        max_num_columns: int = kwargs.pop('max_num_columns', None)
        max_num_rows: int = kwargs.pop('max_num_rows', None)
        debug_print: int = kwargs.pop('debug_print', False)
        super(BasicBinnedImageRenderingWidget, self).__init__(**kwargs)
        self.params = VisualizationParameters(name='BasicBinnedImageRenderingWidget', grid_opacity=grid_opacity, drop_below_threshold=drop_below_threshold,
                                               plot_row_offset=0, max_num_columns=max_num_columns, max_num_rows=max_num_rows, window_title=window_title, debug_print=debug_print, crosshairs_discrete=crosshairs_discrete)
        self.plots_data = RenderPlotsData(name='BasicBinnedImageRenderingWidget')
        self.plots = RenderPlots(name='BasicBinnedImageRenderingWidget')
        self.ui = PhoUIContainer(name='BasicBinnedImageRenderingWidget', connections=None, root_layout=None)
        self.ui.connections = PhoUIContainer(name='BasicBinnedImageRenderingWidget')

        self.params.scrollability_mode = LayoutScrollability.init(scrollability_mode)

        # name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity'
        if variable_label is None:
            variable_label = name.strip("'").replace('_', ' ').title() # Convert to a title-case, space-separated string

        if title is None:
            title = name.strip("'").replace('_', ' ').title() # Convert to a title-case, space-separated string

        if isinstance(color_map, str):        
            self.params.colorMap = pg.colormap.get("viridis")
        else:
            # better be a ColorMap object directly
            assert isinstance(color_map, ColorMap)
            self.params.colorMap = color_map
            
        self.params.color_bar_mode = color_bar_mode
        self.params.wants_crosshairs = wants_crosshairs
        pg.setConfigOption('imageAxisOrder', 'row-major') # Switch default order to Row-major        

        self.init_UI()
        
        # Add the item for the provided data:
        if matrix is not None:    
            self.add_data(row=(self.params.plot_row_offset + row), col=col, matrix=matrix, xbins=xbins, ybins=ybins, name=name, title=title, variable_label=variable_label, drop_below_threshold=drop_below_threshold)
    
        if not defer_show:
            self.show()


    def init_UI(self):
        """ init_UI
        
        """
        self._init_UI_buildColorBarItem()
            
        ## Old (non-scrollable) way:        
        # self.ui.graphics_layout = pg.GraphicsLayoutWidget(show=True)
        # self.setCentralWidget(self.ui.graphics_layout)
        
        ## Build scrollable UI version:
        self.ui = _perform_build_root_graphics_layout_widget_ui(self.ui, is_scrollable=self.params.scrollability_mode.is_scrollable)
        # widget only:
        self.ui.root_layout = QtWidgets.QVBoxLayout(self)  # or any other layout            
        self.ui.root_layout.setContentsMargins(0, 0, 0, 0)
        self.ui.root_layout.setSpacing(0)            

        if self.params.scrollability_mode.is_scrollable:
            # self.setCentralWidget(self.ui.scrollAreaWidget)
            self.ui.root_layout.addWidget(self.ui.scrollAreaWidget)
            self.setLayout(self.ui.root_layout)

        else:
            # self.setCentralWidget(self.ui.graphics_layout)
            self.ui.root_layout.addWidget(self.ui.graphics_layout)
            self.setLayout(self.ui.root_layout)
            self.ui.graphics_layout.resize(1000, 800)

        # Shared:
        self.setWindowTitle(self.params.window_title)
        self.resize(1000, 800)
        
        ## Add Label for debugging:
        self.params.max_num_columns = 1
        self._init_UI_buildMainLabel()
        print(f'self.params.plot_row_offset: {self.params.plot_row_offset}')
        



# @metadata_attributes(short_name=None, tags=['binning', 'image', 'window', 'standalone', 'widget'], input_requires=[], output_provides=[], uses=['_perform_build_root_graphics_layout_widget_ui', 'LayoutScrollability'], used_by=[], creation_date='2023-10-19 02:28', related_items=[])
class BasicBinnedImageRenderingWindow(QtWidgets.QMainWindow, BasicBinnedImageRenderingMixin, SliderUpdatedBinnedImageRenderingMixin):
    """ Renders a Matrix of binned data in the window.NonUniformImage and includes no histogram.
        NOTE: uses basic pg.ImageItem instead of pg.
        Observed to work well to display simple binned heatmaps/grids such as avg velocity across spatial bins, etc.    
        
        History:
            Based off of pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_Matrix.MatrixRenderingWindow
            
        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
            out = BasicBinnedImageRenderingWindow(active_eloy_analysis.avg_2D_speed_per_pos, active_pf_2D_dt.xbin_labels, active_pf_2D_dt.ybin_labels, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity', scrollability_mode=LayoutScrollability.SCROLLABLE)
            out.add_data(row=1, col=0, matrix=active_eloy_analysis.pf_overlapDensity_2D, xbins=active_pf_2D_dt.xbin_labels, ybins=active_pf_2D_dt.ybin_labels, name='pf_overlapDensity', title='pf overlapDensity metric', variable_label='pf overlapDensity')
            out.add_data(row=2, col=0, matrix=active_pf_2D.ratemap.occupancy, xbins=active_pf_2D.xbin, ybins=active_pf_2D.ybin, name='occupancy_seconds', title='Seconds Occupancy', variable_label='seconds')
            out.add_data(row=3, col=0, matrix=active_simpler_pf_densities_analysis.n_neurons_meeting_firing_critiera_by_position_bins_2D, xbins=active_pf_2D.xbin, ybins=active_pf_2D.ybin, name='n_neurons_meeting_firing_critiera_by_position_bins_2D', title='# neurons > 1Hz per Pos (X, Y)', variable_label='# neurons')

            
        NOTE:
            Label for `title` is too large, needs to changed to a smaller font


        from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
        out = BasicBinnedImageRenderingWindow(active_eloy_analysis.avg_2D_speed_per_pos, active_pf_2D_dt.xbin_labels, active_pf_2D_dt.ybin_labels, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity')


    """
    def __init__(self, matrix=None, xbins=None, ybins=None, name='avg_velocity', title=None, variable_label=None,
                 drop_below_threshold: float=0.0000001, color_map='viridis', color_bar_mode=None, wants_crosshairs=True, scrollability_mode=LayoutScrollability.SCROLLABLE, grid_opacity:float=0.65, defer_show=False, **kwargs):
        row = kwargs.pop('row', 0)
        col = kwargs.pop('col', 0)
        window_title: str = kwargs.pop('window_title', title)
        max_num_columns: int = kwargs.pop('max_num_columns', None)
        max_num_rows: int = kwargs.pop('max_num_rows', None)
        debug_print: int = kwargs.pop('debug_print', False)
        super(BasicBinnedImageRenderingWindow, self).__init__(**kwargs)
        self.params = VisualizationParameters(name='BasicBinnedImageRenderingWindow', grid_opacity=grid_opacity, plot_row_offset=0, max_num_columns=max_num_rows, max_num_rows=max_num_rows, window_title=window_title, debug_print=debug_print)
        self.plots_data = RenderPlotsData(name='BasicBinnedImageRenderingWindow')
        self.plots = RenderPlots(name='BasicBinnedImageRenderingWindow')
        self.ui = PhoUIContainer(name='BasicBinnedImageRenderingWindow', connections=None)
        self.ui.connections = PhoUIContainer(name='BasicBinnedImageRenderingWindow')

        self.params.scrollability_mode = LayoutScrollability.init(scrollability_mode)

        # name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity'
        if variable_label is None:
            variable_label = name.strip("'").replace('_', ' ').title() # Convert to a title-case, space-separated string

        if title is None:
            title = name.strip("'").replace('_', ' ').title() # Convert to a title-case, space-separated string

        if isinstance(color_map, str):        
            self.params.colorMap = pg.colormap.get("viridis")
        else:
            # better be a ColorMap object directly
            assert isinstance(color_map, ColorMap)
            self.params.colorMap = color_map
            
        self.params.color_bar_mode = color_bar_mode
        self.params.wants_crosshairs = wants_crosshairs
        pg.setConfigOption('imageAxisOrder', 'row-major') # Switch default order to Row-major        

        self.init_UI()
        
        # Add the item for the provided data:
        if matrix is not None:    
            self.add_data(row=(self.params.plot_row_offset + row), col=col, matrix=matrix, xbins=xbins, ybins=ybins, name=name, title=title, variable_label=variable_label, drop_below_threshold=drop_below_threshold)
    
        if not defer_show:
            self.show()
            
    def init_UI(self):
        """ init_UI
        
        """
        self._init_UI_buildColorBarItem()
            
        ## Old (non-scrollable) way:        
        # self.ui.graphics_layout = pg.GraphicsLayoutWidget(show=True)
        # self.setCentralWidget(self.ui.graphics_layout)
        
        ## Build scrollable UI version:
        self.ui = _perform_build_root_graphics_layout_widget_ui(self.ui, is_scrollable=self.params.scrollability_mode.is_scrollable)
        if self.params.scrollability_mode.is_scrollable:
            self.setCentralWidget(self.ui.scrollAreaWidget)
        else:
            self.setCentralWidget(self.ui.graphics_layout)
            self.ui.graphics_layout.resize(1000, 800)

        # Shared:
        self.setWindowTitle(self.params.window_title)
        self.resize(1000, 800)
        
        ## Add Label for debugging:
        self.params.max_num_columns = 1
        self._init_UI_buildMainLabel()
        print(f'self.params.plot_row_offset: {self.params.plot_row_offset}')
        

        



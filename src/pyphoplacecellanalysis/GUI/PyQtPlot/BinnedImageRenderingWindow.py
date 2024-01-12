import numpy as np
import attrs
from attrs import define, field, Factory, asdict, astuple
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, mkQApp, QtGui
from pyphoplacecellanalysis.External.pyqtgraph.colormap import ColorMap
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.NonUniformImage import NonUniformImage

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

# For scrollable BasicBinnedImageRenderingWindow
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, _perform_build_root_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_with_nested_viewbox_ui


def _add_bin_ticks(plot_item, xbins=None, ybins=None):
    """ adds the ticks/grid for xbins and ybins to the plot_item """
    # show full frame, label tick marks at top and left sides, with some extra space for labels:
    plot_item.showAxes(True, showValues=(True, True, False, False), size=10)
    # define major tick marks and labels:
    if xbins is not None:
        xticks = [(idx, label) for idx, label in enumerate(xbins)]
        for side in ('top','bottom'):
            plot_item.getAxis(side).setTicks((xticks, [])) # add list of major ticks; no minor ticks        
    if ybins is not None:
        yticks = [(idx, label) for idx, label in enumerate(ybins)]
        for side in ('left','right'):
            plot_item.getAxis(side).setTicks((yticks, [])) # add list of major ticks; no minor ticks
    plot_item.showGrid(x = True, y = True, alpha = 0.65)
    return plot_item


def _build_binned_imageItem(plot_item, params, xbins=None, ybins=None, matrix=None, name='avg_velocity', data_label='Avg Velocity', color_bar_mode=None):
    """ 
    color_bar_mode: options for the colorbar of each image
        ### curr_cbar_mode: 'each', 'one', None
    """
    local_plots_data = RenderPlotsData(name=name)
    local_plots_data.matrix = matrix.copy()
    local_plots_data.matrix_min = np.nanmin(matrix)
    local_plots_data.matrix_max = np.nanmax(matrix)
    
    # plotItem.invertY(True)           # orient y axis to run top-to-bottom
    
    local_plots = RenderPlots(name=name)
    # Normal ImageItem():
    local_plots.imageItem = pg.ImageItem(matrix.T)
    plot_item.addItem(local_plots.imageItem)

    plot_item.setAspectLocked(lock=True, ratio=1)


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



@metadata_attributes(short_name=None, tags=['binning', 'image', 'window', 'standalone', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-19 02:28', related_items=[])
class BasicBinnedImageRenderingWindow(QtWidgets.QMainWindow):
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
    """
    
    def __init__(self, matrix=None, xbins=None, ybins=None, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity',
                 drop_below_threshold: float=0.0000001, color_map='viridis', color_bar_mode=None, wants_crosshairs=True, scrollability_mode=LayoutScrollability.SCROLLABLE, defer_show=False, **kwargs):
        super(BasicBinnedImageRenderingWindow, self).__init__(**kwargs)
        self.params = VisualizationParameters(name='BasicBinnedImageRenderingWindow')
        self.plots_data = RenderPlotsData(name='BasicBinnedImageRenderingWindow')
        self.plots = RenderPlots(name='BasicBinnedImageRenderingWindow')
        self.ui = PhoUIContainer(name='BasicBinnedImageRenderingWindow')
        self.ui.connections = PhoUIContainer(name='BasicBinnedImageRenderingWindow')

        self.params.scrollability_mode = LayoutScrollability.init(scrollability_mode)

        
        if isinstance(color_map, str):        
            self.params.colorMap = pg.colormap.get("viridis")
        else:
            # better be a ColorMap object directly
            assert isinstance(color_map, ColorMap)
            self.params.colorMap = color_map
            
        self.params.color_bar_mode = color_bar_mode
        if self.params.color_bar_mode == 'one':
            # Single shared color_bar between all items:
            self.params.shared_colorBarItem = pg.ColorBarItem(values=(0,1), colorMap=self.params.colorMap, label='all_pf_2Ds')
        else:
            self.params.shared_colorBarItem = None
            
        self.params.wants_crosshairs = wants_crosshairs

        pg.setConfigOption('imageAxisOrder', 'row-major') # Switch default order to Row-major

        ## Old (non-scrollable) way:        
        # self.ui.graphics_layout = pg.GraphicsLayoutWidget(show=True)
        # self.setCentralWidget(self.ui.graphics_layout)

        ## Build scrollable UI version:
        self.ui = _perform_build_root_graphics_layout_widget_ui(self.ui, is_scrollable=self.params.scrollability_mode.is_scrollable)
        if self.params.scrollability_mode.is_scrollable:
            self.setCentralWidget(self.ui.scrollAreaWidget)
        else:
            self.setCentralWidget(self.ui.graphics_layout)

        # Shared:
        self.setWindowTitle(title)
        self.resize(1000, 800)
        
        ## Add Label for debugging:
        self.ui.mainLabel = pg.LabelItem(justify='right')
        self.ui.graphics_layout.addItem(self.ui.mainLabel)
        
        # Add the item for the provided data:
        self.add_data(row=0, col=0, matrix=matrix, xbins=xbins, ybins=ybins, name=name, title=title, variable_label=variable_label, drop_below_threshold=drop_below_threshold)
        
        if not defer_show:
            self.show()


    def build_formatted_title_string(self, title: str) -> str:
        return f"<span style = 'font-size : 12px;' >{title}</span>"
        

    def add_data(self, row=1, col=0, matrix=None, xbins=None, ybins=None, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity', drop_below_threshold: float=0.0000001):
        """ adds a new data subplot to the output
        """
        newPlotItem = self.ui.graphics_layout.addPlot(title=title, row=row, col=col) # add PlotItem to the main GraphicsLayoutWidget
        
        # Set the plot title:
        formatted_title = self.build_formatted_title_string(title=title)        
        newPlotItem.setTitle(formatted_title)
        
        newPlotItem.setDefaultPadding(0.0)  # plot without padding data range
        newPlotItem.setMouseEnabled(x=False, y=False)
        newPlotItem = _add_bin_ticks(plot_item=newPlotItem, xbins=xbins, ybins=ybins)

        if drop_below_threshold is not None:
            matrix = matrix.astype(float) # required because NaN isn't available in Integer dtype arrays (in case the matrix is of integer type, this prevents a ValueError)
            matrix[np.where(matrix < drop_below_threshold)] = np.nan # null out the occupancy
            
        
        local_plots, local_plots_data = _build_binned_imageItem(newPlotItem, self.params, xbins=xbins, ybins=ybins, matrix=matrix, name=name, data_label=variable_label, color_bar_mode=self.params.color_bar_mode)
        self.plots_data[name] = local_plots_data
        self.plots[name] = local_plots
        self.plots[name].mainPlotItem = newPlotItem

        # # Set the plot title:
        # formatted_title = self.build_formatted_title_string(title=title)
        # self.plots[name].mainPlotItem.setTitle(formatted_title)
                

        if self.params.color_bar_mode == 'one':
            self.plots[name].colorBarItem = self.params.shared_colorBarItem # shared colorbar item
            self._update_global_shared_colorbaritem()
        
        if self.params.wants_crosshairs:
            self.add_crosshairs(newPlotItem, matrix, name=name)


        ## Scrollable-support:
        active_num_rows = row+1 # get the number of rows after adding the data (adding one to go from an index to a count)

        if self.params.scrollability_mode.is_scrollable:
            self.params.single_plot_fixed_height = 100.0
            self.params.all_plots_height = float(active_num_rows) * float(self.params.single_plot_fixed_height)
            self.ui.graphics_layout.setFixedHeight(self.params.all_plots_height)
            # self.ui.graphics_layout.setMinimumHeight(self.params.all_plots_height)
            
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


    def add_crosshairs(self, plot_item, matrix, name):
        """ adds crosshairs that allow the user to hover a bin and have the label dynamically display the bin (x, y) and value."""
        vLine = pg.InfiniteLine(angle=90, movable=False)
        hLine = pg.InfiniteLine(angle=0, movable=False)
        plot_item.addItem(vLine, ignoreBounds=True)
        plot_item.addItem(hLine, ignoreBounds=True)
        vb = plot_item.vb

        def mouseMoved(evt):
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if plot_item.sceneBoundingRect().contains(pos):
                mousePoint = vb.mapSceneToView(pos)
                # Note that int(...) truncates towards zero (floor effect)
                index_x = int(mousePoint.x())
                index_y = int(mousePoint.y())
                
                matrix_shape = np.shape(matrix)
                # is_valid_x_index = (index_x > 0 and index_x < matrix_shape[0])
                # is_valid_y_index = (index_y > 0 and index_y < matrix_shape[1])
                is_valid_x_index = (index_x >= 0 and index_x < matrix_shape[0])
                is_valid_y_index = (index_y >= 0 and index_y < matrix_shape[1])
                
                if is_valid_x_index and is_valid_y_index:
                    self.ui.mainLabel.setText("<span style='font-size: 12pt'>(x=%0.1f, y=%0.1f), <span style='color: green'>value=%0.3f</span>" % (index_x, index_y, matrix[index_x][index_y]))
                vLine.setPos(mousePoint.x())
                hLine.setPos(mousePoint.y())

        self.ui.connections[name] = pg.SignalProxy(plot_item.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
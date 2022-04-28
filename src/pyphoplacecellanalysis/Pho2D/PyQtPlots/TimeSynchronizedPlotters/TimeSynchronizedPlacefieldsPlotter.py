import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets

from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.indexing_helpers import compute_paginated_grid_config


from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlotterBase import TimeSynchronizedPlotterBase
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import _pyqtplot_build_image_bounds_extent

class TimeSynchronizedPlacefieldsPlotter(TimeSynchronizedPlotterBase):
    """
    
    Usage:
    
        included_epochs = None
        computation_config = active_session_computation_configs[0]
        print('Recomputing active_epoch_placefields2D...', end=' ')
        # PfND version:
        t_list = []
        ratemaps_list = []
        active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
                                          speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                          grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
        curr_occupancy_plotter = TimeSynchronizedPlacefieldsPlotter(active_time_dependent_placefields2D)
        curr_occupancy_plotter.show()

    """
    # Application/Window Configuration Options:
    applicationName = 'TimeSynchronizedPlacefieldsPlotterApp'
    windowName = 'TimeSynchronizedPlacefieldsPlotterWindow'
    
    enable_debug_print = False
    
    def __init__(self, active_time_dependent_placefields2D, drop_below_threshold: float=0.0000001, max_num_columns = 5, application_name=None, parent=None):
        """_summary_
        """
        super().__init__(application_name=application_name, parent=parent) # Call the inherited classes __init__ method
        
        self.active_time_dependent_placefields = active_time_dependent_placefields2D
        
        self.setup()
        self.params.drop_below_threshold = drop_below_threshold
        self.params.max_num_columns = max_num_columns
        
        self.buildUI()
        # self.show()
        
    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        self.app = pg.mkQApp(self.applicationName)
        self.params = VisualizationParameters(self.applicationName)
        ## Build the colormap to be used:
        # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        self.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
        self.params.image_margins = 2.0
        
        self.params.image_bounds_extent, self.params.x_range, self.params.y_range = _pyqtplot_build_image_bounds_extent(self.active_time_dependent_placefields.xbin, self.active_time_dependent_placefields.ybin, margin=self.params.image_margins, debug_print=self.enable_debug_print)
        
        self.params.nMapsToShow = self.active_time_dependent_placefields.ratemap.n_neurons
        
    # def buildUI(self):
    #     """ for QGridLayout
    #         addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
    #     """
    #     self.ui = PhoUIContainer()
        
    #     self.ui.layout = QtWidgets.QGridLayout()
    #     self.ui.layout.setContentsMargins(0, 0, 0, 0)
    #     self.ui.layout.setVerticalSpacing(0)
    #     self.ui.layout.setHorizontalSpacing(0)
    #     # self.setStyleSheet("background : #1B1B1B; color : #727272")
    #     #### Build Graphics Objects #####
    #     self._buildGraphics()        
    #     self.setLayout(self.ui.layout)
    #     self.resize(800,800)
    #     self.setWindowTitle(self.windowName)
        
    def _buildGraphics(self):
        self.ui.img_item_array = []
        self.ui.other_components_array = []
        self.ui.plot_array = []
        
        # root_render_widget
        self.ui.root_graphics_layout_widget = pg.GraphicsLayoutWidget()
        
        curr_ratemap = self.active_time_dependent_placefields.ratemap
        images = curr_ratemap.tuning_curves.copy() # (43, 63, 63)
        occupancy = curr_ratemap.occupancy
        
        # Compute Images:
        included_unit_indicies = np.arange(np.shape(images)[0]) # include all unless otherwise specified
        nMapsToShow = len(included_unit_indicies)

        # Paging Management: Constrain the subplots values to just those that you need
        subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=self.params.max_num_columns, max_subplots_per_page=None, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=True)
        page_idx = 0 # page_idx is zero here because we only have one page:
        

            
        # for i, curr_included_cell_ID in enumerate(self.active_time_dependent_placefields.ratemap.neuron_ids):
        
        for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
            # Need to convert to page specific:
            curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
            curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
            curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
            if self.enable_debug_print:
                print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')
                
            cell_IDX = curr_included_unit_index
            cell_ID = self.active_time_dependent_placefields.ratemap.neuron_ids[cell_IDX]
            curr_cell_identifier_string = f'Cell[{cell_ID}]'
            curr_plot_identifier_string = f'pyqtplot_plot_image_array.{curr_cell_identifier_string}'

            # Build the image item:
            # Update the image:
            image = np.squeeze(images[a_linear_index,:,:])
            # Pre-filter the data:
            with np.errstate(divide='ignore', invalid='ignore'):
                image = np.array(image) / np.nanmax(image) # note scaling by maximum here!
                if self.params.drop_below_threshold is not None:
                    image[np.where(occupancy < self.params.drop_below_threshold)] = np.nan # null out the occupancy
            img_item = pg.ImageItem(image=image, levels=(0,1))
                
            curr_plot = self.ui.root_graphics_layout_widget.addPlot(row=curr_row, col=curr_col, name=curr_plot_identifier_string, title=curr_cell_identifier_string)
            curr_plot.addItem(img_item)  # add ImageItem to PlotItem
            curr_plot.showAxes(True)
   
            # Update the image:
            img_item.setImage(image, rect=self.params.image_bounds_extent, autoLevels=False)
            img_item.setLookupTable(self.params.cmap.getLookupTable(nPts=256))

            # curr_plot.setXRange(global_min_x-margin, global_max_x+margin)
            # curr_plot.setYRange(global_min_y-margin, global_max_y+margin)
            curr_plot.setXRange(*self.params.x_range)
            curr_plot.setYRange(*self.params.y_range)

            # Interactive Color Bar:
            bar = pg.ColorBarItem(values= (0, 1), colorMap=self.params.cmap, width=5, interactive=False) # prepare interactive color bar
            # Have ColorBarItem control colors of img and appear in 'plot':
            bar.setImageItem(img_item, insert_in=curr_plot)

            self.ui.img_item_array.append(img_item)
            self.ui.plot_array.append(curr_plot)
            self.ui.other_components_array.append({'color_bar':bar})
        

        # add the root_graphics_layout_widget to the main layout:
        self.ui.layout.addWidget(self.ui.root_graphics_layout_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
    
    
    def update(self, t):
        # Compute the updated placefields/occupancy for the time t:
        with np.errstate(divide='ignore', invalid='ignore'):
            self.active_time_dependent_placefields.update(t)


    def _update_plots(self):
        """
        
        """
        if self.enable_debug_print:
            print(f'TimeSynchronizedPlacefieldsPlotter._update_plots()')
            
        # Update the existing one:
        
        # Update the plots:
        curr_t = self.active_time_dependent_placefields.last_t
        curr_ratemap = self.active_time_dependent_placefields.ratemap
        images = curr_ratemap.tuning_curves.copy() # (43, 63, 63)
        occupancy = curr_ratemap.occupancy
        # image = curr_ratemap.occupancy
        # image = self.active_time_dependent_placefields.curr_normalized_occupancy
        # image_title = 'curr_normalized_occupancy'
        image_title = 'tuning_curves'
        
        # Update the placefields plot if needed:
        for i, an_img_item in enumerate(self.ui.img_item_array):
            image = np.squeeze(images[i,:,:])
            # Pre-filter the data:
            with np.errstate(divide='ignore', invalid='ignore'):
                image = np.array(image) / np.nanmax(image) # note scaling by maximum here!
                if self.params.drop_below_threshold is not None:
                    image[np.where(occupancy < self.params.drop_below_threshold)] = np.nan # null out the occupancy
            # an_img_item.setImage(np.squeeze(images[i,:,:]))
            an_img_item.setImage(image, autoLevels=False)
            

        self.setWindowTitle(f'{self.windowName} - {image_title} t = {curr_t}')
    
    
    # @QtCore.Slot(float, float)
    # def on_window_changed(self, start_t, end_t):
    #     # called when the window is updated
    #     if self.enable_debug_print:
    #         print(f'TimeSynchronizedPlacefieldsPlotter.on_window_changed(start_t: {start_t}, end_t: {end_t})')
    #     if self.enable_debug_print:
    #         profiler = pg.debug.Profiler(disabled=True, delayed=True)
            
    #     self.update(end_t)
    #     self._update_plots()
        
    #     if self.enable_debug_print:
    #         profiler('Finished calling _update_plots()')
            
            
# included_epochs = None
# computation_config = active_session_computation_configs[0]
# active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
#                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
#                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
# curr_occupancy_plotter = TimeSynchronizedPlacefieldsPlotter(active_time_dependent_placefields2D)
# curr_occupancy_plotter.show()
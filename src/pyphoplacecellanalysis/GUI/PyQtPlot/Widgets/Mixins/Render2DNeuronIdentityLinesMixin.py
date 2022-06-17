


class Render2DNeuronIdentityLinesMixin:
    """ renders the horizontal lines on the 2D plots 
    
    TODO: This is not really a mixin, need to figure out how I want these used.

    """
    @staticmethod
    def add_lines(plot_widget):
        """ Sets up the y-axis ticks/grid to display one tick per unit (meaning one tick per cell) and label them every 5 ticks 

        Inputs:       
            plot_widget <PlotItem>: (background_static_scroll_plot_widget, main_plot_widget)
        
        Returns:
            v_axis_item <AxisItem>: note that this object doesn't need to be retained.

        Usage:
            ## Add the lines to the main_plot_widget: 
            main_plot_widget = spike_raster_window.spike_raster_plt_2d.ui.main_plot_widget # PlotItem
            v_axis_item = Render2DNeuronIdentityLinesMixin.add_lines(main_plot_widget)

            ## Add the lines to the background_static_scroll_window_plot: 
            background_static_scroll_plot_widget = spike_raster_window.spike_raster_plt_2d.ui.background_static_scroll_window_plot # PlotItem
            background_v_axis_item = Render2DNeuronIdentityLinesMixin.add_lines(background_static_scroll_plot_widget)

        """
        v_axis_item = plot_widget.axes['left']['item'] # AxisItem 
        v_axis_item.setGrid(255) # set grid alpha to opaque (Set the alpha value (0-255) for the grid, or False to disable.)
        v_axis_item.setTickSpacing(5, 1) # two levels, all offsets = 0
        return v_axis_item
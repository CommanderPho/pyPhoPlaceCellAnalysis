import numpy as np


class Render2DNeuronIdentityLinesMixin:
    """ renders the horizontal lines separating the neurons on the 2D raster plots 
    
    TODO: This is not really a mixin, need to figure out how I want these used.
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DNeuronIdentityLinesMixin import Render2DNeuronIdentityLinesMixin
        
        v_axis_item = Render2DNeuronIdentityLinesMixin.setup_custom_neuron_ticks(main_plot_widget, spike_raster_window.spike_raster_plt_2d.n_cells)
        v_axis_item = Render2DNeuronIdentityLinesMixin.add_lines(main_plot_widget)
        

    """
    @staticmethod
    def setup_custom_neuron_identity_axis(plot_widget, n_cells):
        """ Completely sets up the custom 2D neuron identity axis vertical/y-axis) by adding one minor tick per neuron and displaying the horizontal grid. """
        v_axis_item = Render2DNeuronIdentityLinesMixin._setup_custom_neuron_ticks(plot_widget, n_cells)
        v_axis_item = Render2DNeuronIdentityLinesMixin._add_lines(plot_widget)
        return v_axis_item
    
    
    @staticmethod
    def _setup_custom_neuron_ticks(plot_widget, n_cells):
        """ Build custom ticks with one for each neuron """
        neuron_id_ticks = [
            [(float(i),f'{i}') for i in np.arange(0, 5, n_cells)], # [ (majorTickValue1, majorTickString1), (majorTickValue2, majorTickString2), ... ],
            [(float(i),'') for i in np.arange(n_cells+1)], #[ (minorTickValue1, minorTickString1), (minorTickValue2, minorTickString2), ... ],
        ]
        v_axis_item = plot_widget.axes['left']['item'] # AxisItem 
        v_axis_item.setTicks(neuron_id_ticks)
        v_axis_item.setStyle(stopAxisAtTick=(True, True)) # (max, min)
        return v_axis_item

        
    @staticmethod
    def _add_lines(plot_widget):
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
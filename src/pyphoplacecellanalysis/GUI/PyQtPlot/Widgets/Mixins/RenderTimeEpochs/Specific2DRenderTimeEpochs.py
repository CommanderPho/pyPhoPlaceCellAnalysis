import numpy as np
import pandas as pd


import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Render2DEventRectanglesHelper import Render2DEventRectanglesHelper
from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource



class RenderTimeEpochsItem(object):
    """docstring for RenderTimeEpochsItem."""
    def __init__(self, name, data):
        super(RenderTimeEpochsItem, self).__init__()
        self.name = name
        self.data = data # data should be either a Epoch object, a dataframe with the appropriate properties, or a callable
        self.plots = [] # IntervalRectsItem
        


    

class Specific2DRenderTimeEpochsHelper:
    """ Analagous to Specific3DTimeCurvesHelper """
    
    
    ##########################################
    ## PBE (Population Burst Events)
    @staticmethod
    def build_PBEs_formatter_datasource(debug_print=False):
        class PBE_IntervalRectFormatter:
            """ An alternative to the simplier _add_interval_dataframe_visualization_columns_PBE(...) function version that can be passed to _add_rendered_epochs(...) as a callback if state is desired.
                adds the remaining _required_interval_visualization_columns specifically for PBEs
            """
            
            def __init__(self) -> None:
                ## PBE parameters:
                # self.pbe_y_location: float = 45.0
                self.pbe_y_location: float = 0.0
                self.pbe_height: float = 2.5
                self.pbe_pen_color: object = pg.mkColor('w')
                self.pbe_brush_color: object = pg.mkColor('grey')
                
            def __call__(self, active_PBEs_df):
                ## Add the missing parameters to the dataframe:
                active_PBEs_df['series_vertical_offset'] = self.pbe_y_location
                active_PBEs_df['series_height'] = self.pbe_height
                active_PBEs_df['pen'] = pg.mkPen(self.pbe_pen_color)
                active_PBEs_df['brush'] = pg.mkBrush(self.pbe_brush_color)
                return active_PBEs_df
            
            
                
        def _add_interval_dataframe_visualization_columns_PBE(active_PBEs_df):
            """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
                Designed to be passed to _add_rendered_epochs(...) as a callback if state is desired.
            """
            num_intervals = np.shape(active_PBEs_df)[0]
            if debug_print:
                print(f'num_intervals: {num_intervals}') # num_intervals: 206

            ## PBE parameters:
            # pbe_y_location = 45.0
            pbe_y_location = 0.0
            pbe_height = 2.5
            pbe_pen_color = pg.mkColor('w')
            pbe_brush_color = pg.mkColor('grey')

            ## Add the missing parameters to the dataframe:
            active_PBEs_df['series_vertical_offset'] = pbe_y_location
            active_PBEs_df['series_height'] = pbe_height
            active_PBEs_df['pen'] = pg.mkPen(pbe_pen_color)
            active_PBEs_df['brush'] = pg.mkBrush(pbe_brush_color)
            
            # new bounds:
            new_y_max = (pbe_y_location+pbe_height)
            print(f'new_y_max: {new_y_max}')
            
            return active_PBEs_df

        # active_pbe_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_epoch(active_PBEs_obj, _add_interval_dataframe_visualization_columns_PBE)
        pbe_interval_rects_formatter = PBE_IntervalRectFormatter()
        return pbe_interval_rects_formatter
        
    @classmethod
    def build_PBEs_2D_render_time_epochs(cls, curr_sess):
        """ builds the animal position 3D Curves and adds them to the spike_raster_plot
        Usage:
            active_plot_curve_datasource = Specific3DTimeCurvesHelper.build_position_3D_time_curves(curr_sess, spike_raster_plt_3d)
        """
        active_PBEs_obj = curr_sess.pbe # <Epoch> object
        pbe_interval_rects_formatter = cls.build_PBEs_formatter_datasource()
        active_pbe_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_epoch(active_PBEs_obj, pbe_interval_rects_formatter) # IntervalRectsItem
        return active_pbe_interval_rects_item
    
    
    ##########################################
    ## PBE (Population Burst Events)
    @staticmethod
    def build_Laps_formatter_datasource(debug_print=False, **kwargs):
        
        
        def _add_interval_dataframe_visualization_columns_Laps(active_Laps_df):
            """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
            """
            num_intervals = np.shape(active_Laps_df)[0]
            if debug_print:
                print(f'num_intervals: {num_intervals}') # num_intervals: 206

            ## PBE parameters:
            y_location = 0.0
            height = 1.0
            # y_location = 43.5            
            # height = 1.5
            pen_color = pg.mkColor('red')
            brush_color = pg.mkColor('red')

            ## Add the missing parameters to the dataframe:
            active_Laps_df['series_vertical_offset'] = kwargs.setdefault('series_vertical_offset', y_location)
            active_Laps_df['series_height'] = kwargs.setdefault('series_height', height)
            active_Laps_df['pen'] = kwargs.setdefault('pen', pg.mkPen(pen_color)) 
            active_Laps_df['brush'] = kwargs.setdefault('brush', pg.mkBrush(brush_color))  
            return active_Laps_df

        return _add_interval_dataframe_visualization_columns_Laps
        
    @classmethod
    def build_Laps_2D_render_time_epochs(cls, curr_sess, **kwargs):
        """ 
        Usage:

        """
        active_Laps_Epochs = curr_sess.laps.as_epoch_obj() # <Epoch> object
        laps_interval_datasource = IntervalsDatasource.init_from_epoch_object(active_Laps_Epochs, cls.build_Laps_formatter_datasource(**kwargs),       datasource_name='intervals_datasource_from_laps_epoch_obj')
        ## build the output tuple list: fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
        # curr_IntervalRectsItem_interval_tuples = Render2DEventRectanglesHelper._build_interval_tuple_list_from_dataframe(test_laps_interval_datasource)
        # active_laps_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_epoch(active_Laps_Epochs, cls.build_Laps_formatter_datasource(**kwargs))
        
        ## IntervalsDatasource version:
        active_laps_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(laps_interval_datasource)
        active_laps_interval_rects_item.setToolTip('Laps')
        return active_laps_interval_rects_item
    
    @classmethod
    def add_Laps_2D_render_time_epochs(cls, curr_sess, destination_plot):
        active_interval_rects_item = cls.build_Laps_2D_render_time_epochs(curr_sess=curr_sess)
        
        destination_plot.addPlot(active_interval_rects_item)
        
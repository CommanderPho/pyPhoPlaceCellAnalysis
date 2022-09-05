import numpy as np
import pandas as pd

from neuropy.core.laps import Laps
from neuropy.core.epoch import Epoch
from neuropy.core.session.dataSession import DataSession


import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Render2DEventRectanglesHelper import Render2DEventRectanglesHelper
from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource


class Specific2DRenderTimeEpochsHelper:
    """ Analagous to Specific3DTimeCurvesHelper """
    
    ##########################################
    ## General Epochs
    @staticmethod
    def build_general_epochs_dataframe_formatter(debug_print=False, **kwargs):
        def _add_interval_dataframe_visualization_columns_general_epoch(active_Laps_df):
            """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
            """
            num_intervals = np.shape(active_Laps_df)[0]
            if debug_print:
                print(f'num_intervals: {num_intervals}') # num_intervals: 206

    #         ## parameters:
    #         y_location = 0.0
    #         height = 1.0
    #         pen_color = pg.mkColor('red')
    #         brush_color = pg.mkColor('red')

    #         ## Add the missing parameters to the dataframe:
    #         active_Laps_df['series_vertical_offset'] = kwargs.setdefault('series_vertical_offset', y_location)
    #         active_Laps_df['series_height'] = kwargs.setdefault('series_height', height)
    #         active_Laps_df['pen'] = kwargs.setdefault('pen', pg.mkPen(pen_color)) 
    #         active_Laps_df['brush'] = kwargs.setdefault('brush', pg.mkBrush(brush_color))  
            ## parameters:
            y_location = 0.0
            height = 1.0
            # pen_color = [pg.mkColor('red'), pg.mkColor('cyan')]
            # brush_color = [pg.mkColor('red'), pg.mkColor('cyan')]
            
            pen_color = [pg.mkColor('red'), pg.mkColor('cyan')]
            brush_color = [pg.mkColor('red'), pg.mkColor('cyan')]

            ## Add the missing parameters to the dataframe:
            active_Laps_df['series_vertical_offset'] = kwargs.setdefault('series_vertical_offset', y_location)
            active_Laps_df['series_height'] = kwargs.setdefault('series_height', height)
            # active_Laps_df['pen'] = kwargs.setdefault('pen', pg.mkPen(pen_color)) 
            # active_Laps_df['brush'] = kwargs.setdefault('brush', pg.mkBrush(brush_color))  
            
            active_Laps_df['pen'] = kwargs.setdefault('pen', [pg.mkPen(a_pen_color) for a_pen_color in pen_color]) 
            active_Laps_df['brush'] = kwargs.setdefault('brush', [pg.mkBrush(a_color) for a_color in brush_color])  

            return active_Laps_df

        return _add_interval_dataframe_visualization_columns_general_epoch

    @classmethod
    def build_general_render_time_epochs_datasource(cls, active_epochs_obj, **kwargs):
        general_epochs_interval_datasource = IntervalsDatasource.init_from_epoch_object(active_epochs_obj, cls.build_general_epochs_dataframe_formatter(**kwargs), datasource_name='intervals_datasource_from_general_Epochs_obj')
        return general_epochs_interval_datasource

    @classmethod
    def build_general_2D_render_time_epochs(cls, curr_sess, **kwargs):
        """ builds the 
        
        """
        if isinstance(curr_sess, DataSession):
            active_Epochs = curr_sess.epochs # <Epoch> object
        elif isinstance(curr_sess, Epoch):
            active_Epochs = curr_sess  # <Epoch> object passed directly
        else:
            raise NotImplementedError
        
        interval_datasource = cls.build_general_render_time_epochs_datasource(active_epochs_obj=active_Epochs, **kwargs)
        active_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasource)
        active_interval_rects_item.setToolTip('General')
        return active_interval_rects_item
    
    @classmethod
    def add_general_2D_render_time_epochs(cls, curr_sess, destination_plot):
        active_interval_rects_item = cls.build_general_2D_render_time_epochs(curr_sess=curr_sess)
        destination_plot.addPlot(active_interval_rects_item)
        
    
    ##########################################
    ## PBE (Population Burst Events)
    @staticmethod
    def build_PBEs_dataframe_formatter(debug_print=False, **kwargs):
        # class PBE_IntervalRectFormatter:
        #     """ An alternative to the simplier _add_interval_dataframe_visualization_columns_PBE(...) function version that can be passed to _add_rendered_epochs(...) as a callback if state is desired.
        #         adds the remaining _required_interval_visualization_columns specifically for PBEs
        #     """
            
        #     def __init__(self) -> None:
        #         ## PBE parameters:
        #         # self.pbe_y_location: float = 45.0
        #         self.pbe_y_location: float = 0.0
        #         self.pbe_height: float = 2.5
        #         self.pbe_pen_color: object = pg.mkColor('w')
        #         self.pbe_brush_color: object = pg.mkColor('grey')
                
        #     def __call__(self, active_PBEs_df):
        #         ## Add the missing parameters to the dataframe:
        #         active_PBEs_df['series_vertical_offset'] = self.pbe_y_location
        #         active_PBEs_df['series_height'] = self.pbe_height
        #         active_PBEs_df['pen'] = pg.mkPen(self.pbe_pen_color)
        #         active_PBEs_df['brush'] = pg.mkBrush(self.pbe_brush_color)
        #         return active_PBEs_df
            
        def _add_interval_dataframe_visualization_columns_PBE(active_PBEs_df):
            """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
                Designed to be passed to _add_rendered_epochs(...) as a callback if state is desired.
            """
            num_intervals = np.shape(active_PBEs_df)[0]
            if debug_print:
                print(f'num_intervals: {num_intervals}') # num_intervals: 206

            ## PBE parameters:
            # pbe_y_location = 45.0
            y_location = 0.0
            height = 2.5
            pen_color = pg.mkColor('w')
            brush_color = pg.mkColor('grey')

            ## Add the missing parameters to the dataframe:
            active_PBEs_df['series_vertical_offset'] = kwargs.setdefault('series_vertical_offset', y_location)
            active_PBEs_df['series_height'] = kwargs.setdefault('series_height', height)
            active_PBEs_df['pen'] = kwargs.setdefault('pen', pg.mkPen(pen_color)) 
            active_PBEs_df['brush'] = kwargs.setdefault('brush', pg.mkBrush(brush_color))  
            
            # new bounds:
            new_y_max = (y_location+height)
            print(f'new_y_max: {new_y_max}')
            
            return active_PBEs_df

        # active_pbe_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_epoch(active_PBEs_obj, _add_interval_dataframe_visualization_columns_PBE)
        # pbe_interval_rects_formatter = PBE_IntervalRectFormatter()
        # return pbe_interval_rects_formatter
        return _add_interval_dataframe_visualization_columns_PBE
        
        

    @classmethod
    def build_PBEs_render_time_epochs_datasource(cls, curr_sess, **kwargs):
        if isinstance(curr_sess, DataSession):
            active_pbe_Epochs = curr_sess.pbe # <Epoch> object
        elif isinstance(curr_sess, Epoch):
            active_pbe_Epochs = curr_sess  # <Epoch> object passed directly
        else:
            raise NotImplementedError
        return IntervalsDatasource.init_from_epoch_object(active_pbe_Epochs, cls.build_PBEs_dataframe_formatter(**kwargs), datasource_name='intervals_datasource_from_PBEs_epoch_obj')
    
    
    @classmethod
    def build_PBEs_2D_render_time_epochs(cls, curr_sess, **kwargs):
        """ builds the animal PBE Rects and adds them to the spike_raster_plot
        
        """
        # active_PBEs_obj = curr_sess.pbe # <Epoch> object
        # pbe_interval_rects_formatter = cls.build_PBEs_formatter_datasource()
        # active_pbe_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_epoch(active_PBEs_obj, pbe_interval_rects_formatter) # IntervalRectsItem
        ## IntervalsDatasource version:
        # PBEs_interval_datasource = IntervalsDatasource.init_from_epoch_object(active_PBEs_obj, cls.build_PBEs_formatter_datasource(**kwargs),       datasource_name='intervals_datasource_from_PBEs_epoch_obj')
        
        PBEs_interval_datasource = cls.build_PBEs_render_time_epochs_datasource(curr_sess=curr_sess, **kwargs)
        active_pbe_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(PBEs_interval_datasource)
        active_pbe_interval_rects_item.setToolTip('PBEs')
        return active_pbe_interval_rects_item
    
    
    ##########################################
    ## Laps
    @staticmethod
    def build_Laps_dataframe_formatter(debug_print=False, **kwargs):
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
    def build_Laps_render_time_epochs_datasource(cls, curr_sess, **kwargs):
        """_summary_

        Args:
            curr_sess (DataSession || Laps || Epoch): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            IntervalsDatasource: _description_
        """
        if isinstance(curr_sess, DataSession):
            active_Laps_Epochs = curr_sess.laps.as_epoch_obj() # <Epoch> object
        elif isinstance(curr_sess, Laps):
            active_Laps_Epochs = curr_sess.as_epoch_obj()
        elif isinstance(curr_sess, Epoch):
            active_Laps_Epochs = curr_sess
        else:
            raise NotImplementedError
        return IntervalsDatasource.init_from_epoch_object(active_Laps_Epochs, cls.build_Laps_dataframe_formatter(**kwargs), datasource_name='intervals_datasource_from_laps_epoch_obj')
    
    
    @classmethod
    def build_Laps_2D_render_time_epochs(cls, curr_sess, **kwargs):
        """ 
        Usage:

        """
        # active_Laps_Epochs = curr_sess.laps.as_epoch_obj() # <Epoch> object
        ## IntervalsDatasource version:
        # laps_interval_datasource = IntervalsDatasource.init_from_epoch_object(active_Laps_Epochs, cls.build_Laps_formatter_datasource(**kwargs), datasource_name='intervals_datasource_from_laps_epoch_obj')
        laps_interval_datasource = cls.build_Laps_render_time_epochs_datasource(curr_sess=curr_sess, **kwargs)
        active_laps_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(laps_interval_datasource)
        active_laps_interval_rects_item.setToolTip('Laps')
        return active_laps_interval_rects_item
    
    
    @classmethod
    def add_Laps_2D_render_time_epochs(cls, curr_sess, destination_plot):
        active_interval_rects_item = cls.build_Laps_2D_render_time_epochs(curr_sess=curr_sess)
        
        destination_plot.addPlot(active_interval_rects_item)
        
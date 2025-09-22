### Graphics Items Children Discovery - Programmatically getting rows/columns of GraphicsLayoutWidget
from collections import namedtuple
import numpy as np
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph.graphicsItems as graphicsItems
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem import PlotItem #, PlotCurveItem
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem #, PlotCurveItem
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsLayout import GraphicsLayout
from qtpy import QtCore, QtWidgets, QtGui

# from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot


# """ 

"""

Analagous to `neuropy.utils.matplotlib_helpers`

# PyQtGraph Properties:
win.setBackground
win.setWindowTitle
win.setBackgroundBrush
win.setXRange
win.setYRange

'Show X Grid'
'Show Y Grid'
"""

# visual_config = dict(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'))

# plot_items = [a_child for a_child in main_graphics_layout_widget.items() if isinstance(a_child, (PlotItem))] # ScatterPlotItem
# plot_items

# graphics_layout = main_graphics_layout_widget.ci # <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsLayout.GraphicsLayout at 0x24affb75820>

# graphics_layout.rows
# # {1: {0: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem at 0x24affb9dc10>},
# #  2: {0: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem at 0x24affb164c0>}}

# graphics_layout.items ## item: [(row, col), (row, col), ...]  lists all cells occupied by the item


# """

def inline_mkColor(color, alpha=1.0):
    """ helps build a new QColor for a pen/brush in an inline (single-line) way. """
    out_color = pg.mkColor(color)
    out_color.setAlphaF(alpha)
    return out_color



@function_attributes(short_name=None, tags=['pyqtgraph', 'important', 'useful'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 13:44', related_items=[])
def recover_graphics_layout_widget_item_indicies(graphics_layout_widget, debug_print=False):
    """ ✅WORKS✅ Recovers the row/column indicies for the items of a graphics_layout_widget 
    
    Example:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.helpers import recover_graphics_layout_widget_item_indicies
        
        found_item_rows, found_item_cols, found_items_list, (found_max_row, found_max_col) = recover_graphics_layout_widget_item_indicies(main_graphics_layout_widget, debug_print=True)
        
    Output:
    
    Found Items:
        row_indicies: [1 4]
        indicies: [0 0]
        items: [<pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem object at 0x000001B5C917B160>, <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem object at 0x000001B5C93B2D30>]
    (found_unique_rows: [1 4], found_unique_cols: [0])
    (found_max_row: 4, found_max_col: 0)

    """
    # Need graphics_layout to be a GraphicsLayout object:
    if isinstance(graphics_layout_widget, GraphicsLayout):
        graphics_layout = graphics_layout_widget # input is already a GraphicsLayout object. <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsLayout.GraphicsLayout at 0x24affb75820>
    else:
        # for GraphicsLayoutWidget or GraphicsWindow:
        graphics_layout = graphics_layout_widget.ci

    assert isinstance(graphics_layout, GraphicsLayout), f"type(graphics_layout): {type(graphics_layout)}"
    plot_items_dict = graphics_layout.items ## item: [(row, col), (row, col), ...]  lists all cells occupied by the item
    num_plot_items = len(plot_items_dict)
    found_items_list = list(plot_items_dict.keys())
    plot_items_index_tuple_list = np.array([list(a_index_tuple[0]) for a_index_tuple in list(plot_items_dict.values())])
    plot_items_row_indicies = plot_items_index_tuple_list[:,0]
    plot_items_col_indicies = plot_items_index_tuple_list[:,1]
    
    found_item_rows = np.unique(plot_items_row_indicies)
    found_item_cols = np.unique(plot_items_col_indicies)
    
    num_found_rows = len(found_item_rows)
    num_found_cols = len(found_item_cols)
    
    found_max_row = max(found_item_rows)
    found_max_col = max(found_item_cols)
    if debug_print:
        print(f'Found Items:\n\trow_indicies: {plot_items_row_indicies}\n\tindicies: {plot_items_col_indicies}\n\titems: {found_items_list}\n(found_unique_rows: {found_item_rows}, found_unique_cols: {found_item_cols})\n(found_max_row: {found_max_row}, found_max_col: {found_max_col})')
        # print(f'plot_items_row_indicies: {plot_items_row_indicies}\nplot_items_col_indicies: {plot_items_col_indicies}\nfound_items_list: {found_items_list}\n(found_unique_rows: {found_item_rows}\nfound_unique_cols: {found_item_cols})\n(found_max_row: {found_max_row}, found_max_col: {found_max_col})')
    return found_item_rows, found_item_cols, found_items_list, (found_max_row, found_max_col)




# ==================================================================================================================== #
# RectangleRenderTupleHelpers                                                                                          #
# ==================================================================================================================== #
QColorTuple = namedtuple('QColorTuple', ['hexColor', 'alpha'])
QPenTuple = namedtuple('QPenTuple', ['color', 'width'])
QBrushTuple = namedtuple('QBrushTuple', ['color'])


QPenFlatTuple = namedtuple('QPenFlatTuple', ['hexColor', 'alpha', 'width'])
QBrushFlatTuple = namedtuple('QBrushFlatTuple', ['hexColor', 'alpha'])



@metadata_attributes(short_name=None, tags=['class', 'helper', 'pyqtgraph', 'QPen', 'Qt', 'QBrush', 'Helpful', 'TO_REFACTOR'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 13:45', related_items=[])
class RectangleRenderTupleHelpers:
    """ class for use in copying, serializing, etc the list of tuples used by IntervalRectsItem

    Refactored out of `pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem.IntervalRectsItem` on 2022-12-05 

    Usage:

        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.helpers import RectangleRenderTupleHelpers
        
    Known Usages:
        Used in `IntervalRectsItem` and `CustomIntervalRectsItem` to copy themselves

        # Copy Constructors: _________________________________________________________________________________________________ #
        def __copy__(self):
            independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
            return CustomIntervalRectsItem(independent_data_copy)
        
        def __deepcopy__(self, memo):
            independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
            return CustomIntervalRectsItem(independent_data_copy)
            # return CustomIntervalRectsItem(copy.deepcopy(self.data, memo))


    """
    @classmethod
    def QColor_to_simple_columns_dict(cls, value):
        """Resolves into basic datatypes:
        color: a HexRgb string (without opacity)
        alpha: a float value indicating the opacity
        """
        return {'hexColor': value.name(QtGui.QColor.HexRgb),'alpha':value.alphaF()}
    
    @staticmethod
    def QColor_to_tuple(value):
        return QColorTuple(hexColor=value.name(QtGui.QColor.HexRgb), alpha=value.alphaF())


    _color_process_fn = lambda a_color: pg.colorStr(a_color) # a_pen.color()
    # _color_process_fn = lambda a_color: RectangleRenderTupleHelpers.QColor_to_simple_columns_dict(a_color)


    @staticmethod
    def QPen_to_dict(a_pen):
        return {'color': RectangleRenderTupleHelpers._color_process_fn(a_pen.color()),'width':a_pen.widthF()}
        # return {**RectangleRenderTupleHelpers.QColor_to_simple_columns_dict(a_pen.color()),'width':a_pen.widthF()}

    @staticmethod
    def QBrush_to_dict(a_brush):
        return {'color': RectangleRenderTupleHelpers._color_process_fn(a_brush.color())} # ,'gradient':a_brush.gradient()
        # return {**RectangleRenderTupleHelpers.QColor_to_simple_columns_dict(a_brush.color())} # ,'gradient':a_brush.gradient()

    @staticmethod
    def QPen_to_tuple(a_pen):
        return QPenTuple(color=RectangleRenderTupleHelpers._color_process_fn(a_pen.color()), width=a_pen.widthF())
        # return QPenTuple(**RectangleRenderTupleHelpers.QColor_to_simple_columns_dict(a_pen.color()), width=a_pen.widthF())

    @staticmethod
    def QBrush_to_tuple(a_brush):
        return QBrushTuple(color=RectangleRenderTupleHelpers._color_process_fn(a_brush.color()))
        # return QBrushTuple(**RectangleRenderTupleHelpers.QColor_to_simple_columns_dict(a_brush.color()))

    
    @classmethod
    def get_serialized_data(cls, tuples_data):
        """ converts the list of (float, float, float, float, QPen, QBrush) tuples into a list of (float, float, float, float, pen_color_hex:str, brush_color_hex:str) for serialization. """            
        return [(start_t, series_vertical_offset, duration_t, series_height, cls.QPen_to_dict(pen), cls.QBrush_to_dict(brush)) for (start_t, series_vertical_offset, duration_t, series_height, pen, brush) in tuples_data]

    
    @staticmethod
    def get_deserialized_data(seralized_tuples_data):
        """ converts the list of (float, float, float, float, pen_color_hex:str, brush_color_hex:str) tuples back to the original (float, float, float, float, QPen, QBrush) list
        
        Inverse operation of .get_serialized_data(...).
        
        Usage:
            seralized_tuples_data = RectangleRenderTupleHelpers.get_serialized_data(tuples_data)
            tuples_data = RectangleRenderTupleHelpers.get_deserialized_data(seralized_tuples_data)
        """        
        return [(start_t, series_vertical_offset, duration_t, series_height, pg.mkPen(pen_color_hex), pg.mkBrush(**brush_color_hex)) for (start_t, series_vertical_offset, duration_t, series_height, pen_color_hex, brush_color_hex) in seralized_tuples_data]
        # return [(start_t, series_vertical_offset, duration_t, series_height, pg.mkPen(inline_mkColor(pen_color_hex)), pg.mkBrush(**brush_color_hex)) for (start_t, series_vertical_offset, duration_t, series_height, pen_color_hex, brush_color_hex) in seralized_tuples_data]




    @classmethod
    def copy_data(cls, tuples_data):
        seralized_tuples_data = cls.get_serialized_data(tuples_data).copy()
        return cls.get_deserialized_data(seralized_tuples_data)





@function_attributes(short_name=None, tags=['pyqtgraph', 'scatterplot', 'clickable', 'interactive'], conforms_to=[], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 13:37')
def _helper_make_scatterplot_clickable(main_scatter_plot, enable_hover:bool=False):
    """ pyqtgraph 
    
    Usage:
    
    lastClicked, clickedPen, (main_scatter_hovered_connection, main_scatter_clicked_connection) = _helper_make_scatterplot_clickable(a_plot)
    
    """
    # Highlights the hovered spikes white:
    # main_scatter_plot.addPoints(hoverable=True,
    # 	# hoverSymbol=vtick, # hoverSymbol='s',
    # 	hoverSize=7, # default is 5
    # 	)


    ## Clickable/Selectable Spikes:
    # global lastClicked  # Declare lastClicked as a global variable
    # Will make all plots clickable
    clickedPen = pg.mkPen('#DDD', width=2)
    lastClicked = []
    def _test_scatter_plot_clicked(plot, points):
        """ captures `lastClicked` """
        global lastClicked  # Declare lastClicked as a global variable
        for p in lastClicked:
            p.resetPen()
        print("clicked points", points)
        for p in points:
            p.setPen(clickedPen)
        lastClicked = points

    main_scatter_clicked_connection = main_scatter_plot.sigClicked.connect(_test_scatter_plot_clicked)

    ## Hoverable Spikes:
    if enable_hover:
        def _test_scatter_plot_hovered(plt, points, ev):
            # sigHovered(self, points, ev)
            print(f'_test_scatter_plot_hovered(plt: {plt}, points: {points}, ev: {ev})')
            if (len(points) > 0):
                curr_point = points[0]
        main_scatter_hovered_connection = main_scatter_plot.sigHovered.connect(_test_scatter_plot_hovered)
    else:
        main_scatter_hovered_connection = None

    return lastClicked, clickedPen, (main_scatter_hovered_connection, main_scatter_clicked_connection)




class ScrollableRasterViewOwnerMixin:
    """ 

        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.helpers import ScrollableRasterViewOwnerMixin

    """
    # ==================================================================================================================== #
    # Events                                                                                                               #
    # ==================================================================================================================== #
        
    @property
    def should_debug_print_interaction_events(self) -> bool:
        """The should_debug_print_interaction_events property."""
        if not hasattr(self.params, 'should_debug_print_interaction_events'):
            self.params.should_debug_print_interaction_events = False ## default value setting
        return self.params.should_debug_print_interaction_events
    @should_debug_print_interaction_events.setter
    def should_debug_print_interaction_events(self, value: bool):
        self.params.should_debug_print_interaction_events = value

    # @QtCore.Property(int) # Note that this ia *pyqt*Property, meaning it's available to pyqt
    # def scheduledAnimationSteps(self):
    #     """The scheduledAnimationSteps property."""
    #     return self._scheduledAnimationSteps
    # @scheduledAnimationSteps.setter
    # def scheduledAnimationSteps(self, value):
    #     if self._scheduledAnimationSteps != value:
    #         # Only update if the value has changed from the previous one:
    #         self._scheduledAnimationSteps = value
    #         # TODO: maybe use a rate-limited signal that's emitted instead so this isn't called too often during interpolation?
    #         # self.shift_animation_frame_val(self._scheduledAnimationSteps) # TODO: this isn't quite right
            
    @property
    def animation_delegate(self):
        """The animation_delegate property."""
        active_2d_plot = self.ui.controlling_widget # Spike2DRaster
        return active_2d_plot
    
    @property
    def animation_active_time_window(self):
        """The animation_active_time_window property."""
        return self.animation_delegate.animation_active_time_window
    @animation_active_time_window.setter
    def animation_active_time_window(self, value):
        self.animation_delegate.animation_active_time_window = value

    @property
    def animation_playback_direction_multiplier(self):
        """The animation_playback_direction_multiplier property."""
        return self.params.animation_playback_direction_multiplier
    @animation_playback_direction_multiplier.setter
    def animation_playback_direction_multiplier(self, value):
        self.params.animation_playback_direction_multiplier = value

    @property
    def animation_time_step(self):
        """The animation_time_step property."""
        return self.params.animation_time_step
    @animation_time_step.setter
    def animation_time_step(self, value):
        self.params.animation_time_step = value
    

    def _setup_ScrollableRasterViewOwnerMixin(self):
        """ MUST BE CALLED ON STARTUP
        
        """
        self.params.should_debug_print_interaction_events = False
        
        ## Scrolling Properties:
        self._scheduledAnimationSteps = 0
        self.params.scrollStepMultiplier = 30.0 # The multiplier by which each scroll step is multiplied. Decrease this value to increase scrolling precision (making the same rotation of the mousewheel scroll less in time).
        self.params.animation_playback_direction_multiplier = 1.0
        self.params.animation_time_step = 2.0
        
        self.enable_smooth_scrolling_animation = False # UNTESTED

        # if self.enable_smooth_scrolling_animation:
        #     ## Add the QPropertyAnimation for smooth scrolling, but do not start it:
        #     self.ui.scrollAnim = QtCore.QPropertyAnimation(self, b"numScheduledScalings") # the animation will act on the self.numScheduledScalings pyqtProperty
        #     # self.ui.scrollAnim.setEndValue(0) # Update the end value
        #     self.ui.scrollAnim.setDuration(250) # set duration in milliseconds
            
        #     ## QTimeLine-style smooth scrolling:
        #     self.ui.scrollAnimTimeline = QtCore.QTimeLine(250, parent=self) # Make a new QTimeLine with a 250ms animation duration
        #     self.ui.scrollAnimTimeline.setUpdateInterval(20)
        #     self.ui.scrollAnimTimeline.setCurveShape(QtCore.QTimeLine.CurveShape.LinearCurve)
        #     self.ui.scrollAnimTimeline.setFrameRange(0, 100)
        #     self.ui.scrollAnimTimeline.frameChanged.connect(self.onScrollingTimelineFired)
        #     # self.ui.scrollAnimTimeline.valueChanged.connect(self.onScrollingTimelineFired)
        #     self.ui.scrollAnimTimeline.finished.connect(self.onScrollingTimelineAnimationFinished)
        #     # self.ui.scrollAnimTimeline.start() # Do not start it

        # else:
        self.ui.scrollAnim = None
        self.ui.scrollAnimTimeline = None


    ###################################
    #### EVENT HANDLERS
    ##################################
    

    # @pyqtExceptionPrintingSlot(float)
    def update_animation(self, next_start_timestamp: float):
        """ Actually updates the animation given the next_start_timestep
            extracted from Spike3DRasterWindowWidget.shift_animation_frame_val(...)
        """
        if self.should_debug_print_interaction_events:
            print(f'ScrollableRasterViewOwnerMixin.update_animation(next_start_timestamp: {next_start_timestamp})')
        # self.animation_active_time_window.update_window_start(next_start_timestamp) # calls update_window_start, so any subscribers should be notified.
        next_end_timestamp = next_start_timestamp + self.animation_active_time_window.window_duration
        
        # Update the windows once before showing the UI:
        self.animation_delegate.update_scroll_window_region(next_start_timestamp, next_end_timestamp, block_signals=True) # self.spike_raster_plt_2d.window_scrolled should be emitted        
        # signal emit:
        self.animation_delegate.window_scrolled.emit(next_start_timestamp, next_end_timestamp)
        # update_scroll_window_region
        # self.ui.spike_raster_plt_3d.spikes_window.update_window_start_end(self.ui.spike_raster_plt_2d.spikes_window.active_time_window[0], self.ui.spike_raster_plt_2d.spikes_window.active_time_window[1])
        # self.bottom_playback_control_bar_widget.on_window_changed(next_start_timestamp, next_end_timestamp) ## direct
        

    # @pyqtExceptionPrintingSlot(int)
    def shift_animation_frame_val(self, shift_frames: int):
        if self.should_debug_print_interaction_events:
            print(f'ScrollableRasterViewOwnerMixin.shift_animation_frame_val(shift_frames: {shift_frames})')
        next_start_timestamp = self.animation_active_time_window.active_window_start_time + (self.animation_playback_direction_multiplier * self.animation_time_step * float(shift_frames)) # Equivalent to self.compute_frame_shifted_start_timestamp(shift_frames)
        self.update_animation(next_start_timestamp)
        


    # def onScrollingTimelineAnimationFinished(self):
    #     """ used for the QTimeline version of the smooth scrolling animation """
    #     print(f'onScrollingTimelineAnimationFinished()')
    #     print(f'\t self._scheduledAnimationSteps: {self._scheduledAnimationSteps}')
    #     self.scheduledAnimationSteps = 0 # updated method that actually zeros out the scheduled scalings        
    #     print('\t zeroing out.')
    #     # if self._scheduledAnimationSteps > 0:
    #     #     self._scheduledAnimationSteps -= 1
    #     # else:
    #     #     self._scheduledAnimationSteps += 1
        

    # def onScrollingTimelineFired(self, x):
    #     """ used for the QTimeline version of the smooth scrolling animation 
        
    #     # OLD VERSION: x appears to be a float between 0.0-1.0 by default that indicates how far along in the animation it is
        
    #     x is an int indicating the number of frames for the timeline that were set with between 0.0-1.0 by default that indicates how far along in the animation it is
        
    #     """
    #     print(f'onScrollingTimelineFired(x: {x})')
    #     # self.shift_animation_frame_val(x)        
    #     curr_shifted_next_start_time = self.compute_frame_shifted_start_timestamp(x)
    #     print(f'\t curr_shifted_next_start_time: {curr_shifted_next_start_time}')
    #     self.update_animation(curr_shifted_next_start_time)
    #     self._scheduledAnimationSteps = self._scheduledAnimationSteps - x # subtract off the frames that have been shifted


    # @function_attributes(short_name=None, tags=['TODO', 'ACTIVE', 'programmatic', 'scrolling', 'time'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-18 12:09', related_items=[])
    # def programmatically_scroll_to_time(self, new_time):
    #     numSteps: int = 3
    #     updatedNumScheduledScalings = self._scheduledAnimationSteps + numSteps
    #     if (updatedNumScheduledScalings * numSteps < 0):
    #         updatedNumScheduledScalings = numSteps # if user moved the wheel in another direction, we reset previously scheduled scalings
        
    #     if self.enable_smooth_scrolling_animation:
    #         ## QTimeline version:
    #         self._scheduledAnimationSteps = updatedNumScheduledScalings # Set the updated number of scalings:
    #         self.ui.scrollAnimTimeline.setEndFrame(self._scheduledAnimationSteps)
    #         self.ui.scrollAnimTimeline.start() # Start the timeline's animation event
    #     else:
    #         # No animation, just update directly ("old way")
    #         self._scheduledAnimationSteps = updatedNumScheduledScalings
    #         self.shift_animation_frame_val(self._scheduledAnimationSteps)
    #         self._scheduledAnimationSteps = 0 # New method: zero it out instead of having it compound


    # self.spikes_window.windowed_data_window_duration_changed_signal.connect(self.on_windowed_data_window_duration_changed)
    # self.spikes_window.windowed_data_window_updated_signal.connect(self.on_windowed_data_window_changed)
    
    def eventFilter(self, watched, event):
        """  has to be installed on an item like:
            self.grid = pg.GraphicsLayoutWidget()
            self.top_left = self.grid.addViewBox(row=1, col=1)
            self.top_left.installEventFilter(self)
        
        """
        from pyphocorehelpers.gui.Qt.qevent_lookup_helpers import QEventLookupHelpers # used for ScrollableRasterViewOwnerMixin

        # print(f'ScrollableRasterViewOwnerMixin.eventFilter(self, watched, event)')
        delta = None
        if (event.type() == QtCore.QEvent.GraphicsSceneWheel):
            # QtCore.QEvent.GraphicsSceneWheel
            """             
            event.delta(): (gives values like +/- 120, 240, etc) # Returns the distance that the wheel is rotated, in eighths (1/8s) of a degree. A positive value indicates that the wheel was rotated forwards away from the user; a negative value indicates that the wheel was rotated backwards toward the user.
                Most mouse types work in steps of 15 degrees, in which case the delta value is a multiple of 120 (== 15 * 8).

            event.orientation(): 1 for alternative scroll wheel dir and 2 for primary scroll wheel dir
            
            """
            if self.should_debug_print_interaction_events:
                print(f'ScrollableRasterViewOwnerMixin.eventFilter(...)\n\t detected event.type() == QtCore.QEvent.GraphicsSceneWheel')
                print(f'\twatched: {watched}\n\tevent: {event}')
                print(f'\tevent.delta(): {event.delta()}')
                print(f'\tevent.orientation(): {event.orientation()}')
                # print(f'\tevent.phase(): {event.phase()}')
                # print(f'\tevent.pixelDelta(): {event.pixelDelta()}')
                
            delta = event.delta()
        
        
        elif (event.type() == QtCore.QEvent.Wheel): # the second case (QtGui.QWheelEvent) doesn't even exist I don't think. IDK why ChatGPT said to use it.
            """ the event is an instance of `QtGui.QWheelEvent`, but the event's .type() is NEVER QtGui.QWheelEvent, that's not even a possible type. """
            if self.should_debug_print_interaction_events:
                print(f'ScrollableRasterViewOwnerMixin.eventFilter(...)\n\t detected event.type() == QtCore.QEvent.Wheel')
                print(f'\twatched: {watched}\n\tevent: {event}')
                print(f'\tevent.angleDelta(): {event.angleDelta()}')
                
            delta = event.angleDelta().x()
            if delta == 0:
                delta = event.angleDelta().y()
            
        else:
            delta = None
            if self.should_debug_print_interaction_events:
                print(f'\t unhandled event {QEventLookupHelpers.get_event_string(event)}')
                
        if (delta is not None) and (abs(delta) > 0):
            ## do the scroll
            if self.should_debug_print_interaction_events:
                print(f'\tperofmring scroll with delta: {delta}')

            numDegrees = delta / 8
            numSteps = numDegrees / 15 # see QWheelEvent documentation
            numSteps = int(round(float(self.params.scrollStepMultiplier) * float(numSteps)))
                       
            updatedNumScheduledScalings = self._scheduledAnimationSteps + numSteps
            if (updatedNumScheduledScalings * numSteps < 0):
                updatedNumScheduledScalings = numSteps # if user moved the wheel in another direction, we reset previously scheduled scalings
            
            # if self.enable_smooth_scrolling_animation:
            #     # ## pyqt Property Animation Method:            
            #     # self.ui.scrollAnim.setEndValue(updatedNumScheduledScalings) # Update the end value
            #     # self.ui.scrollAnim.start() # start the animation
                
            #     ## QTimeline version:
            #     self._scheduledAnimationSteps = updatedNumScheduledScalings # Set the updated number of scalings:
            #     self.ui.scrollAnimTimeline.setEndFrame(self._scheduledAnimationSteps)
            #     self.ui.scrollAnimTimeline.start() # Start the timeline's animation event
            # else:
            # No animation, just update directly ("old way")
            self._scheduledAnimationSteps = updatedNumScheduledScalings
            self.shift_animation_frame_val(self._scheduledAnimationSteps)
            self._scheduledAnimationSteps = 0 # New method: zero it out instead of having it compound

            return True
        # END if (delta is not None) a....
        else:
            # Unknown event type
            if self.should_debug_print_interaction_events:
                print(f'\t unhandled event {QEventLookupHelpers.get_event_string(event)}')


        # if source == self.ui.jumpToHourMinSecTimeEdit:
        #     if event.type() == event.FocusIn:
        #         self.set_jump_time_white_style()
        #     elif event.type() == event.FocusOut:
        #         self.set_jump_time_light_grey_style()
                
        # elif source == self.time_edit and event.type() == event.KeyPress:
        #     # """Handle Enter key to finalize and lose focus."""
        #     if event.key() in (Qt.Key_Return, Qt.Key_Enter):
        #         self.time_edit.clearFocus()  # Finalize and lose focus
        #         return True  # Mark event as handled
            
        # elif (source == self.ui.doubleSpinBox_ActiveWindowStartTime) or (source == self.ui.doubleSpinBox_ActiveWindowEndTime):
        #     if event.type() == event.KeyPress:
        #         # """Handle Enter key to finalize and lose focus."""
        #         if event.key() in (Qt.Key_Return, Qt.Key_Enter):
        #             source.clearFocus()  # Finalize and lose focus
        #             return True  # Mark event as handled
        #         else: 
        #             ## other key presses, such as type numbers and such
        #             pass

        # if self.params.debug_print:
        #     print(f'Spike3DRasterBottomPlaybackControlBar.eventFilter(source: {source}, event: {event})')

        # If not a particularlly handled case, do the default thing.
        return super().eventFilter(watched, event)


    def wheelEvent(self, event):
        super(ScrollableRasterViewOwnerMixin, self).wheelEvent(event)
        if self.should_debug_print_interaction_events:
            print(f'ScrollableRasterViewOwnerMixin.wheelEvent(...)')
            # self.x = self.x + event.delta()/120
            # print self.x
            # self.label.setText("Total Steps: "+QString.number(self.x))        
            print(f'\t wheelEvent(event: {event}')
    

    ########################################################
    ## For Key Press Events:
    ########################################################

    ##-----------------------------------------
    def keyPressEvent(self, event):
        if self.should_debug_print_interaction_events:
            print(f'pressed from ScrollableRasterViewOwnerMixin.keyPressEvent(event): {event.key()}')
            print(f'\t event.modifiers(): {event.modifiers()}')
            # e.Modifiers()
            print('event received @ ScrollableRasterViewOwnerMixin')
        super(ScrollableRasterViewOwnerMixin, self).keyPressEvent(event)
        if self.should_debug_print_interaction_events:
            if event.key() == QtCore.Qt.Key_Space:
                print(f'\t detected event: {event.key()}')
            elif event.key() == QtCore.Qt.Key_0:
                print(f'\t detected event: {event.key()}')
            else:
                print(f'\t undetected event')
        # self.keyPressed.emit(event)



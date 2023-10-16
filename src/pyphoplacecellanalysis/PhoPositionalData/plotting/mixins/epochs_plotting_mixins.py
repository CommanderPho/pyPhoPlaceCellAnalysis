import param
import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtGui # for QColor
from qtpy.QtGui import QColor, QBrush, QPen

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.helpers import inline_mkColor
from pyphoplacecellanalysis.General.Model.Configs.ParamConfigs import BasePlotDataParams





class EpochDisplayConfig(BasePlotDataParams):
    """ NOTE: Upon reviewing many different versions of my plotting implementations, this Param-based one is the most succinct and smooth.

    This class uses the 'param' library to observe changes to its members and perform corresponding updates to the class that holds it when they happen:
    
    From OccupancyPlottingMixin.setup_occupancy_plotting_mixin(self):
        # Setup watchers:    
        self.occupancy_plotting_config.param.watch(self.plot_occupancy_bars, OccupancyPlottingConfig._config_update_watch_labels(), queued=True)
        self.occupancy_plotting_config.param.watch(self.on_occupancy_plot_update_visibility, OccupancyPlottingConfig._config_visibility_watch_labels(), queued=True)
    
    
    Note that _config_update_watch_labels() provides the names/labels of the properties that when updated trigger plot_occupancy_bars(...)
        and _config_visibility_watch_labels() provides those for on_occupancy_plot_update_visibility(...)
    """
    # debug_logging = False

    # @staticmethod
    # def _config_update_watch_labels():
    #     return ['barOpacity', 'labelsShowPoints', 'labelsOpacity', 'dropBelowThreshold']
    # @staticmethod
    # def _config_visibility_watch_labels():
    #     return ['labelsAreVisible', 'isVisible']
    
    # Overriding defaults from parent
    name = param.String(default='SessionEpochs')
    isVisible = param.Boolean(default=False, doc="Whether the epochs are visible") # default to False

    # Bar properties:
    pen_color = param.Color(default='#00ffff', doc="The edge Color")
    pen_opacity = param.Number(default=0.8, bounds=(0.0, 1.0), step=0.1)

    brush_color = param.Color(default='#00ffff', doc="The fill Color")
    brush_opacity = param.Number(default=0.5, bounds=(0.0, 1.0), step=0.1)

    # Smart Sizing/Positioning Properties:
    desired_height_ratio = param.Number(default=1.0, bounds=(0.1, 50.0), step=0.1)

    # Location Properties:
    height = param.Number(default=7.5, bounds=(0.1, 50.0), step=0.1)
    y_location = param.Number(default=-12.0, bounds=(-200.0, 1000.0), step=2.0)

    @classmethod
    def init_from_config_dict(cls, name: str, config_dict: dict):
        """

        Example Usage:

            epochs_update_dict = {
                'Replays':dict(y_location=-10.0, height=7.5, pen_color=inline_mkColor('orange', 0.8), brush_color=inline_mkColor('orange', 0.5)),
                'PBEs':dict(y_location=-2.0, height=1.5, pen_color=inline_mkColor('pink', 0.8), brush_color=inline_mkColor('pink', 0.5)),
                'Ripples':dict(y_location=-12.0, height=1.5, pen_color=inline_mkColor('cyan', 0.8), brush_color=inline_mkColor('cyan', 0.5)),
                'SessionEpochs':dict(y_location=-12.0, height=1.5, pen_color=inline_mkColor('cyan', 0.8), brush_color=inline_mkColor('cyan', 0.5)),
            }
            epochs_update_dict = {k:EpochDisplayConfig.init_from_config_dict(name=k, config_dict=v) for k,v in epochs_update_dict.items()}
            epochs_update_dict


        """
        pen_qcolor = config_dict.pop('pen_color', None)
        brush_qcolor = config_dict.pop('brush_color', None)
        out = cls(name=name, **config_dict)
        if pen_qcolor is not None:
            if isinstance(pen_qcolor, QColor):
                out.pen_Qcolor = pen_qcolor
            else:
                out.pen_color = pen_qcolor # set raw
        if brush_qcolor is not None:
            if isinstance(brush_qcolor, QColor):
                out.brush_QColor = brush_qcolor
            else:
                out.brush_color = brush_qcolor # set raw
        return out
    
    @property
    def pen_QColor(self):
        return inline_mkColor(self.pen_color, self.pen_opacity)
    @pen_QColor.setter
    def pen_QColor(self, value):
        self.pen_color = value.name(QtGui.QColor.HexRgb)
        self.pen_opacity = value.alphaF()


    @property
    def brush_QColor(self):
        return inline_mkColor(self.brush_color, self.brush_opacity)
    @brush_QColor.setter
    def brush_QColor(self, value):
        self.brush_color = value.name(QtGui.QColor.HexRgb)
        self.brush_opacity = value.alphaF()



    # def to_plot_config_dict(self):
    #     issue_labels = {'name': 'OccupancyLabels', 'name': 'Occupancy'}
    #     return {'drop_below_threshold': self.dropBelowThreshold, 'opacity': self.barOpacity, 'shape': 'rounded_rect', 'shape_opacity': self.labelsOpacity, 'show_points': self.labelsShowPoints}
    
    # def to_bars_plot_config_dict(self):
    #     return {'name': 'Occupancy', 'drop_below_threshold': self.dropBelowThreshold, 'opacity': self.barOpacity}
    
    # def to_labels_plot_config_dict(self):
    #     return {'name': 'OccupancyLabels', 'shape': 'rounded_rect', 'shape_opacity': self.labelsOpacity, 'show_points': self.labelsShowPoints}
    
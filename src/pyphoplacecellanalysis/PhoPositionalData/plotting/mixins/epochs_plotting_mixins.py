from typing import List, Dict, Optional, Tuple
import param
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtGui # for QColor
from qtpy.QtGui import QColor, QBrush, QPen

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.helpers import inline_mkColor
from pyphoplacecellanalysis.General.Model.Configs.ParamConfigs import BasePlotDataParams
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

""" 
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.epochs_plotting_mixins import EpochDisplayConfig, _get_default_epoch_configs

"""

# class ColorWithOpacity(param.Parameter):
#     """Integer Parameter that must be even"""

#     def _validate_value(self, val, allow_None):
#         super(ColorWithOpacity, self)._validate_value(val, allow_None)
#         if not isinstance(val, numbers.Number):
#             raise ValueError("ColorWithOpacity parameter %r must be a number, "
#                              "not %r." % (self.name, val))
        
#         if not (val % 2 == 0):
#             raise ValueError("ColorWithOpacity parameter %r must be even, "
#                              "not %r." % (self.name, val))


# @metadata_attributes(short_name=None, tags=['epoch', 'params', 'config'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-17 03:26', related_items=[])
class EpochDisplayConfig(BasePlotDataParams):
    """ NOTE: Upon reviewing many different versions of my plotting implementations, this Param-based one is the most succinct and smooth.

    This class uses the 'param' library to observe changes to its members and perform corresponding updates to the class that holds it when they happen:
    

    from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.epochs_plotting_mixins import EpochDisplayConfig


    ## Use with Panel:

    import panel as pn

    pn.extension()
   


    """
    # debug_logging = False

    # @staticmethod
    # def _config_update_watch_labels():
    #     return ['barOpacity', 'labelsShowPoints', 'labelsOpacity', 'dropBelowThreshold']
    # @staticmethod
    # def _config_visibility_watch_labels():
    #     return ['labelsAreVisible', 'isVisible']
    
    # Overriding defaults from parent
    name = param.String(default='SessionEpochs', doc='Name of the epochs')
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

    @staticmethod
    def _config_update_watch_labels():
        ## any updates
        return ['pen_color', 'pen_opacity', 'brush_color', 'brush_opacity', 'desired_height_ratio', 'height', 'y_location', 'desired_height_ratio', 'height', 'y_location', 'isVisible']
    @staticmethod
    def _config_display_watch_labels():
        return ['pen_color', 'pen_opacity', 'brush_color', 'brush_opacity', 'desired_height_ratio', 'height', 'y_location']
    @staticmethod
    def _config_layout_watch_labels():
        return ['desired_height_ratio', 'height', 'y_location']
    @staticmethod
    def _config_visibility_watch_labels():
        return ['isVisible']
    

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


    @classmethod
    def init_from_visualization_dataframe_row(cls, name: str, y_location, height, pen_tuple, brush_tuple):
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
        # print(f'pen_tuple: {pen_tuple}, brush_tuple: {brush_tuple}')

        ## Try 3-tuple unwrap
        if len(pen_tuple) == 3:
            (pen_color, pen_opacity, pen_width) = pen_tuple
        elif len(pen_tuple) == 2:
            (_temp_pen_color_hex_argb, pen_width) = pen_tuple.color, pen_tuple.width # QPenTuple(color='ff0000ff', width=1.0)
            _temp_pen_qcolor = inline_mkColor(_temp_pen_color_hex_argb)
            pen_color = _temp_pen_qcolor.name(QtGui.QColor.HexRgb)
            pen_opacity = _temp_pen_qcolor.alphaF()
        else:
            raise NotImplementedError

        if len(brush_tuple) == 2:
            (brush_color, brush_opacity) = brush_tuple
        elif len(brush_tuple) == 1:
            # brush_tuple: QBrushTuple(color='ff000080')
            _temp_brush_color_hex_argb = brush_tuple.color
            _temp_brush_qcolor = inline_mkColor(_temp_brush_color_hex_argb)
            brush_color = _temp_brush_qcolor.name(QtGui.QColor.HexRgb)
            brush_opacity = _temp_brush_qcolor.alphaF()
        else:
            raise NotImplementedError

        return cls(name=name, isVisible=True, y_location=y_location, height=height, pen_color=pen_color, pen_opacity=pen_opacity, brush_color=brush_color, brush_opacity=brush_opacity)


    @staticmethod
    def _labels_autoqualify_for_named_multi_epoch_mode(label_names: List[str], series_name: str, a_df, max_rows: int = 20) -> bool:
        """True only for semantic named epochs (pre/maze1/post1), not lap-index labels ('1','2',...) or bulk uniform series."""
        n_rows: int = len(label_names)
        if n_rows == 0 or n_rows > max_rows:
            return False
        unique_labels: List[str] = list(dict.fromkeys(str(x) for x in label_names))
        if len(unique_labels) != n_rows:
            return False
        if 'lap_id' in a_df.columns and np.all(a_df['label'].astype(str).to_numpy() == a_df['lap_id'].astype(str).to_numpy()):
            return False
        if all(lbl.strip().isdigit() for lbl in unique_labels):
            return False
        return True


    @classmethod
    def _init_configs_list_single_series_fallback(cls, name: str, a_ds) -> List["EpochDisplayConfig"]:
        """Single series-level config from compressed viz rows (Laps/PBEs/Ripples and forced-single mode)."""
        a_serializable_df = a_ds.get_serialized_data(drop_duplicates=True)
        assert np.all(np.isin(['series_vertical_offset','series_height','pen','brush'], a_serializable_df.columns))
        return [cls.init_from_visualization_dataframe_row(name, y_location, height, a_pen_tuple, a_brush_tuple) for y_location, height, a_pen_tuple, a_brush_tuple in zip(a_serializable_df['series_vertical_offset'], a_serializable_df['series_height'], a_serializable_df['pen'], a_serializable_df['brush'])]


    @classmethod
    def init_configs_list_from_interval_datasource_df(cls, name: str, a_ds, is_named_multi_epoch_series: Optional[bool] = None) -> List["EpochDisplayConfig"]:
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
        if is_named_multi_epoch_series is None:
            is_named_multi_epoch_series = getattr(a_ds, 'is_named_multi_epoch_series', None)
        if is_named_multi_epoch_series is False:
            return cls._init_configs_list_single_series_fallback(name, a_ds)

        from pyphocorehelpers.gui.Qt.color_helpers import ColorDataframeColumnHelpers

        try:
            a_serializable_df = a_ds.df.copy() ## try this first
            assert np.all(np.isin(['series_vertical_offset','series_height','pen','brush'], a_serializable_df.columns))
            label_names: List[str] = a_serializable_df['label'].to_list()
            unique_label_names: List[str] = np.unique(label_names)
            assert len(unique_label_names) == len(a_serializable_df), f"len(unique_label_names): {len(unique_label_names)}, unique_label_names: {unique_label_names} != len(a_serializable_df): {len(a_serializable_df)}\n\ta_serializable_df: {a_serializable_df}"
            if is_named_multi_epoch_series is not True and not cls._labels_autoqualify_for_named_multi_epoch_mode(label_names, name, a_serializable_df):
                raise ValueError(f"series {name!r} is not a named multi-epoch series (labels={list(unique_label_names)[:8]}{'...' if len(unique_label_names) > 8 else ''})")
            ## serialize the raw QPen/QBrush objects to tuples so `init_from_visualization_dataframe_row` (which does `len(...)`) doesn't crash. isinstance guards pass through already-serialized tuples unchanged.
            pen_tuples = [ColorDataframeColumnHelpers.QPen_to_tuple(a_pen) if isinstance(a_pen, QPen) else a_pen for a_pen in a_serializable_df['pen']]
            brush_tuples = [ColorDataframeColumnHelpers.QBrush_to_tuple(a_brush) if isinstance(a_brush, QBrush) else a_brush for a_brush in a_serializable_df['brush']]
            out_list = [cls.init_from_visualization_dataframe_row(a_name, y_location, height, a_pen_tuple, a_brush_tuple) for a_name, y_location, height, a_pen_tuple, a_brush_tuple in zip(label_names, a_serializable_df['series_vertical_offset'], a_serializable_df['series_height'], pen_tuples, brush_tuples)]
            return out_list

        except Exception as e:
            print(f'WARNING: FALLBACK to old method that does not work for multi-interval items (due to error: {e}).')
            return cls._init_configs_list_single_series_fallback(name, a_ds)
        




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


    def to_dict(self) -> dict:
        """ returns as a dictionary representation """
        return dict(name=self.name, y_location=self.y_location, height=self.height, pen_color=self.pen_QColor, brush_color=self.brush_QColor, isVisible=self.isVisible)


    def to_clipboard_epochs_update_dict_code(self) -> Tuple[str, bool]:
        """
        Returns a tuple of (code_string, clipboard_success) where:
        - code_string: Python code that can be pasted elsewhere to recreate an
          `epochs_update_dict` entry for this EpochDisplayConfig. The output is a
          single dictionary item (with key=self.name) showing the non-default config
          values, suitable for copy-paste.
        - clipboard_success: True if code was successfully copied to clipboard, False otherwise.
        
        Also attempts to copy the code to the system clipboard.
        """
        # Collect the relevant fields for code generation
        fields = [
            'y_location', 'height', 'pen_color', 'pen_opacity',
            'brush_color', 'brush_opacity', 'isVisible'
        ]
        # Build the dictionary for this config
        config_dict = {}
        for field in fields:
            value = getattr(self, field)
            # Format color values as hex strings if necessary
            if 'color' in field and isinstance(value, str):
                value_str = f"'{value}'"
            else:
                value_str = repr(value)
            config_dict[field] = value_str

        # Create the dict display: e.g.,
        # 'MyEpoch': dict(y_location=-12.0, height=1.5, pen_color='#00ffff', pen_opacity=0.8, brush_color='#00ffff', brush_opacity=0.5)
        dict_items = ', '.join([f"{k}={v}" for k, v in config_dict.items()])
        code_str = f"'{self.name}': dict({dict_items})"

        # Try copying to the clipboard with consistent Qt wrapper
        clipboard_success = False
        try:
            # Use consistent Qt wrapper from pyqtgraph
            from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication([])
            cb = app.clipboard()
            cb.setText(code_str)
            clipboard_success = True
        except Exception as e1:
            # Fallback to PyQt5 if pyqtgraph wrapper fails
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is None:
                    app = QApplication([])
                cb = app.clipboard()
                cb.setText(code_str)
                clipboard_success = True
            except Exception as e2:
                # Fallback to pyperclip if Qt methods fail
                try:
                    import pyperclip
                    pyperclip.copy(code_str)
                    clipboard_success = True
                except Exception as e3:
                    # All clipboard methods failed
                    print(f"Warning: Could not copy to clipboard. Qt error: {e1}, PyQt5 error: {e2}, pyperclip error: {e3}")
                    clipboard_success = False

        return (code_str, clipboard_success)



    # @param.depends('height','y_location','pen_color','pen_opacity','brush_color','brush_opacity', watch=True)
    # def on_update(self):
    #     """ unused example callback. Just prints the values when any of them change. """
    #     print(f"on_update(name: {self.name}, {self.to_dict()})")


    # def to_plot_config_dict(self):
    #     issue_labels = {'name': 'OccupancyLabels', 'name': 'Occupancy'}
    #     return {'drop_below_threshold': self.dropBelowThreshold, 'opacity': self.barOpacity, 'shape': 'rounded_rect', 'shape_opacity': self.labelsOpacity, 'show_points': self.labelsShowPoints}
    
    # def to_bars_plot_config_dict(self):
    #     return {'name': 'Occupancy', 'drop_below_threshold': self.dropBelowThreshold, 'opacity': self.barOpacity}
    
    # def to_labels_plot_config_dict(self):
    #     return {'name': 'OccupancyLabels', 'shape': 'rounded_rect', 'shape_opacity': self.labelsOpacity, 'show_points': self.labelsShowPoints}
    



def _get_default_epoch_configs() -> Dict[str, EpochDisplayConfig]:
    epochs_update_dict = {
        # 'SessionEpochs': EpochDisplayConfig(brush_color='#00ffff', brush_opacity=0.5, name='SessionEpochs', pen_color='#00ffff', pen_opacity=0.8, height=2.469135802469136, y_location=-12.34567901234568),
        'Laps': EpochDisplayConfig(brush_color='#ff0000', brush_opacity=0.5, name='Laps', pen_color='#ff0000', pen_opacity=0.8, height=4.938271604938272, y_location=-9.876543209876544),	
        'PBEs': EpochDisplayConfig(brush_color='#ffc0cb', brush_opacity=0.5, name='PBEs', pen_color='#ffc0cb', pen_opacity=0.8, height=4.938271604938272, y_location=-4.938271604938272),
        'Ripples': EpochDisplayConfig(brush_color='#6e00f5', brush_opacity=0.5, name='Ripples', pen_color='#6e00f5', pen_opacity=0.8, height=4.938271604938272, y_location=-12.0),
        'Replays': EpochDisplayConfig(brush_color='#ffa500', brush_opacity=0.5, name='Replays', pen_color='#ffa500', pen_opacity=0.8, height=4.938271604938272, y_location=-10.0), 
    }

    return epochs_update_dict
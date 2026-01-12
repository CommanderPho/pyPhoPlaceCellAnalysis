from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
from silx.gui.plot3d.SceneWidget import SceneWidget
from silx.gui.plot3d.SceneWindow import SceneWindow, items as plot3d_items
from silx.gui.plot3d.items.scatter import Scatter2D, Scatter3D
from silx.gui.colors import Colormap
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from skimage.measure import find_contours
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
import neuropy.utils.type_aliases as types
from attrs import define, field
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from neuropy.core.epoch import Epoch, EpochsAccessor, ensure_dataframe, ensure_Epoch, EpochHelpers
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericSilxContainer


@define(slots=False, eq=False, repr=False)
class TextDataProviderDatasource:
    """ simple datasource to show text for each epoch, t_bin 
    """
    a_df: pd.DataFrame = field()
    text_columns: Optional[List[str]] = field(default=None)

    def on_update_epoch_idx(self, epoch_idx: int) -> pd.DataFrame:
        """ return the filtered df for this epoch_idx (df of all time bins in this epoch) """
        return self.a_df[self.a_df['epoch_idx'] == epoch_idx]
    

    def get_text_label(self, epoch_idx: int, t_bin_idx: int) -> Optional[str]:
        """Return formatted text label for a specific epoch and time bin.
        
        Args:
            epoch_idx: Epoch index
            t_bin_idx: Time bin index within the epoch
            
        Returns:
            Formatted text string or None if no match found
        """
        # Filter dataframe by both epoch_idx and t_bin_idx
        matching_rows = self.a_df[
            (self.a_df['epoch_idx'] == epoch_idx) & 
            (self.a_df['t_bin_idx'] == t_bin_idx)
        ]
        
        if len(matching_rows) == 0:
            return None
        
        # Get the first matching row
        row = matching_rows.iloc[0]
        
        # Determine which columns to format
        if self.text_columns is not None and len(self.text_columns) > 0:
            # Use specified columns only
            columns_to_format = [col for col in self.text_columns if col in row.index]
        else:
            # Use all columns except epoch_idx and t_bin_idx
            columns_to_format = [col for col in row.index if col not in ['epoch_idx', 't_bin_idx']]
        
        # Build label from selected columns
        label_parts = []
        for col in columns_to_format:
            value = row[col]
            # Format the value appropriately
            if pd.isna(value):
                label_parts.append(f"{col}: N/A")
            elif isinstance(value, (int, float)):
                label_parts.append(f"{col}: {value:.3f}" if isinstance(value, float) else f"{col}: {value}")
            else:
                label_parts.append(f"{col}: {value}")
        
        # value_join_sep: str = ", "
        value_join_sep: str = "\n"

        return value_join_sep.join(label_parts) if label_parts else None



@metadata_attributes(short_name=None, tags=['Silx', 'gui', '3D', 'volumetric', 'epoch_idx_slider', 'epoch_t_bin_idx_slider', 'two-slider'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-23', related_items=[])
class EpochTimeBinViewer(qt.QWidget):
    """ Silx volumentric widget - renders two sliders, one controlling the `epoch_idx`, 
            and a sub-slider controlling the `t_bin_idx` within that epoch.

    Args:
        decoded_result: Decoded result object with p_x_given_n_list attribute
        xbin_centers: Optional array of X bin center coordinates
        ybin_centers: Optional array of Y bin center coordinates
        locality_measures_df: Optional DataFrame with 'start' and 'stop' columns for matching time bins
        text_columns: Optional list of column names from locality_measures_df to render as text labels

    Usage:

        from pyphoplacecellanalysis.GUI.Silx.EpochTimeBinViewerWidget import EpochTimeBinViewer
        from silx.gui import qt
        from silx.gui.plot import Plot2D
        from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
        from silx.gui.colors import Colormap
        import numpy as np
        from pyphoplacecellanalysis.GUI.Silx.EpochTimeBinViewerWidget import EpochTimeBinViewer

        # Basic usage:
        a_decoder = a_widget.container.pf1D_Decoder_dict['roam']
        viewer = EpochTimeBinViewer(decoded_result=a_widget.decoded_result,
                                        xbin_centers=a_decoder.xbin_centers,
                                        ybin_centers=a_decoder.ybin_centers)
        viewer.show()
        
        # With text labels:
        viewer = EpochTimeBinViewer(decoded_result=a_widget.decoded_result,
                                        xbin_centers=a_decoder.xbin_centers,
                                        ybin_centers=a_decoder.ybin_centers,
                                        locality_measures_df=measures_df,
                                        text_columns=['measure1', 'measure2'])
        viewer.show()

    """
    def __init__(self, decoded_result, xbin_centers=None, ybin_centers=None, locality_measures_df: Optional[pd.DataFrame] = None, text_columns: Optional[List[str]] = None):
        super().__init__()
        self.decoded_result = decoded_result
        self.xbin_centers = xbin_centers
        self.ybin_centers = ybin_centers
        self.locality_measures_df = locality_measures_df
        self.text_columns = text_columns if text_columns is not None else []
        
        # Current indices
        self.curr_epoch_idx = 0
        self.curr_time_bin_idx = 0
        
        # Store wireframe items for cleanup
        self.wireframe_items = []
        
        # Store text label items for cleanup
        self.text_label_items_2d = []
        self.text_label_items_3d = []
        
        # Create UI
        layout = qt.QVBoxLayout()
        self.setLayout(layout)
        
        # Create splitter for 2D and 3D views
        splitter = qt.QSplitter(qt.Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left side: 2D slice view
        left_widget = qt.QWidget()
        left_layout = qt.QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        self.plot_2d = Plot2D()
        left_layout.addWidget(self.plot_2d)
        splitter.addWidget(left_widget)
        
        # Right side: 3D volume view
        self.view_3d = ScalarFieldView()
        splitter.addWidget(self.view_3d)
        
        # Set splitter proportions (50/50)
        splitter.setSizes([500, 500])
        
        # Create sliders
        slider_layout = qt.QHBoxLayout()
        
        # Epoch slider
        self.epoch_slider = qt.QSlider(qt.Qt.Horizontal)
        self.epoch_slider.setMinimum(0)
        self.ui.epoch_slider.setMaximum(len(self.plot_data.decoded_result.p_x_given_n_list) - 1)
        self.epoch_slider.setTickPosition(qt.QSlider.TicksBelow)
        self.epoch_slider.setTickInterval(1)
        self.epoch_slider.valueChanged.connect(self.on_epoch_changed)
        slider_layout.addWidget(qt.QLabel("Epoch:"))
        slider_layout.addWidget(self.epoch_slider)
        slider_layout.addWidget(qt.QLabel("0"))
        self.epoch_label = slider_layout.itemAt(slider_layout.count()-1).widget()
        
        # Time bin slider - make it discrete
        self.time_bin_slider = qt.QSlider(qt.Qt.Horizontal)
        self.time_bin_slider.setTickPosition(qt.QSlider.TicksBelow)
        self.time_bin_slider.setTickInterval(1)  # Tick at every integer value
        self.time_bin_slider.setSingleStep(1)  # Step by 1
        self.time_bin_slider.valueChanged.connect(self.on_time_bin_changed)
        slider_layout.addWidget(qt.QLabel("Time Bin:"))
        slider_layout.addWidget(self.time_bin_slider)
        slider_layout.addWidget(qt.QLabel("0/0"))
        self.time_bin_label = slider_layout.itemAt(slider_layout.count()-1).widget()
        
        layout.addLayout(slider_layout)
        
        # Initialize
        self.update_time_bin_slider_range()
        self.update_views()
    
    @property
    def curr_n_time_bins(self) -> int:
        """Get number of time bins for current epoch"""
        p_x_given_n = self.decoded_result.p_x_given_n_list[self.params.curr_epoch_idx]
        return p_x_given_n.shape[-1]  # Last dimension is time
    
    def _get_time_bin_centers(self) -> Optional[NDArray]:
        """Get time bin centers for current epoch if available"""
        try:
            if hasattr(self.decoded_result, 'time_bin_containers') and self.decoded_result.time_bin_containers is not None:
                return self.decoded_result.time_bin_containers[self.params.curr_epoch_idx].centers
            elif hasattr(self.decoded_result, 'time_window_centers'):
                # Fallback to time_window_centers if available
                return self.decoded_result.time_window_centers
        except (AttributeError, IndexError, KeyError):
            pass
        return None
    
    def _match_time_bin_to_dataframe_row(self, t_bin_idx: int) -> Optional[pd.Series]:
        """Match a time bin index to a dataframe row using start/stop times.
        
        Args:
            t_bin_idx: Time bin index within current epoch
            
        Returns:
            Matching dataframe row (Series) or None if no match found
        """
        if self.locality_measures_df is None or 'start' not in self.locality_measures_df.columns or 'stop' not in self.locality_measures_df.columns:
            return None
        
        time_bin_centers = self._get_time_bin_centers()
        if time_bin_centers is None:
            return None
        
        if t_bin_idx >= len(time_bin_centers):
            return None
        
        # Get time bin center time
        t_bin_time = time_bin_centers[t_bin_idx]
        
        # Find matching row where start <= t_bin_time <= stop
        matching_rows = self.locality_measures_df[
            (self.locality_measures_df['start'] <= t_bin_time) & 
            (t_bin_time <= self.locality_measures_df['stop'])
        ]
        
        if len(matching_rows) > 0:
            # Return first matching row
            return matching_rows.iloc[0]
        return None
    
    def _get_text_label_string(self, t_bin_idx: int) -> Optional[str]:
        """Get text label string for a time bin.
        
        Args:
            t_bin_idx: Time bin index within current epoch
            
        Returns:
            Formatted text string or None if no label available
        """
        row = self._match_time_bin_to_dataframe_row(t_bin_idx)
        if row is None or not self.text_columns:
            return None
        
        # Build label from specified columns
        label_parts = []
        for col in self.text_columns:
            if col in row.index:
                value = row[col]
                # Format the value appropriately
                if pd.isna(value):
                    label_parts.append(f"{col}: N/A")
                elif isinstance(value, (int, float)):
                    label_parts.append(f"{col}: {value:.3f}" if isinstance(value, float) else f"{col}: {value}")
                else:
                    label_parts.append(f"{col}: {value}")
        
        return ", ".join(label_parts) if label_parts else None
    
    def update_time_bin_slider_range(self):
        """Update time_bin slider range when epoch changes"""
        max_time_bins = self.curr_n_time_bins - 1
        self.time_bin_slider.setMaximum(max_time_bins)
        # Reset to 0 when epoch changes
        self.time_bin_slider.setValue(0)
        self.time_bin_slider.setTickInterval(1)  # Tick at every integer value
        self.time_bin_slider.setSingleStep(1)  # Step by 1        
        self.curr_time_bin_idx = 0
        self.update_time_bin_label()
    
    def update_time_bin_label(self):
        """Update the time bin label to show current/max"""
        max_time_bins = self.curr_n_time_bins - 1
        self.time_bin_label.setText(f"{self.curr_time_bin_idx}/{max_time_bins}")
    
    def on_epoch_changed(self, value):
        """Called when epoch slider changes"""
        self.curr_epoch_idx = int(value)
        self.epoch_label.setText(str(self.curr_epoch_idx))
        self.update_time_bin_slider_range()
        self.update_views()
    
    def on_time_bin_changed(self, value):
        """Called when time_bin slider changes"""
        self.curr_time_bin_idx = int(value)
        self.update_time_bin_label()
        self.update_2d_slice()  # Only update 2D view, 3D stays the same
        # Update wireframe colors to highlight current time bin
        self._update_wireframe_colors()
    
    def _create_wireframe_box(self, x_min: float, x_max: float, y_min: float, y_max: float, z_pos: float, color: str = '#FFFFFF40') -> Optional[Any]:
        """Create a wireframe box outline for a single time bin plane.
        
        Args:
            x_min, x_max: X coordinate bounds
            y_min, y_max: Y coordinate bounds
            z_pos: Z position (time bin position)
            color: Color string (e.g., '#FFFFFF40' for semi-transparent white)
            
        Returns:
            Scatter3D item representing the wireframe box, or None if creation fails
        """
        try:
            # Create 4 corners of the rectangle
            corners = np.array([
                [x_min, y_min, z_pos],  # Bottom-left
                [x_max, y_min, z_pos],  # Bottom-right
                [x_max, y_max, z_pos],  # Top-right
                [x_min, y_max, z_pos],  # Top-left
                [x_min, y_min, z_pos],  # Close the loop
            ])
            
            # Extract x, y, z coordinates
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            z_coords = corners[:, 2]
            
            # Create a scatter3d item in line mode to draw the wireframe
            wireframe_item = plot3d_items.Scatter3D()
            # Create dummy values array for setData (required parameter)
            values = np.ones(len(x_coords))
            wireframe_item.setData(x_coords, y_coords, z_coords, values)
            wireframe_item.setVisualization('lines')
            wireframe_item.setLineWidth(1.0)
            # Set color - convert hex to RGBA tuple
            if color.startswith('#'):
                # Parse hex color: #RRGGBBAA or #RRGGBB
                hex_color = color[1:]
                if len(hex_color) == 8:
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    a = int(hex_color[6:8], 16) / 255.0
                elif len(hex_color) == 6:
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    a = 0.25  # Default alpha
                else:
                    r, g, b, a = 1.0, 1.0, 1.0, 0.25
                wireframe_item.setColor((r, g, b, a))
            else:
                wireframe_item.setColor((1.0, 1.0, 1.0, 0.25))  # Default white with transparency
            
            return wireframe_item
        except Exception as e:
            # If Scatter3D doesn't work, return None
            return None
    
    def _get_scene_widget(self):
        """Get the scene widget from ScalarFieldView, trying multiple methods."""
        # Try different ways to access the scene widget
        try:
            if hasattr(self.view_3d, 'getSceneWidget'):
                return self.view_3d.getSceneWidget()
        except:
            pass
        
        try:
            if hasattr(self.view_3d, 'sceneWidget'):
                return self.view_3d.sceneWidget
        except:
            pass
        
        try:
            if hasattr(self.view_3d, '_sceneWidget'):
                return self.view_3d._sceneWidget
        except:
            pass
        
        # Try to find it through Qt widget hierarchy
        # ScalarFieldView might be a window containing a SceneWidget as central widget
        try:
            if hasattr(self.view_3d, 'centralWidget'):
                central = self.view_3d.centralWidget()
                if central and hasattr(central, 'addItem'):
                    return central
        except:
            pass
        
        # Try to find it through children
        try:
            for child in self.view_3d.children():
                # Check if it's a SceneWidget by looking for addItem method
                if hasattr(child, 'addItem') and hasattr(child, 'getItems'):
                    return child
        except:
            pass
        
        # Last resort: try to access through internal structure
        # This is a fallback that might work depending on silx version
        try:
            # Some versions might store it in a private attribute
            for attr_name in dir(self.view_3d):
                if 'scene' in attr_name.lower() and 'widget' in attr_name.lower():
                    attr = getattr(self.view_3d, attr_name, None)
                    if attr and hasattr(attr, 'addItem'):
                        return attr
        except:
            pass
        
        return None
    
    def _add_wireframe_boxes(self, volume_data: NDArray, x_scale: float, y_scale: float, time_scale: float):
        """Add wireframe boxes for each time bin plane.
        
        Args:
            volume_data: 3D array in (time, y, x) format
            x_scale: Scale factor for X dimension
            y_scale: Scale factor for Y dimension
            time_scale: Scale factor for time (Z) dimension
        """
        # Get scene widget
        scene_widget = self._get_scene_widget()
        if scene_widget is None:
            # If we can't access the scene widget, skip adding wireframes
            return
        
        # Remove existing wireframe items
        for item in self.wireframe_items:
            try:
                if hasattr(scene_widget, 'removeItem'):
                    scene_widget.removeItem(item)
            except:
                pass
        self.wireframe_items.clear()
        
        # Get data dimensions
        n_time_bins, n_y_bins, n_x_bins = volume_data.shape
        
        # Calculate bounds - need to account for data positioning
        # The data is positioned starting at (0, 0, 0) and scaled
        if self.xbin_centers is not None and self.ybin_centers is not None:
            # Use actual bin center values
            x_min = float(self.xbin_centers[0])
            x_max = float(self.xbin_centers[-1])
            y_min = float(self.ybin_centers[0])
            y_max = float(self.ybin_centers[-1])
        else:
            # Use bin indices - data spans from 0 to (n_bins - 1) * scale
            # But we need to account for the fact that data is positioned at integer indices
            x_min = 0.0
            x_max = float(n_x_bins - 1) * x_scale if x_scale > 0 else float(n_x_bins - 1)
            y_min = 0.0
            y_max = float(n_y_bins - 1) * y_scale if y_scale > 0 else float(n_y_bins - 1)
        
        # Create wireframe box for each time bin
        for t_bin_idx in range(n_time_bins):
            # Calculate Z position based on time bin index and scale
            z_pos = t_bin_idx * time_scale
            
            # Use different color for current time bin (highlighted)
            if t_bin_idx == self.curr_time_bin_idx:
                color = '#00FF0080'  # Semi-transparent green for current bin
            else:
                color = '#FFFFFF40'  # Semi-transparent white for other bins
            
            # Create wireframe box
            wireframe_item = self._create_wireframe_box(x_min, x_max, y_min, y_max, z_pos, color)
            
            if wireframe_item is not None:
                # Add to scene
                scene_widget.addItem(wireframe_item)
                self.wireframe_items.append(wireframe_item)
    
    def update_views(self):
        """Update both 2D and 3D views"""
        p_x_given_n = self.decoded_result.p_x_given_n_list[self.params.curr_epoch_idx]
        # Shape: (n_x_bins, n_y_bins, n_time_bins)
        
        # Update 3D volume view with all time bins
        # ScalarFieldView expects data in (Z, Y, X) format where Z is the "depth" dimension
        # We want time to be the Z dimension, so transpose to (time, y, x)
        volume_data = np.transpose(p_x_given_n, (2, 1, 0))  # (time, y, x)
        
        # Set data
        self.view_3d.setData(volume_data)
        
        # After setData, you could also add an isosurface:
        self.view_3d.addIsosurface(np.nanmean(volume_data), '#FF000080')


        # Add a cut plane to visualize the volume (shows a slice through the 3D data)
        # This helps see the stack of heatmaps
        try:
            # Get the scene widget to access volume items
            scene_widget = self._get_scene_widget()
            if scene_widget is not None and hasattr(scene_widget, 'getItems'):
                scene_items = scene_widget.getItems()
                for item in scene_items:
                    if hasattr(item, 'getCutPlanes'):
                        cut_planes = item.getCutPlanes()
                        if len(cut_planes) > 0:
                            # Make the cut plane visible and position it
                            cut_plane = cut_planes[0]
                            cut_plane.setVisible(True)
                            # Set normal to view along time axis (0, 0, 1) means viewing XY plane
                            cut_plane.setNormal((0., 0., 1.))
                            cut_plane.moveToCenter()
                            # Set colormap for the cut plane
                            cut_plane.getColormap().setName('viridis')
                            break
        except (AttributeError, IndexError):
            pass
        
        # Set scale if we have bin centers
        time_scale = 10.0
        if self.xbin_centers is not None and self.ybin_centers is not None:
            x_scale = (self.xbin_centers[-1] - self.xbin_centers[0]) / len(self.xbin_centers)
            y_scale = (self.ybin_centers[-1] - self.ybin_centers[0]) / len(self.ybin_centers)
            self.view_3d.setScale(x_scale, y_scale, time_scale)
            self.view_3d.setAxesLabels('X (cm)', 'Y (cm)', 'Time Bin')
        else:
            x_scale = 1.0
            y_scale = 1.0
            self.view_3d.setScale(1.0, 1.0, time_scale)
            self.view_3d.setAxesLabels('X', 'Y', 'Time Bin')
        
        # Add wireframe boxes for each time bin plane
        self._add_wireframe_boxes(volume_data, x_scale, y_scale, time_scale)
        
        # Add text labels to 3D view
        self._add_text_labels_3d(volume_data, x_scale, y_scale, time_scale)
        
        # Update 2D slice view
        self.update_2d_slice()
    
    def _update_wireframe_colors(self):
        """Update wireframe box colors to highlight the current time bin"""
        if not self.wireframe_items:
            return
        
        try:
            scene_widget = self._get_scene_widget()
            if scene_widget is None:
                return
            
            # Get current scale to calculate positions
            p_x_given_n = self.decoded_result.p_x_given_n_list[self.params.curr_epoch_idx]
            volume_data = np.transpose(p_x_given_n, (2, 1, 0))
            n_time_bins = volume_data.shape[0]
            
            time_scale = 10.0
            if self.xbin_centers is not None and self.ybin_centers is not None:
                x_scale = (self.xbin_centers[-1] - self.xbin_centers[0]) / len(self.xbin_centers)
                y_scale = (self.ybin_centers[-1] - self.ybin_centers[0]) / len(self.ybin_centers)
            else:
                x_scale = 1.0
                y_scale = 1.0
            
            # Update colors for each wireframe item
            for t_bin_idx, wireframe_item in enumerate(self.wireframe_items):
                if t_bin_idx == self.curr_time_bin_idx:
                    # Highlight current time bin with green
                    wireframe_item.setColor((0.0, 1.0, 0.0, 0.5))  # Green, semi-transparent
                else:
                    # Other bins in white
                    wireframe_item.setColor((1.0, 1.0, 1.0, 0.25))  # White, semi-transparent
        except Exception:
            pass  # Silently fail if update isn't possible
    
    def _add_text_labels_2d(self):
        """Add text labels to 2D plot for current time bin"""
        # Remove existing text labels
        for item in self.text_label_items_2d:
            try:
                self.plot_2d.removeItem(item)
            except:
                pass
        self.text_label_items_2d.clear()
        
        if not self.text_columns:
            return
        
        # Get text label for current time bin
        label_text = self._get_text_label_string(self.curr_time_bin_idx)
        if label_text is None:
            return
        
        # Calculate position: center of plot, just below the image
        if self.xbin_centers is not None and self.ybin_centers is not None:
            x_pos = (self.xbin_centers[0] + self.xbin_centers[-1]) / 2.0
            y_pos = self.ybin_centers[0] - (self.ybin_centers[-1] - self.ybin_centers[0]) * 0.05  # 5% below
        else:
            # Use default positioning
            x_pos = 0.5
            y_pos = -0.05
        
        # Add text annotation to 2D plot
        try:
            text_item = self.plot_2d.addText(label_text, x_pos, y_pos, color='white', fontsize=10)
            self.text_label_items_2d.append(text_item)
        except Exception:
            # If addText doesn't work, try alternative method
            try:
                # Some versions might use different API
                if hasattr(self.plot_2d, 'addAnnotation'):
                    text_item = self.plot_2d.addAnnotation(label_text, x_pos, y_pos)
                    self.text_label_items_2d.append(text_item)
            except Exception:
                pass  # Silently fail if text rendering isn't available
    
    def _add_text_labels_3d(self, volume_data: NDArray, x_scale: float, y_scale: float, time_scale: float):
        """Add text labels to 3D scene for all time bins"""
        scene_widget = self._get_scene_widget()
        if scene_widget is None or not self.text_columns:
            return
        
        # Remove existing text labels
        for item in self.text_label_items_3d:
            try:
                if hasattr(scene_widget, 'removeItem'):
                    scene_widget.removeItem(item)
            except:
                pass
        self.text_label_items_3d.clear()
        
        n_time_bins = volume_data.shape[0]
        
        # Calculate bounds for positioning
        if self.xbin_centers is not None and self.ybin_centers is not None:
            x_min = float(self.xbin_centers[0])
            x_max = float(self.xbin_centers[-1])
            y_min = float(self.ybin_centers[0])
            y_max = float(self.ybin_centers[-1])
        else:
            n_x_bins = volume_data.shape[2]
            n_y_bins = volume_data.shape[1]
            x_min = 0.0
            x_max = float(n_x_bins - 1) * x_scale if x_scale > 0 else float(n_x_bins - 1)
            y_min = 0.0
            y_max = float(n_y_bins - 1) * y_scale if y_scale > 0 else float(n_y_bins - 1)
        
        # Add text label for each time bin
        for t_bin_idx in range(n_time_bins):
            label_text = self._get_text_label_string(t_bin_idx)
            if label_text is None:
                continue
            
            # Position text below the time bin plane
            x_pos = (x_min + x_max) / 2.0
            y_pos = y_min - (y_max - y_min) * 0.1  # 10% below
            z_pos = t_bin_idx * time_scale
            
            # Try to add text using Scatter3D with a single point and text visualization
            try:
                # Create a text item using Scatter3D positioned at the label location
                text_item = plot3d_items.Scatter3D()
                # Use a single point to position the text
                text_item.setData(np.array([x_pos]), np.array([y_pos]), np.array([z_pos]), np.array([1.0]))
                text_item.setVisualization('points')
                text_item.setPointSize(0)  # Make point invisible
                # Note: silx may not directly support text labels in 3D, so this is a placeholder
                # The actual text rendering might need to be done differently depending on silx version
                scene_widget.addItem(text_item)
                self.text_label_items_3d.append(text_item)
            except Exception:
                # If direct text rendering isn't available, we'll skip it
                pass
    
    def update_2d_slice(self):
        """Update the 2D slice view for current time bin"""
        p_x_given_n = self.decoded_result.p_x_given_n_list[self.params.curr_epoch_idx]
        # Extract 2D slice: (n_x_bins, n_y_bins, n_time_bins) -> (n_x_bins, n_y_bins)
        slice_2d = p_x_given_n[:, :, self.curr_time_bin_idx]
        
        # Update 2D plot
        self.plot_2d.addImage(slice_2d, legend='p_x_given_n', replace=True, 
                             colormap=Colormap(name="viridis"))
        
        if self.xbin_centers is not None and self.ybin_centers is not None:
            self.plot_2d.getXAxis().setLabel("X (cm)")
            self.plot_2d.getYAxis().setLabel("Y (cm)")
            # Set extent based on bin centers
            xmin, xmax = self.xbin_centers[0], self.xbin_centers[-1]
            ymin, ymax = self.ybin_centers[0], self.ybin_centers[-1]
            self.plot_2d.getXAxis().setLimits(xmin, xmax)
            self.plot_2d.getYAxis().setLimits(ymin, ymax)
        
        # Add text labels
        self._add_text_labels_2d()




@metadata_attributes(short_name=None, tags=['Silx', 'gui', '3D', 'scene', 'epoch_idx_slider', 'height-map'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-23', related_items=[])
@define(slots=False, eq=False)
class Epoch3DSceneTimeBinViewer(GenericSilxContainer, qt.QWidget):
    """ Silx 3D scene widget - displays all time bins from the active epoch as adjacent 3D height map surfaces.
    
    This widget uses a SceneWindow to display multiple time bins from the selected epoch
    as 3D height map surfaces arranged horizontally. Each time bin is displayed as a 
    2D scatter plot with height map enabled, positioned side-by-side along the X axis.
    
    Args:
        decoded_result: Decoded result object with p_x_given_n_list attribute
        xbin_centers: Optional array of X bin center coordinates
        ybin_centers: Optional array of Y bin center coordinates
        locality_measures_df: Optional DataFrame with 'start' and 'stop' columns for matching time bins
        text_columns: Optional list of column names from locality_measures_df to render as text labels
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.Silx.EpochTimeBinViewerWidget import Epoch3DSceneTimeBinViewer
        
        # Basic usage:
        a_decoder = a_widget.container.pf1D_Decoder_dict['roam']
        viewer = Epoch3DSceneTimeBinViewer(decoded_result=a_widget.decoded_result,
                                           xbin_centers=a_decoder.xbin_centers,
                                           ybin_centers=a_decoder.ybin_centers)
        viewer.show()
        
        # With text labels:
        viewer = Epoch3DSceneTimeBinViewer(decoded_result=a_widget.decoded_result,
                                           xbin_centers=a_decoder.xbin_centers,
                                           ybin_centers=a_decoder.ybin_centers,
                                           locality_measures_df=measures_df,
                                           text_columns=['measure1', 'measure2'])
        viewer.show()
    
    """

    sigEpochIndexChanged = QtCore.pyqtSignal(int)
    # sigTimeBinIndexChanged = QtCore.pyqtSignal(int)

    @property
    def curr_n_time_bins(self) -> int:
        """Get number of time bins for current epoch"""
        p_x_given_n = self.plots_data.decoded_result.p_x_given_n_list[self.params.curr_epoch_idx]
        return p_x_given_n.shape[-1]  # Last dimension is time


    def __init__(self, decoded_result, xbin_centers=None, ybin_centers=None, locality_measures_df: Optional[pd.DataFrame] = None, text_columns: Optional[List[str]] = None, text_data_provider: Optional[TextDataProviderDatasource] = None, **kwargs):
        # Extract attrs field names from GenericSilxContainer
        attrs_field_names = {'name', 'plots', 'plot_data', 'ui', 'params'}
        attrs_kwargs = {k: v for k, v in kwargs.items() if k in attrs_field_names}
        qt_kwargs = {k: v for k, v in kwargs.items() if k not in attrs_field_names}
        
        # Initialize attrs class first
        GenericSilxContainer.__init__(self, **attrs_kwargs)
        
        # Initialize Qt widget
        qt.QWidget.__init__(self, **qt_kwargs)
        
        self.plots_data.decoded_result = decoded_result
        self.plots_data.xbin_centers = xbin_centers
        self.plots_data.ybin_centers = ybin_centers
        self.plots_data.locality_measures_df = locality_measures_df
        self.params.text_columns = text_columns if text_columns is not None else []
        self.plots_data.text_data_provider = text_data_provider
        
        self.params.spacing_factor = 1.2  # 20% spacing between bins
        self.plots_data.translation_triple_list = []

        # Detect point-like data mode (when 't' column exists in locality_measures_df)
        self.params.is_point_like_mode = (self.plots_data.locality_measures_df is not None and 't' in self.plots_data.locality_measures_df.columns)
        self.params.use_groupItem = True
        self.plots.time_bin_groupItems = [] ## initialize to empty
        
        # Current epoch index
        self.params.curr_epoch_idx = 0
        
        # Store time bin items for cleanup
        self.time_bin_items = []
        
        # Store text label items for cleanup (Qt QLabel widgets)
        self.text_label_items = []

        # Peak-contour overlays (Silx-specific)
        self.plots_data.peak_prominence_result = None
        # self.plots.peak_contour_items: List[Any] = []
        self.plots.peak_contour_items = []
        
        # Store table widget for locality measures (point-like mode)
        self.locality_measures_table = None

        self._init_UI()
        
        self._init_Graphics()

        # Initialize
        self.on_epoch_changed(0)


    def _init_UI(self):
        # Create UI
        layout = qt.QVBoxLayout()
        self.setLayout(layout)
        
        # Create SceneWindow for 3D visualization
        self.plots.scene_window = SceneWindow()
        
        # Add SceneWindow to layout
        layout.addWidget(self.plots.scene_window)
        
        # Store label data for repositioning on resize
        self._label_data = []  # List of (text, t_bin_idx, x_translation, x_min, x_max, y_min, y_max, bin_spacing) tuples
        
        # Install event filter to catch resize events
        self.plots.scene_window.installEventFilter(self)
        
        # Create epoch slider
        self.ui.slider_container_layout = qt.QHBoxLayout()
        
        self.ui.epoch_slider = qt.QSlider(qt.Qt.Horizontal)
        self.ui.epoch_slider.setMinimum(0)
        self.ui.epoch_slider.setMaximum(len(self.plots_data.decoded_result.p_x_given_n_list) - 1)
        self.ui.epoch_slider.setTickPosition(qt.QSlider.TicksBelow)
        self.ui.epoch_slider.setTickInterval(1)
        self.ui.epoch_slider.valueChanged.connect(self.on_epoch_changed)
        self.ui.slider_container_layout.addWidget(qt.QLabel("Epoch:"))
        self.ui.slider_container_layout.addWidget(self.ui.epoch_slider)
        self.ui.slider_container_layout.addWidget(qt.QLabel("0"))
        self.epoch_label = self.ui.slider_container_layout.itemAt(self.ui.slider_container_layout.count()-1).widget()
        
        layout.addLayout(self.ui.slider_container_layout)

        # Create table tab for point-like mode (delayed to ensure sidebar is available)
        if self.params.is_point_like_mode:
            # Use QTimer to delay table creation until after window is shown
            qt.QTimer.singleShot(100, self._create_locality_measures_table_tab)



    def _init_Graphics(self):
        """ build graphics items 
        """
        self.plots.scene_widget = self.plots.scene_window.getSceneWidget()

        self.plots.scene_widget.setBackgroundColor((0.8, 0.8, 0.8, 1.))
        self.plots.scene_widget.setForegroundColor((1., 1., 1., 1.))
        self.plots.scene_widget.setTextColor((0.1, 0.1, 0.1, 1.))


    @property
    def scene_window(self) -> SceneWindow:
        """The scene_window property."""
        return self.plots.scene_window
    
    @property
    def scene_widget(self) -> SceneWidget:
        """The scene_widget property."""
        return self.plots.scene_widget


    def _get_epoch_time_bin_shape(self, epoch_idx: int) -> Tuple[int, int, int]:
        """Return (n_x_bins, n_y_bins, n_time_bins) for the given epoch."""
        p_x_given_n = self.plots_data.decoded_result.p_x_given_n_list[epoch_idx]
        return p_x_given_n.shape




    def _get_sidebar_tab_widget(self) -> Optional[qt.QTabWidget]:
        """Access the SceneWindow sidebar QTabWidget.
        
        Based on silx documentation, the sidebar can be accessed through the plot3d widget.
        
        Returns:
            QTabWidget if found, None otherwise
        """
        # Helper function to recursively find QTabWidget
        def find_tab_widget(widget):
            if widget is None:
                return None
            if isinstance(widget, qt.QTabWidget):
                return widget
            # Check all children recursively
            for child in widget.children():
                if isinstance(child, qt.QWidget):
                    result = find_tab_widget(child)
                    if result is not None:
                        return result
            return None
        
        # Method 1: Try getPlot3DWidget() if available (as per silx docs)
        try:
            if hasattr(self.plots.scene_window, 'getPlot3DWidget'):
                plot3d_widget = self.plots.scene_window.getPlot3DWidget()
                if plot3d_widget:
                    # Try findChild to locate the sidebar QTabWidget
                    sidebar = plot3d_widget.findChild(qt.QTabWidget)
                    if sidebar is not None:
                        print("DEBUG: Found sidebar via getPlot3DWidget().findChild()")
                        return sidebar
                    # Also try recursive search
                    result = find_tab_widget(plot3d_widget)
                    if result is not None:
                        print("DEBUG: Found sidebar via getPlot3DWidget() recursive search")
                        return result
        except Exception as e:
            print(f"DEBUG: getPlot3DWidget() failed: {e}")
            pass
        
        # Method 2: Try direct access if available
        try:
            if hasattr(self.plots.scene_window, 'getSidebar'):
                sidebar = self.plots.scene_window.getSidebar()
                if isinstance(sidebar, qt.QTabWidget):
                    print("DEBUG: Found sidebar via getSidebar()")
                    return sidebar
        except Exception as e:
            print(f"DEBUG: getSidebar() failed: {e}")
            pass
        
        # Method 3: Use findChild directly on SceneWindow
        try:
            sidebar = self.plots.scene_window.findChild(qt.QTabWidget)
            if sidebar is not None:
                # Verify it's the right one by checking tab names
                tab_count = sidebar.count()
                if tab_count >= 1:
                    print(f"DEBUG: Found sidebar via findChild() with {tab_count} tabs")
                    return sidebar
        except Exception as e:
            print(f"DEBUG: findChild() failed: {e}")
            pass
        
        # Method 4: Traverse children to find QTabWidget
        try:
            result = find_tab_widget(self.plots.scene_window)
            if result is not None:
                tab_count = result.count()
                # Check tab names to confirm this is the sidebar
                has_object_params = False
                has_global_params = False
                for i in range(tab_count):
                    tab_text = result.tabText(i)
                    if "Object parameters" in tab_text or "Object" in tab_text:
                        has_object_params = True
                    if "Global parameters" in tab_text or "Global" in tab_text:
                        has_global_params = True
                if (has_object_params or has_global_params) and tab_count >= 1:
                    print(f"DEBUG: Found sidebar via recursive search with {tab_count} tabs")
                    return result
        except Exception as e:
            print(f"DEBUG: Recursive search failed: {e}")
            pass
        
        # Method 5: Try accessing through centralWidget
        try:
            if hasattr(self.plots.scene_window, 'centralWidget'):
                central = self.plots.scene_window.centralWidget()
                if central:
                    sidebar = central.findChild(qt.QTabWidget)
                    if sidebar is not None:
                        print("DEBUG: Found sidebar via centralWidget().findChild()")
                        return sidebar
                    result = find_tab_widget(central)
                    if result is not None:
                        print("DEBUG: Found sidebar via centralWidget() recursive search")
                        return result
        except Exception as e:
            print(f"DEBUG: centralWidget() search failed: {e}")
            pass
        
        # Method 6: Search through all widgets in the scene window
        try:
            all_widgets = []
            def collect_widgets(widget):
                if isinstance(widget, qt.QWidget):
                    all_widgets.append(widget)
                    for child in widget.children():
                        if isinstance(child, qt.QWidget):
                            collect_widgets(child)
            
            collect_widgets(self.plots.scene_window)
            for widget in all_widgets:
                if isinstance(widget, qt.QTabWidget):
                    tab_count = widget.count()
                    # Check tab names to confirm this is the sidebar
                    has_object_params = False
                    has_global_params = False
                    for i in range(tab_count):
                        tab_text = widget.tabText(i)
                        if "Object parameters" in tab_text or "Object" in tab_text:
                            has_object_params = True
                        if "Global parameters" in tab_text or "Global" in tab_text:
                            has_global_params = True
                    if (has_object_params or has_global_params) and tab_count >= 1:
                        print(f"DEBUG: Found sidebar via comprehensive search with {tab_count} tabs")
                        return widget
        except Exception as e:
            print(f"DEBUG: Comprehensive search failed: {e}")
            pass
        
        print("DEBUG: Could not find sidebar QTabWidget using any method")
        return None
    
    def _create_locality_measures_table_tab(self):
        """Create and add a table tab to the sidebar displaying locality_measures_df."""
        if not self.params.is_point_like_mode or self.plots_data.locality_measures_df is None:
            print(f"DEBUG: Skipping table creation - is_point_like_mode={self.params.is_point_like_mode}, df is None={self.plots_data.locality_measures_df is None}")
            return
        
        # Get the sidebar tab widget
        sidebar_tabs = self._get_sidebar_tab_widget()
        if sidebar_tabs is None:
            # If we can't find the sidebar, try again after a longer delay
            # This can happen if the SceneWindow hasn't fully initialized yet
            # But limit retries to avoid infinite loops
            if not hasattr(self, '_table_creation_retry_count'):
                self._table_creation_retry_count = 0
            self._table_creation_retry_count += 1
            if self._table_creation_retry_count < 5:  # Try up to 5 times
                print(f"DEBUG: Sidebar not found, retrying ({self._table_creation_retry_count}/5)...")
                qt.QTimer.singleShot(500, self._create_locality_measures_table_tab)
            else:
                # Sidebar not found after retries, create dock widget instead
                print("DEBUG: Sidebar not found after retries, creating dock widget instead")
                self._create_locality_measures_dock_widget()
            return
        
        # Reset retry count on success
        if hasattr(self, '_table_creation_retry_count'):
            self._table_creation_retry_count = 0
        
        print(f"DEBUG: Found sidebar with {sidebar_tabs.count()} tabs")
        
        # Create QTableWidget
        table_widget = qt.QTableWidget()
        self.locality_measures_table = table_widget
        
        # Filter dataframe by current epoch time range
        df = self._get_filtered_dataframe_for_current_epoch()
        n_rows, n_cols = df.shape
        table_widget.setRowCount(n_rows)
        table_widget.setColumnCount(n_cols)
        table_widget.setHorizontalHeaderLabels([str(col) for col in df.columns])
        
        # Populate cells
        for row_idx in range(n_rows):
            for col_idx, col_name in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                # Format value appropriately
                if pd.isna(value):
                    item_text = "N/A"
                elif isinstance(value, (int, float)):
                    item_text = f"{value:.6f}" if isinstance(value, float) else str(value)
                else:
                    item_text = str(value)
                
                item = qt.QTableWidgetItem(item_text)
                # Make read-only
                try:
                    item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                except AttributeError:
                    # Fallback: use integer value
                    item.setFlags(item.flags() & ~0x00000002)  # ItemIsEditable flag
                table_widget.setItem(row_idx, col_idx, item)
        
        # Set table properties
        table_widget.setAlternatingRowColors(True)
        # Set selection behavior and mode using Qt enums
        try:
            table_widget.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
            table_widget.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        except AttributeError:
            # Fallback if enum access doesn't work
            table_widget.setSelectionBehavior(1)  # SelectRows
            table_widget.setSelectionMode(1)  # SingleSelection
        table_widget.setSortingEnabled(True)
        
        # Resize columns to fit content
        table_widget.resizeColumnsToContents()
        
        # Create a container widget with layout for the table
        container = qt.QWidget()
        container_layout = qt.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(table_widget)
        container.setLayout(container_layout)
        
        # Add tab to sidebar (insert before last tab to keep "Global parameters" last)
        tab_index = sidebar_tabs.count() - 1 if sidebar_tabs.count() > 0 else 0
        sidebar_tabs.insertTab(tab_index, container, "Locality Measures")
        print(f"DEBUG: Added 'Locality Measures' tab at index {tab_index}. Total tabs: {sidebar_tabs.count()}")
    

    def _create_locality_measures_dock_widget(self):
        """Create a dock widget containing the locality measures table."""
        if not self.params.is_point_like_mode or self.plots_data.locality_measures_df is None:
            return
        
        # Create QTableWidget
        table_widget = qt.QTableWidget()
        self.locality_measures_table = table_widget
        
        # Filter dataframe by current epoch time range
        df = self._get_filtered_dataframe_for_current_epoch()
        n_rows, n_cols = df.shape
        table_widget.setRowCount(n_rows)
        table_widget.setColumnCount(n_cols)
        table_widget.setHorizontalHeaderLabels([str(col) for col in df.columns])
        
        # Populate cells
        for row_idx in range(n_rows):
            for col_idx, col_name in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                # Format value appropriately
                if pd.isna(value):
                    item_text = "N/A"
                elif isinstance(value, (int, float)):
                    item_text = f"{value:.6f}" if isinstance(value, float) else str(value)
                else:
                    item_text = str(value)
                
                item = qt.QTableWidgetItem(item_text)
                # Make read-only
                try:
                    flags = item.flags()
                    item.setFlags(flags & ~qt.Qt.ItemIsEditable)
                except (AttributeError, TypeError):
                    # Fallback: use integer value for ItemIsEditable flag
                    flags = item.flags()
                    item.setFlags(flags & ~0x00000002)  # ItemIsEditable flag
                table_widget.setItem(row_idx, col_idx, item)
        
        # Set table properties
        table_widget.setAlternatingRowColors(True)
        # Set selection behavior and mode using Qt enums
        try:
            table_widget.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
            table_widget.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        except AttributeError:
            # Fallback if enum access doesn't work
            table_widget.setSelectionBehavior(1)  # SelectRows
            table_widget.setSelectionMode(1)  # SingleSelection
        table_widget.setSortingEnabled(True)
        
        # Resize columns to fit content
        table_widget.resizeColumnsToContents()
        
        # Create dock widget
        dock_widget = qt.QDockWidget("Locality Measures", self)
        dock_widget.setWidget(table_widget)
        dock_widget.setAllowedAreas(qt.Qt.RightDockWidgetArea | qt.Qt.LeftDockWidgetArea)
        
        # Check if self is a QMainWindow or has addDockWidget method
        if hasattr(self, 'addDockWidget'):
            self.addDockWidget(qt.Qt.RightDockWidgetArea, dock_widget)
            print("DEBUG: Created dock widget for Locality Measures table")
        else:
            # If not a QMainWindow, try to find parent that is
            parent = self.parent()
            while parent:
                if hasattr(parent, 'addDockWidget'):
                    parent.addDockWidget(qt.Qt.RightDockWidgetArea, dock_widget)
                    print("DEBUG: Created dock widget for Locality Measures table in parent window")
                    break
                parent = parent.parent() if hasattr(parent, 'parent') else None
            else:
                # Last resort: just show as a separate window
                dock_widget.setFloating(True)
                dock_widget.show()
                print("DEBUG: Created floating dock widget for Locality Measures table")
    
    def _highlight_matching_row_in_table(self, t_bin_idx: int):
        """Highlight and scroll to the row in the table matching the current time bin.
        
        Args:
            t_bin_idx: Time bin index within current epoch
        """
        if self.locality_measures_table is None or not self.params.is_point_like_mode:
            return
        
        # Find matching row using the same logic as _match_time_bin_to_dataframe_row
        time_bin_centers = self._get_time_bin_centers()
        if time_bin_centers is None:
            return
        
        if t_bin_idx >= len(time_bin_centers):
            return
        
        # Get time bin center time
        t_bin_time = time_bin_centers[t_bin_idx]
        
        # Find matching row in dataframe
        if 't' in self.plots_data.locality_measures_df.columns:
            time_diffs = (self.plots_data.locality_measures_df['t'] - t_bin_time).abs()
            closest_idx = time_diffs.idxmin()
            closest_time_diff = time_diffs.loc[closest_idx]
            
            # Check if closest match is within tolerance
            atol = 0.100  # 100 millisecond tolerance
            if closest_time_diff <= atol:
                # Find the row index in the dataframe
                # Get the position of closest_idx in the dataframe index
                df_index_pos = self.plots_data.locality_measures_df.index.get_loc(closest_idx)
                # If get_loc returns a slice or boolean array, get the first position
                if isinstance(df_index_pos, (slice, np.ndarray)):
                    if isinstance(df_index_pos, slice):
                        df_index_pos = df_index_pos.start if df_index_pos.start is not None else 0
                    else:
                        df_index_pos = int(np.where(df_index_pos)[0][0]) if np.any(df_index_pos) else 0
                else:
                    df_index_pos = int(df_index_pos)
                
                # Clear previous highlights first
                for row in range(self.locality_measures_table.rowCount()):
                    for col in range(self.locality_measures_table.columnCount()):
                        item = self.locality_measures_table.item(row, col)
                        if item:
                            item.setBackground(qt.QBrush())  # Reset to default
                
                # Clear previous selection
                self.locality_measures_table.clearSelection()
                
                # Select and highlight the row
                self.locality_measures_table.selectRow(df_index_pos)
                
                # Scroll to make the row visible
                first_item = self.locality_measures_table.item(df_index_pos, 0)
                if first_item:
                    # Scroll to item using Qt enum
                    try:
                        self.locality_measures_table.scrollToItem(
                            first_item,
                            qt.QAbstractItemView.EnsureVisible
                        )
                    except AttributeError:
                        # Fallback if enum access doesn't work
                        self.locality_measures_table.scrollToItem(first_item, 1)
                
                # Set background color for highlighting
                for col_idx in range(self.locality_measures_table.columnCount()):
                    item = self.locality_measures_table.item(df_index_pos, col_idx)
                    if item:
                        item.setBackground(qt.QColor(255, 255, 0, 100))  # Yellow highlight with transparency
            else:
                # No match within tolerance, clear selection
                self.locality_measures_table.clearSelection()
                # Clear any previous highlights
                for row in range(self.locality_measures_table.rowCount()):
                    for col in range(self.locality_measures_table.columnCount()):
                        item = self.locality_measures_table.item(row, col)
                        if item:
                            item.setBackground(qt.QBrush())  # Reset to default
    

    
    def _get_time_bin_centers(self) -> Optional[NDArray]:
        """Get time bin centers for current epoch if available"""
        try:
            if hasattr(self.plots_data.decoded_result, 'time_bin_containers') and self.plots_data.decoded_result.time_bin_containers is not None:
                return self.plots_data.decoded_result.time_bin_containers[self.params.curr_epoch_idx].centers
            elif hasattr(self.plots_data.decoded_result, 'time_window_centers'):
                # Fallback to time_window_centers if available
                return self.plots_data.decoded_result.time_window_centers
        except (AttributeError, IndexError, KeyError):
            pass
        return None
    
    def _get_epoch_time_range(self) -> Optional[Tuple[float, float]]:
        """Get the time range (earliest start, latest stop) for the current epoch.
        
        Returns:
            Tuple of (start_time, stop_time) or None if not available
        """
        try:
            if hasattr(self.plots_data.decoded_result, 'time_bin_containers') and self.plots_data.decoded_result.time_bin_containers is not None:
                time_bin_container = self.plots_data.decoded_result.time_bin_containers[self.params.curr_epoch_idx]
                print(f'time_bin_container: {time_bin_container}')
                # Check if container has start/stop attributes
                if hasattr(time_bin_container, 'start') and hasattr(time_bin_container, 'stop'):
                    # If it's a single value, return it
                    if hasattr(time_bin_container.start, '__len__') and len(time_bin_container.start) > 0:
                        start = float(np.nanmin(time_bin_container.start))
                        stop = float(np.nanmax(time_bin_container.stop))
                    else:
                        start = float(time_bin_container.start)
                        stop = float(time_bin_container.stop)
                    return (start, stop)
                # If not, try to get from centers (use first and last with some margin)
                elif hasattr(time_bin_container, 'centers'):
                    centers = time_bin_container.centers
                    print(f'\tlen(centers): {len(centers)}')
                    if len(centers) > 0:
                        # Estimate range from centers (assuming bins are evenly spaced)
                        if len(centers) > 1:
                            bin_width = centers[1] - centers[0]
                            start = float(centers[0] - bin_width / 2)
                            stop = float(centers[-1] + bin_width / 2)
                        else:
                            # Single bin, use a small range around it
                            start = float(centers[0] - 0.1)
                            stop = float(centers[0] + 0.1)
                        return (start, stop)
        except (AttributeError, IndexError, KeyError, TypeError) as e:
            pass
        
        # Fallback: try to get from time_window_centers
        try:
            if hasattr(self.plots_data.decoded_result, 'time_window_centers'):
                centers = self.plots_data.decoded_result.time_window_centers
                if len(centers) > 0:
                    if len(centers) > 1:
                        bin_width = centers[1] - centers[0]
                        start = float(centers[0] - bin_width / 2)
                        stop = float(centers[-1] + bin_width / 2)
                    else:
                        start = float(centers[0] - 0.1)
                        stop = float(centers[0] + 0.1)
                    return (start, stop)
        except (AttributeError, TypeError):
            pass
        
        
        return None
    

    def _get_filtered_dataframe_for_current_epoch(self) -> pd.DataFrame:
        """Filter locality_measures_df to show only rows within the current epoch's time range.
        
        Returns:
            Filtered DataFrame containing only rows where 't' is between epoch start and stop
        """
        if self.plots_data.locality_measures_df is None or not self.params.is_point_like_mode:
            return pd.DataFrame()
        
        # Get epoch time range
        time_range = self._get_epoch_time_range()
        print(f'time_range: {time_range}')
        if time_range is None:
            # If we can't get time range, return full dataframe
            return self.plots_data.locality_measures_df
        
        start_time, stop_time = time_range
        
        # Filter dataframe: keep rows where 't' is between start_time and stop_time
        if 't' in self.plots_data.locality_measures_df.columns:
            filtered_df = self.plots_data.locality_measures_df[
                (self.plots_data.locality_measures_df['t'] >= start_time) & 
                (self.plots_data.locality_measures_df['t'] <= stop_time)
            ].copy()
            return filtered_df
        
        # If no 't' column, return full dataframe
        return self.plots_data.locality_measures_df
    

    def _update_table_for_current_epoch(self):
        """Update the table to show only rows for the current epoch."""
        if self.locality_measures_table is None or not self.params.is_point_like_mode:
            return
        
        # Get filtered dataframe
        df = self._get_filtered_dataframe_for_current_epoch()
        
        # Clear existing table
        self.locality_measures_table.setRowCount(0)
        
        # Repopulate with filtered data
        n_rows, n_cols = df.shape
        self.locality_measures_table.setRowCount(n_rows)
        self.locality_measures_table.setColumnCount(n_cols)
        self.locality_measures_table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        
        # Populate cells
        for row_idx in range(n_rows):
            for col_idx, col_name in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                # Format value appropriately
                if pd.isna(value):
                    item_text = "N/A"
                elif isinstance(value, (int, float)):
                    item_text = f"{value:.6f}" if isinstance(value, float) else str(value)
                else:
                    item_text = str(value)
                
                item = qt.QTableWidgetItem(item_text)
                # Make read-only
                try:
                    flags = item.flags()
                    item.setFlags(flags & ~qt.Qt.ItemIsEditable)
                except (AttributeError, TypeError):
                    # Fallback: use integer value for ItemIsEditable flag
                    flags = item.flags()
                    item.setFlags(flags & ~0x00000002)  # ItemIsEditable flag
                self.locality_measures_table.setItem(row_idx, col_idx, item)
        
        # Resize columns to fit content
        self.locality_measures_table.resizeColumnsToContents()
    

    def _match_time_bin_to_dataframe_row(self, t_bin_idx: int) -> Optional[pd.Series]:
        """Match a time bin index to a dataframe row using start/stop times.
        
        Args:
            t_bin_idx: Time bin index within current epoch
            
        Returns:
            Matching dataframe row (Series) or None if no match found
        """
        if (self.plots_data.locality_measures_df is None):
            return None

        # if self.plot_data.locality_measures_df is None or ('start' not in self.plot_data.locality_measures_df.columns) or ('stop' not in self.plot_data.locality_measures_df.columns):
        #     return None
        
        time_bin_centers = self._get_time_bin_centers()
        if time_bin_centers is None:
            return None
        
        if t_bin_idx >= len(time_bin_centers):
            return None
        
        # Get time bin center time
        t_bin_time = time_bin_centers[t_bin_idx]
        
        # Find matching row where start <= t_bin_time <= stop
        if ('stop' in self.plots_data.locality_measures_df) and ('stop' in self.plots_data.locality_measures_df):
            ## epoch-style updates
            matching_rows = self.plots_data.locality_measures_df[
                (self.plots_data.locality_measures_df['start'] <= t_bin_time) & 
                (t_bin_time <= self.plots_data.locality_measures_df['stop'])
            ]
        elif ('t' in self.plots_data.locality_measures_df):
            ## point/position-style updates: find closest match within tolerance
            print(f'debug: point-style data: t_bin_time: {t_bin_time}')
            # Find the index with the closest time value
            time_diffs = (self.plots_data.locality_measures_df['t'] - t_bin_time).abs()
            closest_idx = time_diffs.idxmin()
            closest_time_diff = time_diffs.loc[closest_idx]
            print(f'\tclosest_idx: {closest_idx}, closest_time_diff: {closest_time_diff}, self.plot_data.locality_measures_df.loc[[closest_idx]]: {self.plots_data.locality_measures_df.loc[[closest_idx]]}')
            # Check if closest match is within tolerance (default 1ms)
            atol = 0.100  # 100 millisecond tolerance, kinda insanely high
            if closest_time_diff <= atol:
                matching_rows = self.plots_data.locality_measures_df.loc[[closest_idx]]
            else:
                # No match within tolerance
                matching_rows = pd.DataFrame()  # Empty DataFrame

        else:
            print(f'WARN: attempting to get time_bin_to_dataframe_row text failed because self.plot_data.locality_measures_df does not look epoch-like or point-like: self.plot_data.locality_measures_df.columns: {list(self.plots_data.locality_measures_df.columns)}')
            return None
        
        if len(matching_rows) > 0:
            # Return first matching row
            return matching_rows.iloc[0]
        return None
    

    def _get_text_label_string(self, t_bin_idx: int) -> Optional[str]:
        """Get text label string for a time bin.
        
        Args:
            t_bin_idx: Time bin index within current epoch
            
        Returns:
            Formatted text string or None if no label available
        """
        # Check if text_data_provider is provided (takes precedence)
        if self.plots_data.text_data_provider is not None:
            return self.plots_data.text_data_provider.get_text_label(self.params.curr_epoch_idx, t_bin_idx)
        
        # Fall back to existing text_columns approach
        row = self._match_time_bin_to_dataframe_row(t_bin_idx)
        if row is None or not self.params.text_columns:
            return None
        
        # Build label from specified columns
        label_parts = []
        for col in self.params.text_columns:
            if col in row.index:
                value = row[col]
                # Format the value appropriately
                if pd.isna(value):
                    label_parts.append(f"{col}: N/A")
                elif isinstance(value, (int, float)):
                    label_parts.append(f"{col}: {value:.3f}" if isinstance(value, float) else f"{col}: {value}")
                else:
                    label_parts.append(f"{col}: {value}")
        
        return "\n".join(label_parts) if label_parts else None
    
    
    def _clear_text_label_items(self):
        """Remove all text label items (Qt QLabel widgets) from the scene"""
        for label in self.text_label_items:
            try:
                if label and label.parent():
                    label.setParent(None)
                    label.deleteLater()
            except:
                pass
        self.text_label_items.clear()
    

    def _add_text_label_3d(self, label_text: str, t_bin_idx: int, x_translation: float, x_min: float, x_max: float, y_min: float, y_max: float, bin_spacing: float):
        """Add a text label as a Qt QLabel widget positioned below a time bin surface.
        
        Args:
            label_text: Text to display
            t_bin_idx: Time bin index
            x_translation: X translation of the time bin surface
            x_min, x_max: X bounds of the surface
            y_min, y_max: Y bounds of the surface
            bin_spacing: Spacing between bins
        """
        try:
            # Create a QLabel for the text
            label = qt.QLabel(label_text, self.plots.scene_window)
            label.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    color: white;
                    padding: 3px 6px;
                    border: none;
                    border-radius: 3px;
                    font-size: 9px;
                    font-weight: normal;
                }
            """)
            label.setAlignment(qt.Qt.AlignCenter | qt.Qt.AlignVCenter)
            label.setWordWrap(False)
            
            # Get scene window dimensions
            scene_width = self.plots.scene_window.width()
            scene_height = self.plots.scene_window.height()
            
            # Get slider height to avoid collision
            slider_height = 60  # Estimated height for slider + labels + spacing
            
            # If window not yet sized, use default or wait
            if scene_width <= 0 or scene_height <= 0:
                scene_width = 800  # Default width
                scene_height = 600  # Default height
            
            # Simple approach: distribute labels evenly based on time bin index
            # This avoids issues with 3D-to-2D coordinate mapping and camera transformations
            n_time_bins = self.curr_n_time_bins
            margin = 50  # pixels on each side
            usable_width = scene_width - 2 * margin  # Define usable_width before conditional
            
            if n_time_bins > 0:
                # Calculate evenly spaced positions across the window width
                # Leave margins on both sides
                if n_time_bins > 1:
                    # Distribute evenly: first label at margin, last label at (scene_width - margin)
                    # Each label is centered at its position
                    normalized_pos = t_bin_idx / (n_time_bins - 1)
                    x_pos_center = margin + (normalized_pos * usable_width)
                else:
                    # Single time bin: center it
                    x_pos_center = scene_width / 2
            else:
                x_pos_center = scene_width / 2
            
            # Calculate label width
            label_width = min(150, max(80, int(usable_width / max(n_time_bins, 1) * 0.8)))  # Scale with available space
            label_height = 200  # 4x taller (was 25px)
            
            # Center the label on the calculated x position
            x_pos = int(x_pos_center - label_width / 2)
            
            # Position above the slider with some padding
            y_pos = scene_height - label_height - slider_height - 10  # 10px padding above slider
            
            # Ensure label doesn't go outside bounds
            x_pos = max(5, min(x_pos, scene_width - label_width - 5))
            
            label.setGeometry(x_pos, y_pos, label_width, label_height)
            label.show()
            label.raise_()  # Bring to front
            label.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)  # Don't block mouse events
            
            self.text_label_items.append(label)
        except Exception:
            # Silently fail if label creation doesn't work
            pass


    def _configure_root_data_bounding_box(self):
        """Disable bounding box visualization on the root 'Data' node if present."""
        try:
            if not hasattr(self.plots.scene_widget, 'getItems'):
                return
            for item in self.plots.scene_widget.getItems():
                try:
                    name = item.getName() if hasattr(item, 'getName') else None
                    if name == 'Data' and hasattr(item, 'setBoundingBoxVisible'):
                        item.setBoundingBoxVisible(False)
                        break
                except Exception:
                    continue
        except Exception:
            # Silently ignore if scene introspection fails
            pass



    def _update_text_label_positions(self):
        """Update positions of all text labels after window resize"""
        if not self._label_data:
            return
        
        # Clear and recreate labels with updated positions
        self._clear_text_label_items()
        for label_data in self._label_data:
            label_text, t_bin_idx, x_translation, x_min, x_max, y_min, y_max, bin_spacing = label_data
            self._add_text_label_3d(label_text, t_bin_idx, x_translation, x_min, x_max, y_min, y_max, bin_spacing)



    # ==================================================================================================================================================================================================================================================================================== #
    # peak_contours Functions                                                                                                                                                                                                                                                              #
    # ==================================================================================================================================================================================================================================================================================== #

    def _extract_contours_for_epoch_timebin(self, epoch_idx: int, t_bin_idx: int) -> List[NDArray]:
        """Extract contour vertex arrays for a given (epoch_idx, t_bin_idx) from peak_prominence_result.

        Returns:
            List of (N, 2) arrays of (x, y) vertices in world coordinates.
        """
        from pyphoplacecellanalysis.External.peak_prominence2d import DecodedEpochIndex, DecodedEpochTimeBinIndex, DecodedEpochTimeBinIndexTuple

        if self.plots_data.peak_prominence_result is None:
            return []

        try:
            a_peaks_results: Dict[DecodedEpochTimeBinIndexTuple, Dict] = self.plots_data.peak_prominence_result.results
        except AttributeError:
            return []

        a_epoch_t_bin_tuple: DecodedEpochTimeBinIndexTuple = (int(epoch_idx), int(t_bin_idx))
        if a_epoch_t_bin_tuple not in a_peaks_results:
            return []

        an_epoch_t_bin_peaks_result: Dict = a_peaks_results[a_epoch_t_bin_tuple]
        peaks_dict = an_epoch_t_bin_peaks_result.get('peaks', {})
        if len(peaks_dict) == 0:
            return []

        shapes_data: List[NDArray] = []
        for _, peak_info in peaks_dict.items():
            level_slices = peak_info.get('level_slices', {})
            for _, slice_info in level_slices.items():
                contour = slice_info.get('contour', None)
                if contour is None:
                    continue
                vertices_world = getattr(contour, 'vertices', None)
                if vertices_world is None or len(vertices_world) == 0:
                    continue
                shapes_data.append(np.asarray(vertices_world, dtype=float))
        return shapes_data


    def _clear_peak_contour_items(self):
        """Remove all existing peak contour items from the scene widget."""
        if not self.plots.peak_contour_items:
            return
        for item in self.plots.peak_contour_items:
            try:
                if hasattr(self.plots.scene_widget, 'removeItem'):
                    self.plots.scene_widget.removeItem(item)
            except Exception:
                pass
        self.plots.peak_contour_items = []


    def _add_contours_for_current_epoch(self, edge_color: str = '#ffaaff', line_width: float = 1.0, z_offset: float = 0.01):
        """Add Silx 3D line items for all contours in the current epoch."""
        if self.plots_data.peak_prominence_result is None:
            return


        try:
            p_x_given_n = self.plots_data.decoded_result.p_x_given_n_list[self.params.curr_epoch_idx]
        except Exception:
            return

        n_x_bins, n_y_bins, n_time_bins = p_x_given_n.shape

        # mask_included_bins_list = self.plots_data.peak_contours['mask_included_bins_list'][self.params.curr_epoch_idx]
        mask_included_bins_list = self.plots_data.peak_contours['epoch_prom_t_bin_high_prob_pos_mask'][self.params.curr_epoch_idx]
        
        # epoch_prom_t_bin_high_prob_pos_mask: (n_x_bins, n_y_bins, n_t_bins)



        # [np.sum(a_mask) for t_bin_idx, a_mask in enumerate(mask_included_bins_list)]
        # [np.count_nonzero(a_mask) for t_bin_idx, a_mask in enumerate(mask_included_bins_list)]

        # Parse edge_color hex string into RGBA
        def _parse_hex_color(hex_color: str) -> Tuple[float, float, float, float]:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 8:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                a = int(hex_color[6:8], 16) / 255.0
            elif len(hex_color) == 6:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                a = 1.0
            else:
                r, g, b, a = 1.0, 0.0, 1.0, 1.0
            return (r, g, b, a)

        rgba_color = _parse_hex_color(edge_color)
        
        # Calculate a reasonable z_offset based on the data range
        # Height maps use the posterior values as z, so we need contours above the max value
        max_posterior_value = np.nanmax(p_x_given_n)
        # Position contours slightly above the maximum height map value
        effective_z_offset = max_posterior_value + (max_posterior_value * 0.1) if max_posterior_value > 0 else 0.1

        # Get coordinate arrays for converting pixel coordinates to world coordinates
        if (self.plots_data.xbin_centers is not None) and (self.plots_data.ybin_centers is not None):
            x_coords = np.array(self.plots_data.xbin_centers)
            y_coords = np.array(self.plots_data.ybin_centers)
        else:
            # Use bin indices as coordinates
            x_coords = np.arange(n_x_bins)
            y_coords = np.arange(n_y_bins)
        
        total_contours_added = 0
        # for t_bin_idx in range(n_time_bins):
        for t_bin_idx, a_t_bin_masks in enumerate(mask_included_bins_list):
            # a_t_bin_masks = mask_included_bins_list[t_bin_idx]
            num_masks: int = len(a_t_bin_masks)
            translation_triple = self.plots_data.translation_triple_list[t_bin_idx]
            curr_p_x_given_n = p_x_given_n[:, :, t_bin_idx]
            
            contours_list = []
            
            for slice_idx, a_mask in enumerate(a_t_bin_masks):
                if np.count_nonzero(a_mask) == 0:
                    continue
                
                # Extract contour from boolean mask using find_contours
                # find_contours expects a 2D array and returns contours in (row, col) format
                # where row corresponds to y-axis and col corresponds to x-axis
                try:
                    # Find contours at level 0.5 (midpoint between False=0 and True=1)
                    contours = find_contours(a_mask.astype(float), level=0.5)
                    
                    for contour in contours:
                        # contour is in (row, col) format where:
                        # - row (first column) corresponds to y-axis (second dimension of array)
                        # - col (second column) corresponds to x-axis (first dimension of array)
                        # Convert from pixel coordinates to world coordinates
                        row_indices = contour[:, 0]  # y-axis pixel coordinates
                        col_indices = contour[:, 1]  # x-axis pixel coordinates
                        
                        # Interpolate to world coordinates
                        # For x: use col_indices to index into x_coords
                        # For y: use row_indices to index into y_coords
                        # Handle fractional indices by interpolating
                        x_world = np.interp(col_indices, np.arange(len(x_coords)), x_coords)
                        y_world = np.interp(row_indices, np.arange(len(y_coords)), y_coords)
                        
                        # Create (N, 2) array of (x, y) coordinates
                        vertices = np.column_stack([x_world, y_world])
                        
                        # Only add if contour has at least 2 points
                        if len(vertices) >= 2:
                            contours_list.append(vertices)
                            
                except Exception as e:
                    print(f"DEBUG: Failed to extract contour from mask: {e}")
                    continue

            ## OUTPUTS: contours_list
            # contours_list = self._extract_contours_for_epoch_timebin(epoch_idx=self.params.curr_epoch_idx, t_bin_idx=t_bin_idx)
            if len(contours_list) == 0:
                continue

            # x_translation = t_bin_idx * bin_spacing
            for vertices in contours_list:
                if vertices.shape[1] != 2 or len(vertices) < 2:
                    continue

                # vertices are in world (x, y) coordinates; apply time-bin translation along X
                # x_coords_translated = vertices[:, 0] + translation_triple[0] # x_translation
                # y_coords_translated = vertices[:, 1] + translation_triple[1]
                # # Use the effective z offset calculated from data range
                # z_coords = np.full_like(x_coords_translated, effective_z_offset, dtype=float) + translation_triple[2]

                x_coords_translated = vertices[:, 0]
                y_coords_translated = vertices[:, 1]
                # Use the effective z offset calculated from data range
                z_coords = np.full_like(x_coords_translated, effective_z_offset, dtype=float)

                # # vertices are in world (x, y) coordinates; apply time-bin translation along X
                # x_coords_translated = x_coords_translated + translation_triple[0] # x_translation
                # y_coords_translated = y_coords_translated + translation_triple[1]
                # z_coords = z_coords + translation_triple[2]

                try:
                    values = np.ones_like(x_coords_translated, dtype=float)

                    # self.plots.time_bin_groupItems
                    
                    line_item = plot3d_items.Scatter3D()
                    line_item.setData(x_coords_translated, y_coords_translated, z_coords, values)
                    # line_item.setVisualization('lines')
                    # line_item.setLineWidth(float(line_width))
                    # line_item.setColor(rgba_color)
                    # Explicitly set visibility
                    if hasattr(line_item, 'setVisible'):
                        line_item.setVisible(True)
                    
                    # Ensure the item is added to the scene
                    if not self.params.use_groupItem:
                        self.plots.scene_widget.addItem(line_item)
                    else:
                        ## add to group item:
                        self.plots.time_bin_groupItems[t_bin_idx].addItem(line_item)

                    self.plots.peak_contour_items.append(line_item)
                    
                    line_item.setBoundingBoxVisible(False)
                    if not self.params.use_groupItem:
                        line_item.setTranslation(*translation_triple)
                    
                    line_item.setScale(1.0, 1.0, 1000.0) # Anisotropic scale: emphasize Z (time/height) dimension

                    total_contours_added += 1
                except Exception as e:
                    print(f"DEBUG: Failed to add contour line item: {e}")
                    continue
        ## END for t_bin_idx in range(n_tim...
        
        print(f"DEBUG: Added {total_contours_added} contour line items for epoch {self.params.curr_epoch_idx}")


    def add_peak_contours_overlays(self, peak_prominence_result, edge_color: str = '#ffaaff78', line_width: float = 1.0, z_offset: Optional[float] = None):
        """Adds peak contours as Silx 3D line overlays that update when the epoch slider changes.

        Mirrors the Napari add_peak_contours_layer conceptually but renders into the SceneWindow.

        Args:
            peak_prominence_result: PosteriorPeaksPeakProminence2dResult containing per-epoch, per-time-bin contours.
            edge_color: Hex RGBA string for contour color (default '#ffaaff78').
            line_width: Width of contour lines.
            z_offset: Constant Z offset above the base plane for the contour lines.
        """
        from pyphoplacecellanalysis.External.peak_prominence2d import PosteriorPeaksPeakProminence2dResult

        self.params.slice_level_multipliers = [0.9]
        self.plots_data.peak_contours = {}
        self.plots_data.peak_prominence_result = peak_prominence_result

        active_contour_level: float = 0.9
        mask_included_bins_list, summit_slice_levels_list, mask_included_p_x_given_n_list_dict, epoch_prom_t_bin_high_prob_pos_masks, epoch_prom_high_prob_pos_masks, *extra_outs = peak_prominence_result.compute_discrete_contour_masks(p_x_given_n_list=self.plots_data.decoded_result.p_x_given_n_list, slice_level_multipliers=self.params.slice_level_multipliers)

        # mask_included_bins_list
        # summit_slice_levels_list
        epoch_prom_high_prob_pos_mask = epoch_prom_high_prob_pos_masks[0.9] ## high
        # np.shape(epoch_prom_high_prob_pos_mask) # (74, 41, 63)

        self.plots_data.peak_contours['mask_included_bins_list'] = mask_included_bins_list
        self.plots_data.peak_contours['epoch_prom_t_bin_high_prob_pos_masks'] = epoch_prom_t_bin_high_prob_pos_masks
        self.plots_data.peak_contours['epoch_prom_high_prob_pos_masks'] = epoch_prom_high_prob_pos_masks


        self.plots_data.peak_contours['epoch_prom_t_bin_high_prob_pos_mask'] = epoch_prom_t_bin_high_prob_pos_masks[active_contour_level] # epoch_prom_t_bin_high_prob_pos_mask: (n_epochs, n_x_bins, n_ybins)
        self.plots_data.peak_contours['epoch_prom_high_prob_pos_mask'] = epoch_prom_high_prob_pos_masks[active_contour_level] # epoch_prom_high_prob_pos_mask: (n_epochs, n_x_bins, n_ybins)


        # Clear any existing contour items and rebuild for current epoch
        self._clear_peak_contour_items()
        self._add_contours_for_current_epoch(edge_color=edge_color, line_width=line_width, z_offset=z_offset)
    
        

    # ==================================================================================================================================================================================================================================================================================== #
    # Lifecycle Functions                                                                                                                                                                                                                                                                  #
    # ==================================================================================================================================================================================================================================================================================== #

    def _build_time_bin_positions(self):
        """ builds the mapping between time-bin-index and item positions for all items 

        Updates: 
            self.plots_data.translation_triple_list = []

            (self.params.all_epochs_p_x_given_n_min, self.params.all_epochs_p_x_given_n_max)

        """
        assert self.plots_data.decoded_result is not None
        assert len(self.plots_data.decoded_result.p_x_given_n_list) > 0
        
        ## max positions
        max_n_t_bins: int = np.nanmax(self.plots_data.decoded_result.nbins)

        self.params.max_n_t_bins = max_n_t_bins

        ## Build min/max for data
        # per_epoch_p_x_given_n_min_max_tuples_list = [(np.nanmin(p_x_given_n), np.nanmax(p_x_given_n))  for p_x_given_n in self.plots_data.decoded_result.p_x_given_n_list]
        per_epoch_p_x_given_n_min_list = np.array([np.nanmin(p_x_given_n) for p_x_given_n in self.plots_data.decoded_result.p_x_given_n_list])
        per_epoch_p_x_given_n_max_list = np.array([np.nanmax(p_x_given_n) for p_x_given_n in self.plots_data.decoded_result.p_x_given_n_list])

        all_epochs_p_x_given_n_min: float = np.nanmin(per_epoch_p_x_given_n_min_list)
        all_epochs_p_x_given_n_max: float = np.nanmax(per_epoch_p_x_given_n_min_list)


        self.params.per_epoch_p_x_given_n_min_list = per_epoch_p_x_given_n_min_list
        self.params.per_epoch_p_x_given_n_max_list = per_epoch_p_x_given_n_max_list

        self.params.all_epochs_p_x_given_n_min = all_epochs_p_x_given_n_min
        self.params.all_epochs_p_x_given_n_max = all_epochs_p_x_given_n_max

        # per_epoch_p_x_given_n_min_max_tuples_list)


        p_x_given_n = self.plots_data.decoded_result.p_x_given_n_list[0] ## get first item to determine shape
        # Shape: (n_x_bins, n_y_bins, n_time_bins)
        n_x_bins, n_y_bins, n_time_bins = p_x_given_n.shape
        
        self.plots_data.n_x_bins = n_x_bins
        self.plots_data.n_y_bins = n_y_bins

        # Calculate coordinate arrays
        if (self.plots_data.xbin_centers is not None) and (self.plots_data.ybin_centers is not None):
            # Use actual bin center values
            x_coords = np.array(self.plots_data.xbin_centers)
            y_coords = np.array(self.plots_data.ybin_centers)
            x_min, x_max = float(x_coords[0]), float(x_coords[-1])
            y_min, y_max = float(y_coords[0]), float(y_coords[-1])
            
        else:
            # Use bin indices
            x_coords = np.arange(n_x_bins)
            y_coords = np.arange(n_y_bins)
            x_min, x_max = 0.0, float(n_x_bins - 1)
            y_min, y_max = 0.0, float(n_y_bins - 1)
            # x_extent = float(n_x_bins - 1)


        x_extent: float = x_max - x_min
        y_extent: float = y_max - y_min

        self.plots_data.x_min = x_min
        self.plots_data.x_max = x_max
        self.plots_data.y_min = y_min
        self.plots_data.y_max = y_max
        self.plots_data.x_extent = x_extent
        self.plots_data.y_extent = y_extent

        # Create meshgrid for scatter plot coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        x_flat = X.flatten()
        y_flat = Y.flatten()

        self.plots_data.X = X
        self.plots_data.Y = Y
        self.plots_data.x_flat = x_flat
        self.plots_data.y_flat = y_flat

        # Calculate spacing between adjacent time bins
        self.plots_data.bin_spacing = self.plots_data.x_extent * self.params.spacing_factor
        
        # Create a height map surface for each time bin
        self.plots_data.translation_triple_list = []
        
        for t_bin_idx in range(self.params.max_n_t_bins):
            # Position horizontally: each bin offset along X axis
            x_translation = t_bin_idx * self.plots_data.bin_spacing

            translation_triple = (x_translation, 0.0, 0.0)
            self.plots_data.translation_triple_list.append(translation_triple)




    def _configure_time_bin_item(self, t_bin_idx: int, group_item=None, mesh_item=None, points_item=None, raise_on_error: bool=True):
        """Configure per-time-bin scatter item appearance and bounding box."""
        # Position horizontally: each bin offset along X axis
        translation_triple = self.plots_data.translation_triple_list[t_bin_idx] # (x_translation, 0.0, 0.0)
        assert len(translation_triple) == 3
        has_group_item: bool = (group_item is not None)
        
        if group_item is not None:
            try:
                group_item.setBoundingBoxVisible(True)
                group_item.setTranslation(*translation_triple)
                # Anisotropic scale: emphasize Z (time/height) dimension
                # group_item.setScale(1.0, 1.0, 1000.0)

            except Exception as e:
                # Keep failures non-fatal so the scene still renders
                print(f'ERROR: item[t_bin_idx]: {t_bin_idx}. Error {e}')
                if raise_on_error:
                    raise
                else:
                    pass
                        
        
        if mesh_item is not None:
            try:
                # # Get colormap
                # colormap: Colormap = item.getColormap()
                # # Set colormap
                # colormap.setName('viridis')
                # colormap.setVRange(self.params.all_epochs_p_x_given_n_min, self.params.all_epochs_p_x_given_n_max) # autoscaleMode='percentile_1_99'
                # # setAutoscaleMode
                # colormap: Colormap = Colormap(name='viridis', normalization='linear', vmin=self.params.all_epochs_p_x_given_n_min, vmax=self.params.all_epochs_p_x_given_n_max, autoscaleMode='percentile_1_99')
                colormap: Colormap = Colormap(name='viridis', normalization='linear', 
                    # vmin=self.params.all_epochs_p_x_given_n_min, vmax=self.params.all_epochs_p_x_given_n_max,
                    vmin=self.params.per_epoch_p_x_given_n_min_list[t_bin_idx], vmax=self.params.per_epoch_p_x_given_n_max_list[t_bin_idx], ## per epoch specific
                    # vmin=0.0, vmax=self.params.all_epochs_p_x_given_n_max,
                    # autoscaleMode='percentile_1_99',
                )

                mesh_item.setColormap(colormap)
                # Enable height map visualization
                mesh_item.setHeightMap(True)
                mesh_item.setVisualization('solid')
                mesh_item.setBoundingBoxVisible(not has_group_item)

                if group_item is None:
                    mesh_item.setTranslation(*translation_triple)
                    
                # Anisotropic scale: emphasize Z (time/height) dimension
                mesh_item.setScale(1.0, 1.0, 1000.0)

            except Exception as e:
                # Keep failures non-fatal so the scene still renders
                print(f'ERROR: item[t_bin_idx]: {t_bin_idx}. Error {e}')
                if raise_on_error:
                    raise
                else:
                    pass

        if points_item is not None:
            try:
                # Point/marker appearance (APIs may vary across silx versions)
                points_item.setVisualization('points') # setLineWidth
                points_item.setSymbol('s') # square
                points_item.setSymbolSize(5.0)

                points_colormap: Colormap = Colormap(name='viridis', normalization='linear', 
                    # vmin=self.params.all_epochs_p_x_given_n_min, vmax=self.params.all_epochs_p_x_given_n_max,
                    vmin=self.params.per_epoch_p_x_given_n_min_list[t_bin_idx], vmax=self.params.per_epoch_p_x_given_n_max_list[t_bin_idx], ## per epoch specific
                    # vmin=0.0, vmax=self.params.all_epochs_p_x_given_n_max,
                    # autoscaleMode='percentile_1_99',
                )
                points_item.setColormap(points_colormap)
                # Enable height map visualization
                points_item.setHeightMap(True)

                points_item.setBoundingBoxVisible(False)

                if group_item is None:
                    points_item.setTranslation(*translation_triple)

                # Anisotropic scale: emphasize Z (time/height) dimension
                points_item.setScale(1.0, 1.0, 1000.0)

            except Exception as e:
                # Keep failures non-fatal so the scene still renders
                print(f'ERROR: points_item[t_bin_idx]: {t_bin_idx}. Error {e}')
                if raise_on_error:
                    raise

        ## END if points_item is not None..


    def _create_time_bin_items(self):
        """Create and position all time bin height maps for current epoch"""
        self.params.use_groupItem = True

        p_x_given_n = self.plots_data.decoded_result.p_x_given_n_list[self.params.curr_epoch_idx]
        # Shape: (n_x_bins, n_y_bins, n_time_bins)
        n_x_bins, n_y_bins, n_time_bins = p_x_given_n.shape
        
        # Calculate coordinate arrays
        if (self.plots_data.xbin_centers is not None) and (self.plots_data.ybin_centers is not None):
            # Use actual bin center values
            x_coords = np.array(self.plots_data.xbin_centers)
            y_coords = np.array(self.plots_data.ybin_centers)
            x_min, x_max = float(x_coords[0]), float(x_coords[-1])
            y_min, y_max = float(y_coords[0]), float(y_coords[-1])
            x_extent = x_max - x_min
        else:
            # Use bin indices
            x_coords = np.arange(n_x_bins)
            y_coords = np.arange(n_y_bins)
            x_min, x_max = 0.0, float(n_x_bins - 1)
            y_min, y_max = 0.0, float(n_y_bins - 1)
            x_extent = float(n_x_bins - 1)
        
        # Create meshgrid for scatter plot coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Calculate spacing between adjacent time bins
        spacing_factor = 1.2  # 20% spacing between bins
        bin_spacing = x_extent * spacing_factor

        self._build_time_bin_positions()

        # Create a height map surface for each time bin
        for t_bin_idx in range(n_time_bins):
            # Get translation for this time bin
            translation_triple = self.plots_data.translation_triple_list[t_bin_idx]
            x_translation = translation_triple[0]  # Extract x_translation from translation_triple
            
            # Extract 2D slice for this time bin
            slice_2d = p_x_given_n[:, :, t_bin_idx]  # (n_x_bins, n_y_bins)
            values_flat = slice_2d.flatten()
            
            if self.params.use_groupItem:
                # Create a group item and add it to the scene
                # The group children share the group transform
                groupItem = plot3d_items.GroupItem()  # Create a new group item
                # groupItem.setTranslation(*translation_triple) # Translate the group
            else:
                groupItem = None

            # Create 2D scatter item with height map
            # mesh_item = self.scene_widget.add2DScatter(x_flat, y_flat, values_flat)
            mesh_item = None
            if (mesh_item is not None) and (groupItem is not None):
                groupItem.addItem(mesh_item)

            # points_item = self.plots.scene_widget.add2DScatter(x_flat, y_flat, values_flat)
            points_item: Scatter2D = Scatter2D()
            points_item.setData(x_flat, y_flat, values_flat)
            if (points_item is not None) and (groupItem is not None):
                groupItem.addItem(points_item)

            added_items = []
            if groupItem is not None:
                added_items.append(groupItem)
            if mesh_item is not None:
                added_items.append(mesh_item)
            if points_item is not None:
                added_items.append(points_item)

            # Per-item visualization and bounding box configuration
            self._configure_time_bin_item(t_bin_idx=t_bin_idx, group_item=groupItem, mesh_item=mesh_item, points_item=points_item)
            
            # if len(added_items) == 1:
            #     added_items = added_items[0] ## extract just the single item
            if groupItem is not None:
                self.scene_widget.addItem(groupItem)  # Add the group as an item of the scene
                self.plots.time_bin_groupItems.append(groupItem)
                
            # Store item for cleanup
            for an_item in added_items: 
                self.time_bin_items.append(an_item)
                
            # self.time_bin_items.append(points_item)
            # self.time_bin_items.extend(added_items)
            
            # Add text label below this time bin if available
            if self.plots_data.text_data_provider is not None or self.params.text_columns:
                label_text = self._get_text_label_string(t_bin_idx)
                if label_text is not None:
                    # Store label data for later positioning
                    self._label_data.append((label_text, t_bin_idx, x_translation, x_min, x_max, y_min, y_max, bin_spacing))
                    self._add_text_label_3d(label_text, t_bin_idx, x_translation, x_min, x_max, y_min, y_max, bin_spacing)

        ## END for t_bin_idx in range(n_time_bins)...

        # After items exist in the scene, configure the root 'Data' node bounding box if present
        self._configure_root_data_bounding_box()


    def _clear_time_bin_items(self):
        """Remove all time bin items from the scene"""
        def _perform_remove(an_item):
            try:
                if hasattr(self.plots.scene_widget, 'removeItem'):
                    self.plots.scene_widget.removeItem(an_item)
            except:
                pass


        for item in self.time_bin_items:
            if isinstance(item, (Tuple, List)):
                for an_item in item:
                    try:
                        _perform_remove(an_item=an_item)
                    except:
                        pass
            else:
                try:
                    _perform_remove(an_item=item)
                except:
                    pass
                
        self.time_bin_items.clear()
        self.plots.time_bin_groupItems.clear()        

    def on_epoch_changed(self, value):
        """Called when epoch slider changes"""
        self.params.curr_epoch_idx = int(value)
        self.epoch_label.setText(str(self.params.curr_epoch_idx))
        
        # Clear existing items
        self._clear_time_bin_items()
        self._clear_text_label_items()
        self._label_data = []  # Clear label data
        self._clear_peak_contour_items()
        
        # Create new time bin items for selected epoch
        self._create_time_bin_items()

        # Recreate peak-contour overlays for this epoch if available
        if self.plots_data.peak_prominence_result is not None:
            self._add_contours_for_current_epoch()
        
        # Update label positions after a short delay to ensure window is sized
        qt.QTimer.singleShot(100, self._update_text_label_positions)
        
        # Update table for current epoch if in point-like mode
        if self.params.is_point_like_mode:
            self._update_table_for_current_epoch()
            # Highlight matching row based on first time bin of the epoch (since this widget shows all time bins at once)
            self._highlight_matching_row_in_table(0)

        self.sigEpochIndexChanged.emit(self.params.curr_epoch_idx)
        


    def eventFilter(self, obj, event):
        """Event filter to catch scene window resize events"""
        try:
            # Check if this is a resize event
            # QEvent.Resize is typically 14 in Qt5/Qt6
            if obj == self.plots.scene_window:
                event_type = event.type()
                # Check for resize event (works for both Qt5 and Qt6)
                if hasattr(qt.QEvent, 'Resize') and event_type == qt.QEvent.Resize:
                    qt.QTimer.singleShot(50, self._update_text_label_positions)
                elif int(event_type) == 14:  # Resize event type code
                    qt.QTimer.singleShot(50, self._update_text_label_positions)
        except:
            pass
        return super().eventFilter(obj, event)

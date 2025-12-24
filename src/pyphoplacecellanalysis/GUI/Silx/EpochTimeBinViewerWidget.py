from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
from silx.gui.plot3d.SceneWindow import SceneWindow, items as plot3d_items
from silx.gui.colors import Colormap
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
import neuropy.utils.type_aliases as types
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


@metadata_attributes(short_name=None, tags=['Silx', 'gui', '3D', 'volumetric', 'epoch_idx_slider', 'epoch_t_bin_idx_slider', 'two-slider'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-23', related_items=[])
class EpochTimeBinViewer(qt.QWidget):
    """ Silx volumentric widget - renders two sliders, one controlling the `epoch_idx`, 
            and a sub-slider controlling the `t_bin_idx` within that epoch.

    Usage:

        from pyphoplacecellanalysis.GUI.Silx.EpochTimeBinViewerWidget import EpochTimeBinViewer
        from silx.gui import qt
        from silx.gui.plot import Plot2D
        from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
        from silx.gui.colors import Colormap
        import numpy as np
        from pyphoplacecellanalysis.GUI.Silx.EpochTimeBinViewerWidget import EpochTimeBinViewer

        # Usage:
        a_decoder = a_widget.container.pf1D_Decoder_dict['roam']
        # p_x_given_n = a_widget.decoded_result.p_x_given_n_list[a_widget.active_epoch_idx]
        viewer = EpochTimeBinViewer(decoded_result=a_widget.decoded_result,
                                        xbin_centers=a_decoder.xbin_centers,
                                        ybin_centers=a_decoder.ybin_centers)
        viewer.show()



    """
    def __init__(self, decoded_result, xbin_centers=None, ybin_centers=None):
        super().__init__()
        self.decoded_result = decoded_result
        self.xbin_centers = xbin_centers
        self.ybin_centers = ybin_centers
        
        # Current indices
        self.curr_epoch_idx = 0
        self.curr_time_bin_idx = 0
        
        # Store wireframe items for cleanup
        self.wireframe_items = []
        
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
        self.epoch_slider.setMaximum(len(decoded_result.p_x_given_n_list) - 1)
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
        p_x_given_n = self.decoded_result.p_x_given_n_list[self.curr_epoch_idx]
        return p_x_given_n.shape[-1]  # Last dimension is time
    
    def update_time_bin_slider_range(self):
        """Update time_bin slider range when epoch changes"""
        max_time_bins = self.curr_n_time_bins - 1
        self.time_bin_slider.setMaximum(max_time_bins)
        # Reset to 0 when epoch changes
        self.time_bin_slider.setValue(0)
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
        p_x_given_n = self.decoded_result.p_x_given_n_list[self.curr_epoch_idx]
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
            p_x_given_n = self.decoded_result.p_x_given_n_list[self.curr_epoch_idx]
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
    
    def update_2d_slice(self):
        """Update the 2D slice view for current time bin"""
        p_x_given_n = self.decoded_result.p_x_given_n_list[self.curr_epoch_idx]
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


@metadata_attributes(short_name=None, tags=['Silx', 'gui', '3D', 'scene', 'epoch_idx_slider', 'height-map'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-23', related_items=[])
class Epoch3DSceneTimeBinViewer(qt.QWidget):
    """ Silx 3D scene widget - displays all time bins from the active epoch as adjacent 3D height map surfaces.
    
    This widget uses a SceneWindow to display multiple time bins from the selected epoch
    as 3D height map surfaces arranged horizontally. Each time bin is displayed as a 
    2D scatter plot with height map enabled, positioned side-by-side along the X axis.
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.Silx.EpochTimeBinViewerWidget import Epoch3DSceneTimeBinViewer
        
        # Usage:
        a_decoder = a_widget.container.pf1D_Decoder_dict['roam']
        viewer = Epoch3DSceneTimeBinViewer(decoded_result=a_widget.decoded_result,
                                           xbin_centers=a_decoder.xbin_centers,
                                           ybin_centers=a_decoder.ybin_centers)
        viewer.show()
    
    """
    def __init__(self, decoded_result, xbin_centers=None, ybin_centers=None):
        super().__init__()
        self.decoded_result = decoded_result
        self.xbin_centers = xbin_centers
        self.ybin_centers = ybin_centers
        
        # Current epoch index
        self.curr_epoch_idx = 0
        
        # Store time bin items for cleanup
        self.time_bin_items = []
        
        # Create UI
        layout = qt.QVBoxLayout()
        self.setLayout(layout)
        
        # Create SceneWindow for 3D visualization
        self.scene_window = SceneWindow()
        self.scene_widget = self.scene_window.getSceneWidget()
        
        # Add SceneWindow to layout
        layout.addWidget(self.scene_window)
        
        # Create epoch slider
        slider_layout = qt.QHBoxLayout()
        
        self.epoch_slider = qt.QSlider(qt.Qt.Horizontal)
        self.epoch_slider.setMinimum(0)
        self.epoch_slider.setMaximum(len(decoded_result.p_x_given_n_list) - 1)
        self.epoch_slider.setTickPosition(qt.QSlider.TicksBelow)
        self.epoch_slider.setTickInterval(1)
        self.epoch_slider.valueChanged.connect(self.on_epoch_changed)
        slider_layout.addWidget(qt.QLabel("Epoch:"))
        slider_layout.addWidget(self.epoch_slider)
        slider_layout.addWidget(qt.QLabel("0"))
        self.epoch_label = slider_layout.itemAt(slider_layout.count()-1).widget()
        
        layout.addLayout(slider_layout)
        
        # Initialize
        self.on_epoch_changed(0)
    
    @property
    def curr_n_time_bins(self) -> int:
        """Get number of time bins for current epoch"""
        p_x_given_n = self.decoded_result.p_x_given_n_list[self.curr_epoch_idx]
        return p_x_given_n.shape[-1]  # Last dimension is time
    
    def _clear_time_bin_items(self):
        """Remove all time bin items from the scene"""
        for item in self.time_bin_items:
            try:
                if hasattr(self.scene_widget, 'removeItem'):
                    self.scene_widget.removeItem(item)
            except:
                pass
        self.time_bin_items.clear()
    
    def _create_time_bin_items(self):
        """Create and position all time bin height maps for current epoch"""
        p_x_given_n = self.decoded_result.p_x_given_n_list[self.curr_epoch_idx]
        # Shape: (n_x_bins, n_y_bins, n_time_bins)
        n_x_bins, n_y_bins, n_time_bins = p_x_given_n.shape
        
        # Calculate coordinate arrays
        if self.xbin_centers is not None and self.ybin_centers is not None:
            # Use actual bin center values
            x_coords = np.array(self.xbin_centers)
            y_coords = np.array(self.ybin_centers)
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
        
        # Create a height map surface for each time bin
        for t_bin_idx in range(n_time_bins):
            # Extract 2D slice for this time bin
            slice_2d = p_x_given_n[:, :, t_bin_idx]  # (n_x_bins, n_y_bins)
            values_flat = slice_2d.flatten()
            
            # Create 2D scatter item with height map
            item = self.scene_widget.add2DScatter(x_flat, y_flat, values_flat)
            
            # Enable height map visualization
            item.setHeightMap(True)
            item.setVisualization('solid')
            
            # Set colormap
            item.getColormap().setName('viridis')
            
            # Position horizontally: each bin offset along X axis
            x_translation = t_bin_idx * bin_spacing
            item.setTranslation(x_translation, 0.0, 0.0)
            
            # Set scale to maintain proper aspect ratio
            if self.xbin_centers is not None and self.ybin_centers is not None:
                # Use actual scale based on bin centers
                x_scale = 1.0
                y_scale = 1.0
            else:
                # Scale based on number of bins
                x_scale = 1.0
                y_scale = 1.0
            item.setScale(x_scale, y_scale, 1.0)
            
            # Store item for cleanup
            self.time_bin_items.append(item)
    
    def on_epoch_changed(self, value):
        """Called when epoch slider changes"""
        self.curr_epoch_idx = int(value)
        self.epoch_label.setText(str(self.curr_epoch_idx))
        
        # Clear existing time bin items
        self._clear_time_bin_items()
        
        # Create new time bin items for selected epoch
        self._create_time_bin_items()

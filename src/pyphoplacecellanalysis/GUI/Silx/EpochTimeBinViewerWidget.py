from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
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
            scene_widget = self.view_3d.getSceneWidget()
            items = scene_widget.getItems()
            for item in items:
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
            self.view_3d.setScale(1.0, 1.0, time_scale)
            self.view_3d.setAxesLabels('X', 'Y', 'Time Bin')
        
        # Update 2D slice view
        self.update_2d_slice()
    
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

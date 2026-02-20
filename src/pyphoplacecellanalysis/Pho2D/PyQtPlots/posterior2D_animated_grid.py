import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyqtgraph.Qt import QtWidgets, QtCore
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent
from PyQt5.QtGui import QPainter

@metadata_attributes(short_name=None, tags=['animated', 'PBE', 'loop'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-02-20 12:22', related_items=[])
class AnimatedLoopingPosteriorViewer(QtWidgets.QMainWindow):
    """ Awesome, animated PBE viewer that loops over PBEs
    
    Usage:
    
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.posterior2D_animated_grid import AnimatedLoopingPosteriorViewer

        app = pg.mkQApp('AnimatedLoopingPosteriorViewer') # QtWidgets.QApplication(sys.argv)
        viewer = AnimatedLoopingPosteriorViewer(active_decoded_PBE_result)
        viewer.resize(800, 1000)
        viewer.show()


    """
    enable_debug_print: bool = False

    def __init__(self, active_decoded_PBE_result, n_columns: int = 10, xbin=None, ybin=None, drop_below_threshold: float=0.0000001):
        super().__init__()

        self.active_decoded_PBE_result = active_decoded_PBE_result
        self.n_columns = n_columns

        self.applicationName = 'AnimatedLoopingPosteriorViewer'
        self.xbin = xbin
        self.ybin = ybin
        self.drop_below_threshold = drop_below_threshold
        self.setup() ## call setup

        self._buildGraphics()

        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(50)  # ms per frame


    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        self.app = pg.mkQApp(self.applicationName)
        self.params = VisualizationParameters(self.applicationName, debug_view_mode=False, drop_below_threshold=self.drop_below_threshold) #  xbin=xbin, ybin=ybin, n_columns=n_columns
        # self.params.shared_axis_order = 'row-major'
        self.params.shared_axis_order = 'col-major'
        # self.params.shared_axis_order = None # #TODO 2025-06-30 17:42: - [ ] was like this, but posteriors plotted seem wrong
        self.params.decoded_time_bins_info_df = None
        
        ## Build the colormap to be used:
        # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        # self.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
        self.params.cmap = pg.colormap.get('viridis','matplotlib')
        self.params.debug_view_mode = False
        
        self.params.image_margins = 0.0
        self.params.image_bounds_extent, self.params.x_range, self.params.y_range = pyqtplot_build_image_bounds_extent(self.xbin, self.ybin, margin=self.params.image_margins, debug_print=self.enable_debug_print)
        


    def _buildGraphics(self):
        """ basic """
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        # 🔥 Changed from QVBoxLayout → QGridLayout
        self.main_layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.main_layout)

        self.image_items = []
        self.current_t_bins = []

        debug_view_mode: bool = self.params.debug_view_mode

        # Build one animated cell per epoch
        for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs):

            an_epoch_p_x_given_n = self.active_decoded_PBE_result.p_x_given_n_list[an_epoch_idx]
            an_epoch_n_bins: int = self.active_decoded_PBE_result.nbins[an_epoch_idx]

            plot_widget = pg.PlotWidget()
            plot_widget.setAspectLocked(True)
            plot_widget.invertY(True)
            plot_widget.hideAxis('left')
            plot_widget.hideAxis('bottom')

            img_item = pg.ImageItem(border='w')
            plot_widget.addItem(img_item, defaultPadding=0.0)
            # img_item.setRange(QtCore.QRectF(*self.params.image_bounds_extent))

            # plot_widget.showAxes(False)
            # plot_widget.hideButtons() # Hides the auto-scale button
            
            # # self.ui.root_plot.showAxes(False)        
            plot_widget.setRange(xRange=self.params.x_range, yRange=self.params.y_range, padding=0.0)
            # Sets only the panning limits:
            plot_widget.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1])

            plot_widget.setMouseEnabled(x=debug_view_mode, y=debug_view_mode)
            plot_widget.setMenuEnabled(enableMenu=debug_view_mode)
            
            # Set the color map:
            img_item.setColorMap(self.params.cmap)

            if isinstance(self.params.cmap, NDArray):
                img_item.setLookupTable(self.params.cmap, update=True)
            else:
                img_item.setLookupTable(self.params.cmap.getLookupTable(nPts=256), update=True)

            # Compute grid position
            row = an_epoch_idx // self.n_columns
            col = an_epoch_idx % self.n_columns

            self.main_layout.addWidget(plot_widget, row, col)

            self.image_items.append((img_item, an_epoch_p_x_given_n, an_epoch_n_bins))
            self.current_t_bins.append(0)
    ## END for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs)...




    def update_frames(self):
        """ updates/animates the PBE frames """        
        for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs):

            img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]

            an_epoch_t_bin = self.current_t_bins[an_epoch_idx]

            if an_epoch_t_bin < an_epoch_n_bins:
                a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin]

                image = np.squeeze(a_t_bin_p_x_given_n).copy()
                # image_title = f'{self.name}'
            
                if self.params.drop_below_threshold is not None:
                    image[np.where(image < self.params.drop_below_threshold)] = np.nan # null out the low values if needed

                is_all_nan = np.all(np.isnan(image))

                # Optional normalization for better contrast
                # img_item.setImage(a_t_bin_p_x_given_n.T, autoLevels=True)
                # img_item.setImage(image, autoLevels=True)

                ## get the image item to draw:
                imv: pg.ImageItem = img_item
                # imv.setCompositionMode(QPainter.CompositionMode_Plus) ## Set this mode so that the heatmap overlays the occupancy map
                if is_all_nan:
                    # Optimization: Don't just set data to None/NaN, actually remove it from the render pipeline
                    if imv.isVisible():
                        print(f'WARNING: is_all_nan == True for image. Hiding item.')
                        imv.hide()
                        # Optional: clear the image data to free memory if the object stays hidden for long
                        # imv.clear() 
                else:
                    # Ensure it is visible if it was previously hidden
                    if not imv.isVisible():
                        imv.show()

                    # Update Image:
                    if self.params.shared_axis_order is None:
                        imv.setImage(image, rect=self.params.image_bounds_extent)
                    else:
                        imv.setImage(image, rect=self.params.image_bounds_extent, axisOrder=self.params.shared_axis_order)

                self.current_t_bins[an_epoch_idx] += 1
            else:
                self.current_t_bins[an_epoch_idx] = 0



        ## END for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs)...


    def export_grid_as_gif(self, output_path: str, frame_duration_ms: int = 50):
        """ Exports the entire animated grid as an animated GIF.

        Parameters
        ----------
        output_path : str
            Path to save GIF.
        frame_duration_ms : int
            Duration per frame in milliseconds.
        """

        import imageio
        from PyQt5.QtGui import QImage

        frames = []

        # Determine maximum number of frames across all epochs
        max_n_bins = np.max(self.active_decoded_PBE_result.nbins)

        # Save current state so we can restore later
        original_t_bins = self.current_t_bins.copy()

        for global_frame_idx in range(max_n_bins):

            # Manually set each epoch to correct frame (looping behavior preserved)
            for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs):

                img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]

                if global_frame_idx < an_epoch_n_bins:
                    an_epoch_t_bin = global_frame_idx
                else:
                    an_epoch_t_bin = 0  # loop shorter epochs

                a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin]

                img_item.setImage(
                    a_t_bin_p_x_given_n.T,
                    autoLevels=False,
                    levels=(0, 1)
                )

            # Process Qt events so rendering completes
            QtWidgets.QApplication.processEvents()

            # Grab full window image
            qpixmap = self.grab()
            qimage = qpixmap.toImage().convertToFormat(QImage.Format_RGBA8888)

            width = qimage.width()
            height = qimage.height()

            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)

            frames.append(arr.copy())

        ## END for global_frame_idx in range(max_n_bins)
        
        # Restore previous animation state
        self.current_t_bins = original_t_bins

        # Save GIF
        imageio.mimsave(
            output_path,
            frames,
            duration=frame_duration_ms / 1000.0,
            loop=1
        )

        print(f"Exported GIF to: {output_path}")
        


# if __name__ == "__main__":
# 	app = pg.mkQApp('AnimatedLoopingPosteriorViewer') # QtWidgets.QApplication(sys.argv)
# 	viewer = AnimatedLoopingPosteriorViewer(active_decoded_PBE_result)
# 	viewer.resize(1200, 900)
# 	viewer.show()
# 	sys.exit(app.exec_())
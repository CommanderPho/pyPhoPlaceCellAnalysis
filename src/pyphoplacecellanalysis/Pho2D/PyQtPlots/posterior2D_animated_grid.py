import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

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
    def __init__(self, active_decoded_PBE_result, n_columns: int = 10):
        super().__init__()

        self.active_decoded_PBE_result = active_decoded_PBE_result
        self.n_columns = n_columns

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        # 🔥 Changed from QVBoxLayout → QGridLayout
        self.layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.layout)

        self.image_items = []
        self.current_t_bins = []

        # Build one animated cell per epoch
        for an_epoch_idx in np.arange(active_decoded_PBE_result.n_epochs):

            an_epoch_p_x_given_n = active_decoded_PBE_result.p_x_given_n_list[an_epoch_idx]
            an_epoch_n_bins: int = active_decoded_PBE_result.nbins[an_epoch_idx]

            plot_widget = pg.PlotWidget()
            plot_widget.setAspectLocked(True)
            plot_widget.invertY(True)
            plot_widget.hideAxis('left')
            plot_widget.hideAxis('bottom')

            img_item = pg.ImageItem()
            plot_widget.addItem(img_item)

            # Compute grid position
            row = an_epoch_idx // self.n_columns
            col = an_epoch_idx % self.n_columns

            self.layout.addWidget(plot_widget, row, col)

            self.image_items.append((img_item, an_epoch_p_x_given_n, an_epoch_n_bins))
            self.current_t_bins.append(0)

        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(50)  # ms per frame

    def update_frames(self):
        """ updates/animates the PBE frames """        
        for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs):

            img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]

            an_epoch_t_bin = self.current_t_bins[an_epoch_idx]

            if an_epoch_t_bin < an_epoch_n_bins:
                a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin]

                # Optional normalization for better contrast
                img_item.setImage(a_t_bin_p_x_given_n.T, autoLevels=True)

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
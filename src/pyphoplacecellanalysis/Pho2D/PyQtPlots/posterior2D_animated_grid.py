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

        # 🔥 Changed from QVBoxLayout → QGridLayout (name avoids shadowing QMainWindow.layout())
        self.grid_layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.grid_layout)

        self.image_items = []
        self.current_t_bins = []

        
        # Build one animated cell per epoch
        for an_epoch_idx in np.arange(active_decoded_PBE_result.n_epochs):

            an_epoch_p_x_given_n = active_decoded_PBE_result.p_x_given_n_list[an_epoch_idx]
            an_epoch_n_bins: int = active_decoded_PBE_result.nbins[an_epoch_idx]

            plot_widget = pg.PlotWidget()
            plot_widget.setAspectLocked(True)
            # plot_widget.invertY(True)
            plot_widget.invertY(False)
            plot_widget.hideAxis('left')
            plot_widget.hideAxis('bottom')

            img_item = pg.ImageItem(border='w')
            # lut = pg.colormap.get('viridis','matplotlib').getLookupTable(256)
            # img_item = pg.ImageItem(lut=lut)
            img_item.setColorMap(pg.colormap.get('viridis','matplotlib'))

            plot_widget.addItem(img_item)

            # Compute grid position
            row = an_epoch_idx // self.n_columns
            col = an_epoch_idx % self.n_columns

            self.grid_layout.addWidget(plot_widget, row, col)

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
                # img_item.setImage(a_t_bin_p_x_given_n.T, autoLevels=False, levels=(0, 1))


                self.current_t_bins[an_epoch_idx] += 1
            else:
                self.current_t_bins[an_epoch_idx] = 0

        ## END for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs)...


    def export_grid_as_gif(self, output_path: str, frame_duration_ms: int = 50, render_passes: int = 5):
        """ Exports the entire animated grid as an animated GIF.

        Parameters
        ----------
        output_path : str
            Path to save GIF.
        frame_duration_ms : int
            Duration per frame in milliseconds.
        render_passes : int
            Number of Qt event processing passes per frame before grab (more = smoother capture).
        """
        import imageio
        from pyqtgraph.Qt import QtGui

        # Global min/max across all epochs so colormap is consistent (smoother than per-frame levels)
        all_data = np.concatenate([p for _, p, _ in self.image_items], axis=2)
        gmin, gmax = float(np.nanmin(all_data)), float(np.nanmax(all_data))
        if gmax <= gmin:
            gmax = gmin + 1.0
        levels = (gmin, gmax)

        frames = []
        max_n_bins = np.max(self.active_decoded_PBE_result.nbins)
        original_t_bins = self.current_t_bins.copy()

        for global_frame_idx in range(max_n_bins):
            for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs):
                img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]
                an_epoch_t_bin = global_frame_idx if global_frame_idx < an_epoch_n_bins else 0
                a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin]
                img_item.setImage(a_t_bin_p_x_given_n.T, autoLevels=False, levels=levels)

            for _ in range(render_passes):
                QtWidgets.QApplication.processEvents()

            qpixmap = self.grab()
            qimage = qpixmap.toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
            width, height = qimage.width(), qimage.height()
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.ascontiguousarray(np.array(ptr).reshape(height, width, 4).copy())
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
            rgb[:] = (arr[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
            frames.append(rgb)

        self.current_t_bins = original_t_bins

        try:
            imageio.mimsave(output_path, frames, format="GIF-PIL", duration=frame_duration_ms / 1000.0, loop=0, quantizer="nq", palettesize=256)
        except Exception:
            imageio.mimsave(output_path, frames, duration=frame_duration_ms / 1000.0, loop=0)

        print(f"Exported GIF to: {output_path}")
        



@metadata_attributes(short_name=None, tags=['NEW', 'replacement', 'higher-efficiency'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-02-23 19:14', related_items=[])
class AnimatedLoopingPosteriorGraphicsGridViewer(QtWidgets.QMainWindow):
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

        self.active_decoded_filter_epochs_result = active_decoded_PBE_result
        self.n_columns = n_columns
        max_n_t_bins: int = np.nanmax(self.active_decoded_filter_epochs_result.nbins) ## get the maximum number of t_bins in any epoch
        self.max_n_t_bins = max_n_t_bins
        
        # self.central_widget = QtWidgets.QWidget()
        # self.setCentralWidget(self.central_widget)

        self.setup()
        self._build_graphics()        

        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(50)  # ms per frame


    def setup(self):
        self.image_items = []
        self.text_items = []
        self.current_t_bins = []
        self.black_image_items = []
    

    def _build_graphics(self):
        """ builds the bank of animated cells """
        # 🔥 Changed from QVBoxLayout → QGridLayout (name avoids shadowing QMainWindow.layout())
        # self.grid_layout = QtWidgets.QGridLayout()
        # self.central_widget.setLayout(self.grid_layout)    
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        # self.central_widget.setLayout(self.graphics_layout_widget)
        # self.central_widget.addWidget(self.graphics_layout_widget)
        self.setCentralWidget(self.graphics_layout_widget)

        # Build one animated cell per epoch
        for an_epoch_idx in np.arange(self.active_decoded_filter_epochs_result.n_epochs):
            an_epoch_p_x_given_n = self.active_decoded_filter_epochs_result.p_x_given_n_list[an_epoch_idx]
            an_epoch_n_bins: int = self.active_decoded_filter_epochs_result.nbins[an_epoch_idx]

            # Compute grid position
            row = an_epoch_idx // self.n_columns
            col = an_epoch_idx % self.n_columns
            
            plot_title: str = f'{an_epoch_idx}'
            plot_widget = self.graphics_layout_widget.addPlot(row=row, col=col) # , title=plot_title
            
            # plot_widget = pg.PlotWidget()
            plot_widget.setAspectLocked(True)
            # plot_widget.invertY(True)
            plot_widget.invertY(False)
            plot_widget.hideAxis('left')
            plot_widget.hideAxis('bottom')

            img_item = pg.ImageItem(border='w')
            # lut = pg.colormap.get('viridis','matplotlib').getLookupTable(256)
            # img_item = pg.ImageItem(lut=lut)
            img_item.setColorMap(pg.colormap.get('viridis','matplotlib'))

            plot_widget.addItem(img_item)

            ## Build text item to display title
            txt_item: pg.TextItem = pg.TextItem(text=plot_title, color='#FFFFFF77', anchor=(1, 1))
            plot_widget.addItem(txt_item)
            img_rect = img_item.boundingRect()
            txt_item.setPos(img_rect.right(), img_rect.bottom())

            # a_black_image_item = np.zeros_like(an_epoch_p_x_given_n[:, :, 0])
            a_black_image_item = np.full_like(an_epoch_p_x_given_n[:, :, 0], np.nan)
            
            self.black_image_items.append(a_black_image_item)

            # img_item.addItem(txt_item)
            # self.grid_layout.addWidget(plot_widget, row, col)
            self.image_items.append((img_item, an_epoch_p_x_given_n, an_epoch_n_bins))
            # self.current_t_bins.append(an_epoch_n_bins)
            self.current_t_bins.append(0) ## starting at zero
            self.text_items.append(txt_item)



    def update_frames(self):
        """ updates/animates the PBE frames """        
        
        max_n_t_bins: int = self.max_n_t_bins
        

        for an_epoch_idx in np.arange(self.active_decoded_filter_epochs_result.n_epochs):

            img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]

            an_epoch_t_bin: int = self.current_t_bins[an_epoch_idx] ## get current time bin iterator for this index

            if (an_epoch_t_bin < an_epoch_n_bins):
                a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin] ## current time bin posterior

                # Optional normalization for better contrast
                img_item.setImage(a_t_bin_p_x_given_n, autoLevels=True)
                # img_item.setImage(a_t_bin_p_x_given_n.T, autoLevels=False, levels=(0, 1))

                self.current_t_bins[an_epoch_idx] += 1
            else:
                ## BEHAVIOR: resets to zero and starts over, allowing them all to operate "out of sync"
                # self.current_t_bins[an_epoch_idx] = 0 ## Resets to zero, I don't think this is desirable

                if max_n_t_bins == self.current_t_bins[an_epoch_idx]:
                    ## truely reset to zero (for all of them
                    self.current_t_bins[an_epoch_idx] = 0 ## Resets to zero, I don't think this is desirable
                else:                
                    ## render a black frame and keep accruing:
                    img_item.setImage(self.black_image_items[an_epoch_idx], autoLevels=True)
                    self.current_t_bins[an_epoch_idx] += 1
                

        ## END for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs)...


    def export_grid_as_gif(self, output_path: str, frame_duration_ms: int = 50, render_passes: int = 5):
        """ Exports the entire animated grid as an animated GIF.

        Parameters
        ----------
        output_path : str
            Path to save GIF.
        frame_duration_ms : int
            Duration per frame in milliseconds.
        render_passes : int
            Number of Qt event processing passes per frame before grab (more = smoother capture).
        """
        import imageio
        from pyqtgraph.Qt import QtGui

        # Global min/max across all epochs so colormap is consistent (smoother than per-frame levels)
        all_data = np.concatenate([p for _, p, _ in self.image_items], axis=2)
        gmin, gmax = float(np.nanmin(all_data)), float(np.nanmax(all_data))
        if gmax <= gmin:
            gmax = gmin + 1.0
        levels = (gmin, gmax)

        frames = []
        max_n_bins = np.max(self.active_decoded_filter_epochs_result.nbins)
        original_t_bins = self.current_t_bins.copy()

        for global_frame_idx in range(max_n_bins):
            for an_epoch_idx in np.arange(self.active_decoded_filter_epochs_result.n_epochs):
                img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]
                an_epoch_t_bin = global_frame_idx if global_frame_idx < an_epoch_n_bins else 0
                a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin]
                img_item.setImage(a_t_bin_p_x_given_n.T, autoLevels=False, levels=levels)

            for _ in range(render_passes):
                QtWidgets.QApplication.processEvents()

            qpixmap = self.grab()
            qimage = qpixmap.toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
            width, height = qimage.width(), qimage.height()
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.ascontiguousarray(np.array(ptr).reshape(height, width, 4).copy())
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
            rgb[:] = (arr[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
            frames.append(rgb)

        self.current_t_bins = original_t_bins

        try:
            imageio.mimsave(output_path, frames, format="GIF-PIL", duration=frame_duration_ms / 1000.0, loop=0, quantizer="nq", palettesize=256)
        except Exception:
            imageio.mimsave(output_path, frames, duration=frame_duration_ms / 1000.0, loop=0)

        print(f"Exported GIF to: {output_path}")
        



# if __name__ == "__main__":
# 	app = pg.mkQApp('AnimatedLoopingPosteriorViewer') # QtWidgets.QApplication(sys.argv)
# 	viewer = AnimatedLoopingPosteriorViewer(active_decoded_PBE_result)
# 	viewer.resize(1200, 900)
# 	viewer.show()
# 	sys.exit(app.exec_())
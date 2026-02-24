import sys
import numpy as np
from enum import Enum, auto
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
        self.current_t_bin_index = []

        
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
            self.current_t_bin_index.append(0)

        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(50)  # ms per frame


    def update_frames(self):
        """ updates/animates the PBE frames """        
        for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs):

            img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]

            an_epoch_t_bin = self.current_t_bin_index[an_epoch_idx]

            if an_epoch_t_bin < an_epoch_n_bins:
                a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin]

                # Optional normalization for better contrast
                img_item.setImage(a_t_bin_p_x_given_n.T, autoLevels=True)
                # img_item.setImage(a_t_bin_p_x_given_n.T, autoLevels=False, levels=(0, 1))


                self.current_t_bin_index[an_epoch_idx] += 1
            else:
                self.current_t_bin_index[an_epoch_idx] = 0

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
        original_t_bins = self.current_t_bin_index.copy()

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

        self.current_t_bin_index = original_t_bins

        try:
            imageio.mimsave(output_path, frames, format="GIF-PIL", duration=frame_duration_ms / 1000.0, loop=0, quantizer="nq", palettesize=256)
        except Exception:
            imageio.mimsave(output_path, frames, duration=frame_duration_ms / 1000.0, loop=0)

        print(f"Exported GIF to: {output_path}")
        

class AnimationLoopMode(Enum):
    """ controls how each independnent epoch loops (with coherent frames across all which go black when they run out, or item-wise. """
    common_frame_indexing = auto()
    individual_independent_looping = auto()


class AnimationExportMode(Enum):
    """ controls how animated gifs are exported """
    separate_images = auto()
    single_image = auto()


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
    def __init__(self, active_decoded_filter_epochs_result, n_columns: int = 10, loop_mode: AnimationLoopMode=AnimationLoopMode.common_frame_indexing):
        super().__init__()

        self.active_decoded_filter_epochs_result = active_decoded_filter_epochs_result
        self.n_columns = n_columns
        max_n_t_bins: int = np.nanmax(self.active_decoded_filter_epochs_result.nbins) ## get the maximum number of t_bins in any epoch
        self.max_n_t_bins = max_n_t_bins
        self.loop_mode = loop_mode


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
        self.current_t_bin_index = []
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
            an_epoch_single_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, 0]

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
            img_item.setImage(an_epoch_single_t_bin_p_x_given_n, autoLevels=True) ## set initial image to get proper shape

            plot_widget.addItem(img_item)

            ## Build text item to display title
            txt_item: pg.TextItem = pg.TextItem(text=plot_title, color='#FFFFFF77', anchor=(1, 1))
            ## Create text object, use HTML tags to specify color/size
            # txt_item: pg.TextItem = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">This is the</span><br><span style="color: #FF0; font-size: 16pt;">PEAK</span></div>', anchor=(-0.3,0.5), angle=45, border='w', fill=(0, 0, 255, 100))
            # plot.addItem(txt_item)
            # txt_item.setPos(0, y.max())

            plot_widget.addItem(txt_item)
            img_rect = img_item.boundingRect()
            # txt_item.setPos(img_rect.right(), img_rect.bottom())
            txt_item.setPos(img_rect.right(), img_rect.top()) ## .top() is the correct choice because the y-axis is flipped or something by default. This line actually positions the text at the bottom-right inside corner.

            # a_black_image_item = np.zeros_like(an_epoch_p_x_given_n[:, :, 0])
            a_black_image_item = np.full_like(an_epoch_single_t_bin_p_x_given_n, np.nan)
            
            self.black_image_items.append(a_black_image_item)

            # img_item.addItem(txt_item)
            # self.grid_layout.addWidget(plot_widget, row, col)
            self.image_items.append((img_item, an_epoch_p_x_given_n, an_epoch_n_bins))
            # self.current_t_bins.append(an_epoch_n_bins)
            self.current_t_bin_index.append(0) ## starting at zero
            self.text_items.append(txt_item)



    def update_frames(self):
        """ updates/animates the PBE frames """        
        
        max_n_t_bins: int = self.max_n_t_bins
        
        for an_epoch_idx in np.arange(self.active_decoded_filter_epochs_result.n_epochs):

            img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]

            an_epoch_t_bin: int = self.current_t_bin_index[an_epoch_idx] ## get current time bin iterator for this index

            if (an_epoch_t_bin < an_epoch_n_bins):
                a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin] ## current time bin posterior

                # Optional normalization for better contrast
                img_item.setImage(a_t_bin_p_x_given_n, autoLevels=True)
                # img_item.setImage(a_t_bin_p_x_given_n.T, autoLevels=False, levels=(0, 1))

                self.current_t_bin_index[an_epoch_idx] += 1
            else:
                ## BEHAVIOR: resets to zero and starts over, allowing them all to operate "out of sync"
                if self.loop_mode.value == AnimationLoopMode.individual_independent_looping.value:
                    self.current_t_bin_index[an_epoch_idx] = 0 ## Resets to zero, I don't think this is desirable

                elif self.loop_mode.value == AnimationLoopMode.common_frame_indexing.value:
                    if max_n_t_bins == self.current_t_bin_index[an_epoch_idx]:
                        ## truly reset to zero (for all of them)
                        self.current_t_bin_index[an_epoch_idx] = 0 ## Resets to zero
                    else:                
                        ## render a black frame and keep accruing:
                        img_item.setImage(self.black_image_items[an_epoch_idx], autoLevels=True)
                        self.current_t_bin_index[an_epoch_idx] += 1
                else:
                    raise NotImplementedError(f'unimplemented looping mode: {self.loop_mode}.')                

            ## update text items:
            txt_item = self.text_items[an_epoch_idx]
            epoch_display_str: str = f"{an_epoch_idx}[{an_epoch_t_bin}/{an_epoch_n_bins}]"
            txt_item.setText(epoch_display_str)
        ## END for an_epoch_idx in np.arange(self.active_decoded_PBE_result.n_epochs)...


    def export_grid_as_gif(self, output_path: str, frame_duration_ms: int = 50, render_passes: int = 5, mode: AnimationExportMode=AnimationExportMode.separate_images):
        """ Exports the entire animated grid as an animated GIF.

        Parameters
        ----------
        output_path : str
            Path to save GIF.
        frame_duration_ms : int
            Duration per frame in milliseconds.
        render_passes : int
            Number of Qt event processing passes per frame before grab (more = smoother capture).



        Usage:

            from pyphoplacecellanalysis.Pho2D.PyQtPlots.posterior2D_animated_grid import AnimatedLoopingPosteriorViewer, AnimatedLoopingPosteriorGraphicsGridViewer, AnimationExportMode

            laps_viewer_app = pg.mkQApp('AnimatedLoopingPosteriorViewer - Laps') # QtWidgets.QApplication(sys.argv)
            # laps_viewer: AnimatedLoopingPosteriorViewer = AnimatedLoopingPosteriorViewer(a_masked_decoded_laps_epochs_result)
            laps_viewer: AnimatedLoopingPosteriorGraphicsGridViewer = AnimatedLoopingPosteriorGraphicsGridViewer(a_masked_decoded_laps_epochs_result)
            # viewer = AnimatedLoopingPosteriorViewer(active_decoded_PBE_result, xbin=a_decoder.xbin, ybin=a_decoder.ybin, drop_below_threshold=None)
            laps_viewer.resize(1000, 450)
            laps_viewer.show()


            ## Export to a single image .gif:
            animated_epochs_export_folder = curr_active_pipeline.get_output_path().joinpath('videos', 'Laps')
            animated_epochs_export_folder.mkdir(exist_ok=True)
            animated_epochs_export_path = animated_epochs_export_folder.joinpath('single_file.gif')
            laps_viewer.export_grid_as_gif(output_path=animated_epochs_export_path, render_passes=5, mode=AnimationExportMode.single_image)

            ## Export to separate .gif images for each epoch:
            animated_epochs_export_folder = curr_active_pipeline.get_output_path().joinpath('videos', 'Laps')
            animated_epochs_export_folder.mkdir(exist_ok=True)
            exported_image_paths = laps_viewer.export_grid_as_gif(output_path=animated_epochs_export_folder, mode=AnimationExportMode.separate_images)


        """
        import imageio
        from pyqtgraph.Qt import QtGui

        frame_duration_seconds: float = frame_duration_ms / 1000.0

        if isinstance(output_path, str):
            output_path = Path(output_path)

        # Global min/max across all epochs so colormap is consistent (smoother than per-frame levels)
        all_data = np.concatenate([p for _, p, _ in self.image_items], axis=2)
        gmin, gmax = float(np.nanmin(all_data)), float(np.nanmax(all_data))
        if gmax <= gmin:
            gmax = gmin + 1.0
        levels = (gmin, gmax)

        self.timer.stop()
        self.current_t_bin_index = np.zeros(self.active_decoded_filter_epochs_result.n_epochs, dtype='uint16') ## reset to zeros
        original_t_bins = self.current_t_bin_index.copy()

        if mode.value == AnimationExportMode.single_image.value:
            frames = []
            max_n_bins: int = np.max(self.active_decoded_filter_epochs_result.nbins)
            for global_frame_idx in range(max_n_bins):

                ## iterate through each epoch
                for an_epoch_idx in np.arange(self.active_decoded_filter_epochs_result.n_epochs):
                    img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]
                    an_epoch_t_bin = global_frame_idx if (global_frame_idx < an_epoch_n_bins) else 0
                    a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin]
                    img_item.setImage(a_t_bin_p_x_given_n, autoLevels=False, levels=levels)

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

            self.current_t_bin_index = original_t_bins

            try:
                imageio.mimsave(output_path, frames, format="GIF-PIL", duration=frame_duration_seconds, loop=0, quantizer="nq", palettesize=256)
            except Exception:
                imageio.mimsave(output_path, frames, duration=frame_duration_seconds, loop=0)

            print(f"Exported GIF to: {output_path}")

        elif mode.value == AnimationExportMode.separate_images.value:
            ## iterate through each epoch
            output_paths = []
            for an_epoch_idx in np.arange(self.active_decoded_filter_epochs_result.n_epochs):
                curr_epoch_label: str = f'epoch[{an_epoch_idx}]'
                print(f'processing epoch: {curr_epoch_label}...')
                curr_output_path = output_path.joinpath(f"{curr_epoch_label}.gif")
                frames = [] ## reset frames for each epoch:
                img_item, an_epoch_p_x_given_n, an_epoch_n_bins = self.image_items[an_epoch_idx]
                for an_epoch_t_bin in np.arange(an_epoch_n_bins):
                    a_t_bin_p_x_given_n = an_epoch_p_x_given_n[:, :, an_epoch_t_bin]
                    img_item.setImage(a_t_bin_p_x_given_n, autoLevels=False, levels=levels)
                    QtWidgets.QApplication.processEvents() ## needed to refresh the view I think?
                    ## Seems super inefficient:
                    # qpixmap = img_item.grab()
                    qimage = img_item.qimage.convertToFormat(QtGui.QImage.Format_RGBA8888)
                    width, height = qimage.width(), qimage.height()
                    ptr = qimage.bits()
                    ptr.setsize(qimage.byteCount())
                    arr = np.ascontiguousarray(np.array(ptr).reshape(height, width, 4).copy())
                    rgb = np.empty((height, width, 3), dtype=np.uint8)
                    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
                    rgb[:] = (arr[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
                    frames.append(rgb)
                ## for an_epoch_t_bin in np.arange(an_epoch_n_bins):
                print(f'\tattempting to export {len(frames)} frames to GIF at path: {curr_output_path.as_posix()}')
                try:
                    imageio.mimsave(curr_output_path, frames, format="GIF-PIL", duration=frame_duration_seconds, loop=0, quantizer="nq", palettesize=256)
                except Exception:
                    imageio.mimsave(curr_output_path, frames, duration=frame_duration_seconds, loop=0)

                
                print(f"\tExported GIF to: {curr_output_path}")
                output_paths.append(curr_output_path)
            ## END for an_epoch_idx in np.arange(self.active_dec
            print(f'exported {len(output_paths)} total images.')
            return output_paths

        else:
            raise ValueError(f'mode: {mode} is unimplemented!')


# if __name__ == "__main__":
# 	app = pg.mkQApp('AnimatedLoopingPosteriorViewer') # QtWidgets.QApplication(sys.argv)
# 	viewer = AnimatedLoopingPosteriorViewer(active_decoded_PBE_result)
# 	viewer.resize(1200, 900)
# 	viewer.show()
# 	sys.exit(app.exec_())
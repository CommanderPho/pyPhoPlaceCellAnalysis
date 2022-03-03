# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import InteractivePlaceCellConfig, PlottingConfig

from pathlib import Path
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

# Old class Names: VideoOutputModeConfig, PlottingConfig, InteractivePlaceCellConfig

""" 
TODO: See OccupancyPlottingConfig, decide if whether to refactor


"""
class VideoOutputModeConfig(DynamicParameters):
    
    def __init__(self, active_frame_range, video_output_parent_dir, active_is_video_output_mode): 
        super(VideoOutputModeConfig, self).__init__(active_frame_range=active_frame_range, video_output_parent_dir=video_output_parent_dir, active_is_video_output_mode=active_is_video_output_mode)
        self.active_is_video_output_mode = active_is_video_output_mode
        self.active_frame_range = active_frame_range
  
        # Computed variables:
        if video_output_parent_dir is None:
            self.active_video_output_parent_dir = Path('output')
        else:
            self.active_video_output_parent_dir = video_output_parent_dir
    
        self.active_video_output_filename = f'complete_plotting_full_curve_F{self.active_frame_range[0]}_F{self.active_frame_range[-1]}.mp4'
        self.active_video_output_fullpath = self.active_video_output_parent_dir.joinpath(self.active_video_output_filename)
    
    

class PlottingConfig(DynamicParameters):
    def __init__(self, output_subplots_shape=(1,1), output_parent_dir=None, use_age_proportional_spike_scale=False, plotter_type='BackgroundPlotter'): 
        # output_subplots_shape="3|1" means 3 plots on the left and 1 on the right,
        # output_subplots_shape="4/2" means 4 plots on top of 2 at bottom.
        # use_age_proportional_spike_scale: if True, the scale of the recent spikes is inversely proportional to their age.
        super(PlottingConfig, self).__init__(output_subplots_shape=output_subplots_shape, output_parent_dir=output_parent_dir, use_age_proportional_spike_scale=use_age_proportional_spike_scale, plotter_type=plotter_type)
        if output_subplots_shape is None:
            output_subplots_shape = (1,1) # By default, only a single plot is needed
        self.subplots_shape = output_subplots_shape
        if output_parent_dir is None:
            self.active_output_parent_dir = Path('output')
        else:
            self.active_output_parent_dir = output_parent_dir

        self.use_age_proportional_spike_scale = use_age_proportional_spike_scale
        self.plotter_type = plotter_type
        
    @property
    def figure_output_directory(self):
        return self.active_output_parent_dir     

    def get_figure_save_path(self, *args):
        # print('get_figure_save_path(...):')
        args_list = list(args)
        basename = args_list.pop()
        subdirectories = args_list
        # print(f'\tsubdirectories: {subdirectories}\n basename: {basename}')
        curr_parent_out_path = self.active_output_parent_dir.joinpath(*subdirectories)
        # print(f'\t curr_parent_out_path: {curr_parent_out_path}')
        curr_parent_out_path.mkdir(parents=True, exist_ok=True)
        return curr_parent_out_path.joinpath(basename)
    
    
    def change_active_out_parent_dir(self, new_parent):
        self.active_output_parent_dir = new_parent
        return self.active_output_parent_dir
        
        


# class InteractivePlaceCellConfig:
class InteractivePlaceCellConfig(DynamicParameters):
    def __init__(self, active_session_config=None, active_epochs=None, video_output_config=None, plotting_config=None, computation_config=None):
        super(InteractivePlaceCellConfig, self).__init__(active_session_config=active_session_config, active_epochs=active_epochs, video_output_config=video_output_config, plotting_config=plotting_config, computation_config=computation_config)


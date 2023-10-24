# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import InteractivePlaceCellConfig, PlottingConfig
from typing import Optional, List, Dict, Tuple, Union
from attr import define, field, Factory, asdict, astuple
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib.colors import ListedColormap # used in PlottingConfig

from neuropy.core.session.Formats.SessionSpecifications import SessionConfig
from neuropy.core.epoch import NamedTimerange
from neuropy.utils.dynamic_container import DynamicContainer

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

# Old class Names: VideoOutputModeConfig, PlottingConfig, InteractivePlaceCellConfig

@define(slots=False)
class BaseConfig:
    """ 2023-10-24 - Base class to enable successful unpickling from old pre-attrs-based classes (based on `DynamicParameters`) to attrs-based classes.`

    """

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes (_mapping and _keys_at_init). Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        # del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        if ('_mapping' in state) and ('_keys_at_init' in state):
            # unpickling from the old DynamicParameters-based ComputationResult
            print(f'unpickling from old DynamicParameters-based computationResult')
            self.__dict__.update(state['_mapping'])
        else:
             # typical update
            self.__dict__.update(state)



@define(slots=False)
class VideoOutputModeConfig(BaseConfig):
    # Docstring for VideoOutputModeConfig.
    active_frame_range: np.ndarray = field()
    # video_output_parent_dir: Path = field()
    active_video_output_parent_dir: Path = field(default=Path('output'), alias='video_output_parent_dir') # init=False
    active_is_video_output_mode: bool = field(default=False)

    ## Derived
    active_video_output_filename: str = field(init=False)
    active_video_output_fullpath: Path = field(init=False)

    def __attrs_post_init__(self):
        # Computed variables:
        self.active_video_output_filename = f'complete_plotting_full_curve_F{self.active_frame_range[0]}_F{self.active_frame_range[-1]}.mp4'
        self.active_video_output_fullpath = self.active_video_output_parent_dir.joinpath(self.active_video_output_filename)

    @classmethod
    def init_from_params(cls, active_frame_range, video_output_parent_dir, active_is_video_output_mode) -> "VideoOutputModeConfig": 
        _obj = cls(active_frame_range, video_output_parent_dir, active_is_video_output_mode)
        _obj.active_is_video_output_mode = active_is_video_output_mode
        _obj.active_frame_range = active_frame_range

        # Computed variables:
        if video_output_parent_dir is None:
            _obj.active_video_output_parent_dir = Path('output')
        else:
            _obj.active_video_output_parent_dir = video_output_parent_dir

        _obj.active_video_output_filename = f'complete_plotting_full_curve_F{_obj.active_frame_range[0]}_F{_obj.active_frame_range[-1]}.mp4'
        _obj.active_video_output_fullpath = _obj.active_video_output_parent_dir.joinpath(_obj.active_video_output_filename)
        return _obj




@define(slots=False)
class PlottingConfig(BaseConfig):
    # Docstring for PlottingConfig. 

    # output_subplots_shape: tuple = field(default=(1,1))
    # output_parent_dir: Path = field(default=Path('output'))
    subplots_shape: tuple = field(default=(1,1)) # , alias='output_subplots_shape'
    active_output_parent_dir: Path = field(default=Path('output')) # , alias='output_parent_dir'
    use_age_proportional_spike_scale: bool = field(default=False)
    plotter_type: str = field(default='BackgroundPlotter')
    
    ## Typically "derived" properties:
    pf_neuron_identities: Optional[list] = field(default=None)
    pf_sort_ind: Optional[np.ndarray] = field(default=None)
    pf_colors: Optional[np.ndarray] = field(default=None)
    pf_colormap: np.ndarray = field(default=None)
    pf_listed_colormap: Optional[ListedColormap] = field(default=None)
    use_smoothed_maze_rendering: Optional[bool] = field(default=None)


    @property
    def figure_output_directory(self) -> Path:
        return self.active_output_parent_dir     

    def get_figure_save_path(self, *args, enable_creating_directory:bool=True) -> Path:
        """ If no *args are passed just returns the computed parent basepath. """
        # print('get_figure_save_path(...):')
        args_list = list(args)
        if len(args) == 0:
            curr_parent_out_path = self.active_output_parent_dir
            out_path = curr_parent_out_path
        else:            
            basename = args_list.pop()
            subdirectories = args_list
            curr_parent_out_path = self.active_output_parent_dir.joinpath(*subdirectories)
            out_path = curr_parent_out_path.joinpath(basename)
        
        if enable_creating_directory:
            curr_parent_out_path.mkdir(parents=True, exist_ok=True)
        return out_path
    
    
    def change_active_out_parent_dir(self, new_parent) -> Path:
        self.active_output_parent_dir = new_parent
        return self.active_output_parent_dir


    # def __attrs_post_init__(self):
    #     # Computed variables:
    #     if output_subplots_shape is None:
    #         output_subplots_shape = (1,1) # By default, only a single plot is needed
    #     self.subplots_shape = output_subplots_shape
    #     if output_parent_dir is None:
    #         self.active_output_parent_dir = Path('output')
    #     else:
    #         self.active_output_parent_dir = output_parent_dir

    #     self.use_age_proportional_spike_scale = use_age_proportional_spike_scale
    #     self.plotter_type = plotter_type

    
    @classmethod
    def init_from_params(cls, output_subplots_shape=(1,1), output_parent_dir=None, use_age_proportional_spike_scale=False, plotter_type='BackgroundPlotter') -> "PlottingConfig": 
        if output_subplots_shape is None:
            output_subplots_shape = (1,1) # By default, only a single plot is needed
        if output_parent_dir is None:
            active_output_parent_dir = Path('output')
        else:
            active_output_parent_dir = output_parent_dir

        _obj = cls(subplots_shape=output_subplots_shape, active_output_parent_dir=active_output_parent_dir, use_age_proportional_spike_scale=use_age_proportional_spike_scale, plotter_type=plotter_type)
        _obj.subplots_shape = output_subplots_shape
        _obj.use_age_proportional_spike_scale = use_age_proportional_spike_scale
        _obj.plotter_type = plotter_type
        return _obj


@define(slots=False)
class InteractivePlaceCellConfig(BaseConfig):
    """ 
        Initially represented a single computation_config



    """ 

    active_session_config: SessionConfig = field()
    active_epochs: NamedTimerange = field()
    video_output_config: VideoOutputModeConfig = field(default=Factory(VideoOutputModeConfig))
    plotting_config: PlottingConfig = field(default=Factory(PlottingConfig))
    computation_config: DynamicContainer = field(default=Factory(DynamicContainer))
    filter_config: dict = field(default=Factory(dict))



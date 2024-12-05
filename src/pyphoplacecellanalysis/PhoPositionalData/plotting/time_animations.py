from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
from attrs import define, field, Factory

@define(slots=False)
class CustomTimeAnimationRoutine:
    """ Implementors define animations over time, for example changing what renders as the user adjusts the time slider.
    """
    debug_print: bool = field() # default=False

    def on_update_current_window(self, t_start: float, t_stop: float):
        """ called to update the current window. 
        
        """
        if self.debug_print:
            print(f'.on_update_current_window(t_start: {t_start}, t_stop: {t_stop})')

        raise NotImplementedError(f'Implementor must override')
    
    # def __call__(self, param, value):
    #     # called whenever a param is updated with a provided value:
    #     self.kwargs[param] = value
    #     self.update()

    # def update(self):
    #     # This is where you update your plot from the values:

    #     result = pv.Sphere(**self.kwargs)
    #     self.output.copy_from(result)


    #     ## post-delta:
    #     self.long_maze_bg.GetProperty().SetOpacity(self.hidden_track_opacity)
    #     self.short_maze_bg.GetProperty().SetOpacity(self.visible_track_opacity)

    #     return
        

@define(slots=False)
class TrackConfigurationTimeAnimationRoutine(CustomTimeAnimationRoutine):
    """ used to animate the transition between active tracks (long/short) by adjusting the opacity of the 3D track
     NOTE: this is not actually needed because there is a hardcoded implementation built in to `InteractivePyvistaPlotter_MazeRenderingMixin`: see `on_update_current_window_MazeRenderingMixin`
      
    Usage:

        from pyphoplacecellanalysis.PhoPositionalData.plotting.time_animations import TrackConfigurationTimeAnimationRoutine
    
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        custom_track_animatior: TrackConfigurationTimeAnimationRoutine = TrackConfigurationTimeAnimationRoutine(t_start=t_start, t_delta=t_delta, t_end=t_end, 
                long_maze_bg=ipspikesDataExplorer.plots['long_maze_bg'], short_maze_bg=ipspikesDataExplorer.plots['short_maze_bg'],
            )

        
        self.params.custom_track_animatior = TrackConfigurationTimeAnimationRoutine(t_start=self.active_config.plotting_config.t_start, t_delta=self.active_config.plotting_config.t_delta, t_end=self.active_config.plotting_config.t_end, 
            long_maze_bg=self.long_maze_bg, short_maze_bg=self.short_maze_bg,
        )

            
    """
    t_start: float = field()
    t_delta: float = field()
    t_end: float = field()

    long_maze_bg: Any = field()
    short_maze_bg: Any = field()

    hidden_track_opacity: float = field(default=0.1)
    visible_track_opacity: float = field(default=1.0)

    


    def on_update_current_window(self, t_start: float, t_stop: float):
        """ called to update the current window. 
        
        """
        if self.debug_print:
            print(f'.on_update_current_window(t_start: {t_start}, t_stop: {t_stop})')

        if t_start >= self.t_delta:
            ## long track inivisible:
            long_track_opacity: float = self.hidden_track_opacity
            ## short track visible
            short_track_opacity: float = self.visible_track_opacity
        else:
            ## long track visible:
            long_track_opacity: float = self.visible_track_opacity

            if t_stop < self.t_delta:
                ## short track inivisible
                short_track_opacity: float = self.hidden_track_opacity
            else:                
                ## short track visible
                short_track_opacity: float = self.visible_track_opacity


        ## Now we have: long_track_opacity, short_track_opacity
        ## post-delta:
        self.long_maze_bg.GetProperty().SetOpacity(long_track_opacity)
        self.short_maze_bg.GetProperty().SetOpacity(short_track_opacity)




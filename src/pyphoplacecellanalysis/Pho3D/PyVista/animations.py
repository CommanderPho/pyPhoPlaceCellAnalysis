#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho

TODO: REORGANIZE_PLOTTER_SCRIPTS: PyVista

"""
import sys
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import pyvista as pv
import pyvistaqt as pvqt
import numpy as np
from pathlib import Path

# from pyphoplacecellanalysis.PhoPositionalData.plotting.animations import * # make_mp4_from_plotter

## For animation/movie creation, see:
# https://github.com/pyvista/pyvista-support/issues/81


def compute_active_frame_range(desired_start_stop_indicies: Tuple[int, int], desired_playback_duration_sec: int, video_framerate: int):
    """ 

        from pyphoplacecellanalysis.Pho3D.PyVista.animations import compute_active_frame_range

        video_framerate = curr_active_pipeline.active_configs[active_config_name].video_output_config.framerate # fps
        active_frame_range = compute_active_frame_range(desired_start_stop_indicies = (14408, 40741), desired_playback_duration_sec = 45, video_framerate=video_framerate)
        active_frame_range

    """
    

    num_total_indicies_to_cover: int = (desired_start_stop_indicies[-1] - desired_start_stop_indicies[0])
    print(f'num_total_indicies_to_cover: {num_total_indicies_to_cover}')

    required_indicies_per_sec: float = float(num_total_indicies_to_cover) / float(desired_playback_duration_sec)
    print(f'required_indicies_per_sec: {required_indicies_per_sec}')


    total_available_num_frames: int = int(round(float(video_framerate) * float(desired_playback_duration_sec)))
    print(f'total_available_num_frames: {total_available_num_frames}')


    required_indicies_jump_step_per_frame = int(round(num_total_indicies_to_cover / total_available_num_frames))
    print(f'required_indicies_jump_step_per_frame: {required_indicies_jump_step_per_frame}')

    ## final output: active_frame_range
    active_frame_range = np.arange(desired_start_stop_indicies[0], desired_start_stop_indicies[1], required_indicies_jump_step_per_frame)
    active_frame_range

    print(f'num_frames: {len(active_frame_range)}')
    return active_frame_range



## Save out to MP4 Movie
def make_mp4_from_plotter(active_plotter, active_frame_range, update_callback, filename='sphere-shrinking.mp4', framerate=30):
    # Open a movie file
    print('active_frame_range: {}'.format(active_frame_range))
    try:
        # Further file processing goes here
        print('Trying to open mp4 movie file at {}...\n'.format(filename))
        if isinstance(active_plotter, pv.plotting.Plotter):
            active_plotter.show(auto_close=False)  # only necessary for an off-screen movie
        elif isinstance(active_plotter, pvqt.BackgroundPlotter):
            active_plotter.show()
        else:
            print('ERROR: active_plotter is not a Plotter or a BackgroundPlotter! Is it valid?')
            
        active_plotter.open_movie(filename, framerate=framerate)
        # Run through each frame
        active_plotter.write_frame()  # write initial data
        total_number_frames = np.size(active_frame_range)
        print('\t opened. Planning to write {} frames...\n'.format(total_number_frames))
        # Update scalars on each frame
        for i in active_frame_range:
            # print('\t Frame[{} of {}]'.format(i, total_number_frames))
            # Call the provided update_callback function:
            update_callback(i)
            # active_plotter.add_text(f"Iteration: {i}", name='time-label')
            # active_plotter.render()
            active_plotter.write_frame()  # Write this frame

    finally:
        # Be sure to close the plotter when finished
        active_plotter.close()
        print('File reader closed!')
        
    print('done.')
    

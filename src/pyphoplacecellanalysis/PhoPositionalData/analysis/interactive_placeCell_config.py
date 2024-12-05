#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import numpy as np
from pathlib import Path

# refactored to pyphoplacecellanalysis.General.Configs.DynamicConfigs
from pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs import VideoOutputModeConfig, PlottingConfig, InteractivePlaceCellConfig


## For building the configs used to filter the session by epoch: 

# General.Pipeline.Stages.Filtering.FilterablePipelineStage.select_filters
def build_configs(session_config, active_epoch, active_subplots_shape = (1,1)):
    ## Get the config corresponding to this epoch/session settings:
    active_config = InteractivePlaceCellConfig(active_session_config=session_config, active_epochs=active_epoch, video_output_config=None, plotting_config=None) # '3|1    
    active_config.video_output_config = VideoOutputModeConfig.init_from_params(active_frame_range=np.arange(100.0, 120.0), 
                                                              video_output_parent_dir=Path('output', session_config.session_name, active_epoch.name),
                                                              active_is_video_output_mode=False)
    active_config.plotting_config = PlottingConfig.init_from_params(output_subplots_shape=active_subplots_shape,
                                                   output_parent_dir=Path('output', session_config.session_name, active_epoch.name)
                                                  )
    # Make the directories:
    active_config.plotting_config.active_output_parent_dir.mkdir(parents=True, exist_ok=True) # makes the directory if it isn't already there
    return active_config



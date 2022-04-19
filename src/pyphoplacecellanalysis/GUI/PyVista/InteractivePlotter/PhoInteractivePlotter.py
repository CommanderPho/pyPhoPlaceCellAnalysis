#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho

A class wrapper for PyVista's plotter class used to simplify adding interactive playback elements (animations) and storing common state for the purpose of 3D plotting.
"""
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractiveSliderWrapper import InteractiveSliderWrapper
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Model.InterfaceProperties import InterfaceProperties
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.AnimationStateMixin import AnimationStateBaseMixin
# from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter import InterfaceProperties
# from pyphoplacecellanalysis.PhoPositionalData.plotting.spikeAndPositions import InteractiveSliderWrapper
# import InterfaceProperties


## TODO: note that the .add_ui() is what makes this VTK centric
class AnimationStateVTKMixin(AnimationStateBaseMixin):
    """ specialized for VTK """
    def add_ui(self):
        """ VTK A checkbox that decides whether we're playing back at a constant rate or not."""
        self.interactive_checkbox_actor = self.p.add_checkbox_button_widget(self.toggle_animation, value=False, color_on='green')


class PhoInteractivePlotter(AnimationStateVTKMixin):
    """A class wrapper for PyVista's plotter class used to simplify adding interactive playback elements (animations) and storing common state for the purpose of 3D plotting."""
    
    def __init__(self, pyvista_plotter, interactive_timestamp_slider_actor):
        # interactive_timestamp_slider_actor: the slider actor object to use for the interactive slider        
        self.p = pyvista_plotter # The actual plotter object, must be either a pyvista.plotter or pyvistaqt.BackgroundPlotter
        interactive_timestamp_slider_wrapper = InteractiveSliderWrapper(interactive_timestamp_slider_actor)
        self.interface_properties = InterfaceProperties(interactive_timestamp_slider_wrapper)
        self.add_ui()
        # An unused constant-time callback that calls back every so often to perform updates
        self.p.add_callback(self.interface_properties, interval=16)  # to be smooth on 60Hz



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho

A class wrapper for PyVista's plotter class used to simplify adding interactive playback elements (animations) and storing common state for the purpose of 3D plotting.
"""
from typing import Any, Union
from attrs import define, field, Factory
import pyvista as pv
from pyvistaqt import BackgroundPlotter, MultiPlotter


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


@define(slots=False)
class PhoInteractivePlotter(AnimationStateVTKMixin):
    """A class wrapper for PyVista's plotter class used to simplify adding interactive playback elements (animations) and storing common state for the purpose of 3D plotting."""
    p: Union[BackgroundPlotter, MultiPlotter, pv.Plotter] = field()
    interface_properties: InterfaceProperties = field()
    step_size: int = field(default=15)
    animation_callback_interval_milliseconds: int = field(default=16)


    # def __init__(self, pyvista_plotter, interactive_timestamp_slider_actor):
    #     # interactive_timestamp_slider_actor: the slider actor object to use for the interactive slider        
    #     self.p = pyvista_plotter # The actual plotter object, must be either a pyvista.plotter or pyvistaqt.BackgroundPlotter
    #     interactive_timestamp_slider_wrapper = InteractiveSliderWrapper(slider_obj=interactive_timestamp_slider_actor)
    #     self.interface_properties = InterfaceProperties(active_timestamp_slider_wrapper=interactive_timestamp_slider_wrapper, animation_state=False, step_size=1)
    #     self.add_ui()
    #     # An unused constant-time callback that calls back every so often to perform updates
    #     self.p.add_callback(self.interface_properties, interval=16)  # to be smooth on 60Hz


    def __attrs_post_init__(self):
        self.add_ui()
        # An unused constant-time callback that calls back every so often to perform updates
        self.p.add_callback(self.interface_properties, interval=self.animation_callback_interval_milliseconds)  # to be smooth on 60Hz


    @classmethod
    def init_from_plotter_and_slider(cls, pyvista_plotter, interactive_timestamp_slider_actor, step_size: float=15, animation_callback_interval_ms: float=16):
        # interactive_timestamp_slider_actor: the slider actor object to use for the interactive slider        
        # self.p = pyvista_plotter # The actual plotter object, must be either a pyvista.plotter or pyvistaqt.BackgroundPlotter
        
        # self.interface_properties = InterfaceProperties(active_timestamp_slider_wrapper=interactive_timestamp_slider_wrapper, animation_state=False, step_size=1)
        # self.add_ui()
        # # An unused constant-time callback that calls back every so often to perform updates
        # self.p.add_callback(self.interface_properties, interval=16)  # to be smooth on 60Hz
        interactive_timestamp_slider_wrapper = InteractiveSliderWrapper(slider_obj=interactive_timestamp_slider_actor)
        interface_properties = InterfaceProperties(active_timestamp_slider_wrapper=interactive_timestamp_slider_wrapper, animation_state=False, step_size=step_size)
        _obj = cls(p=pyvista_plotter, interface_properties=interface_properties, step_size=step_size, animation_callback_interval_milliseconds=animation_callback_interval_ms)
        return _obj


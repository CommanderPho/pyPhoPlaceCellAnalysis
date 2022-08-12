from ...Pho3D.PyVista.animations import make_mp4_from_plotter
from ...Pho3D.PyVista.camera_manipulation import apply_camera_view, apply_close_perspective_camera_view, apply_close_overhead_zoomed_camera_view
from ...Pho3D.PyVista.gui import customize_default_pyvista_theme, print_controls_helper_text
from ...Pho3D.PyVista.spikeAndPositions import build_active_spikes_plot_data, perform_plot_flat_arena, build_spike_spawn_effect_light_actor #, plot_placefields2D, update_plotVisiblePlacefields2D
#from .placefield import plot_placefields2D, update_plotVisiblePlacefields2D
from .visualization_window import VisualizationWindow
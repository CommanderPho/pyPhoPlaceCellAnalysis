from pathlib import Path
import pyvista as pv
import pyvistaqt as pvqt

from pyphocorehelpers.gui.Qt.widget_positioning_helpers import WidgetPositioningHelpers
from pyphoplacecellanalysis.General.Configs.DynamicConfigs import PlottingConfig, InteractivePlaceCellConfig
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

# from neuropy.utils.mixins.unwrap_placefield_computation_parameters import unwrap_placefield_computation_parameters

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper

from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.ImagePlaneRendering import ImagePlaneRendering

from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractivePlaceCellDataExplorer import InteractivePlaceCellDataExplorer

from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractiveCustomDataExplorer import InteractiveCustomDataExplorer

from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractivePlaceCellTuningCurvesDataExplorer import InteractivePlaceCellTuningCurvesDataExplorer
from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.ConnectionControlsMenuMixin import ConnectionControlsMenuMixin



class Interactive3dDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    
    ## Tuning Curves 3D Plot:
    def _display_3d_interactive_tuning_curves_plotter(computation_result, active_config, **kwargs):
        """ 
        Inputs: {'extant_plotter': None} 
        Outputs: {'ipcDataExplorer', 'plotter', 'pane'}
        
        Optional Keywords:
            override_pf2D: the placefield PfND or PfND_TimeDependent object to use as the source data.
        
        """
        # Panel library based Placefield controls
        from pyphoplacecellanalysis.GUI.Panel.panel_placefield import build_panel_interactive_placefield_visibility_controls
        # Qt-based Placefield controls:
        from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControls.qt_placefield import build_all_placefield_output_panels
        
        panel_controls_mode = kwargs.pop('panel_controls_mode', 'Qt') # valid options are 'Qt', 'Panel', or None
        should_use_separate_window = kwargs.pop('separate_window', True)
        pActiveTuningCurvesPlotter = kwargs.get('extant_plotter', None)
        active_pf2D = kwargs.get('override_pf2D', computation_result.computed_data['pf2D'])
        ipcDataExplorer = InteractivePlaceCellTuningCurvesDataExplorer(active_config, computation_result.sess, active_pf2D, active_config.plotting_config.pf_colors, **({'extant_plotter':None} | kwargs))
        pActiveTuningCurvesPlotter = ipcDataExplorer.plot(pActiveTuningCurvesPlotter) # [2, 17449]
        # Update the ipcDataExplorer's colors for spikes and placefields from its configs on init:
        ipcDataExplorer.on_config_update({neuron_id:a_config.color for neuron_id, a_config in ipcDataExplorer.active_neuron_render_configs_map.items()}, defer_update=False)

        
        # build the output panels if desired:
        if panel_controls_mode == 'Qt':
            # pane: (placefieldControlsContainerWidget, pf_widgets)
            placefieldControlsContainerWidget, pf_widgets = build_all_placefield_output_panels(ipcDataExplorer)
            placefieldControlsContainerWidget.show()
            
            # Adds the placefield controls container widget and each individual pf widget to the ipcDataExplorer.ui in case it needs to reference them later:
            ipcDataExplorer.ui['placefieldControlsContainerWidget'] = placefieldControlsContainerWidget
            
            # Visually align the widgets:
            WidgetPositioningHelpers.align_window_edges(ipcDataExplorer.p, placefieldControlsContainerWidget, relative_position = 'above', resize_to_main=(1.0, None))
            
            # Wrap:
            if not should_use_separate_window:
                active_root_main_widget = ipcDataExplorer.p.window()
                root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(active_root_main_widget, placefieldControlsContainerWidget, title=ipcDataExplorer.data_explorer_name)
            else:
                print(f'Skipping separate window because should_use_separate_window == True')
                root_dockAreaWindow = None
            pane = (root_dockAreaWindow, placefieldControlsContainerWidget, pf_widgets)
            
        elif panel_controls_mode == 'Panel':        
            ### Build Dynamic Panel Interactive Controls for configuring Placefields:
            pane = build_panel_interactive_placefield_visibility_controls(ipcDataExplorer)
        else:
            # no controls
            pane = None
        
        
        return {'ipcDataExplorer': ipcDataExplorer, 'plotter': pActiveTuningCurvesPlotter, 'pane': pane}
            
        

    ## Interactive 3D Spike and Behavior Browser: 
    def _display_3d_interactive_spike_and_behavior_browser(computation_result, active_config, **kwargs):
        """ 
        Inputs: {'extant_plotter': None} 
        Outputs: {'ipspikesDataExplorer', 'plotter'}
        """
        active_config.plotting_config.show_legend = True        
        active_session = computation_result.sess # this is unfiltered, shouldn't be used... actually, it should be filtered. Don't know what's wrong here.

        print(f'computation_result.sess: {computation_result.sess}')
        # try: pActiveInteractivePlaceSpikesPlotter
        # except NameError: pActiveInteractivePlaceSpikesPlotter = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
        pActiveInteractivePlaceSpikesPlotter = kwargs.get('extant_plotter', None)
        ipspikesDataExplorer = InteractivePlaceCellDataExplorer(active_config, computation_result.sess, **({'extant_plotter':None} | kwargs))
        pActiveInteractivePlaceSpikesPlotter = ipspikesDataExplorer.plot(pActivePlotter=pActiveInteractivePlaceSpikesPlotter)
        # Add Connection Controls to the window:
        
        # Setup Connections Menu:
        root_window, menuConnections, actions_dict = ConnectionControlsMenuMixin.try_add_connections_menu(ipspikesDataExplorer.p.app_window) # none of these properties need to be saved directly, as they're accessible via ipspikesDataExplorer.p.app_window.window()
        
        
        return {'ipspikesDataExplorer': ipspikesDataExplorer, 'plotter': pActiveInteractivePlaceSpikesPlotter}

    ## CustomDataExplorer 3D Plotter:
    def _display_3d_interactive_custom_data_explorer(computation_result, active_config, **kwargs):
        """ 
        Inputs: {'extant_plotter': None} 
        Outputs: {'iplapsDataExplorer', 'plotter'}
        """
        active_laps_config = InteractivePlaceCellConfig(active_session_config=computation_result.sess.config, active_epochs=None, video_output_config=None, plotting_config=None) # '3|1    
        active_laps_config.plotting_config = PlottingConfig(output_subplots_shape='1|5', output_parent_dir=Path('output', computation_result.sess.config.session_name, 'custom_laps'))
        # try: pActiveInteractiveLapsPlotter
        # except NameError: pActiveInteractiveLapsPlotter = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
        pActiveInteractiveLapsPlotter = kwargs.get('extant_plotter', None)
        iplapsDataExplorer = InteractiveCustomDataExplorer(active_laps_config, computation_result.sess, **({'extant_plotter':None} | kwargs))
        pActiveInteractiveLapsPlotter = iplapsDataExplorer.plot(pActivePlotter=pActiveInteractiveLapsPlotter)
        return {'iplapsDataExplorer': iplapsDataExplorer, 'plotter': pActiveInteractiveLapsPlotter}



    def _display_3d_image_plotter(computation_result, active_config, **kwargs):
        """ 
        Inputs: {'extant_plotter': None} 
        Outputs: {'plotter'}
        """
        def plot_3d_image_plotter(active_epoch_placefields2D, image_file=r'output\2006-6-07_11-26-53\maze\speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.10\pf2D-Occupancy-maze-odd_laps-speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.png'):
            loaded_image_tex = pv.read_texture(image_file)
            pActiveImageTestPlotter = pvqt.BackgroundPlotter()
            return ImagePlaneRendering.plot_3d_image(pActiveImageTestPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy, loaded_image_tex=loaded_image_tex)
            
        # Texture from file:
        image_file = r'output\2006-6-07_11-26-53\maze\speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.10\pf2D-Occupancy-maze-odd_laps-speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.png'
        pActiveImageTestPlotter = plot_3d_image_plotter(computation_result.computed_data['pf2D'], image_file=image_file)
        return {'plotter': pActiveImageTestPlotter}

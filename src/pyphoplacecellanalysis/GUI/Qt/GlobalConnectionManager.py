# GlobalConnectionManager

from indexed import IndexedOrderedDict
from qtpy import QtCore, QtWidgets, QtGui

class GlobalConnectionManager(QtCore.QObject):
    """ A singleton owned by the QApplication instance that owns connections between widgets/windows and includes tools for discovering widgets to control/be controlled by. """
    _currentInstance = None
    
    def __init__(self, owning_application: QtWidgets.QApplication):
        if GlobalConnectionManager._currentInstance is not None:
            print(f'GlobalConnectionManager already exists! Returning extant instance!')
            self = GlobalConnectionManager._currentInstance
            return
        else:
            print(f'GlobalConnectionManager: does not already exist, creating new instance!')
        
        if owning_application is None or not isinstance(owning_application, QtWidgets.QApplication):
            # app was never constructed is already deleted or is an
            # QCoreApplication/QGuiApplication and not a full QApplication
            raise NotImplementedError
        
        super(GlobalConnectionManager, self).__init__()
        # Setup member variables:
        self._registered_available_drivers = IndexedOrderedDict({})
        self._registered_available_drivables = IndexedOrderedDict({})
  
        self._active_connections = IndexedOrderedDict({})
        
        # Setup internal connections:
        # owning_application.aboutToQuit.connect(self.on_application_quit)

        # Set the class variable to this newly created instance
        GlobalConnectionManager._currentInstance = self

    @property
    def registered_available_drivers(self):
        """ an IndexedOrderedDict of widget/objects that can drive a certain property (currently limited to time or time windows) """
        return self._registered_available_drivers
    @property
    def registered_available_drivables(self):
        """ an IndexedOrderedDict of widgets/objects that can be driven by a driver."""
        return self._registered_available_drivables
    
    @property
    def active_connections(self):
        """ an IndexedOrderedDict of widgets/objects that can be driven by a driver."""
        return self._active_connections
    

    #### ================ Registration Methods:
    def register_driver(self, driver, driver_identifier=None):
        """Registers a new driver object/widget """
        return GlobalConnectionManager.register_control_object(self._registered_available_drivers, driver, driver_identifier) # return the new identifier            
                
    def register_drivable(self, drivable, drivable_identifier=None):
        return GlobalConnectionManager.register_control_object(self._registered_available_drivables, drivable, drivable_identifier) # return the new identifier 
    
    def unregister_object(self, control_object, debug_print=True):
        # unregisters object from both drivers and drivables
        # For Driver list:
        found_driver_key, found_object = GlobalConnectionManager._unregister_object(self._registered_available_drivers, control_object=control_object)
        if found_driver_key is not None:
            print(f'removed object with key {found_driver_key} from drivers list.')
        
        # For Drivable List:
        found_drivable_key, found_object = GlobalConnectionManager._unregister_object(self._registered_available_drivables, control_object=control_object)
        if found_drivable_key is not None:
            print(f'removed object with key {found_drivable_key} from drivers list.')    
        
        return found_driver_key, found_drivable_key
        
    def connect_drivable_to_driver(self, drivable, driver):
        """ attempts to connect the drivable to the driver. 
        drivable/driver can either be a key for a drivable/driver already registered or the drivable/driver itself.
        """
        # Get key for drivable:
        if isinstance(drivable, str):
            drivable_key = drivable
            drivable = self.registered_available_drivables[drivable_key]
        else:
            # already have the object, just find the key:
            drivable_key = GlobalConnectionManager._try_find_object_key(self.registered_available_drivables, control_object=drivable)
        
        # Get Key for driver:
        if isinstance(driver, str):
            driver_key = driver
            driver = self.registered_available_drivers[driver_key]
        else:
            # already have the object, just find the key:
            driver_key = GlobalConnectionManager._try_find_object_key(self.registered_available_drivers, control_object=driver)

        ## Make sure the connection doesn't already exist:
        extant_connection = self.active_connections.get(drivable, None)
        if extant_connection is None:
            new_connection_obj = connect_additional_controlled_plotter(self.spike_raster_plt_2d, controlled_plt=drivable)
            self.active_connections[drivable] = new_connection_obj # add the connection object to the self.active_connections array
            return self.active_connections[drivable]
        else:
            print(f'connection already existed!')
            return extant_connection
                
        ## Make the connection:
        ## Sync ipspikesDataExplorer to raster window:
        extra_interactive_spike_behavior_browser_sync_connection = spike_raster_window.connect_additional_controlled_plotter(controlled_plt=ipspikesDataExplorer)
        # extra_interactive_spike_behavior_browser_sync_connection = _connect_additional_controlled_plotter(spike_raster_window.spike_raster_plt_2d, ipspikesDataExplorer)

    
        #### ================ Access Methods:
    def get_available_drivers(self):
        """ gets a list of the available widgets that could be used to drive a time widget. """
        return self.registered_available_drivers
    
    def get_available_drivables(self):
        """ gets a list of the available widgets that could be driven via a time widget. """
        return self.registered_available_drivables

    #### ================ Utility Methods:
    def disambiguate_driver_name(self, extant_name):
        """ attempts to create a unique name for the driver that doesn't already exist in the dict and return it """
        return GlobalConnectionManager.disambiguate_registered_name(self._registered_available_drivers, extant_name)
    
    def disambiguate_drivable_name(self, extant_name):
        """ attempts to create a unique name for the drivable that doesn't already exist in the dict and return it """
        return GlobalConnectionManager.disambiguate_registered_name(self._registered_available_drivables, extant_name)
    
    #### ================ Slots Methods:
    # @QtCore.Slot()
    # def on_application_quit(self):
    #     print(f'GlobalConnectionManager.on_application_quit')
    #     GlobalConnectionManager._currentInstance = None
        

    #### ================ Static Methods:
    @classmethod
    def disambiguate_registered_name(cls, registraction_dict, extant_name):
        """ attempts to create a unique name for the driver/drivee that doesn't already exist in the dict and return it """
        matching_names_with_prefix = filter(lambda x: x.startswith(extant_name), list(registraction_dict.keys()))
        itr_index = len(matching_names_with_prefix) # get the next number after the previous matching names to build a string like # "RasterPlot2D_1"
        proposed_driver_identifier = f'{extant_name}_{itr_index}'
        # Proposed name shouldn't exist:
        extant_driver_with_identifier = registraction_dict.get(proposed_driver_identifier, None)
        assert extant_driver_with_identifier is None, f"Driver with new name {extant_driver_with_identifier} already exists too!"
        # return the new name
        return proposed_driver_identifier
    


    @classmethod
    def register_control_object(cls, registraction_dict, control_object, control_identifier=None):
        """Registers a new driver or driven object/widget

        Args:
            control_object (_type_): _description_
            control_identifier (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if control_identifier is None:
            control_identifier = control_object.windowName # 'Spike3DRasterWindow'
            
        try:
            extant_driver_index = list(registraction_dict.values()).index(control_object)
            # Driver already exists somewhere in the registered drivers:
            return registraction_dict.keys()[extant_driver_index] # return its key
        except ValueError as e:
            # driver doesn't exist anywhere in the registered drivers:
            pass
        
        extant_driver_with_identifier = registraction_dict.get(control_identifier, None)
        if extant_driver_with_identifier is not None:
            # driver already exists with this identifier:
            # check and see if it's the same object
            if extant_driver_with_identifier == control_object:
                # driver with this key already exists, but it's the same driver, so it's just attempting to be re-registered for some reason. No problem.
                return
            else:
                print(f'driver with key {control_identifier} already exists and is a different object. Disambiguating name...')
                # control_identifier = self.disambiguate_driver_name(control_identifier)
                control_identifier = GlobalConnectionManager.disambiguate_registered_name(registraction_dict, control_identifier)
                print(f'\t proposed_driver_name is now {control_identifier}')
                # now has a unique driver identifier
                
        # register the driver provided:
        registraction_dict[control_identifier] = control_object
        return control_identifier # return the new identifier           
    
    
    @classmethod
    def _try_find_object_key(cls, registraction_dict, control_object):
        # tries to find the key of the object in the provided registration_dict
        found_key = None
        try:
            extant_item_index = list(registraction_dict.values()).index(control_object)
            found_key = registraction_dict.keys()[extant_item_index]
            return found_key
        except ValueError as e:
            pass
        except KeyError as e:
            pass
        return found_key
    
    
    @classmethod
    def _unregister_object(cls, registraction_dict, control_object):
        # unregisters object from both drivers and drivables
        found_key = cls._try_find_object_key(registraction_dict, control_object=control_object)
        found_object = None
        if found_key is not None:
            found_object = registraction_dict.pop(found_key) # pop the key
            ## TODO: tear down any connections that use it.             
        return found_key, found_object
         
         
    #### ================ Static Methods factored out of SyncedTimelineWindowLink.py on 2022-05-25
    @classmethod
    def connect_additional_controlled_plotter(cls, source_spike_raster_plt, controlled_plt):
        """ allow the window to control InteractivePlaceCellDataExplorer (ipspikesDataExplorer) objects;
        source_spike_raster_plt: the spike raster plotter to connect to as the source
        controlled_plt: should be a InteractivePlaceCellDataExplorer object (ipspikesDataExplorer), but can be any function with a valid update_window_start_end @QtCore.Slot(float, float) slot.
        
        Requirements:
            source_spike_raster_plt:
                .spikes_window.active_time_window
                .window_scrolled
            
            controlled_plt:
                .disable_ui_window_updating_controls()
                .update_window_start_end(float, float)
        
        
        Usage:
        
            from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget

            # Build the controlled ipspikesDataExplorer:
            display_output = dict()
            pActiveSpikesBehaviorPlotter = None
            display_output = display_output | curr_active_pipeline.display(DefaultDisplayFunctions._display_3d_interactive_spike_and_behavior_browser, active_config_name, extant_plotter=display_output.get('pActiveSpikesBehaviorPlotter', None)) # Works now!
            ipspikesDataExplorer = display_output['ipspikesDataExplorer']
            display_output['pActiveSpikesBehaviorPlotter'] = display_output.pop('plotter') # rename the key from the generic "plotter" to "pActiveSpikesBehaviorPlotter" to avoid collisions with others
            pActiveSpikesBehaviorPlotter = display_output['pActiveSpikesBehaviorPlotter']

            # Build the contolling raster window:
            spike_raster_window = Spike3DRasterWindowWidget(curr_spikes_df)
            # Call this function to connect them:
            extra_interactive_spike_behavior_browser_sync_connection = connect_additional_controlled_plotter(spike_raster_window.spike_raster_plt_2d, ipspikesDataExplorer)
        
        """
        # Perform Initial (one-time) update from source -> controlled:
        controlled_plt.disable_ui_window_updating_controls() # disable the GUI for manual updates.
        controlled_plt.update_window_start_end(source_spike_raster_plt.spikes_window.active_time_window[0], source_spike_raster_plt.spikes_window.active_time_window[1])
        # Connect to update self when video window playback position changes
        sync_connection = source_spike_raster_plt.window_scrolled.connect(controlled_plt.update_window_start_end)
        return sync_connection

    @classmethod
    def connect_controlled_time_synchornized_plotter(cls, source_spike_raster_plt, controlled_plt):
        """ 
        source_spike_raster_plt: TimeSynchronizedPlotterBase
        
        spike_raster_window.spike_raster_plt_2d
        """
        controlled_plt.on_window_changed(source_spike_raster_plt.spikes_window.active_time_window[0], source_spike_raster_plt.spikes_window.active_time_window[1])
        sync_connection = source_spike_raster_plt.window_scrolled.connect(controlled_plt.on_window_changed) # connect the window_scrolled event to the _on_window_updated function
        return sync_connection


    # Perform Initial (one-time) update from source -> controlled:
    @classmethod
    def connect_additional_controlled_spike_raster_plotter(cls, spike_raster_plt_2d, controlled_spike_raster_plt):
        """ Connect an additional plotter to a source that's driving the update of the data-window:
        
        Requirements:
            source_spike_raster_plt:
                .spikes_window.active_time_window
                .window_scrolled
            
            controlled_spike_raster_plt:
                .spikes_window.update_window_start_end(float, float)
            
        Usage:
            
            spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection = build_spike_3d_raster_with_2d_controls(curr_spikes_df)
            spike_raster_plt_3d_vedo = Spike3DRaster_Vedo(curr_spikes_df, window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None)
            extra_vedo_sync_connection = connect_additional_controlled_spike_raster_plotter(spike_raster_plt_2d, spike_raster_plt_3d_vedo)
        
        """
        controlled_spike_raster_plt.spikes_window.update_window_start_end(spike_raster_plt_2d.spikes_window.active_time_window[0], spike_raster_plt_2d.spikes_window.active_time_window[1])
        # Connect to update self when video window playback position changes
        sync_connection = spike_raster_plt_2d.window_scrolled.connect(controlled_spike_raster_plt.spikes_window.update_window_start_end)
        return sync_connection


### Usesful Examples:


### Checking if application instance exists yet:
# if QtGui.QApplication.instance() is None:
# 	return


### Checking if an object is still alive/extant:
# from ...Qt import isQObjectAlive
#  for k in ViewBox.AllViews:
# 	if isQObjectAlive(k) and getConfigOption('crashWarning'):
# 		sys.stderr.write('Warning: ViewBox should be closed before application exit.\n')
        
# 	try:
# 		k.destroyed.disconnect()
# 	except RuntimeError:  ## signal is already disconnected.
# 		pass
# 	except TypeError:  ## view has already been deleted (?)
# 		pass
# 	except AttributeError:  # PySide has deleted signal
# 		pass
    
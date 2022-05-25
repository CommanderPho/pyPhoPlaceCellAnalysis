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
        found_driver_key, found_object = GlobalConnectionManager.unregister_object(self._registered_available_drivers, control_object=control_object)
        if found_driver_key is not None:
            print(f'removed object with key {found_driver_key} from drivers list.')
        
        # For Drivable List:
        found_drivable_key, found_object = GlobalConnectionManager.unregister_object(self._registered_available_drivables, control_object=control_object)
        if found_drivable_key is not None:
            print(f'removed object with key {found_drivable_key} from drivers list.')
        
        return found_driver_key, found_drivable_key
        
    
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
    def _unregister_object(cls, registraction_dict, control_object):
        # unregisters object from both drivers and drivables
        found_key = None
        found_object = None
        try:
            extant_item_index = list(registraction_dict.values()).index(control_object)
            found_key = registraction_dict.keys()[extant_item_index]
            found_key, found_object = registraction_dict.pop(found_key) # pop the key
            return found_key, found_object
            ## TODO: tear down any connections that use it.             
        except ValueError as e:
            # driver doesn't exist anywhere in the registered drivers:
            pass
        except KeyError as e:
            pass
        return found_key, found_object
         
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
    
# GlobalConnectionManager

from indexed import IndexedOrderedDict
from qtpy import QtCore, QtWidgets, QtGui

class GlobalConnectionManager(QtCore.QObject):
    """ A singleton owned by the QApplication instance that owns connections between widgets/windows and includes tools for discovering widgets to control/be controlled by. """
    _currentInstance = None
    
    def __init__(self, owning_application: QtWidgets.QApplication):
        if GlobalConnectionManager._currentInstance is not None:
            print(f'GlobalConnectionManager already exists! Returning extant instance!')
            return GlobalConnectionManager._currentInstance
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
        
#   items


        # Setup internal connections:
        owning_application.aboutToQuit.connect(self.on_application_quit)

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
    
    def disambiguate_driver_name(self, extant_name):
        """ attempts to create a unique name for the driver that doesn't already exist in the dict and return it """
        return GlobalConnectionManager.disambiguate_registered_name(self._registered_available_drivers, extant_name)
    
    def disambiguate_drivable_name(self, extant_name):
        """ attempts to create a unique name for the drivable that doesn't already exist in the dict and return it """
        return GlobalConnectionManager.disambiguate_registered_name(self._registered_available_drivables, extant_name)
    
        

    def register_driver(self, driver, driver_identifier=None):
        """Registers a new driver object/widget

        Args:
            driver (_type_): _description_
            driver_identifier (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if driver_identifier is None:
            driver_identifier = driver.windowName # 'Spike3DRasterWindow'
            
        try:
            extant_driver_index = list(self._registered_available_drivers.values()).index(driver)
            # Driver already exists somewhere in the registered drivers:
            return self._registered_available_drivers.keys()[extant_driver_index] # return its key
        except ValueError as e:
            # driver doesn't exist anywhere in the registered drivers:
            pass
        
        extant_driver_with_identifier = self._registered_available_drivers.get(driver_identifier, None)
        if extant_driver_with_identifier is not None:
            # driver already exists with this identifier:
            # check and see if it's the same object
            if extant_driver_with_identifier == driver:
                # driver with this key already exists, but it's the same driver, so it's just attempting to be re-registered for some reason. No problem.
                return
            else:
                print(f'driver with key {driver_identifier} already exists and is a different object. Disambiguating name...')
                driver_identifier = self.disambiguate_driver_name(driver_identifier)
                print(f'\t proposed_driver_name is now {driver_identifier}')
                # now has a unique driver identifier
                
        # register the driver provided:
        self._registered_available_drivers[driver_identifier] = driver
        return driver_identifier # return the new identifier            
        
        
    def register_drivable(self, drivable):
        self._registered_available_drivables.append(drivable)
        
    
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
    
    
    def get_available_drivers(self):
        """ gets a list of the available widgets that could be used to drive a time widget. """
        return self.registered_available_drivers
    
    def get_available_drivables(self):
        """ gets a list of the available widgets that could be driven via a time widget. """
        return self.registered_available_drivables


    @QtCore.Slot()
    def on_application_quit(self):
        print(f'GlobalConnectionManager.on_application_quit')
        GlobalConnectionManager._currentInstance = None



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
    
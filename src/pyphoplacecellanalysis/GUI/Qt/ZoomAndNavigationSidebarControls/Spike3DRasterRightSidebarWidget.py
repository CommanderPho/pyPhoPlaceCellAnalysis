# Spike3DRasterRightSidebarWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\ZoomAndNavigationSidebarControls\Spike3DRasterRightSidebarWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic

## IMPORTS:
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphoplacecellanalysis.External.pyqtgraph.widgets.LayoutWidget import LayoutWidget
## For dock widget
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget import NestedDockAreaWidget
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig


## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'Spike3DRasterRightSidebarWidget.ui')

# LayoutWidget

class Spike3DRasterRightSidebarWidget(QtWidgets.QWidget):
    """ A simple container to hold interactive widgets
    
    btnToggleDockManager
    # btnToggleIntervalManager
    
    btnToggleIntervalConfigManager
    btnToggleIntervalTableManager
    btnToggleNeuronVisualConfigManager
    
    btnAddDockTrack
    
    
    btnToggleCollapseExpand
    """
    sigToggleIntervalEpochsDisplayManagerPressed = QtCore.pyqtSignal()
    sigToggleIntervalActiveWindowTableManagerPressed = QtCore.pyqtSignal()
    sigToggleNeuronDisplayConfigManagerPressed = QtCore.pyqtSignal()
    sigToggleDockManagerPressed = QtCore.pyqtSignal()

    
    @property
    def right_sidebar_contents_container(self) -> LayoutWidget:
        try:
            return self.ui.layout_widget # AttributeError: 'Spike3DRasterRightSidebarWidget' object has no attribute 'ui'
        except AttributeError as e:
            ## occurs before class is initialized by uic.loadUi(...)
            pass
        except Exception as e:
            raise e        

    @property
    def right_sidebar_contents_container_dockarea(self) -> NestedDockAreaWidget:
        try:
            return self.ui.dynamic_docked_widget_container
        except AttributeError as e:
            ## occurs before class is initialized by uic.loadUi(...)
            pass
        except Exception as e:
            raise e
        

    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        self.setVisible(False) # collapses and hides the sidebar
        # self.ui.layout_widget # a LayoutWidget

        # self.ui.btnToggleCollapseExpand # a LayoutWidget
        self.ui.layout_widget.setMinimumWidth(200.0)
        
        # ==================================================================================================================== #
        # Build the nested dock areas via NestedDockAreaWidget                                                                 #
        # ==================================================================================================================== #

        # Create the dynamic docked widget container
        self.ui.dynamic_docked_widget_container = NestedDockAreaWidget()
        self.ui.dynamic_docked_widget_container.setObjectName("dynamic_docked_widget_container")
        
        # # Create a layout for the wrapper
        # self.ui.wrapper_layout = pg.QtWidgets.QVBoxLayout() # parent_widget
        # self.ui.wrapper_layout.setSpacing(0)
        # self.ui.wrapper_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add the container to the wrapper layout
        # self.ui.wrapper_layout.addWidget(self.ui.dynamic_docked_widget_container)
        
        ## Add to the main layout widget
        # self.ui.layout_widget.addLayout(self.ui.wrapper_layout)
        
        self.ui.layout_widget.addWidget(self.ui.dynamic_docked_widget_container)
        self.ui.layout_widget.setContentsMargins(0, 0, 0, 0)
    
        self.ui.dock_items = {}  # Store dock items
        
        self.ui.btnToggleDockManager.clicked.connect(self.toggle_dock_manager)
        self.ui.btnToggleIntervalTableManager.clicked.connect(self.toggle_interval_active_window_table_manager)
        self.ui.btnToggleIntervalConfigManager.clicked.connect(self.toggle_interval_visual_configs_manager)
        self.ui.btnToggleNeuronVisualConfigManager.clicked.connect(self.toggle_neuron_configs_manager)
        

        # self.setVisible(True) # shows the sidebar

    def toggle_dock_manager(self):
        """ Toggles the visibility of the dock manager """
        print("toggle_dock_manager()")
        self.sigToggleDockManagerPressed.emit()
        
    # def toggle_interval_manager(self):
    #     """ Toggles the visibility of the interval manager """
    #     print("toggle_interval_manager()")
    #     self.sigToggleDockManagerPressed.emit()
    

    def toggle_interval_visual_configs_manager(self):
        """ Toggles the visibility of the interval manager """
        print("toggle_interval_visual_configs_manager()")
        self.sigToggleIntervalEpochsDisplayManagerPressed.emit()
        

    def toggle_interval_active_window_table_manager(self):
        """ Toggles the visibility of the interval manager """
        print("toggle_interval_active_window_table_manager()")
        self.sigToggleIntervalActiveWindowTableManagerPressed.emit()
        

    def toggle_neuron_configs_manager(self):
        """ Toggles the visibility of the interval manager """
        print("toggle_neuron_configs_manager()")
        self.sigToggleNeuronDisplayConfigManagerPressed.emit()



class SpikeRasterRightSidebarOwningMixin:
    """ renders the UI 

        Currently used in Spike3DRasterWindowWidget to implement the right sidebar
    """
    
    @property
    def right_sidebar_widget(self) -> Spike3DRasterRightSidebarWidget:
        return self.ui.rightSideContainerWidget
    
    @property
    def right_sidebar_contents_container(self) -> LayoutWidget:
        return self.right_sidebar_widget.ui.layout_widget
    
    @property
    def right_sidebar_contents_container_dockarea(self) -> NestedDockAreaWidget:
        return self.right_sidebar_widget.ui.dynamic_docked_widget_container
    
    
    def toggle_right_sidebar(self):
        is_visible = self.right_sidebar_widget.isVisible()
        self.right_sidebar_widget.setVisible(not is_visible) # collapses and hides the sidebar
        # self.right_sidebar_widget.setVisible(True) # shows the sidebar

    @pyqtExceptionPrintingSlot(bool)
    def set_right_sidebar_visibility(self, is_visible:bool):
        """ called when the right sidebar is made Visible or non_visible"""
        self.right_sidebar_widget.setVisible(is_visible) 

    # @pyqtExceptionPrintingSlot()
    # def on_toggle_interval_manager(self):
    #     """ Toggles the visibility of the interval manager """
    #     print(f'SpikeRasterRightSidebarOwningMixin.on_toggle_interval_manager()')
    #     self.build_epoch_intervals_visual_configs_widget()
    #     print(f'\tdone.')


    @pyqtExceptionPrintingSlot()
    def on_toggle_interval_visual_configs_manager(self):
        """ Toggles the visibility of the interval manager """
        print(f'SpikeRasterRightSidebarOwningMixin.on_toggle_interval_visual_configs_manager()')
        self.build_epoch_intervals_visual_configs_widget()
        print(f'\tdone.')
        

    @pyqtExceptionPrintingSlot()
    def on_toggle_interval_active_window_tables_manager(self):
        """ Toggles the visibility of the interval manager """
        print(f'SpikeRasterRightSidebarOwningMixin.on_toggle_interval_active_window_tables_manager()')
        self.on_update_right_sidebar_visible_interval_info_tables()
        print(f'\tdone.')

    @pyqtExceptionPrintingSlot()
    def on_toggle_neuron_visual_configs_manager(self):
        """ Toggles the visibility of the interval manager """
        print(f'SpikeRasterRightSidebarOwningMixin.on_toggle_neuron_visual_configs_manager()')
        self.build_neuron_visual_configs_widget()
        print(f'\tdone.')

    @pyqtExceptionPrintingSlot()
    def on_toggle_dock_manager(self):
        """ Toggles the visibility of the dock manager """
        print(f'SpikeRasterRightSidebarOwningMixin.on_toggle_dock_manager()')
        self.build_dock_area_managing_tree_widget()
        print(f'\tdone.')

            

    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_connectSignals(self, right_side_bar_controls: Spike3DRasterRightSidebarWidget):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        right_side_bar_connections = []
        # right_side_bar_connections.append(right_side_bar_controls.animation_time_step_changed.connect(self.on_animation_timestep_valueChanged))
        # right_side_bar_connections.append(right_side_bar_controls.temporal_zoom_factor_changed.connect(self.on_temporal_zoom_factor_valueChanged))
        # right_side_bar_connections.append(right_side_bar_controls.render_window_duration_changed.connect(self.on_render_window_duration_valueChanged))        
        right_side_bar_connections.append(right_side_bar_controls.sigToggleDockManagerPressed.connect(self.on_toggle_dock_manager))
        right_side_bar_connections.append(right_side_bar_controls.sigToggleIntervalActiveWindowTableManagerPressed.connect(self.on_toggle_interval_active_window_tables_manager))
        right_side_bar_connections.append(right_side_bar_controls.sigToggleIntervalEpochsDisplayManagerPressed.connect(self.on_toggle_interval_visual_configs_manager))
        right_side_bar_connections.append(right_side_bar_controls.sigToggleNeuronDisplayConfigManagerPressed.connect(self.on_toggle_neuron_visual_configs_manager))
        return right_side_bar_connections
        


    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        # CALLED:
         # Set Initial values:
        # right_side_bar_controls = self.ui.rightSidebarWidget
        
        # right_side_bar_controls.ui.verticalSliderZoom.blockSignals(True)
        # right_side_bar_controls.ui.spinAnimationTimeStep.blockSignals(True)
        # right_side_bar_controls.ui.spinTemporalZoomFactor.blockSignals(True)
        # right_side_bar_controls.ui.spinRenderWindowDuration.blockSignals(True)
        
        # right_side_bar_controls.ui.verticalSliderZoom.setValue(round(self.temporal_zoom_factor))
        # right_side_bar_controls.ui.spinAnimationTimeStep.setValue(self.animation_time_step)
        # right_side_bar_controls.ui.spinTemporalZoomFactor.setValue(round(self.temporal_zoom_factor))
        # right_side_bar_controls.ui.spinRenderWindowDuration.setValue(self.render_window_duration)

        # right_side_bar_controls.ui.verticalSliderZoom.blockSignals(False)
        # right_side_bar_controls.ui.spinAnimationTimeStep.blockSignals(False)
        # right_side_bar_controls.ui.spinTemporalZoomFactor.blockSignals(False)
        # right_side_bar_controls.ui.spinRenderWindowDuration.blockSignals(False)
        pass
        
        
    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # TODO: NOT CALLED
        pass

    @pyqtExceptionPrintingSlot(float, float)
    def SpikeRasterRightSidebarOwningMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        # TODO: NOT CALLED
        pass


## Start Qt event loop
if __name__ == '__main__':
    app = pg.mkQApp()
    widget = Spike3DRasterRightSidebarWidget()
    widget.show()
    sys.exit(app.exec_())

from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import warnings
from attrs import define, field, Factory
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop

def debug_widget_geometry(a_widget, widget_name="Unknown"):
    """Print comprehensive debug information about a DockLabel to diagnose layout issues."""
    widget_type = type(a_widget)
    print(f"\n--- {widget_type} Debug: {widget_name} ---")
    
    # Basic geometry info
    print(f"Position: ({a_widget.x()}, {a_widget.y()})")
    print(f"Size: {a_widget.width()} × {a_widget.height()}")
    print(f"Geometry: {a_widget.geometry()}")
    print(f"Content rect: {a_widget.rect()}")
    print(f"Size hint: {a_widget.sizeHint()}")
    print(f"Minimum size hint: {a_widget.minimumSizeHint()}")
    
    # Orientation and layout issues
    print(f"Orientation: {a_widget.orientation}")
    # print(f"Dim state: {a_widget.dim}")
    print(f"Size policy: {a_widget.sizePolicy().horizontalPolicy()}, {a_widget.sizePolicy().verticalPolicy()}")
    print(f"stretch: {a_widget.stretch()}")
    print(f"container: {a_widget.container()}")
    
    # Visibility and enablement
    print(f"Is visible: {a_widget.isVisible()}")
    print(f"Is enabled: {a_widget.isEnabled()}")
    print(f"Is shown: {not a_widget.isHidden()}")
    
    # Parent and layout context
    print(f"Parent type: {type(a_widget.parent()).__name__}")
    
    print("--- End Debug Info ---\n")
    




def debug_print_dock(a_dock, widget_name="Unknown"):
    """Print comprehensive debug information about a DockLabel to diagnose layout issues."""
    print(f"\n--- Dock Debug: {widget_name} ---")
    
    # Basic geometry info
    print(f"Position: ({a_dock.x()}, {a_dock.y()})")
    print(f"Size: {a_dock.width()} × {a_dock.height()}")
    print(f"Geometry: {a_dock.geometry()}")
    print(f"Content rect: {a_dock.rect()}")
    print(f"Size hint: {a_dock.sizeHint()}")
    print(f"Minimum size hint: {a_dock.minimumSizeHint()}")
    
    # Orientation and layout issues
    print(f"Orientation: {a_dock.orientation}")
    # print(f"Dim state: {a_widget.dim}")
    print(f"Size policy: {a_dock.sizePolicy().horizontalPolicy()}, {a_dock.sizePolicy().verticalPolicy()}")
    print(f"stretch: {a_dock.stretch()}")
    print(f"container: {a_dock.container()}")
    print(f"title: {a_dock.title()}")
    
    # Visibility and enablement
    print(f"Is visible: {a_dock.isVisible()}")
    print(f"Is enabled: {a_dock.isEnabled()}")
    print(f"Is shown: {not a_dock.isHidden()}")
    
    # Parent and layout context
    print(f"Parent type: {type(a_dock.parent()).__name__}")
    
    print("--- End Debug Info ---\n")





@define(slots=False)
class DockDisplayConfig(object):
    """Holds the display and configuration options for a Dock, such as how to format its title bar (color and font), whether it's closable, etc."""
    showCloseButton: bool = field(default=True)
    showCollapseButton: bool = field(default=False)
    showGroupButton: bool = field(default=False)
    showOrientationButton: bool = field(default=False)
    
    hideTitleBar: bool = field(default=False)
    fontSize: str = field(default='10px')
    corner_radius: str = field(default='2px')
    # fontSize: str = field(default='10px')
    custom_get_stylesheet_fn: Callable = field(default=None) #(self, orientation, is_dim)
    _orientation: Optional[str] = field(default=None, alias="orientation", metadata={'valid_values': [None, 'auto', 'vertical', 'horizontal']}) # alias="orientation" just refers to the initializer, it doesn't interfere with the @property

    additional_metadata: Dict = field(default=Factory(dict)) ## optional metadata


    @property
    def orientation(self) -> str:
        """The orientation property."""
        return (self._orientation or 'horizontal')   
    @orientation.setter
    def orientation(self, value):
        self._orientation = value

    @property
    def shouldAutoOrient(self) -> bool:
        """ Whether the dock should auto-orient based on the aspect ratioy."""
        if self.orientation is None:
            return True
        else:
            return (self.orientation == 'auto') ## only if "auto" instead of ['vertical', 'horizontal']
    @shouldAutoOrient.setter
    def shouldAutoOrient(self, value: bool):
        if value:
            self.orientation = 'auto' # only if True: change the orientation to 'auto'
        else:
            print(f"WARN: setting shouldAutoOrient to False does nothing, as we do not know which concrete value (['vertical', 'horizontal']) is wanted.")


    def get_colors(self, orientation, is_dim):
        """ point of customization """
        if is_dim:
            fg_color = '#aaa'
            bg_color = '#44a'
            border_color = '#339'
        else:
            fg_color = '#fff'
            bg_color = '#66c'
            border_color = '#55B'
        return fg_color, bg_color, border_color
    
    def get_stylesheet(self, orientation, is_dim):
        """ Gets the appropriate stylesheet for the given state. This method can be overriden to customize the appearance 
        
        Usage:
            updated_stylesheet = config.get_stylesheet(self, orientation=self.orientation, is_dim=self.dim)
            
        """            
        fg_color, bg_color, border_color = self.get_colors(orientation, is_dim)

        if orientation == 'vertical':
            # """
            #     padding-top: 3px;
            #     padding-bottom: 3px;
            # """
            return """DockLabel {
                background-color : %s;
                color : %s;
                border-top-right-radius: 0px;
                border-top-left-radius: %s;
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: %s;
                border-width: 0px;
                border-right: 2px solid %s;
                padding-top: 0px;
                padding-bottom: 1px;
                font-size: %s;
            }""" % (bg_color, fg_color, self.corner_radius, self.corner_radius, border_color, self.fontSize)
            
        else:
            return """DockLabel {
                background-color : %s;
                color : %s;
                border-top-right-radius: %s;
                border-top-left-radius: %s;
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: 0px;
                border-width: 0px;
                border-bottom: 2px solid %s;
                padding-left: 3px;
                padding-right: 3px;
                font-size: %s;
            }""" % (bg_color, fg_color, self.corner_radius, self.corner_radius, border_color, self.fontSize)

    
buttonIconOrientation = {'horizontal': QtWidgets.QStyle.StandardPixmap.SP_ToolBarHorizontalExtensionButton, 'vertical': QtWidgets.QStyle.StandardPixmap.SP_ToolBarVerticalExtensionButton}
buttonIconShade = {'shade': QtWidgets.QStyle.StandardPixmap.SP_TitleBarShadeButton, 'unshade': QtWidgets.QStyle.StandardPixmap.SP_TitleBarUnshadeButton}



class Dock(QtWidgets.QWidget, DockDrop):
    """ 
    
    self.widgetArea: QWidget - the main container for the contents of the dock, aside from the titlebar at the top
    
    """
    sigStretchChanged = QtCore.Signal()
    sigClosed = QtCore.Signal(object)
    ## passthrough signals from `DockLabel`
    # sigClicked = QtCore.Signal(object, object)
    sigCollapseClicked = QtCore.Signal(object)
    sigGroupClicked = QtCore.Signal(object)
    sigToggleOrientationClicked = QtCore.Signal(object, bool)
    

    @property
    def config(self) -> Optional[DockDisplayConfig]:
        """The config property."""
        if self.label is None:
            return None
        return self.label.config
    @config.setter
    def config(self, value: Optional[DockDisplayConfig]):
        assert self.label is not None
        self.label.config = value


    def __init__(self, name, area=None, size=(10, 10), widget=None, hideTitle=False, autoOrientation=True, display_config:Optional[DockDisplayConfig]=None, **kwargs): # , closable=False, fontSize="12px"
        QtWidgets.QWidget.__init__(self)
        DockDrop.__init__(self)
        self._container = None
        self._name = name
        self.area = area
        # self.label = DockLabel(name, self, closable, fontSize)
        
        if display_config is None:
            print(f"WARNING: Dock.__init__(...): display_config is None... using old-mode fallback. This will be eventually depricated.")
            display_config = DockDisplayConfig(showCloseButton=kwargs.get('closable', False), fontSize=kwargs.get('fontSize', "10px"), corner_radius='2px', orientation='horizontal')
            if autoOrientation:
                display_config.orientation = 'auto' # only if True: change the orientation to 'auto'

            # raise NotImplementedError
        else:
            # print(f"WARNING: Dock.__init__(...): display_config is set, so the explicitly passed parameters 'closable' and 'fontSize' will be ignored.")
            pass
        self.label = DockLabel(name, self, display_config)
        if display_config.showCloseButton:
            self.label.sigCloseClicked.connect(self.close)
        if display_config.showCollapseButton:
            self.label.sigCollapseClicked.connect(self.on_collapse_btn_clicked)
        if display_config.showGroupButton:
            self.label.sigGroupClicked.connect(self.on_group_btn_clicked)
        if display_config.showOrientationButton:
            self.label.sigToggleOrientationClicked.connect(self.on_orientation_btn_toggled)
        # Add this line to connect the new rename signal
        self.label.sigRenamed.connect(self.on_renamed)
        
        self.contentsHidden = False
        self.labelHidden = False
        self.moveLabel = True  ## If false, the dock is no longer allowed to move the label.
        # self.autoOrient = autoOrientation
        # self.orientation = 'horizontal'
        self.autoOrient = display_config.shouldAutoOrient
        self.orientation = (display_config.orientation or 'horizontal')
        #self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.topLayout = QtWidgets.QGridLayout()
        self.topLayout.setContentsMargins(0, 0, 0, 0)
        self.topLayout.setSpacing(0)
        self.setLayout(self.topLayout)
        self.topLayout.addWidget(self.label, 0, 1)
        self.widgetArea = QtWidgets.QWidget()
        self.topLayout.addWidget(self.widgetArea, 1, 1)
        self.layout = QtWidgets.QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.widgetArea.setLayout(self.layout)
        self.widgetArea.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.widgets = []
        self.currentRow = 0
        #self.titlePos = 'top'
        self.raiseOverlay()
        self.hStyle = """
        Dock > QWidget {
            border: 1px solid #000;
            border-radius: 5px;
            border-top-left-radius: 0px;
            border-top-right-radius: 0px;
            border-top-width: 0px;
        }"""
        self.vStyle = """
        Dock > QWidget {
            border: 1px solid #000;
            border-radius: 5px;
            border-top-left-radius: 0px;
            border-bottom-left-radius: 0px;
            border-left-width: 0px;
        }"""
        self.nStyle = """
        Dock > QWidget {
            border: 1px solid #000;
            border-radius: 5px;
        }"""
        self.dragStyle = """
        Dock > QWidget {
            border: 4px solid #00F;
            border-radius: 5px;
        }"""
        self.setAutoFillBackground(False)
        self.widgetArea.setStyleSheet(self.hStyle)

        self.setStretch(*size)

        if widget is not None:
            self.addWidget(widget)

        if hideTitle:
            self.hideTitleBar()


    def debug_print(self, widget_name: str="Unknown"):
        """Print comprehensive debug information about a Dock to diagnose layout issues."""
        return debug_print_dock(self, widget_name=widget_name)
    

    def implements(self, name=None):
        if name is None:
            return ['dock']
        else:
            return name == 'dock'

    def setStretch(self, x=None, y=None):
        """
        Set the 'target' size for this Dock.
        The actual size will be determined by comparing this Dock's
        stretch value to the rest of the docks it shares space with.
        """
        if x is None:
            x = 0
        if y is None:
            y = 0
        self._stretch = (x, y)
        self.sigStretchChanged.emit()
        
    def stretch(self):
        return self._stretch

    def hideTitleBar(self):
        """
        Hide the title bar for this Dock.
        This will prevent the Dock being moved by the user.
        """
        self.label.hide()
        self.labelHidden = True
        if 'center' in self.allowedAreas:
            self.allowedAreas.remove('center')
        self.updateStyle()
        # Create a special tiny button that can restore the title bar
        if not hasattr(self, 'restoreTitleButton'):
            self.restoreTitleButton = QtWidgets.QToolButton(self)
            self.restoreTitleButton.setIcon(QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton))
            self.restoreTitleButton.setToolTip("Show title bar")
            self.restoreTitleButton.clicked.connect(self.showTitleBar)
            self.restoreTitleButton.setMaximumSize(16, 16)
            self.restoreTitleButton.setStyleSheet("""
                QToolButton {
                    background-color: rgba(100, 100, 100, 60);
                    border-radius: 2px;
                }
                QToolButton:hover {
                    background-color: rgba(100, 100, 100, 100);
                }
            """)
            
        # Position in top-right corner and make visible
        self.restoreTitleButton.show()
        self.restoreTitleButton.raise_()
        




    def showTitleBar(self):
        """
        Show the title bar for this Dock.
        """
        self.label.show()
        self.labelHidden = False
        self.allowedAreas.add('center')
        self.updateStyle()

        # Hide the restore button if it exists
        if hasattr(self, 'restoreTitleButton'):
            self.restoreTitleButton.hide()


    def toggleContentVisibility(self):
        """ toggles the visibility of the contents (everything except the title bar) for this Dock.
        """
        new_is_hidden: bool = (not self.contentsHidden)
        if new_is_hidden:
            ## now hidden
            self.hideContents()
        else:
            ## now visible
            self.showContents()

    def setContentVisibility(self, is_visible: bool):
        """ toggles the visibility of the contents (everything except the title bar) for this Dock.
        """
        new_is_hidden: bool = (not is_visible)
        if new_is_hidden:
            ## now hidden
            self.hideContents()
        else:
            ## now visible
            self.showContents()

    def hideContents(self):
        """
        Hide the contents (everything except the title bar) for this Dock.
        """
        print(f'hideContents()')
        self.widgetArea.hide()
        self.contentsHidden = True
        self.updateStyle()
        print(f'\tdone.')
        
    def showContents(self):
        """
        Show the contents (everything except the title bar) for this Dock.
        """
        print(f'showContents()')
        self.widgetArea.show()
        self.contentsHidden = False
        self.updateStyle()
        print(f'\tdone.')
        



    def title(self):
        """
        Gets the text displayed in the title bar for this dock.
        """
        return self.label.text()

    def setTitle(self, text):
        """
        Sets the text displayed in title bar for this Dock.
        """
        self.label.setText(text)

    def setOrientation(self, o='auto', force=False):
        """
        Sets the orientation of the title bar for this Dock.
        Must be one of 'auto', 'horizontal', or 'vertical'.
        By default ('auto'), the orientation is determined
        based on the aspect ratio of the Dock.
        """
        # setOrientation may be called before the container is set in some cases
        # (via resizeEvent), so there's no need to do anything here until called
        # again by containerChanged
        if self.container() is None:
            return

        if o == 'auto' and self.autoOrient:
            if self.container().type() == 'tab':
                o = 'horizontal'
            elif self.width() > self.height()*1.5:
                o = 'vertical'
            else:
                o = 'horizontal'
        if force or self.orientation != o:
            self.orientation = o
            self.label.setOrientation(o)
            self.updateStyle()

    def updateStyle(self):
        ## updates orientation and appearance of title bar
        if self.labelHidden:
            self.widgetArea.setStyleSheet(self.nStyle)
        elif self.orientation == 'vertical':
            self.label.setOrientation('vertical')
            if self.moveLabel:
                self.topLayout.addWidget(self.label, 1, 0)
            self.widgetArea.setStyleSheet(self.vStyle)
        else:
            self.label.setOrientation('horizontal')
            if self.moveLabel:
                self.topLayout.addWidget(self.label, 0, 1)
            self.widgetArea.setStyleSheet(self.hStyle)

    def resizeEvent(self, ev):
        self.setOrientation()
        self.resizeOverlay(self.size())
        super().resizeEvent(ev) ## call super
        

    def name(self):
        return self._name

    def addWidget(self, widget, row=None, col=0, rowspan=1, colspan=1):
        """
        Add a new widget to the interior of this Dock.
        Each Dock uses a QGridLayout to arrange widgets within.
        """
        if row is None:
            row = self.currentRow
        self.currentRow = max(row+1, self.currentRow)
        self.widgets.append(widget)
        self.layout.addWidget(widget, row, col, rowspan, colspan)
        self.raiseOverlay()
        
    def startDrag(self):
        self.drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        self.drag.setMimeData(mime)
        self.widgetArea.setStyleSheet(self.dragStyle)
        self.update()
        action = self.drag.exec() if hasattr(self.drag, 'exec') else self.drag.exec_()
        self.updateStyle()

    def float(self):
        self.area.floatDock(self)
            
    def container(self):
        return self._container

    def containerChanged(self, c):
        if self._container is not None:
            # ask old container to close itself if it is no longer needed
            self._container.apoptose()
        self._container = c
        if c is None:
            self.area = None
        else:
            self.area = c.area
            if c.type() != 'tab':
                self.moveLabel = True
                self.label.setDim(False)
            else:
                self.moveLabel = False
                
            self.setOrientation(force=True)

    def raiseDock(self):
        """If this Dock is stacked underneath others, raise it to the top."""
        self.container().raiseDock(self)

    def close(self):
        """Remove this dock from the DockArea it lives inside."""
        if self._container is None:
            warnings.warn(f"Cannot close dock {self} because it is not open.", RuntimeWarning, stacklevel=2)
            return

        self.setParent(None)
        QtWidgets.QLabel.close(self.label)
        self.label.setParent(None)
        self._container.apoptose()
        self._container = None
        self.sigClosed.emit(self)

    def __repr__(self):
        return "<Dock %s %s>" % (self.name(), self.stretch())

    ## PySide bug: We need to explicitly redefine these methods
    ## or else drag/drop events will not be delivered.
    def dragEnterEvent(self, *args):
        DockDrop.dragEnterEvent(self, *args)

    def dragMoveEvent(self, *args):
        DockDrop.dragMoveEvent(self, *args)

    def dragLeaveEvent(self, *args):
        DockDrop.dragLeaveEvent(self, *args)

    def dropEvent(self, *args):
        DockDrop.dropEvent(self, *args)

    ## pass-through events:
    def on_collapse_btn_clicked(self):
        """Remove this dock from the DockArea it lives inside."""
        self.toggleContentVisibility()
        self.label.updateCollapseButtonStyle(is_collapse_active=self.contentsHidden)
        self.sigCollapseClicked.emit(self)
        
    def on_group_btn_clicked(self):
        """Remove this dock from the DockArea it lives inside."""
        self.sigGroupClicked.emit(self)
        
    def on_orientation_btn_toggled(self, is_checked):
        """Remove this dock from the DockArea it lives inside."""
        if is_checked:
            print(f'changing to horizontal.')
            new_icon_name = buttonIconOrientation['horizontal']
        else:
            print(f'changing to vertical')
            new_icon_name = buttonIconOrientation['vertical']
        
        icon = QtWidgets.QApplication.style().standardIcon(new_icon_name)
        self.label.orientationButton.setIcon(icon)
        self.sigToggleOrientationClicked.emit(self)
        

    # ==================================================================================================================== #
    # Context Menu Providing                                                                                               #
    # ==================================================================================================================== #
    def buildContextMenu(self):
        """
        Build and return a context menu for this dock.
        This method can be overridden by subclasses to customize the context menu.
        Widgets in the dock can also add their own menu items by connecting to 
        the label's sigContextMenuRequested signal.
        
        Returns:
            QtWidgets.QMenu: The context menu to display
        """
        menu = QtWidgets.QMenu()
        
        # Create standard actions
        showTitleAction = menu.addAction("Show title bar")
        showTitleAction.triggered.connect(self.showTitleBar)
        menu.addSeparator()

        renameAction = menu.addAction("Rename dock...")
        renameAction.triggered.connect(lambda: self.label.promptRename())
        
        toggleOrientationAction = menu.addAction("Toggle orientation")
        toggleOrientationAction.triggered.connect(
            lambda: self.setOrientation('vertical' if self.orientation == 'horizontal' else 'horizontal', force=True)
        )
        
        # Add Close action if close button is available
        if self.config.showCloseButton:
            closeAction = menu.addAction("Close dock")
            closeAction.triggered.connect(self.close)
        
        # Add buttons visibility submenu
        buttonVisibilityMenu = menu.addMenu("Show dock buttons")
        
        # Close button toggle
        showCloseAction = buttonVisibilityMenu.addAction("Close button")
        showCloseAction.setCheckable(True)
        showCloseAction.setChecked(self.config.showCloseButton)
        showCloseAction.toggled.connect(lambda checked: self.updateButtonVisibility('close', checked))
        
        # Collapse button toggle
        showCollapseAction = buttonVisibilityMenu.addAction("Collapse button")
        showCollapseAction.setCheckable(True)
        showCollapseAction.setChecked(self.config.showCollapseButton)
        showCollapseAction.toggled.connect(lambda checked: self.updateButtonVisibility('collapse', checked))
        
        # Group button toggle
        showGroupAction = buttonVisibilityMenu.addAction("Group button")
        showGroupAction.setCheckable(True)
        showGroupAction.setChecked(self.config.showGroupButton)
        showGroupAction.toggled.connect(lambda checked: self.updateButtonVisibility('group', checked))
        
        # Orientation button toggle
        showOrientationAction = buttonVisibilityMenu.addAction("Orientation button")
        showOrientationAction.setCheckable(True)
        showOrientationAction.setChecked(self.config.showOrientationButton)
        showOrientationAction.toggled.connect(lambda checked: self.updateButtonVisibility('orientation', checked))
        
        # Add visibility options
        collapseAction = menu.addAction("Toggle content visibility")
        collapseAction.triggered.connect(self.toggleContentVisibility)
        
        # Allow widgets to extend this menu
        self.extendContextMenu(menu)
        
        return menu
    
    def extendContextMenu(self, menu):
        """
        This method can be overridden by subclasses to add additional items to the context menu.
        The default implementation checks if any widgets in the dock have a method named
        'extendDockContextMenu' and calls it if present.
        
        Args:
            menu (QtWidgets.QMenu): The context menu to extend
        """
        # Check if any widgets want to add menu items
        for widget in self.widgets:
            if hasattr(widget, 'extendDockContextMenu') and callable(getattr(widget, 'extendDockContextMenu')):
                widget.extendDockContextMenu(menu, self)

    def contextMenuEvent(self, event):
        """
        Handle right-click events on the dock itself (but not its contents).
        This gives access to the context menu when clicking on the dock frame.
        """
        # Check if the event occurred directly on the dock widget, not on a child widget
        if self.childAt(event.pos()) is None:
            menu = self.buildContextMenu()
            if menu and not menu.isEmpty():
                menu.exec(event.globalPos())
            event.accept()
        else:
            # Let the event propagate to children
            super().contextMenuEvent(event)
            
        

    # ==================================================================================================================== #
    # Specific context menu action handlers                                                                                #
    # ==================================================================================================================== #
    # @function_attributes(short_name=None, tags=['context_menu', 'action'], input_requires=[], output_provides=[], uses=[], used_by=['buildContextMenu'], creation_date='2025-03-25 16:56', related_items=[])
    def updateButtonVisibility(self, button_type, visible):
        """Update the visibility of a specific button in the dock label."""
        if button_type == 'close':
            self.config.showCloseButton = visible
        elif button_type == 'collapse':
            self.config.showCollapseButton = visible
        elif button_type == 'group':
            self.config.showGroupButton = visible
        elif button_type == 'orientation':
            self.config.showOrientationButton = visible
        
        # Update the buttons in the UI
        self.label.updateButtonsFromConfig()
    
    # @function_attributes(short_name=None, tags=['context_menu', 'action'], input_requires=[], output_provides=[], uses=[], used_by=['buildContextMenu'], creation_date='2025-03-25 16:55', related_items=[])
    def on_renamed(self, dock, new_name):
        """Handle renaming of the dock."""
        self._name = new_name
        self.sigRenamed.emit(self, new_name)
        


def debug_dock_label(dock_label, label_name="Unknown"):
    """Print comprehensive debug information about a DockLabel to diagnose layout issues."""
    print(f"\n--- DockLabel Debug: {label_name} ---")
    
    # Basic geometry info
    print(f"Position: ({dock_label.x()}, {dock_label.y()})")
    print(f"Size: {dock_label.width()} × {dock_label.height()}")
    print(f"Geometry: {dock_label.geometry()}")
    print(f"Content rect: {dock_label.rect()}")
    print(f"Size hint: {dock_label.sizeHint()}")
    print(f"Minimum size hint: {dock_label.minimumSizeHint()}")
    
    # Text content issues
    print(f"Text content: '{dock_label.text()}'")
    print(f"Text length: {len(dock_label.text())}")
    print(f"Has tooltip: {bool(dock_label.toolTip())}")
    
    # Orientation and layout issues
    print(f"Orientation: {dock_label.orientation}")
    print(f"Dim state: {dock_label.dim}")
    print(f"Size policy: {dock_label.sizePolicy().horizontalPolicy()}, {dock_label.sizePolicy().verticalPolicy()}")
    
    # Button presence and sizes
    print(f"Has closeButton: {dock_label.closeButton is not None}")
    if dock_label.closeButton:
        print(f"  - closeButton size: {dock_label.closeButton.size()}")
    
    print(f"Has collapseButton: {dock_label.collapseButton is not None}")
    if dock_label.collapseButton:
        print(f"  - collapseButton size: {dock_label.collapseButton.size()}")
    
    print(f"Has groupButton: {dock_label.groupButton is not None}")
    if dock_label.groupButton:
        print(f"  - groupButton size: {dock_label.groupButton.size()}")
    
    print(f"Has orientationButton: {dock_label.orientationButton is not None}")
    if dock_label.orientationButton:
        print(f"  - orientationButton size: {dock_label.orientationButton.size()}")
    
    # Visibility and enablement
    print(f"Is visible: {dock_label.isVisible()}")
    print(f"Is enabled: {dock_label.isEnabled()}")
    print(f"Is shown: {not dock_label.isHidden()}")
    
    # Parent and layout context
    print(f"Parent type: {type(dock_label.parent()).__name__}")
    
    # Font metrics for text rendering
    font_metrics = dock_label.fontMetrics()
    if dock_label.text():
        text_width = font_metrics.horizontalAdvance(dock_label.text())
        print(f"Text width by font metrics: {text_width}px")
    
    # Check elided text mode
    if hasattr(dock_label, 'elided_text_mode'):
        print(f"Elided text mode: {dock_label.elided_text_mode}")
    
    # Check stylesheet
    if dock_label.styleSheet():
        print(f"Has custom stylesheet: Yes")
        # print(f"Stylesheet: {dock_label.styleSheet()}")
    else:
        print(f"Has custom stylesheet: No")
    
    print("--- End Debug Info ---\n")




class DockLabel(VerticalLabel):
    """ the label and 'title bar' at the top of the Dock widget that displays the title and allows dragging/closing.
    VerticalLabel: .forceWidth, .orientation
    
    """
    sigClicked = QtCore.Signal(object, object)
    sigCloseClicked = QtCore.Signal()
    sigCollapseClicked = QtCore.Signal()
    sigGroupClicked = QtCore.Signal()
    sigToggleOrientationClicked = QtCore.Signal(bool)
    
    sigContextMenuRequested = QtCore.Signal(object, object)  # Emits dock label and QPoint
    sigRenamed = QtCore.Signal(object, str)  # Emits dock and new name
    


    def __init__(self, text, dock, display_config:DockDisplayConfig):
        self.dim = False
        self.fixedWidth = False
        self.config = display_config
        # self.elided_text_mode = None
        self.elided_text_mode = QtCore.Qt.TextElideMode.ElideLeft # True
        VerticalLabel.__init__(self, text, orientation='horizontal', forceWidth=False)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop|QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.dock = dock
        self.updateStyle()
        self.setAutoFillBackground(False)
        self.mouseMoved = False
        self.setToolTip(self.text()) ## the original text is only preserved in the label's .toolTip()
        self.closeButton = None
        self.collapseButton = None
        self.groupButton = None
        self.orientationButton = None
        
        # Create all possible buttons (always create them)
        MIN_BUTTON_SIZE = 12
        
        # Create close button
        self.closeButton = QtWidgets.QToolButton(self)
        self.closeButton.clicked.connect(self.sigCloseClicked)
        self.closeButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarCloseButton))
        self.closeButton.setMinimumSize(MIN_BUTTON_SIZE, MIN_BUTTON_SIZE)
        self.closeButton.setFixedSize(MIN_BUTTON_SIZE, MIN_BUTTON_SIZE)
        
        # Create collapse button
        self.collapseButton = QtWidgets.QToolButton(self)
        self.collapseButton.clicked.connect(self.sigCollapseClicked)
        self.collapseButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarMinButton))
        self.collapseButton.setMinimumSize(MIN_BUTTON_SIZE, MIN_BUTTON_SIZE)
        self.collapseButton.setFixedSize(MIN_BUTTON_SIZE, MIN_BUTTON_SIZE)
        
        # Create group button
        self.groupButton = QtWidgets.QToolButton(self)
        self.groupButton.clicked.connect(self.sigGroupClicked)
        self.groupButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogListView))
        self.groupButton.setMinimumSize(MIN_BUTTON_SIZE, MIN_BUTTON_SIZE)
        self.groupButton.setFixedSize(MIN_BUTTON_SIZE, MIN_BUTTON_SIZE)
        
        # Create orientation button
        self.orientationButton = QtWidgets.QToolButton(self)
        self.orientationButton.setCheckable(True)
        self.orientationButton.toggled.connect(self.sigToggleOrientationClicked)
        self.orientationButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ToolBarHorizontalExtensionButton))
        self.orientationButton.setMinimumSize(MIN_BUTTON_SIZE, MIN_BUTTON_SIZE)
        self.orientationButton.setFixedSize(MIN_BUTTON_SIZE, MIN_BUTTON_SIZE)
        
        # Set initial visibility based on config
        self.updateButtonsFromConfig()
        
        # Connect config property changes to UI updates
        # self.config.propertyChanged.connect(self.updateButtonsFromConfig)
    


    def updateButtonsFromConfig(self):
        """Updates button visibility and state based on current config."""
        # Set visibility based on config
        self.closeButton.setVisible(self.config.showCloseButton)
        self.collapseButton.setVisible(self.config.showCollapseButton)
        self.groupButton.setVisible(self.config.showGroupButton)
        self.orientationButton.setVisible(self.config.showOrientationButton)
        
        # Update count of buttons
        self.num_total_title_bar_buttons = (int(self.config.showCloseButton) + 
                                        int(self.config.showCollapseButton) + 
                                        int(self.config.showGroupButton) +
                                        int(self.config.showOrientationButton))
        
        # Force a resize to update button positions
        self.updateGeometry()
        self.update()




    def debug_print(self, label_name: str="Unknown"):
        """Print comprehensive debug information about a DockLabel to diagnose layout issues."""
        return debug_dock_label(self, label_name=label_name)
    

    """Print comprehensive debug information about a DockLabel to diagnose layout issues."""
    def updateStyle(self):
        updated_stylesheet = self.config.get_stylesheet(orientation=self.orientation, is_dim=self.dim)
        if self.orientation == 'vertical':
            self.vStyle = updated_stylesheet
            self.setStyleSheet(self.vStyle)
        else:
            self.hStyle = updated_stylesheet
            self.setStyleSheet(self.hStyle)


    def setDim(self, d):
        """ Note that `self.dim` refers to whether the tab is a background tab or not. """
        if self.dim != d:
            self.dim = d
            self.updateStyle()

    def setOrientation(self, o):
        """ 
        sets self.orientation
        
        """
        # self.config.orientation = o ## do not do this, that would update the desired orientation
        VerticalLabel.setOrientation(self, o)
        self.updateStyle()

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.pressPos = lpos
        self.mouseMoved = False
        ev.accept()

    def mouseMoveEvent(self, ev):
        if not self.mouseMoved:
            lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
            self.mouseMoved = (lpos - self.pressPos).manhattanLength() > QtWidgets.QApplication.startDragDistance()

        if self.mouseMoved and ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.dock.startDrag()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        ev.accept()
        if not self.mouseMoved:
            self.sigClicked.emit(self, ev)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.dock.float()

    def resizeEvent(self, ev):
        debug_print: bool = False
        num_total_title_bar_buttons: int = self.num_total_title_bar_buttons

        button_size = None  # Track the common size for all buttons
        current_x = 0  # Track horizontal positioning for buttons in horizontal layout
        current_y = 0  # Track vertical positioning for buttons in vertical layout

        # Calculate button sizes and positions
        if self.closeButton or self.collapseButton or self.groupButton:
            if self.orientation == 'vertical':
                ## sideways mode with bar on left
                button_size = ev.size().width()
            else:
                ## regular mode with bar on top
                button_size = ev.size().height()
        ## END if self.closeButton or self.collapseButton or self.groupButton...
        
        # Position closeButton if it exists
        if self.closeButton:
            self.closeButton.setFixedSize(QtCore.QSize(button_size, button_size))
            if self.orientation == 'vertical':
                ## sideways mode with bar on left
                self.closeButton.move(0, current_y)
                current_y += button_size  # Stack buttons vertically
            else:
                ## regular mode with bar on top
                # button_x = ((ev.size().width() - button_size) - current_x) ## right aligned
                button_x = current_x ## left aligned
                self.closeButton.move(button_x, 0)
                current_x += button_size  # Stack buttons horizontally
        ## END if self.closeButton...
        
        # Position collapseButton if it exists
        if self.collapseButton:
            self.collapseButton.setFixedSize(QtCore.QSize(button_size, button_size))
            if self.orientation == 'vertical':
                ## sideways mode with bar on left
                self.collapseButton.move(0, current_y)
                current_y += button_size
            else:
                ## regular mode with bar on top
                # button_x = ((ev.size().width() - button_size) - current_x) ## right aligned
                button_x = current_x ## left aligned
                self.collapseButton.move(button_x, 0)
                current_x += button_size
        ## END if self.collapseButton...
        
        # Position orientationButton if it exists
        if self.orientationButton:
            self.orientationButton.setFixedSize(QtCore.QSize(button_size, button_size))
            if self.orientation == 'vertical':
                ## sideways mode with bar on left
                self.orientationButton.move(0, current_y)
            else:
                ## regular mode with bar on top
                # button_x = ((ev.size().width() - button_size) - current_x) ## right aligned
                button_x = current_x ## left aligned
                self.orientationButton.move(button_x, 0)
                current_x += button_size
        ## END if self.orientationButton...   

        # Position groupButton if it exists
        if self.groupButton:
            self.groupButton.setFixedSize(QtCore.QSize(button_size, button_size))
            if self.orientation == 'vertical':
                ## sideways mode with bar on left
                self.groupButton.move(0, current_y)
            else:
                ## regular mode with bar on top
                # button_x = ((ev.size().width() - button_size) - current_x) ## right aligned
                button_x = current_x ## left aligned
                self.groupButton.move(button_x, 0)
                current_x += button_size
        ## END if self.groupButton...   
        
        ## See how much space is left for the text label after subtracting away the buttons:
        button_occupied_space = ((button_size if button_size else 0) * num_total_title_bar_buttons)

        if self.elided_text_mode is not None:
            # Add elided text logic
            if debug_print:
                print(f'self.elided_text_mode: {self.elided_text_mode} -- self.orientation: {self.orientation} -- w: {self.width()}, h: {self.height()}')
            font_metrics = QtGui.QFontMetrics(self.font())
            
            if self.orientation == 'vertical':
                ## sideways mode with bar on left
                # available_text_space = max(0, self.height() - (size if self.closeButton else 0))
                available_text_space = max(0, self.height() - button_occupied_space)
                
            else:
                ## regular mode with bar on top
                # available_text_space = max(0, self.width() - (size if self.closeButton else 0))
                available_text_space = max(0, self.width() - button_occupied_space)
                

            if debug_print:
                print(f'\tavailable_text_space: {available_text_space}')
            # Skip elision if available space is insufficient
            if available_text_space > 0:
                original_text: str = self.toolTip()
                elided_text = font_metrics.elidedText(original_text, self.elided_text_mode, available_text_space)
                self.setText(elided_text)
            else:
                if debug_print:
                    print("Insufficient space for elision; skipping.")

        else:
            if debug_print:
                print(f'self.elided_text_mode == None so skipping eliding -- self.orientation: {self.orientation}')
        ## END if self.elided_text_mode is not None...
        super(DockLabel,self).resizeEvent(ev)


    def updateCollapseButtonStyle(self, is_collapse_active: bool):
        """Updates the collapse button style based on the current state."""
        if is_collapse_active:
            self.collapseButton.setStyleSheet("""
                QToolButton {
                    background-color: #0078d7; /* Highlighted blue for collapsed state */
                }
            """)
        else:
            self.collapseButton.setStyleSheet("") ## clear the stylesheet
                        
            # self.collapseButton.setStyleSheet("""
            #     QToolButton {
            #         background-color: #000000; /* Light gray for collapsed state */
            #     }
            # """)
        # self.collapseButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarMinButton))
        

    def contextMenuEvent(self, event):
        """Handle right-click events on the dock label by showing a context menu."""
        # First emit signal to allow external handlers to process this event
        self.sigContextMenuRequested.emit(self, event.globalPos())
        
        # Create the context menu
        menu = self.dock.buildContextMenu()
        if menu and not menu.isEmpty():
            menu.exec(event.globalPos())

        # menu = QtWidgets.QMenu(self)
        
        # # Create standard actions
        # renameAction = menu.addAction("Rename dock...")
        # toggleOrientationAction = menu.addAction("Toggle orientation")
        
        # # Add Close action if close button is available
        # closeAction = None
        # if self.config.showCloseButton:
        #     closeAction = menu.addAction("Close dock")
        
        # # Add buttons visibility submenu
        # buttonVisibilityMenu = menu.addMenu("Show dock buttons")
        # showCloseAction = buttonVisibilityMenu.addAction("Close button")
        # showCloseAction.setCheckable(True)
        # showCloseAction.setChecked(self.config.showCloseButton)
        
        # showCollapseAction = buttonVisibilityMenu.addAction("Collapse button")
        # showCollapseAction.setCheckable(True)
        # showCollapseAction.setChecked(self.config.showCollapseButton)
        
        # showGroupAction = buttonVisibilityMenu.addAction("Group button")
        # showGroupAction.setCheckable(True)
        # showGroupAction.setChecked(self.config.showGroupButton)
        
        # showOrientationAction = buttonVisibilityMenu.addAction("Orientation button")
        # showOrientationAction.setCheckable(True)
        # showOrientationAction.setChecked(self.config.showOrientationButton)
        
        # # Add visibility options
        # collapseAction = menu.addAction("Toggle content visibility")
        
        # # Show menu and get selected action
        # action = menu.exec(event.globalPos())
        
        # # Handle actions
        # if action == renameAction:
        #     self.promptRename()
        # elif action == toggleOrientationAction:
        #     new_orientation = 'vertical' if self.orientation == 'horizontal' else 'horizontal'
        #     self.dock.setOrientation(new_orientation, force=True)
        # elif closeAction is not None and action == closeAction:
        #     self.sigCloseClicked.emit()
        # elif action == collapseAction:
        #     self.sigCollapseClicked.emit()
        # elif action == showCloseAction:
        #     self.config.showCloseButton = showCloseAction.isChecked()
        #     self.updateButtonsFromConfig()
        # elif action == showCollapseAction:
        #     self.config.showCollapseButton = showCollapseAction.isChecked()
        #     self.updateButtonsFromConfig()
        # elif action == showGroupAction:
        #     self.config.showGroupButton = showGroupAction.isChecked()
        #     self.updateButtonsFromConfig()
        # elif action == showOrientationAction:
        #     self.config.showOrientationButton = showOrientationAction.isChecked()
        #     self.updateButtonsFromConfig()
        

    def promptRename(self):
        """Show a dialog to rename the dock."""
        current_name = self.text()
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, 
            "Rename Dock", 
            "New dock name:", 
            QtWidgets.QLineEdit.Normal, 
            current_name
        )
        
        if ok and new_name:
            self.dock.setTitle(new_name)
            self.sigRenamed.emit(self.dock, new_name)
            
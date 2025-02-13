from typing import Optional, Callable
import warnings
from attrs import define, field, Factory
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop


@define(slots=False)
class DockDisplayConfig(object):
    """Holds the display and configuration options for a Dock, such as how to format its title bar (color and font), whether it's closable, etc."""
    showCloseButton: bool = field(default=True)
    showCollapseButton: bool = field(default=False)
    showGroupButton: bool = field(default=False)
    hideTitleBar: bool = field(default=False)
    fontSize: str = field(default='10px')
    corner_radius: str = field(default='2px')
    # fontSize: str = field(default='10px')
    custom_get_stylesheet_fn: Callable = field(default=None) #(self, orientation, is_dim)
    _orientation: Optional[str] = field(default=None, alias="orientation", metadata={'valid_values': [None, 'auto', 'vertical', 'horizontal']}) # alias="orientation" just refers to the initializer, it doesn't interfere with the @property

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

    def showTitleBar(self):
        """
        Show the title bar for this Dock.
        """
        self.label.show()
        self.labelHidden = False
        self.allowedAreas.add('center')
        self.updateStyle()
        

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
        






class DockLabel(VerticalLabel):
    """ the label and 'title bar' at the top of the Dock widget that displays the title and allows dragging/closing. """
    sigClicked = QtCore.Signal(object, object)
    sigCloseClicked = QtCore.Signal()
    sigCollapseClicked = QtCore.Signal()
    sigGroupClicked = QtCore.Signal()
    

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
        self.setToolTip(self.text())
        self.closeButton = None
        self.collapseButton = None
        self.groupButton = None
        self.num_total_title_bar_buttons: int = int(display_config.showCloseButton) + int(display_config.showCollapseButton) + int(display_config.showGroupButton)
        # if self.num_total_title_bar_buttons > 1:
        #     ## use a layout instead
        #     # self.buttonBarLayout = QtWidgets.
        #     pass
        if display_config.showCloseButton:
            self.closeButton = QtWidgets.QToolButton(self)
            self.closeButton.clicked.connect(self.sigCloseClicked)
            self.closeButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarCloseButton))
        if display_config.showCollapseButton:
            self.collapseButton = QtWidgets.QToolButton(self)
            self.collapseButton.clicked.connect(self.sigCollapseClicked)
            self.collapseButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarMinButton))
        if display_config.showGroupButton:
            self.groupButton = QtWidgets.QToolButton(self)
            self.groupButton.clicked.connect(self.sigGroupClicked)
            self.groupButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogListView))




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
        
        # if self.closeButton:
        #     if self.orientation == 'vertical':
        #         ## sideways mode with bar on left
        #         size = ev.size().width()
        #         pos = QtCore.QPoint(0, 0)
        #     else:
        #         ## regular mode with bar on top
        #         size = ev.size().height()
        #         pos = QtCore.QPoint((ev.size().width() - size), 0)
        #     self.closeButton.setFixedSize(QtCore.QSize(size, size))
        #     self.closeButton.move(pos)
        # ## END if self.closeButton...
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
                elided_text = font_metrics.elidedText(self.text(), self.elided_text_mode, available_text_space)
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
        
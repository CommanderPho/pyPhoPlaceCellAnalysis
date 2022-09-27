from typing import Optional
import warnings

from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop


class DockDisplayConfig(object):
    """docstring for DockDisplayConfig."""
    def __init__(self, showCloseButton=True, fontSize='12px', corner_radius='3px'):
        super(DockDisplayConfig, self).__init__()
        self.fontSize = fontSize
        self.showCloseButton = showCloseButton
        self.corner_radius = corner_radius
        
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
            return """DockLabel {
                background-color : %s;
                color : %s;
                border-top-right-radius: 0px;
                border-top-left-radius: %s;
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: %s;
                border-width: 0px;
                border-right: 2px solid %s;
                padding-top: 3px;
                padding-bottom: 3px;
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

    sigStretchChanged = QtCore.Signal()
    sigClosed = QtCore.Signal(object)

    def __init__(self, name, area=None, size=(10, 10), widget=None, hideTitle=False, autoOrientation=True, closable=False, fontSize="12px", display_config:Optional[DockDisplayConfig]=None): # , closable=False, fontSize="12px"
        QtWidgets.QWidget.__init__(self)
        DockDrop.__init__(self)
        self._container = None
        self._name = name
        self.area = area
        # self.label = DockLabel(name, self, closable, fontSize)
        
        if display_config is None:
            print(f"WARNING: Dock.__init__(...): display_config is None... using old-mode fallback. This will be eventually depricated.")
            display_config = DockDisplayConfig(closable, fontSize=fontSize, corner_radius='3px')
        else:
            print(f"WARNING: Dock.__init__(...): display_config is set, so the explicitly passed parameters 'closable' and 'fontSize' will be ignored.")
        
        self.label = DockLabel(name, self, display_config)
        if display_config.showCloseButton:
            self.label.sigCloseClicked.connect(self.close)
        self.labelHidden = False
        self.moveLabel = True  ## If false, the dock is no longer allowed to move the label.
        self.autoOrient = autoOrientation
        self.orientation = 'horizontal'
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


class DockLabel(VerticalLabel):
    """ the label at the top of the Dock widget that displays the title and allows dragging/closing. """
    sigClicked = QtCore.Signal(object, object)
    sigCloseClicked = QtCore.Signal()

    def __init__(self, text, dock, display_config:DockDisplayConfig):
        self.dim = False
        self.fixedWidth = False
        self.config = display_config
        VerticalLabel.__init__(self, text, orientation='horizontal', forceWidth=False)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop|QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.dock = dock
        self.updateStyle()
        self.setAutoFillBackground(False)
        self.mouseMoved = False

        self.closeButton = None
        if display_config.showCloseButton:
            self.closeButton = QtWidgets.QToolButton(self)
            self.closeButton.clicked.connect(self.sigCloseClicked)
            self.closeButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarCloseButton))

    def updateStyle(self):
        updated_stylesheet = self.config.get_stylesheet(orientation=self.orientation, is_dim=self.dim)
        if self.orientation == 'vertical':
            self.vStyle = updated_stylesheet
            self.setStyleSheet(self.vStyle)
        else:
            self.hStyle = updated_stylesheet
            self.setStyleSheet(self.hStyle)
                    
        ## Old (non-config) method:
        # corner_radius = '3px'
        # if self.dim:
        #     fg_color = '#aaa'
        #     bg_color = '#44a'
        #     border_color = '#339'
        # else:
        #     fg_color = '#fff'
        #     bg_color = '#66c'
        #     border_color = '#55B'

        # if self.orientation == 'vertical':
        #     self.vStyle = """DockLabel {
        #         background-color : %s;
        #         color : %s;
        #         border-top-right-radius: 0px;
        #         border-top-left-radius: %s;
        #         border-bottom-right-radius: 0px;
        #         border-bottom-left-radius: %s;
        #         border-width: 0px;
        #         border-right: 2px solid %s;
        #         padding-top: 3px;
        #         padding-bottom: 3px;
        #         font-size: %s;
        #     }""" % (bg_color, fg_color, corner_radius, corner_radius, border_color, self.fontSize)
        #     self.setStyleSheet(self.vStyle)
        # else:
        #     self.hStyle = """DockLabel {
        #         background-color : %s;
        #         color : %s;
        #         border-top-right-radius: %s;
        #         border-top-left-radius: %s;
        #         border-bottom-right-radius: 0px;
        #         border-bottom-left-radius: 0px;
        #         border-width: 0px;
        #         border-bottom: 2px solid %s;
        #         padding-left: 3px;
        #         padding-right: 3px;
        #         font-size: %s;
        #     }""" % (bg_color, fg_color, corner_radius, corner_radius, border_color, self.fontSize)
        #     self.setStyleSheet(self.hStyle)

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

    def resizeEvent (self, ev):
        if self.closeButton:
            if self.orientation == 'vertical':
                size = ev.size().width()
                pos = QtCore.QPoint(0, 0)
            else:
                size = ev.size().height()
                pos = QtCore.QPoint(ev.size().width() - size, 0)
            self.closeButton.setFixedSize(QtCore.QSize(size, size))
            self.closeButton.move(pos)
        super(DockLabel,self).resizeEvent(ev)

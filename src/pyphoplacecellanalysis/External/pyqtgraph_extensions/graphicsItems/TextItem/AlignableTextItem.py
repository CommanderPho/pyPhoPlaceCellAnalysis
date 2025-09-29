import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
# import pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph_extensions.mixins.SelectableItemMixin import SelectableItemMixin

__all__ = ['CustomRectBoundedTextItem']

# class RectLabel(pg.GraphicsObject):
#     def __init__(self, text, rect, font=None):
#         super().__init__()
#         self.rect = QtCore.QRectF(rect)
#         self.textItem = QtWidgets.QGraphicsTextItem(text, parent=self)
#         # self.textItem = pg.TextItem(text=text)
#         self.textItem.setParentItem(self)
#         if font:
#             self.textItem.setFont(font)
#         self.textItem.setTextWidth(self.rect.width())
#         self.textItem.setPos(self.rect.topLeft())

#     def boundingRect(self):
#         return self.rect

#     def paint(self, p, *args):
#         # optional: draw rect for debugging
#         p.setPen(pg.mkPen('r'))
#         p.drawRect(self.rect)
        


class CustomRectBoundedTextItem(pg.TextItem):
    """ based off of `InfLineLabel`
    A TextItem that attaches itself to an InfiniteLine.
    
    This class extends pg.TextItem with the following features:
    
      * Automatically positions adjacent to the line at a fixed position along
        the line and within the view box.
      * Automatically reformats text when the line value has changed.
      * Can optionally be dragged to change its location along the line.
      * Optionally aligns to its parent line.

    =============== ==================================================================
    **Arguments:**
    line            The InfiniteLine to which this label will be attached.
    text            String to display in the label. May contain a {value} formatting
                    string to display the current value of the line.
    movable         Bool; if True, then the label can be dragged along the line.
    position        Relative position (0.0-1.0) within the view to position the label
                    along the line.
    anchors         List of (x,y) pairs giving the text anchor positions that should
                    be used when the line is moved to one side of the view or the
                    other. This allows text to switch to the opposite side of the line
                    as it approaches the edge of the view. These are automatically
                    selected for some common cases, but may be specified if the 
                    default values give unexpected results.
    =============== ==================================================================
    
    All extra keyword arguments are passed to pg.TextItem. A particularly useful
    option here is to use `rotateAxis=(1, 0)`, which will cause the text to
    be automatically rotated parallel to the line.
    
    Usage:
    
        from pyphoplacecellanalysis.External.pyqtgraph_extensions.graphicsItems.TextItem.AlignableTextItem import CustomRectBoundedTextItem
    """
    def __init__(self, rect: QtCore.QRectF, text: str="", parent=None, **kwds): # , movable=False, position=0.5, anchors=None, 
        self._original_text = text ## full text
        self._parent_rect = rect
        # self.movable = movable
        # self.moving = False
        # self.orthoPos = position  # text will always be placed on the line at a position relative to view bounds
        # self.format = text
        # self.line.sigPositionChanged.connect(self.valueChanged)
        # self._endpoints = (None, None)
        # if anchors is None:
        #     # automatically pick sensible anchors
        #     rax = kwds.get('rotateAxis', None)
        #     if rax is not None:
        #         if tuple(rax) == (1,0):
        #             anchors = [(0.5, 0), (0.5, 1)]
        #         else:
        #             anchors = [(0, 0.5), (1, 0.5)]
        #     else:
        #         anchors = [(0.5, 0), (0.5, 1)]
            
        # self.anchors = anchors
        pg.TextItem.__init__(self, text=text, **kwds)
        self.setParentItem(parent)
        # self.valueChanged()
        

    @property
    def desired_text_rect(self) -> QtCore.QRectF:
        """The desired_text_rect property."""
        return self._parent_rect
    @desired_text_rect.setter
    def desired_text_rect(self, value: QtCore.QRectF):
        self._parent_rect = value
        

    @property
    def original_text(self) -> str:
        """The original_text property."""
        return self._original_text
    @original_text.setter
    def original_text(self, value: str):
        self._original_text = value


    # def required_text_rect(self) -> QtCore.QRectF:
    #     """ local bounding rect in data coords 
    #     """
    #     br = self.boundingRect()
    #     # map its corners into data coords
    #     p1 = self.mapToParent(br.topLeft())
    #     p2 = self.mapToParent(br.bottomRight())
    #     data_rect = pg.QtCore.QRectF(p1, p2).normalized()
    #     return data_rect
    
    def compute_required_full_text_rect(self, a_text: str) -> QtCore.QRectF:
    # def compute_required_full_text_rect(self, a_text: str) -> QtCore.QSizeF:
        """ local bounding rect in data coords 
        """
        font = self.textItem.font()
        metrics = pg.QtGui.QFontMetricsF(font)
        _out = metrics.boundingRect(QtCore.QRectF(), 0, a_text)
        return self.toDataCoords(_out) ## return in data coords
        # full_required_text_size = _out.size()
        # return self.toDataCoords(full_required_text_size) ## return in data coords

    def required_text_rect(self) -> QtCore.QRectF:
        """ local bounding rect in data coords 
        """
        # br = self.boundingRect()
        # br = self.compute_required_full_text_rect(self.original_text)
        # # map its corners into data coords
        # p1 = self.mapToParent(br.topLeft())
        # p2 = self.mapToParent(br.bottomRight())
        # data_rect = pg.QtCore.QRectF(p1, p2).normalized()
        # return data_rect
        return self.compute_required_full_text_rect(self.original_text)

    def active_text_rect(self) -> QtCore.QRectF:
        """ local bounding rect in data coords 
        """
        br = self.boundingRect()
        # map its corners into data coords
        p1 = self.mapToParent(br.topLeft())
        p2 = self.mapToParent(br.bottomRight())
        data_rect = pg.QtCore.QRectF(p1, p2).normalized()
        return data_rect
    
    
    def required_text_size(self) -> QtCore.QSizeF:
        """ size in data coordinates """
        return self.required_text_rect().size()
    
    def req_avail_size_diff(self) -> QtCore.QSizeF:
        """ size in data coordinates """
        required_text_size = self.required_text_size()
        available_size = self._parent_rect.size()
        req_available_size_diff = ((available_size.width() - required_text_size.width()), (available_size.height() - required_text_size.height()))
        return QtCore.QSizeF(*req_available_size_diff)

    def needs_additional_size(self) -> bool:
        req_avail_size_diff = self.req_avail_size_diff()
        if req_avail_size_diff.width() < 0.0:
            return True
        if req_avail_size_diff.height() < 0.0:
            return True
        return False
    
    # def paintEvent(self, ev):
    #     pg.TextItem.paint(
    #     p = QtGui.QPainter(self)
    #     #p.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 200)))
    #     #p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 100)))
    #     #p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
    #     #p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        
    #     if self.orientation == 'vertical':
    #         p.rotate(-90)
    #         rgn = QtCore.QRect(-self.height(), 0, self.height(), self.width())
    #     else:
    #         rgn = self.contentsRect()
    #     align = self.alignment()
    #     #align  = QtCore.Qt.AlignmentFlag.AlignTop|QtCore.Qt.AlignmentFlag.AlignHCenter
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         self.hint = p.drawText(rgn, align, self.text())
    #     p.end()
        
    #     if self.orientation == 'vertical':
    #         self.setMaximumWidth(self.hint.height())
    #         self.setMinimumWidth(0)
    #         self.setMaximumHeight(16777215)
    #         if self.forceWidth:
    #             self.setMinimumHeight(self.hint.width())
    #         else:
    #             self.setMinimumHeight(0)
    #     else:
    #         self.setMaximumHeight(self.hint.height())
    #         self.setMinimumHeight(0)
    #         self.setMaximumWidth(16777215)
    #         if self.forceWidth:
    #             self.setMinimumWidth(self.hint.width())
    #         else:
    #             self.setMinimumWidth(0)

    def sizeHint(self):
        return pg.TextItem.sizeHint(self)
        
        # if self.orientation == 'vertical':
        #     if hasattr(self, 'hint'):
        #         return QtCore.QSize(self.hint.height(), self.hint.width())
        #     else:
        #         return QtCore.QSize(19, 50)
        # else:
        #     if hasattr(self, 'hint'):
        #         return QtCore.QSize(self.hint.width(), self.hint.height())
        #     else:
        #         return QtCore.QSize(50, 19)
            
    def relayout_text(self):
        """ forces text update by directly manipulating the label"""
        # ellided_text = self.text()
        original_text: str = self.original_text
        print(f'\toriginal_text: "{original_text}"')
        # self.elided_text_mode = QtCore.Qt.TextElideMode.ElideLeft  # Ensure elision is enabled
        self.setText("")  # Clear text temporarily
        self.setText(original_text)  # Reset text to trigger recalculation

        # self.updateStyle()
        # self.resizeEvent(QtGui.QResizeEvent(self.size(), self.size()))
        

    # def valueChanged(self):
    #     if not self.isVisible():
    #         return
    #     value = self.line.value()
    #     self.setText(self.format.format(value=value))
    #     self.updatePosition()

    # def getEndpoints(self):
    #     # calculate points where line intersects view box
    #     # (in line coordinates)
    #     if self._endpoints[0] is None:
    #         lr = self.line.boundingRect()
    #         pt1 = pg.Point(lr.left(), 0)
    #         pt2 = pg.Point(lr.right(), 0)
            
    #         if self.line.angle % 90 != 0:
    #             # more expensive to find text position for oblique lines.
    #             view = self.getViewBox()
    #             if not self.isVisible() or not isinstance(view, pg.ViewBox):
    #                 # not in a viewbox, skip update
    #                 return (None, None)
    #             p = QtGui.QPainterPath()
    #             p.moveTo(pt1)
    #             p.lineTo(pt2)
    #             p = self.line.itemTransform(view)[0].map(p)
    #             vr = QtGui.QPainterPath()
    #             vr.addRect(view.boundingRect())
    #             paths = vr.intersected(p).toSubpathPolygons(QtGui.QTransform())
    #             if len(paths) > 0:
    #                 l = list(paths[0])
    #                 pt1 = self.line.mapFromItem(view, l[0])
    #                 pt2 = self.line.mapFromItem(view, l[1])
    #         self._endpoints = (pt1, pt2)
    #     return self._endpoints
    
    def updatePosition(self):
        # # update text position to relative view location along line
        # self._endpoints = (None, None)
        # pt1, pt2 = self.getEndpoints()
        # if pt1 is None:
        #     return
        # pt = pt2 * self.orthoPos + pt1 * (1-self.orthoPos)
        # self.setPos(pt)
        
        # # update anchor to keep text visible as it nears the view box edge
        # vr = self.line.viewRect()
        # if vr is not None:
        #     self.setAnchor(self.anchors[0 if vr.center().y() < 0 else 1])
        
        a_rect = self._parent_rect
        a_center_point = a_rect.center()
        self.setAnchor(pg.Point(0.5, 0.5))
        # self.setText(f'TEST_ITEM[{i}]')
        # a_label.setPos(a_label._parent_rect.x(), a_label._parent_rect.y())
        self.setPos(a_center_point.x(), a_center_point.y())
        


    # def setVisible(self, v):
    #     pg.TextItem.setVisible(self, v)
    #     if v:
    #         self.valueChanged()
            
    # def setMovable(self, m):
    #     """Set whether this label is movable by dragging along the line.
    #     """
    #     self.movable = m
    #     self.setAcceptHoverEvents(m)
        
    # def setPosition(self, p):
    #     """Set the relative position (0.0-1.0) of this label within the view box
    #     and along the line. 
        
    #     For horizontal (angle=0) and vertical (angle=90) lines, a value of 0.0
    #     places the text at the bottom or left of the view, respectively. 
    #     """
    #     self.orthoPos = p
    #     self.updatePosition()
        
    # def setFormat(self, text):
    #     """Set the text format string for this label.
        
    #     May optionally contain "{value}" to include the lines current value
    #     (the text will be reformatted whenever the line is moved).
    #     """
    #     self.format = text
    #     self.valueChanged()
        
    # def mouseDragEvent(self, ev):
    #     if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
    #         if ev.isStart():
    #             self._moving = True
    #             self._cursorOffset = self._posToRel(ev.buttonDownPos())
    #             self._startPosition = self.orthoPos
    #         ev.accept()

    #         if not self._moving:
    #             return

    #         rel = self._posToRel(ev.pos())
    #         self.orthoPos = fn.clip_scalar(self._startPosition + rel - self._cursorOffset, 0., 1.)
    #         self.updatePosition()
    #         if ev.isFinish():
    #             self._moving = False

    # def mouseClickEvent(self, ev):
    #     if self.moving and ev.button() == QtCore.Qt.MouseButton.RightButton:
    #         ev.accept()
    #         self.orthoPos = self._startPosition
    #         self.moving = False

    # def hoverEvent(self, ev):
    #     if not ev.isExit() and self.movable:
    #         ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)

    # def viewTransformChanged(self):
    #     pg.GraphicsItem.viewTransformChanged(self)
    #     self.updatePosition()
    #     pg.TextItem.viewTransformChanged(self)

    def _pointToDataCoords(self, point: Union[QtCore.QPoint, QtCore.QPointF]) -> QtCore.QPointF:
        # convert local position to relative position along line between view bounds
        return self.mapToParent(point)
    


    def _rectToDataCoords(self, rect: Union[QtCore.QRect, QtCore.QRectF]) -> QtCore.QRectF:
        # convert local position to relative position along line between view bounds

        # map its corners into data coords
        p1 = self.mapToParent(rect.topLeft())
        p2 = self.mapToParent(rect.bottomRight())
        if isinstance(rect, pg.QtCore.QRectF):
            data_rect = pg.QtCore.QRectF(p1, p2).normalized()
        else:
            data_rect = pg.QtCore.QRect(p1, p2).normalized()
        return data_rect
    

    def _sizeToDataCoords(self, size: Union[QtCore.QSize, QtCore.QSizeF]) -> QtCore.QSizeF:
        # convert local position to relative position along line between view bounds
        a_rect = self._rectToDataCoords(QtCore.QRectF(0, 0, size.width(), size.height()))        
        # a_pt = self._pointToDataCoords(point=QtCore.QPointF(size.width(), size.height()))
        # return QtCore.QSizeF(a_pt.x(), a_pt.y())
        return QtCore.QSizeF(a_rect.width(), a_rect.height())
    
    
    def toDataCoords(self, obj: Union[QtCore.QPoint, QtCore.QPointF, QtCore.QRect, QtCore.QRectF, QtCore.QSize, QtCore.QSizeF]) -> Union[QtCore.QPoint, QtCore.QPointF, QtCore.QRect, QtCore.QRectF, QtCore.QSize, QtCore.QSizeF]:
        # convert local position to relative position along line between view bounds
        if isinstance(obj, (QtCore.QPoint, QtCore.QPointF)):
            return self._pointToDataCoords(obj)
        elif isinstance(obj, (QtCore.QSize, QtCore.QSizeF)):
            return self._sizeToDataCoords(obj)
        elif isinstance(obj, (QtCore.QRect, QtCore.QRectF)):
            return self._rectToDataCoords(obj)
        else:
            raise TypeError(f'unexpected type: {type(obj)}')
        
    


    # # def _posToRel(self, pos):
    # #     # convert local position to relative position along line between view bounds

    # #     br = self.boundingRect()
    # #     # map its corners into data coords
    # #     p1 = self.mapToParent(br.topLeft())
    # #     p2 = self.mapToParent(br.bottomRight())
    # #     data_rect = pg.QtCore.QRectF(p1, p2).normalized()
    # #     return data_rect
            

    #     pt1, pt2 = self.getEndpoints()
    #     if pt1 is None:
    #         return 0
    #     pos = self.mapToParent(pos)
    #     return (pos.x() - pt1.x()) / (pt2.x()-pt1.x())

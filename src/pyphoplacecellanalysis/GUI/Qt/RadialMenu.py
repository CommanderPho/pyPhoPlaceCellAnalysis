from PyQt5 import QtWidgets, QtGui, QtCore
from math import sqrt

import pyphoplacecellanalysis.External.pyqtgraph as pg


class RadialMenu(QtWidgets.QGraphicsObject):
    """ A PyQt-based Radial GUI Menu Widget 
     From https://stackoverflow.com/questions/60310219/radial-menu-with-pyside-pyqt
    
     https://stackoverflow.com/a/60312581/9732163 
     answered Feb 20, 2020 at 3:42 by user: musicamante
     

    """
    buttonClicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptHoverEvents(True)
        self.buttons = {}

    def addButton(self, id, innerRadius, size, startAngle, angleSize, pen=None, 
                  brush=None, icon=None):
        # if a button already exists with the same id, remove it
        if id in self.buttons:
            oldItem = self.buttons.pop(id)
            if self.scene():
                self.scene().removeItem(oldItem)
            oldItem.setParent(None)

        # compute the extents of the inner and outer "circles"
        startRect = QtCore.QRectF(
            -innerRadius, -innerRadius, innerRadius * 2, innerRadius * 2)
        outerRadius = innerRadius + size
        endRect = QtCore.QRectF(
            -outerRadius, -outerRadius, outerRadius * 2, outerRadius * 2)

        # create the circle section path
        path = QtGui.QPainterPath()
        # move to the start angle, using the outer circle
        path.moveTo(QtCore.QLineF.fromPolar(outerRadius, startAngle).p2())
        # draw the arc to the end of the angle size
        path.arcTo(endRect, startAngle, angleSize)
        # draw a line that connects to the inner circle
        path.lineTo(QtCore.QLineF.fromPolar(innerRadius, startAngle + angleSize).p2())
        # draw the inner circle arc back to the start angle
        path.arcTo(startRect, startAngle + angleSize, -angleSize)
        # close the path back to the starting position; theoretically unnecessary,
        # but better safe than sorry
        path.closeSubpath()

        # create a child item for the "arc"
        item = QtWidgets.QGraphicsPathItem(path, self)
        item.setPen(pen if pen else (QtGui.QPen(QtCore.Qt.transparent)))
        item.setBrush(brush if brush else QtGui.QColor(180, 140, 70))
        self.buttons[id] = item

        if icon is not None:
            # the maximum available size is at 45 degrees, use the Pythagorean
            # theorem to compute it and create a new pixmap based on the icon
            iconSize = int(sqrt(size ** 2 / 2))
            pixmap = icon.pixmap(iconSize)
            # create the child icon (pixmap) item
            iconItem = QtWidgets.QGraphicsPixmapItem(pixmap, self)
            # push it above the "arc" item
            iconItem.setZValue(item.zValue() + 1)
            # find the mid of the angle and put the icon there
            midAngle = startAngle + angleSize / 2
            iconPos = QtCore.QLineF.fromPolar(innerRadius + size * .5, midAngle).p2()
            iconItem.setPos(iconPos)
            # use the center of the pixmap as the offset for centering
            iconItem.setOffset(-pixmap.rect().center())

    def itemAtPos(self, pos):
        for button in self.buttons.values():
            if button.shape().contains(pos):
                return button

    def checkHover(self, pos):
        hoverButton = self.itemAtPos(pos)
        for button in self.buttons.values():
            # set a visible border only for the hovered item
            button.setPen(QtCore.Qt.red if button == hoverButton else QtCore.Qt.transparent)

    def hoverEnterEvent(self, event):
        self.checkHover(event.pos())

    def hoverMoveEvent(self, event):
        self.checkHover(event.pos())

    def hoverLeaveEvent(self, event):
        for button in self.buttons.values():
            button.setPen(QtCore.Qt.transparent)

    def mousePressEvent(self, event):
        clickButton = self.itemAtPos(event.pos())
        if clickButton:
            for id, btn in self.buttons.items():
                if btn == clickButton:
                    self.buttonClicked.emit(id)

    def boundingRect(self):
        return self.childrenBoundingRect()

    def paint(self, qp, option, widget):
        # required for QGraphicsObject subclasses
        pass


# ==================================================================================================================== #
# Testing                                                                                                              #
# ==================================================================================================================== #
ButtonData = [
    (50, 40, QtWidgets.QStyle.SP_MessageBoxInformation), 
    (90, 40, QtWidgets.QStyle.SP_MessageBoxQuestion), 
    (180, 20, QtWidgets.QStyle.SP_FileDialogBack), 
    (200, 20, QtWidgets.QStyle.SP_DialogOkButton), 
    (220, 20, QtWidgets.QStyle.SP_DialogOpenButton), 
    (290, 30, QtWidgets.QStyle.SP_ArrowDown), 
    (320, 30, QtWidgets.QStyle.SP_ArrowUp), 
]

class RadialTest(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.scene = QtWidgets.QGraphicsScene(self)

        buttonItem = RadialMenu()
        self.scene.addItem(buttonItem)
        buttonItem.buttonClicked.connect(self.buttonClicked)
        for index, (startAngle, extent, icon) in enumerate(ButtonData):
            icon = self.style().standardIcon(icon, None, self)
            buttonItem.addButton(index, 64, 20, startAngle, extent, icon=icon)

        buttonItem.setPos(150, 150)
        buttonItem.setZValue(1000)

        self.view = QtWidgets.QGraphicsView(self.scene, self)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing)
        self.scene.setSceneRect(0, 0, 300, 300)
        self.setGeometry(50, 50, 305, 305)
        self.show()

    def buttonClicked(self, id):
        print('Button id {} has been clicked'.format(id))
        
        
        
        
        
def main():
    app = pg.mkQApp("RadialMenu Test")
	# app.setStyleSheet(stylesheet_data_stream.readAll())
	# app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5()) # QDarkStyle version

    test_widget = RadialTest()
    test_widget.show()
    pg.exec()
        
if __name__ == '__main__':
    main()

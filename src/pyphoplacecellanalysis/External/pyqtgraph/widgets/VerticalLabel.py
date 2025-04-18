import warnings

from ..Qt import QtCore, QtGui, QtWidgets

__all__ = ['VerticalLabel']
#class VerticalLabel(QtWidgets.QLabel):
    #def paintEvent(self, ev):
        #p = QtGui.QPainter(self)
        #p.rotate(-90)
        #self.hint = p.drawText(QtCore.QRect(-self.height(), 0, self.height(), self.width()), QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter, self.text())
        #p.end()
        #self.setMinimumWidth(self.hint.height())
        #self.setMinimumHeight(self.hint.width())

    #def sizeHint(self):
        #if hasattr(self, 'hint'):
            #return QtCore.QSize(self.hint.height(), self.hint.width())
        #else:
            #return QtCore.QSize(16, 50)

class VerticalLabel(QtWidgets.QLabel):
    """ A flexible label that adjusts its orientation and size based on the orientation and available space.
    
    When text is ellided (which means to cut out a portion of it and replace it with ellipses ("...") so it fits in the available space, the label's text is actualy replaced by the ellided version.
    The original is only accessible via the .toolTip()
    
    
    VerticalLabel: .forceWidth, .orientation
    """
    def __init__(self, text, orientation='vertical', forceWidth=True):
        QtWidgets.QLabel.__init__(self, text)
        self.forceWidth = forceWidth
        self.orientation = None
        self.setOrientation(orientation)
        
    def setOrientation(self, o):
        if self.orientation == o:
            ## no change in orientation values
            return
        self.orientation = o
        self.update()
        self.updateGeometry()
        
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        #p.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 200)))
        #p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 100)))
        #p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        #p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        
        if self.orientation == 'vertical':
            p.rotate(-90)
            rgn = QtCore.QRect(-self.height(), 0, self.height(), self.width())
        else:
            rgn = self.contentsRect()
        align = self.alignment()
        #align  = QtCore.Qt.AlignmentFlag.AlignTop|QtCore.Qt.AlignmentFlag.AlignHCenter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hint = p.drawText(rgn, align, self.text())
        p.end()
        
        if self.orientation == 'vertical':
            self.setMaximumWidth(self.hint.height())
            self.setMinimumWidth(0)
            self.setMaximumHeight(16777215)
            if self.forceWidth:
                self.setMinimumHeight(self.hint.width())
            else:
                self.setMinimumHeight(0)
        else:
            self.setMaximumHeight(self.hint.height())
            self.setMinimumHeight(0)
            self.setMaximumWidth(16777215)
            if self.forceWidth:
                self.setMinimumWidth(self.hint.width())
            else:
                self.setMinimumWidth(0)

    def sizeHint(self):
        if self.orientation == 'vertical':
            if hasattr(self, 'hint'):
                return QtCore.QSize(self.hint.height(), self.hint.width())
            else:
                return QtCore.QSize(19, 50)
        else:
            if hasattr(self, 'hint'):
                return QtCore.QSize(self.hint.width(), self.hint.height())
            else:
                return QtCore.QSize(50, 19)
            
    def relayout_text(self):
        """ forces text update by directly manipulating the label"""
        # ellided_text = self.text()
        original_text: str = self.toolTip()
        print(f'\toriginal_text: "{original_text}"')
        # self.elided_text_mode = QtCore.Qt.TextElideMode.ElideLeft  # Ensure elision is enabled
        self.setText("")  # Clear text temporarily
        self.setText(original_text)  # Reset text to trigger recalculation

        # self.updateStyle()
        # self.resizeEvent(QtGui.QResizeEvent(self.size(), self.size()))


if __name__ == '__main__':
    app = QtWidgets.QApplication([])  # noqa: qapplication must be stored to variable to avoid gc
    win = QtWidgets.QMainWindow()
    w = QtWidgets.QWidget()
    l = QtWidgets.QGridLayout()
    w.setLayout(l)
    
    l1 = VerticalLabel("text 1", orientation='horizontal')
    l2 = VerticalLabel("text 2")
    l3 = VerticalLabel("text 3")
    l4 = VerticalLabel("text 4", orientation='horizontal')
    l.addWidget(l1, 0, 0)
    l.addWidget(l2, 1, 1)
    l.addWidget(l3, 2, 2)
    l.addWidget(l4, 3, 3)
    win.setCentralWidget(w)
    win.show()

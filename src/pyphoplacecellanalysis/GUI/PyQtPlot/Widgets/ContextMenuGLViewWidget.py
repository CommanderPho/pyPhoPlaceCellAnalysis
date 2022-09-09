import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot


class ContextMenuGLViewWidget(gl.GLViewWidget):
    
    def __init__(self, parent=None, devicePixelRatio=None, rotationMethod='euler'):
        """    
        Basic widget for displaying 3D data
          - Rotation/scale controls
          - Axis/grid display
          - Export options

        ================ ==============================================================
        **Arguments:**
        parent           (QObject, optional): Parent QObject. Defaults to None.
        devicePixelRatio No longer in use. High-DPI displays should automatically
                         detect the correct resolution.
        rotationMethod   (str): Mechanimsm to drive the rotation method, options are 
                         'euler' and 'quaternion'. Defaults to 'euler'.
        ================ ==============================================================
        """
        super(ContextMenuGLViewWidget, self).__init__(parent=parent, devicePixelRatio=devicePixelRatio, rotationMethod=rotationMethod)

    # ==================================================================================================================== #
    # Overrides for events                                                                                                 #
    # ==================================================================================================================== #
 
    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePos = lpos
        
    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        diff = lpos - self.mousePos
        self.mousePos = lpos
        
        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == QtCore.Qt.MouseButton.MiddleButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')
        
    def mouseReleaseEvent(self, ev):
        print(f'mouseReleaseEvent(ev: {ev}) detected')
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            pos = ev.pos()
            print(f'\tpos: {pos}')
            menu = QtWidgets.QMenu(self)
            debugTestAction = menu.addAction('Debug Test')
            debugTestAction1 = menu.addAction('Debug Test 1')
            debugTestAction2 = menu.addAction('Debug Test 2')
            globalPosition = self.mapToGlobal(pos)
            # globalPosition = active_3d_plot.mapToGlobal(pos)
            print(f'\tglobalPosition: {globalPosition}')
            menu.exec(globalPosition)
            print(f'\tdone.')

        gl.GLViewWidget.mouseReleaseEvent(self, ev) # supposed to pass to parent like this?
        # Example item selection code:
        #region = (ev.pos().x()-5, ev.pos().y()-5, 10, 10)
        #print(self.itemsAt(region))
        
        ## debugging code: draw the picking region
        #glViewport(*self.getViewport())
        #glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )
        #region = (region[0], self.height()-(region[1]+region[3]), region[2], region[3])
        #self.paintGL(region=region)
        #self.swapBuffers()
        
    def wheelEvent(self, ev):
        delta = ev.angleDelta().x()
        if delta == 0:
            delta = ev.angleDelta().y()
        if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            self.opts['fov'] *= 0.999**delta
        else:
            self.opts['distance'] *= 0.999**delta
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
      
    def keyReleaseEvent(self, ev):
        if ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except KeyError:
                self.keysPressed = {}
            self.evalKeyState()
        
        
        
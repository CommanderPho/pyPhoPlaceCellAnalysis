from OpenGL.GL import *  # noqa
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphoplacecellanalysis.External.pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


__all__ = ['GLDebugAxisItem']

class GLDebugAxisItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`
		
		Displays three lines indicating origin and orientation of local coordinate system. 
		Fixed by Pho to use the default (x, y, z) => (R, G, B) color pallet instead of the arbitrary and unusual one that GLAxisItem used
		
    """
    
    def __init__(self, size=None, antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        if size is None:
            size = QtGui.QVector3D(1,1,1)
        self.antialias = antialias
        self.setSize(size=size)
        self.setGLOptions(glOptions)
    
    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x,y,z]
        self.update()
        
    def size(self):
        return self.__size[:]
    
    
    def paint(self):

        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glEnable( GL_BLEND )
        #glEnable( GL_ALPHA_TEST )
        self.setupGLState()
        
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            
        glBegin( GL_LINES )
        
        x,y,z = self.size()
        glColor4f(0, 0, 1, .6)  # z is blue
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, z)

        glColor4f(0, 1, 0, .6)  # y is green
        glVertex3f(0, 0, 0)
        glVertex3f(0, y, 0)

        glColor4f(1, 0, 0, .6)  # x is red
        glVertex3f(0, 0, 0)
        glVertex3f(x, 0, 0)
        glEnd()

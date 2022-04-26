from copy import deepcopy
from typing import OrderedDict
import OpenGL.GL as GL

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.opengl import GLAxisItem, GLGraphicsItem, GLGridItem, GLViewWidget
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui


class GLViewportOverlayPainterItem(GLGraphicsItem.GLGraphicsItem):
    """ Draws simple overlay text on the viewport. 
    
    Updating self.additional_overlay_text_lines adds items to the top-left by default
    
    
    Usage:
        paintitem = GLViewportOverlayPainterItem()
        glv.addItem(paintitem)

    """
    def __init__(self, **kwds):
        super().__init__() # should pass kwargs?
        self.additional_overlay_text_lines = []
        self.additional_overlay_text_dict = dict() 
        
        # OrderedDict(QtCore.Qt.AlignmentFlag,)
        
        
        
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)


    def compute_projection(self):
        modelview = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        projection = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        mvp = projection.T @ modelview.T
        mvp = QtGui.QMatrix4x4(mvp.ravel().tolist())

        # note that QRectF.bottom() != QRect.bottom()
        rect = QtCore.QRectF(self.view().rect())
        ndc_to_viewport = QtGui.QMatrix4x4()
        ndc_to_viewport.viewport(rect.left(), rect.bottom(), rect.width(), -rect.height())
        return ndc_to_viewport * mvp

    def paint(self):
        self.setupGLState()

        painter = QtGui.QPainter(self.view())
        self.draw(painter)
        painter.end()

    def draw(self, painter):
        painter.setPen(QtCore.Qt.GlobalColor.white)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)

        rect = self.view().rect()
        af = QtCore.Qt.AlignmentFlag

        # painter.drawText(rect, af.AlignTop | af.AlignRight, 'TR')
        # painter.drawText(rect, af.AlignBottom | af.AlignLeft, 'BL')
        # painter.drawText(rect, af.AlignBottom | af.AlignRight, 'BR')

        # Gets current info from the camera
        opts = self.view().cameraParams()
        
        
        active_overlay_text_dict = self.additional_overlay_text_dict.copy()
        # active_overlay_text_dict = deepcopy(self.additional_overlay_text_dict)
        
        # curr_text_list = active_overlay_text_dict.get((af.AlignTop | af.AlignLeft), list()) # try to get the extant text list at the specified position
        
        center = opts['center']
        xyz = self.view().cameraPosition()
        
        # curr_text_list.extend(lines) # add the text items to the list if needed
        # self.update_text(lines, (af.AlignTop | af.AlignLeft))

        # # Build the final text output
        # info = "\n".join(lines)
        # painter.drawText(rect, af.AlignTop | af.AlignLeft, info)


        # Extended Text Ouput:
        for (alignment_flag, text_lines) in active_overlay_text_dict.items():
            # Build the final text output
            if (alignment_flag == (af.AlignTop | af.AlignLeft)):
                builtin_lines = []
                builtin_lines.append(f"center : ({center.x():.1f}, {center.y():.1f}, {center.z():.1f})")
                for key in ['distance', 'fov', 'elevation', 'azimuth']:
                    builtin_lines.append(f"{key} : {opts[key]:.1f}")
                builtin_lines.append(f"xyz : ({xyz.x():.1f}, {xyz.y():.1f}, {xyz.z():.1f})")
                # Add any additional lines that may have been set:
                builtin_lines.extend(self.additional_overlay_text_lines)
                # curr_text_list.extend(builtin_lines) # add the text items to the list if needed
            else:
                builtin_lines = []
                
            curr_text_flat_str = "\n".join(text_lines + builtin_lines)
            painter.drawText(rect, alignment_flag, curr_text_flat_str)

        # # Draws some dots over the top of a grid
        # project = self.compute_projection()

        # hsize = SIZE // 2
        # for xi in range(-hsize, hsize+1):
        #     for yi in range(-hsize, hsize+1):
        #         if xi == -hsize and yi == -hsize:
        #             # skip one corner for visual orientation
        #             continue
        #         vec3 = QtGui.QVector3D(xi, yi, 0)
        #         pos = project.map(vec3).toPointF()
        #         painter.drawEllipse(pos, 1, 1)
        
    def update_text(self, text, alignment_flag: QtCore.Qt.AlignmentFlag):
        """ Update the text at a given place. """
        curr_text_list = self.additional_overlay_text_dict.get(alignment_flag, list()) # try to get the extant text list at the specified position
        curr_text_list.extend(text) # add the text items to the list if needed
        
        self.additional_overlay_text_dict[alignment_flag] = curr_text_list # update the specific list
        
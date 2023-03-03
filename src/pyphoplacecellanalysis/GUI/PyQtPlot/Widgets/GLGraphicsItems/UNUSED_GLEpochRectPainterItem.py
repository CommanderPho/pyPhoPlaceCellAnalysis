"""
Demonstrate using QPainter on a subclass of GLGraphicsItem.
"""

from io import StringIO
from turtle import st # for loading pandas dataframe from literal string
import OpenGL.GL as GL
import numpy as np
try:
    import modin.pandas as pd # modin is a drop-in replacement for pandas that uses multiple cores
except ImportError:
    import pandas as pd # fallback to pandas when modin isn't available
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.opengl import GLAxisItem, GLGraphicsItem, GLGridItem, GLViewWidget
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters

from pyphocorehelpers.geometry_helpers import find_ranges_in_window

SIZE = 32

class GLEpochRectPainterItem(GLGraphicsItem.GLGraphicsItem):
    """ WARNING: NOT YET IMPLEMENTED 
    
    TODO: NEXT_STEPS: Use the new find_ranges_in_window() function (imported above from pyphocorehelpers.geometry_helpers) to compute the relevant rectangles to draw given the active time window.
    
    
    """
    def __init__(self, starts_t=[], durations=[], **kwds):
        super().__init__()
        raise NotImplementedError
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        # Helper container variables
        self.params = VisualizationParameters('')
        self.params.starts_t = starts_t
        self.params.durations = durations
        self.params.epoch_rects = []
        # build the correct rectangles
        # self.build_epoch_rects(starts_t, durations)
        

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

        # Draw Viewport (overlay) items:
        rect = self.view().rect()
        af = QtCore.Qt.AlignmentFlag

        # painter.drawText(rect, af.AlignTop | af.AlignRight, 'TR')
        # painter.drawText(rect, af.AlignBottom | af.AlignLeft, 'BL')
        # painter.drawText(rect, af.AlignBottom | af.AlignRight, 'BR')

        # opts = self.view().cameraParams()
        # lines = []
        # center = opts['center']
        # lines.append(f"center : ({center.x():.1f}, {center.y():.1f}, {center.z():.1f})")
        # for key in ['distance', 'fov', 'elevation', 'azimuth']:
        #     lines.append(f"{key} : {opts[key]:.1f}")
        # xyz = self.view().cameraPosition()
        # lines.append(f"xyz : ({xyz.x():.1f}, {xyz.y():.1f}, {xyz.z():.1f})")
        # info = "\n".join(lines)
        # painter.drawText(rect, af.AlignTop | af.AlignLeft, info)
        
        # Draw 3D Items:
        project = self.compute_projection()
        print(f'project: {project}') # project: PyQt5.QtGui.QMatrix4x4(0.0, 1939.89697265625, -320.0, 19731.28125, 665.1074829101562, 0.0, -240.0, 18262.5625, 0.0, 0.0, -1.0000009536743164, 52.95005416870117, 0.0, 0.0, -1.0, 53.0)
        ## Need to build an array of QRect objects for use with painter.drawRects()
                
        self.params.epoch_rects = self.build_epoch_rects(self.params.starts_t, self.params.durations)
        
        # transformed_rects = project.map(self.params.epoch_rects).toRectF()
        # painter.setTransform(QtGui.QTransform())
        painter.setTransform(project)
        
        painter.drawRects(self.params.epoch_rects)

        # hsize = SIZE // 2
        # for xi in range(-hsize, hsize+1):
        #     for yi in range(-hsize, hsize+1):
        #         if xi == -hsize and yi == -hsize:
        #             # skip one corner for visual orientation
        #             continue
                
        #         rect = QtCore.QRectF(self.view().rect())
                
                
        #         # vec3 = QtGui.QVector3D(xi, yi, 0)
        #         pos = project.map(vec3).toPointF()
        #         painter.drawRects(epoch_rect_array)
        #         painter.drawEllipse(pos, 1, 1)


    def build_epoch_rects(self, starts_t, durations):
        
        find_ranges_in_window(starts_t, starts_t + durations, active_window=
                              
                              
        half_durations = durations / 2.0
        t_centers = starts_t + half_durations
        # Convert to projected coordinates
        self.params.epoch_rects = []
        curr_width = self.view().rect().width()
        curr_height = self.view().rect().height()
        curr_center = self.view().rect().center()
        # y_center = curr_center
        curr_top = float(self.view().rect().top())
        
        for i in np.arange(len(t_centers)):
            rect = QtCore.QRectF(starts_t[i], curr_top, durations[i], curr_height)
            # rect = QtCore.QRectF(starts_t[i], -float(y_center), durations[i], curr_height)
            # rect = QtCore.QRectF(starts_t[i], -float(self.n_half_cells), durations[i], 1.0)
            self.params.epoch_rects.append(rect)
        return self.params.epoch_rects


    def update_epoch_meshes(self, starts_t, durations):
        
        self.build_epoch_rects(starts_t, durations)
        
        
        
        # t_shifted_centers = t_centers - self.spikes_window.active_time_window[0] # offset by the start of the current window
        t_shifted_centers = t_centers
        for (i, aCube) in enumerate(self.plots.new_cube_objects):
            # aCube.setPos(t_centers[i], self.n_half_cells, 0)
            aCube.resetTransform()
            aCube.translate(t_shifted_centers[i], -self.n_half_cells, self.floor_z)
            aCube.scale(durations[i], self.n_full_cell_grid, 0.25)
            # aCube.setData(pos=(t_centers[i], self.n_half_cells, 0))
            # aCube.setParent(None)
            # aCube.deleteLater()
            
            
    

pg.mkQApp("GLEpochRectPainterItem Example")
glv = GLViewWidget()
glv.show()
glv.setWindowTitle('GLEpochRectPainterItem')
glv.setCameraPosition(distance=50, elevation=90, azimuth=0)

griditem = GLGridItem()
griditem.setSize(SIZE, SIZE)
griditem.setSpacing(1, 1)
glv.addItem(griditem)

axisitem = GLAxisItem()
axisitem.setSize(SIZE/2, SIZE/2, 1)
glv.addItem(axisitem)

example_epoch_data_csv_string = StringIO(""",start,stop,duration,label\r\n0,57.32,57.472,0.15200000000000102,\r\n1,78.83,79.154,0.32399999999999807,\r\n2,80.412,80.592,0.1799999999999926,\r\n3,83.792,84.07000000000001,0.2780000000000058,\r\n4,85.005,85.144,0.13900000000001,\r\n5,89.348,89.759,0.41100000000000136,\r\n6,93.95,94.115,0.16499999999999204,\r\n7,99.805,99.936,0.13100000000000023,\r\n8,125.276,125.418,0.14200000000001012,\r\n9,139.835,140.15800000000002,0.3230000000000075,\r\n10,147.648,147.842,0.19400000000001683,\r\n11,148.927,149.068,0.14100000000001955,\r\n12,176.844,177.33,0.4860000000000184,\r\n13,183.631,183.874,0.242999999999995,\r\n14,201.058,201.257,0.19900000000001228,\r\n15,212.357,212.516,0.15899999999999181,\r\n16,240.886,241.019,0.13300000000000978,\r\n17,242.76500000000001,242.91400000000002,0.1490000000000009,\r\n18,245.6,245.937,0.3370000000000175,\r\n19,248.00400000000002,248.139,0.1349999999999909,\r\n20,274.12600000000003,274.516,0.38999999999998636,\r\n21,299.409,299.624,0.21500000000003183,\r\n22,300.184,300.308,0.1239999999999668,\r\n23,305.567,306.325,0.7579999999999814,\r\n24,318.704,318.944,0.2400000000000091,\r\n25,321.404,321.594,0.18999999999999773,\r\n26,327.33,327.596,0.26600000000001955,\r\n27,331.399,331.54,0.14100000000001955,\r\n28,333.931,334.101,0.17000000000001592,\r\n29,336.473,336.608,0.1349999999999909,\r\n30,339.333,339.502,0.16899999999998272,\r\n31,342.207,342.324,0.11700000000001864,\r\n32,342.411,342.586,0.17500000000001137,\r\n33,345.411,345.60200000000003,0.19100000000003092,\r\n34,353.54900000000004,353.796,0.24699999999995725,\r\n35,367.128,367.333,0.20500000000004093,\r\n36,368.848,369.039,0.19099999999997408,\r\n37,370.661,370.91700000000003,0.25600000000002865,\r\n38,373.08,373.371,0.2909999999999968,\r\n39,379.355,379.601,0.2459999999999809,\r\n40,380.404,380.509,0.10500000000001819,\r\n41,385.653,385.889,0.23599999999999,\r\n42,386.428,386.615,0.18700000000001182,\r\n43,392.867,393.03700000000003,0.17000000000001592,\r\n44,398.271,398.648,0.37700000000000955,\r\n45,400.012,400.219,0.20699999999999363,\r\n46,401.11,401.399,0.28899999999998727,\r\n47,404.916,405.072,0.1560000000000059,\r\n48,407.197,407.445,0.24799999999999045,\r\n49,416.095,416.366,0.27099999999995816,\r\n50,417.294,417.683,0.38900000000001,\r\n51,427.039,427.161,0.1220000000000141,\r\n52,430.531,430.757,0.2259999999999991,\r\n53,434.69,434.837,0.14699999999999136,\r\n54,435.822,435.92900000000003,0.10700000000002774,\r\n55,456.574,456.927,0.35300000000000864,\r\n56,478.076,478.354,0.27799999999996317,\r\n57,483.985,484.212,0.22699999999997544,\r\n58,489.569,489.724,0.15499999999997272,\r\n59,491.58,491.735,0.15500000000002956,\r\n60,492.742,492.954,0.21199999999998909,\r\n61,498.669,498.818,0.1490000000000009,\r\n62,519.033,519.176,0.1430000000000291,\r\n63,529.016,529.197,0.18100000000004002,\r\n64,535.8290000000001,535.999,0.16999999999995907,\r\n65,540.38,540.623,0.24300000000005184,\r\n66,543.284,543.476,0.19200000000000728,\r\n67,547.034,547.271,0.23699999999996635,\r\n68,550.17,550.294,0.12400000000002365,\r\n69,553.788,553.917,0.1290000000000191,\r\n70,554.763,555.0360000000001,0.27300000000002456,\r\n71,563.53,563.695,0.1650000000000773,\r\n72,566.97,567.256,0.2859999999999445,\r\n73,567.918,568.093,0.17499999999995453,\r\n74,573.824,574.014,0.19000000000005457,\r\n75,577.3770000000001,577.532,0.15499999999997272,\r\n76,585.466,585.687,0.22100000000000364,\r\n77,592.663,593.003,0.34000000000003183,\r\n78,594.552,594.77,0.2179999999999609,\r\n79,598.635,598.773,0.13800000000003365,\r\n80,600.773,600.884,0.11099999999999,\r\n81,602.885,603.118,0.23300000000006094,\r\n82,611.827,611.943,0.11599999999998545,\r\n83,615.653,615.794,0.1409999999999627,\r\n84,618.146,618.355,0.20900000000006003,\r\n85,624.298,624.426,0.12800000000004275,\r\n86,627.626,627.794,0.16800000000000637,\r\n87,645.028,645.294,0.2659999999999627,\r\n88,657.562,657.87,0.3079999999999927,\r\n89,660.967,661.351,0.38400000000001455,\r\n90,667.472,667.705,0.23300000000006094,\r\n91,672.054,672.269,0.21500000000003183,\r\n92,680.97,681.25,0.2799999999999727,\r\n93,689.048,689.25,0.20199999999999818,\r\n94,694.205,694.418,0.21299999999996544,\r\n95,695.519,695.662,0.1430000000000291,\r\n96,702.464,702.664,0.1999999999999318,\r\n97,703.864,704.014,0.14999999999997726,\r\n98,707.251,707.468,0.21699999999998454,\r\n99,708.61,708.7470000000001,0.1370000000000573,\r\n100,713.5260000000001,713.785,0.25899999999990087,\r\n101,718.119,718.242,0.1229999999999336,\r\n102,721.128,721.293,0.16499999999996362,\r\n103,731.023,731.184,0.16099999999994452,\r\n104,732.316,732.541,0.22500000000002274,\r\n105,745.001,745.157,0.15600000000006276,\r\n106,762.33,762.52,0.18999999999994088,\r\n107,770.556,770.745,0.18899999999996453,\r\n108,772.206,772.5310000000001,0.3250000000000455,\r\n109,777.561,778.465,0.9039999999999964,\r\n110,788.6220000000001,788.783,0.16099999999994452,\r\n111,794.7810000000001,795.014,0.23299999999994725,\r\n112,803.999,804.3000000000001,0.30100000000004457,\r\n113,806.559,806.687,0.12800000000004275,\r\n114,807.407,807.7570000000001,0.35000000000002274,\r\n115,812.899,813.029,0.12999999999999545,\r\n116,815.054,815.168,0.11400000000003274,\r\n117,819.519,819.628,0.10900000000003729,\r\n118,820.695,821.0310000000001,0.33600000000001273,\r\n119,828.7810000000001,829.119,0.33799999999996544,\r\n120,831.056,831.375,0.31899999999996,\r\n121,833.678,833.813,0.1349999999999909,\r\n122,837.658,838.614,0.9560000000000173,\r\n123,861.128,861.3720000000001,0.2440000000000282,\r\n124,863.915,864.186,0.27100000000007185,\r\n125,876.171,876.3720000000001,0.20100000000002183,\r\n126,881.962,882.1800000000001,0.21800000000007458,\r\n127,890.326,890.4540000000001,0.12800000000004275,\r\n128,892.698,892.89,0.19200000000000728,\r\n129,908.196,908.415,0.21899999999993724,\r\n130,919.932,920.143,0.21100000000001273,\r\n131,935.4590000000001,935.706,0.24699999999995725,\r\n132,939.331,939.485,0.15399999999999636,\r\n133,943.789,944.022,0.23300000000006094,\r\n134,965.4590000000001,965.745,0.2859999999999445,\r\n135,981.674,981.91,0.23599999999999,\r\n136,983.785,984.0020000000001,0.21700000000009823,\r\n137,1016.335,1016.476,0.1409999999999627,\r\n138,1017.761,1017.957,0.19600000000002638,\r\n139,1029.26,1029.422,0.16200000000003456,\r\n140,1030.044,1030.194,0.14999999999986358,\r\n141,1037.156,1037.432,0.2760000000000673,\r\n142,1045.623,1045.782,0.15899999999987813,\r\n143,1046.897,1047.448,0.5510000000001583,\r\n144,1049.864,1050.067,0.20299999999997453,\r\n145,1063.924,1064.159,0.23500000000012733,\r\n146,1068.26,1068.577,0.3170000000000073,\r\n147,1081.695,1081.827,0.13200000000006185,\r\n148,1083.282,1083.494,0.21199999999998909,\r\n149,1085.09,1085.301,0.21100000000001273,\r\n150,1104.819,1105.198,0.3790000000001328,\r\n151,1111.363,1111.74,0.3769999999999527,\r\n152,1134.018,1134.199,0.18100000000004002,\r\n153,1136.71,1136.9,0.19000000000005457,\r\n154,1142.528,1142.748,0.22000000000002728,\r\n155,1181.758,1181.904,0.14599999999995816,\r\n156,1189.151,1189.278,0.1269999999999527,\r\n157,1194.279,1194.549,0.2699999999999818,\r\n158,1224.21,1224.3790000000001,0.1690000000000964,\r\n159,1228.653,1228.82,0.16699999999991633,\r\n160,1230.519,1230.699,0.18000000000006366,\r\n161,1233.843,1233.9470000000001,0.10400000000004184,\r\n162,1239.08,1239.25,0.17000000000007276,\r\n163,1240.618,1240.72,0.10200000000008913,\r\n164,1250.0240000000001,1250.147,0.12299999999981992,\r\n165,1253.343,1253.536,0.19299999999998363,\r\n166,1256.728,1256.932,0.2039999999999509,\r\n167,1266.395,1266.71,0.31500000000005457,\r\n168,1276.409,1276.555,0.14599999999995816,\r\n169,1304.038,1304.228,0.19000000000005457,\r\n170,1313.451,1313.576,0.125,\r\n171,1316.15,1316.39,0.2400000000000091,\r\n172,1317.462,1317.6580000000001,0.19600000000014006,\r\n173,1337.51,1337.741,0.23099999999999454,\r\n174,1359.215,1359.346,0.1310000000000855,\r\n175,1367.954,1368.069,0.1150000000000091,\r\n176,1377.606,1377.751,0.1449999999999818,\r\n177,1388.288,1389.159,0.8710000000000946,\r\n178,1390.621,1390.917,0.29599999999982174,\r\n179,1393.832,1393.993,0.16099999999983083,\r\n180,1428.15,1428.432,0.2819999999999254,\r\n181,1429.087,1429.238,0.1510000000000673,\r\n182,1441.0,1441.124,0.12400000000002365,\r\n183,1453.997,1454.2920000000001,0.29500000000007276,\r\n184,1464.442,1464.605,0.16300000000001091,\r\n185,1467.941,1468.257,0.3160000000000309,\r\n186,1473.903,1474.051,0.14799999999991087,\r\n187,1500.933,1501.179,0.2460000000000946,\r\n188,1508.7930000000001,1509.09,0.2969999999997981,\r\n189,1510.001,1510.19,0.18900000000007822,\r\n190,1514.893,1515.008,0.1150000000000091,\r\n191,1517.015,1517.209,0.19399999999995998,\r\n192,1525.5430000000001,1525.7,0.15699999999992542,\r\n193,1528.416,1528.553,0.13700000000017099,\r\n194,1544.021,1544.2250000000001,0.20400000000017826,\r\n195,1558.1290000000001,1558.4170000000001,0.2880000000000109,\r\n196,1561.311,1561.509,0.19800000000009277,\r\n197,1584.151,1584.343,0.19200000000000728,\r\n198,1609.8700000000001,1610.194,0.32399999999984175,\r\n199,1624.631,1624.739,0.10799999999994725,\r\n200,1627.07,1627.257,0.1870000000001255,\r\n201,1643.512,1643.777,0.26500000000010004,\r\n202,1714.259,1714.422,0.16300000000001091,\r\n203,1715.307,1715.49,0.18299999999999272,\r\n204,1729.6870000000001,1729.806,0.11899999999991451,\r\n205,1730.93,1731.099,0.16899999999986903,\r\n""")
test_epochs_df = pd.read_csv(example_epoch_data_csv_string, sep=",")

# print(test_epochs_df)


paintitem = GLEpochRectPainterItem(starts_t=test_epochs_df['start'].to_numpy(), durations=test_epochs_df['duration'].to_numpy())
paintitem.translate(10, 5, -3)
paintitem.scale(1.2, 3.5, 1)
glv.addItem(paintitem)

if __name__ == '__main__':
    pg.exec()

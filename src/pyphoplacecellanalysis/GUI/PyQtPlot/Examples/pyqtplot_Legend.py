import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, mkQApp, QtGui


class PyQtPlotLegendMixin:
    
    def add_legend():
        """  https://www.geeksforgeeks.org/pyqtgraph-symbols/
        o : Default symbol, round circle symbol
        t : Triangle pointing downwards symbol
        t1 : Triangle pointing upwards symbol
        t2 : Triangle pointing right side symbol
        t3 : Triangle pointing left side symbol
        s : Square symbol
        p : Pentagon symbol
        h : Hexagon symbol
        star : Star symbol
        + : Plus symbol
        d : Prism symbol
        x : Cross symbol
        """
        # setting pyqtgraph configuration
        pg.setConfigOptions(antialias = True)
 
        # creating a plot window
        plot = pg.plot()
 
        # adding legend to the plot window
        plot.addLegend()
 
        # plot the line1 with symbol o
        # having color blue
        line1 = plot.plot([1, 1, 1, 1, 1], pen =(0, 0, 200), symbolBrush =(0, 0, 200),
                          symbolPen ='w', symbol ='o', symbolSize = 14, name ="symbol ='o'")
 
        # plot the line2 with symbol t
        # having color green
        line2 = plot.plot([2, 2, 2, 2, 2], pen =(0, 128, 0), symbolBrush =(0, 128, 0),
                          symbolPen ='w', symbol ='t', symbolSize = 14, name ="symbol ='t'")
 
        # plot the line3 with symbol t1
        # having color bluegreen
        line3 = plot.plot([3, 3, 3, 3, 3], pen =(19, 234, 201), symbolBrush =(19, 234, 201),
                          symbolPen ='w', symbol ='t1', symbolSize = 14, name ="symbol ='t1'")
 
        # plot the line4 with symbol t2
        # having color purple
        line4 = plot.plot([4, 4, 4, 4, 4], pen =(195, 46, 212), symbolBrush =(195, 46, 212),
                          symbolPen ='w', symbol ='t2', symbolSize = 14, name ="symbol ='t2'")
 
        # plot the line5 with symbol t3
        # having color brownish
        line5 = plot.plot([5, 5, 5, 5, 5], pen =(250, 194, 5), symbolBrush =(250, 194, 5),
                          symbolPen ='w', symbol ='t3', symbolSize = 14, name ="symbol ='t3'")
 
        # plot the line6 with symbol s
        # having color grey
        line6 = plot.plot([6, 6, 6, 6, 6], pen =(54, 55, 55), symbolBrush =(55, 55, 55), symbolPen ='w', symbol ='s', symbolSize = 14,
                  name ="symbol ='s'")
 
        # plot the line7 with symbol p
        # having color sky blue
        line7 = plot.plot([7, 7, 7, 7, 7], pen =(0, 114, 189), symbolBrush =(0, 114, 189),
                         symbolPen ='w', symbol ='p', symbolSize = 14, name ="symbol ='p'")
 
        # plot the line8 with symbol h
        # having color orange
        line8 = plot.plot([8, 8, 8, 8, 8], pen =(217, 83, 25), symbolBrush =(217, 83, 25),
                          symbolPen ='w', symbol ='h', symbolSize = 14, name ="symbol ='h'")
 
        # plot the line9 with symbol star
        # having yellow color
        line9 = plot.plot([9, 9, 9, 9, 9], pen =(237, 177, 32), symbolBrush =(237, 177, 32),
                          symbolPen ='w', symbol ='star', symbolSize = 14, name ="symbol ='star'")
 
        # plot the line10 with symbol +
        # having blueish color
        line10 = plot.plot([10, 10, 10, 10, 10], pen =(126, 47, 142), symbolBrush =(126, 47, 142),
                           symbolPen ='w', symbol ='+', symbolSize = 14, name ="symbol ='+'")
 
        # plot the line11 with symbol d
        # having greenish color
        line11 = plot.plot([11, 11, 11, 11, 11], pen =(119, 172, 48), symbolBrush =(119, 172, 48),
                           symbolPen ='w', symbol ='d', symbolSize = 14, name ="symbol ='d'")
 
        # plot the line12 with symbol x
        # having off white color
        line12 = plot.plot([12, 12, 12, 12, 12], pen =(180, 180, 180), symbolBrush =(180, 180, 180),
                           symbolPen ='w', symbol ='x', symbolSize = 14, name ="symbol ='x'")
 
        # setting x-axis range
        plot.setXRange(-3, 4)
        
        
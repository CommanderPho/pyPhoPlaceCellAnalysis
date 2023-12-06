import sys
import time
from collections import OrderedDict

import numpy as np

from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.AxisItem import AxisItem

__all__ = ['NeuronsAxisItem']


class NeuronsAxisItem(AxisItem):
    """
    **Bases:** :class:`AxisItem <pyqtgraph.AxisItem>`
    
    An AxisItem that displays dates from unix timestamps.

    The display format is adjusted automatically depending on the current time
    density (seconds/point) on the axis. For more details on changing this
    behaviour, see :func:`setZoomLevelForDensity() <pyqtgraph.NeuronsAxisItem.setZoomLevelForDensity>`.
    
    Can be added to an existing plot e.g. via 
    :func:`setAxisItems({'bottom':axis}) <pyqtgraph.PlotItem.setAxisItems>`.

    """

    def __init__(self, orientation='left', utcOffset=None, **kwargs):
        """
        Create a new NeuronsAxisItem.
        
        For `orientation` and `**kwargs`, see
        :func:`AxisItem.__init__ <pyqtgraph.AxisItem.__init__>`.
        
        """

        super(NeuronsAxisItem, self).__init__(orientation, **kwargs)
        # Set the zoom level to use depending on the time density on the axis
        if utcOffset is None:
            utcOffset = getOffsetFromUtc()
        self.utcOffset = utcOffset
        
        self.zoomLevels = OrderedDict([
            (np.inf,      YEAR_MONTH_ZOOM_LEVEL),
            (5 * 3600*24, MONTH_DAY_ZOOM_LEVEL),
            (6 * 3600,    DAY_HOUR_ZOOM_LEVEL),
            (15 * 60,     HOUR_MINUTE_ZOOM_LEVEL),
            (30,          HMS_ZOOM_LEVEL),
            (1,           MS_ZOOM_LEVEL),
            ])
        self.autoSIPrefix = False
    
    def tickStrings(self, values, scale, spacing):
        tickSpecs = self.zoomLevel.tickSpecs
        tickSpec = next((s for s in tickSpecs if s.spacing == spacing), None)
        try:
            dates = [utcfromtimestamp(v - self.utcOffset) for v in values]
        except (OverflowError, ValueError, OSError):
            # should not normally happen
            return ['%g' % ((v-self.utcOffset)//SEC_PER_YEAR + 1970) for v in values]
            
        formatStrings = []
        for x in dates:
            try:
                s = x.strftime(tickSpec.format)
                if '%f' in tickSpec.format:
                    # we only support ms precision
                    s = s[:-3]
                elif '%Y' in tickSpec.format:
                    s = s.lstrip('0')
                formatStrings.append(s)
            except ValueError:  # Windows can't handle dates before 1970
                formatStrings.append('')
        return formatStrings

    def tickValues(self, minVal, maxVal, size):
        density = (maxVal - minVal) / size
        self.setZoomLevelForDensity(density)
        values = self.zoomLevel.tickValues(minVal, maxVal, minSpc=self.minSpacing)
        return values

    def setZoomLevelForDensity(self, density):
        """
        Setting `zoomLevel` and `minSpacing` based on given density of seconds per pixel
        
        The display format is adjusted automatically depending on the current time
        density (seconds/point) on the axis. You can customize the behaviour by 
        overriding this function or setting a different set of zoom levels
        than the default one. The `zoomLevels` variable is a dictionary with the
        maximal distance of ticks in seconds which are allowed for each zoom level
        before the axis switches to the next coarser level. To customize the zoom level
        selection, override this function.
        """
        padding = 10
        
        # Size in pixels a specific tick label will take
        if self.orientation in ['bottom', 'top']:
            def sizeOf(text):
                return self.fontMetrics.boundingRect(text).width() + padding
        else:
            def sizeOf(text):
                return self.fontMetrics.boundingRect(text).height() + padding
        
        # Fallback zoom level: Years/Months
        self.zoomLevel = YEAR_MONTH_ZOOM_LEVEL
        for maximalSpacing, zoomLevel in self.zoomLevels.items():
            size = sizeOf(zoomLevel.exampleText)

            # Test if zoom level is too fine grained
            if maximalSpacing/size < density:
                break
            
            self.zoomLevel = zoomLevel
        
        # Set up zoomLevel
        self.zoomLevel.utcOffset = self.utcOffset
        
        # Calculate minimal spacing of items on the axis
        size = sizeOf(self.zoomLevel.exampleText)
        self.minSpacing = density*size
        
    def linkToView(self, view):
        """Link this axis to a ViewBox, causing its displayed range to match the visible range of the view."""
        self._linkToView_internal(view) # calls original linkToView code
        
        # Set default limits
        _min = MIN_REGULAR_TIMESTAMP
        _max = MAX_REGULAR_TIMESTAMP
        
        if self.orientation in ['right', 'left']:
            view.setLimits(yMin=_min, yMax=_max)
        else:
            view.setLimits(xMin=_min, xMax=_max)
        
    def generateDrawSpecs(self, p):
        # Get font metrics from QPainter
        # Not happening in "paint", as the QPainter p there is a different one from the one here,
        # so changing that font could cause unwanted side effects
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        
        self.fontMetrics = p.fontMetrics()
        
        # Get font scale factor by current window resolution
        
        return super(NeuronsAxisItem, self).generateDrawSpecs(p)

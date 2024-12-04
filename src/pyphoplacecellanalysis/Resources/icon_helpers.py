from pyphoplacecellanalysis.External.breeze_style_sheets import breeze_resources as breeze
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons, silx_resources_rc

import pyphoplacecellanalysis.External.pyqtgraph as pg

def try_get_icon(icon_path: str):
    """ 
    Usage:
		from pyphoplacecellanalysis.Resources.icon_helpers import try_get_icon
		icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico")
		if icon is not None:
			_out_rank_order_event_raster_debugger.ui.root_dockAreaWindow.setWindowIcon(icon)
    """
    if icon_path is not None:
        icon = pg.QtGui.QIcon()
        icon.addPixmap(pg.QtGui.QPixmap(icon_path), pg.QtGui.QIcon.Normal, pg.QtGui.QIcon.Off)
        return icon
    else:
        return None
    


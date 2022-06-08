# For qdarkstyle support:
import qdarkstyle
# For BreezeStylesheets support:
from qtpy import QtWidgets
from qtpy.QtCore import QFile, QTextStream
import pyphoplacecellanalysis.External.breeze_style_sheets.breeze_resources
# set stylesheet:
stylesheet_qss_file = QFile(":/dark/stylesheet.qss")
stylesheet_qss_file.open(QFile.ReadOnly | QFile.Text)
stylesheet_data_stream = QTextStream(stylesheet_qss_file)


class StylesheetThemingHelpers:
    """ Helper Qt Theme/Apperance class that uses tatic methods to apply the stylesheet to the passed object """
    @classmethod
    def get_breeze_theme(cls):
        return stylesheet_data_stream.readAll()

    @classmethod
    def get_qdarkstyle_theme(cls):
        raise NotImplementedError

    
    @classmethod
    def apply_breeze_theme(cls, object):
        # Load Stylesheet:
        object.setStyleSheet(stylesheet_data_stream.readAll())
  
    @classmethod
    def apply_qdarkstyle_theme(cls, object):
        # Load Stylesheet:
        # qdarkstyle.load_stylesheet_pyqt()
        raise NotImplementedError
        # object.setStyleSheet(stylesheet_data_stream.readAll())
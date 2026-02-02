from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QPushButton, QWidget, QVBoxLayout, QFrame
from PyQt5.QtCore import Qt, pyqtSignal

from pyphoplacecellanalysis.External.pyqt_color_picker.colorPickerWidget import ColorPickerWidget


class ColorPickerDialog(QDialog):
    """
        from pyphoplacecellanalysis.External.pyqt_color_picker.colorPickerDialog import ColorPickerDialog

    QColorDialog-compatible signals for drop-in use (e.g. ColorButton).
    """
    currentColorChanged = pyqtSignal(QColor)
    colorSelected = pyqtSignal(QColor)

    def __init__(self, color=QColor(255, 255, 255), orientation='horizontal', use_alpha=True):
        super().__init__()
        if isinstance(color, QColor):
            pass
        elif isinstance(color, str):
            color = QColor(color)
        self.__use_alpha = bool(use_alpha)
        self.__initUi(color=color, orientation=orientation)

    @property
    def use_alpha(self):
        return self.__use_alpha

    def __initUi(self, color, orientation):
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.MSWindowsFixedSizeDialogHint)
        self.setWindowTitle('Pick the color')

        self.__colorPickerWidget = ColorPickerWidget(color, orientation, use_alpha=self.__use_alpha)
        self.__colorPickerWidget.colorChanged.connect(self.currentColorChanged.emit)

        okBtn = QPushButton('OK')
        cancelBtn = QPushButton('Cancel')

        okBtn.clicked.connect(self.accept)
        cancelBtn.clicked.connect(self.reject)

        if orientation == 'horizontal':
            lay = QHBoxLayout()
            lay.addWidget(self.__colorPickerWidget)
            lay.setContentsMargins(0, 0, 0, 0)

            topWidget = QWidget()
            topWidget.setLayout(lay)

            lay = QHBoxLayout()
            lay.setAlignment(Qt.AlignRight)
            lay.addWidget(okBtn)
            lay.addWidget(cancelBtn)
            lay.setContentsMargins(0, 0, 0, 0)

            bottomWidget = QWidget()
            bottomWidget.setLayout(lay)

            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setFrameShadow(QFrame.Sunken)
            sep.setContentsMargins(0, 0, 0, 0)

            lay = QVBoxLayout()
            lay.addWidget(topWidget)
            lay.addWidget(sep)
            lay.addWidget(bottomWidget)
        elif orientation == 'vertical':
            lay = QHBoxLayout()
            lay.addWidget(self.__colorPickerWidget)
            lay.setContentsMargins(0, 0, 0, 0)

            leftWidget = QWidget()
            leftWidget.setLayout(lay)

            lay = QVBoxLayout()
            lay.setAlignment(Qt.AlignBottom)
            lay.addWidget(okBtn)
            lay.addWidget(cancelBtn)
            lay.setContentsMargins(0, 0, 0, 0)

            rightWidget = QWidget()
            rightWidget.setLayout(lay)

            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            sep.setFrameShadow(QFrame.Sunken)
            sep.setContentsMargins(0, 0, 0, 0)

            lay = QHBoxLayout()
            lay.addWidget(leftWidget)
            lay.addWidget(sep)
            lay.addWidget(rightWidget)

        self.setLayout(lay)

    def accept(self) -> None:
        self.colorSelected.emit(self.getColor())
        return super().accept()

    def setCurrentColor(self, color) -> None:
        if isinstance(color, QColor):
            pass
        elif isinstance(color, str):
            color = QColor(color)
        self.__colorPickerWidget.setCurrentColor(color)

    def currentColor(self) -> QColor:
        return self.__colorPickerWidget.getCurrentColor()

    def getColor(self) -> QColor:
        return self.__colorPickerWidget.getCurrentColor()

    # def getWidget(self):
    #     return self.__colorPickerWidget
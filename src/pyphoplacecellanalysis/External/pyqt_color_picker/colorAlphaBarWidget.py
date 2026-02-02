from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QSizePolicy
from PyQt5.QtGui import QColor


class ColorAlphaBarWidget(QWidget):
    """Vertical slider for alpha (0-255), intended to sit to the right of the hue bar."""
    alphaChanged = pyqtSignal(int)

    def __init__(self, color):
        super().__init__()
        self.__width = 20
        self.__height = 300
        self.__initUi(color)

    def __initUi(self, color):
        self.setFixedWidth(self.__width)
        self.setMinimumHeight(self.__height)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)

        self.__slider = QSlider(Qt.Vertical)
        self.__slider.setRange(0, 255)
        self.__slider.setValue(color.alpha() if isinstance(color, QColor) else 255)
        self.__slider.valueChanged.connect(self.alphaChanged.emit)

        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.__slider, 0, Qt.AlignHCenter)
        self.setLayout(lay)

    def setAlpha(self, alpha: int) -> None:
        alpha = max(0, min(255, int(alpha)))
        self.__slider.blockSignals(True)
        self.__slider.setValue(alpha)
        self.__slider.blockSignals(False)

    def getAlpha(self) -> int:
        return self.__slider.value()

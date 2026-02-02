from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSpinBox, QLineEdit, QMenu, QApplication,
)


class ColorEditorWidget(QWidget):
    colorChanged = pyqtSignal(QColor)

    def __init__(self, color, orientation, use_alpha=True):
        super().__init__()
        self.__current_color = color
        self.__use_alpha = bool(use_alpha)
        self.__initVal()
        self.__initUi(color, orientation)

    @property
    def use_alpha(self):
        return self.__use_alpha

    def __initVal(self):
        # default width and height
        self.__w = 200
        self.__h = 75

    def __hexString(self, color):
        """Format color as #RRGGBB or #RRGGBBAA depending on use_alpha."""
        r, g, b = color.red(), color.green(), color.blue()
        if self.__use_alpha:
            a = color.alpha()
            return f'#{r:02x}{g:02x}{b:02x}{a:02x}'
        return f'#{r:02x}{g:02x}{b:02x}'

    def __initUi(self, color, orientation):
        self.__colorPreviewWithGraphics = QWidget()
        self.__colorPreviewWithGraphics.setFixedWidth(self.__w)
        self.__colorPreviewWithGraphics.setMinimumHeight(self.__h)
        self.setColorPreviewWithGraphics()

        self.__hLineEdit = QLineEdit()
        self.__hLineEdit.setReadOnly(True)

        self.__rSpinBox = QSpinBox()
        self.__gSpinBox = QSpinBox()
        self.__bSpinBox = QSpinBox()

        self.__rSpinBox.valueChanged.connect(self.__rColorChanged)
        self.__gSpinBox.valueChanged.connect(self.__gColorChanged)
        self.__bSpinBox.valueChanged.connect(self.__bColorChanged)

        self.__hLineEdit.setAlignment(Qt.AlignCenter)
        self.__hLineEdit.setFont(QFont('Arial', 12))

        spinBoxs = [self.__rSpinBox, self.__gSpinBox, self.__bSpinBox]
        for spinBox in spinBoxs:
            spinBox.setRange(0, 255)
            spinBox.setAlignment(Qt.AlignCenter)
            spinBox.setFont(QFont('Arial', 12))

        lay = QFormLayout()
        lay.addRow('#', self.__hLineEdit)
        lay.addRow('R', self.__rSpinBox)
        lay.addRow('G', self.__gSpinBox)
        lay.addRow('B', self.__bSpinBox)
        if self.__use_alpha:
            self.__aSpinBox = QSpinBox()
            self.__aSpinBox.setRange(0, 255)
            self.__aSpinBox.setAlignment(Qt.AlignCenter)
            self.__aSpinBox.setFont(QFont('Arial', 12))
            self.__aSpinBox.valueChanged.connect(self.__aColorChanged)
            lay.addRow('A', self.__aSpinBox)
        else:
            self.__aSpinBox = None
        lay.setContentsMargins(0, 0, 0, 0)

        # Right-click Copy for hex and all spinboxes
        copyable_widgets = [self.__hLineEdit, self.__rSpinBox, self.__gSpinBox, self.__bSpinBox]
        if self.__aSpinBox is not None:
            copyable_widgets.append(self.__aSpinBox)
        for w in copyable_widgets:
            w.setContextMenuPolicy(Qt.CustomContextMenu)
            w.customContextMenuRequested.connect(self.__showCopyContextMenu)

        colorEditor = QWidget()
        colorEditor.setLayout(lay)
        if orientation == 'horizontal':
            lay = QVBoxLayout()
        elif orientation == 'vertical':
            lay = QHBoxLayout()
        lay.addWidget(self.__colorPreviewWithGraphics)
        lay.addWidget(colorEditor)

        lay.setContentsMargins(0, 0, 0, 0)

        self.setLayout(lay)

        self.setCurrentColor(color)

    def setColorPreviewWithGraphics(self):
        fmt = QColor.HexArgb if self.__use_alpha else QColor.HexRgb
        self.__colorPreviewWithGraphics.setStyleSheet(f' border-radius: 5px; '
                                                      f'background-color: {self.__current_color.name(fmt)}; ')

    def setCurrentColor(self, color):
        self.__current_color = QColor(color)
        if not self.__use_alpha:
            self.__current_color.setAlpha(255)
        self.setColorPreviewWithGraphics()
        self.__hLineEdit.setText(self.__hexString(self.__current_color))

        # Prevent infinite valueChanged event loop
        self.__rSpinBox.valueChanged.disconnect(self.__rColorChanged)
        self.__gSpinBox.valueChanged.disconnect(self.__gColorChanged)
        self.__bSpinBox.valueChanged.disconnect(self.__bColorChanged)

        r = self.__current_color.red()
        g = self.__current_color.green()
        b = self.__current_color.blue()
        self.__rSpinBox.setValue(r)
        self.__gSpinBox.setValue(g)
        self.__bSpinBox.setValue(b)

        if self.__aSpinBox is not None:
            self.__aSpinBox.valueChanged.disconnect(self.__aColorChanged)
            self.__aSpinBox.setValue(self.__current_color.alpha())
            self.__aSpinBox.valueChanged.connect(self.__aColorChanged)

        self.__rSpinBox.valueChanged.connect(self.__rColorChanged)
        self.__gSpinBox.valueChanged.connect(self.__gColorChanged)
        self.__bSpinBox.valueChanged.connect(self.__bColorChanged)

    def __rColorChanged(self, r):
        self.__current_color.setRed(r)
        self.__procColorChanged()

    def __gColorChanged(self, g):
        self.__current_color.setGreen(g)
        self.__procColorChanged()

    def __bColorChanged(self, b):
        self.__current_color.setBlue(b)
        self.__procColorChanged()

    def __aColorChanged(self, a):
        self.__current_color.setAlpha(a)
        self.__procColorChanged()

    def __procColorChanged(self):
        if not self.__use_alpha:
            self.__current_color.setAlpha(255)
        self.__hLineEdit.setText(self.__hexString(self.__current_color))
        self.setColorPreviewWithGraphics()
        self.colorChanged.emit(QColor(self.__current_color))

    def __showCopyContextMenu(self, pos):
        widget = self.sender()
        if widget is None:
            return
        menu = QMenu(self)
        copy_act = menu.addAction("Copy")
        copy_act.triggered.connect(lambda checked, w=widget: self.__copyWidgetValue(w))
        menu.exec_(widget.mapToGlobal(pos))

    def __copyWidgetValue(self, widget):
        app = QApplication.instance()
        if app is None:
            return
        if widget == self.__hLineEdit:
            text = self.__hLineEdit.text()
        elif widget in (self.__rSpinBox, self.__gSpinBox, self.__bSpinBox):
            text = str(widget.value())
        elif widget == self.__aSpinBox:
            text = str(widget.value())
        else:
            return
        app.clipboard().setText(text)

    def getCurrentColor(self):
        return self.__current_color
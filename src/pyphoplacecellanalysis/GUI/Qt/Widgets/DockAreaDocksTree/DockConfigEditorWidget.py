import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QFormLayout, QCheckBox, QLineEdit, QComboBox, 
                            QGroupBox, QLabel)
from PyQt5.QtCore import Qt, QtCore

# Import the DockDisplayConfig class
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import DockDisplayConfig

class DockConfigOwningMixin:
    sigDockConfigChanged = QtCore.pyqtSignal(object) # (self)    
    sigSave = QtCore.pyqtSignal(str, str, str) # signal emitted when the mapping from the temporal window to the spatial layout is changed
    sigRefresh = QtCore.pyqtSignal(str, str, str) # signal emitted when the mapping from the temporal window to the spatial


class DockConfigEditor(DockConfigOwningMixin, QWidget):
    """ allows the user to display and edit a `DockDisplayConfig` via a GUI
    
    """
    def __init__(self, config=None):
        super().__init__()
        
        # Create a new config if none is provided
        self.config = config if config is not None else DockDisplayConfig()
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Dock Display Config Editor')
        self.resize(400, 500)
        
        main_layout = QVBoxLayout()
        
        # Boolean properties group
        bool_group = QGroupBox("Button Options")
        bool_layout = QFormLayout()
        
        # Create checkboxes for boolean properties
        self.close_btn_cb = QCheckBox()
        self.close_btn_cb.setChecked(self.config.showCloseButton)
        self.close_btn_cb.stateChanged.connect(self.updateBoolProperty('showCloseButton'))
        bool_layout.addRow("Show Close Button:", self.close_btn_cb)
        
        self.collapse_btn_cb = QCheckBox()
        self.collapse_btn_cb.setChecked(self.config.showCollapseButton)
        self.collapse_btn_cb.stateChanged.connect(self.updateBoolProperty('showCollapseButton'))
        bool_layout.addRow("Show Collapse Button:", self.collapse_btn_cb)
        
        self.group_btn_cb = QCheckBox()
        self.group_btn_cb.setChecked(self.config.showGroupButton)
        self.group_btn_cb.stateChanged.connect(self.updateBoolProperty('showGroupButton'))
        bool_layout.addRow("Show Group Button:", self.group_btn_cb)
        
        self.orientation_btn_cb = QCheckBox()
        self.orientation_btn_cb.setChecked(self.config.showOrientationButton)
        self.orientation_btn_cb.stateChanged.connect(self.updateBoolProperty('showOrientationButton'))
        bool_layout.addRow("Show Orientation Button:", self.orientation_btn_cb)
        
        self.hide_title_cb = QCheckBox()
        self.hide_title_cb.setChecked(self.config.hideTitleBar)
        self.hide_title_cb.stateChanged.connect(self.updateBoolProperty('hideTitleBar'))
        bool_layout.addRow("Hide Title Bar:", self.hide_title_cb)
        
        bool_group.setLayout(bool_layout)
        
        # String properties group
        string_group = QGroupBox("Appearance")
        string_layout = QFormLayout()
        
        self.font_size_edit = QLineEdit(self.config.fontSize)
        self.font_size_edit.textChanged.connect(self.updateStringProperty('fontSize'))
        string_layout.addRow("Font Size:", self.font_size_edit)
        
        self.corner_radius_edit = QLineEdit(self.config.corner_radius)
        self.corner_radius_edit.textChanged.connect(self.updateStringProperty('corner_radius'))
        string_layout.addRow("Corner Radius:", self.corner_radius_edit)
        
        # Orientation dropdown
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["auto", "horizontal", "vertical"])
        current_orientation = self.config.orientation
        if current_orientation == "horizontal" or current_orientation is None:
            self.orientation_combo.setCurrentIndex(1)
        elif current_orientation == "auto":
            self.orientation_combo.setCurrentIndex(0)
        else:  # vertical
            self.orientation_combo.setCurrentIndex(2)
        
        self.orientation_combo.currentTextChanged.connect(self.updateOrientation)
        string_layout.addRow("Orientation:", self.orientation_combo)
        
        string_group.setLayout(string_layout)
        
        # Preview section
        preview_group = QGroupBox("Preview (Configuration Output)")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setAlignment(Qt.AlignTop)
        self.updatePreview()
        
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        
        # Add all groups to main layout
        main_layout.addWidget(bool_group)
        main_layout.addWidget(string_group)
        main_layout.addWidget(preview_group)
        
        self.setLayout(main_layout)
        
    def updateBoolProperty(self, property_name):
        def update(state):
            setattr(self.config, property_name, state == Qt.Checked)
            self.updatePreview()
        return update
    
    def updateStringProperty(self, property_name):
        def update(text):
            setattr(self.config, property_name, text)
            self.updatePreview()
        return update
    
    def updateOrientation(self, text):
        self.config.orientation = text
        self.updatePreview()
    
    def updatePreview(self):
        # Display the current configuration as text
        props = [
            f"showCloseButton: {self.config.showCloseButton}",
            f"showCollapseButton: {self.config.showCollapseButton}",
            f"showGroupButton: {self.config.showGroupButton}",
            f"showOrientationButton: {self.config.showOrientationButton}",
            f"hideTitleBar: {self.config.hideTitleBar}",
            f"fontSize: {self.config.fontSize}",
            f"corner_radius: {self.config.corner_radius}",
            f"orientation: {self.config.orientation}",
            f"shouldAutoOrient: {self.config.shouldAutoOrient}"
        ]
        
        # Get stylesheet for both orientations to show in preview
        horizontal_stylesheet = self.config.get_stylesheet('horizontal', False)
        vertical_stylesheet = self.config.get_stylesheet('vertical', False)
        
        preview_text = "\n".join(props)
        preview_text += "\n\nHorizontal Stylesheet Sample:\n" + horizontal_stylesheet[:100] + "..."
        preview_text += "\n\nVertical Stylesheet Sample:\n" + vertical_stylesheet[:100] + "..."
        
        self.preview_label.setText(preview_text)



def main():
    app = QApplication(sys.argv)
    
    # Create a config instance
    config = DockDisplayConfig()
    
    # Create and show the editor
    editor = DockConfigEditor(config)
    editor.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QFormLayout, QCheckBox, QLineEdit, QComboBox, 
                            QGroupBox, QLabel, QTabWidget, QListWidget, QPushButton, QInputDialog)
from PyQt5.QtCore import Qt

# Import the DockDisplayConfig class
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import DockDisplayConfig
from pyqtgraph.widgets.ColorButton import ColorButton

from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig


class DockConfigOwningMixin:
    sigDockConfigChanged = QtCore.pyqtSignal(object) # (self) - signal emitted when any property of the internal dock config is modified.
    sigSave = QtCore.pyqtSignal(object) # signal emitted when the user wants to finalize and commit the changes to the config
    sigRevert = QtCore.pyqtSignal(object) # signal emitted when the user wants to revert the changes to the config back to the pre-edited values
    

class DockConfigEditor(DockConfigOwningMixin, QWidget):
    """ allows the user to display and edit a `DockDisplayConfig` via a GUI
    
    """
    sigDockConfigChanged = QtCore.pyqtSignal(object) # (self) - signal emitted when any property of the internal dock config is modified.
    sigSave = QtCore.pyqtSignal(object) # signal emitted when the user wants to finalize and commit the changes to the config
    sigRevert = QtCore.pyqtSignal(object) # signal emitted when the user wants to revert the changes to the config back to the pre-edited values    

    def __init__(self, config=None):
        super().__init__()
        
        # Check which type of config we have
        if config is None:
            self.config = DockDisplayConfig()
            self.is_custom_config = False
        elif isinstance(config, CustomDockDisplayConfig):
            self.config = config
            self.is_custom_config = True
        else:
            self.config = config
            self.is_custom_config = False
            
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Dock Display Config Editor')
        self.resize(600, 700)
        
        main_layout = QVBoxLayout()
        
        # Create the tab widget
        self.tab_widget = QTabWidget()
        self.base_tab = QWidget()
        
        # Base properties tab
        base_layout = QVBoxLayout()
        
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
        
        # Add groups to base layout
        base_layout.addWidget(bool_group)
        base_layout.addWidget(string_group)
        self.base_tab.setLayout(base_layout)
        
        # Add base tab
        self.tab_widget.addTab(self.base_tab, "Base Config")
        
        # Add custom config tab if applicable
        if self.is_custom_config:
            self.custom_tab = QWidget()
            custom_layout = QVBoxLayout()
            
            # Color configuration group
            color_group = QGroupBox("Color Configuration")
            color_layout = QVBoxLayout()
            
            # Normal and Dim states in separate groups
            states_tab = QTabWidget()
            
            # Normal state (is_dim=False)
            normal_tab = QWidget()
            normal_layout = QFormLayout()
            
            self.normal_colors = self.config.custom_get_colors_dict[False]
            
            self.normal_fg_btn = ColorButton(self, color=self.normal_colors.fg_color)
            self.normal_bg_btn = ColorButton(self, color=self.normal_colors.bg_color)
            self.normal_border_btn = ColorButton(self, color=self.normal_colors.border_color)
            
            normal_layout.addRow("Foreground Color:", self.normal_fg_btn)
            normal_layout.addRow("Background Color:", self.normal_bg_btn)
            normal_layout.addRow("Border Color:", self.normal_border_btn)
            
            normal_tab.setLayout(normal_layout)
            states_tab.addTab(normal_tab, "Normal State")
            
            # Dim state (is_dim=True)
            dim_tab = QWidget()
            dim_layout = QFormLayout()
            
            self.dim_colors = self.config.custom_get_colors_dict[True]
            
            self.dim_fg_btn = ColorButton(self, color=self.dim_colors.fg_color)
            self.dim_bg_btn = ColorButton(self, color=self.dim_colors.bg_color)
            self.dim_border_btn = ColorButton(self, color=self.dim_colors.border_color)
            
            dim_layout.addRow("Foreground Color:", self.dim_fg_btn)
            dim_layout.addRow("Background Color:", self.dim_bg_btn)
            dim_layout.addRow("Border Color:", self.dim_border_btn)
            
            dim_tab.setLayout(dim_layout)
            states_tab.addTab(dim_tab, "Dim State")
            
            color_layout.addWidget(states_tab)
            color_group.setLayout(color_layout)
            
            # Dock Group Names
            group_names_group = QGroupBox("Dock Group Names")
            group_names_layout = QVBoxLayout()
            
            self.group_names_list = QListWidget()
            for group_name in self.config.dock_group_names:
                self.group_names_list.addItem(group_name)
            
            group_names_buttons = QHBoxLayout()
            add_group_btn = QPushButton("Add Group")
            add_group_btn.clicked.connect(self.addDockGroup)
            remove_group_btn = QPushButton("Remove Group")
            remove_group_btn.clicked.connect(self.removeDockGroup)
            
            group_names_buttons.addWidget(add_group_btn)
            group_names_buttons.addWidget(remove_group_btn)
            
            group_names_layout.addWidget(self.group_names_list)
            group_names_layout.addLayout(group_names_buttons)
            
            group_names_group.setLayout(group_names_layout)
            
            # Add groups to custom layout
            custom_layout.addWidget(color_group)
            custom_layout.addWidget(group_names_group)
            
            self.custom_tab.setLayout(custom_layout)
            self.tab_widget.addTab(self.custom_tab, "Custom Config")
        
        # Preview section
        preview_group = QGroupBox("Preview (Configuration Output)")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setAlignment(Qt.AlignTop)
        self.updatePreview()
        
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        
        # Add tab widget and preview to main layout
        main_layout.addWidget(self.tab_widget)
        main_layout.addWidget(preview_group)
        
        self.setLayout(main_layout)

        # self.setWindowTitle('Dock Display Config Editor')
        # self.resize(400, 500)
        
        # main_layout = QVBoxLayout()
        
        # # Boolean properties group
        # bool_group = QGroupBox("Button Options")
        # bool_layout = QFormLayout()
        
        # # Create checkboxes for boolean properties
        # self.close_btn_cb = QCheckBox()
        # self.close_btn_cb.setChecked(self.config.showCloseButton)
        # self.close_btn_cb.stateChanged.connect(self.updateBoolProperty('showCloseButton'))
        # bool_layout.addRow("Show Close Button:", self.close_btn_cb)
        
        # self.collapse_btn_cb = QCheckBox()
        # self.collapse_btn_cb.setChecked(self.config.showCollapseButton)
        # self.collapse_btn_cb.stateChanged.connect(self.updateBoolProperty('showCollapseButton'))
        # bool_layout.addRow("Show Collapse Button:", self.collapse_btn_cb)
        
        # self.group_btn_cb = QCheckBox()
        # self.group_btn_cb.setChecked(self.config.showGroupButton)
        # self.group_btn_cb.stateChanged.connect(self.updateBoolProperty('showGroupButton'))
        # bool_layout.addRow("Show Group Button:", self.group_btn_cb)
        
        # self.orientation_btn_cb = QCheckBox()
        # self.orientation_btn_cb.setChecked(self.config.showOrientationButton)
        # self.orientation_btn_cb.stateChanged.connect(self.updateBoolProperty('showOrientationButton'))
        # bool_layout.addRow("Show Orientation Button:", self.orientation_btn_cb)
        
        # self.hide_title_cb = QCheckBox()
        # self.hide_title_cb.setChecked(self.config.hideTitleBar)
        # self.hide_title_cb.stateChanged.connect(self.updateBoolProperty('hideTitleBar'))
        # bool_layout.addRow("Hide Title Bar:", self.hide_title_cb)
        
        # bool_group.setLayout(bool_layout)
        
        # # String properties group
        # string_group = QGroupBox("Appearance")
        # string_layout = QFormLayout()
        
        # self.font_size_edit = QLineEdit(self.config.fontSize)
        # self.font_size_edit.textChanged.connect(self.updateStringProperty('fontSize'))
        # string_layout.addRow("Font Size:", self.font_size_edit)
        
        # self.corner_radius_edit = QLineEdit(self.config.corner_radius)
        # self.corner_radius_edit.textChanged.connect(self.updateStringProperty('corner_radius'))
        # string_layout.addRow("Corner Radius:", self.corner_radius_edit)
        
        # # Orientation dropdown
        # self.orientation_combo = QComboBox()
        # self.orientation_combo.addItems(["auto", "horizontal", "vertical"])
        # current_orientation = self.config.orientation
        # if current_orientation == "horizontal" or current_orientation is None:
        #     self.orientation_combo.setCurrentIndex(1)
        # elif current_orientation == "auto":
        #     self.orientation_combo.setCurrentIndex(0)
        # else:  # vertical
        #     self.orientation_combo.setCurrentIndex(2)
        
        # self.orientation_combo.currentTextChanged.connect(self.updateOrientation)
        # string_layout.addRow("Orientation:", self.orientation_combo)
        
        # string_group.setLayout(string_layout)
        
        # # Preview section
        # preview_group = QGroupBox("Preview (Configuration Output)")
        # preview_layout = QVBoxLayout()
        
        # self.preview_label = QLabel()
        # self.preview_label.setWordWrap(True)
        # self.preview_label.setAlignment(Qt.AlignTop)
        # self.updatePreview()
        
        # preview_layout.addWidget(self.preview_label)
        # preview_group.setLayout(preview_layout)
        
        # # Add all groups to main layout
        # main_layout.addWidget(bool_group)
        # main_layout.addWidget(string_group)
        # main_layout.addWidget(preview_group)
        
        # self.setLayout(main_layout)

        
    def updateBoolProperty(self, property_name):
        def update(state):
            setattr(self.config, property_name, state == Qt.Checked)
            self.updatePreview()
              self.sigDockConfigChanged.emit(self)
        return update
    
    def updateStringProperty(self, property_name):
        def update(text):
            setattr(self.config, property_name, text)
            self.updatePreview()
              self.sigDockConfigChanged.emit(self)
        return update
    
    def updateOrientation(self, text):
        self.config.orientation = text
        self.updatePreview()
          self.sigDockConfigChanged.emit(self)
    
    def updateColors(self):
        if self.is_custom_config:
            # Update normal state colors
            self.normal_colors.fg_color = self.normal_fg_btn.color
            self.normal_colors.bg_color = self.normal_bg_btn.color
            self.normal_colors.border_color = self.normal_border_btn.color
            
            # Update dim state colors
            self.dim_colors.fg_color = self.dim_fg_btn.color
            self.dim_colors.bg_color = self.dim_bg_btn.color
            self.dim_colors.border_color = self.dim_border_btn.color
            
            # Update the config
            self.config.custom_get_colors_dict[False] = self.normal_colors
            self.config.custom_get_colors_dict[True] = self.dim_colors
            
            self.updatePreview()
              self.sigDockConfigChanged.emit(self)
    
    def addDockGroup(self):
        group_name, ok = QInputDialog.getText(self, "Add Dock Group", "Group Name:")
        if ok and group_name:
            if group_name not in self.config.dock_group_names:
                self.config.dock_group_names.append(group_name)
                self.group_names_list.addItem(group_name)
                self.updatePreview()
                  self.sigDockConfigChanged.emit(self)
    
    def removeDockGroup(self):
        selected_items = self.group_names_list.selectedItems()
        if selected_items:
            for item in selected_items:
                group_name = item.text()
                if group_name in self.config.dock_group_names:
                    self.config.dock_group_names.remove(group_name)
                    self.group_names_list.takeItem(self.group_names_list.row(item))
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
        
        # Add custom properties if applicable
        if self.is_custom_config:
            props.append("\nCustom Config Properties:")
            props.append(f"dock_group_names: {self.config.dock_group_names}")
            
            # Add color information
            props.append("\nNormal State Colors:")
            normal_colors = self.config.custom_get_colors_dict[False]
            props.append(f"  fg_color: {normal_colors.fg_color}")
            props.append(f"  bg_color: {normal_colors.bg_color}")
            props.append(f"  border_color: {normal_colors.border_color}")
            
            props.append("\nDim State Colors:")
            dim_colors = self.config.custom_get_colors_dict[True]
            props.append(f"  fg_color: {dim_colors.fg_color}")
            props.append(f"  bg_color: {dim_colors.bg_color}")
            props.append(f"  border_color: {dim_colors.border_color}")
        
        # Get stylesheet for both orientations to show in preview
        horizontal_stylesheet = self.config.get_stylesheet('horizontal', False)
        vertical_stylesheet = self.config.get_stylesheet('vertical', False)
        
        preview_text = "\n".join(props)
        preview_text += "\n\nHorizontal Stylesheet Sample:\n" + horizontal_stylesheet[:100] + "..."
        preview_text += "\n\nVertical Stylesheet Sample:\n" + vertical_stylesheet[:100] + "..."
        
        self.preview_label.setText(preview_text)



def main():
    app = QApplication(sys.argv)
    
    # Create a config instance - you can choose either type
    # For basic config:
    # config = DockDisplayConfig()
    
    # For custom config with colors:
    config = CustomDockDisplayConfig(
        showCloseButton=True,
        showCollapseButton=True,
        fontSize='12px',
        corner_radius='4px',
        orientation='auto'
    )
    
    # Create and show the editor
    editor = DockConfigEditor(config)
    editor.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
              self.sigDockConfigChanged.emit(self)

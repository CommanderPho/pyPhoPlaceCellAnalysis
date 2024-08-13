# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\DockPlanningHelperWidget\DockPlanningHelperWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 310)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(-1, -1, 0, 0)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName("gridLayout")
        self.btnAddWidgetRight = QtWidgets.QToolButton(Form)
        self.btnAddWidgetRight.setArrowType(QtCore.Qt.RightArrow)
        self.btnAddWidgetRight.setObjectName("btnAddWidgetRight")
        self.gridLayout.addWidget(self.btnAddWidgetRight, 6, 1, 1, 1)
        self.line = QtWidgets.QFrame(Form)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 4, 0, 1, 1)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.formLayout.setVerticalSpacing(0)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.txtDockTitle = QtWidgets.QLineEdit(Form)
        self.txtDockTitle.setObjectName("txtDockTitle")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.txtDockTitle)
        self.txtDockIdentifier = QtWidgets.QLineEdit(Form)
        self.txtDockIdentifier.setReadOnly(True)
        self.txtDockIdentifier.setObjectName("txtDockIdentifier")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.txtDockIdentifier)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.horizontalLayout_Size = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Size.setSpacing(3)
        self.horizontalLayout_Size.setObjectName("horizontalLayout_Size")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_Size.addItem(spacerItem)
        self.spinBox_Width = QtWidgets.QSpinBox(Form)
        self.spinBox_Width.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.spinBox_Width.setToolTip("DockWidth")
        self.spinBox_Width.setStatusTip("DockWidth")
        self.spinBox_Width.setFrame(False)
        self.spinBox_Width.setReadOnly(True)
        self.spinBox_Width.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_Width.setKeyboardTracking(False)
        self.spinBox_Width.setMinimum(1)
        self.spinBox_Width.setMaximum(2000)
        self.spinBox_Width.setSingleStep(100)
        self.spinBox_Width.setProperty("value", 600)
        self.spinBox_Width.setObjectName("spinBox_Width")
        self.horizontalLayout_Size.addWidget(self.spinBox_Width)
        self.spinBox_Height = QtWidgets.QSpinBox(Form)
        self.spinBox_Height.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.spinBox_Height.setToolTip("Height")
        self.spinBox_Height.setStatusTip("Height")
        self.spinBox_Height.setFrame(False)
        self.spinBox_Height.setReadOnly(True)
        self.spinBox_Height.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_Height.setKeyboardTracking(False)
        self.spinBox_Height.setMinimum(1)
        self.spinBox_Height.setMaximum(8000)
        self.spinBox_Height.setProperty("value", 300)
        self.spinBox_Height.setObjectName("spinBox_Height")
        self.horizontalLayout_Size.addWidget(self.spinBox_Height)
        self.horizontalLayout_Size.setStretch(0, 1)
        self.horizontalLayout_Size.setStretch(1, 1)
        self.horizontalLayout_Size.setStretch(2, 1)
        self.formLayout.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_Size)
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setContentsMargins(2, 0, 2, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radBtnOrientation_Auto = QtWidgets.QRadioButton(self.groupBox)
        self.radBtnOrientation_Auto.setChecked(True)
        self.radBtnOrientation_Auto.setObjectName("radBtnOrientation_Auto")
        self.horizontalLayout.addWidget(self.radBtnOrientation_Auto)
        self.radBtnOrientation_Horizontal = QtWidgets.QRadioButton(self.groupBox)
        self.radBtnOrientation_Horizontal.setObjectName("radBtnOrientation_Horizontal")
        self.horizontalLayout.addWidget(self.radBtnOrientation_Horizontal)
        self.radBtnOrientation_Vertical = QtWidgets.QRadioButton(self.groupBox)
        self.radBtnOrientation_Vertical.setObjectName("radBtnOrientation_Vertical")
        self.horizontalLayout.addWidget(self.radBtnOrientation_Vertical)
        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 1)
        self.txtExtendedLabel = QtWidgets.QPlainTextEdit(Form)
        self.txtExtendedLabel.setEnabled(True)
        self.txtExtendedLabel.setUndoRedoEnabled(True)
        self.txtExtendedLabel.setReadOnly(True)
        self.txtExtendedLabel.setObjectName("txtExtendedLabel")
        self.gridLayout.addWidget(self.txtExtendedLabel, 5, 0, 1, 1)
        self.lblInfoTextLine = QtWidgets.QLabel(Form)
        self.lblInfoTextLine.setText("")
        self.lblInfoTextLine.setObjectName("lblInfoTextLine")
        self.gridLayout.addWidget(self.lblInfoTextLine, 2, 0, 1, 1)
        self.btnAddWidgetBelow = QtWidgets.QToolButton(Form)
        self.btnAddWidgetBelow.setArrowType(QtCore.Qt.DownArrow)
        self.btnAddWidgetBelow.setObjectName("btnAddWidgetBelow")
        self.gridLayout.addWidget(self.btnAddWidgetBelow, 6, 0, 1, 1)
        self.groupBoxActions = QtWidgets.QGroupBox(Form)
        self.groupBoxActions.setObjectName("groupBoxActions")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBoxActions)
        self.horizontalLayout_2.setContentsMargins(-1, 2, -1, 2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btnRefresh = QtWidgets.QPushButton(self.groupBoxActions)
        self.btnRefresh.setObjectName("btnRefresh")
        self.horizontalLayout_2.addWidget(self.btnRefresh)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.btnLog = QtWidgets.QPushButton(self.groupBoxActions)
        self.btnLog.setObjectName("btnLog")
        self.horizontalLayout_2.addWidget(self.btnLog)
        self.btnSave = QtWidgets.QPushButton(self.groupBoxActions)
        self.btnSave.setObjectName("btnSave")
        self.horizontalLayout_2.addWidget(self.btnSave)
        self.gridLayout.addWidget(self.groupBoxActions, 3, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setRowStretch(0, 1)
        self.actionActionEditingFinished = QtWidgets.QAction(Form)
        self.actionActionEditingFinished.setObjectName("actionActionEditingFinished")
        self.label.setBuddy(self.txtDockTitle)
        self.label_2.setBuddy(self.txtDockIdentifier)
        self.label_3.setBuddy(self.spinBox_Width)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.txtDockTitle, self.txtDockIdentifier)
        Form.setTabOrder(self.txtDockIdentifier, self.spinBox_Width)
        Form.setTabOrder(self.spinBox_Width, self.spinBox_Height)
        Form.setTabOrder(self.spinBox_Height, self.radBtnOrientation_Auto)
        Form.setTabOrder(self.radBtnOrientation_Auto, self.btnSave)
        Form.setTabOrder(self.btnSave, self.btnAddWidgetBelow)
        Form.setTabOrder(self.btnAddWidgetBelow, self.btnAddWidgetRight)
        Form.setTabOrder(self.btnAddWidgetRight, self.radBtnOrientation_Horizontal)
        Form.setTabOrder(self.radBtnOrientation_Horizontal, self.btnLog)
        Form.setTabOrder(self.btnLog, self.radBtnOrientation_Vertical)
        Form.setTabOrder(self.radBtnOrientation_Vertical, self.txtExtendedLabel)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.btnAddWidgetRight.setToolTip(_translate("Form", "Add DockPlanningHelperWidget right"))
        self.btnAddWidgetRight.setText(_translate("Form", "..."))
        self.label.setText(_translate("Form", "Title"))
        self.txtDockTitle.setText(_translate("Form", "Position Decoder"))
        self.txtDockTitle.setPlaceholderText(_translate("Form", "Position Decoder"))
        self.txtDockIdentifier.setText(_translate("Form", "Position Decoder"))
        self.txtDockIdentifier.setPlaceholderText(_translate("Form", "PositionDecoder"))
        self.label_2.setText(_translate("Form", "Identifer"))
        self.label_3.setText(_translate("Form", "dockSize"))
        self.groupBox.setTitle(_translate("Form", "Orientation"))
        self.radBtnOrientation_Auto.setText(_translate("Form", "Auto"))
        self.radBtnOrientation_Horizontal.setText(_translate("Form", "Horizontal"))
        self.radBtnOrientation_Vertical.setText(_translate("Form", "Vertical"))
        self.btnAddWidgetBelow.setToolTip(_translate("Form", "Add DockPlanningHelperWidget below"))
        self.btnAddWidgetBelow.setText(_translate("Form", "..."))
        self.groupBoxActions.setTitle(_translate("Form", "Actions"))
        self.btnRefresh.setText(_translate("Form", "Refresh"))
        self.btnLog.setText(_translate("Form", "Log"))
        self.btnSave.setText(_translate("Form", "Save"))
        self.actionActionEditingFinished.setText(_translate("Form", "actionEditingFinished"))
        self.actionActionEditingFinished.setToolTip(_translate("Form", "Editing Finished"))

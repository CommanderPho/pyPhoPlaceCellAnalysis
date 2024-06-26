# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\DecoderPlotSelectorControls\DecoderPlotSelectorWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(394, 300)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(-1, -1, -1, 2)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.cmbDecoder = QtWidgets.QComboBox(Form)
        self.cmbDecoder.setObjectName("cmbDecoder")
        self.cmbDecoder.addItem("")
        self.cmbDecoder.addItem("")
        self.horizontalLayout.addWidget(self.cmbDecoder)
        spacerItem = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.cmbVariableName = QtWidgets.QComboBox(Form)
        self.cmbVariableName.setObjectName("cmbVariableName")
        self.cmbVariableName.addItem("")
        self.cmbVariableName.setItemText(0, "p_x_given_n")
        self.cmbVariableName.addItem("")
        self.cmbVariableName.setItemText(1, "p_x_given_n_and_x_prev")
        self.horizontalLayout.addWidget(self.cmbVariableName)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.decoderPlotContainerWidget = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.decoderPlotContainerWidget.sizePolicy().hasHeightForWidth())
        self.decoderPlotContainerWidget.setSizePolicy(sizePolicy)
        self.decoderPlotContainerWidget.setMinimumSize(QtCore.QSize(200, 100))
        self.decoderPlotContainerWidget.setObjectName("decoderPlotContainerWidget")
        self.gridLayout.addWidget(self.decoderPlotContainerWidget, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Decoder"))
        self.cmbDecoder.setCurrentText(_translate("Form", "one-step"))
        self.cmbDecoder.setItemText(0, _translate("Form", "one-step"))
        self.cmbDecoder.setItemText(1, _translate("Form", "two-step"))
        self.label_2.setText(_translate("Form", "Variable"))
        self.cmbVariableName.setToolTip(_translate("Form", "Corresponds to _temp_debug_two_step_plots_animated_imshow\'s variable_name arg"))
        self.cmbVariableName.setCurrentText(_translate("Form", "p_x_given_n"))

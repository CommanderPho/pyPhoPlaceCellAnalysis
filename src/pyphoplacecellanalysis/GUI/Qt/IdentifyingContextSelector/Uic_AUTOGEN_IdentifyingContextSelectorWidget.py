# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\IdentifyingContextSelector\IdentifyingContextSelectorWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(393, 348)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cmbIdentifyingContext = QtWidgets.QComboBox(self.groupBox)
        self.cmbIdentifyingContext.setObjectName("cmbIdentifyingContext")
        self.horizontalLayout.addWidget(self.cmbIdentifyingContext)
        self.btnRevert = QtWidgets.QToolButton(self.groupBox)
        self.btnRevert.setObjectName("btnRevert")
        self.horizontalLayout.addWidget(self.btnRevert)
        self.btnConfirm = QtWidgets.QPushButton(self.groupBox)
        self.btnConfirm.setObjectName("btnConfirm")
        self.horizontalLayout.addWidget(self.btnConfirm)
        self.verticalLayout.addWidget(self.groupBox)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "Identifying Context"))
        self.btnRevert.setText(_translate("Form", "Revert"))
        self.btnConfirm.setText(_translate("Form", "Change"))

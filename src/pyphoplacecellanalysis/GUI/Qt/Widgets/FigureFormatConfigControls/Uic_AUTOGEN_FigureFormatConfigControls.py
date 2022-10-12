# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\FigureFormatConfigControls\FigureFormatConfigControls.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(536, 753)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/light/file_dialog_detailed.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tupleCtrl_0 = SingleGroupOptionalMembersCtrl(Form)
        self.tupleCtrl_0.setObjectName("tupleCtrl_0")
        self.verticalLayout.addWidget(self.tupleCtrl_0)
        self.tupleCtrl_1 = SingleGroupOptionalMembersCtrl(Form)
        self.tupleCtrl_1.setObjectName("tupleCtrl_1")
        self.verticalLayout.addWidget(self.tupleCtrl_1)
        self.tupleCtrl_2 = SingleGroupOptionalMembersCtrl(Form)
        self.tupleCtrl_2.setObjectName("tupleCtrl_2")
        self.verticalLayout.addWidget(self.tupleCtrl_2)
        self.line = QtWidgets.QFrame(Form)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.horizontalFrame_2 = QtWidgets.QFrame(Form)
        self.horizontalFrame_2.setObjectName("horizontalFrame_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalFrame_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalWidget = QtWidgets.QWidget(self.horizontalFrame_2)
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.horizontalWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.chkEnableSpikeOverlay = QtWidgets.QCheckBox(self.horizontalWidget)
        self.chkEnableSpikeOverlay.setObjectName("chkEnableSpikeOverlay")
        self.verticalLayout_2.addWidget(self.chkEnableSpikeOverlay)
        self.chkDebugPrint = QtWidgets.QCheckBox(self.horizontalWidget)
        self.chkDebugPrint.setObjectName("chkDebugPrint")
        self.verticalLayout_2.addWidget(self.chkDebugPrint)
        self.chkEnableSavingToDisk = QtWidgets.QCheckBox(self.horizontalWidget)
        self.chkEnableSavingToDisk.setChecked(True)
        self.chkEnableSavingToDisk.setObjectName("chkEnableSavingToDisk")
        self.verticalLayout_2.addWidget(self.chkEnableSavingToDisk)
        self.horizontalLayout_3.addWidget(self.horizontalWidget)
        self.frame = QtWidgets.QFrame(self.horizontalFrame_2)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.frame)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 333, 382))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.txtEditExtraArguments = QtWidgets.QPlainTextEdit(self.scrollAreaWidgetContents)
        self.txtEditExtraArguments.setPlainText("")
        self.txtEditExtraArguments.setBackgroundVisible(False)
        self.txtEditExtraArguments.setObjectName("txtEditExtraArguments")
        self.gridLayout_3.addWidget(self.txtEditExtraArguments, 0, 0, 1, 1)
        self.codeConsoleWidget = PhoCodeConsoleWidget(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.codeConsoleWidget.sizePolicy().hasHeightForWidth())
        self.codeConsoleWidget.setSizePolicy(sizePolicy)
        self.codeConsoleWidget.setMinimumSize(QtCore.QSize(0, 200))
        self.codeConsoleWidget.setBaseSize(QtCore.QSize(300, 200))
        self.codeConsoleWidget.setObjectName("codeConsoleWidget")
        self.gridLayout_3.addWidget(self.codeConsoleWidget, 1, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_2.addWidget(self.scrollArea, 1, 0, 1, 1)
        self.horizontalLayout_3.addWidget(self.frame)
        self.verticalLayout.addWidget(self.horizontalFrame_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.toolBox = QtWidgets.QToolBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolBox.sizePolicy().hasHeightForWidth())
        self.toolBox.setSizePolicy(sizePolicy)
        self.toolBox.setBaseSize(QtCore.QSize(400, 200))
        self.toolBox.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.toolBox.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.toolBox.setLineWidth(2)
        self.toolBox.setMidLineWidth(1)
        self.toolBox.setObjectName("toolBox")
        self.page_figExport = QtWidgets.QWidget()
        self.page_figExport.setGeometry(QtCore.QRect(0, 0, 516, 69))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.page_figExport.sizePolicy().hasHeightForWidth())
        self.page_figExport.setSizePolicy(sizePolicy)
        self.page_figExport.setObjectName("page_figExport")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.page_figExport)
        self.horizontalLayout.setContentsMargins(4, 0, 4, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.filepkr_FigureOutputPath = InlineFilesystemPathSelectWidget(self.page_figExport)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filepkr_FigureOutputPath.sizePolicy().hasHeightForWidth())
        self.filepkr_FigureOutputPath.setSizePolicy(sizePolicy)
        self.filepkr_FigureOutputPath.setObjectName("filepkr_FigureOutputPath")
        self.verticalLayout_3.addWidget(self.filepkr_FigureOutputPath)
        self.filepkr_ProgrammaticDisplayFcnOutputPath = InlineFilesystemPathSelectWidget(self.page_figExport)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filepkr_ProgrammaticDisplayFcnOutputPath.sizePolicy().hasHeightForWidth())
        self.filepkr_ProgrammaticDisplayFcnOutputPath.setSizePolicy(sizePolicy)
        self.filepkr_ProgrammaticDisplayFcnOutputPath.setObjectName("filepkr_ProgrammaticDisplayFcnOutputPath")
        self.verticalLayout_3.addWidget(self.filepkr_ProgrammaticDisplayFcnOutputPath)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/light/desktop.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBox.addItem(self.page_figExport, icon1, "")
        self.page_debugging = QtWidgets.QWidget()
        self.page_debugging.setGeometry(QtCore.QRect(0, 0, 516, 207))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.page_debugging.sizePolicy().hasHeightForWidth())
        self.page_debugging.setSizePolicy(sizePolicy)
        self.page_debugging.setObjectName("page_debugging")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.page_debugging)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.textBrowser = QtWidgets.QTextBrowser(self.page_debugging)
        self.textBrowser.setObjectName("textBrowser")
        self.gridLayout.addWidget(self.textBrowser, 1, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.page_debugging)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 2, 0, 1, 1)
        self.lineEdit = FilesystemPathLineEdit(self.page_debugging)
        self.lineEdit.setClearButtonEnabled(True)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 0, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/light/file_dialog_list.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBox.addItem(self.page_debugging, icon2, "")
        self.verticalLayout.addWidget(self.toolBox)
        self.line_2 = QtWidgets.QFrame(Form)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.buttonBox = QtWidgets.QDialogButtonBox(Form)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Apply|QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.RestoreDefaults)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Form)
        self.toolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Figure Format Configuration"))
        self.chkEnableSpikeOverlay.setText(_translate("Form", "enable_spike_overlay"))
        self.chkDebugPrint.setText(_translate("Form", "debug_print"))
        self.chkEnableSavingToDisk.setText(_translate("Form", "enable_saving_to_disk"))
        self.txtEditExtraArguments.setToolTip(_translate("Form", "Extra Arguments"))
        self.txtEditExtraArguments.setStatusTip(_translate("Form", "Extra Arguments"))
        self.txtEditExtraArguments.setPlaceholderText(_translate("Form", "Optional Arguments"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_figExport), _translate("Form", "Figure Output/Export"))
        self.pushButton_2.setText(_translate("Form", "PushButton"))
        self.lineEdit.setText(_translate("Form", "C:\\Users\\pho\\repos\\PhoPy3DPositionAnalysis2021\\EXTERNAL\\Screenshots\\ProgrammaticDisplayFunctionTesting"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_debugging), _translate("Form", "Extra/Debugging"))
from pyphocorehelpers.gui.Qt.InlineFilesystemPathSelectWidget.FilesystemPathLineEdit import FilesystemPathLineEdit
from pyphocorehelpers.gui.Qt.InlineFilesystemPathSelectWidget.InlineFilesystemPathSelectWidget import InlineFilesystemPathSelectWidget
from pyphoplacecellanalysis.GUI.Qt.Widgets.PhoCodeConsoleWidget import PhoCodeConsoleWidget
from pyphoplacecellanalysis.GUI.Qt.Widgets.SingleGroupOptionalMembersCtrl.SingleGroupOptionalMembersCtrl import SingleGroupOptionalMembersCtrl
from pyphoplacecellanalysis.Resources import breeze

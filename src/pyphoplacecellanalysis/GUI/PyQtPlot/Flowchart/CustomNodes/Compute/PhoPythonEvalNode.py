import os
import re
from typing import final
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore, QtWidgets
translate = QtCore.QCoreApplication.translate

from pyphoplacecellanalysis.External.pyqtgraph.flowchart.library.common import Node
from pyphoplacecellanalysis.External.pyqtgraph.parametertree.parameterTypes.file import popupFilePicker
from pyphoplacecellanalysis.External.pyqtgraph.widgets.FileDialog import FileDialog
from pyphoplacecellanalysis.External.pyqtgraph import configfile

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.MiscNodes.ExtendedCtrlNode import ExtendedCtrlNode


class TextEdit(QtWidgets.QTextEdit):
    def __init__(self, on_update):
        super().__init__()
        self.on_update = on_update
        self.lastText = None

    def focusOutEvent(self, ev):
        text = self.toPlainText()
        if text != self.lastText:
            self.lastText = text
            self.on_update()
        super().focusOutEvent(ev)


class PhoPythonEvalNode(Node):
    """Return the output of a string evaluated/executed by the python interpreter.
    The string may be either an expression or a python script, and inputs are accessed as the name of the terminal. 
    For expressions, a single value may be evaluated for a single output, or a dict for multiple outputs.
    For a script, the text will be executed as the body of a function."""
    nodeName = 'PhoPythonEval'
    
    def __init__(self, name):
        # Setup member variables
        self.filePath = None
        self.currentFileName = None
        self._widget = None
        Node.__init__(self, name, 
            terminals = {
                'input': {'io': 'in', 'renamable': True, 'multiable': True},
                'output': {'io': 'out', 'renamable': True, 'multiable': True},
            },
            allowAddInput=True, allowAddOutput=True)
        
        self.ctrlWidget() # initializes the control widget

    def ctrlWidget(self):
        # return self.ui.root
        if self._widget is None:
            self._widget = PhoPythonEvalNodeCtrlWidget(self)
        return self._widget
        
    def setCode(self, code):
        """ unindent code; this allows nicer inline code specification when calling this method. """
        self.ctrlWidget().setCode(code)

    def code(self):
        """ returns the code of this node from the control. """
        return self.ctrlWidget().code()


    # Adding functions to load/save the current node text:
    def saveAsCustomNode(self):
        """ save the current node's code and inputs/outputs to a file for this node. """
        # Get the node's current state:
        # curr_state = self.saveState()
        # self.save(curr_state)
        try:
            if self.currentFileName is None:
                newFile = self.ctrlWidget().saveFile()
            else:
                newFile = self.ctrlWidget().saveFile(suggestedFileName=self.currentFileName)
            #self.ui.saveAsBtn.success("Saved.")
            #print "Back to saveAsClicked."
        except:
            # self.ui.saveBtn.failure("Error")
            raise
        

    def loadCustomNodeCode(self):
        """ load - get the loaded state from file or whereever """
        # TODO: get the loaded state from file
        # loaded_state = None
        # loaded_state = 
        # self.restoreState(loaded_state)
        newFile = self.ctrlWidget().loadFile()
    
    
    # Node functions:
    # def terminalRenamed(self, term, oldName):
    #     Node.terminalRenamed(self, term, oldName)
    #     item = term.joinItem
    #     item.setText(0, term.name())
    #     self.update()
        
    # def nodeRenamed(self, node, oldName):
    #     del self._nodes[oldName]
    #     self._nodes[node.name()] = node
    #     self.widget().nodeRenamed(node, oldName)
    #     self.sigChartChanged.emit(self, 'rename', node)
        
    # def chartGraphicsItem(self):
    #     """Return the graphicsItem that displays the internal nodes and
    #     connections of this flowchart.
    #     Note that the similar method `graphicsItem()` is inherited from Node
    #     and returns the *external* graphical representation of this flowchart."""
    #     return self.viewBox
    
    # def graphicsItem(self):
    #     """Return the GraphicsItem for this node. Subclasses may re-implement
    #     this method to customize their appearance in the flowchart."""
    #     if self._graphicsItem is None:
    #         self._graphicsItem = NodeGraphicsItem(self)
    #     return self._graphicsItem
    
    
    # TODO: eventually update the code inputs/outputs on sigTerminalRenamed, sigTerminalAdded, and sigTerminalRemoved (all emitted by the parent Node class)

    def rebuild_terminals_from_code(self):
        """ rebuilds the node's input/output terminals from the current code inputs/outputs """
        curr_code = self.code()
        print(f'.rebuild_terminals_from_code():\n{curr_code}')

    def update_code_from_node(self):
        """ updates the header/footer of the code from the the node's input/output terminals """
        # curr_code = self.code()
        curr_terminals = self.terminals
        print(f'.update_code_from_node():\nself.terminals: {curr_terminals}')
        

    def clear(self):
        """Remove all nodes from this flowchart except the original input/output nodes.
        """
        #self.clearTerminals()
        self.ctrlWidget().clear()
    
    def clearTerminals(self):
        Node.clearTerminals(self)
        
        
    # Process 
        
    def process(self, display=True, **args):
        l = locals()
        l.update(args)
        ## try eval first, then exec
        try:  
            text = self.ctrlWidget().ui.text.toPlainText().replace('\n', ' ')
            output = eval(text, globals(), l)
        except SyntaxError:
            fn = "def fn(**args):\n"
            run = "\noutput=fn(**args)\n"
            text = fn + "\n".join(["    "+l for l in self.ctrlWidget().ui.text.toPlainText().split('\n')]) + run
            ldict = locals()
            exec(text, globals(), ldict)
            output = ldict['output']
        except:
            print(f"Error processing node: {self.name()}")
            raise
        return output
        
    def saveState(self):
        state = Node.saveState(self)
        state['text'] = self.ctrlWidget().ui.text.toPlainText() # should be the same as self.ctrlWidget().code()
        #state['terminals'] = self.saveTerminals()
        return state
        
    def restoreState(self, state, clear=False):
        self.blockSignals(True)
        try:
            if clear:
                self.clear()
            Node.restoreState(self, state)
            self.setCode(state['text'])
            self.restoreTerminals(state['terminals'])
            self.update()
        finally:
            self.blockSignals(False)
        

        


# rebuild_terminals_from_code


class PhoPythonEvalNodeCtrlWidget(QtWidgets.QWidget):
    """The widget that contains the list of all the nodes in a flowchart and their controls, as well as buttons for loading/saving flowcharts."""
        
    sigFileLoaded = QtCore.Signal(object)
    sigFileSaved = QtCore.Signal(object)
    
    def __init__(self, eval_node):
        #self.loadDir = loadDir  ## where to look initially for chart files
        self.currentFileName = None
        QtWidgets.QWidget.__init__(self)
        self.eval_node = eval_node # the actual node object
        
        self.ui = PhoUIContainer()
        self._buildUI()
        # self.ui.loadBtn.clicked.connect(self.loadClicked)
        # self.ui.saveBtn.clicked.connect(self.saveClicked)
        # self.ui.saveAsBtn.clicked.connect(self.saveAsClicked)
        # self.ui.showChartBtn.toggled.connect(self.chartToggled)
        # self.chart.sigFileLoaded.connect(self.setCurrentFile)
        # self.ui.reloadBtn.clicked.connect(self.reloadClicked)
        # self.chart.sigFileSaved.connect(self.fileSaved)
        
        self.sigFileLoaded.connect(self.setCurrentFile)
        self.sigFileSaved.connect(self.on_file_saved)
        
        self.ui.load_btn.clicked.connect(self.eval_node.loadCustomNodeCode)
        self.ui.save_btn.clicked.connect(self.eval_node.saveAsCustomNode)
        
        self.ui.reload_from_code_btn.clicked.connect(self.eval_node.rebuild_terminals_from_code)
        self.ui.update_code_from_node_btn.clicked.connect(self.eval_node.update_code_from_node)
        
        
    def _build_inputs_header_text(self):
        
        
        return "'# Access inputs as args['input_name']"
    
    def _build_outputs_return_footer_text(self):
        return "{'output': None} ## one key per output terminal"
        # return f"{'output': None} ## one key per output terminal"

    def _buildUI(self):
        ## Build UI:
        # self.ui.root = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.ui.text = TextEdit(self.update)
        self.ui.text.setTabStopWidth(30)
        default_fcn_body = ''
        default_textfield_body = '\n'.join([self._build_inputs_header_text(), default_fcn_body, self._build_outputs_return_footer_text()])
        # default_textfield_body = "# Access inputs as args['input_name']\nreturn {'output': None} ## one key per output terminal"
        self.ui.text.setPlainText(default_textfield_body)
        self.layout.addWidget(self.ui.text, 1, 0, 1, 2)        
        # Add load/save button widgets:
        self.ui.loadSaveBtnWidget = QtWidgets.QWidget()
        self.ui.loadSaveBtnWidget.setObjectName("loadSaveBtnWidget")
        self.ui.metaBtnLayout = QtWidgets.QHBoxLayout(self.ui.loadSaveBtnWidget)
        self.ui.metaBtnLayout.setContentsMargins(0, 0, 0, 0)
        self.ui.metaBtnLayout.setSpacing(2)
        # self.ui.metaBtnLayout.addStretch(0)
        self.ui.load_btn = QtWidgets.QPushButton()
        self.ui.load_btn.setMinimumSize(QtCore.QSize(24, 24))
        self.ui.load_btn.setText('Load')
        self.ui.load_btn.setObjectName('btnLoad')
        # self.ui.load_btn.clicked.connect(self.loadCustomNodeCode)
        self.ui.metaBtnLayout.addWidget(self.ui.load_btn)
        self.ui.save_btn = QtWidgets.QPushButton()
        self.ui.save_btn.setMinimumSize(QtCore.QSize(24, 24))
        self.ui.save_btn.setText('Save')
        self.ui.save_btn.setObjectName('btnSave')
        # self.ui.save_btn.clicked.connect(self.saveAsCustomNode)
        self.ui.metaBtnLayout.addWidget(self.ui.save_btn)
        
        self.ui.reload_from_code_btn = QtWidgets.QPushButton()
        self.ui.reload_from_code_btn.setMinimumSize(QtCore.QSize(24, 24))
        self.ui.reload_from_code_btn.setText('Reload from Code')
        self.ui.reload_from_code_btn.setObjectName('btnReloadFromCode')
        self.ui.metaBtnLayout.addWidget(self.ui.reload_from_code_btn)
        
        self.ui.update_code_from_node_btn = QtWidgets.QPushButton()
        self.ui.update_code_from_node_btn.setMinimumSize(QtCore.QSize(24, 24))
        self.ui.update_code_from_node_btn.setText('Code from Node')
        self.ui.update_code_from_node_btn.setObjectName('btnUpdateCodeFromNode')
        self.ui.metaBtnLayout.addWidget(self.ui.update_code_from_node_btn)
        
        # Set the button container layout:
        # self.ui.loadSaveBtnWidget.setLayout(self.ui.metaBtnLayout)
        self.layout.addWidget(self.ui.loadSaveBtnWidget, 2, 0, 1, 2)
        # self.ui.root.setLayout(self.layout)
        self.setLayout(self.layout)

        
        # Build custom context menu:
        self.contextMenu = QtWidgets.QMenu()
        self.contextMenu.addAction(translate("PhoPythonEval", 'Custom Copy Selection')).triggered.connect(self.copySel)
        self.contextMenu.addAction(translate("PhoPythonEval", 'Custom Copy All')).triggered.connect(self.copyAll)
        self.contextMenu.addAction(translate("PhoPythonEval", 'Custom Save Selection')).triggered.connect(self.saveSel)
        self.contextMenu.addAction(translate("PhoPythonEval", 'Custom Save All')).triggered.connect(self.saveAll)
        
        # Dialogs:
        self.fileDialog = None
        
    # UI Handling Functions:
    def contextMenuEvent(self, ev):
        self.contextMenu.popup(ev.globalPos())
        
    def serialize(self):
        """Convert entire table (or just selected area) into tab-separated text values"""
        state = self.saveState()
        if 'terminals' not in state:
            state['terminals'] = self.saveTerminals()
        # TODO: convert to JSON or something?
        return state


    def copySel(self):
        """Copy selected data to clipboard."""
        QtWidgets.QApplication.clipboard().setText(self.serialize())

    def copyAll(self):
        """Copy all data to clipboard."""
        QtWidgets.QApplication.clipboard().setText(self.serialize())

    def saveSel(self):
        """Save selected data to file."""
        self.save(self.serialize())

    def saveAll(self):
        """Save all data to file."""
        self.save(self.serialize())
        
        
    # Operations:
    def clear(self):
        """Remove all nodes from this flowchart except the original input/output nodes.
        """
        self.ui.text.clear()
        
    def setCode(self, code):
        # unindent code; this allows nicer inline code specification when 
        # calling this method.
        ind = []
        lines = code.split('\n')
        for line in lines:
            stripped = line.lstrip()
            if len(stripped) > 0:
                ind.append(len(line) - len(stripped))
        if len(ind) > 0:
            ind = min(ind)
            code = '\n'.join([line[ind:] for line in lines])
        
        self.ui.text.clear()
        self.ui.text.insertPlainText(code)

    def code(self):
        """ returns the code of this node from the control. """
        return self.ui.text.toPlainText()
        
     # Extended GUI        
    # def save(self, data):
    #     """ displays the user save prompt and writes out the file """
    #     # getSaveFileName(parent: QWidget = None, caption: str = '', directory: str = '', filter: str = '', initialFilter: str = '', options: Union[QFileDialog.Options, QFileDialog.Option] = 0)
    #     fileName = QtWidgets.QFileDialog.getSaveFileName(self.ui.root,
    #         f"{translate('PhoPythonEval', 'Save As')}...",
    #         'EXTERNAL/CustomNodes',
    #         f"{translate('PhoPythonEval', 'Custom Eval Node')} (*.pEval)"
    #     )
    #     if isinstance(fileName, tuple):
    #         fileName = fileName[0]  # Qt4/5 API difference
    #     if fileName == '':
    #         return
    #     with open(fileName, 'w') as fd:
    #         # write out the data
    #         fd.write(data)
            
            
    def loadFile(self, fileName=None, startDir=None):
        """Load a Custom Eval Node (``*.pEval``) file.
        """
        if fileName is None:
            if startDir is None:
                startDir = self.eval_node.filePath
            if startDir is None:
                startDir = '.'
            self.fileDialog = FileDialog(None, "Load Custom Eval Node..", startDir, "Custom Eval Node (*.pEval)")
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.loadFile)
            return
            ## NOTE: was previously using a real widget for the file dialog's parent, but this caused weird mouse event bugs..
        state = configfile.readConfigFile(fileName)
        self.eval_node.restoreState(state, clear=True)
        self.sigFileLoaded.emit(fileName)

    def saveFile(self, fileName=None, startDir=None, suggestedFileName='custom_node.pEval'):
        """Save this Custom Eval Node to a .pEval file
        """
        if fileName is None:
            if startDir is None:
                startDir = self.eval_node.filePath
            if startDir is None:
                startDir = '.'
            self.fileDialog = FileDialog(None, "Save Custom Eval Node..", startDir, "Custom Eval Node (*.pEval)")
            self.fileDialog.setDefaultSuffix("pEval")
            self.fileDialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave) 
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.saveFile)
            return
        configfile.writeConfigFile(self.eval_node.saveState(), fileName)
        self.sigFileSaved.emit(fileName)

        
    # def _retrieveFileSelection_gui(self):
    #     curVal = self.param.value()
    #     if isinstance(curVal, list) and len(curVal):
    #         # All files should be from the same directory, in principle
    #         # Since no mechanism exists for preselecting multiple, the most sensible
    #         # thing is to select nothing in the preview dialog
    #         curVal = curVal[0]
    #         if os.path.isfile(curVal):
    #             curVal = os.path.dirname(curVal)
    #     opts = self.param.opts.copy()
    #     useDir = curVal or opts.get('directory') or os.getcwd()
    #     startDir = os.path.abspath(useDir)
    #     if os.path.isfile(startDir):
    #         opts['selectFile'] = os.path.basename(startDir)
    #         startDir = os.path.dirname(startDir)
    #     if os.path.exists(startDir):
    #         opts['directory'] = startDir
    #     if opts.get('windowTitle') is None:
    #         opts['windowTitle'] = self.param.title()

    #     fname = popupFilePicker(None, **opts)
    #     if not fname:
    #         return
    #     self.param.setValue(fname)
    
    
    # Events
    @QtCore.pyqtSlot(object)
    def setCurrentFile(self, fileName):
        print(f'.setCurrentFile(fileName: {fileName})')
        self.eval_node.currentFileName = fileName
        # if fileName is None:
        #     self.ui.fileNameLabel.setText("<b>[ new ]</b>")
        # else:
        #     self.ui.fileNameLabel.setText("<b>%s</b>" % os.path.split(self.currentFileName)[1])
        # self.resizeEvent(None)
        
        
    @QtCore.pyqtSlot(object)
    def on_file_saved(self, fileName):
        print(f'.on_file_saved(fileName: {fileName})')
        self.setCurrentFile(fileName)
        # self.ui.saveBtn.success("Saved.")
        
        
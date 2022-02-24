from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.flowchart.library.common import Node

from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.ExtendedCtrlNode import ExtendedCtrlNode


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
        Node.__init__(self, name, 
            terminals = {
                'input': {'io': 'in', 'renamable': True, 'multiable': True},
                'output': {'io': 'out', 'renamable': True, 'multiable': True},
            },
            allowAddInput=True, allowAddOutput=True)
        
        self.ui = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.text = TextEdit(self.update)
        self.text.setTabStopWidth(30)
        self.text.setPlainText("# Access inputs as args['input_name']\nreturn {'output': None} ## one key per output terminal")
        self.layout.addWidget(self.text, 1, 0, 1, 2)
        self.ui.setLayout(self.layout)
        
    def ctrlWidget(self):
        return self.ui
        
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
        
        self.text.clear()
        self.text.insertPlainText(code)

    def code(self):
        return self.text.toPlainText()
        
    def process(self, display=True, **args):
        l = locals()
        l.update(args)
        ## try eval first, then exec
        try:  
            text = self.text.toPlainText().replace('\n', ' ')
            output = eval(text, globals(), l)
        except SyntaxError:
            fn = "def fn(**args):\n"
            run = "\noutput=fn(**args)\n"
            text = fn + "\n".join(["    "+l for l in self.text.toPlainText().split('\n')]) + run
            ldict = locals()
            exec(text, globals(), ldict)
            output = ldict['output']
        except:
            print(f"Error processing node: {self.name()}")
            raise
        return output
        
    def saveState(self):
        state = Node.saveState(self)
        state['text'] = self.text.toPlainText()
        #state['terminals'] = self.saveTerminals()
        return state
        
    def restoreState(self, state):
        Node.restoreState(self, state)
        self.setCode(state['text'])
        self.restoreTerminals(state['terminals'])
        self.update()

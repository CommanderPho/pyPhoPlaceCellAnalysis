from pyqtgraph.flowchart.library.common import generateUi, CtrlNode
# from pyqtgraph.Node import Node
from pyqtgraph.widgets.FeedbackButton import FeedbackButton
from pyqtgraph.widgets.CheckTable import CheckTable

from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.CustomControlWidgets.ExtendedCheckTable import ExtendedCheckTable


class ExtendedCtrlNode(CtrlNode):
	"""docstring for ExtendedCtrlNode."""
 
	@classmethod
	def process_uiTemplate(cls, opts):
		# ctrl_node_ops = dict()
		# custom_ops = dict()
		ctrl_node_ops = list()
		custom_ops = list()
		for opt in opts:
			if len(opt) == 2:
				k, t = opt
				o = {}
			elif len(opt) == 3:
				k, t, o = opt
			else:
				raise Exception("Widget specification must be (name, type) or (name, type, {opts})")
			
			if t in ['intSpin','doubleSpin','spin','check','combo','color']:
				# ctrl_node_ops[t] = opt
				ctrl_node_ops.append(opt)
			else:
				# custom_ops[t] = opt
				custom_ops.append(opt)
		return ctrl_node_ops, custom_ops

	@classmethod
	def generate_extended_Ui(cls, node, unhandled_opts):
		layout = node.ui.layout()
		ctrls = node.ctrls
		num_curr_ctrls = len(ctrls)
		row = num_curr_ctrls - 1 # index of the last row number
		# row = node.ctrls[-1].rowNum # get the row number of the last row
		row = row + 1 # set for the next row
		for opt in unhandled_opts:
			if len(opt) == 2:
				k, t = opt
				o = {}
			elif len(opt) == 3:
				k, t, o = opt
			else:
				raise Exception("Widget specification must be (name, type) or (name, type, {unhandled_opts})")
				
			## clean out these options so they don't get sent to SpinBox
			hidden = o.pop('hidden', False)
			tip = o.pop('tip', None)

			if t == 'extendedchecktable':
				if 'columns' in o:
					col_labels = o['columns']
				else:
					raise
				w = ExtendedCheckTable(col_labels)
			elif t == 'action':
				w = FeedbackButton()
			elif t == 'checktable':
				if 'columns' in o:
					col_labels = o['columns']
				else:
					raise
				w = CheckTable(col_labels)
				# if 'rows' in o:
				#     row_labels = o['rows']
				#     w.updateRows(row_labels)
				# else:
				#     # add a single row with a default label if no labels are provided
				#     row_labels = [f'row[{i}]' for i in np.arange(1)]
				#     w.updateRows(row_labels)
					
			else:
				raise Exception("Unknown widget type '%s'" % str(t))

			if tip is not None:
				w.setToolTip(tip)
			w.setObjectName(k)
			layout.addRow(k, w)
			if hidden:
				w.hide()
				label = layout.labelForField(w)
				label.hide()
				
			ctrls[k] = w
			w.rowNum = row
			row += 1
			

	def __init__(self, name, ui=None, terminals=None):
		custom_ops = None
		if ui is None:
			if hasattr(self, 'uiTemplate'):
				ui = self.uiTemplate
				# Get the CtrlNode safe ui elements:
				ctrl_node_ops, custom_ops = ExtendedCtrlNode.process_uiTemplate(ui)
				
				ui = ctrl_node_ops
				setattr(self, 'uiTemplate', ctrl_node_ops) # set the self.uiTemplate
				self.uiTemplate = ctrl_node_ops
				print(f'ctrl_node_ops: {ctrl_node_ops}\n custom_ops:{custom_ops}\n self.uiTemplate: {self.uiTemplate}\n')
			else:
				ui = []
    
		

				
		CtrlNode.__init__(self, name, ui=ui, terminals=terminals)
		if custom_ops is not None:
			ExtendedCtrlNode.generate_extended_Ui(self, custom_ops)
  
		# self.ui_build()
  
  
		

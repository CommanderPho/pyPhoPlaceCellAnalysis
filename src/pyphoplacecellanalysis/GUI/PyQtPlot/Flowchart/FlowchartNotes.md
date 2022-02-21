
# Custom Nodes

Notes:
"subclasses should call update() whenever thir internal state has changed
        (such as when the user interacts with the Node's control widget). Update
        is automatically called when the inputs to the node are changed."
        

From the Node class, you can override several interesting functions:

### Customizing Node appearance in flowchart: 
See from pyqtgraph.flowchart.Node import NodeGraphicsItem for example
```python
def graphicsItem(self):
	"""Return the GraphicsItem for this node. Subclasses may re-implement
	this method to customize their appearance in the flowchart."""
	if self._graphicsItem is None:
		self._graphicsItem = NodeGraphicsItem(self)
	return self._graphicsItem
```


### Customizing the control interface in the list of nodes (where you set parameters)
```python
def ctrlWidget(self):
	"""Return this Node's control widget. 
	
	By default, Nodes have no control widget. Subclasses may reimplement this 
	method to provide a custom widget. This method is called by Flowcharts
	when they are constructing their Node list."""
	return None
```


## Class Notes:

### Terminal 
The input/output terminal on a Node

multi           bool, for inputs: whether this terminal may make multiple connections
                        for outputs: whether this terminal creates a different value for each connection



boundingRect


### NodeGraphicsItem

setPos




# Adding Controls to Nodes:

# 'combo': Combo Box with Dynamic Keys:
```python
	uiTemplate = [
			('included_configs', 'combo', {'values': [], 'index': 0}),
	]
	# __init__(self, ...):
	self.keys = [] # the active config keys
	# Updating Keys:
	self.updateKeys(updated_configs) # Update the possible keys

	# Getting value:
	selected_config_value = str(self.ctrls['included_configs'].currentText())
	print(f'selected_config_value: {selected_config_value}; updated_configs: {updated_configs}')
        
	s = self.stateGroup.state()
	s['dtype']
```




# Computation Functions:

Each computation function should take a `computation_result: ComputationResult` argument followed by any optional arguments it needs.

From the computation_result, the function has access to:

1. computation_result.sess
2. All previously computed computation results (accessible by knowing their keys)
	`prev_one_step_bayesian_decoder = computation_result.computed_data['pf2D_Decoder']`


Within the function body it adds its specific computed data to one or more (usually one) key in the `computation_result.computed_data` dictionary using a short version of its function name. This dict can have as many items added as desired.
```python
	computation_result.computed_data['pf2D_TwoStepDecoder'] = {'xbin':active_xbins, 'ybin':active_ybins,
		'avg_speed_per_pos': avg_speed_per_pos,
		'K':K, 'V':V,
		'sigma_t_all':sigma_t_all, 'flat_sigma_t_all': np.squeeze(np.reshape(sigma_t_all, (-1, 1)))
	}
```




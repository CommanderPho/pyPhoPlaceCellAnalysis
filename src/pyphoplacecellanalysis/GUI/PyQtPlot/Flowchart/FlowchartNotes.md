
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


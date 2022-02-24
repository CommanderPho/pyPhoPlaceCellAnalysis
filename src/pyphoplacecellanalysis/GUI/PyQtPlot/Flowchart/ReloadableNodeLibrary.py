import importlib
from pyqtgraph.flowchart.NodeLibrary import NodeLibrary


class ReloadableNodeLibrary(NodeLibrary):
	""" A version of NodeLibrary that enables reloading node classes dynamically (to enable updating them after running). """

	def __init__(self):
		NodeLibrary.__init__(self)

	def copy(self):
		"""
		Return a copy of this library.
		"""
		lib = ReloadableNodeLibrary()
		lib.nodeList = self.nodeList.copy()
		lib.nodeTree = self.treeCopy(self.nodeTree)
		return lib

	@classmethod
	def from_node_library(cls, nodelibrary):
		""" Copies a standard NodeLibrary's items to build an upgraded ReloadableNodeLibrary() """
		lib = ReloadableNodeLibrary()
		lib.nodeList = nodelibrary.nodeList.copy()
		lib.nodeTree = nodelibrary.treeCopy(nodelibrary.nodeTree)
		return lib

	def reload(self):
		"""
		Reload Node classes in this library.
		"""
		# raise NotImplementedError()
		# Get all known nodes:
		print(f'reload(): ReloadableNodeLibrary contains nodes: {self.nodeList.keys()}')
		for aName, aClass in self.nodeList.items():
			print(f'reloading node module {aName} with {aClass}')
			importlib.reload(aClass)
		


import importlib
from pyqtgraph.flowchart.NodeLibrary import NodeLibrary


class ReloadableNodeLibrary(NodeLibrary):
    """ A version of NodeLibrary that enables reloading node classes dynamically (to enable updating them after running). """

    @property
    def register_custom_nodes_function(self):
        """The register_custom_nodes_function property."""
        return self._register_custom_nodes_function
    @register_custom_nodes_function.setter
    def register_custom_nodes_function(self, value):
        self._register_custom_nodes_function = value
    
    
    def __init__(self):
        NodeLibrary.__init__(self)
        self._register_custom_nodes_function = None


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


    def addNodeType(self, nodeClass, paths, override=True):
        """
        Register a new node type. If the type's name is already in use,
        an exception will be raised (unless override=True).
        
        ============== =========================================================
        **Arguments:**
        
        nodeClass      a subclass of Node (must have typ.nodeName)
        paths          list of tuples specifying the location(s) this 
                       type will appear in the library tree.
        override       if True, overwrite any class having the same name
        ============== =========================================================
        """
        return NodeLibrary.addNodeType(self, nodeClass=nodeClass, paths=paths, override=override)


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
            # self.addNodeType(aClass, aClass. 
            
        if self.register_custom_nodes_function is not None:
            self.register_custom_nodes_function(self) # call to re-register the custom nodes. Not sure if this will work.
        


import importlib
from pyphoplacecellanalysis.External.pyqtgraph.flowchart.NodeLibrary import NodeLibrary
import pyphoplacecellanalysis.External.pyqtgraph.flowchart.library as fclib

# Import the custom nodes:
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.DisplayNodes.ImageViewNode import ImageViewNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.DisplayNodes.UnsharpMaskNode import UnsharpMaskNode

from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.BasePipeline.PipelineInputDataNode import PipelineInputDataNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.BasePipeline.PipelineFilteringDataNode import PipelineFilteringDataNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.BasePipeline.PipelineComputationsNode import PipelineComputationsNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.DisplayNodes.PipelineDisplayNode import PipelineDisplayNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.Compute.PhoPythonEvalNode import PhoPythonEvalNode
from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.CustomNodes.DisplayNodes.PipelineResultVisNode import PipelineResultVisNode


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
        

    @classmethod
    def _register_only_custom_node_types(cls, library):

        # Custom Nodes:
        library.addNodeType(PhoPythonEvalNode, [('Data',), 
                                            ('Pho Pipeline','Eval')])
            
        # Pipeline Nodes:
        library.addNodeType(PipelineInputDataNode, [('Data',), 
                                            ('Pho Pipeline','Input')])
        library.addNodeType(PipelineFilteringDataNode, [('Filters',), 
                                            ('Pho Pipeline','Filtering')])
        library.addNodeType(PipelineComputationsNode, [('Data',), 
                                            ('Pho Pipeline','Computation')])
        library.addNodeType(PipelineDisplayNode, [('Display',), 
                                            ('Pho Pipeline','Display')])    
        library.addNodeType(PipelineResultVisNode, [('Display',), 
                                            ('Pho Pipeline','Display')])
        return library

    @classmethod
    def setup_custom_node_library(cls, fc):
        """Register Custom Nodes so they appear in the flowchart context menu
        
        fc: an actual Flowchart object
        """
        ## Method 1: Register to global default library:
        #fclib.registerNodeType(ImageViewNode, [('Display',)])
        #fclib.registerNodeType(UnsharpMaskNode, [('Image',)])

        ## Method 2: If we want to make our custom node available only to this flowchart,
        ## then instead of registering the node type globally, we can create a new 
        ## NodeLibrary:
        # library = fclib.LIBRARY.copy() # start with the default node set
        library = ReloadableNodeLibrary.from_node_library(fclib.LIBRARY.copy())  # start with the default node set
        
        library.addNodeType(ImageViewNode, [('Display',)])
        # Add the unsharp mask node to two locations in the menu to demonstrate
        # that we can create arbitrary menu structures
        library.addNodeType(UnsharpMaskNode, [('Image',)])
        
        library = cls._register_only_custom_node_types(library=library)
        library.register_custom_nodes_function = cls._register_only_custom_node_types # set the reload custom nodes function to the function used to register the custom nodes

        fc.setLibrary(library)
        
        
    
import pyphoplacecellanalysis.External.pyqtgraph.parametertree.parameterTypes as pTypes
from pyphoplacecellanalysis.External.pyqtgraph.parametertree import Parameter, ParameterTree
from pyphoplacecellanalysis.GUI.PyQtPlot.Params.SaveRestoreStateParamHelpers import default_parameters_save_restore_state_button_children, add_save_restore_btn_functionality
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import get_default_pipeline_data_keys, _test_save_pipeline_data_to_h5, get_h5_data_keys, save_some_pipeline_data_to_h5, load_pipeline_data_from_h5  #ExportHelpers

""" ExportPipelineParametersTree
Usage:
    from pyphoplacecellanalysis.GUI.PyQtPlot.Params.ParameterTrees.ExportPipelineParametersTree import _build_export_parameters_tree
    p = build_export_parameters_tree(curr_active_pipeline, parameter_names='ExportParams', finalized_output_cache_file='data/pipeline_cache_store.h5', include_state_save_restore_buttons=False, debug_print=True)

"""

## this group includes a menu allowing the user to add new parameters into its child list
class ExportHdf5KeysGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add Key"
        opts['addList'] = ['str', 'float', 'int'] # dropdown list of items to add (shows up in the combo box)
        pTypes.GroupParameter.__init__(self, **opts)
    
    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(dict(name="/filtered_sessions/YOUR_SESSION_NAME/YOUR_KEY", type=typ, value=val, removable=True, renamable=True))



def build_export_parameters_tree(curr_active_pipeline, parameter_names='ExportParams', finalized_output_cache_file='data/pipeline_cache_store.h5', include_state_save_restore_buttons=True, debug_print=False):
    """ Builds a ParameterTree widget to allow specification of the export parameters.
        
        curr_active_pipeline: the pipeline object captured to actually perform the export
    """
    def _build_current_export_keys(all_computed_config_names, include_whitelist = None):
        """ builds the list of default export keys for the keys list given the currently selected configs """
        # List existing keys in the file:
        # loaded_extant_keys = get_h5_data_keys(finalized_output_cache_file=finalized_output_cache_file)
        # if debug_print:
        #     print(f'loaded_extant_keys: {loaded_extant_keys}')
        
        # Filter existing keys to selections only:
        key_children_list = []
        if include_whitelist is None:
            active_config_names_list = all_computed_config_names
        else:
            # otherwise only include the items in include_whitelist
            active_config_names_list = [value for value in include_whitelist if value in all_computed_config_names] # entry must be in all_computed_config_names
            
        for a_key in active_config_names_list:
            curr_default_key_children_dict = get_default_pipeline_data_keys(a_key)
            curr_children_list = [{'name': value_key_name, 'type': 'str', 'value': value_name} for value_name, value_key_name in curr_default_key_children_dict.items()]
            key_children_list = key_children_list + curr_children_list
            
        if debug_print:
            print(f'key_children_list: {key_children_list}')
            
        return key_children_list
        
    def _simple_export_dict_params(all_computed_config_names):
        key_children_list = _build_current_export_keys(all_computed_config_names, include_whitelist=None)
        children = [
                dict(name='Export Path', type='file', dialogLabel='test label', value=finalized_output_cache_file, default='<Select Path>'),
                dict(name='Included Exports', type='checklist', value=all_computed_config_names, limits=all_computed_config_names),
                ExportHdf5KeysGroup(name="Export Keys", tip='Click to add children', children=key_children_list),
                dict(name='Export', type='action', tip='Perform the export', value='Export'),
            ]

        # Use save/restore state buttons
        if include_state_save_restore_buttons:
            children.append(default_parameters_save_restore_state_button_children())

        ## Create tree of Parameter objects
        p = Parameter.create(name=parameter_names, type='group', children=children)
        return p
    
    
    all_computed_config_names = curr_active_pipeline.active_completed_computation_result_names
    p = _simple_export_dict_params(all_computed_config_names)
    
    ## If anything changes in the tree, print a message
    def on_tree_value_change(param, changes):
        """ 
        Implicitly captures:
            p
        """
        if debug_print:
            print("tree changes:")
        for param, change, data in changes:
            path = p.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            if debug_print:
                print('  parameter: %s'% childName)
                print('  change:    %s'% change)
                print('  data:      %s'% str(data))
                print('  ----------')

            # Handle unchecking events: update the children keys when the selection changes. TODO: note that any custom-added keys will be currently overwritten
            uncheck_event_type = ('Included Exports', 'value') # parameter: 'Included Exports', change: 'value', data: []
            if (childName == uncheck_event_type[0]) and (change == uncheck_event_type[1]):
                if debug_print:
                    print(f'matched uncheck event. data: {data}')
                # data: the list of currently checked changes:
                updated_key_children_list = _build_current_export_keys(include_whitelist=data)
                if debug_print:
                    print(f'\tupdated_key_children_list: {updated_key_children_list}')    
                # replace the children keys:
                export_keys_list = p.param("Export Keys")
                export_keys_list.clearChildren()
                export_keys_list.addChildren(children=updated_key_children_list)
                
    def valueChanging(param, value):
        # called whenever a child value is changed:
        print("Value changing (not finalized): %s %s" % (param, value))
        
    # Connect the sigTreeStateChanged signal:
    p.sigTreeStateChanged.connect(on_tree_value_change)

    ## Connect the children's sigValueChanging signals:
    for child in p.children():
        # Only listen for changes of the 'widget' child:
        if 'widget' in child.names:
            child.child('widget').sigValueChanging.connect(valueChanging)

    if include_state_save_restore_buttons:
        # Setup the save/restore button functionality for p
        add_save_restore_btn_functionality(p)


    # Setup the export button action:
    def action_perform_export():
        """ Do the actual export task here:
        
            Implicitly captures: curr_active_pipeline, 
        """
        print(f'action_perform_export()')
        # direct from widget (WORKS):
        curr_export_path_str = p['Export Path']
        # from saved state:
        # curr_export_path_str = state['children']['Export Path']['value']

        if debug_print:
            print(f'\tcurr_export_path_str: {curr_export_path_str}')

        ## TODO: validate the path and make sure it's valid:
            
        finalized_output_cache_file = curr_export_path_str
        output_save_result = save_some_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file=finalized_output_cache_file)
        if debug_print:
            print(f'\toutput_save_result: {output_save_result}')
        

    btnExport = p.param('Export')
    btnExport.sigActivated.connect(action_perform_export)
    
    return p


from pyphocorehelpers.mixins.auto_registering import RegistryHolder
from typing import Dict

class ComputationFunctionRegistryHolder(RegistryHolder):
    REGISTRY: Dict[str, "ComputationFunctionRegistryHolder"] = {}
    
    @classmethod
    def get_registry(cls):
        """ ensures that registry items are returned sorted by their ._computationPrecidence """
        return dict(sorted(dict(cls.REGISTRY).items(), key=lambda item: item[1]._computationPrecidence))
    
    @classmethod
    def get_registry_items_functions_dict(cls, absolute_flat_path_keys=False, applying_disable_dict=None):
        """Returns a nested hierarchy of registry items and their children

        Args:
            absolute_flat_path_keys (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        out_dict = {}
        for (a_computation_class_name, a_computation_class) in reversed(cls.get_registry().items()):
            if not absolute_flat_path_keys:
                curr_class_relative_out_dict = {}
            
            for (a_computation_fn_name, a_computation_fn) in reversed(a_computation_class.get_all_functions(use_definition_order=True)):
                if not absolute_flat_path_keys:
                    # Relative dict mode:
                    curr_class_relative_out_dict[a_computation_fn_name] = a_computation_fn

                else:
                    curr_absolute_path_list = [a_computation_class_name, a_computation_fn_name]
                    curr_absolute_path_key = '.'.join(curr_absolute_path_list)
                    out_dict[curr_absolute_path_key] = a_computation_fn # absolute path
                    
            if not absolute_flat_path_keys:
                out_dict[a_computation_class_name] = curr_class_relative_out_dict
            
        if applying_disable_dict is not None:
            # Must apply disable dict:
            out_dict = cls.applying_disable_dict(out_dict, applying_disable_dict)
            
        return out_dict
    
    
    
    @classmethod
    def _debug_print_relative_computations_functions_list(cls, rel_compuitations_functions_list):
        """ Prints the hierarchy produced by _temp_compuitations_functions_list = ComputationFunctionRegistryHolder.get_registry_items_functions_dict(absolute_flat_path_keys=False)
        
        Usage:
            _temp_compuitations_functions_list = ComputationFunctionRegistryHolder.get_registry_items_functions_dict()
            _debug_print_relative_computations_functions_list(_temp_compuitations_functions_list)
            
            >>
            SpikeAnalysisComputations (1 functions):
                _perform_spike_burst_detection_computation
            ExtendedStatsComputations (3 functions):
                _perform_placefield_overlap_computation
                _perform_firing_rate_trends_computation
                _perform_extended_statistics_computation
            DefaultComputationFunctions (3 functions):
                _perform_velocity_vs_pf_density_computation
                _perform_two_step_position_decoding_computation
                _perform_position_decoding_computation
            PlacefieldComputations (2 functions):
                _perform_time_dependent_placefield_computation
                _perform_baseline_placefield_computation

        """
        for (a_computation_class_name, a_computation_class_functions_dict) in rel_compuitations_functions_list.items():
            print(f'{a_computation_class_name} ({len(a_computation_class_functions_dict)} functions):')
            for (a_computation_fn_name, a_computation_fn) in a_computation_class_functions_dict.items():
                print(f'\t {a_computation_fn_name}')
            
    @classmethod
    def applying_disable_dict(cls, rel_compuitations_functions_list, disable_dict):
        """ returns a copy of the registery items after applying the disable_dict """
        backup_temp_compuitations_functions_list = rel_compuitations_functions_list.copy()
        backup_temp_compuitations_functions_list = cls._apply_disable_dict(backup_temp_compuitations_functions_list, disable_dict)
        return backup_temp_compuitations_functions_list
            
    @classmethod
    def _apply_disable_dict(cls, rel_compuitations_functions_list, disable_dict):
        """ Applies a dictionary containing keys specifiying computation functions to drop from the available computations list to return only the desired functions
        
        _disable_dict: dict
            {'SpikeAnalysisComputations': False,
            'ExtendedStatsComputations': {'_perform_placefield_overlap_computation': False}}
        
        Usage:
            _disable_dict = {} # empty dict by default
            _disable_dict['SpikeAnalysisComputations'] = False # disables all members of SpikeAnalysisComputations
            _disable_dict['ExtendedStatsComputations'] = {'_perform_placefield_overlap_computation': False} # Disables only the '_perform_placefield_overlap_computation' function of 'ExtendedStatsComputations'
            
            backup_temp_compuitations_functions_list = _temp_compuitations_functions_list.copy()
            backup_temp_compuitations_functions_list = _apply_disable_dict(backup_temp_compuitations_functions_list, _disable_dict)
            backup_temp_compuitations_functions_list
        
        """
        for a_key, a_value in disable_dict.items():
            if isinstance(a_value, dict):
                # keep recurrsing:
                curr_dict_item = rel_compuitations_functions_list[a_key]
                return cls._apply_disable_dict(curr_dict_item, a_value) # a_value is a sub-disable dict, which should be applied to the subset of the rel_computations_functions_list
                
            elif isinstance(a_value, bool):
                # otherwise it's concrete!
                if not a_value:
                    # if false, remove the item
                    del rel_compuitations_functions_list[a_key] # remove the item
                    return rel_compuitations_functions_list
            else:
                # otherwise it's unknown type
                raise


    # @classmethod
    # def get_registry_items_info(cls):
    #     """ ensures that registry items are returned sorted by their ._computationPrecidence """
    #     return dict(sorted(dict(cls.REGISTRY).items(), key=lambda item: item[1]._computationPrecidence))
    
    
# class BaseRegisteredDisplayClass(metaclass=ComputationFunctionRegistryHolder):
#     pass

# class TestDisplayClass(BaseRegisteredDisplayClass):
#     pass

# ComputationFunctionRegistryHolder.get_registry()

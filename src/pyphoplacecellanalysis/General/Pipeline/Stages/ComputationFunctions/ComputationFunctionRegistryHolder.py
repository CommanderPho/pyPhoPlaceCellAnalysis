from pyphocorehelpers.mixins.auto_registering import RegistryHolder
from typing import Dict, Callable, List, Tuple

"""
Computation order is determined by `computation_precidence`.
    Lower values => earlier computations

    If a function `fB` depends on the result added to the pipeline by a function `fA`, then `fA.computation_precidence < fB.computation_precidence`

Each computation functions class has its own `._computationPrecidence`.
All of its functions get their `.computation_precidence` based on the order that they are defined in the class, with the earliest-running functions appearing at the top of the class.
    The numbers are assigned as:  `a_class.fn[i].computation_precidence = float(a_class._computationPrecidence) + (0.01 * float(i))
    Each individual function can override their default directly by using the `@computation_precidence_specifying_function(1.92)` decorator or including the `@function_attributes(..., computation_precidence=1.92)`. 
    This override allows functions to be ran before others even if the others are in a separate class file.


```python
class DirectionalPlacefieldGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    _computationGroupName = 'directional_pfs'
    _computationPrecidence = 1000
    _is_global = True

    ...
```

```python
...
    @computation_precidence_specifying_function(overriden_computation_precidence=-0.1)
    @function_attributes(short_name='lap_direction_determination', tags=['laps'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-24 13:04', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].sess.laps.to_dataframe(), curr_active_pipeline.computation_results[computation_filter_name].sess.laps.to_dataframe()['is_LR_dir']), is_global=False)
    def _perform_lap_direction_determination(computation_result: ComputationResult, **kwargs):
            ...
```
-- OR --
```python
...
    @function_attributes(short_name='lap_direction_determination', tags=['laps'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-24 13:04', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].sess.laps.to_dataframe(), curr_active_pipeline.computation_results[computation_filter_name].sess.laps.to_dataframe()['is_LR_dir']), is_global=False,
        computation_precidence=(-0.1))
    def _perform_lap_direction_determination(computation_result: ComputationResult, **kwargs):
            ...
```


"""

def global_function(is_global:bool=True):
    """Adds function attributes to a function that marks it as global

    ```python
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import global_function

        @global_function()
        def _perform_time_dependent_pf_sequential_surprise_computation(computation_result, debug_print=False):
            # function body
    ```

    func.is_global
    """
    def decorator(func):
        func.is_global = is_global
        return func
    return decorator

def computation_precidence_specifying_function(overriden_computation_precidence: float):
    """Adds function attributes to a function that specify its computation_precidence

    ```python
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import computation_precidence_specifying_function

        @computation_precidence_specifying_function(1.2)
        def _perform_time_dependent_pf_sequential_surprise_computation(computation_result, debug_print=False):
            # function body
    ```

    func.is_global
    """
    def decorator(func):
        func.computation_precidence = overriden_computation_precidence
        return func
    return decorator



class ComputationFunctionRegistryHolder(RegistryHolder):
    REGISTRY: Dict[str, "ComputationFunctionRegistryHolder"] = {}
    
    @classmethod
    def get_registry(cls) -> Dict[str, Callable]:
        """ ensures that registry items are returned sorted by their ._computationPrecidence """
        return dict(sorted(dict(cls.REGISTRY).items(), key=lambda item: item[1]._computationPrecidence))

    @classmethod
    def get_non_global_registry_items(cls, **kwargs) -> Dict[str, Callable]:
        """ ensures that registry items are returned sorted by their ._computationPrecidence """
        # return {k:v for k,v in cls.get_registry().items() if not v._is_global}
        return cls.get_ordered_registry_items_functions(include_local=True, include_global=False, **kwargs)
    

    @classmethod
    def get_global_registry_items(cls, **kwargs) -> Dict[str, Callable]:
        """ ensures that registry items are returned sorted by their ._computationPrecidence
         # Global Items:
            _out_global_only = ComputationFunctionRegistryHolder.get_ordered_registry_items_functions_list(include_local=False, include_global=True)
            _out_global_only

        """
        # return {k:v for k,v in cls.get_registry().items() if v._is_global}
        return cls.get_ordered_registry_items_functions(include_local=False, include_global=True, **kwargs)

    @classmethod
    def get_all_computation_fn_names(cls, non_global_all_exclude_list=['EloyAnalysis', '_DEP_ratemap_peaks', '_perform_specific_epochs_decoding', 'extended_stats', 'placefield_overlap', 'recursive_latent_pf_decoding', 'velocity_vs_pf_simplified_count_density'],
                                        global_all_exclude_list=['PBE_stats']) -> Tuple[List[str], List[str]]:
        """ Gets the hardcoded or dynamically loaded computation names
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
        non_global_comp_names, global_comp_names = ComputationFunctionRegistryHolder.get_all_batch_computation_names()
        
        
        """
        ## functions to exclude from the outputs:
        either_exclude_list = non_global_all_exclude_list + global_all_exclude_list

        global_comp_names = [getattr(v, 'short_name', v.__name__) for k,v in cls.get_global_registry_items().items() if ((getattr(v, 'short_name', v.__name__) not in either_exclude_list) and (v.__name__ not in either_exclude_list))]
        non_global_comp_names = [getattr(v, 'short_name', v.__name__) for k,v in cls.get_non_global_registry_items().items() if ((getattr(v, 'short_name', v.__name__) not in either_exclude_list) and (v.__name__ not in either_exclude_list))]
        return non_global_comp_names, global_comp_names


    @classmethod
    def get_ordered_registry_items_functions(cls, include_local:bool = True, include_global: bool = True, applying_disable_dict=None) -> Dict[str, Callable]:
        """Returns a computationPrecidence ordered list of functions

        Args:
            applying_disable_dict (dict, optional): a dictionary to exclude returned items 

        Returns:
            _type_: _description_
        """
        out_dict = {}
        # out_computation_list = []
        
        for (a_computation_class_name, a_computation_class) in reversed(cls.get_registry().items()):
            # if not absolute_flat_path_keys:
            #     curr_class_relative_out_dict = {}
                
            # curr_all_computation_functions = list(reversed(a_computation_class.get_all_functions(use_definition_order=True))) # reversed definition order means that functions at the bottom of the class are given the lowest precidence, while those at the top are given the highest (and will be computed first)
            curr_all_computation_functions = list(a_computation_class.get_all_functions(use_definition_order=True)) # 2024-05-02 - Correct order

            class_num_computations: int = len(curr_all_computation_functions)
            # not going to have more than 100 computation functions ever
            assert class_num_computations < 100, f"100 is the max number of functions a class can have right now. Change the default_computation_precidence_step_size: float = 0.001 to have 1000."
            default_computation_precidence_step_size: float = 0.01

            curr_class_is_global: bool = a_computation_class._is_global

            for i, (a_computation_fn_name, a_computation_fn) in enumerate(curr_all_computation_functions):
                # determine its actual computation precidence:
                curr_fn_computation_precidence_score: float = getattr(a_computation_fn, 'computation_precidence', None)
                if curr_fn_computation_precidence_score is None:
                    # set the default score:
                    curr_fn_computation_precidence_score = float(a_computation_class._computationPrecidence) + (default_computation_precidence_step_size * float(i))
                    a_computation_fn.computation_precidence = curr_fn_computation_precidence_score

                curr_fn_is_global: float = getattr(a_computation_fn, 'is_global', None)
                if curr_fn_is_global is None:
                    # set whether it's global to the class default:
                    a_computation_fn.is_global = curr_class_is_global

                curr_fn_short_name: str = getattr(a_computation_fn, 'short_name', None)
                if curr_fn_short_name is None:
                    # set whether it's global to the class default:
                    a_computation_fn.short_name = a_computation_fn.__name__ # use the full name

                ## build the path
                curr_absolute_path_list = [a_computation_class_name, a_computation_fn_name]
                curr_absolute_path_key = '.'.join(curr_absolute_path_list)
                out_dict[curr_absolute_path_key] = a_computation_fn # absolute path
                # out_computation_list.append(out_computation_list)
                

        ## now they all have computation precidence, sort them accordingly:
        # out_dict = dict(sorted(out_dict.items(), key=lambda item: float(item[1].computation_precidence)))
        out_dict = dict(sorted(out_dict.items(), key=lambda item: float(item[1].computation_precidence), reverse=False)) # I think we need reverse=True so that the highest precidence items are first in the list and they descend in order.
        
        if applying_disable_dict is not None:
            # Must apply disable dict:
            out_dict = cls.applying_disable_dict(out_dict, applying_disable_dict)

        if (not include_local):
            ## exclude non-global (local) items:
            out_dict = {k: v for k, v in out_dict.items() if v.is_global}

        if (not include_global):
            ## exclude global items:
            out_dict = {k: v for k, v in out_dict.items() if (not v.is_global)}

        return out_dict
    

    # Disable Dicts ______________________________________________________________________________________________________ #
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


    # Old/Obsolite _______________________________________________________________________________________________________ #
                
    # @classmethod
    # def get_registry_items_functions_dict(cls, absolute_flat_path_keys=False, applying_disable_dict=None):
    #     """Returns a nested hierarchy of registry items and their children

    #     Args:
    #         absolute_flat_path_keys (bool, optional): _description_. Defaults to False.
    #         applying_disable_dict (dict, optional): a dictionary to exclude returned items 
    #     Returns:
    #         _type_: _description_
    #     """
    #     out_dict = {}
    #     for (a_computation_class_name, a_computation_class) in reversed(cls.get_registry().items()):
    #         if not absolute_flat_path_keys:
    #             curr_class_relative_out_dict = {}
            
    #         for (a_computation_fn_name, a_computation_fn) in reversed(a_computation_class.get_all_functions(use_definition_order=True)):
    #             if not absolute_flat_path_keys:
    #                 # Relative dict mode:
    #                 curr_class_relative_out_dict[a_computation_fn_name] = a_computation_fn

    #             else:
    #                 curr_absolute_path_list = [a_computation_class_name, a_computation_fn_name]
    #                 curr_absolute_path_key = '.'.join(curr_absolute_path_list)
    #                 out_dict[curr_absolute_path_key] = a_computation_fn # absolute path
                    
    #         if not absolute_flat_path_keys:
    #             out_dict[a_computation_class_name] = curr_class_relative_out_dict
            
    #     if applying_disable_dict is not None:
    #         # Must apply disable dict:
    #         out_dict = cls.applying_disable_dict(out_dict, applying_disable_dict)
            
    #     return out_dict
    
    # @classmethod
    # def _debug_print_relative_computations_functions_list(cls, rel_compuitations_functions_list):
    #     """ Prints the hierarchy produced by _temp_compuitations_functions_list = ComputationFunctionRegistryHolder.get_registry_items_functions_dict(absolute_flat_path_keys=False)
        
    #     Usage:
    #         _temp_compuitations_functions_list = ComputationFunctionRegistryHolder.get_registry_items_functions_dict()
    #         _debug_print_relative_computations_functions_list(_temp_compuitations_functions_list)
            
    #         >>
    #         SpikeAnalysisComputations (1 functions):
    #             _perform_spike_burst_detection_computation
    #         ExtendedStatsComputations (3 functions):
    #             _perform_placefield_overlap_computation
    #             _perform_firing_rate_trends_computation
    #             _perform_extended_statistics_computation
    #         DefaultComputationFunctions (3 functions):
    #             _perform_velocity_vs_pf_density_computation
    #             _perform_two_step_position_decoding_computation
    #             _perform_position_decoding_computation
    #         PlacefieldComputations (2 functions):
    #             _perform_time_dependent_placefield_computation
    #             _perform_baseline_placefield_computation

    #     """
    #     for (a_computation_class_name, a_computation_class_functions_dict) in rel_compuitations_functions_list.items():
    #         print(f'{a_computation_class_name} ({len(a_computation_class_functions_dict)} functions):')
    #         for (a_computation_fn_name, a_computation_fn) in a_computation_class_functions_dict.items():
    #             print(f'\t {a_computation_fn_name}')
            


    # @classmethod
    # def get_registry_items_info(cls):
    #     """ ensures that registry items are returned sorted by their ._computationPrecidence """
    #     return dict(sorted(dict(cls.REGISTRY).items(), key=lambda item: item[1]._computationPrecidence))
    
    
# class BaseRegisteredDisplayClass(metaclass=ComputationFunctionRegistryHolder):
#     pass

# class TestDisplayClass(BaseRegisteredDisplayClass):
#     pass

# ComputationFunctionRegistryHolder.get_registry()

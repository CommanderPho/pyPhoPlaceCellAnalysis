import numpy as np
from attrs import define, Factory, field, fields
from typing import Callable, List, Dict, Optional
import networkx as nx

@define(slots=False, repr=True)
class SpecificComputationResultsSpecification:
    """ This encapsulates the specification for required/provided global results 

    Usage:
    	from pyphoplacecellanalysis.General.Model.SpecificComputationValidation import SpecificComputationResultsSpecification
    
    Can specify like:
    
    	@function_attributes(short_name='merged_directional_placefields', tags=['directional_pf', 'laps', 'epoch', 'session', 'pf1D', 'pf2D'], input_requires=[], output_provides=[], uses=['PfND.build_merged_directional_placefields('], used_by=[], creation_date='2023-10-25 09:33', related_items=[],
		    results_specification = SpecificComputationResultsSpecification(provides_global_keys=['DirectionalMergedDecoders']),
		    provides_global_keys = ['DirectionalMergedDecoders'],
		validate_computation_test=DirectionalPseudo2DDecodersResult.validate_has_directional_merged_placefields, is_global=True)
        def a_fn(...
            ...
            
        results_specification=SpecificComputationResultsSpecification(provides_global_keys=['DirectionalMergedDecoders']),
		provides_global_keys=['DirectionalMergedDecoders'],


    """
    provides_global_keys: List[str] = field(default=Factory(list), repr=True) 
    # provides_local_keys: List[str] = field(default=Factory(list), repr=True)
    requires_global_keys: List[str] = field(default=Factory(list), repr=True)
    # requires_local_keys: List[str] = field(default=Factory(list), repr=True)
    
    def has_provided_keys(self, global_computation_results):
        """ check if the provided_global_keys are in the global_computation_results already.

        # Check for existing result:
        are_all_included, (matching_keys_dict, missing_keys_dict) = comp_specifier.results_specification.has_provided_keys(curr_active_pipeline.global_computation_results)
        if are_all_included:
            print(f'all provided {len(matching_keys_dict)} keys are already present and valid: {list(matching_keys_dict.keys())}')
        else:
            print(f'{len(matching_keys_dict)}/{len(matching_keys_dict) + len(missing_keys_dict)} provided keys are already present and valid:\n\tmatching: {list(matching_keys_dict.keys())}\n\tmissing: {list(missing_keys_dict.keys())}')
            # missing_keys_dict


        """
        matching_keys_dict = {}
        missing_keys_dict = {}
        for a_key in self.provides_global_keys:
            prev_result = global_computation_results.computed_data.get(a_key, None)
            if prev_result is not None:
                matching_keys_dict[a_key] = True
            else:
                missing_keys_dict[a_key] = False

        are_all_included: bool = (len(matching_keys_dict) == len(self.provides_global_keys))
        return are_all_included, (matching_keys_dict, missing_keys_dict)

    def remove_provided_keys(self, global_computation_results) -> Dict:
        """ try to find and remove any existing results that match provides_global_keys """
        removed_keys_dict = {}
        for a_key in self.provides_global_keys:
            prev_result = global_computation_results.computed_data.pop(a_key, None)
            if prev_result is not None:
                removed_keys_dict[a_key] = prev_result
        return removed_keys_dict     
            


@define(slots=False, repr=True)
class SpecificComputationValidator:
    """ This encapsulates the logic for testing if a computation already complete or needs to be completed, and calling the compute function if needed.

    Usage:
    	from pyphoplacecellanalysis.General.Model.SpecificComputationValidation import SpecificComputationValidator
    
        ## Specify the computations and the requirements to validate them.
        _comp_specifiers = [
            SpecificComputationValidator(short_name='firing_rate_trends', computation_fn_name='_perform_firing_rate_trends_computation', validate_computation_test=lambda curr_active_pipeline: (curr_active_pipeline.computation_results[global_epoch_name].computed_data['firing_rate_trends'], curr_active_pipeline.computation_results[global_epoch_name].computed_data['extended_stats']['time_binned_position_df']), is_global=False),
            SpecificComputationValidator(short_name='pf_dt_sequential_surprise', computation_fn_name='_perform_time_dependent_pf_sequential_surprise_computation', validate_computation_test=lambda curr_active_pipeline: (np.sum(curr_active_pipeline.global_computation_results.computed_data['pf_dt_sequential_surprise']['flat_relative_entropy_results'], axis=1), np.sum(curr_active_pipeline.global_computation_results.computed_data['pf_dt_sequential_surprise']['flat_jensen_shannon_distance_results'], axis=1)), is_global=False),  # flat_surprise_across_all_positions
            SpecificComputationValidator(short_name='jonathan_firing_rate_analysis', computation_fn_name='_perform_jonathan_replay_firing_rate_analyses', validate_computation_test=lambda curr_active_pipeline: curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis'].neuron_replay_stats_df, is_global=True),  # active_context
            SpecificComputationValidator(short_name='short_long_pf_overlap_analyses', computation_fn_name='_perform_long_short_pf_overlap_analyses', validate_computation_test=lambda curr_active_pipeline: (curr_active_pipeline.global_computation_results.computed_data['short_long_pf_overlap_analyses']['relative_entropy_overlap_scalars_df'], curr_active_pipeline.global_computation_results.computed_data['short_long_pf_overlap_analyses']['relative_entropy_overlap_dict']), is_global=True),  # relative_entropy_overlap_scalars_df
            SpecificComputationValidator(short_name='long_short_fr_indicies_analyses', computation_fn_name='_perform_long_short_firing_rate_analyses', validate_computation_test=lambda curr_active_pipeline: curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']['x_frs_index'], is_global=True),  # active_context
            SpecificComputationValidator(short_name='long_short_decoding_analyses', computation_fn_name='_perform_long_short_decoding_analyses', validate_computation_test=lambda curr_active_pipeline: (curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'].long_results_obj, curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis'].short_results_obj), is_global=True),
            SpecificComputationValidator(short_name='long_short_post_decoding', computation_fn_name='_perform_long_short_post_decoding_analysis', validate_computation_test=lambda curr_active_pipeline: curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding'].rate_remapping.rr_df, is_global=True)
        ]

        for _comp_specifier in _comp_specifiers:
            if (not _comp_specifier.is_global) or include_global_functions:
                if _comp_specifier.short_name in include_includelist:
                    newly_computed_values += _comp_specifier.try_computation_if_needed(curr_active_pipeline, on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)



    """
    short_name:str = field() # 'long_short_post_decoding'
    computation_fn_name:str = field() # '_perform_long_short_post_decoding_analysis'
    validate_computation_test:Callable = field(repr=False) # lambda curr_active_pipeline, computation_filter_name='maze'
    computation_precidence: float = field()
    results_specification: SpecificComputationResultsSpecification = field(default=Factory(SpecificComputationResultsSpecification), repr=True) # (provides_global_keys=['DirectionalMergedDecoders']) # results_specification=SpecificComputationResultsSpecification(provides_global_keys=['DirectionalMergedDecoders'])
    computation_fn_kwargs:dict = field(default=Factory(dict), repr=True)  # {'perform_cache_load': False}]`
    is_global:bool = field(default=False)
    
    @property
    def has_results_spec(self) -> bool:
       return (self.results_specification is not None) 


    @property
    def provides_global_keys(self) -> List[str]:
        if not self.has_results_spec:
            return []
        return (self.results_specification.provides_global_keys) 


    @property
    def requires_global_keys(self) -> List[str]:
        if not self.has_results_spec:
            return []
        return (self.results_specification.requires_global_keys) 


    @classmethod
    def init_from_decorated_fn(cls, a_fn):
        """
        Produces a validator from a function `a_fn` decorated with `@function_attributes` specifying a valid `validate_computation_test`
        
        Example:
        
        @function_attributes(short_name='pfdt_computation', tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-30 19:58', related_items=[],
                         provides_global_keys=[], requires_global_keys=[],
                         validate_computation_test=lambda curr_active_pipeline, global_epoch_name='maze': (curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf1D_dt'], curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf2D_dt']))
        def _perform_time_dependent_placefield_computation(computation_result: ComputationResult, debug_print=False):
            # ... function definiition

                        
        """
        assert hasattr(a_fn, 'validate_computation_test') and (a_fn.validate_computation_test is not None)

        ## Try to retrieve the `results_specification` from the function validator
        results_specification = None
        if hasattr(a_fn, 'results_specification') and (a_fn.results_specification is not None):
            # ensure the alternative non-import syntax isn't used simultaneously
            assert not (hasattr(a_fn, 'provides_global_keys') and (a_fn.provides_global_keys is not None))
            assert not (hasattr(a_fn, 'requires_global_keys') and (a_fn.requires_global_keys is not None))
            results_specification = a_fn.results_specification
            
        else:
            # allows specifying provides_global_keys=['DirectionalMergedDecoders] as a List without importing:
            results_specification = SpecificComputationResultsSpecification()

            if hasattr(a_fn, 'provides_global_keys') and (a_fn.provides_global_keys is not None):
                results_specification.provides_global_keys = list(a_fn.provides_global_keys)
            if hasattr(a_fn, 'requires_global_keys') and (a_fn.requires_global_keys is not None):
                results_specification.requires_global_keys = list(a_fn.requires_global_keys)

        assert (hasattr(a_fn, 'computation_precidence') and (a_fn.computation_precidence is not None))
        computation_precidence = a_fn.computation_precidence

        return cls(short_name=a_fn.short_name, computation_fn_name=a_fn.__name__, validate_computation_test=a_fn.validate_computation_test, results_specification=results_specification, computation_precidence=computation_precidence, is_global=a_fn.is_global)

    def does_name_match(self, name_str: str) -> bool:
        """ checks if either short_name or computation_name"""        
        return ((self.short_name == name_str) or (self.computation_fn_name == name_str))
    
    def is_name_in(self, name_list: List[str]) -> bool:
        """ checks if either short_name or computation_name is contained in the name_list provided. """        
        return ((self.short_name in name_list) or (self.computation_fn_name in name_list))

    def is_dependency_in_required_global_keys(self, provided_global_keys: List[str]) -> bool:
        """ checks if either short_name or computation_name is contained in the name_list provided. """        
        if not self.has_results_spec:
            return False
        return np.any(np.isin(provided_global_keys, (self.results_specification.requires_global_keys or [])))



    # Main Operation Functions ___________________________________________________________________________________________ #
    def try_validate_is_computation_valid(self, curr_active_pipeline, **kwargs) -> bool:
        """ returns True if the existing computation result is present and valid. """
        return self._perform_try_validate_is_computation_valid(self, curr_active_pipeline, **kwargs)


    def try_computation_if_needed(self, curr_active_pipeline, **kwargs):
        return self._perform_try_computation_if_needed(self, curr_active_pipeline, **kwargs)


    def try_remove_provided_keys(self, curr_active_pipeline, **kwargs):
        """Remove any existing results:
            if (removed_results_dict is not None) and len(removed_results_dict) > 0:
                print(f'removed results: {list(removed_results_dict.keys())} because force_recompute was True.')
        """
        if self.has_results_spec:
            removed_results_dict = self.results_specification.remove_provided_keys(curr_active_pipeline.global_computation_results)
            return removed_results_dict
        else:
            return None

    def try_check_missing_provided_keys(self, curr_active_pipeline, debug_print=False) -> bool:
        """ Check for known provided results that are missing, indicating that it needs to be recomputed either way. If no self.results_specification is provided we can't conlude either way and must fall back to the normal validation function. """
        is_known_missing_provided_keys: bool = False # we don't know if we are or not
        if self.has_results_spec:
            are_all_included, (matching_keys_dict, missing_keys_dict) = self.results_specification.has_provided_keys(curr_active_pipeline.global_computation_results)
            if are_all_included:
                if debug_print:
                    print(f'all provided {len(matching_keys_dict)} keys are already present and valid: {list(matching_keys_dict.keys())}')
            else:
                if debug_print:
                    print(f'{len(matching_keys_dict)}/{len(matching_keys_dict) + len(missing_keys_dict)} provided keys are already present and valid:\n\tmatching: {list(matching_keys_dict.keys())}\n\tmissing: {list(missing_keys_dict.keys())}')
                # missing_keys_dict
            is_known_missing_provided_keys = (not are_all_included)
            return is_known_missing_provided_keys
        else:
            # inconclusive:
            return is_known_missing_provided_keys


    def debug_comp_validator_status(self, curr_active_pipeline):
        # Check for existing result:
        are_all_included = None
        matching_keys_dict, missing_keys_dict = None, None
        if self.has_results_spec:
            are_all_included, (matching_keys_dict, missing_keys_dict) = self.results_specification.has_provided_keys(curr_active_pipeline.global_computation_results)
            if are_all_included:
                print(f'all provided {len(matching_keys_dict)} keys are already present and valid: {list(matching_keys_dict.keys())}')
            else:
                print(f'{len(matching_keys_dict)}/{len(matching_keys_dict) + len(missing_keys_dict)} provided keys are already present and valid:\n\tmatching: {list(matching_keys_dict.keys())}\n\tmissing: {list(missing_keys_dict.keys())}')
            return are_all_included

        else:
            print(f'does not have any comp_specifier.results_specification properties so the value is None.') 
            return None


    # Implementations ____________________________________________________________________________________________________ #
    @classmethod
    def _perform_try_validate_is_computation_valid(cls, comp_specifier: "SpecificComputationValidator", curr_active_pipeline, computation_filter_name:str, fail_on_exception=False, progress_print=True, debug_print=False, force_recompute:bool=False) -> bool:
        """ 2023-06-08 - tries to validate (but not perform) the computation to see if it needs to becomputed. 
        
        It can return False for several independent reasons:
            - a critical result is missing
            - force_recompute = True
            - validation function failed or threw an error

        Usage:
            if _comp_name in include_includelist:
                # newly_computed_values += _try_computation_if_needed(curr_active_pipeline, comp_specifier=SpecificComputationValidator(short_name='long_short_post_decoding', computation_fn_name='_perform_long_short_post_decoding_analysis', validate_computation_test=a_validate_computation_test), on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)
        
                did_successfully_validate: bool = cls._perform_try_validate_is_computation_valid(comp_specifier=comp_specifier, curr_active_pipeline=curr_active_pipeline, computation_filter_name=computation_filter_name,
                                                                    fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)
                                                                    
        #TODO 2023-08-31 11:08: - [ ] Made for global computations, but `computation_filter_name` was just added post-hoc. Needs to be updated to use computation_filter_name in perform_specific_computation for non-global functions
                
        """
        def _subfn_try_validate(validate_fail_on_exception:bool=False, is_post_recompute:bool=False) -> bool:
            """ captures: comp_specifier, curr_active_pipeline, computation_filter_name, debug_print """
            try:
                # try the validation again.
                _is_valid = comp_specifier.validate_computation_test(curr_active_pipeline, computation_filter_name=computation_filter_name) # passed the validation
                if (_is_valid is not None) and (isinstance(_is_valid, bool)):
                    return _is_valid

                return True # some are valid just by not throwing errors?

            except (AttributeError, KeyError, TypeError, ValueError, AssertionError) as validation_err:
                # Handle the inner exception
                if is_post_recompute:
                    print(f'Exception occured while validating (`validate_computation_test(...)`) after recomputation:\n Validation exception: {validation_err}')
                if validate_fail_on_exception:
                    raise validation_err
                if debug_print:
                    import traceback # for stack trace formatting
                    print(f'\t encountered error while validating (is_post_recompute: {is_post_recompute}):\n\tValidation exception: {validation_err}\n{traceback.format_exc()}\n.')
            except BaseException:
                raise # unhandled exception
            return False


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        # comp_short_name: str = comp_specifier.short_name
        did_successfully_validate: bool = False
        
        if force_recompute:
            # if force_recompute is True, we always return that it is invalid. We'll need to remove the previous computations before running the new ones though.
            return False

        # Check for existing result:
        is_known_missing_provided_keys: bool = comp_specifier.try_check_missing_provided_keys(curr_active_pipeline)
        if (is_known_missing_provided_keys):
            if debug_print:
                print(f"missing required value, so we don't need to call .validate_computation_test(...) to know it isn't valid!")
            return False

        did_successfully_validate = _subfn_try_validate(validate_fail_on_exception=False, is_post_recompute=False)  # 2024-05-02 - this is returning True indicating successful validation when it should fail! 
        return did_successfully_validate ## #TODO 2024-04-03 11:03: - [ ] this should NEVER be assigned to `needs_compute` 
    

    @classmethod
    def _perform_try_computation_if_needed(cls, comp_specifier: "SpecificComputationValidator", curr_active_pipeline, computation_filter_name:str, on_already_computed_fn=None, fail_on_exception=False, progress_print=True, debug_print=False, force_recompute:bool=False):
        """ 2023-06-08 - tries to perform the computation if the results are missing and it's needed. 
        
        Usage:
            if _comp_name in include_includelist:
                newly_computed_values += _try_computation_if_needed(curr_active_pipeline, comp_specifier=SpecificComputationValidator(short_name='long_short_post_decoding', computation_fn_name='_perform_long_short_post_decoding_analysis', validate_computation_test=a_validate_computation_test), on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)
        
        #TODO 2023-08-31 11:08: - [ ] Made for global computations, but `computation_filter_name` was just added post-hoc. Needs to be updated to use computation_filter_name in perform_specific_computation for non-global functions
                
        """
        def _subfn_try_validate(validate_fail_on_exception:bool=False, is_post_recompute:bool=False) -> bool:
            """ captures: comp_specifier, curr_active_pipeline, computation_filter_name, debug_print """
            try:
                # try the validation again.
                comp_specifier.validate_computation_test(curr_active_pipeline, computation_filter_name=computation_filter_name)
                return True
            except (AttributeError, KeyError, TypeError, ValueError, AssertionError) as validation_err:
                # Handle the inner exception
                if is_post_recompute:
                    print(f'Exception occured while validating (`validate_computation_test(...)`) after recomputation:\n Validation exception: {validation_err}')
                if validate_fail_on_exception:
                    raise validation_err
                if debug_print:
                    import traceback # for stack trace formatting
                    print(f'\t encountered error while validating (is_post_recompute: {is_post_recompute}):\n\tValidation exception: {validation_err}\n{traceback.format_exc()}\n.')
            except BaseException:
                raise # unhandled exception
            return False


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        comp_short_name: str = comp_specifier.short_name
        newly_computed_values = []
        did_successfully_validate: bool = False

        if force_recompute:
            ## Remove any existing results:
            print(f'2024-01-02 - {comp_short_name} _perform_try_computation_if_needed, remove_provided_keys')
            removed_results_dict = comp_specifier.try_remove_provided_keys(curr_active_pipeline)
            if (removed_results_dict is not None) and len(removed_results_dict) > 0:
                print(f'removed results: {list(removed_results_dict.keys())} because force_recompute was True.')

        # Check for existing result:
        did_successfully_validate: bool = cls._perform_try_validate_is_computation_valid(comp_specifier=comp_specifier, curr_active_pipeline=curr_active_pipeline, computation_filter_name=computation_filter_name,
                                                                         fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)
        needs_computation: bool = (not did_successfully_validate)
        
        if needs_computation:
            ## validate_computation_test(...) failed, so we need to recompute.
            if progress_print or debug_print:
                print(f'{comp_short_name} missing.')
            if progress_print or debug_print:
                print(f'\t Recomputing {comp_short_name}...')
            # When this fails due to unwrapping from the load, add `, computation_kwargs_list=[{'perform_cache_load': False}]` as an argument to the `perform_specific_computation` call below
            try:
                curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=[comp_specifier.computation_fn_name], computation_kwargs_list=[comp_specifier.computation_fn_kwargs], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
                if progress_print or debug_print:
                    print(f'\t done.')
            except (AttributeError, KeyError, TypeError, ValueError, AssertionError) as inner_e:
                # Handle the inner exception
                print(f'Exception occured while computing (`perform_specific_computation(...)`):\n Inner exception: {inner_e}')
                if fail_on_exception:
                    raise inner_e
            except BaseException:
                raise # unhandled exception
        
            # Re-validate after computation ______________________________________________________________________________________ #
            did_successfully_validate: bool = _subfn_try_validate(validate_fail_on_exception=False, is_post_recompute=True)
            if did_successfully_validate:
                newly_computed_values.append((comp_short_name, computation_filter_name)) # append the new computation
                
        else:
            if debug_print:
                print(f'\t no recomputation needed! did_successfully_validate: {did_successfully_validate}.\t done.')

        return newly_computed_values

    # ==================================================================================================================== #
    # Dependency Parsing/Determination                                                                                     #
    # ==================================================================================================================== #
    @classmethod
    def find_matching_validators(cls, remaining_comp_specifiers_dict: Dict[str, "SpecificComputationValidator"], probe_fn_names: List[str], debug_print=False):
        """
        Usage:
            remaining_comp_specifiers_dict = deepcopy(_comp_specifiers_dict)
            remaining_comp_specifiers_dict, found_matching_validators, provided_global_keys = SpecificComputationValidator.find_matching_validators(remaining_comp_specifiers_dict=remaining_comp_specifiers_dict,
                                                                                                probe_fn_names=['long_short_decoding_analyses','long_short_fr_indicies_analyses'])

            provided_global_keys
        """
        found_matching_validators = {}
        provided_global_keys = []
        for a_name, a_validator in remaining_comp_specifiers_dict.items():
            for a_probe_fn_name in probe_fn_names:
                # if a_validator.is_name_in(probe_fn_names):
                if a_validator.does_name_match(a_probe_fn_name):
                    found_matching_validators[a_probe_fn_name] = a_validator
                    if debug_print:
                        print(f'found matching validator: {a_validator}')
                    
        # Get each validator's provided keys:
        for a_name, a_found_validator in found_matching_validators.items():
            new_provided_global_keys = a_found_validator.results_specification.provides_global_keys
            provided_global_keys.extend(new_provided_global_keys)

        remaining_comp_specifiers_dict = {k:v for k,v in remaining_comp_specifiers_dict.items() if k not in found_matching_validators}
        if debug_print:
            print(f'len(remaining_comp_specifiers_dict): {len(remaining_comp_specifiers_dict)}, found_matching_validators: {found_matching_validators}')
        return remaining_comp_specifiers_dict, found_matching_validators, provided_global_keys

    @classmethod
    def find_immediate_dependencies(cls, remaining_comp_specifiers_dict: Dict[str, "SpecificComputationValidator"], provided_global_keys: List[str], debug_print=False):
        """
        Usage:

        remaining_comp_specifiers_dict, dependent_validators, provided_global_keys = SpecificComputationValidator.find_immediate_dependencies(remaining_comp_specifiers_dict=remaining_comp_specifiers_dict, provided_global_keys=provided_global_keys)
        provided_global_keys

        """
        dependent_validators = {}
        for a_name, a_validator in remaining_comp_specifiers_dict.items():
            # set(provided_global_keys)
            # set(a_validator.results_specification.requires_global_keys)
            if a_validator.is_dependency_in_required_global_keys(provided_global_keys):
                dependent_validators[a_name] = a_validator
            # (provided_global_keys == (a_validator.results_specification.requires_global_keys or []))

        for a_name, a_found_validator in dependent_validators.items():
            new_provided_global_keys = a_found_validator.results_specification.provides_global_keys
            provided_global_keys.extend(new_provided_global_keys)
            remaining_comp_specifiers_dict.pop(a_name) # remove

        remaining_comp_specifiers_dict = {k:v for k,v in remaining_comp_specifiers_dict.items() if k not in dependent_validators}

        if debug_print:
            print(f'len(remaining_comp_specifiers_dict): {len(remaining_comp_specifiers_dict)}, dependent_validators: {dependent_validators}')
        return remaining_comp_specifiers_dict, dependent_validators, provided_global_keys

    @classmethod
    def find_provided_result_keys(cls, remaining_comp_specifiers_dict: Dict[str, "SpecificComputationValidator"], probe_fn_names: List[str]) -> List[str]:
        """ returns a list of computed properties that the specified functions provide. 
        
        Usage:
            provided_global_keys = SpecificComputationValidator.find_provided_result_keys(remaining_comp_specifiers_dict=remaining_comp_specifiers_dict,
                                                                                                probe_fn_names=['perform_wcorr_shuffle_analysis',  'merged_directional_placefields', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring'],
                                                                                                )
            provided_global_keys # ['DirectionalMergedDecoders', 'DirectionalDecodersEpochsEvaluations', 'SequenceBased']

        """
        provided_global_keys = []
        for a_name, a_validator in remaining_comp_specifiers_dict.items():
            for a_probe_fn_name in probe_fn_names:
                if a_validator.does_name_match(a_probe_fn_name):
                    provided_global_keys.extend(a_validator.results_specification.provides_global_keys) # Get each validator's provided keys:
                    
        return provided_global_keys
    

    @classmethod
    def find_validators_providing_results(cls, remaining_comp_specifiers_dict: Dict[str, "SpecificComputationValidator"], probe_provided_result_keys: List[str], return_flat_list:bool=True) -> List[str]:
        """ returns a list of computed properties that the specified functions provide. 
        
        Usage:

            found_validators_dict = SpecificComputationValidator.find_validators_providing_results(remaining_comp_specifiers_dict=remaining_comp_specifiers_dict,
                                                                                                probe_provided_result_keys=['DirectionalMergedDecoders', 'DirectionalDecodersEpochsEvaluations', 'SequenceBased'])
            [v.computation_fn_name for v in found_validators_dict]

        """
        if return_flat_list:
            found_matching_validators_list = []
        else:
            found_matching_validators = {}

        for a_name, a_validator in remaining_comp_specifiers_dict.items():
            for a_probe_result_name in probe_provided_result_keys:
                if a_probe_result_name in (a_validator.provides_global_keys or []):
                    ## this validator matches
                    if return_flat_list:
                        # found_matching_validators[a_probe_result_name] = a_validator
                        found_matching_validators_list.append(a_validator)
                    else:              
                        if a_probe_result_name not in found_matching_validators:
                            found_matching_validators[a_probe_result_name] = [] # make a new list
                        found_matching_validators[a_probe_result_name].append(a_validator)
                    # 
                        
        if return_flat_list:
            return found_matching_validators_list
        else:
            return found_matching_validators
    
        

        

# I have a class `SpecificComputationValidator` that is used to keep track of computations in a custom pipeline and manage computation order. The key fields are `computation_precidence`, `provides_global_keys`, and `requires_global_keys`. `provides_global_keys` specifies which keys are provided after the computation completes, and `requires_global_keys` specifies which keys are required before computation of the function is possible. This means that a function with a `a_key` in `requires_global_keys` is dependent on all functions with `a_key` in their `provides_global_keys`. I'd like you to propose a datastructure that can be used efficiently track dependencies, for example if the key `a_changed_key` is modified, determine all downstream dependent functions that will need to be recomputed. Here are some example validators:


# ```python
# {'_split_to_directional_laps': SpecificComputationValidator(short_name='split_to_directional_laps', computation_fn_name='_split_to_directional_laps', computation_precidence=1000.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['DirectionalLaps'], requires_global_keys=[]), computation_fn_kwargs={}, is_global=True),
#  '_build_merged_directional_placefields': SpecificComputationValidator(short_name='merged_directional_placefields', computation_fn_name='_build_merged_directional_placefields', computation_precidence=1000.01, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['DirectionalMergedDecoders'], requires_global_keys=['DirectionalLaps']), computation_fn_kwargs={}, is_global=True),
#  '_perform_PBE_stats_analyses': SpecificComputationValidator(short_name='PBE_stats', computation_fn_name='_perform_PBE_stats_analyses', computation_precidence=1001.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=True),
#  'perform_rank_order_shuffle_analysis': SpecificComputationValidator(short_name='rank_order_shuffle_analysis', computation_fn_name='perform_rank_order_shuffle_analysis', computation_precidence=1001.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['RankOrder'], requires_global_keys=['DirectionalLaps']), computation_fn_kwargs={}, is_global=True),
#  '_decode_continuous_using_directional_decoders': SpecificComputationValidator(short_name='directional_decoders_decode_continuous', computation_fn_name='_decode_continuous_using_directional_decoders', computation_precidence=1002.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['DirectionalDecodersDecoded'], requires_global_keys=['DirectionalLaps', 'RankOrder', 'DirectionalMergedDecoders']), computation_fn_kwargs={}, is_global=True),
#  '_decode_and_evaluate_epochs_using_directional_decoders': SpecificComputationValidator(short_name='directional_decoders_evaluate_epochs', computation_fn_name='_decode_and_evaluate_epochs_using_directional_decoders', computation_precidence=1002.1, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['DirectionalDecodersEpochsEvaluations'], requires_global_keys=['DirectionalLaps', 'RankOrder', 'DirectionalMergedDecoders']), computation_fn_kwargs={}, is_global=True),
#  '_decoded_epochs_heuristic_scoring': SpecificComputationValidator(short_name='directional_decoders_epoch_heuristic_scoring', computation_fn_name='_decoded_epochs_heuristic_scoring', computation_precidence=1002.2, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=['DirectionalLaps', 'RankOrder', 'DirectionalMergedDecoders', 'DirectionalDecodersDecoded', 'DirectionalDecodersEpochsEvaluations']), computation_fn_kwargs={}, is_global=True),
#  '_split_train_test_laps_data': SpecificComputationValidator(short_name='directional_train_test_split', computation_fn_name='_split_train_test_laps_data', computation_precidence=1002.3, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['TrainTestSplit'], requires_global_keys=['DirectionalLaps', 'RankOrder', 'DirectionalMergedDecoders']), computation_fn_kwargs={}, is_global=True),
#  '_build_trial_by_trial_activity_metrics': SpecificComputationValidator(short_name='trial_by_trial_metrics', computation_fn_name='_build_trial_by_trial_activity_metrics', computation_precidence=1002.4, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['TrialByTrialActivity'], requires_global_keys=['DirectionalLaps', 'RankOrder', 'DirectionalMergedDecoders']), computation_fn_kwargs={}, is_global=True),
#  '_perform_long_short_decoding_analyses': SpecificComputationValidator(short_name='long_short_decoding_analyses', computation_fn_name='_perform_long_short_decoding_analyses', computation_precidence=1003.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['long_short_leave_one_out_decoding_analysis'], requires_global_keys=[]), computation_fn_kwargs={}, is_global=True),
#  '_perform_long_short_pf_overlap_analyses': SpecificComputationValidator(short_name='short_long_pf_overlap_analyses', computation_fn_name='_perform_long_short_pf_overlap_analyses', computation_precidence=1003.02, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['short_long_pf_overlap_analyses'], requires_global_keys=[]), computation_fn_kwargs={}, is_global=True),
#  '_perform_long_short_firing_rate_analyses': SpecificComputationValidator(short_name='long_short_fr_indicies_analyses', computation_fn_name='_perform_long_short_firing_rate_analyses', computation_precidence=1003.03, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['long_short_fr_indicies_analysis'], requires_global_keys=[]), computation_fn_kwargs={}, is_global=True),
#  '_perform_jonathan_replay_firing_rate_analyses': SpecificComputationValidator(short_name='jonathan_firing_rate_analysis', computation_fn_name='_perform_jonathan_replay_firing_rate_analyses', computation_precidence=1003.04, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['jonathan_firing_rate_analysis'], requires_global_keys=['long_short_fr_indicies_analysis']), computation_fn_kwargs={}, is_global=True),
#  '_perform_long_short_post_decoding_analysis': SpecificComputationValidator(short_name='long_short_post_decoding', computation_fn_name='_perform_long_short_post_decoding_analysis', computation_precidence=1003.05, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['long_short_post_decoding'], requires_global_keys=['long_short_fr_indicies_analysis', 'long_short_leave_one_out_decoding_analysis']), computation_fn_kwargs={}, is_global=True),
#  '_perform_long_short_instantaneous_spike_rate_groups_analysis': SpecificComputationValidator(short_name='long_short_inst_spike_rate_groups', computation_fn_name='_perform_long_short_instantaneous_spike_rate_groups_analysis', computation_precidence=1003.06, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['long_short_inst_spike_rate_groups'], requires_global_keys=['jonathan_firing_rate_analysis', 'long_short_fr_indicies_analysis']), computation_fn_kwargs={}, is_global=True),
#  '_perform_long_short_endcap_analysis': SpecificComputationValidator(short_name='long_short_endcap_analysis', computation_fn_name='_perform_long_short_endcap_analysis', computation_precidence=1003.07, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['long_short_endcap'], requires_global_keys=['jonathan_firing_rate_analysis']), computation_fn_kwargs={}, is_global=True),
#  'perform_wcorr_shuffle_analysis': SpecificComputationValidator(short_name='wcorr_shuffle_analysis', computation_fn_name='perform_wcorr_shuffle_analysis', computation_precidence=1005.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=['SequenceBased'], requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders', 'RankOrder', 'DirectionalDecodersEpochsEvaluations']), computation_fn_kwargs={}, is_global=True),
#  '_add_extended_pf_peak_information': SpecificComputationValidator(short_name='extended_pf_peak_information', computation_fn_name='_add_extended_pf_peak_information', computation_precidence=1005.4, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=['DirectionalLaps', 'RankOrder', 'jonathan_firing_rate_analysis']), computation_fn_kwargs={}, is_global=True),
#  '_perform_lap_direction_determination': SpecificComputationValidator(short_name='lap_direction_determination', computation_fn_name='_perform_lap_direction_determination', computation_precidence=-0.1, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_baseline_placefield_computation': SpecificComputationValidator(short_name='pf_computation', computation_fn_name='_perform_baseline_placefield_computation', computation_precidence=0.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_position_decoding_computation': SpecificComputationValidator(short_name='position_decoding', computation_fn_name='_perform_position_decoding_computation', computation_precidence=1.01, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_two_step_position_decoding_computation': SpecificComputationValidator(short_name='position_decoding_two_step', computation_fn_name='_perform_two_step_position_decoding_computation', computation_precidence=1.02, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_recursive_latent_placefield_decoding': SpecificComputationValidator(short_name='recursive_latent_pf_decoding', computation_fn_name='_perform_recursive_latent_placefield_decoding', computation_precidence=1.03, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_specific_epochs_decoding': SpecificComputationValidator(short_name='_perform_specific_epochs_decoding', computation_fn_name='_perform_specific_epochs_decoding', computation_precidence=1.04, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_velocity_vs_pf_density_computation': SpecificComputationValidator(short_name='EloyAnalysis', computation_fn_name='_perform_velocity_vs_pf_density_computation', computation_precidence=2.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_velocity_vs_pf_simplified_count_density_computation': SpecificComputationValidator(short_name='velocity_vs_pf_simplified_count_density', computation_fn_name='_perform_velocity_vs_pf_simplified_count_density_computation', computation_precidence=2.01, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_DEP_perform_pf_find_ratemap_peaks_computation': SpecificComputationValidator(short_name='_DEP_ratemap_peaks', computation_fn_name='_DEP_perform_pf_find_ratemap_peaks_computation', computation_precidence=2.02, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation': SpecificComputationValidator(short_name='ratemap_peaks_prominence2d', computation_fn_name='_perform_pf_find_ratemap_peaks_peak_prominence2d_computation', computation_precidence=2.03, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_placefield_overlap_computation': SpecificComputationValidator(short_name='placefield_overlap', computation_fn_name='_perform_placefield_overlap_computation', computation_precidence=2.04, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_extended_statistics_computation': SpecificComputationValidator(short_name='extended_stats', computation_fn_name='_perform_extended_statistics_computation', computation_precidence=3.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_spike_burst_detection_computation': SpecificComputationValidator(short_name='spike_burst_detection', computation_fn_name='_perform_spike_burst_detection_computation', computation_precidence=4.0, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_firing_rate_trends_computation': SpecificComputationValidator(short_name='firing_rate_trends', computation_fn_name='_perform_firing_rate_trends_computation', computation_precidence=4.01, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_time_dependent_placefield_computation': SpecificComputationValidator(short_name='pfdt_computation', computation_fn_name='_perform_time_dependent_placefield_computation', computation_precidence=9, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False),
#  '_perform_time_dependent_pf_sequential_surprise_computation': SpecificComputationValidator(short_name='pf_dt_sequential_surprise', computation_fn_name='_perform_time_dependent_pf_sequential_surprise_computation', computation_precidence=9.01, results_specification=SpecificComputationResultsSpecification(provides_global_keys=[], requires_global_keys=[]), computation_fn_kwargs={}, is_global=False)}
# ```

# 'perform_wcorr_shuffle_analysis': requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders', 'RankOrder', 'DirectionalDecodersEpochsEvaluations']


class DependencyGraph:
    """
    # Example usage

    from pyphoplacecellanalysis.General.Model.SpecificComputationValidation import DependencyGraph, SpecificComputationValidator, SpecificComputationResultsSpecification

    _comp_specifiers_dict: Dict[str, SpecificComputationValidator] = curr_active_pipeline.get_merged_computation_function_validators()
    validators = deepcopy(_comp_specifiers_dict) # { ... }  # Your validators here
    print(validators)
    graph = DependencyGraph(validators)
    """
    def __init__(self, validators: Dict[str, SpecificComputationValidator]):
        self.graph = nx.DiGraph()
        self.validators = validators
        self.build_graph(validators)
    
    def build_graph(self, validators):
        for key, validator in validators.items():
            self.graph.add_node(key, provides_global_keys=validator.results_specification.provides_global_keys, requires_global_keys=validator.results_specification.requires_global_keys, validator=validator)
            for required_key in validator.results_specification.requires_global_keys:
                for provider_key, provider_validator in validators.items():
                    if required_key in provider_validator.results_specification.provides_global_keys:
                        self.graph.add_edge(provider_key, key)

    # def __init__(self, validators):
    #     self.graph = nx.DiGraph()
    #     self.build_graph(validators)
    
    # def build_graph(self, validators):
    #     for key, validator in validators.items():
    #         self.graph.add_node(key)
    #         for required_key in validator.results_specification.requires_global_keys:
    #             for provider_key, provider_validator in validators.items():
    #                 if required_key in provider_validator.results_specification.provides_global_keys:
    #                     self.graph.add_edge(provider_key, key)
    
    # def get_downstream_dependents(self, modified_key):
    #     dependents = set()
    #     for node in self.graph.nodes:
    #         if modified_key in validators[node].results_specification.provides_global_keys:
    #             dependents.update(nx.descendants(self.graph, node))
    #     return dependents
    
    def get_downstream_dependents(self, modified_key: str, debug_print=False):
        dependents = set()
        for node in self.graph.nodes:
            if debug_print:
                print(f'node: {node}:')
                print(f"\tself.graph.nodes[node]['provides_global_keys']: {self.graph.nodes[node]['provides_global_keys']}")
            if modified_key in self.graph.nodes[node]['provides_global_keys']:
                if debug_print:
                    print(f'modified_key: {modified_key}')
                dependents.update(nx.descendants(self.graph, node))
        return dependents

    def get_upstream_requirements(self, target_key: str, debug_print=False):
        """
        upstream_requirements = graph.get_upstream_requirements('perform_wcorr_shuffle_analysis')
        print(upstream_requirements)
        """

        requirements = set()
        for node in self.graph.nodes:
            if debug_print:
                print(f'node: {node}:')
                print(f"\tself.graph.nodes[node]['requires_global_keys']: {self.graph.nodes[node]['requires_global_keys']}")
            if target_key in self.graph.nodes[node]['requires_global_keys']:
                if debug_print:
                    print(f'target_key: {target_key}')
                requirements.update(nx.descendants(self.graph, node))
        return requirements

    
    def visualize(self):
        import matplotlib.pyplot as plt
        # Filter out nodes with no parents or children
        nodes_to_draw = [node for node in self.graph.nodes if list(self.graph.predecessors(node)) or list(self.graph.successors(node))]
        subgraph = self.graph.subgraph(nodes_to_draw)
        
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")
        plt.show()

    # def visualize(self):
    #     # Filter out nodes with no parents or children
    #     nodes_to_draw = [node for node in self.graph.nodes if list(self.graph.predecessors(node)) or list(self.graph.successors(node))]
    #     subgraph = self.graph.subgraph(nodes_to_draw)
        
    #     # Use pygraphviz for better visualization
    #     pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot')
    #     nx.draw(subgraph, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray", arrows=True)
    #     plt.show()

    # def visualize(self):
    #     # Filter out nodes with no parents or children
    #     nodes_to_draw = [node for node in self.graph.nodes if list(self.graph.predecessors(node)) or list(self.graph.successors(node))]
    #     subgraph = self.graph.subgraph(nodes_to_draw)
        
    #     # Use a spring layout for better spacing
    #     pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        
    #     plt.figure(figsize=(12, 12))
    #     nx.draw(subgraph, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray", arrows=True)
        
    #     # Draw edge labels if needed
    #     edge_labels = {(u, v): f'{u} -> {v}' for u, v in subgraph.edges}
    #     nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red')
        
    #     plt.show()


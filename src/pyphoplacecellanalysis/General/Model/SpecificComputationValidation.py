from attrs import define, Factory, fields
from typing import Callable

@define(slots=False, repr=False)
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
    short_name:str # 'long_short_post_decoding'
    computation_fn_name:str # '_perform_long_short_post_decoding_analysis'
    validate_computation_test:Callable # lambda curr_active_pipeline, computation_filter_name='maze'
    computation_fn_kwargs:dict = Factory(dict) # {'perform_cache_load': False}]`
    is_global:bool = False
    
    @classmethod
    def init_from_decorated_fn(cls, a_fn):
        """
        Produces a validator from a function `a_fn` decorated with `@function_attributes` specifying a valid `validate_computation_test`
        
        Example:
        
        @function_attributes(short_name='pfdt_computation', tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-30 19:58', related_items=[],
                         validate_computation_test=lambda curr_active_pipeline, global_epoch_name='maze': (curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf1D_dt'], curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf2D_dt']))
        def _perform_time_dependent_placefield_computation(computation_result: ComputationResult, debug_print=False):
            # ... function definiition

                        
        """
        assert hasattr(a_fn, 'validate_computation_test') and (a_fn.validate_computation_test is not None)
        return cls(short_name=a_fn.short_name, computation_fn_name=a_fn.__name__, validate_computation_test=a_fn.validate_computation_test, is_global=a_fn.is_global)
        

    def try_computation_if_needed(self, curr_active_pipeline, **kwargs):
        return self._perform_try_computation_if_needed(self, curr_active_pipeline, **kwargs)

    @classmethod
    def _perform_try_computation_if_needed(cls, comp_specifier: "SpecificComputationValidator", curr_active_pipeline, computation_filter_name:str, on_already_computed_fn=None, fail_on_exception=False, progress_print=True, debug_print=False, force_recompute:bool=False):
        """ 2023-06-08 - tries to perform the computation if the results are missing and it's needed. 
        
        Usage:
            if _comp_name in include_includelist:
                newly_computed_values += _try_computation_if_needed(curr_active_pipeline, comp_specifier=SpecificComputationValidator(short_name='long_short_post_decoding', computation_fn_name='_perform_long_short_post_decoding_analysis', validate_computation_test=a_validate_computation_test), on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)
        
        #TODO 2023-08-31 11:08: - [ ] Made for global computations, but `computation_filter_name` was just added post-hoc. Needs to be updated to use computation_filter_name in perform_specific_computation for non-global functions
                
        """
        comp_short_name: str = comp_specifier.short_name
        newly_computed_values = []
        try:
            comp_specifier.validate_computation_test(curr_active_pipeline, computation_filter_name=computation_filter_name)
            if on_already_computed_fn is not None:
                on_already_computed_fn(comp_short_name, computation_filter_name)
        except (AttributeError, KeyError, TypeError) as initial_validation_err:
            ## validate_computation_test(...) failed, so we need to recompute.
            if progress_print or debug_print:
                print(f'{comp_short_name} missing.')
            if debug_print:
                import traceback # for stack trace formatting
                print(f'\t encountered error while initially validating: initial_validation_err: {initial_validation_err}\n{traceback.format_exc()}\n.')
            if progress_print or debug_print:
                print(f'\t Recomputing {comp_short_name}...')
            # When this fails due to unwrapping from the load, add `, computation_kwargs_list=[{'perform_cache_load': False}]` as an argument to the `perform_specific_computation` call below
            try:
                curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=[comp_specifier.computation_fn_name], computation_kwargs_list=[comp_specifier.computation_fn_kwargs], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
                if progress_print or debug_print:
                    print(f'\t done.')
                # try the validation again.
                comp_specifier.validate_computation_test(curr_active_pipeline, computation_filter_name=computation_filter_name)
                newly_computed_values.append((comp_short_name, computation_filter_name))
            except (AttributeError, KeyError, TypeError, ValueError) as inner_e:
                # Handle the inner exception
                print(f'Exception occured while computing (`perform_specific_computation(...)`) or validating (`validate_computation_test(...)`) after recomputation:\n Inner exception: {inner_e}')
                if fail_on_exception:
                    raise inner_e

        except BaseException:
            raise

        return newly_computed_values



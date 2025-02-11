from copy import deepcopy
from pathlib import Path
from neuropy.core.parameters import ParametersContainer
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.function_helpers import get_fn_kwargs_with_defaults, get_decorated_function_attributes, fn_best_name
from pyphocorehelpers.print_helpers import strip_type_str_to_classname
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from typing import Dict, List, Tuple, Optional, Callable, Union, Any


# from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import *

# print_keys_if_possible('active_sess_config', curr_active_pipeline.active_sess_config, max_depth=4)

    # │   ├── preprocessing_parameters: neuropy.core.session.Formats.SessionSpecifications.ParametersContainer  = ParametersContainer(epoch_estimation_parameters=DynamicContainer({'laps': DynamicContainer({'N': 20, 'should_backup_extant_laps_obj': True, 'use_direction_dependent_laps': True}), 'PBEs': DynamicContainer({'thresh': (0, 1.5), 'min_dur': 0.03, 'merge_dur': 0.1, 'max_dur': 0.6})...
    # 	│   ├── epoch_estimation_parameters: neuropy.utils.dynamic_container.DynamicContainer  = DynamicContainer({'laps': DynamicContainer({'N': 20, 'should_backup_extant_laps_obj': True, 'use_direction_dependent_laps': True}), 'PBEs': DynamicContainer({'thresh': (0, 1.5), 'min_dur': 0.03, 'merge_dur': 0.1, 'max_dur': 0.6}), 'replays': DynamicContainer({'require_intersec...
    # 		│   ├── laps: neuropy.utils.dynamic_container.DynamicContainer  = DynamicContainer({'N': 20, 'should_backup_extant_laps_obj': True, 'use_direction_dependent_laps': True})
    # 			│   ├── N: int  = 20
    # 			│   ├── should_backup_extant_laps_obj: bool  = True
    # 			│   ├── use_direction_dependent_laps: bool  = True
    # 		│   ├── PBEs: neuropy.utils.dynamic_container.DynamicContainer  = DynamicContainer({'thresh': (0, 1.5), 'min_dur': 0.03, 'merge_dur': 0.1, 'max_dur': 0.6})
    # 			│   ├── thresh: tuple  = (0, 1.5) - (2,)
    # 			│   ├── min_dur: float  = 0.03
    # 			│   ├── merge_dur: float  = 0.1
    # 			│   ├── max_dur: float  = 0.6
    # 		│   ├── replays: neuropy.utils.dynamic_container.DynamicContainer  = DynamicContainer({'require_intersecting_epoch': 85 epochs<br>array([[-inf, 10.578],<br>       [13.1822, 18.7536],<br>       [21.4567, 32.9346],<br>       [35.6037, 35.6373],<br>       [41.2429, 46.6152],<br>       [49.9527, 52.5542],<br>       [56.4928, 59.9956],<br>       [62...
    # 			│   ├── require_intersecting_epoch: neuropy.core.epoch.Epoch  = 85 epochs<br>array([[-inf, 10.578],<br>       [13.1822, 18.7536],<br>       [21.4567, 32.9346],<br>       [35.6037, 35.6373],<br>       [41.2429, 46.6152],<br>       [49.9527, 52.5542],<br>       [56.4928, 59.9956],<br>       [62.5318, 67.8689],<br>       [73.1757, 75.2434],<b...
    # 			│   ├── min_epoch_included_duration: float  = 0.06
    # 			│   ├── max_epoch_included_duration: float  = 0.6
    # 			│   ├── maximum_speed_thresh: NoneType  = None
    # 			│   ├── min_inclusion_fr_active_thresh: float  = 1.0
    # 			│   ├── min_num_unique_aclu_inclusions: int  = 5


# curr_active_pipeline.registered_global_computation_function_docs_dict
# curr_active_pipeline.registered_merged_computation_function_dict

@metadata_attributes(short_name=None, tags=['template', 'jninja2', 'parameters', 'code-gen'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-07 14:20', related_items=[])
class GlobalComputationParametersAttrsClassTemplating:
    """ Generates Special Classes to hold the parameters for global computation functions
    
    Used to programmatically generate:
        `General/Model/SpecificComputationParameterTypes.py` classes - on 2024-10-07
    
    
    from pyphoplacecellanalysis.General.PipelineParameterClassTemplating import GlobalComputationParametersAttrsClassTemplating
    
    # ## Version with nested keys expressed as '/' separated flat strings
    # flat_computation_function_default_kwargs_values_dict = flatten_dict(registered_merged_computation_function_default_kwargs_dict)
    # _defn_lines, _flat_fields_tuples_list, _base_variable_name_only_values_dict, _base_variable_name_only_types_dict = TempAttrsParamsClass._build_kwargs_class_defns(flat_computation_function_default_kwargs_values_dict=flat_computation_function_default_kwargs_values_dict)
    # print(_defn_lines)
    # _flat_fields_tuples_list


    """
    types_override_dict = {'time_bin_size': Optional[float], 'decoding_time_bin_size': Optional[float], 'instantaneous_time_bin_size_seconds': Optional[float],
        'override_long_epoch_name': Optional[str], 'override_short_epoch_name': Optional[str], 'debug_output_hdf5_file_path': Optional[Path], 
    }
    ignore_kwarg_names = ['include_includelist', 'debug_print']
    
    @function_attributes(short_name=None, tags=['template', 'definitions'], input_requires=[], output_provides=[], uses=[], used_by=['_subfn_build_attrs_parameters_classes'], creation_date='2024-10-07 12:04', related_items=[])
    @classmethod
    def _build_kwargs_class_defns(cls, flat_computation_function_default_kwargs_values_dict, flat_computation_function_default_kwargs_types_dict=None):
        """ 
        Captures: types_override_dict
        
        Outputs:
            _flat_fields_tuples_list: List[Tuple[str, str, str]]: (a_name, a_type, a_default_value) tuples
            
            
        Usage:
            flat_computation_function_default_kwargs_values_dict = flatten_dict(registered_merged_computation_function_default_kwargs_dict)        
            _defn_lines, _flat_fields_tuples_list, _base_variable_name_only_values_dict, _base_variable_name_only_types_dict = _build_kwargs_class_defns(flat_computation_function_default_kwargs_values_dict=flat_computation_function_default_kwargs_values_dict)
            print(_defn_lines)

        """
        if flat_computation_function_default_kwargs_types_dict is None:
            flat_computation_function_default_kwargs_types_dict = {k:type(v) for k, v in flat_computation_function_default_kwargs_values_dict.items()}

        _defn_lines: str = ''
        _base_variable_name_only_values_dict = {}
        _base_variable_name_only_types_dict = {}
        _flat_fields_tuples_list = [] # (a_name, a_type, a_default_value) tuples

        for k, v in flat_computation_function_default_kwargs_values_dict.items():
            curr_type = flat_computation_function_default_kwargs_types_dict[k]
            final_var_name: str = k.split('/')[-1] # 'directional_decoders_evaluate_epochs/should_skip_radon_transform' gets 'should_skip_radon_transform'
            override_curr_type = cls.types_override_dict.get(final_var_name, None)
            if override_curr_type is not None:
                curr_type = override_curr_type
            if isinstance(v, str):
                v = f"'{v}'" # add quotes around the actual string value, so the rendered value has the quotes
            curr_type_str: str = strip_type_str_to_classname(str(curr_type))
            curr_defn_line = f"{k}: {curr_type_str} = {v}\n"

            if final_var_name in _base_variable_name_only_values_dict:
                _prev_variable_value = _base_variable_name_only_values_dict[final_var_name]
                print(f'WARNING: variable "{final_var_name}" already exists. original value: {_prev_variable_value}, new_value: {v} ')
            _base_variable_name_only_values_dict[final_var_name] = v
            _base_variable_name_only_types_dict[final_var_name] = curr_type
            _defn_lines += curr_defn_line
            _flat_fields_tuples_list.append((final_var_name, curr_type_str, v,))

        return _defn_lines, _flat_fields_tuples_list, _base_variable_name_only_values_dict, _base_variable_name_only_types_dict


    @function_attributes(short_name=None, tags=['jninja2', 'template', 'private'], input_requires=[], output_provides=[], uses=['cls._build_kwargs_class_defns'], used_by=['main_generate_params_classes'], creation_date='2024-10-07 12:03', related_items=[])
    @classmethod
    def _subfn_build_attrs_parameters_classes(cls, registered_merged_computation_function_default_kwargs_dict, params_defn_save_path=None, print_defns: bool = True, **render_kwargs):
        """ builds the parameters classes for the global computations
        """
        import sys
        import os
        import platform
        import pkg_resources # for Slurm templating
        from jinja2 import Environment, FileSystemLoader # for Slurm templating
        
        # ## Default:
        # attrs_class_defn_template_filename: str = 'attrs_class_defn_template.py.j2' # used for the 
        # attrs_container_class_defn_template_filename: str = 'attrs_container_class_defn_template.py.j2' ## used for the individual computation-specific parameter subclasses

        ## with `param.Parameterized` support:
        attrs_class_defn_template_filename: str = 'attrs_plus_param_class_defn_template.py.j2' # used for the 
        attrs_container_class_defn_template_filename: str = 'attrs_plus_param_container_class_defn_template.py.j2' ## used for the individual computation-specific parameter subclasses

        # Set up Jinja2 environment
        template_path = pkg_resources.resource_filename('pyphoplacecellanalysis.Resources', 'Templates')
        env = Environment(loader=FileSystemLoader(template_path))
        attrs_class_defn_template = env.get_template(attrs_class_defn_template_filename)

        nested_classes_dict = {}
        imports_dict = {}

        for k, v in registered_merged_computation_function_default_kwargs_dict.items():
            k = k.removeprefix('_') # do not allow starting with underscores
            _param_class_name: str = f'{k}_Parameters'
            # _param_class_name = _param_class_name.removeprefix('_') # remove leading underscores
            
            # nested_classes_dict[k] = CodeConversion.convert_dictionary_to_class_defn(v, _param_class_name, copy_to_clipboard=False, include_initializer_default_values=True)
            _defn_lines, _flat_fields_tuples_list, _base_variable_name_only_values_dict, _base_variable_name_only_types_dict = cls._build_kwargs_class_defns(v)
            attrs_defn_str: str = attrs_class_defn_template.render(class_name=_param_class_name, fields_tuples_list=_flat_fields_tuples_list, **render_kwargs)
            # Remove empty lines using generator expression
            text_without_empty_lines: str = '\n'.join(line for line in attrs_defn_str.split('\n') if line.strip())
            nested_classes_dict[k] = text_without_empty_lines
            imports_dict[k] = _param_class_name


        # ==================================================================================================================== #
        # Template Container Class                                                                                             #
        # ==================================================================================================================== #
        contained_parameter_type_names: List[str] = list(imports_dict.values())
        # contained_parameter_type_names: List[str] = [
        #         "merged_directional_placefields_Parameters",
        #         "rank_order_shuffle_analysis_Parameters",
        #         "directional_decoders_decode_continuous_Parameters",
        #         "directional_decoders_evaluate_epochs_Parameters",
        #         "directional_train_test_split_Parameters",
        #         "long_short_decoding_analyses_Parameters",
        #         "long_short_rate_remapping_Parameters",
        #         "long_short_inst_spike_rate_groups_Parameters",
        #         "wcorr_shuffle_analysis_Parameters",
        #         "perform_specific_epochs_decoding_Parameters",
        #         "DEP_ratemap_peaks_Parameters",
        #         "ratemap_peaks_prominence2d_Parameters",
        #     ]            
        # Set up Jinja2 environment
        template_path = pkg_resources.resource_filename('pyphoplacecellanalysis.Resources', 'Templates')
        env = Environment(loader=FileSystemLoader(template_path))
        # attrs_container_class_defn_template = env.get_template('attrs_container_class_defn_template.py.j2')
        # attrs_container_class_defn_str: str = attrs_container_class_defn_template.render(
        #     container_class_name="ComputationKWargParameters",
        #     base_classes=["HDF_SerializationMixin", "AttrsBasedClassHelperMixin", "BaseGlobalComputationParameters"],
        #     class_names= contained_parameter_type_names,
        #     container_class_docstring="The base class for computation parameter types."
        # )


        # `attrs_plus_param_container_class_defn_template.py.j2`: Plus `param.Parameterized` implementation
        attrs_plus_param_container_class_defn_template = env.get_template(attrs_container_class_defn_template_filename)
        attrs_container_class_defn_str: str = attrs_plus_param_container_class_defn_template.render(
            container_class_name="ComputationKWargParameters",
            base_classes=["HDF_SerializationMixin", "AttrsBasedClassHelperMixin", "BaseGlobalComputationParameters"], # , "param.Parameterized"
            class_names= contained_parameter_type_names,
            container_class_docstring="The base class for computation parameter types."
        )






        attrs_container_class_defn_str = '\n'.join(line for line in attrs_container_class_defn_str.split('\n') if line.strip())
        # print(f'attrs_container_class_defn_str\n{attrs_container_class_defn_str}\n\n')

        code_str: str = '\n\n'.join(list(nested_classes_dict.values())) # add comment above code
        ## add in `attrs_container_class_defn_str`
        code_str = code_str + '\n\n' + attrs_container_class_defn_str + '\n\n'

        if print_defns:
            print(code_str)

        if params_defn_save_path:
            with open(params_defn_save_path, 'w') as script_file:
                script_file.write(code_str)

        return code_str, nested_classes_dict, imports_dict



    @classmethod
    def main_extract_params_default_values(cls, curr_active_pipeline):
        """ 
        from pyphoplacecellanalysis.General.PipelineParameterClassTemplating import GlobalComputationParametersAttrsClassTemplating
        registered_merged_computation_function_default_kwargs_dict = GlobalComputationParametersAttrsClassTemplating.main_extract_params_default_values(curr_active_pipeline=curr_active_pipeline)
        """
        ignore_kwarg_names = cls.ignore_kwarg_names
        registered_merged_computation_function_default_kwargs_dict = {fn_best_name(v):get_fn_kwargs_with_defaults(v, ignore_kwarg_names=ignore_kwarg_names) for k, v in curr_active_pipeline.registered_merged_computation_function_dict.items()}
        registered_merged_computation_function_default_kwargs_dict = {k:v for k, v in registered_merged_computation_function_default_kwargs_dict.items() if len(v)>0} # filter empty lists
        return registered_merged_computation_function_default_kwargs_dict
        


    @classmethod
    def main_generate_params_classes(cls, curr_active_pipeline, print_defns=False):
        """ Main function called with a `curr_active_pipeline` to explicitly generate the boilerplate parameters classes
        
        from pyphoplacecellanalysis.General.PipelineParameterClassTemplating import GlobalComputationParametersAttrsClassTemplating
        registered_merged_computation_function_default_kwargs_dict, code_str, nested_classes_dict, (imports_dict, imports_list, imports_string) = GlobalComputationParametersAttrsClassTemplating.main_generate_params_classes(curr_active_pipeline=curr_active_pipeline)
        """        
        registered_merged_computation_function_default_kwargs_dict = cls.main_extract_params_default_values(curr_active_pipeline=curr_active_pipeline)

        code_str, nested_classes_dict, imports_dict = cls._subfn_build_attrs_parameters_classes(registered_merged_computation_function_default_kwargs_dict=registered_merged_computation_function_default_kwargs_dict, 
                                                                                                                params_defn_save_path=None, should_build_hdf_class=True, print_defns=print_defns,
                                                                                                                additional_bases=["BaseGlobalComputationParameters"])
        imports_list = list(imports_dict.keys())
        imports_string: str = 'import ' + ', '.join(imports_list)
        if print_defns:
            print(imports_string)

        return registered_merged_computation_function_default_kwargs_dict, code_str, nested_classes_dict, (imports_dict, imports_list, imports_string)
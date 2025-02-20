
import inspect
import numpy as np
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters # to replace simple PlacefieldComputationParameters

from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import build_configs # TODO: should be replaced by a better and internal config

# ==================================================================================================================== #
# PIPELINE STAGE (MIXIN HERE?)                                                                                         
# ==================================================================================================================== #
class FilterablePipelineStage:
    """ 
    Adds the self.filtered_sessions, self.filtered_epochs, self.active_configs, self.computation_results properties:
    """    
    
    def select_filters(self, active_session_filter_configurations, clear_filtered_results=True, progress_logger=None):
        """ 
        Clears/Updates:
            self.filtered_sessions, self.filtered_epochs, self.filtered_contexts,
            self.active_configs
            self.computation_results

        """
        if clear_filtered_results:
            # if clear_filtered_results is True, initialize the filtered_* properties. Otherwise just continue with the extant values (they must exist)
            self.filtered_sessions = dict()
            self.filtered_epochs = dict()
            self.filtered_contexts = DynamicParameters()
            self.active_configs = dict() # active_config corresponding to each filtered session/epoch
            self.computation_results = dict()
            
        if progress_logger is not None:
            progress_logger.info(f'select_filters(...) with: {list(active_session_filter_configurations.values())}')
            

        for a_filter_config_name, a_select_config_filter_function in active_session_filter_configurations.items():
            print(f'Applying session filter named "{a_filter_config_name}"...')
            if progress_logger is not None:
                progress_logger.info(f'\tApplying session filter named "{a_filter_config_name}"...')
            self.filtered_sessions[a_filter_config_name], self.filtered_epochs[a_filter_config_name], self.filtered_contexts[a_filter_config_name] = a_select_config_filter_function(self.sess) # `a_select_config_filter_function` call the filter select function
            ## Add the filter to the active context (IdentifyingContext)
            # self.filtered_contexts[a_filter_config_name] = active_identifying_session_ctx.adding_context('filter', filter_name=a_filter_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'

            # build the active filter config from the session's config and the filtered epoch
            self.active_configs[a_filter_config_name] = build_configs(self.filtered_sessions[a_filter_config_name].config, self.filtered_epochs[a_filter_config_name]) 
            self.computation_results[a_filter_config_name] = None # Note that computation config is currently None because computation hasn't been performed yet at this stage.
            self.active_configs[a_filter_config_name].filter_config = {'filter_function': a_select_config_filter_function} # add the makeshift filter config (which is currently just a dictionary)
            



    # Filtered Properties: _______________________________________________________________________________________________ #
    @property
    def is_filtered(self):
        """The is_filtered property."""
        raise NotImplementedError

    # @function_attributes(short_name=None, tags=['filter', 'filtered_sessions'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-20 06:51', related_items=[])
    def filter_sessions(self, active_session_filter_configurations, changed_filters_ignore_list=None, debug_print=False, progress_logger=None):
        """ 
            changed_filters_ignore_list: <list> a list of names of changed filters which will be ignored if they exists
            
            Uses: `self.active_configs, self.logger, self.is_filtered, self.stage
            
            Call Hierarchy:
                - `self.active_configs`
            
            
            Stage's equivalent is:
            
            stage.select_filters(active_session_filter_configurations, progress_logger=self.logger) # select filters when done
            
            # 2023-10-25 - refactored from pipeline's function of the same name.
            Compared to `pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline.NeuropyPipeline.filter_sessions`, the only thing this stage version can't do is upgrade itself (replacing `self`) to a `ComputedPipelineStage` if it isn't already. 
            To get around this, it requires that it's already a `ComputedPipelineStage` by checking its implementor's `is_filtered` property.


            progress_logger=self.logger
        """
        if changed_filters_ignore_list is None:
            changed_filters_ignore_list = []

        assert self.is_filtered
        
        # RESUSE LOADED FILTERING: If the loaded pipeline is already filtered, check to see if the filters match those that were previously applied. If they do, don't re-filter unless the user specifies to.
        prev_session_filter_configurations = {a_config_name:a_config.filter_config['filter_function'] for a_config_name, a_config in self.active_configs.items()}
        # print(f'prev_session_filter_configurations: {prev_session_filter_configurations}')
        # Check for any non-equal ones:
        is_common_filter_name = np.isin(list(active_session_filter_configurations.keys()), list(prev_session_filter_configurations.keys()))
        is_novel_filter_name = np.logical_not(is_common_filter_name)
        if debug_print:
            print(f'is_common_filter_name: {is_common_filter_name}')
            print(f'is_novel_filter_name: {is_novel_filter_name}')
        # novel_filter_names = list(active_session_filter_configurations.keys())[np.logical_not(np.isin(list(active_session_filter_configurations.keys()), list(prev_session_filter_configurations.keys())))]
        # novel_filter_names = [a_name for a_name in list(active_session_filter_configurations.keys()) if a_name not in list(prev_session_filter_configurations.keys())]
        common_filter_names = np.array(list(active_session_filter_configurations.keys()))[is_common_filter_name]
        novel_filter_names = np.array(list(active_session_filter_configurations.keys()))[is_novel_filter_name]
        if debug_print:
            print(f'common_filter_names: {common_filter_names}')
        if len(novel_filter_names) > 0:
            if progress_logger is not None:
                progress_logger.info(f'novel_filter_names: {novel_filter_names}')
            print(f'novel_filter_names: {novel_filter_names}')
        ## Deal with filters with the same name, but different filter functions:
        # changed_filters_names_list = [a_config_name for a_config_name in common_filter_names if (inspect.getsource(prev_session_filter_configurations[a_config_name]) != inspect.getsource(active_session_filter_configurations[a_config_name]))] 
        changed_filters_names_list = [] # changed_filters_names_list: a list of filter names for filters that have changed but have the same name
        for a_config_name in common_filter_names:
            try:
                prev_filter_src = inspect.getsource(prev_session_filter_configurations[a_config_name])
                active_filter_src = inspect.getsource(active_session_filter_configurations[a_config_name])
                if (prev_filter_src != active_filter_src):
                    if debug_print:
                        print(f'prev_filter_src != active_filter_src\nprev_filter_src:')
                        print(prev_filter_src)
                        print(f'active_filter_src:')
                        print(active_filter_src)
                    changed_filters_names_list.append(a_config_name) # if inspect works and there is a difference, add it to the changed list
            except OSError as e:
                # OSError: source code not available
                # if inspect fails for some reason, we should assume a difference to be safe and add it to the changed list
                print(f'WARNING: inspect failed for {a_config_name} with error {e}. Assuming changed.')
                changed_filters_names_list.append(a_config_name)
            except Exception as e:
                raise e
        
        if debug_print:
            print(f'changed_filters_names_list: {changed_filters_names_list}')
        unprocessed_filters = {a_config_name:active_session_filter_configurations[a_config_name] for a_config_name in changed_filters_names_list if a_config_name not in changed_filters_ignore_list}
        ignored_changed_filters_list = [a_config_name for a_config_name in changed_filters_names_list if a_config_name in changed_filters_ignore_list]
        if len(ignored_changed_filters_list) > 0:
            print(f'WARNING: changed_filters_names_list > 0!: {changed_filters_names_list} but these filters are in the changed_filters_ignore_list: {changed_filters_ignore_list}\nignored_changed_filters_list: {ignored_changed_filters_list}')
        # assert len(changed_filters_names_list) == 0, f"WARNING: changed_filters_names_list > 0!: {changed_filters_names_list}"
        # if len(changed_filters_names_list) > 0:
        #     print(f'WARNING: changed_filters_names_list > 0!: {changed_filters_names_list}')
        for a_novel_filter_name in novel_filter_names:
            unprocessed_filters[a_novel_filter_name] = active_session_filter_configurations[a_novel_filter_name]

        ## TODO: filter for the new and changed filters here:
        self.select_filters(unprocessed_filters, clear_filtered_results=False, progress_logger=progress_logger) # select filters when done
    
        # else:
        #     # Not previously filtered. Perform the filtering:
        #     self.stage = ComputedPipelineStage.init_from_previous_stage(self.stage)
        #     self.select_filters(active_session_filter_configurations, progress_logger=with_logger) # select filters when done


       








# ==================================================================================================================== #
# PIPELINE MIXIN                                                                                                       #
# ==================================================================================================================== #
class FilteredPipelineMixin:
    """ To be added to the pipeline to enable conveninece access ot its pipeline stage post Filtered stage. """
    ## Filtered Properties:
    @property
    def filtered_epochs(self):
        """The filtered_sessions property, accessed through the stage."""
        return self.stage.filtered_epochs
        
    @property
    def filtered_sessions(self):
        """The filtered_sessions property, accessed through the stage."""
        return self.stage.filtered_sessions
    
    @property
    def filtered_session_names(self):
        """The names that identify each filtered session in the self.stage.filtered_sessions dictionary. Should be the same as self.active_config_names I believe."""
        return list(self.stage.filtered_sessions.keys())

    @property
    def filtered_contexts(self):
        """ filtered_contexts holds the corresponding contexts for each filtered config."""
        return self.stage.filtered_contexts
    @filtered_contexts.setter
    def filtered_contexts(self, value):
        self.stage.filtered_contexts = value

    @property
    def active_config_names(self):
        """The names of the active configs that can be used to index into the other properties (which are dictionaries)."""
        return list(self.stage.active_configs.keys())

    @property
    def active_configs(self):
        """The active_configs property corresponding to the InteractivePlaceCellConfig obtained by filtering the session. Accessed through the stage."""
        return self.stage.active_configs

            

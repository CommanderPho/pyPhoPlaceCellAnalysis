from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import build_configs # TODO: should be replaced by a better and internal config

# ==================================================================================================================== #
# PIPELINE STAGE (MIXIN HERE?)                                                                                         
# ==================================================================================================================== #
class FilterablePipelineStage:
    """ 
    Adds the self.filtered_sessions, self.filtered_epochs, self.active_configs, self.computation_results properties:
    """    
    
    def select_filters(self, active_session_filter_configurations, clear_filtered_results=True, progress_logger=None):
        if clear_filtered_results:
            # if clear_filtered_results is True, initialize the filtered_* properties. Otherwise just continue with the extant values (they must exist)
            self.filtered_sessions = dict()
            self.filtered_epochs = dict()
            self.active_configs = dict() # active_config corresponding to each filtered session/epoch
            self.computation_results = dict()
            
        if progress_logger is not None:
            progress_logger.info(f'select_filters(...) with: {list(active_session_filter_configurations.values())}')
            
        for a_select_config_name, a_select_config_filter_function in active_session_filter_configurations.items():
            print(f'Applying session filter named "{a_select_config_name}"...')
            if progress_logger is not None:
                progress_logger.info(f'\tApplying session filter named "{a_select_config_name}"...')
            self.filtered_sessions[a_select_config_name], self.filtered_epochs[a_select_config_name] = a_select_config_filter_function(self.sess)
            # build the active filter config from the session's config and the filtered epoch
            self.active_configs[a_select_config_name] = build_configs(self.filtered_sessions[a_select_config_name].config, self.filtered_epochs[a_select_config_name]) 
            self.computation_results[a_select_config_name] = None # Note that computation config is currently None because computation hasn't been performed yet at this stage.
            self.active_configs[a_select_config_name].filter_config = {'filter_function': a_select_config_filter_function} # add the makeshift filter config (which is currently just a dictionary)
            
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
    def active_config_names(self):
        """The names of the active configs that can be used to index into the other properties (which are dictionaries)."""
        return list(self.stage.active_configs.keys())

    @property
    def active_configs(self):
        """The active_configs property corresponding to the InteractivePlaceCellConfig obtained by filtering the session. Accessed through the stage."""
        return self.stage.active_configs

            

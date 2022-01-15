from PhoPositionalData.analysis.interactive_placeCell_config import build_configs # TODO: should be replaced by a better and internal config

class FilterablePipelineStage:
    
    # @property
    # def filtered_data(self):
    #     """The filtered_data property."""
    #     return self._filtered_data
    # @filtered_data.setter
    # def filtered_data(self, value):
    #     self._filtered_data = value
    
    def select_filters(self, active_session_filter_configurations):
        self.filtered_sessions = dict()
        self.filtered_epochs = dict()
        self.active_configs = dict() # active_config corresponding to each filtered session/epoch
        self.computation_results = dict()
        for a_select_config_name, a_select_config_filter_function in active_session_filter_configurations.items():
            print(f'Applying session filter named "{a_select_config_name}"...')
            self.filtered_sessions[a_select_config_name], self.filtered_epochs[a_select_config_name] = a_select_config_filter_function(self.sess)
            # build the active filter config from the session's config and the filtered epoch
            self.active_configs[a_select_config_name] = build_configs(self.filtered_sessions[a_select_config_name].config, self.filtered_epochs[a_select_config_name]) # Note that computation config is currently None because computation hasn't been performed yet at this stage.
            self.computation_results[a_select_config_name] = None
            self.active_configs[a_select_config_name].filter_config = {'filter_function': a_select_config_filter_function} # add the makeshift filter config (which is currently just a dictionary)
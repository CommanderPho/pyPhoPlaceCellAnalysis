
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
        self.computation_results = dict()
        for a_select_config_name, a_select_config_filter_function in active_session_filter_configurations.items():
            print(f'Applying session filter named "{a_select_config_name}"...')
            self.filtered_sessions[a_select_config_name], self.filtered_epochs[a_select_config_name] = a_select_config_filter_function(self.sess)
            self.computation_results[a_select_config_name] = None
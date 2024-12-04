
class PipelineOwningMixin:
    """ Implementors own a pipeline or have access through a parent

    """
    @property
    def owning_pipeline(self):
        """The owning_pipeline property."""
        if hasattr(self, '_owning_pipeline') and (self._owning_pipeline is not None):
            return self._owning_pipeline
        else:
            return None
            # # No direct property, check parent
            # if hasattr(self.parent(), '_owning_pipeline'):
            #     return self.parent()._owning_pipeline
            # else:
            #     return None
            

    @property
    def all_filtered_session_keys(self):
        """Gets the names of the filters applied and updates the config rows with them."""
        if self.owning_pipeline is None:
            return []
        return list(self.owning_pipeline.filtered_sessions.keys())

    @property
    def all_filtered_session_contexts(self):
        """Gets the names of the filters applied and updates the config rows with them."""
        if self.owning_pipeline is None:
            return []
        return self.owning_pipeline.filtered_contexts

    @property
    def all_filtered_session_context_descriptions(self):
        """Gets the names of the filters applied and updates the config rows with them."""
        if self.owning_pipeline is None:
            return []
        return [a_context.get_description() for a_context in self.owning_pipeline.filtered_contexts.values()]



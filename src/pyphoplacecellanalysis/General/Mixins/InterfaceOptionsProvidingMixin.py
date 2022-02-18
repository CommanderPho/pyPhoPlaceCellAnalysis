

class InterfaceOptionsProvidingMixin:
    @property
    def epoch_labels(self):
        """The epoch_labels property."""
        return list(self.sess.epochs.labels) # ['pre', 'maze1', 'post1', 'maze2', 'post2']

    @property
    def epoch_named_timeranges(self):
        """The epoch_labels property.
  
        [<NamedTimerange: {'name': 'pre', 'start_end_times': array([    0, 11066], dtype=int64)};>,
        <NamedTimerange: {'name': 'maze1', 'start_end_times': array([11070, 13970], dtype=int64)};>,
        <NamedTimerange: {'name': 'post1', 'start_end_times': array([13972, 20754], dtype=int64)};>,
        <NamedTimerange: {'name': 'maze2', 'start_end_times': array([20756, 24004], dtype=int64)};>,
        <NamedTimerange: {'name': 'post2', 'start_end_times': array([24006, 42305], dtype=int64)};>]
        
      """
        return [self.sess.epochs.get_named_timerange(a_label) for a_label in self.epoch_labels]


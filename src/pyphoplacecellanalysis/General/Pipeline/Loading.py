from pathlib import Path


class LoadableInput:
    def _check(self):
        assert (self.load_function is not None), "self.load_function must be a valid single-argument load function that isn't None!"
        assert callable(self.load_function), "self.load_function must be callable!"
        assert self.basedir is not None, "self.basedir must not be None!"
        assert isinstance(self.basedir, Path), "self.basedir must be a pathlib.Path type object (or a pathlib.Path subclass)"
        if not self.basedir.exists():
            raise FileExistsError
        else:
            return True

    def load(self):
        self._check()
        self.loaded_data = dict()
        # call the internal load_function with the self.basedir.
        self.loaded_data["sess"] = self.load_function(self.basedir)
        
        pass



class LoadableSessionInput:
    """ Provides session (self.sess) and other helper properties to Stages that load a session """
    @property
    def sess(self):
        """The sess property."""
        return self.loaded_data["sess"]

    @sess.setter
    def sess(self, value):
        self.loaded_data["sess"] = value

    @property
    def active_sess_config(self):
        """The active_sess_config property."""
        return self.sess.config

    @active_sess_config.setter
    def active_sess_config(self, value):
        self.sess.config = value

    @property
    def session_name(self):
        """The session_name property."""
        return self.sess.name

    @session_name.setter
    def session_name(self, value):
        self.sess.name = value

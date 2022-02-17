import inspect # for getting the member function names


class AllFunctionEnumeratingMixin:
    """Implementors can enumerate their functions. """

    @classmethod
    def get_all_functions(cls):
        return inspect.getmembers(cls, predicate=inspect.isfunction) # return the list of tuples for each function. The first element contains the function name, and the second element contains the function itself.
    
    
    
    @classmethod
    def get_all_function_names(cls):
        all_fcn_tuples = list(inspect.getmembers(cls, predicate=inspect.isfunction))
        # all_fcn_tuples = cls.get_all_functions()
        return [a_name for (a_name, a_fn) in all_fcn_tuples] # returns the list of names
        
    # @property
    # def all_functions(self):
    #     """The all_functions property."""
    #     return self._all_functions
    
    

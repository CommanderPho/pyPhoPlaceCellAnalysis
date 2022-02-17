import inspect # for getting the member function names


class AllFunctionEnumeratingMixin:
    """Implementors can enumerate their functions. """

    @classmethod
    def get_all_functions(cls, use_definition_order=True):
        # return inspect.getmembers(cls, predicate=inspect.isfunction) # return the list of tuples for each function. The first element contains the function name, and the second element contains the function itself.
        
        # use_definition_order: if True, the tuples are returned in the order they are defined in the file. Otherwise, in alphabetical order.
    
         # all_function_tuples is a list of tuples for each function. The first element contains the function name, and the second element contains the function itself.
        all_function_tuples = inspect.getmembers(cls, predicate=inspect.isfunction)
        
        if use_definition_order:
            # members = []
            member_line_numbers = []
            for name, obj in all_function_tuples:
                source, start_line = inspect.getsourcelines(obj) # get the line number for the function
                # members.append([name, obj, start_line])
                member_line_numbers.append(start_line)

            # def _line_order(value):
            #     return value[2]

            # returns the ordered result
            # return members.sort(key = _line_order)
            return [x for _, x in sorted(zip(member_line_numbers, all_function_tuples))] # sort the returned function tuples by line number
        else:
            return all_function_tuples
        
    
    @classmethod
    def get_all_function_names(cls, use_definition_order=True):
        # all_fcn_tuples = list(inspect.getmembers(cls, predicate=inspect.isfunction))
        all_fcn_tuples = list(cls.get_all_functions(use_definition_order=use_definition_order))
        return [a_name for (a_name, a_fn) in all_fcn_tuples] # returns the list of names
        
    # @property
    # def all_functions(self):
    #     """The all_functions property."""
    #     return self._all_functions
    
    

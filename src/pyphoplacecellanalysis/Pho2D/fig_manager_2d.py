from traitlets import SingletonConfigurable, HasTraits, Int, Unicode, Enum

class FigureManager2D(SingletonConfigurable):
	""" A singleton manager for 2D figures.  """
    

    autocall = Enum((0,1,2), default_value=0, help=
        """
        Make IPython automatically call any callable object even if you didn't
        type explicit parentheses. For example, 'str 43' becomes 'str(43)'
        automatically. The value can be '0' to disable the feature, '1' for
        'smart' autocall, where it is not applied if there are no more
        arguments on the line, and '2' for 'full' autocall, where all callable
        objects are automatically called (even if no arguments are present).
        """
    ).tag(config=True)
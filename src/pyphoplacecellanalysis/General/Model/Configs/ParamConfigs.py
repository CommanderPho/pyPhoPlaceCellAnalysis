import param
import numpy as np
import pandas as pd
from qtpy import QtGui # for QColor

## Parameters (Param):
class BaseDisplayStateParams(param.Parameterized):
    # name = param.Parameter(default="Not editable", constant=True)
    name = param.String(default='name', doc='The name of the placefield')
    isVisible = param.Boolean(default=False, doc="Whether the plot is visible")


class BasePlotDataParams(param.Parameterized):
    # name = param.Parameter(default="Not editable", constant=True)
    name = param.String(default='name', doc='The name of the placefield')
    # name = param.Parameter(default='name', doc='The name of the placefield')
    isVisible = param.Boolean(default=False, doc="Whether the plot is visible")


class ExtendedPlotDataParams(BasePlotDataParams):
    color = param.Color(default='#FF0000', doc="The placefield's Color")
    extended_values_dictionary = param.Dict(default={}, doc="Extra values stored in a dictionary.")


class ExampleExtended(BasePlotDataParams):
    color                   = param.Color(default='#BBBBBB')
    dictionary              = param.Dict(default={"a": 2, "b": 9})
    select_string           = param.ObjectSelector(default="yellow", objects=["red", "yellow", "green"])
    select_fn               = param.ObjectSelector(default=list,objects=[list, set, dict])
    int_list                = param.ListSelector(default=[3, 5], objects=[1, 3, 5, 7, 9], precedence=0.5)



# checkbutton_group = pn.widgets.CheckButtonGroup(name='Check Button Group', value=[], options=pf_options_list_strings) # checkbutton_group.value
# cross_selector = pn.widgets.CrossSelector(name='Active Placefields', value=[], options=pf_options_list_strings) # cross_selector.value

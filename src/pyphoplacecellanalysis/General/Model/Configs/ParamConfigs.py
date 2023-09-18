import param
import numpy as np
import pandas as pd
from qtpy import QtGui # for QColor


## Parameters (Param):
class BaseDisplayStateParams(param.Parameterized):
    # name = param.Parameter(default="Not editable", constant=True)
    name = param.String(default='name', doc='The name of the placefield')
    isVisible = param.Boolean(default=False, doc="Whether the plot is visible")



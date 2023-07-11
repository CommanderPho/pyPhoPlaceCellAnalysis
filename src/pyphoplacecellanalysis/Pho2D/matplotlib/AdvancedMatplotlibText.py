from typing import Any, Callable, List
from attrs import define, field, Factory
import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from flexitext import flexitext ## flexitext version


@define()
class FormattedFigureText:
    """ builds flexitext matplotlib figure title and footers 

    Consistent color scheme:
        Long: Red
        Short: Blue

        Context footer is along the bottom of the figure in gray.


    Usage:
        from pyphoplacecellanalysis.Pho2D.matplotlib.AdvancedMatplotlibText import FormattedFigureText

        # `flexitext` version:
        text_formatter = FormattedFigureText()
        plt.title('')
        plt.suptitle('')
        text_formatter.setup_margins(fig)

        ## Need to extract the track name ('maze1') for the title in this plot. 
        track_name = active_context.get_description(subset_includelist=['filter_name'], separator=' | ') # 'maze1'
        # TODO: do we want to convert this into "long" or "short"?
        flexitext(text_formatter.left_margin, text_formatter.top_margin, f'<size:22><weight:bold>{track_name}</> replay|laps <weight:bold>firing rate</></>', va="bottom", xycoords="figure fraction")
        footer_text_obj = flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")



    """
    # fig.subplots_adjust(top=top_margin, left=left_margin, bottom=bottom_margin)
    top_margin: float = 0.8
    # left_margin: float = 0.090
    # right_margin: float = 0.91 # (1.0-0.090)
    left_margin: float = 0.15
    right_margin: float = 0.85 # (1.0-0.15)
    bottom_margin: float = 0.150

    

    @classmethod
    def _build_formatted_title_string(cls, epochs_name) -> str:
        """ buidls the two line colored string figure's footer that is passed into `flexitext`.
        """
        return (f"<size:22><weight:bold>{epochs_name}</> Firing Rates\n"
                "<size:14>for the "
                "<color:crimson, weight:bold>Long</>/<color:royalblue, weight:bold>Short</> eXclusive Cells on each track</></>"
                )


    @classmethod
    def _build_footer_string(cls, active_context) -> str:
        """ buidls the dim, grey string for the figure's footer that is passed into `flexitext`.
        Usage:
            footer_text_obj = flexitext((left_margin*0.1), (bottom_margin*0.25), cls._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")
        """
        first_portion_sess_ctxt_str = active_context.get_description(subset_includelist=['format_name', 'animal', 'exper_name'], separator=' | ')
        session_name_sess_ctxt_str = active_context.get_description(subset_includelist=['session_name'], separator=' | ') # 2006-6-08_14-26-15
        return (f"<color:silver, size:10>{first_portion_sess_ctxt_str} | <weight:bold>{session_name_sess_ctxt_str}</></>")


    def setup_margins(self, fig, **kwargs):
        top_margin, left_margin, right_margin, bottom_margin = kwargs.get('top_margin', self.top_margin), kwargs.get('left_margin', self.left_margin), kwargs.get('right_margin', self.right_margin), kwargs.get('bottom_margin', self.bottom_margin)
        fig.subplots_adjust(top=top_margin, left=left_margin, right=right_margin, bottom=bottom_margin) # perform the adjustment on the figure

    def add_flexitext_context_footer(self, active_context):
        """ adds the default footer  """
        return flexitext((self.left_margin*0.1), (self.bottom_margin*0.25), self._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")



    def add_flexitext(self, fig, **kwargs):
        self.setup_margins(fig, **kwargs)
        # Add flexitext
        top_margin, left_margin, bottom_margin = kwargs.get('top_margin', self.top_margin), kwargs.get('left_margin', self.left_margin), kwargs.get('bottom_margin', self.bottom_margin)
        title_text_obj = flexitext(left_margin, top_margin, 'long ($L$)|short($S$) firing rate indicies', va="bottom", xycoords="figure fraction")
        footer_text_obj = flexitext((self.left_margin*0.1), (self.bottom_margin*0.25), self._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")
        return title_text_obj, footer_text_obj




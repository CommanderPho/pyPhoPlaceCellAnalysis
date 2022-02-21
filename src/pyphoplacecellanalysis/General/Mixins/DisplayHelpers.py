import matplotlib.pyplot as plt

from pyphocorehelpers.print_helpers import WrappingMessagePrinter
from pyphocorehelpers.plotting.mixins.figure_param_text_box import add_figure_text_box # for _display_add_computation_param_text_box


# Used by _display_2d_placefield_result_plot_ratemaps_2D
def _save_displayed_figure_if_needed(plotting_config, plot_type_name='plot', active_variant_name=None, active_figures=list(), debug_print=False):
    if active_variant_name is not None:
        active_plot_filename = '-'.join([plot_type_name, active_variant_name])
    else:
        active_plot_filename = plot_type_name
    active_plot_filepath = plotting_config.get_figure_save_path(active_plot_filename).with_suffix('.png')
    if debug_print:
        print(f'active_plot_filepath: {active_plot_filepath}')
    with WrappingMessagePrinter('Saving 2D Placefield image out to "{}"...'.format(active_plot_filepath), begin_line_ending='...', finished_message='done.'):
        for aFig in active_figures:
            aFig.savefig(active_plot_filepath)
    
    
# Post plotting figure helpers:
def _display_add_computation_param_text_box(fig, computation_config):
    """ Adds a small box containing the computation parmaters to the matplotlib figure. 
    Usage:
        _display_add_computation_param_text_box(plt.gcf(), active_session_computation_config)
    """
    if fig is None:
        fig = plt.gcf()
    render_text = computation_config.str_for_attributes_list_display(key_val_sep_char=':')
    return add_figure_text_box(fig, render_text=render_text)
# used by _display_2d_placefield_result_plot_ratemaps_2D




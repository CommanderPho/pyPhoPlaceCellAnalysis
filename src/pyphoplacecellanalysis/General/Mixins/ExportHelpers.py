from copy import deepcopy
from datetime import datetime
from enum import Enum # for getting the current date to set the ouptut folder name
from pathlib import Path
import pandas as pd
import numpy as np

from neuropy.utils.dynamic_container import overriding_dict_with # required for programmatic_display_to_PDF

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

# ==================================================================================================================== #
# FIGURE/GRAPHICS EXPORT                                                                                               #
# ==================================================================================================================== #

import pyphoplacecellanalysis.External.pyqtgraph as pg
# import pyphoplacecellanalysis.External.pyqtgraph.exporters
# import pyphoplacecellanalysis.External.pyqtgraph.widgets.GraphicsLayoutWidget
from pyphoplacecellanalysis.External.pyqtgraph.widgets.GraphicsView import GraphicsView

# ==================================================================================================================== #
# GRAPHICS/FIGURES EXPORTING                                                                                           #
# ==================================================================================================================== #

class ExportFiletype(Enum):
    """Used by `export_pyqtgraph_plot(.)` to specify the filetype of the export to do"""
    PNG = '.png'
    SVG =  '.svg'

def export_pyqtgraph_plot(graphics_item, savepath='fileName.png', progress_print=True, **kwargs):
    """Takes a PlotItem, A GraphicsLayoutWidget, or other pyqtgraph item to be exported.

    Uses the extension of the `savepath` to determine which type of Exporter to use (png, SVG, etc.)

    Args:
        graphics_item (_type_): _description_
        savepath (str, optional): _description_. Defaults to 'fileName.png'.

    Usage:
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers.export_pyqtgraph_plot

        main_graphics_layout_widget = active_2d_plot.ui.main_graphics_layout_widget # GraphicsLayoutWidget
        main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
        background_static_scroll_plot_widget = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem
        # Export:
        export_pyqtgraph_plot(main_graphics_layout_widget, savepath='main_graphics_layout_widget.png') # works
        export_pyqtgraph_plot(main_plot_widget, savepath='main_plot_widget.png') # works
        export_pyqtgraph_plot(background_static_scroll_plot_widget, savepath='background_static_scroll_plot_widget_HUGE.png') # works

        export_pyqtgraph_plot(background_static_scroll_plot_widget, savepath='background_static_scroll_plot_widget_VECTOR.svg') # works

    """
    if not isinstance(savepath, Path):
        savepath = Path(savepath).resolve() # convert to a path

    if isinstance(graphics_item, (GraphicsView, pg.widgets.GraphicsLayoutWidget.GraphicsLayoutWidget)):
        ## To export the overall layout of a GraphicsLayoutWidget grl, the exporter initialization is:
        graphics_item = graphics_item.scene()

    # Get the extension from the path to determine the filetype:
    file_extensions = savepath.suffixes
    assert len(file_extensions)>0, f"savepath {savepath} must have a recognizable file extension"
    file_extension = file_extensions[-1].lower() # the last is the suffix
    
    ## create an exporter instance, as an argument give it the item you wish to export
    if file_extension == ExportFiletype.PNG.value:
        exporter = pg.exporters.ImageExporter(graphics_item)
        kwargs = ({'width': 4096} | kwargs) # add 'width' to kwargs if not specified
    elif file_extension == ExportFiletype.SVG.value:
        exporter = pg.exporters.SVGExporter(graphics_item)
    else:
        print(f'Unknown file_extension: {file_extension}')
        raise NotImplementedError

    ## set export parameters if needed
    for k, v in kwargs.items():
        # exporter.parameters()['width'] = 4096*4   # (note this also affects height parameter)   
        exporter.parameters()[k] = v
    ## save to file
    exporter.export(str(savepath))
    if progress_print:
        print(f'exported plot to {savepath}')




# ==================================================================================================================== #
# PDF Output                                                                                                           #
# ==================================================================================================================== #

def build_pdf_export_metadata(session_descriptor_string, filter_name, out_path=None, debug_print=False):
    """ OLD - Pre 2022-10-04 - Builds the PDF metadata generating function from the passed info
    
        session_descriptor_string: a string describing the context of the session like 'sess_kdiba_2006-6-07_11-26-53'
            Can be obtained from pipleine via `curr_active_pipeline.sess.get_description()`
        filter_name: a name like 'maze1'
        out_path: an optional Path to use instead of generating a new one
        
    Returns:
        a function that takes one argument, the display function name, and returns the PDF metadata
        
    History:
        Refactored from PhoPy3DPositionAnalysis2021.PendingNotebookCode._build_programmatic_display_function_testing_pdf_metadata on 2022-08-17
        
    Usage:
        session_descriptor_string = curr_active_pipeline.sess.get_description()
        ## PDF Output, NOTE this is single plot stuff: uses active_config_name
        from matplotlib.backends import backend_pdf, backend_pgf, backend_ps
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_pdf_export_metadata

        filter_name = active_config_name
        _build_pdf_pages_output_info, out_parent_path = build_pdf_export_metadata(session_descriptor_string, filter_name=active_config_name, out_path=None)
        _build_pdf_pages_output_info, programmatic_display_function_testing_output_parent_path = build_pdf_export_metadata(curr_active_pipeline.sess.get_description(), filter_name=filter_name)
        print(f'Figure Output path: {str(programmatic_display_function_testing_output_parent_path)}')

        
        curr_display_function_name = '_display_1d_placefield_validations'
        built_pdf_metadata, curr_pdf_save_path = _build_pdf_pages_output_info(curr_display_function_name)
        with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=built_pdf_metadata) as pdf:
            # plt.ioff() # disable displaying the plots inline in the Jupyter-lab notebook. NOTE: does not work in Jupyter-Lab, figures still show
            plots = curr_active_pipeline.display(curr_display_function_name, active_config_name) # works, but generates a TON of plots!
            # plt.ion()
            for fig_idx, a_fig in enumerate(plots):
                # print(f'saving fig: {fig_idx+1}/{len(plots)}')
                pdf.savefig(a_fig)
                # pdf.savefig(a_fig, transparent=True)
            # When no figure is specified the current figure is saved
            # pdf.savefig()

        
    """
    if out_path is None:   
        out_day_date_folder_name = datetime.today().strftime('%Y-%m-%d') # A string with the day's date like '2022-01-16'
        out_path = Path(r'EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting').joinpath(out_day_date_folder_name).resolve()
    else:
        out_path = Path(out_path) # make sure it's a Path
    out_path.mkdir(exist_ok=True)

    
    pho_pdf_metadata = {'Creator': 'Spike3D - TestNeuroPyPipeline116', 'Author': 'Pho Hale', 'Title': session_descriptor_string, 'Subject': '', 'Keywords': [session_descriptor_string]}
    if debug_print:
        print(f'filter_name: {filter_name}')

    def _build_pdf_pages_output_info(display_function_name):
        """ 
        Implicitly captures:
            programmatic_display_fcn_out_path
            session_descriptor_string
            pho_pdf_metadata
            filter_name
        """
        built_pdf_metadata = pho_pdf_metadata.copy()
        context_tuple = [session_descriptor_string, filter_name, display_function_name]
        built_pdf_metadata['Title'] = '_'.join(context_tuple)
        built_pdf_metadata['Subject'] = display_function_name
        built_pdf_metadata['Keywords'] = ' | '.join(context_tuple)
        curr_pdf_save_path = out_path.joinpath(('_'.join(context_tuple) + '.pdf'))
        return built_pdf_metadata, curr_pdf_save_path
    
    return _build_pdf_pages_output_info, out_path


# ==================================================================================================================== #
# Modern 2022-10-04 PDF                                                                                                #
# from pyphoplacecellanalysis.General.Mixins.ExportHelpers import create_daily_programmatic_display_function_testing_folder_if_needed, build_pdf_metadata_from_display_context
# ==================================================================================================================== #
def create_daily_programmatic_display_function_testing_folder_if_needed(out_path=None):
    if out_path is None:   
        out_day_date_folder_name = datetime.today().strftime('%Y-%m-%d') # A string with the day's date like '2022-01-16'
        out_path = Path(r'EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting').joinpath(out_day_date_folder_name).resolve()
    else:
        out_path = Path(out_path) # make sure it's a Path
    out_path.mkdir(exist_ok=True)
    return out_path

def session_context_to_relative_path(parent_path, session_ctx):
    """_summary_

    Args:
        parent_path (Path): _description_
        session_ctx (IdentifyingContext): _description_

    Returns:
        _type_: _description_

    Usage:

        curr_sess_ctx = local_session_contexts_list[0]
        # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
        figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
        session_context_to_relative_path(figures_parent_out_path, curr_sess_ctx)

    """
    parent_path = Path(parent_path)
    subset_whitelist=['format_name','animal','exper_name', 'session_name']
    all_keys_found, found_keys, missing_keys = session_ctx.check_keys(subset_whitelist, debug_print=False)
    if not all_keys_found:
        print(f'WARNING: missing {len(missing_keys)} keys from context: {missing_keys}. Building path anyway.')
    curr_sess_ctx_tuple = session_ctx.as_tuple(subset_whitelist=subset_whitelist, drop_missing=True) # ('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')
    return parent_path.joinpath(*curr_sess_ctx_tuple).resolve()



def build_figure_basename_from_display_context(active_identifying_ctx, subset_whitelist=None, subset_blacklist=None, context_tuple_join_character='_', debug_print=False):
    """ 
    Usage:
        curr_fig_save_basename = build_figure_basename_from_display_context(active_identifying_ctx, context_tuple_join_character='_')
        >>> 'kdiba_2006-6-09_1-22-43_batch_plot_test_long_only'
    """
    ## Note that active_identifying_ctx.as_tuple() can have non-string elements (e.g. debug_test_max_num_slices=128, which is an int). This is what we want, but for setting the metadata we need to convert them to strings
    context_tuple = [str(v) for v in list(active_identifying_ctx.as_tuple(subset_whitelist=subset_whitelist, subset_blacklist=subset_blacklist, drop_missing=True))]
    fig_save_basename = context_tuple_join_character.join(context_tuple) # joins the elements of the context_tuple with '_'
    if debug_print:
        print(f'fig_save_basename: "{fig_save_basename}"')
    return fig_save_basename


def build_pdf_metadata_from_display_context(active_identifying_ctx, subset_whitelist=None, subset_blacklist=None, debug_print=False):
    """ Internally uses `build_figure_basename_from_display_context(...)` 
    Usage:
        curr_built_pdf_metadata, curr_pdf_save_filename = build_pdf_metadata_from_display_context(active_identifying_ctx)

    """
    session_descriptor_string = '_'.join([active_identifying_ctx.format_name, active_identifying_ctx.session_name]) # 'kdiba_2006-6-08_14-26-15'
    if debug_print:
        print(f'session_descriptor_string: "{session_descriptor_string}"')
    built_pdf_metadata = {'Creator': 'Spike3D - TestNeuroPyPipeline227', 'Author': 'Pho Hale', 'Title': session_descriptor_string, 'Subject': '', 'Keywords': [session_descriptor_string]}
    # context_tuple = [str(v) for v in list(active_identifying_ctx.as_tuple())]
    curr_fig_save_basename = build_figure_basename_from_display_context(active_identifying_ctx, subset_whitelist=subset_whitelist, subset_blacklist=subset_blacklist, context_tuple_join_character='_')
    built_pdf_metadata['Title'] = curr_fig_save_basename
    built_pdf_metadata['Subject'] = active_identifying_ctx.display_fn_name
    built_pdf_metadata['Keywords'] = build_figure_basename_from_display_context(active_identifying_ctx, context_tuple_join_character=' | ') # ' | '.join(context_tuple)
    curr_pdf_save_filename = curr_fig_save_basename + '.pdf'
    if debug_print:
        print(f'curr_pdf_save_filename: "{curr_pdf_save_filename}"')
    return built_pdf_metadata, curr_pdf_save_filename


import matplotlib.pyplot as plt
## PDF Output, NOTE this is single plot stuff: uses active_config_name
from matplotlib.backends import backend_pdf # Needed for
# from pyphoplacecellanalysis.General.Mixins.ExportHelpers import create_daily_programmatic_display_function_testing_folder_if_needed, build_pdf_metadata_from_display_context
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # required for programmatic_display_to_PDF


## 2022-10-04 Modern Programmatic PDF outputs:
def programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_plot_decoded_epoch_slices', subset_whitelist=None, subset_blacklist=None,  debug_print=False, **kwargs):
    """
    2022-10-04 Modern Programmatic PDF outputs
    curr_display_function_name = '_display_plot_decoded_epoch_slices' 

    Looks it this is done for EACH filtered context (in the loop below) whereas the original just did a single specific context
    """

    ## Get the output path (active_session_figures_out_path) for this session (and all of its filtered_contexts as well):
    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
    figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
    active_session_figures_out_path = session_context_to_relative_path(figures_parent_out_path, active_identifying_session_ctx)
    if debug_print:
        print(f'curr_session_parent_out_path: {active_session_figures_out_path}')
    active_session_figures_out_path.mkdir(parents=True, exist_ok=True) # make folder if needed

    with plt.ioff():
        ## Disables showing the figure by default from within the context manager.
        # active_display_fn_kwargs = overriding_dict_with(lhs_dict=dict(filter_epochs='ripple', debug_test_max_num_slices=128), **kwargs)
        active_display_fn_kwargs = overriding_dict_with(lhs_dict=dict(), **kwargs) # active_display_fn_kwargs = overriding_dict_with(lhs_dict=dict(filter_epochs='ripple', debug_test_max_num_slices=128), **kwargs)

        # Perform for each filtered context:
        for filter_name, a_filtered_context in curr_active_pipeline.filtered_contexts.items():
            if debug_print:
                print(f'filter_name: {filter_name}: "{a_filtered_context.get_description()}"')
            # Get the desired display function context:
            active_identifying_display_ctx = a_filtered_context.adding_context('display_fn', display_fn_name=curr_display_function_name)
            # final_context = active_identifying_display_ctx # Display only context    

            # # Add in the desired display variable:
            active_identifying_ctx = active_identifying_display_ctx.adding_context('filter_epochs', **active_display_fn_kwargs) # , filter_epochs='ripple' ## TODO: this is only right for a single function!
            final_context = active_identifying_ctx # Display/Variable context mode

            active_identifying_ctx_string = final_context.get_description(separator='|') # Get final discription string
            if debug_print:
                print(f'active_identifying_ctx_string: "{active_identifying_ctx_string}"')

            ## Build PDF Output Info
            active_pdf_metadata, active_pdf_save_filename = build_pdf_metadata_from_display_context(final_context, subset_whitelist=subset_whitelist, subset_blacklist=subset_blacklist)
            active_pdf_save_path = active_session_figures_out_path.joinpath(active_pdf_save_filename) # build the final output pdf path from the pdf_parent_out_path (which is the daily folder)

            ## BEGIN DISPLAY/SAVE
            with backend_pdf.PdfPages(active_pdf_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
                out_fig_list = [] # Separate PDFs mode:

                if debug_print:
                    print(f'active_pdf_save_path: {active_pdf_save_path}\nactive_pdf_metadata: {active_pdf_metadata}')
                    print(f'active_display_fn_kwargs: {active_display_fn_kwargs}')
                out_display_var = curr_active_pipeline.display(curr_display_function_name, a_filtered_context, **active_display_fn_kwargs) # , filter_epochs='ripple', debug_test_max_num_slices=128
                # , fignum=active_identifying_ctx_string, **figure_format_config
    
                if debug_print:
                    print(f'completed display(...) call. type(out_display_var): {type(out_display_var)}\n out_display_var: {out_display_var}, active_display_fn_kwargs: {active_display_fn_kwargs}')

                if isinstance(out_display_var, dict):
                    main_out_display_context = list(out_display_var.keys())[0]
                    if debug_print:
                        print(f'main_out_display_context: "{main_out_display_context}"')
                    main_out_display_dict = out_display_var[main_out_display_context]
                    ui = main_out_display_dict['ui']
                    # out_plot_tuple = curr_active_pipeline.display(curr_display_function_name, filter_name, filter_epochs='ripple', fignum=active_identifying_ctx_string, **figure_format_config)
                    # params, plots_data, plots, ui = out_plot_tuple 
                    out_fig = ui.mw.getFigure() # TODO: Only works for MatplotlibWidget wrapped figures
                    out_fig_list.append(out_fig)
                elif isinstance(out_display_var, MatplotlibRenderPlots):
                    # Newest style plots: 2022-12-09
                    out_fig_list.extend(out_display_var.figures)

                else:
                    # Non-dictionary type item, older style:
                    if not isinstance(out_display_var, (list, tuple)):
                        # not a list, just a scalar object
                        plots = [out_display_var] # make a single-element list
                    else:
                        # it is a list
                        if len(out_display_var) == 2:
                            fig0, figList1 = out_display_var # unpack
                            plots = [fig0, *figList1]
                        else:
                            # otherwise just try and set the plots to the list
                            plots = out_display_var

                    out_fig_list.extend(plots)


                if debug_print:
                    print(f'out_fig_list: {out_fig_list}')

                # Finally iterate through and do the saving to PDF
                for i, a_fig in enumerate(out_fig_list):
                    pdf.savefig(a_fig, transparent=True)
                    pdf.attach_note(f'Page {i + 1}: "{active_identifying_ctx_string}"')



# ==================================================================================================================== #
# Output PDF Merging/Manipulation                                                                                      #
# ==================================================================================================================== #


def merge_output_pdfs(out_file_path='merged-pdf.pdf', *input_files):
    """ merges the input PDF files into a single output 
    Requires: PyPDF2
    """
    from PyPDF2 import PdfMerger
    merger = PdfMerger()
    for pdf in input_files: # ["file1.pdf", "file2.pdf", "file3.pdf"]
        merger.append(pdf)
    merger.write(out_file_path)
    merger.close()




# ==================================================================================================================== #
# Potentially obsolite PDF wrapper method                                                                              #

# ==================================================================================================================== #


from pyphocorehelpers.plotting.figure_management import PhoActiveFigureManager2D, capture_new_figures_decorator
fig_man = PhoActiveFigureManager2D(name=f'fig_man') # Initialize a new figure manager
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper


# ==================================================================================================================== #
# DATA EXPORT                                                                                                          #
# ==================================================================================================================== #

def get_default_pipeline_data_keys(active_config_name):
    if active_config_name is None:
        active_config_name = 'sess' # the default keys for no filter config are '/sess/spikes_df'
        
    return {'spikes_df': f'/filtered_sessions/{active_config_name}/spikes_df',
            'positions_df': f'/filtered_sessions/{active_config_name}/pos_df'
        }
        
def save_some_pipeline_data_to_h5(active_pipeline, included_session_identifiers=None, custom_key_prefix=None, finalized_output_cache_file='./pipeline_cache_store.h5', debug_print=False):
    """ 
    
    Inputs:
        included_session_identifiers: [] -  a list of session names to include in the output (e.g. ['maze','maze1','maze2']
       finalized_output_cache_file: str - a string specifying the prefix to prepend to each h5 key created, or None to use the default
        
    # Saves out ['/spikes_df', '/sess/spikes_df', '/filtered_sessions/maze2/spikes_df', '/filtered_sessions/maze1/spikes_df', '/filtered_sessions/maze/spikes_df'] to a .h5 file which can be loaded with
    # with pd.HDFStore(finalized_spike_df_cache_file) as store:
        # print(store.keys())
        # reread = pd.read_hdf(finalized_spike_df_cache_file, key='spikes_df')
        # reread
    Usage:
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import _test_save_pipeline_data_to_h5, get_h5_data_keys, save_some_pipeline_data_to_h5, load_pipeline_data_from_h5  #ExportHelpers
        finalized_output_cache_file='data/pipeline_cache_store.h5'
        output_save_result = save_some_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file=finalized_output_cache_file)
        output_save_result
        
        >> DynamicParameters({'finalized_output_cache_file': 'data/pipeline_cache_store.h5', 'sess': {'spikes_df': 'sess/spikes_df', 'pos_df': 'sess/pos_df'}, 'filtered_sessions/maze1': {'spikes_df': 'filtered_sessions/maze1/spikes_df', 'pos_df': 'filtered_sessions/maze1/pos_df'}, 'filtered_sessions/maze2': {'spikes_df': 'filtered_sessions/maze2/spikes_df', 'pos_df': 'filtered_sessions/maze2/pos_df'}, 'filtered_sessions/maze': {'spikes_df': 'filtered_sessions/maze/spikes_df', 'pos_df': 'filtered_sessions/maze/pos_df'}})

    
    Example: Loading Saved Dataframe:
        # Load the saved .h5 spikes dataframe for testing:
        finalized_spike_df_cache_file='./pipeline_cache_store.h5'
        desired_spikes_df_key = '/filtered_sessions/maze1/spikes_df'
        spikes_df = pd.read_hdf(finalized_spike_df_cache_file, key=desired_spikes_df_key)
        spikes_df
    """
    def _perform_save_cache_pipeline_data_to_h5(spikes_df, pos_df, sess_identifier_key='sess', finalized_output_cache_file='./pipeline_cache_store.h5'):
        """ 
            sess_identifier_key: str: like 'sess' or 'filtered_sessions/maze1'
        
        """
        # local_output_structure = output_structure.setdefault(sess_identifier_key, {})
        local_output_keys = get_default_pipeline_data_keys(sess_identifier_key)
        spikes_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/spikes_df')
        pos_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/pos_df', format='table')
        return sess_identifier_key, local_output_keys

    output_structure = DynamicParameters(finalized_output_cache_file=finalized_output_cache_file)
    
    if included_session_identifiers is None:
        included_session_identifiers = ['sess'] + active_pipeline.filtered_session_names
                            
        
    # Save out the non-filtered (sess) if desired:
    if 'sess' in included_session_identifiers:
        if custom_key_prefix is not None:
            curr_sess_identifier_key = '/'.join([custom_key_prefix, 'sess'])
        else:
            curr_sess_identifier_key = 'sess'
        local_sess_identifier_key, local_output_keys = _perform_save_cache_pipeline_data_to_h5(active_pipeline.sess.spikes_df, active_pipeline.sess.position.to_dataframe(), sess_identifier_key=curr_sess_identifier_key, finalized_output_cache_file=finalized_output_cache_file)
        output_structure[local_sess_identifier_key] = local_output_keys
    else:
        if debug_print:
            print("skipping 'sess' because it is not included in included_session_identifiers")
    
    for (a_key, a_filtered_session) in active_pipeline.filtered_sessions.items():
        if a_key in included_session_identifiers:
            if debug_print:
                print(f'a_filtered_session: {a_filtered_session}')
                
            # curr_sess_identifier_key = f'filtered_sessions/{a_key}'
            if custom_key_prefix is not None:
                curr_sess_identifier_key = '/'.join([custom_key_prefix, 'filtered_sessions', a_key])
            else:
                curr_sess_identifier_key = '/'.join(['filtered_sessions', a_key])
            
            local_sess_identifier_key, local_output_keys = _perform_save_cache_pipeline_data_to_h5(a_filtered_session.spikes_df, a_filtered_session.position.to_dataframe(), sess_identifier_key=curr_sess_identifier_key, finalized_output_cache_file=finalized_output_cache_file)
            output_structure[local_sess_identifier_key] = local_output_keys 
        else:
            if debug_print:
                print(f'skipping {a_key} because it is not included in included_session_identifiers')
    return output_structure

def load_pipeline_data_from_h5(finalized_output_cache_file, desired_spikes_df_key, desired_positions_df_key):
    """  Load the saved .h5 spikes dataframe for testing:
    
    Usage:
        desired_spikes_df_key = f'/filtered_sessions/{active_config_name}/spikes_df'
        desired_positions_df_key = f'/filtered_sessions/{active_config_name}/pos_df'    
        spikes_df, pos_df = load_pipeline_data_from_h5(finalized_output_cache_file=finalized_output_cache_file, desired_spikes_df_key=desired_spikes_df_key, desired_positions_df_key=desired_positions_df_key)

        spikes_df
        pos_df    
    """
    # Load the saved .h5 spikes dataframe for testing:
    spikes_df = pd.read_hdf(finalized_output_cache_file, key=desired_spikes_df_key)
    pos_df = pd.read_hdf(finalized_output_cache_file, key=desired_positions_df_key)
    # spikes_df
    # pos_df
    return spikes_df, pos_df

def get_h5_data_keys(finalized_output_cache_file, enable_debug_print=False):
    """ Returns the keys (variables) in the .h5 file
    Usage:
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import _test_save_pipeline_data_to_h5, get_h5_data_keys, save_some_pipeline_data_to_h5, load_pipeline_data_from_h5  #ExportHelpers
        finalized_output_cache_file='data/pipeline_cache_store.h5'
        out_keys = get_h5_data_keys(finalized_output_cache_file=finalized_output_cache_file)
        print(out_keys)
        >>> ['/spikes_df', '/sess/pos_df', '/sess/spikes_df', '/filtered_sessions/maze2/pos_df', '/filtered_sessions/maze2/spikes_df', '/filtered_sessions/maze2/pos_df/meta/values_block_2/meta', '/filtered_sessions/maze2/pos_df/meta/values_block_1/meta', '/filtered_sessions/maze1/pos_df', '/filtered_sessions/maze1/spikes_df', '/filtered_sessions/maze1/pos_df/meta/values_block_2/meta', '/filtered_sessions/maze1/pos_df/meta/values_block_1/meta', '/filtered_sessions/maze/pos_df', '/filtered_sessions/maze/spikes_df', '/filtered_sessions/maze/pos_df/meta/values_block_2/meta', '/filtered_sessions/maze/pos_df/meta/values_block_1/meta']
        
    """
    out_keys = None
    with pd.HDFStore(finalized_output_cache_file) as store:
        out_keys = store.keys()
        if enable_debug_print:
            print(out_keys)
    return out_keys


def _test_save_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file=None, data_output_directory=None, enable_dry_run=True, enable_debug_print=True):
    """ 
    
    Usage:
        finalized_output_cache_file = _test_save_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file=finalized_output_cache_file, enable_dry_run=False, enable_debug_print=True)
        finalized_output_cache_file

    """
    # Define Saving/Loading Directory and paths:
    if finalized_output_cache_file is None:
        if data_output_directory is None:
            data_output_directory = Path('./data')
        finalized_output_cache_file = data_output_directory.joinpath('pipeline_cache_store.h5') # '../../data/pipeline_cache_store.h5'

    if enable_debug_print:
        print(f'finalized_output_cache_file: "{str(finalized_output_cache_file)}"')

    curr_epoch_labels = list(curr_active_pipeline.sess.epochs.labels) # ['pre', 'maze1', 'post1', 'maze2', 'post2']
    curr_named_timeranges = [curr_active_pipeline.sess.epochs.get_named_timerange(a_label) for a_label in curr_epoch_labels]

    if enable_debug_print:
        print(f'curr_named_timeranges: {curr_named_timeranges}')
    

    all_filters_list = list(curr_active_pipeline.filtered_sessions.keys())
    if enable_debug_print:
        print(f'all_filters_list: {all_filters_list}')
        
    active_config_name = 'maze1'
    # active_config_name = 'maze'

    desired_spikes_df_key = f'/filtered_sessions/{active_config_name}/spikes_df'
    desired_positions_df_key = f'/filtered_sessions/{active_config_name}/pos_df'
    # desired_spikes_df_key = f'/filtered_sessions/{active_config_name}/spikes_df'

    if enable_debug_print:
        print(f'desired_spikes_df_key: "{desired_spikes_df_key}"')
        print(f'desired_positions_df_key: "{desired_positions_df_key}"')


    # Get relevant variables:
    # curr_active_pipeline is set above, and usable here
    sess = curr_active_pipeline.filtered_sessions[active_config_name]
    active_computed_data = curr_active_pipeline.computation_results[active_config_name].computed_data
    if enable_debug_print:
        print(f'active_computed_data.keys(): {active_computed_data.keys()}')
    
    pf = curr_active_pipeline.computation_results[active_config_name].computed_data['pf1D']
    active_one_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data['pf2D_Decoder']
    active_two_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None)
    active_measured_positions = curr_active_pipeline.computation_results[active_config_name].sess.position.to_dataframe()

    if not enable_dry_run:
        save_some_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file=finalized_output_cache_file)
    else:
        print(f'dry run only because enable_dry_run == True. No changes will be made.')
        print(f'final command would have been: save_some_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file="{finalized_output_cache_file}")')
    # save_spikes_data_to_h5(curr_active_pipeline, finalized_output_cache_file=finalized_output_cache_file)
    
    return finalized_output_cache_file
from copy import deepcopy
from datetime import datetime
from enum import Enum # for getting the current date to set the ouptut folder name
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from neuropy.core.user_annotations import metadata_attributes
import pandas as pd
import numpy as np
from attrs import define, field, Factory, fields

from neuropy.utils.result_context import IdentifyingContext

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.assertion_helpers import Assert
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path, find_first_extant_path, sanitize_filename_for_Windows
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field


@function_attributes(short_name=None, tags=['filesystem', 'collected_output', 'platform_independent'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-03 08:10', related_items=[])
def try_discover_default_collected_outputs_dir(known_collected_outputs_paths: Optional[List[Path]]=None) -> Optional[Path]:
    """ Tries to find the appropriate "collected_outputs" directory on various known computers.

    Usage:
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import try_discover_default_collected_outputs_dir

        collected_outputs_directory = try_discover_default_collected_outputs_dir()

        assert collected_outputs_directory.exists(), f"collected_outputs_directory: {collected_outputs_directory} does not exist! Is the right computer's config commented out above?"
        # fullwidth_path_widget(scripts_output_path, file_name_label='Scripts Output Path:')
        print(f'collected_outputs_directory: "{collected_outputs_directory}"')
        # Create a 'figures' subfolder if it doesn't exist
        figures_folder: Path = collected_outputs_directory.joinpath('figures', '_temp_individual_posteriors').resolve()
        figures_folder.mkdir(parents=False, exist_ok=True)
        assert figures_folder.exists()
        print(f'\tfigures_folder: "{figures_folder}"')
        ## this is good
        parent_output_folder = figures_folder


    """
    ## if none is provided it tries to find one in collected_outputs
    if known_collected_outputs_paths is None:
        known_collected_outputs_paths = [Path(v).resolve() for v in ['/Users/pho/data/collected_outputs',
                                                                    '/Volumes/SwapSSD/Data/collected_outputs', r"K:/scratch/collected_outputs", '/Users/pho/Dropbox (University of Michigan)/MED-DibaLabDropbox/Data/Pho/Outputs/output/collected_outputs', r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs',
                                                                    '/home/halechr/FastData/collected_outputs/', '/home/halechr/cloud/turbo/Data/Output/collected_outputs']]

    collected_outputs_directory = find_first_extant_path(known_collected_outputs_paths)
    return collected_outputs_directory


## General Output Helpers
@custom_define(slots=False)
class OutputsSpecifier:
    """ outputs_specifier: a class that specifies how to save outputs from a pipeline.

    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import OutputsSpecifier

    """
    output_basepath: Path # path to write out to

    def get_output_path(self) -> Path:
        """ returns the appropriate output path to store the outputs for this session. Usually '$session_folder/outputs/' """
        return self.output_basepath.joinpath('output').resolve()


    # def get_global_computations_output_path(self) -> Path:
    #     """ could be customized to redirect global computations outputs """
    #     return self.output_basepath.joinpath('output').resolve()

    def get_global_computations_output_path(self) -> Path:
        """ could be customized to redirect global computations outputs """
        return self.output_basepath.joinpath('output').resolve()

    # @property
    # def global_computations_basepath(self) -> Path:
    #     return self.output_basepath.joinpath('output').resolve()


# ==================================================================================================================== #
# FIGURE/GRAPHICS EXPORT                                                                                               #
# ==================================================================================================================== #

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.widgets.GraphicsView import GraphicsView


# ==================================================================================================================== #
# GRAPHICS/FIGURES EXPORTING                                                                                           #
# ==================================================================================================================== #

class ExportFiletype(Enum):
    """Used by `export_pyqtgraph_plot(.)` to specify the filetype of the export to do"""
    PNG = '.png'
    SVG =  '.svg'

@function_attributes(tags=['pyqtgraph', 'export', 'graphics', 'plot'])
def export_pyqtgraph_plot(graphics_item, savepath='fileName.png', progress_print=True, **kwargs):
    """Takes a PlotItem, A GraphicsLayoutWidget, or other pyqtgraph item to be exported.

    Uses the extension of the `savepath` to determine which type of Exporter to use (png, SVG, etc.)

    Args:
        graphics_item (_type_): _description_
        savepath (str, optional): _description_. Defaults to 'fileName.png'.

    Usage:
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
        
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
        from pyphoplacecellanalysis.External.pyqtgraph.exporters.ImageExporter import ImageExporter
        exporter = ImageExporter(graphics_item)
        bg = pg.mkColor(0,0,0,0.0) # clear color unless a different one is specified
        kwargs = ({'background': bg} | kwargs) # add 'width' to kwargs if not specified
        kwargs = ({'width': 4096} | kwargs) # add 'width' to kwargs if not specified
    elif file_extension == ExportFiletype.SVG.value:
        from pyphoplacecellanalysis.External.pyqtgraph.exporters.SVGExporter import SVGExporter
        bg = pg.mkColor(0,0,0,0.0) # clear color unless a different one is specified
        kwargs = ({'background': bg} | kwargs) # add 'width' to kwargs if not specified
        exporter = SVGExporter(graphics_item)
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
        print(f'exported plot to "{savepath}"')
    return savepath


# ==================================================================================================================== #
# Modern 2022-10-04 PDF                                                                                                #
# from pyphoplacecellanalysis.General.Mixins.ExportHelpers import create_daily_programmatic_display_function_testing_folder_if_needed, build_pdf_metadata_from_display_context
# ==================================================================================================================== #
@function_attributes(tags=['folder','programmatic','daily','output','path','important','if_needed','filesystem'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-25 12:54')
def create_daily_programmatic_display_function_testing_folder_if_needed(out_path=None):
    """ Creates a folder with today's date like '2022-01-16' located in the `out_path` if provided or in 'EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting' by default if none is specified. 
    
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import create_daily_programmatic_display_function_testing_folder_if_needed
    
    """
    if out_path is None:   
        out_day_date_folder_name = datetime.today().strftime('%Y-%m-%d') # A string with the day's date like '2022-01-16'
        out_path = Path(r'EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting').joinpath(out_day_date_folder_name).resolve()
    else:
        out_path = Path(out_path) # make sure it's a Path
    out_path.mkdir(exist_ok=True, parents=True) # parents=True creates all necessary parent folders
    return out_path

@function_attributes(tags=['context','output','path','important'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-25 12:54')
def session_context_to_relative_path(parent_path, session_ctx):
    """Only uses the keys that define session: ['format_name','animal','exper_name', 'session_name'] to build the relative path

    Args:
        parent_path (Path): _description_
        session_ctx (IdentifyingContext): _description_

    Returns:
        _type_: _description_

    Usage:
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import session_context_to_relative_path
        
        curr_sess_ctx = local_session_contexts_list[0]
        # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
        figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
        session_context_to_relative_path(figures_parent_out_path, curr_sess_ctx)

    """
    parent_path = Path(parent_path)
    subset_includelist=['format_name','animal','exper_name', 'session_name']
    all_keys_found, found_keys, missing_keys = session_ctx.check_keys(subset_includelist, debug_print=False)
    if not all_keys_found:
        print(f'WARNING: missing {len(missing_keys)} keys from context: {missing_keys}. Building path anyway.')
    curr_sess_ctx_tuple = session_ctx.as_tuple(subset_includelist=subset_includelist, drop_missing=True) # ('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')
    return parent_path.joinpath(*curr_sess_ctx_tuple).resolve()

@function_attributes(tags=['figure','context','output','path','important'], input_requires=[], output_provides=[], uses=[], used_by=['build_pdf_metadata_from_display_context'], creation_date='2023-05-25 12:54')
def build_figure_basename_from_display_context(active_identifying_ctx, subset_includelist=None, subset_excludelist=None, context_tuple_join_character='_', debug_print=False):
    """ 
    Usage:
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_figure_basename_from_display_context
        curr_fig_save_basename = build_figure_basename_from_display_context(active_identifying_ctx, context_tuple_join_character='_')
        >>> 'kdiba_2006-6-09_1-22-43_batch_plot_test_long_only'
    """
    ## Note that active_identifying_ctx.as_tuple() can have non-string elements (e.g. debug_test_max_num_slices=128, which is an int). This is what we want, but for setting the metadata we need to convert them to strings
    context_tuple = [str(v) for v in list(active_identifying_ctx.as_tuple(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist, drop_missing=True))]
    fig_save_basename = context_tuple_join_character.join(context_tuple) # joins the elements of the context_tuple with '_'
    if debug_print:
        print(f'fig_save_basename: "{fig_save_basename}"')
    return fig_save_basename


# ==================================================================================================================== #
# 2023-06-14 - Configurable Figure Output Functions                                                                    #
# ==================================================================================================================== #

class FigureOutputLocation(Enum):
    """Specifies the filesystem location for the parent folder where figures are output."""
    DAILY_PROGRAMMATIC_OUTPUT_FOLDER = "daily_programmatic_output_folder" # the common folder for today's date
    SESSION_OUTPUT_FOLDER = "session_output_folder" # the session-specific output folder. f"{session_path}/output/figures"
    CUSTOM = "custom" # other folder. Must be specified.
    
    def get_figures_output_parent_path(self, overriding_root_path=None, make_folder_if_needed:bool=True) -> Path:
        """ DAILY_PROGRAMMATIC_OUTPUT_FOLDER: All figures are located in a subdirectory of a daily programmatic output folder:
        /c/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2023-06-14/kdiba/gor01/two/2006-6-08_21-16-25/kdiba_gor01_two_2006-6-08_21-16-25_batch_pho_jonathan_replay_firing_rate_comparison.png
        """
        if self.name == FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER.name:
            # figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
            out_day_date_folder_name = datetime.today().strftime('%Y-%m-%d') # A string with the day's date like '2022-01-16'
            relative_out_path = Path(r'EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting').joinpath(out_day_date_folder_name)
            
            if overriding_root_path is not None:
                if isinstance(overriding_root_path, str):
                    overriding_root_path = Path(overriding_root_path)
                assert isinstance(overriding_root_path, Path)
                absolute_out_path = overriding_root_path.joinpath(relative_out_path).resolve()
            else:
                absolute_out_path = relative_out_path.resolve()
            
        elif self.name == FigureOutputLocation.SESSION_OUTPUT_FOLDER.name:
            raise NotImplementedError
            absolute_out_path = None
        elif self.name == FigureOutputLocation.CUSTOM.name:
            # custom mode:        
            assert overriding_root_path is not None, f"in FigureOutputLocation.CUSTOM mode, the `overriding_root_path` must be passed!"
            if not isinstance(overriding_root_path, Path):
                overriding_root_path = Path(overriding_root_path)
            absolute_out_path = overriding_root_path
        else:
            raise NotImplementedError(f"unknown type: {self}")

        # end if		
        if make_folder_if_needed:
            absolute_out_path.mkdir(exist_ok=True, parents=True) # parents=True creates all necessary parent folders
        return absolute_out_path


class ContextToPathMode(Enum):
    """ Controls how hierarchical contexts (IdentityContext) are mapped to relative output paths.
    In HIERARCHY_UNIQUE mode the folder hierarchy partially specifies the context (mainly the session part, e.g. './kdiba/gor01/two/2006-6-08_21-16-25/') so the filenames don't need to be completely unique (they can drop the 'kdiba_gor01_two_2006-6-08_21-16-25_' portion)
        'output/kdiba/gor01/two/2006-6-08_21-16-25/batch_pho_jonathan_replay_firing_rate_comparison.png

    In GLOBAL_UNIQUE mode the outputs are placed in a flat folder structure ('output/'), meaning the filenames need to be completely unique and specify all parts of the context:
        'output/kdiba_gor01_two_2006-6-08_21-16-25_batch_pho_jonathan_replay_firing_rate_comparison.png'
    """
    HIERARCHY_UNIQUE = "hierarchy_unique"
    GLOBAL_UNIQUE = "global_unique"

    @function_attributes(tags=['context','output','path','important'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-25 12:54')
    def session_context_to_relative_path(self, parent_path, session_ctx) -> Path:
        """Only uses the keys that define session: ['format_name','animal','exper_name', 'session_name'] to build the relative path

        Args:
            parent_path (Path): _description_
            session_ctx (IdentifyingContext): _description_

        Returns:
            _type_: _description_

        Usage:
            from pyphoplacecellanalysis.General.Mixins.ExportHelpers import session_context_to_relative_path
            
            curr_sess_ctx = local_session_contexts_list[0]
            # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
            figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
            session_context_to_relative_path(figures_parent_out_path, curr_sess_ctx)

        """
        if isinstance(parent_path, str):
            parent_path = Path(parent_path)

        if self.name == ContextToPathMode.GLOBAL_UNIQUE.name:
            return parent_path.resolve() # in this mode everything is globally unique, so it's all output in the same base folder. Just return the unaltered base folder.

        elif self.name == ContextToPathMode.HIERARCHY_UNIQUE.name:
            subset_includelist=['format_name','animal','exper_name', 'session_name']
            all_keys_found, found_keys, missing_keys = session_ctx.check_keys(subset_includelist, debug_print=False)
            if not all_keys_found:
                print(f'WARNING: missing {len(missing_keys)} keys from context: {missing_keys}. Building path anyway.')
            curr_sess_ctx_tuple = session_ctx.as_tuple(subset_includelist=subset_includelist, drop_missing=True) # ('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')
            return parent_path.joinpath(*curr_sess_ctx_tuple).resolve()
        else:
            raise NotImplementedError

    @function_attributes(tags=['figure','context','output','path','important'], input_requires=[], output_provides=[], uses=[], used_by=['build_pdf_metadata_from_display_context'], creation_date='2023-05-25 12:54')
    def build_figure_basename_from_display_context(self, active_identifying_ctx, subset_includelist=None, subset_excludelist=None, context_tuple_join_character='_', debug_print=False) -> str:
        """ 
        Usage:
            from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_figure_basename_from_display_context
            curr_fig_save_basename = build_figure_basename_from_display_context(active_identifying_ctx, context_tuple_join_character='_')
            >>> 'kdiba_2006-6-09_1-22-43_batch_plot_test_long_only'
        """
        subset_excludelist = (subset_excludelist or [])
        
        if self.name == ContextToPathMode.GLOBAL_UNIQUE.name:
            ## Note that active_identifying_ctx.as_tuple() can have non-string elements (e.g. debug_test_max_num_slices=128, which is an int). This is what we want, but for setting the metadata we need to convert them to strings
            pass # nothing needs to be added to the subset_exclude list

        elif self.name == ContextToPathMode.HIERARCHY_UNIQUE.name:
            session_subset_excludelist = ['format_name','animal','exper_name', 'session_name']
            subset_excludelist = subset_excludelist + session_subset_excludelist # add the session keys to the subset_excludelist
        else:
            raise NotImplementedError
        
        context_tuple = [str(v) for v in list(active_identifying_ctx.as_tuple(subset_includelist=subset_includelist, subset_excludelist=subset_excludelist, drop_missing=True))]
        fig_save_basename = context_tuple_join_character.join(context_tuple) # joins the elements of the context_tuple with '_'
        if debug_print:
            print(f'fig_save_basename: "{fig_save_basename}"')
        return fig_save_basename


@define(slots=False)
class FileOutputManager:
    """ 2023-06-14 - Manages figure output. Singleton/not persisted.

    Usage:
    
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode
        fig_man = FileOutputManager(figure_output_location=FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER, context_to_path_mode=ContextToPathMode.GLOBAL_UNIQUE)
        test_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='display_long_short_laps')
        fig_man.get_figure_save_file_path(test_context, make_folder_if_needed=False)
        >>> Path('/home/halechr/repo/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2023-06-14/kdiba_gor01_one_2006-6-08_14-26-15_display_long_short_laps')
    """
    figure_output_location: FigureOutputLocation = field()
    context_to_path_mode: ContextToPathMode = field()
    override_output_parent_path: Optional[Path] = field(default=None)
    
    def get_figure_output_parent_and_basename(self, final_context: IdentifyingContext, make_folder_if_needed:bool=True, **kwargs) -> tuple[Path, str]:
        """ gets the final output path for the figure to be saved specified by final_context """
        if self.figure_output_location.name == FigureOutputLocation.CUSTOM.name:
            assert self.override_output_parent_path is not None
        else:
            assert self.override_output_parent_path is None, f"for all modes other than FigureOutputLocation.CUSTOM, the override_output_parent_path should be None!"
                        

        figures_parent_out_path = self.figure_output_location.get_figures_output_parent_path(overriding_root_path=self.override_output_parent_path, make_folder_if_needed=make_folder_if_needed)            
        fig_save_path = self.context_to_path_mode.session_context_to_relative_path(figures_parent_out_path, session_ctx=final_context)
        if make_folder_if_needed:
            fig_save_path.mkdir(parents=True, exist_ok=True) # make folder if needed
        fig_save_basename = self.context_to_path_mode.build_figure_basename_from_display_context(final_context, **kwargs)
        return fig_save_path.resolve(), fig_save_basename

    def get_figure_output_parent_path(self, final_context: IdentifyingContext, make_folder_if_needed:bool=True, **kwargs) -> Path:
        """ Returns the parent path for figure output. Typically shouldn't be used except for drop-in compatibility.        
        """
        parent_save_path, _ = self.get_figure_output_parent_and_basename(final_context, make_folder_if_needed=make_folder_if_needed, **kwargs)
        return parent_save_path

    def get_figure_save_file_path(self, final_context: IdentifyingContext, make_folder_if_needed:bool=True, **kwargs) -> Path:
        """ Returns a complete path to a file without the extension (as a basepath). Same information output by `get_figure_output_parent_and_basename` but returns a single output path instead of the parent_path and basename.
        
        """
        parent_save_path, fig_save_basename = self.get_figure_output_parent_and_basename(final_context, make_folder_if_needed=make_folder_if_needed, **kwargs)
        fig_save_basename = sanitize_filename_for_Windows(fig_save_basename)
        return parent_save_path.joinpath(fig_save_basename).resolve()






# ==================================================================================================================== #
# Split Pre-2023-06-14 Functions                                                                                       #
# ==================================================================================================================== #

@function_attributes(tags=['figure','pdf','context','output','path','important'], input_requires=[], output_provides=[], uses=['build_figure_basename_from_display_context'], used_by=[], creation_date='2023-05-25 12:54')
def build_pdf_metadata_from_display_context(active_identifying_ctx: IdentifyingContext, subset_includelist=None, subset_excludelist=None, debug_print=False):
    """ Internally uses `build_figure_basename_from_display_context(...)` 
    Usage:
        curr_built_pdf_metadata, curr_pdf_save_filename = build_pdf_metadata_from_display_context(active_identifying_ctx)

    """
    # Filename:
    curr_fig_save_basename = build_figure_basename_from_display_context(active_identifying_ctx, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist, context_tuple_join_character='_')
    curr_pdf_save_filename = curr_fig_save_basename + '.pdf'
    if debug_print:
        print(f'curr_pdf_save_filename: "{curr_pdf_save_filename}"')

    # PDF metadata:
    active_identifying_ctx.get_subset(subset_includelist=['format_name', 'session_name'])
    if active_identifying_ctx.check_keys(keys_list=['format_name', 'session_name'])[0]:
        session_descriptor_string: str = '_'.join([active_identifying_ctx.format_name, active_identifying_ctx.session_name]) # 'kdiba_2006-6-08_14-26-15'
    else:
        # print(f'no session. in context (err: {err}). Just using context description')
        session_descriptor_string: str = active_identifying_ctx.get_description(separator='_', include_property_names=False)    
    if debug_print:
        print(f'session_descriptor_string: "{session_descriptor_string}"')
    built_pdf_metadata = {'Creator': 'Spike3D - TestNeuroPyPipeline227', 'Author': 'Pho Hale', 'Title': session_descriptor_string, 'Subject': '', 'Keywords': [session_descriptor_string]}   
    built_pdf_metadata['Title'] = curr_fig_save_basename
    if active_identifying_ctx.check_keys(keys_list=['display_fn_name'])[0]:
        built_pdf_metadata['Subject'] = active_identifying_ctx.display_fn_name
                                                               
    built_pdf_metadata['Keywords'] = build_figure_basename_from_display_context(active_identifying_ctx, context_tuple_join_character=' | ') # ' | '.join(context_tuple)

    return built_pdf_metadata, curr_pdf_save_filename


import matplotlib.pyplot as plt
## PDF Output, NOTE this is single plot stuff: uses active_config_name
from matplotlib.backends import backend_pdf # Needed for
# from pyphoplacecellanalysis.General.Mixins.ExportHelpers import create_daily_programmatic_display_function_testing_folder_if_needed, build_pdf_metadata_from_display_context
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # required for extract_figures_from_display_function_output

@function_attributes(short_name=None, tags=[], input_requires=[], output_provides=[], uses=['MatplotlibRenderPlots'], used_by=['programmatic_render_to_file'], creation_date='2023-06-08 12:15')
def extract_figures_from_display_function_output(out_display_var, out_fig_list:List=None, debug_print=False)->List:
    """ overcomes the lack of standardization in the display function outputs (some return dicts, some lists of figures, some wrapped with `MatplotlibRenderPlots` to extract the figures and add them to the `out_fig_list` """
    if out_fig_list is None:
        out_fig_list = []

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
        
    return out_fig_list
    

## 2022-10-04 Modern Programmatic PDF outputs:
@function_attributes(short_name=None, tags=['Depricating', 'PDF', 'export', 'output', 'matplotlib', 'display', 'file', 'active'], input_requires=[], output_provides=[], uses=['create_daily_programmatic_display_function_testing_folder_if_needed', 'session_context_to_relative_path', 'build_pdf_metadata_from_display_context'], used_by=[], creation_date='2022-10-04 00:00', related_items=[])
def programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_plot_decoded_epoch_slices', subset_includelist=None, subset_excludelist=None,  debug_print=False, **kwargs):
    """
    2022-10-04 Modern Programmatic PDF outputs
    curr_display_function_name = '_display_plot_decoded_epoch_slices' 

    Looks it this is done for EACH filtered context (in the loop below) whereas the original just did a single specific context
    """

    #TODO 2023-07-06 15:33: - [ ] Currently required for only one display function: `_display_1d_placefield_validations`
    # raise PendingDeprecationWarning

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
        # active_display_fn_kwargs = overriding_dict_with(lhs_dict=dict(), **kwargs) # this is always an error, if lhs_dict is empty the result will be empty regardless of the value of kwargs.
        active_display_fn_kwargs = kwargs
        
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
            active_pdf_metadata, active_pdf_save_filename = build_pdf_metadata_from_display_context(final_context, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist)
            active_pdf_save_path = active_session_figures_out_path.joinpath(active_pdf_save_filename) # build the final output pdf path from the pdf_parent_out_path (which is the daily folder)

            ## BEGIN DISPLAY/SAVE
            with backend_pdf.PdfPages(active_pdf_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
                out_fig_list = [] # Separate PDFs mode:

                if debug_print:
                    print(f'active_pdf_save_path: {active_pdf_save_path}\nactive_pdf_metadata: {active_pdf_metadata}')
                    print(f'active_display_fn_kwargs: {active_display_fn_kwargs}')
                    

                # All display is done here:
                out_display_var = curr_active_pipeline.display(curr_display_function_name, a_filtered_context, **active_display_fn_kwargs) # , filter_epochs='ripple', debug_test_max_num_slices=128
                # , fignum=active_identifying_ctx_string, **figure_format_config
    
                if debug_print:
                    print(f'completed display(...) call. type(out_display_var): {type(out_display_var)}\n out_display_var: {out_display_var}, active_display_fn_kwargs: {active_display_fn_kwargs}')

                out_fig_list = extract_figures_from_display_function_output(out_display_var=out_display_var, out_fig_list=out_fig_list)

                if debug_print:
                    print(f'out_fig_list: {out_fig_list}')

                # Finally iterate through and do the saving to PDF
                for i, a_fig in enumerate(out_fig_list):
                    pdf.savefig(a_fig, transparent=True)
                    pdf.attach_note(f'Page {i + 1}: "{active_identifying_ctx_string}"')
                    
                curr_active_pipeline.register_output_file(output_path=active_pdf_save_path, output_metadata={'filtered_context': a_filtered_context, 'context': active_identifying_ctx, 'fig': out_fig_list})


@metadata_attributes(short_name=None, tags=['pdf', 'export', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-01 01:13', related_items=[])
class FigureToImageHelpers:
    """ Exports the entire active_2d_plot timeline (all tracks) to a multi-page PDF file 
    """
    @classmethod
    def _DEP_helper_extract_renderables_from_track_widgets(cls, active_2d_plot, included_track_dock_identifiers: Optional[List]=None):
        """ Gets the renderable embedded in the track widget for the provided `included_track_dock_identifiers`.
        Usage:

            found_heterogeneous_stack, normalized_track_heights, included_track_dock_identifiers = cls._helper_extract_renderables_from_track_widgets(active_2d_plot, included_track_dock_identifiers=None)

        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget import PyqtgraphTimeSynchronizedWidget
        from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget
        import matplotlib.image as mimage
        from matplotlib.axes import Axes
        from matplotlib.artist import Artist

        if included_track_dock_identifiers is None:
            ## all tracks:
            included_track_dock_identifiers = active_2d_plot.dock_manager_widget.get_leaf_only_flat_dock_identifiers_list()

        found_widgets = [active_2d_plot.find_dock_item_tuple(an_id)[-1] for an_id in included_track_dock_identifiers]

        found_heterogeneous_stack = []

        track_heights = []
        for a_widget in found_widgets:
            if isinstance(a_widget, PyqtgraphTimeSynchronizedWidget):
                widget: PyqtgraphTimeSynchronizedWidget = a_widget
                root_plot_item: pg.PlotItem = widget.getRootPlotItem()
                # found_pyqtgraph_stack.append(root_plot_item)
                found_heterogeneous_stack.append(root_plot_item)
            elif isinstance(a_widget, MatplotlibTimeSynchronizedWidget):
                found_renderable = None
                try:
                    found_renderable = a_widget.plots.im_posterior_x
                except KeyError as e:
                    ## try to discover the axes images directly
                    fig = a_widget.plots.fig # plt.gcf()
                    assert fig is not None
                    axes_images = [im for ax in fig.axes for im in ax.get_images() if isinstance(im, mimage.AxesImage)]
                    
                    if len(axes_images) > 0:
                        # Found AxesImage objects
                        # assert len(axes_images) == 1, f"TODO - only allow the first (single) AxesImage to be added. len(axes_images): {len(axes_images)}"
                        if len(axes_images) > 1:
                            print(f'WARN: TODO - only allow the first (single) AxesImage to be added. len(axes_images): {len(axes_images)}\n\tONLY THE FIRST WILL BE USED!')
                        found_renderable = axes_images[0] ## Only add the first
                    else:
                        # No AxesImage found, look for Axes objects with plot artists (lines, patches, collections, etc.)
                        # Check for axes that have actual plot data: lines, patches, collections, or containers
                        axes_with_plot_data = [ax for ax in fig.axes 
                                               if isinstance(ax, Axes) and 
                                               (len(ax.lines) > 0 or len(ax.patches) > 0 or 
                                                len(ax.collections) > 0 or len(ax.containers) > 0)]
                        
                        if len(axes_with_plot_data) > 0:
                            if len(axes_with_plot_data) > 1:
                                print(f'WARN: Multiple axes found without AxesImage. len(axes_with_plot_data): {len(axes_with_plot_data)}\n\tONLY THE FIRST WILL BE USED!')
                            found_renderable = axes_with_plot_data[0] ## Only add the first
                        else:
                            raise ValueError(f'No AxesImage or Axes with plot data found in MatplotlibTimeSynchronizedWidget. fig.axes: {fig.axes}')
                except Exception as e:
                    found_renderable = None
                    raise e

                if found_renderable is not None:
                    # found_matplotlib_stack.append(found_renderable)
                    found_heterogeneous_stack.append(found_renderable)
            else:
                raise NotImplementedError(f'unexpected widget type: {type(a_widget), a_widget: {a_widget}}')

            ## get height
            track_heights.append(a_widget.height())

        # found_matplotlib_stack
        # found_pyqtgraph_stack
        found_heterogeneous_stack

        track_heights = np.array(track_heights)
        normalized_track_heights = track_heights / np.sum(track_heights)

        return found_heterogeneous_stack, normalized_track_heights, included_track_dock_identifiers
    


    @function_attributes(short_name=None, tags=['DEP', 'OLD', 'pdf', 'export', 'wrapped', 'multi-track', 'pyqtgraph', 'matplotlib'], creation_date='2025-08-22 02:30')
    @classmethod
    def _DEP_perform_export_wrapped_tracks_to_paged_pdf(cls, tracks: List, x_extent: tuple, chunk_width: float, output_pdf_path: str, rows_per_page: int=5, figsize=(8, 11), dpi=150, normalized_track_heights: Optional[List]=None, debug_max_num_pages: Optional[int]=5, track_labels: Optional[List[str]]=None, debug_print:bool=False):
        """
        Export a mixed list of matplotlib AxesImages and PyQtGraph PlotItems to a wrapped, paged PDF.

        A "chunk" is a block of all tracks (stacked vertically) that makes up the conceptual row (but they themselves have several vertically stacked tracks looking like subrows). 
        Each "page" has a fixed number of rows `rows_per_page`. 

        """
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from pyphoplacecellanalysis.External.pyqtgraph.exporters.ImageExporter import ImageExporter
        from PyQt5.QtGui import QImage
        import matplotlib.image as mimage
        from matplotlib.axes import Axes
        from matplotlib.artist import Artist
        import matplotlib.pyplot as plt
        import io
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # Styling like matplotlib version
        track_separator_line_kwargs = dict(color='white', linewidth=2, linestyle='-', alpha=0.8)
        time_label_formatting_kwargs = dict(fontsize=10, color='black')
        multi_track_label_formatting_kwargs = dict(fontsize=9, color='black')

        if not isinstance(tracks, (list, tuple)):
            tracks = [tracks]

        if track_labels is not None and len(track_labels) != len(tracks):
            print(f"Warning: track_labels length ({len(track_labels)}) != tracks length ({len(tracks)}). Ignoring labels.")
            track_labels = None
        has_labels = track_labels is not None


        if normalized_track_heights is None:
            raise NotImplementedError(f'normalized_track_heights is now required! The previous implementation did not work.')

        if normalized_track_heights is not None and len(normalized_track_heights) != len(tracks):
            print(f"Warning: track_heights length ({len(normalized_track_heights)}) != tracks length ({len(tracks)}). Ignoring track_heights.")
            normalized_track_heights = None
        has_track_heights = normalized_track_heights is not None
        if normalized_track_heights is not None:
            ## converts the track heights into matplotlib figure units (inches):
            fig_total_height: float = float(figsize[1])
            # track_rel_heights = deepcopy(normalized_track_heights)
            track_heights = (normalized_track_heights * fig_total_height)


        x_min, x_max = x_extent

        # Collect metadata dictionary for stacking
        export_infos = []
        y_offset = 0
        for track_IDX, t in enumerate(tracks):

            if isinstance(t, mimage.AxesImage):
                # ## Data units version:
                # #t.get_extent() is like [-2.84147705365001e-15, 1458.5500000000002, 0.0, 287.7697841726619] and in data units
                # y_min, y_max = t.get_extent()[2:4] ## these are in data units, like [0.0, 287.7697841726619] and so the same for many tracks
                # h = y_max - y_min ## in data units
                # extent = [t.get_extent()[0], t.get_extent()[1], y_offset, (y_offset+h)]

                ## Figure units version:
                #t.get_extent() is like [-2.84147705365001e-15, 1458.5500000000002, 0.0, 287.7697841726619] and in data units
                y_min = 0.0
                y_max = track_heights[track_IDX] ## these are in data units, like [0.0, 287.7697841726619] and so the same for many tracks
                h = y_max - y_min ## in data units
                extent = [t.get_extent()[0], t.get_extent()[1], y_offset, (y_offset+h)]
                export_infos.append(dict(kind="mpl", subkind="AxesImage", obj=t, extent=extent, y_height=h))
                
            elif isinstance(t, (Axes, Artist)):
                ## matplotlib general axes or Artist
                ## Figure units version:
                # For Axes objects, use x_min and x_max from the function parameters (similar to pyqtgraph PlotItems)
                y_min = 0.0
                y_max = track_heights[track_IDX] ## these are in data units, like [0.0, 287.7697841726619] and so the same for many tracks
                h = y_max - y_min ## in data units
                extent = [x_min, x_max, y_offset, (y_offset+h)]
                export_infos.append(dict(kind="mpl", subkind="Axes", obj=t, extent=extent, y_height=h))
                

            else:  # assume pg.PlotItem
                # ## Data units version: for 3 tracks, we get [[-4.4, 0.4], [-4.0, 45.5], [0, 1]]
                # y_min, y_max = t.getViewBox().viewRange()[1]
                # h = y_max - y_min
                # extent = [x_min, x_max, y_offset, y_offset+h]

                ## Figure units version:
                #t.get_extent() is like [-2.84147705365001e-15, 1458.5500000000002, 0.0, 287.7697841726619] and in data units
                y_min = 0.0
                y_max = track_heights[track_IDX] ## these are in data units, like [0.0, 287.7697841726619] and so the same for many tracks
                h = y_max - y_min ## in data units
                extent = [x_min, x_max, y_offset, (y_offset+h)]

                export_infos.append(dict(kind="pg", subkind="PlotItem", obj=t, extent=extent, y_height=h))

            ## must spit out `h`
            y_offset += h

        total_y_min, total_y_max = 0, y_offset

        if debug_print:
            print(f'export_infos: {export_infos}')

        # Chunking/building output images:
        chunks = []
        start = x_min
        while start < x_max:
            end = min(start+chunk_width, x_max)
            chunks.append((start, end))
            start = end
        pages = [chunks[i:(i+rows_per_page)] for i in range(0, len(chunks), rows_per_page)]
        if debug_max_num_pages is not None:
            pages = pages[:debug_max_num_pages]

        with backend_pdf.PdfPages(output_pdf_path) as pdf:
            for page_chunks in pages:
                fig, axes = plt.subplots(nrows=len(page_chunks), figsize=figsize, dpi=dpi, constrained_layout=True)
                if len(page_chunks) == 1:
                    axes = [axes]

                first_chunk = True
                for ax, (start, end) in zip(axes, page_chunks):
                    # render each track
                    for info in export_infos:
                        if debug_print:
                            print(f'info["extent"]: {info["extent"]}')

                        if info['kind'] == "mpl":
                            if info['subkind'] == "AxesImage":
                                arr = info['obj'].get_array()
                                cmap = info['obj'].get_cmap()
                                ax.imshow(arr, extent=[info['extent'][0], info['extent'][1], info['extent'][2], info['extent'][3]], aspect='auto', cmap=cmap, origin=info['obj'].origin)
                            elif info['subkind'] == "Axes":
                                ## Copy the general matplotlib Axes object to the temporary render axes:
                                source_ax = info['obj']
                                
                                # Get original limits
                                orig_xlim = source_ax.get_xlim()
                                orig_ylim = source_ax.get_ylim()
                                
                                # Temporarily set X limits to chunk range
                                source_ax.set_xlim(start, end)
                                
                                # Get the source figure and ensure it's drawn
                                source_fig = source_ax.figure
                                source_fig.canvas.draw()
                                
                                # Get the axes bbox in display coordinates
                                bbox = source_ax.get_tightbbox(source_fig.canvas.renderer)
                                
                                # Convert bbox to inches for rendering
                                # bbox is in display coordinates (pixels at source_fig.dpi), convert to inches
                                bbox_inches = bbox.transformed(source_fig.dpi_scale_trans.inverted())
                                
                                # Calculate the output size in pixels when rendered at the specified DPI
                                output_width_pixels = bbox_inches.width * dpi
                                output_height_pixels = bbox_inches.height * dpi
                                
                                # Check if dimensions exceed the maximum (2^16 - 1 = 65535)
                                max_dimension = 65535
                                if output_width_pixels > max_dimension or output_height_pixels > max_dimension:
                                    # Calculate scale factor to fit within limits
                                    scale = min(max_dimension / output_width_pixels, max_dimension / output_height_pixels)
                                    # Adjust DPI to achieve the desired size
                                    effective_dpi = int(dpi * scale)
                                else:
                                    effective_dpi = dpi
                                
                                # Create a buffer to render to
                                buf = io.BytesIO()
                                
                                # Render source figure with bbox_inches and adjusted DPI
                                source_fig.savefig(buf, format='png', dpi=effective_dpi, bbox_inches=bbox_inches, pad_inches=0, facecolor='white')
                                
                                # Read the image array
                                buf.seek(0)
                                img_arr = mimage.imread(buf)
                                buf.close()
                                
                                # Restore original limits
                                source_ax.set_xlim(*orig_xlim)
                                source_ax.set_ylim(*orig_ylim)
                                
                                # Display the rendered image in the target axes
                                ax.imshow(img_arr, extent=[start, end, info['extent'][2], info['extent'][3]], aspect='auto', origin='upper')
                            
                        else:  
                            # pyqtgraph-backed tracks
                            pi = info['obj']
                            vb = pi.getViewBox()
                            orig_x, orig_y = vb.viewRange()
                            
                            # Temporarily break X-link if present (e.g., for new_curves_separate_plot)
                            # This prevents the linked plot from overriding the X range change during export
                            orig_x_link = vb.linkedView(pg.ViewBox.XAxis)  # Get current X-axis link
                            if orig_x_link is not None:
                                pi.setXLink(None)  # Temporarily unlink
                            
                            pi.setXRange(start, end, padding=0) ## set to this chunk
                            pi.setYRange(*orig_y, padding=0)
                            exporter = ImageExporter(pi)
                            # exporter.parameters()['width'] = int(figsize[0]*dpi)
                            # exporter.parameters()['height'] = int(((figsize[1]/len(page_chunks))*dpi)/len(tracks))
                            exporter.parameters()['width'] = int((end - start) * dpi) # AI suggests I should be using `figsize[0] * dpi` - I don't think this is right.
                            exporter.parameters()['height'] = int((info['extent'][3] - info['extent'][2]) * dpi)
                            if debug_print:
                                print(f"\texporter.parameters(): w: {exporter.parameters()['width']}, h: {exporter.parameters()['height']}")
                            # exporter.parameters()['width'] = int(figsize[0]*dpi)
                            # exporter.parameters()['height'] = int((figsize[1]/len(page_chunks))*dpi/len(tracks))
                            img = exporter.export(toBytes=True)
                            if isinstance(img, QImage):
                                w, h = img.width(), img.height()
                                ptr = img.bits(); ptr.setsize(img.byteCount())
                                # QImage from pyqtgraph is typically in BGRA byte order.
                                raw = np.array(ptr).reshape(h, w, 4).astype(np.float32) / 255.0
                                b = raw[:, :, 0]
                                g = raw[:, :, 1]
                                r = raw[:, :, 2]
                                a = raw[:, :, 3]
                                rgb = np.stack([r, g, b], axis=-1)
                                # Composite over white background so grid and image blend as on-screen
                                bg = np.ones_like(rgb)
                                comp = rgb * a[..., None] + bg * (1.0 - a[..., None])
                                arr = (comp * 255).astype(np.uint8)
                            else:
                                arr = np.array(img)
                                
                            ## render the image into the temporary matplotlib ax using `ax.imshow(...)`
                            ax.imshow(arr, extent=[start, end, info['extent'][2], info['extent'][3]], aspect='auto', origin='upper') 
                            # ax.imshow(arr, extent=[info['extent'][0], info['extent'][1], info['extent'][2], info['extent'][3]], aspect='auto', origin='upper') ## tried this, and it's markedly wrong
                            
                            ## restore previous ranges and X-link
                            pi.setXRange(*orig_x, padding=0)
                            pi.setYRange(*orig_y, padding=0)
                            if orig_x_link is not None:
                                pi.setXLink(orig_x_link)  # Restore X-link

                    # separators between tracks
                    if len(export_infos) > 1:
                        for i, info in enumerate(export_infos[:-1]):
                            sep_y = info['extent'][3]
                            ax.axhline(y=sep_y, **track_separator_line_kwargs)

                    ax.set_xlim(start, end)
                    ax.set_ylim(total_y_min, total_y_max)

                    # labels (only first chunk per page)
                    if first_chunk and has_labels:
                        for info, lbl in zip(export_infos, track_labels):
                            yc = (info['extent'][2]+info['extent'][3])/2
                            ynorm = (yc-total_y_min)/(total_y_max-total_y_min)
                            ax.text(-0.01, ynorm, lbl, rotation=90, va='center', ha='center', transform=ax.transAxes, **multi_track_label_formatting_kwargs)

                    # start/end time outside edges
                    ax.text(-0.02 if has_labels else -0.01, 0.5, f"{start:.0f}", rotation=90, va='center', ha='center', transform=ax.transAxes, **time_label_formatting_kwargs)
                    ax.text(1.02, 0.5, f"{end:.0f}", rotation=90, va='center', ha='center', transform=ax.transAxes, **time_label_formatting_kwargs)

                    ax.set_xticks([]); ax.set_yticks([])
                    first_chunk = False

                pdf.savefig(fig)
                plt.close(fig)
            ## END for page_chunks in pages...
        ## END with backend_pdf.PdfPages(output_pdf_path) as pdf:...
        
        print(f"PDF saved to {output_pdf_path}")
        return output_pdf_path

    @function_attributes(short_name=None, tags=['DEP', 'OLD', 'tracks', 'MAIN', 'save', 'export', 'pdf', 'multi-page-pdf', 'timeline'], input_requires=[], output_provides=[], uses=['_helper_extract_renderables_from_track_widgets', 'perform_export_wrapped_tracks_to_paged_pdf'], used_by=[], creation_date='2025-08-22 08:13', related_items=[])
    @classmethod
    def _DEP_export_wrapped_tracks_to_paged_df(cls, active_2d_plot, output_pdf_path: str, included_track_dock_identifiers: Optional[List]=None, **kwargs):
        """ Exports the entire timeline (all tracks) out to a multi-paged PDF
        
        Usage:

            saved_output_pdf_path = FigureToImageHelpers.export_wrapped_tracks_to_paged_df(active_2d_plot, output_pdf_path=output_pdf_path)

        """


        found_heterogeneous_stack, normalized_track_heights, included_track_dock_identifiers = cls._DEP_helper_extract_renderables_from_track_widgets(active_2d_plot, included_track_dock_identifiers=included_track_dock_identifiers)

        return cls._DEP_perform_export_wrapped_tracks_to_paged_pdf(tracks=found_heterogeneous_stack, x_extent=(active_2d_plot.total_data_start_time, active_2d_plot.total_data_end_time), chunk_width=active_2d_plot.active_window_duration, output_pdf_path=output_pdf_path,
                                                        normalized_track_heights = normalized_track_heights, **kwargs,
                                                        )









    @function_attributes(short_name=None, tags=['NEW', 'tracks', 'MAIN', 'save', 'export', 'pdf', 'multi-page-pdf', 'timeline'], input_requires=[], output_provides=[], uses=['_helper_extract_renderables_from_track_widgets', 'perform_export_wrapped_tracks_to_paged_pdf'], used_by=[], creation_date='2025-08-22 08:13', related_items=[])
    @classmethod
    def export_wrapped_tracks_to_paged_df(cls, active_2d_plot, output_pdf_path: str, included_track_dock_identifiers: Optional[List]=None, **kwargs):
        """ Exports the entire timeline (all tracks) out to a multi-paged PDF
        
        Usage:

            saved_output_pdf_path = FigureToImageHelpers.export_wrapped_tracks_to_paged_df(active_2d_plot, output_pdf_path=output_pdf_path)

        """
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from pyphoplacecellanalysis.External.pyqtgraph.exporters.ImageExporter import ImageExporter
        from PyQt5.QtGui import QImage
        import matplotlib.image as mimage
        from matplotlib.axes import Axes
        from matplotlib.artist import Artist
        import matplotlib.pyplot as plt
        import io
        
        if included_track_dock_identifiers is None:
            ## all tracks:
            included_track_dock_identifiers = active_2d_plot.dock_manager_widget.get_leaf_only_flat_dock_identifiers_list()

        found_track_widgets = [active_2d_plot.find_dock_item_tuple(an_id)[-1] for an_id in included_track_dock_identifiers]
        track_heights = np.array([a_widget.height() for a_widget in found_track_widgets])
        normalized_track_heights = track_heights / np.sum(track_heights)


        # Unpack keyword arguments with defaults (kwargs with default values as individual assignments)
        # found_track_widgets = kwargs.pop('tracks', None)
        x_extent = kwargs.pop('x_extent', None)
        chunk_width = kwargs.pop('chunk_width', None)
        # output_pdf_path = kwargs.pop('output_pdf_path', None)
        rows_per_page = kwargs.pop('rows_per_page', 5)
        figsize = kwargs.pop('figsize', (8, 11))
        dpi = kwargs.pop('dpi', 150)
        # normalized_track_heights = kwargs.pop('normalized_track_heights', None)
        debug_max_num_pages = kwargs.pop('debug_max_num_pages', 5)
        track_labels = kwargs.pop('track_labels', None)
        debug_print = kwargs.pop('debug_print', False)

        # found_heterogeneous_stack, normalized_track_heights, included_track_dock_identifiers = cls._helper_extract_renderables_from_track_widgets(active_2d_plot, included_track_dock_identifiers=included_track_dock_identifiers)

        # If argument assignment was previously done in call, do it explicitly now:
        # found_track_widgets = found_heterogeneous_stack
        x_extent = (active_2d_plot.total_data_start_time, active_2d_plot.total_data_end_time)
        chunk_width = active_2d_plot.active_window_duration


        # Styling like matplotlib version
        track_separator_line_kwargs = dict(color='white', linewidth=2, linestyle='-', alpha=0.8)
        time_label_formatting_kwargs = dict(fontsize=10, color='black')
        multi_track_label_formatting_kwargs = dict(fontsize=9, color='black')

        if not isinstance(found_track_widgets, (list, tuple)):
            found_track_widgets = [found_track_widgets]

        if track_labels is not None and len(track_labels) != len(found_track_widgets):
            print(f"Warning: track_labels length ({len(track_labels)}) != tracks length ({len(found_track_widgets)}). Ignoring labels.")
            track_labels = None
        has_labels = track_labels is not None

        if normalized_track_heights is None:
            raise NotImplementedError(f'normalized_track_heights is now required! The previous implementation did not work.')

        if normalized_track_heights is not None and len(normalized_track_heights) != len(found_track_widgets):
            print(f"Warning: track_heights length ({len(normalized_track_heights)}) != tracks length ({len(found_track_widgets)}). Ignoring track_heights.")
            normalized_track_heights = None
        has_track_heights = normalized_track_heights is not None
        if normalized_track_heights is not None:
            ## converts the track heights into matplotlib figure units (inches):
            fig_total_height: float = float(figsize[1])
            # track_rel_heights = deepcopy(normalized_track_heights)
            track_heights = (normalized_track_heights * fig_total_height)


        x_min, x_max = x_extent

        # Collect metadata dictionary for stacking

        y_offsets = np.cumsum(np.concatenate([[0], track_heights])) ## this better be correct
        # Assert.same_length(y_offsets, found_track_widgets)
        Assert.len_equals(y_offsets, required_length=(len(found_track_widgets)+1)) # same_length(y_offsets, found_track_widgets)

        export_infos = [dict(extent=[x_min, x_max, y_offsets[track_IDX], (y_offsets[track_IDX]+(track_heights[track_IDX] - 0.0))], y_height=(track_heights[track_IDX] - 0.0)) for track_IDX, t in enumerate(found_track_widgets)]
        total_y_min = 0.0
        total_y_max = y_offsets[-1]

        if debug_print:
            print(f'export_infos: {export_infos}')

        # Chunking/building output images:
        chunks = []
        start = x_min
        while start < x_max:
            end = min(start+chunk_width, x_max)
            chunks.append((start, end))
            start = end
        pages = [chunks[i:(i+rows_per_page)] for i in range(0, len(chunks), rows_per_page)]
        if debug_max_num_pages is not None:
            pages = pages[:debug_max_num_pages]

        with backend_pdf.PdfPages(output_pdf_path) as pdf:
            for page_chunks in pages:
                fig, axes = plt.subplots(nrows=len(page_chunks), figsize=figsize, dpi=dpi, constrained_layout=True)
                if len(page_chunks) == 1:
                    axes = [axes]

                first_chunk = True
                for ax, (start, end) in zip(axes, page_chunks): 
                    ## one temp axes to draw into:
                    # render each track
                    for track_IDX, (t, info) in enumerate(zip(found_track_widgets, export_infos)):
                        # for info in export_infos:
                        if debug_print:
                            print(f'track_IDX: {track_IDX} \t info["extent"]: {info["extent"]}')


                        arr = t.export_as_img_arr(start=start, end=end, dpi=dpi, info=info)
                        ## render the image into the temporary matplotlib ax using `ax.imshow(...)`
                        ax.imshow(arr, extent=[start, end, info['extent'][2], info['extent'][3]], aspect='auto', origin='upper') 

                    # separators between tracks
                    if len(export_infos) > 1:
                        for i, info in enumerate(export_infos[:-1]):
                            sep_y = info['extent'][3]
                            ax.axhline(y=sep_y, **track_separator_line_kwargs)

                    ax.set_xlim(start, end)
                    ax.set_ylim(total_y_min, total_y_max)

                    # labels (only first chunk per page)
                    if first_chunk and has_labels:
                        for info, lbl in zip(export_infos, track_labels):
                            yc = (info['extent'][2]+info['extent'][3])/2
                            ynorm = (yc-total_y_min)/(total_y_max-total_y_min)
                            ax.text(-0.01, ynorm, lbl, rotation=90, va='center', ha='center', transform=ax.transAxes, **multi_track_label_formatting_kwargs)

                    # start/end time outside edges
                    ax.text(-0.02 if has_labels else -0.01, 0.5, f"{start:.0f}", rotation=90, va='center', ha='center', transform=ax.transAxes, **time_label_formatting_kwargs)
                    ax.text(1.02, 0.5, f"{end:.0f}", rotation=90, va='center', ha='center', transform=ax.transAxes, **time_label_formatting_kwargs)

                    ax.set_xticks([]); ax.set_yticks([])
                    first_chunk = False

                pdf.savefig(fig)
                try:
                    plt.close(fig)
                except ValueError as e:
                    print(f'failed to close temp figure with error e: {e}.')
                    pass
                except Exception as e:
                    print(f'ERROR: unhandled exception {e} while trying to close temp fig.')
                    raise
                
            ## END for page_chunks in pages...
        ## END with backend_pdf.PdfPages(output_pdf_path) as pdf:...
        
        print(f"PDF saved to {output_pdf_path}")
        return output_pdf_path




@function_attributes(short_name=None, tags=['PDF', 'export', 'output', 'matplotlib', 'display', 'file', 'active'], input_requires=[], output_provides=[], uses=['extract_figures_from_display_function_output'], used_by=[], creation_date='2023-06-08 11:55', related_items=[])
def programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_plot_decoded_epoch_slices', subset_includelist=None, subset_excludelist=None, write_vector_format=False, write_png=True, debug_print=False, **kwargs):
    """ Loops through the individual epochs in a session (e.g. ['maze1', 'maze2', 'maze']) analagous to the structure of `programmatic_display_to_PDF` and programmatically calls `perform_write_to_file` with the appropriate parameters.
    Newer Programmatic .png and .pdf outputs
    curr_display_function_name = '_display_plot_decoded_epoch_slices' 

    Looks it this is done for EACH filtered context (in the loop below) whereas the original just did a single specific context
    """
    from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import BatchPlotting

    if BatchPlotting._fig_out_man is None:
        collected_figures_folder, fig_out_man = BatchPlotting.find_batch_programmatic_figures_output_dir()
        assert fig_out_man is not None

    out_man: FileOutputManager = deepcopy(BatchPlotting._fig_out_man) # curr_active_pipeline.get_output_manager(figure_output_location=FigureOutputLocation.CUSTOM, context_to_path_mode=ContextToPathMode.GLOBAL_UNIQUE, override_output_parent_path=collected_outputs_path)
    _batch_figure_kwargs = dict(override_fig_man=out_man)


    ## Get the output path (active_session_figures_out_path) for this session (and all of its filtered_contexts as well):
    # fig_man = curr_active_pipeline.get_output_manager() # get the output manager
    # figures_parent_out_path, fig_save_basename = fig_man.get_figure_output_parent_and_basename(final_context, make_folder_if_needed=True)
    # active_session_figures_out_path = figures_parent_out_path
    
    # if debug_print:
    #     print(f'curr_session_parent_out_path: {active_session_figures_out_path}')
        
    all_out_fig_paths = []
    
    with plt.ioff():
        ## Disables showing the figure by default from within the context manager.
        # active_display_fn_kwargs = overriding_dict_with(lhs_dict=dict(filter_epochs='ripple', debug_test_max_num_slices=128), **kwargs)
        # active_display_fn_kwargs = overriding_dict_with(lhs_dict=dict(), **kwargs) # this is always an error, if lhs_dict is empty the result will be empty regardless of the value of kwargs.
        active_display_fn_kwargs = (_batch_figure_kwargs | kwargs)
        
        # Perform for each filtered context:
        for filter_name, a_filtered_context in curr_active_pipeline.filtered_contexts.items():
            if debug_print:
                print(f'filter_name: {filter_name}: "{a_filtered_context.get_description()}"')
            # Get the desired display function context:
            
            # final_context = active_identifying_display_ctx # Display only context    

            # # Add in the desired display variable:
            # active_identifying_ctx = active_identifying_display_ctx.adding_context('filter_epochs', **active_display_fn_kwargs) # , filter_epochs='ripple' ## TODO: this is only right for a single function!
            # final_context = active_identifying_ctx # Display/Variable context mode

            # out_fig_list = [] # list just for figures of this filtered context.
            out_display_var = curr_active_pipeline.display(curr_display_function_name, a_filtered_context, **active_display_fn_kwargs)
            
            try:
                extracted_context = out_display_var.context
            except Exception as e:
                print(f'could not extract the context: {e}')
                # raise e
                extracted_context = None

            if extracted_context is None:
                active_identifying_display_ctx = a_filtered_context.adding_context('display_fn', display_fn_name=curr_display_function_name)
                extracted_context = active_identifying_display_ctx
                if hasattr(out_display_var, 'context'):
                    out_display_var.context = extracted_context
                
            # Extract the figures:
            out_fig_list = extract_figures_from_display_function_output(out_display_var=out_display_var, out_fig_list=[]) # I think out_fig_list needs to be [] so it doesn't accumulate figures over the filtered_context?

            if debug_print:
                print(f'extracted_context: {extracted_context}')

            for fig in out_fig_list:
                active_out_figure_paths = curr_active_pipeline.output_figure(extracted_context, fig, write_vector_format=write_vector_format, write_png=write_png, **_batch_figure_kwargs, debug_print=debug_print)                 
                all_out_fig_paths.extend(active_out_figure_paths)

            # ## Build PDF Output Info
            # active_pdf_metadata, active_pdf_save_filename = build_pdf_metadata_from_display_context(final_context, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist)
            # active_pdf_save_path = active_session_figures_out_path.joinpath(active_pdf_save_filename) # build the final output pdf path from the pdf_parent_out_path (which is the daily folder)

            # ## BEGIN DISPLAY/SAVE
            # with backend_pdf.PdfPages(active_pdf_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
            #     out_fig_list = [] # Separate PDFs mode:

            #     if debug_print:
            #         print(f'active_pdf_save_path: {active_pdf_save_path}\nactive_pdf_metadata: {active_pdf_metadata}')
            #         print(f'active_display_fn_kwargs: {active_display_fn_kwargs}')
                    
            #     out_display_var = curr_active_pipeline.display(curr_display_function_name, a_filtered_context, **active_display_fn_kwargs) # , filter_epochs='ripple', debug_test_max_num_slices=128
            #     # , fignum=active_identifying_ctx_string, **figure_format_config
    
            #     if debug_print:
            #         print(f'completed display(...) call. type(out_display_var): {type(out_display_var)}\n out_display_var: {out_display_var}, active_display_fn_kwargs: {active_display_fn_kwargs}')

            #     out_fig_list = extract_figures_from_display_function_output(out_display_var=out_display_var, out_fig_list=out_fig_list)

            #     if debug_print:
            #         print(f'out_fig_list: {out_fig_list}')

            #     # Finally iterate through and do the saving to PDF
            #     for i, a_fig in enumerate(out_fig_list):
            #         pdf.savefig(a_fig, transparent=True)
            #         pdf.attach_note(f'Page {i + 1}: "{active_identifying_ctx_string}"')
                    
            #     curr_active_pipeline.register_output_file(output_path=active_pdf_save_path, output_metadata={'filtered_context': a_filtered_context, 'context': active_identifying_ctx, 'fig': out_fig_list})
    # end with plt.ioff():
    return all_out_fig_paths





@function_attributes(short_name=None, tags=['file','export','output','matplotlib','display','active','PDF','batch','automated'], input_requires=[], output_provides=[], uses=['write_to_file'], used_by=[], creation_date='2023-06-14 19:06', related_items=[])
def build_and_write_to_file(a_fig, active_identifying_ctx, fig_man:Optional[FileOutputManager]=None, subset_includelist=None, subset_excludelist=None, context_tuple_join_character='_', write_vector_format=False, write_png=True, register_output_file_fn=None, progress_print=True, debug_print=False, **kwargs):
    """ From the context, fig_man, and arguments builds the final save path for the figure and calls `write_to_file` with these values. """
    active_out_figure_paths = []
    write_any_figs = write_vector_format or write_png
    if not write_any_figs:
        return active_out_figure_paths # return empty list if no output formats are requested.

    # Use fig_man to build the path
    fig_man = fig_man or FileOutputManager(figure_output_location=FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER, context_to_path_mode=ContextToPathMode.GLOBAL_UNIQUE)
    curr_fig_save_path = fig_man.get_figure_save_file_path(active_identifying_ctx, make_folder_if_needed=True, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist, context_tuple_join_character=context_tuple_join_character)

    return write_to_file(a_fig, active_identifying_ctx, final_fig_save_basename_path=curr_fig_save_path, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist, write_vector_format=write_vector_format, write_png=write_png, register_output_file_fn=register_output_file_fn, progress_print=progress_print, debug_print=debug_print, **kwargs)
    

@function_attributes(short_name=None, tags=['file','export','output','matplotlib','display','active','PDF','batch','automated'], input_requires=[], output_provides=[], uses=[], used_by=['build_and_write_to_file'], creation_date='2023-05-31 19:16', related_items=['build_and_write_to_file'])
def write_to_file(a_fig, active_identifying_ctx: IdentifyingContext, final_fig_save_basename_path:Path, subset_includelist=None, subset_excludelist=None, write_vector_format=False, write_png=True, register_output_file_fn=None, progress_print=True, debug_print=False, **kwargs):
    """ Lowest level figure write function. Writes a single matplotlib figure out to one or more files based on whether write_png and write_vector_format are specified AND registers the output using `register_output_file_fn` if one is provided. 
    
    History: `perform_write_to_file`
    
    Aims to eventually replace `programmatic_display_to_PDF` (working for both PDF and PNG outputs, working for global plots, along with successfully registering output files with the pipeline via `register_output_file_fn` argument)

    
    Usage:
        # Plots in a shared folder for this session with fully distinct figure names:
        active_session_figures_out_path = curr_active_pipeline.get_daily_programmatic_session_output_path()
        final_context = curr_active_pipeline.sess.get_context().adding_context('display_fn', display_fn_name='plot_expected_vs_observed').adding_context('display_kwargs', **display_kwargs)
        active_out_figure_paths = perform_write_to_file(fig, final_context, figures_parent_out_path=active_session_figures_out_path)

        register_output_file_fn=curr_active_pipeline.register_output_file

    Inputs:
        register_output_file_fn: Callable[output_path:Path, output_metadata:dict] - function called to register outputs, by default should be `curr_active_pipeline.register_output_file`
        
        ## Replace any '.' characters with a suitable alternative so that .with_suffix('.pdf') doesn't incorrect replace everything after the period (overwriting floating-point values in the basename, for example)
        ·
        •
        •
        ⁃
        ∙
        ➗
    """
    import PIL
    from PIL import Image # Pillow Images
    from plotly.basedatatypes import BaseFigure as PlotlyBaseFigure ## for plotly figure detection

    # period_replacement_char: str = '-'
    # period_replacement_char: str = '➗'
    period_replacement_char: str = '•'
    
    active_out_figure_paths = []
    write_any_figs = write_vector_format or write_png
    if not write_any_figs:
        return active_out_figure_paths # return empty list if no output formats are requested.

    assert final_fig_save_basename_path is not None, f"Disabled automatic parent output path generation."
    ## Replace any '.' characters with a suitable alternative so that .with_suffix('.pdf') doesn't incorrect replace everything after the period (overwriting floating-point values in the basename, for example)
    erronious_suffixes = final_fig_save_basename_path.suffixes # ['.5']
    if len(erronious_suffixes) > 0:
        if debug_print:
            print(f'final_fig_save_basename_path should have no suffixes because it is a basename, but it has erronious_suffixes: {erronious_suffixes}\nfinal_fig_save_basename_path: {final_fig_save_basename_path}')
        filename_replaced = str(final_fig_save_basename_path).replace('.', period_replacement_char)
        if debug_print:
            print(f'filename_replaced: {filename_replaced}')
        final_fig_save_basename_path = Path(filename_replaced).resolve()
        if debug_print:
            print(f'final_fig_save_basename_path: {final_fig_save_basename_path}')
        # check the suffixes again:
        erronious_suffixes = final_fig_save_basename_path.suffixes
        assert len(erronious_suffixes) == 0, f"erronious_suffixes: {erronious_suffixes} is still not empty after renaming!"
        
    is_matplotlib_figure: bool = isinstance(a_fig, plt.FigureBase)
    is_plotly_figure: bool = isinstance(a_fig, PlotlyBaseFigure)
    is_pillow_image: bool = isinstance(a_fig, Image.Image) # PIL.Image
    

    # Extract saving kwargs
    

    # PDF: .pdf versions:
    if write_vector_format:
        try:
            ## MATPLOTLIB only:
            if is_matplotlib_figure:
                active_pdf_metadata, _unused_old_pdf_save_filename = build_pdf_metadata_from_display_context(active_identifying_ctx, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist) # ignores `active_pdf_save_filename`
                fig_vector_save_path = final_fig_save_basename_path.with_suffix('.pdf')
                with backend_pdf.PdfPages(fig_vector_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
                    # Save out PDF page:
                    pdf.savefig(a_fig, **kwargs)
                    
                additional_output_metadata = {'fig_format':'matplotlib', 'pdf_metadata': active_pdf_metadata}

            elif is_plotly_figure:
                ## Plotly only:
                fig_vector_save_path = final_fig_save_basename_path.with_suffix('.svg')
                a_fig.write_image(fig_vector_save_path)
                additional_output_metadata = {'fig_format':'plotly'}

            elif is_pillow_image:
                ## Pillow Image only:
                raise NotImplementedError(f'Pillow Images cannot be saved out to a vector format!')
                # fig_vector_save_path = final_fig_save_basename_path.with_suffix('.pdf')
                # a_fig.save(fig_vector_save_path)                 # Save image to file
                # additional_output_metadata = {'fig_format':'pillow'} # Unused here
                
            else:
                # pyqtgraph figure: pyqtgraph's exporter can't currently do PDF, so we'll do .svg instead:
                fig_vector_save_path = final_fig_save_basename_path.with_suffix('.svg')
                export_pyqtgraph_plot(a_fig, savepath=fig_vector_save_path, **kwargs)
                additional_output_metadata = {'fig_format':'pyqtgraph'}

            if register_output_file_fn is not None:
                register_output_file_fn(output_path=fig_vector_save_path, output_metadata={'context': active_identifying_ctx, 'fig': (a_fig), **additional_output_metadata})
            if progress_print:
                print(f'\t saved "{file_uri_from_path(fig_vector_save_path)}"')
            active_out_figure_paths.append(fig_vector_save_path)

        except Exception as e:
            print(f'Error occured while writing vector format for fig. {e}. Skipping.')

    # PNG: .png versions:
    if write_png:
        # curr_page_str = f'pg{i+1}of{num_pages}'
        try:
            fig_png_out_path = final_fig_save_basename_path.with_suffix('.png')
            # fig_png_out_path = fig_png_out_path.with_stem(f'{curr_pdf_save_path.stem}_{curr_page_str}') # note this replaces the current .pdf extension with .png, resulting in a good filename for a .png
            if is_matplotlib_figure:
                ## MATPLOTLIB only:
                a_fig.savefig(fig_png_out_path, **kwargs) ## pass kwargs so things like dpi and etc can be passed.
            
            elif is_plotly_figure:
                ## Plotly only:
                a_fig.write_image(fig_png_out_path)
                additional_output_metadata = {'fig_format':'plotly'} # Unused here

            elif is_pillow_image:
                ## Pillow Image only:
                # Save image to file
                a_fig.save(fig_png_out_path)
                additional_output_metadata = {'fig_format':'pillow'} # Unused here

            else:
                # pyqtgraph
                export_pyqtgraph_plot(a_fig, savepath=fig_png_out_path, **kwargs)

            if register_output_file_fn is not None:
                register_output_file_fn(output_path=fig_png_out_path, output_metadata={'context': active_identifying_ctx, 'fig': (a_fig)})
            if progress_print:
                print(f'\t saved "{file_uri_from_path(fig_png_out_path)}"')
            active_out_figure_paths.append(fig_png_out_path)
        except Exception as e:
            print(f'Error occured while writing .png for fig. {e}. Skipping.')
        
    return active_out_figure_paths


# ==================================================================================================================== #
# 2023-06-13 - Conceptual Outline                                                                                      #
# ==================================================================================================================== #
"""

# 1. Setting up initial configuration prior to plot (making plot not display, removing plot toolbar, setting rendering defaults, etc).

# 2. Iterating through filtered contexts (e.g. ['maze1', 'maze2', 'maze']) and results for non-global display functions

    # 2a. Generating Final Display Context

    # 2b. Performing display function call to actually generate figure

    # 3b. Writing file to disk

    

"""

# @function_attributes(short_name=None, tags=['UNUSED','UNFINISHED','batch','filtered_context','session','figures'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-13 10:27', related_items=[])
# def batch_plot_local_context(curr_active_pipeline, curr_display_function_name='_display_plot_decoded_epoch_slices', subset_filtered_context_includelist=None, subset_filtered_context_excludelist=None, write_vector_format=False, write_png=True, debug_print=False, **kwargs):
#     """ For a local display function specified by `curr_display_function_name`, loops through the individual epochs in a session (e.g. ['maze1', 'maze2', 'maze']) analagous to the structure of `programmatic_display_to_PDF` and for each:
#         # 2a. Generates Final Display Context

#         # 2b. Performing display function call to actually generate figure

#         # 3b. Writing file to disk - programmatically calls `perform_write_to_file` with the appropriate parameters.
    
            
#     Newer Programmatic .png and .pdf outputs
#     curr_display_function_name = '_display_plot_decoded_epoch_slices' 

#     """

#     ## Get the output path (active_session_figures_out_path) for this session (and all of its filtered_contexts as well):
#     # active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
#     # figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
#     # active_session_figures_out_path = session_context_to_relative_path(figures_parent_out_path, active_identifying_session_ctx)
#     # if debug_print:
#     #     print(f'curr_session_parent_out_path: {active_session_figures_out_path}')
#     # active_session_figures_out_path.mkdir(parents=True, exist_ok=True) # make folder if needed

#     active_included_contexts_list = []
#     if subset_filtered_context_includelist is None:
#         if subset_filtered_context_excludelist is not None:
#             print(f'using excludelist: {subset_filtered_context_excludelist}')
#             active_included_contexts_list = {k:v for k, v in curr_active_pipeline.filtered_contexts.items() if ((k not in subset_filtered_context_excludelist) and (v not in subset_filtered_context_excludelist))} 
#         else:
#             # no exclude list: include all
#             active_included_contexts_list = curr_active_pipeline.filtered_contexts # include all by default
#     else:
#         # use the includelist only
#         assert subset_filtered_context_excludelist is None, f"No excludelist can be used if includelist is provided!"
#         print(f'using includelist: {subset_filtered_context_includelist}')
#         if isinstance(subset_filtered_context_includelist, dict):
#             active_included_contexts_list = subset_filtered_context_includelist
#         else:
#             # assume it's a list of keys like ['maze1','maze2']
#             active_included_contexts_list = {k:v for k, v in curr_active_pipeline.filtered_contexts.items() if (k in subset_filtered_context_includelist)}

        

#     with plt.ioff():
#         ## Disables showing the figure by default from within the context manager.
#         # active_display_fn_kwargs = overriding_dict_with(lhs_dict=dict(filter_epochs='ripple', debug_test_max_num_slices=128), **kwargs)
#         # active_display_fn_kwargs = overriding_dict_with(lhs_dict=dict(), **kwargs) # this is always an error, if lhs_dict is empty the result will be empty regardless of the value of kwargs.
#         active_display_fn_kwargs = kwargs
        
#         # Perform for each filtered context:
#         for filter_name, a_filtered_context in active_included_contexts_list.items():
#             if debug_print:
#                 print(f'filter_name: {filter_name}: "{a_filtered_context.get_description()}"')
#             # Get the desired display function context:
#             active_identifying_display_ctx = a_filtered_context.adding_context('display_fn', display_fn_name=curr_display_function_name)
#             # final_context = active_identifying_display_ctx # Display only context    

#             # # Add in the desired display variable:
#             # active_identifying_ctx = active_identifying_display_ctx.adding_context('filter_epochs', **active_display_fn_kwargs) # , filter_epochs='ripple' ## TODO: this is only right for a single function!
#             # final_context = active_identifying_ctx # Display/Variable context mode
#             final_context = active_identifying_display_ctx

#             active_identifying_ctx_string = final_context.get_description(separator='|') # Get final discription string
#             if debug_print:
#                 print(f'active_identifying_ctx_string: "{active_identifying_ctx_string}"')



#             out_fig_list = []
            
#             # all display functions should return: their final display context, their figures, their optional save function override?


#             # ## Build PDF Output Info
#             # active_pdf_metadata, active_pdf_save_filename = build_pdf_metadata_from_display_context(final_context, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist)
#             # active_pdf_save_path = active_session_figures_out_path.joinpath(active_pdf_save_filename) # build the final output pdf path from the pdf_parent_out_path (which is the daily folder)

#             # ## BEGIN DISPLAY/SAVE
#             # with backend_pdf.PdfPages(active_pdf_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
#             #     out_fig_list = [] # Separate PDFs mode:

#             #     if debug_print:
#             #         print(f'active_pdf_save_path: {active_pdf_save_path}\nactive_pdf_metadata: {active_pdf_metadata}')
#             #         print(f'active_display_fn_kwargs: {active_display_fn_kwargs}')
                    
#             #     out_display_var = curr_active_pipeline.display(curr_display_function_name, a_filtered_context, **active_display_fn_kwargs) # , filter_epochs='ripple', debug_test_max_num_slices=128
#             #     # , fignum=active_identifying_ctx_string, **figure_format_config
    
#             #     if debug_print:
#             #         print(f'completed display(...) call. type(out_display_var): {type(out_display_var)}\n out_display_var: {out_display_var}, active_display_fn_kwargs: {active_display_fn_kwargs}')

#             #     out_fig_list = extract_figures_from_display_function_output(out_display_var=out_display_var, out_fig_list=out_fig_list)

#             #     if debug_print:
#             #         print(f'out_fig_list: {out_fig_list}')

#             #     # Finally iterate through and do the saving to PDF
#             #     for i, a_fig in enumerate(out_fig_list):
#             #         pdf.savefig(a_fig, transparent=True)
#             #         pdf.attach_note(f'Page {i + 1}: "{active_identifying_ctx_string}"')
                    
#             #     curr_active_pipeline.register_output_file(output_path=active_pdf_save_path, output_metadata={'filtered_context': a_filtered_context, 'context': active_identifying_ctx, 'fig': out_fig_list})








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




@metadata_attributes(short_name=None, tags=['video', 'timeline', 'tracks', 'export', 'mp4'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-11-24 17:40', related_items=[])
class TimelineVideoExporter:
    """ export the timeline out to a video file. Actually just programmatically advances the timeline as if a video were being output, but you have to use an external screen recorder to write the frames to video.
    
    
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import TimelineVideoExporter
    
    
    """

    @classmethod
    def main(cls, active_2d_plot, sync_plotters, win, desired_playback_duration: float = 8 * 60.0, session_start_t: float = 11023.018433333335, desired_framerate: float = 2.0):
        from PyQt5 import QtWidgets, QtGui, QtCore
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from pyqtgraph.exporters import ImageExporter
        from PIL import Image
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.CustomGraphicsLayoutWidget import CustomGraphicsLayoutWidget

        ## "playback" refers to output video:
        
        # "session" refers to the actual recording session:
        desired_session_time_range_duration: float = (16.0 * 60.0) # 1m
        # session_start_t: float = 23170.37047485091 #20740.63578640229 # 11631.186907154472 # 11451.186907154472 # 11391.186907154472 # 7665.232126354053 # active_2d_plot.total_data_start_time + 4.0 * 60.0 # 4 minutes into start of recording ## Day 5
        # session_start_t: float = 11023.018433333335 # 10843.018433333334 # active_2d_plot.total_data_start_time + 4.0 * 60.0 # 4 minutes into start of recording
        # session_start_t: float = 23170.37047485091 #20740.63578640229 # 11631.186907154472 # 11451.186907154472 # 11391.186907154472 # active_2d_plot.total_data_start_time + 4.0 * 60.0 # 4 minutes into start of recording ## day 4
        desired_session_time_range: Tuple[float, float] = (session_start_t, (session_start_t + desired_session_time_range_duration))

        playback_speed_factor: float = (desired_playback_duration / desired_session_time_range_duration)

        print(f'playback_speed_factor: {playback_speed_factor}')
        time_window_duration: float = active_2d_plot.active_window_duration
        print(f'time_window_duration: {time_window_duration}')

        ## INPUTS: _out_container, active_2d_plot, _out_container, sync_plotters, 
        
        desired_frame_duration_sec: float = 1.0/desired_framerate
        print(f'desired_frame_duration_sec: {desired_frame_duration_sec}')

        # ## All Frames from entire recording (too long)
        # total_duration: float = active_2d_plot.total_data_duration
        # desired_num_total_frames: int = int(np.ceil((total_duration * desired_framerate)))
        # frame_start_indicies = np.linspace(active_2d_plot.total_data_start_time,  active_2d_plot.total_data_end_time, num=desired_num_total_frames)

        ## Plot only for the range of interest:
        desired_num_total_frames: int = int(np.ceil((desired_session_time_range_duration * desired_framerate)))
        frame_start_indicies = np.linspace(desired_session_time_range[0], desired_session_time_range[1], num=desired_num_total_frames)
        frame_end_indices = frame_start_indicies + desired_frame_duration_sec

        print(f'desired_num_total_frames: {desired_num_total_frames}')

        # ## Disable debug print to speed up animation
        for a_plotter_name, a_plotter in sync_plotters.items():
            a_plotter.params.debug_print = False
            

        # next_end_timestamp = next_start_timestamp + self.animation_active_time_window.window_duration

        def _frame_update(frame_start_t, frame_end_t):
            active_2d_plot.update_scroll_window_region(frame_start_t, frame_end_t, block_signals=True)
            active_2d_plot.window_scrolled.emit(frame_start_t, frame_end_t)
            QtWidgets.QApplication.processEvents()
            win.repaint()


        _frame_update(desired_session_time_range[0], (desired_session_time_range[0]+time_window_duration))


        for i, (frame_start_t, frame_end_t) in enumerate(zip(frame_start_indicies, frame_end_indices)):
            print(f'frame[{i}]: ({frame_start_t}, {frame_end_t}):')
            _frame_update(frame_start_t, (frame_start_t + time_window_duration))

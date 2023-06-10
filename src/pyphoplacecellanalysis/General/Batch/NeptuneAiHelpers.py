import os
from attrs import define, Factory, field
import neptune # for logging progress and results
from neptune.types import File

import pathlib
from pathlib import Path

from neuropy.utils.result_context import IdentifyingContext

from pyphocorehelpers.function_helpers import function_attributes

"""
if enable_neptune:
        project = neptune.init_project()
        project["general/global_batch_result_filename"] = active_global_batch_result_filename
        project["general/global_data_root_parent_path"] = global_data_root_parent_path.as_posix()

    ## Currently contains an explicit neptune dependency:
    with neptune.init_run() as run:
        if enable_neptune:
            run['parameters/perform_execute'] = perform_execute
            run['parameters/global_batch_result_file_path'] = global_batch_result_file_path.as_posix()
            # project["general/data_analysis"].upload("data_analysis.ipynb")
            run["dataset/latest"].track_files(f"file://{global_batch_result_file_path}") # "s3://datasets/images"

if enable_neptune:
            # Pre-execution dataframe view:
            run["dataset/global_batch_run_progress_df"].upload(File.as_html(global_batch_run.to_dataframe(expand_context=True, good_only=False))) # "path/to/test_preds.csv"
if enable_neptune:
            run["dataset/latest"].track_files(f"file://{global_batch_result_file_path}") # "s3://datasets/images" # update file progress post-load
            # Post-execution dataframe view:
            run["dataset/global_batch_run_progress_df"].upload(File.as_html(global_batch_run.to_dataframe(expand_context=True, good_only=False))) # "path/to/test_preds.csv"
        
if enable_neptune:
        ## POST Run
        project.stop()
"""


def set_environment_variables(neptune_kwargs=None, enable_neptune=True):
    """ sets the environment variables from the neptune_kwargs """
    if enable_neptune:
        if neptune_kwargs is None:
            # set defaults:
            neptune_kwargs = {'project':"commander.pho/PhoDibaLongShort2023",
            'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ=="}

        os.environ["NEPTUNE_API_TOKEN"] = neptune_kwargs['api_token']
        os.environ["NEPTUNE_PROJECT"] = neptune_kwargs['project']



@define()
class Neptuner(object):
    """An object to maintain state for neptune.ai outputs.
      
    from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import Neptuner
    neptuner = Neptuner()
    
    """
    project_name: str="commander.pho/PhoDibaLongShortUpdated"
    api_token: str="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ=="
    project: neptune.Project = field(init=False)
    run: neptune.Run = field(init=False)

    outputs = field(init=False)
    figures = field(init=False)

    def __attrs_post_init__(self):
        self.project = neptune.init_project(project=self.project_name, api_token=self.api_token)
        self.run = None
        
    def stop(self):
        print(f'Neptuner.stop() called!')
        if self.run is not None:
            self.run.stop()
        self.project.stop()
        print(f'Neptuner stoppped.')

    ## The destructor apparently isn't called reliably and sometimes it is called even when the object still exists?                
    # def __del__(self):
    #     # body of destructor
    #     self.stop()
    #     self.run = None
    #     self.project = None


    def run_with_pipeline(self, curr_active_pipeline):
        """ starts a new run with the provided pipeline. """
        assert self.run is None, f"run_new_pipeline(...) called while the Neptuner object already has an active self.run object!"
        if self.run is None:
            self.run = neptune.init_run(project=self.project_name, api_token=self.api_token, source_files=[]) # see git_ref=GitRef(repository_path="/path/to/repo")
            # Add the session_context properties to the run: {'format_name': 'kdiba', 'animal': 'vvp01', 'exper_name': 'two', 'session_name': '2006-4-09_16-40-54'}
            for k, v in curr_active_pipeline.get_session_context().to_dict().items():
                self.run[k] = v # add the properties to the run

            self.run["sys/tags"].add(list(curr_active_pipeline.get_session_context().as_tuple())) # adds the session tags ('kdiba', 'gor01', 'one', '2006-6-09_1-22-43')
            

            # session_descriptor_string: a string describing the context of the session like 'sess_kdiba_2006-6-07_11-26-53'
            self.run['session_descriptor_string'] = curr_active_pipeline.sess.get_description() # 'kdiba_vvp01_two_2006-4-09_16-40-54_sess'
            self.outputs = self.run['outputs']
            self.figures = self.outputs['figures']

    @classmethod
    def init_with_pipeline(cls, curr_active_pipeline):
        """ creates a new Neptuner object corresponding to a particular session and starts a run. """
        new = cls()
        new.run_with_pipeline(curr_active_pipeline=curr_active_pipeline)
        return new


    def upload_figures(self, curr_active_pipeline, should_use_fig_objects=False, debug_print=False):
        """ 
            should_use_fig_objects: bool (False):  if True, the raw matplotlib.Figure objects are uploaded. Otherwise, the previously saved files are uploaded.
        """
        assert not should_use_fig_objects, f"should_use_fig_objects = True is not currently working. Save the figures to file first and then use should_use_fig_objects = False"
        sess_context = curr_active_pipeline.get_session_context()
        succeeded_fig_paths = []
        failed_fig_paths = []

        for a_fig_path, fig_metadata in curr_active_pipeline.registered_output_files.items():
            ## Get list of figures output and register them to neptune:
            a_fig_context: IdentifyingContext = fig_metadata['context']
            a_fig_specific_context: IdentifyingContext = a_fig_context.subtracting(sess_context) # subtract off the common session-level context.
            # a_full_figure_path_key = f"session/{active_session_context_string}/figures/{str(a_fig_context)}"
            a_full_figure_path_key: str = a_fig_specific_context.get_description(separator='/', include_property_names=True, key_value_separator=':')
            if debug_print:
                # print(f'a_fig_specific_context: {a_fig_specific_context}, a_full_figure_path_key: {a_full_figure_path_key}\n\ta_fig_path: {a_fig_path}, a_fig_context: {a_fig_context}\n')
                print(f'Processing: {a_fig_specific_context}, full_key: {a_full_figure_path_key}')

            fig_objects = fig_metadata.get('fig', None)
            if (fig_objects is not None) and should_use_fig_objects:
                # upload the actual figure objects
                try:
                    # Try to extract and upload the actual figure objects
                    if not isinstance(fig_objects, (tuple, list)):
                        fig_objects = (fig_objects,) # wrap in a tuple or convert list to tuple
                    elif isinstance(fig_objects, list):
                        fig_objects = tuple(fig_objects) # convert the list to a tuple
                        
                    # now fig_objects better be a tuple
                    assert isinstance(fig_objects, tuple), f"{fig_objects} should be a tuple!"

                    if len(fig_objects) == 1:
                        # single element, use .upload(...)
                        a_fig_obj = fig_objects[0]
                        self.figures[a_full_figure_path_key].upload(a_fig_obj)
                    else:
                        # otherwise multi-figure, use append for a series list:
                        for a_fig_obj in fig_objects:
                            # also ,  step=epoch for custom index
                            self.figures[a_full_figure_path_key].append(a_fig_obj) #, name=f"{a_full_figure_path_key}", description=f"{a_full_figure_path_key}" append the whole series of related figures
                            
                    if debug_print:
                        print(f'\tSuccess: "{a_full_figure_path_key}": {len(fig_objects)} uploaded!')
                    succeeded_fig_paths.append(a_fig_path)
                    
                except Exception as e:
                    if debug_print:
                        print(f'\tSkipping "{a_full_figure_path_key}": fig_objects found but raised exception {e}.')
                    failed_fig_paths.append(a_fig_path)
                    # raise e		
            else:
                if should_use_fig_objects:
                    if debug_print:
                        print(f'\tSkipping "{a_full_figure_path_key}": No figure objects found')
                    failed_fig_paths.append(a_fig_path)
                else:
                    # this is the mode we expected -- upload as file
                    try:
                        self.figures[a_full_figure_path_key].upload(File(str(a_fig_path)))
                        succeeded_fig_paths.append(a_fig_path)
                        if debug_print:
                            print(f'\tSuccess: "{a_full_figure_path_key}": file at path {str(a_fig_path)} uploaded!')
                    except Exception as e:
                        raise e
                    
        # end for loop
        return succeeded_fig_paths, failed_fig_paths



# ==================================================================================================================== #
# Independent Helper Functions                                                                                         #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['neptune', 'figures', 'output', 'cloud', 'logging', 'pipeline'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-25 04:34', related_items=[])
def neptune_output_figures(curr_active_pipeline):
    """ Uploads the completed figures to neptune.ai from the pipeline's `.registered_output_files` items. 
    
    Usage:
        from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
        neptune_output_figures(curr_active_pipeline)
    """
    import neptune # for logging progress and results
    from neptune.types import File

    neptune_kwargs = {'project':"commander.pho/PhoDibaLongShort2023",
    'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ=="}

    # kdiba_gor01_one_2006-6-09_1-22-43_sess
    active_session_context_string = str(curr_active_pipeline.active_sess_config)

    with neptune.init_run(**neptune_kwargs) as run:
        for a_fig_path, fig_metadata in curr_active_pipeline.registered_output_files.items():
            ## Get list of figures output and register them to neptune:
            a_fig_context = fig_metadata['context']
            a_full_figure_path_key = f"session/{active_session_context_string}/figures/{str(a_fig_context)}"
            print(f'a_fig_path: {a_fig_path}, a_fig_context: {a_fig_context}\n\t{a_full_figure_path_key}')
            # fig_objects = fig_metadata.get('fig', None)
            # if fig_objects is not None:
            # 	# upload the actual figure objects
            # 	if not isinstance(fig_objects, (tuple, list)):
            # 		fig_objects = (fig_objects,) # wrap in a tuple or convert list to tuple
            # 	# now fig_objects better be a tuple
            # 	assert isinstance(fig_objects, tuple), f"{fig_objects} should be a tuple!"
            # 	# run[a_full_figure_path_key].upload(fig_metadata['fig'])
            # 	for a_fig_obj in fig_objects:
            # 		run[a_full_figure_path_key].append(a_fig_obj) # append the whole series of related figures
            # else:
            # upload as file
            run[a_full_figure_path_key].upload(str(a_fig_path))
from pathlib import Path
import pathlib
import numpy as np
import pandas as pd
from attrs import define, field 

from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows


# Find all previously computed ripple data:
def find_externally_computed_ripple_files(project_path, exclude_dirs=[]):
    # Find all .py files in the project directory and its subdirectories
    if not isinstance(project_path, pathlib.Path):
        project_path = pathlib.Path(project_path)
    py_files = project_path.glob("**/ripple.pkl")
    py_files = [file_path for file_path in py_files] # to list

    excluded_py_files = []
    if exclude_dirs is not None:
        # Find all .py files in the project directory and its subdirectories, excluding the 'my_exclude_dir' directory
        exclude_paths = [project_path.joinpath(a_dir) for a_dir in exclude_dirs]
        for an_exclude_path in exclude_paths:
            excluded_py_files.extend([file_path for file_path in an_exclude_path.glob("**/*.py")])

    included_py_files = [x for x in py_files if x not in excluded_py_files]
    return included_py_files


def find_externally_computed_session_h5_files(project_path, exclude_dirs=[]):
    # Find all .py files in the project directory and its subdirectories
    if not isinstance(project_path, pathlib.Path):
        project_path = pathlib.Path(project_path)
    # py_files = project_path.glob("**/output/global_computations.h5")
    py_files = project_path.glob("**/output/pipeline_results.h5")
    py_files = [file_path for file_path in py_files] # to list

    excluded_py_files = []
    if exclude_dirs is not None:
        # Find all .py files in the project directory and its subdirectories, excluding the 'my_exclude_dir' directory
        exclude_paths = [project_path.joinpath(a_dir) for a_dir in exclude_dirs]
        for an_exclude_path in exclude_paths:
            excluded_py_files.extend([file_path for file_path in an_exclude_path.glob("**/*.py")])

    included_py_files = [x for x in py_files if x not in excluded_py_files]
    return included_py_files
# # 2023-07-06 - New Ideas and External Ripple Detection Progress
# @define()
# class PipelineDataModificationRecipie:
#     """
#     # Define a format for saving a specific modification to the current pipeline:

#     # Requires a criteria for matching the pipeline (to prevent being loaded into the wrong pipeline):

#     # A method that defines how the data is loaded from disk into the pipeline (that takes the pipeline as an argument and modifies it in in place).

    
#     Conceptualyl you can save a bunch of modifications or results to teh session directory, adn then add them specifically instead of computing/etc.
#     """
#     def modify_pipeline(self, curr_active_pipeline):
#         curr_active_pipeline.global_data.a_new_value = self.a_loaded_value
#         curr_active_pipeline.computed_data['maze1'].another_new_value = self.another_loaded_value['maze1']
        
#         raise NotImplementedError()
    
# # Formula for adding a new row to the batch result dataframe after doing some processing:
# # e.g.: for the list of all sessions, add a new column with whether the external ripple generation result is present for that sessions (found by searching that session's folder for the file. 




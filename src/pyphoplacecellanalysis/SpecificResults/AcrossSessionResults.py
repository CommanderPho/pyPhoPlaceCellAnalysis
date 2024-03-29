"""
Concerned with aggregating data (raw and computed results) across multiple sessions.
    Previously (Pre 2023-07-31) everything was designed in terms of a single session: the entire `NeuropyPipeline` object represents a single recording session - although it might slice this session several different ways and process it with several different sets of computation parameters
    
    All NeuropyPipelines (each session) is designed to be processed completely independently from each other with no codependencies. This enables trivial parallel processing of each session and complete encapsulation of the loogic for that session.

    As a result of this design decision, anything that aims to compute statistics aggregated across multiple sessions or combine/compare values between sessions must be implemented here.
"""


""" DESIGN DECISION/CONSTRAINT: This file should not focus on the process of directing the individual session pipeline computations (it's not a parallel processing manager) but instead on processing the data with a specified set of completed session pipelines.

"""

import sys
import os
import re
import pathlib
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Tuple
import numpy as np
import pandas as pd
from copy import deepcopy
from attrs import define, field, Factory
import shutil # copy_files_in_filelist_to_dest
from pyphocorehelpers.exception_helpers import CapturedException
import tables as tb
from tables import (
    Group, Int8Col, Int16Col, Int32Col, Int64Col, NoSuchNodeError,
    UInt8Col, UInt16Col, UInt32Col, UInt64Col,
    Float32Col, Float64Col,
    TimeCol, ComplexCol, StringCol, BoolCol, EnumCol
)
import seaborn as sns
# from pyphocorehelpers.indexing_helpers import partition, safe_pandas_get_group
from datetime import datetime

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path, set_posix_windows, convert_filelist_to_new_parent, find_matching_parent_path
from pyphocorehelpers.function_helpers import function_attributes

# NeuroPy (Diba Lab Python Repo) Loading
## For computation parameters:
from neuropy.core.epoch import Epoch
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths
from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update
from neuropy.core.neuron_identities import NeuronExtendedIdentityTuple, neuronTypesEnum, NeuronIdentityTable
from neuropy.utils.mixins.HDF5_representable import HDF_Converter

from neuropy.utils.mixins.AttrsClassHelpers import custom_define, AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field
# from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata, get_file_metadata


from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData

# from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PaperFigureTwo # for `BatchSessionCompletionHandler`
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation # for `BatchSessionCompletionHandler`, `AcrossSessionsAggregator`
from neuropy.core.user_annotations import UserAnnotationsManager
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode, build_and_write_to_file

from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots


"""
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults, AcrossSessionsVisualizations

"""

trackMembershipTypesList: List[str] = ['long_only', 'short_only', 'both', 'neither']
trackMembershipTypesEnum = tb.Enum(trackMembershipTypesList)
trackExclusiveToMembershipTypeDict: Dict = dict(zip(['LxC', 'SxC', 'shared', 'neither'], trackMembershipTypesList))
trackExclusiveToMembershipTypeReverseDict: Dict = dict(zip(trackMembershipTypesList, ['LxC', 'SxC', 'shared', 'neither'])) # inverse of `trackExclusiveToMembershipTypeDict`



class FiringRatesDeltaTable(tb.IsDescription):
    delta_minus = tb.Float64Col()
    delta_plus = tb.Float64Col()
    



# @pd.api.extensions.register_dataframe_accessor("inst_fr_results")
class InstantaneousFiringRatesDataframeAccessor():
    """ A Pandas pd.DataFrame representation of results from the batch processing of sessions
    # 2023-07-07
    Built from `BatchRun`
    
    Used for FigureTwo: the cross-session scatter plot of firing rates during laps v replays for the LxCs vs. SxCs.

    """

    # _required_column_names = ['session_name', 'basedirs', 'status', 'errors']
    _required_column_names = ['context', 'basedirs', 'status', 'errors']

    # ==================================================================================================================== #
    # ScatterPlotResultsTable                                                                                              #
    # ==================================================================================================================== #
    
    class ScatterPlotResultsTable(tb.IsDescription):
        """ """        
        neuron_identity = NeuronIdentityTable()
        
        lap_firing_rates_delta = FiringRatesDeltaTable()
        replay_firing_rates_delta = FiringRatesDeltaTable()
        
        active_set_membership = EnumCol(trackMembershipTypesEnum, 'neither', base='uint8')


    @classmethod
    def scatter_plot_results_table_to_hdf(cls, file_path, result_df: pd.DataFrame, file_mode='a'):
        """ writes the table to a .h5 file at the specified file path


        common_file_path = Path('output/test_across_session_scatter_plot.h5')
        print(f'common_file_path: {common_file_path}')
        AcrossSessionsResults.scatter_plot_results_table_to_hdf(file_path=common_file_path, result_df=result_df, file_mode='w')

        """
        with tb.open_file(file_path, mode=file_mode) as file:    
            # Check if the table exists
            # if file.root.has_node('ScatterPlotResults'):
            if 'ScatterPlotResults' in file.root:
                table = file.root.ScatterPlotResults
                # The table exists; you can now append to it
            else:
                # The table doesn't exist; you can create it
                table = file.create_table('/', 'ScatterPlotResults', cls.ScatterPlotResultsTable)



            # Serialization
            row = table.row
            for i in np.arange(len(result_df)):
                row_data = result_df.iloc[i]
                
                session_uid: str = f"|".join([row_data['format_name'], row_data['animal'], row_data['exper_name'], row_data['session_name']])
                
                # NeuronIdentityTable
                row['neuron_identity/neuron_uid'] = f"{session_uid}|{row_data['aclu']}"
                row['neuron_identity/session_uid'] = session_uid
                row['neuron_identity/neuron_id'] = row_data['aclu']
                row['neuron_identity/neuron_type'] = neuronTypesEnum[row_data['neuron_type'].hdfcodingClassName]
                row['neuron_identity/shank_index'] = row_data['shank']
                row['neuron_identity/cluster_index'] = row_data['cluster']
                row['neuron_identity/qclu'] = row_data['qclu']

                # # LapFiringRatesDeltaTable
                row['lap_firing_rates_delta/delta_minus'] = row_data['lap_delta_minus']
                row['lap_firing_rates_delta/delta_plus'] = row_data['lap_delta_plus']

                # # ReplayFiringRatesDeltaTable
                row['replay_firing_rates_delta/delta_minus'] = row_data['replay_delta_minus']
                row['replay_firing_rates_delta/delta_plus'] = row_data['replay_delta_plus']

                # active_set_membership
                row['active_set_membership'] = trackMembershipTypesEnum[trackExclusiveToMembershipTypeDict[row_data['active_set_membership']]]
                
                row.append()
                
            table.flush()
            
    @classmethod
    def read_scatter_plot_results_table(cls, file_path) -> pd.DataFrame:
        """ the reciprocal operation to `scatter_plot_results_table_to_hdf(..)`. Reads the table from file to produce a dataframe.
        
        common_file_path = Path('output/test_across_session_scatter_plot.h5')
        print(f'common_file_path: {common_file_path}')
        loaded_result_df = AcrossSessionsResults.read_scatter_plot_results_table(file_path=common_file_path)
        
        
        """
        with tb.open_file(file_path, mode='r') as file:
            table = file.root.ScatterPlotResults

            data = []
            for row in table.iterrows():
                neuron_uid = row['neuron_identity/neuron_uid'].decode()
                session_uid = row['neuron_identity/session_uid'].decode()
                session_uid_parts = session_uid.split("|")
                # global_uid_parts = neuron_uid.split("|")
                # print(f'neuron_uid: {neuron_uid}, global_uid_parts: {global_uid_parts}')
            
                # neuron_uid, session_uid, neuron_id, neuron_type, shank_index, cluster_index, qclu = neuron_identity
                
                row_data = {
                    'neuron_uid': neuron_uid,
                    'format_name': session_uid_parts[0],
                    'animal': session_uid_parts[1],
                    'exper_name': session_uid_parts[2],
                    'session_name': session_uid_parts[3],
                    'aclu': row['neuron_identity/neuron_id'],
                    'shank': row['neuron_identity/shank_index'],
                    'cluster': row['neuron_identity/cluster_index'],
                    'qclu': row['neuron_identity/qclu'],
                    # 'neuron_type': neuronTypesEnum(row['neuron_identity/neuron_type']).hdfcodingClassName, # Assuming reverse mapping is available
                    # 'active_set_membership': trackMembershipTypesEnum(row['active_set_membership']).name, # Assuming reverse mapping is available
                    'neuron_type': neuronTypesEnum(row['neuron_identity/neuron_type']),
                    'active_set_membership': trackExclusiveToMembershipTypeReverseDict[trackMembershipTypesEnum(row['active_set_membership'])], # Assuming reverse mapping is available
                    'lap_delta_minus': row['lap_firing_rates_delta/delta_minus'],
                    'lap_delta_plus': row['lap_firing_rates_delta/delta_plus'],
                    'replay_delta_minus': row['replay_firing_rates_delta/delta_minus'],
                    'replay_delta_plus': row['replay_firing_rates_delta/delta_plus'],
                }
                data.append(row_data)

            loaded_result_df = pd.DataFrame(data)
            
        return loaded_result_df


    @classmethod
    def add_results_to_inst_fr_results_table(cls, curr_active_pipeline, common_file_path, file_mode='w') -> bool:
        """ computes the InstantaneousSpikeRateGroupsComputation needed for FigureTwo and serializes it out to an HDF file.
        Our final output table will be indexed by unique cells but the `InstantaneousSpikeRateGroupsComputation` data structure is currently organized by graph results.

        Usage:
            ## Specify the output file:
            common_file_path = Path('output/test_across_session_scatter_plot_new.h5')
            print(f'common_file_path: {common_file_path}')
            InstantaneousFiringRatesDataframeAccessor.add_results_to_inst_fr_results_table(curr_active_pipeline, common_file_path)

        """
        curr_session_context = curr_active_pipeline.get_session_context() 

        try:
            print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
            _out_inst_fr_comps = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01) # 10ms
            _out_inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_session_context)
            
            ## Build the Output Dataframe:
            cell_firing_rate_summary_df: pd.DataFrame = _out_inst_fr_comps.get_summary_dataframe() # Returns the dataframe with columns ['aclu', 'lap_delta_minus', 'lap_delta_plus', 'replay_delta_minus', 'replay_delta_plus', 'active_set_membership']

            # Get the aclu information for each aclu in the dataframe. Adds the ['aclu', 'shank', 'cluster', 'qclu', 'neuron_type'] columns
            # unique_aclu_information_df: pd.DataFrame = curr_active_pipeline.sess.spikes_df.spikes.extract_unique_neuron_identities()
            unique_aclu_information_df: pd.DataFrame = curr_active_pipeline.get_session_unique_aclu_information()

            # Horizontally join (merge) the dataframes
            result_df: pd.DataFrame = pd.merge(unique_aclu_information_df, cell_firing_rate_summary_df, left_on='aclu', right_on='aclu', how='inner')

            # Add this session context columns for each entry: creates the columns ['format_name', 'animal', 'exper_name', 'session_name']
            result_df[curr_session_context._get_session_context_keys()] = curr_session_context.as_tuple()

            # Reordering the columns to place the new columns on the left
            result_df = result_df[['format_name', 'animal', 'exper_name', 'session_name', 'aclu', 'shank', 'cluster', 'qclu', 'neuron_type', 'active_set_membership', 'lap_delta_minus', 'lap_delta_plus', 'replay_delta_minus', 'replay_delta_plus']]
            
            cls.scatter_plot_results_table_to_hdf(file_path=common_file_path, result_df=result_df, file_mode=file_mode)

            print(f'\t\t done (success).') 
            return True

        except Exception as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {e} while trying to compute the instantaneous firing rates and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
            _out_inst_fr_comps = None
            return False


    @classmethod
    def load_and_prepare_for_plot(cls, common_file_path) -> Tuple[InstantaneousSpikeRateGroupsComputation, pd.DataFrame]:
        """ loads the previously saved out inst_fr_scatter_plot_results_table and prepares it for plotting. 

        returns a `InstantaneousSpikeRateGroupsComputation` _shell_obj which can be plotted
        
        Usage:        
            _shell_obj, loaded_result_df = InstantaneousFiringRatesDataframeAccessor.load_and_prepare_for_plot(common_file_path)
            # Perform the actual plotting:
            AcrossSessionsVisualizations.across_sessions_bar_graphs(_shell_obj, num_sessions=1, save_figure=False, enable_tiny_point_labels=False, enable_hover_labels=False)
        
        """
        from matplotlib.markers import MarkerStyle
        from matplotlib.lines import Line2D
        from matplotlib.transforms import Affine2D

        ## Read the previously saved-out result:
        loaded_result_df = cls.read_scatter_plot_results_table(file_path=common_file_path)
        
        ## Scatter props:
        def build_unique_colors_mapping_for_column(df, column_name:str):
            # Get unique values and map them to colors
            unique_values = df[column_name].unique()
            colors = sns.color_palette('husl', n_colors=len(unique_values)) # Using seaborn to get a set of distinct colors
            # Create a mapping from unique values to colors
            color_mapping = {value: color for value, color in zip(unique_values, colors)}
            return color_mapping

        def build_unique_markers_mapping_for_column(df, column_name:str):
            # Get unique values and map them to colors
            unique_values = df[column_name].unique()
            
            pho_custom_allowed_filled_markers_list = ['o','^','8','s','p','d','P','X'] # all of these filled markers were chosen because they look like they represent similar quantities (they are the approx. same size and area.
            
            # marker_list = [(5, i) for i in np.arange(len(unique_values))] # [(5, 0), (5, 1), (5, 2)]
            marker_list = [pho_custom_allowed_filled_markers_list[i] for i in np.arange(len(unique_values))] # Each marker is of the form: (numsides, 1, angle)
            # Create a mapping from unique values to colors
            marker_mapping = {value: a_marker for value, a_marker in zip(unique_values, marker_list)}
            return marker_mapping

        scatter_props_column_names = ['color', 'marker']
        # column_name_to_colorize:str = 'session_name'
        column_name_to_colorize:str = 'qclu'
        color_mapping = build_unique_colors_mapping_for_column(loaded_result_df, column_name_to_colorize)
        # Apply the mapping to the 'property' column to create a new 'color' column
        loaded_result_df['color'] = loaded_result_df[column_name_to_colorize].map(color_mapping)

        column_name_to_markerize:str = 'animal'
        marker_mapping =  build_unique_markers_mapping_for_column(loaded_result_df, column_name_to_markerize)
        loaded_result_df['marker'] = loaded_result_df[column_name_to_markerize].map(marker_mapping)

        # build the final 'scatter_props' column
        # loaded_result_df['scatter_props'] = [{'edgecolor': a_color, 'marker': a_marker} for a_color, a_marker in zip(loaded_result_df['color'], loaded_result_df['marker'])]
        loaded_result_df['scatter_props'] = [{'marker': a_marker} for a_color, a_marker in zip(loaded_result_df['color'], loaded_result_df['marker'])]


        # For `loaded_result_df`, to recover the plottable FigureTwo points:
        table_columns = ['neuron_uid', 'aclu', 'lap_delta_minus', 'lap_delta_plus', 'replay_delta_minus', 'replay_delta_plus', 'active_set_membership']
        # 1. Group by 'active_set_membership' (to get LxC and SxC groups which are processed separately)

        # loaded_result_df.groupby('active_set_membership')
        # 2. FigureTwo_a uses the lap_* columns and FigureTwo_b uses the replay_* columns

        # 3. Compute the mean and error bars for each of the four columns
        data_columns = ['lap_delta_minus', 'lap_delta_plus', 'replay_delta_minus', 'replay_delta_plus']  

        grouped_df = loaded_result_df.groupby(['active_set_membership'])
        LxC_df, SxC_df = [grouped_df.get_group(aValue) for aValue in ['LxC','SxC']] # Note that in general LxC and SxC might have differing numbers of cells.

        #TODO 2023-08-11 02:09: - [ ] These LxC/SxC_aclus need to be globally unique probably.
        # LxC_aclus = LxC_df.aclu.values
        # SxC_aclus = SxC_df.aclu.values
        LxC_aclus = LxC_df.neuron_uid.values
        SxC_aclus = SxC_df.neuron_uid.values
        # The arguments should be determined by the neuron information or the session, etc. Let's color based on session here.

        # LxC_scatter_props = [{'edgecolor': a_color, 'marker': a_marker} for a_color, a_marker in zip(LxC_df['color'], LxC_df['marker'])]
        # SxC_scatter_props = [{'edgecolor': a_color, 'marker': a_marker} for a_color, a_marker in zip(SxC_df['color'], SxC_df['marker'])]

        ## Hardcoded-override here:
        LxC_scatter_props = [{'alpha': 0.5, 'edgecolors':'black', 'linewidths':1, 'marker':a_marker} for a_color, a_marker in zip(LxC_df['color'], LxC_df['marker'])]
        SxC_scatter_props = [{'alpha': 0.5, 'edgecolors':'black', 'linewidths':1, 'marker':a_marker} for a_color, a_marker in zip(SxC_df['color'], SxC_df['marker'])] # a_marker, 's':80

        # , markeredgecolor="orange", markeredgewidth=5

        # LxC_scatter_props = LxC_df['scatter_props'].values
        # SxC_scatter_props = SxC_df['scatter_props'].values

        # ## Empty scatter_props
        # LxC_scatter_props = [{} for a_color, a_marker in zip(LxC_df['color'], LxC_df['marker'])]
        # SxC_scatter_props = [{} for a_color, a_marker in zip(SxC_df['color'], SxC_df['marker'])]

        ## Convert back to `InstantaneousSpikeRateGroupsComputation`'s language:
        Fig2_Laps_FR: list[SingleBarResult] = [SingleBarResult(v.mean(), v.std(), v, LxC_aclus, SxC_aclus, LxC_scatter_props, SxC_scatter_props) for v in (LxC_df['lap_delta_minus'].values, LxC_df['lap_delta_plus'].values, SxC_df['lap_delta_minus'].values, SxC_df['lap_delta_plus'].values)]
        Fig2_Replay_FR: list[SingleBarResult] = [SingleBarResult(v.mean(), v.std(), v, LxC_aclus, SxC_aclus, LxC_scatter_props, SxC_scatter_props) for v in (LxC_df['replay_delta_minus'].values, LxC_df['replay_delta_plus'].values, SxC_df['replay_delta_minus'].values, SxC_df['replay_delta_plus'].values)]

        _shell_obj = InstantaneousSpikeRateGroupsComputation()
        _shell_obj.Fig2_Laps_FR = Fig2_Laps_FR
        _shell_obj.Fig2_Replay_FR = Fig2_Replay_FR
        _shell_obj.LxC_aclus = LxC_aclus
        _shell_obj.SxC_aclus = SxC_aclus
        # _shell_obj.LxC_scatter_props = LxC_scatter_props
        # _shell_obj.SxC_scatter_props = SxC_scatter_props

        return _shell_obj, loaded_result_df





from neuropy.core.user_annotations import UserAnnotationsManager, SessionCellExclusivityRecord
from neuropy.utils.result_context import IdentifyingContext
from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_perform_diagonal_line_binomial_test, pho_stats_bar_graph_t_tests


from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FigureOutputLocation, ContextToPathMode, FileOutputManager # used in post_compute_all_sessions_processing
from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PaperFigureTwo # used in post_compute_all_sessions_processing


class AcrossSessionsResults:
    """ 

    Holds a reference to a centralized HDF5 file and a way of registering entries into it.
    
    Entries consist of:
        - AcrossSessionAggregating-level Results (such as those used in PhoDiba2023Paper
        - Links or References
     
    Batch Processing goes like:
    1. Discover Sessions
    2. Load the Session Data to gain access to the pipeline
    
    an_active_pipeline: NeuropyPipeline = all_sessions.get_pipeline(context: a_session_context)
    
        
    """
    


    #TODO 2023-08-10 21:34: - [ ] Ready to accumulate results!

 
    class ProcessedSessionResultsTable(tb.IsDescription):
        """ represents a single session's processing results in the scope of multiple sessions for use in a PyTables table or HDF5 output file """
        neuron_uid = StringCol(68)   # 16-character String, globally unique neuron identifier (across all sessions) composed of a session_uid and the neuron's (session-specific) aclu
        session_uid = StringCol(64)
        neuron_identities = NeuronIdentityTable()

        class LongShortNeuronComputedPropertiesTable(tb.IsDescription):
            """ Represents computed properties for a single neuron """
            long_pf_peak_x = tb.Float64Col()
            has_long_pf = tb.BoolCol()
            short_pf_peak_x = tb.Float64Col()
            has_short_pf = tb.BoolCol()
            has_na = tb.BoolCol()
            track_membership = EnumCol(trackMembershipTypesEnum, 'neither', base='uint8') 
            long_non_replay_mean = tb.Float64Col()
            short_non_replay_mean = tb.Float64Col()
            non_replay_diff = tb.Float64Col()
            long_replay_mean = tb.Float64Col()
            short_replay_mean = tb.Float64Col()
            replay_diff = tb.Float64Col()
            long_mean = tb.Float64Col()
            short_mean = tb.Float64Col()
            mean_diff = tb.Float64Col()
            neuron_IDX = tb.Int64Col()
            num_replays = tb.Int32Col()
            long_num_replays = tb.Int32Col()
            short_num_replays = tb.Int32Col()
            neuron_type = tb.StringCol(itemsize=50)  # Adjust 'itemsize' based on your maximum string length
        



        # class GlobalComputationsTable(tb.IsDescription):
        #     """ represents a single session's processing results in the scope of multiple sessions for use in a PyTables table or HDF5 output file """
        #     session_uid = StringCol(32) # globally unique session identifier (across all sessions)
        #     format_name = StringCol(16)
        #     animal = StringCol(16)
        #     exper_name  = StringCol(32)
        #     session_name  = StringCol(32)
    

    # ==================================================================================================================== #
    # NeuronIdentityTable                                                                                                  #
    # ==================================================================================================================== #
    
    ## This seems definitionally a single-session result! It can be concatenated across sessions to make a multi-session one though!
    @classmethod
    def build_neuron_identity_table_to_hdf(cls, file_path, key: str, spikes_df: pd.DataFrame, session_uid:str="test_session_uid"):
        """ extracts a NeuronIdentityTable from the complete session spikes_df """
        unique_rows_df = spikes_df.spikes.extract_unique_neuron_identities()
        # Extract the selected columns as NumPy arrays
        aclu_array = unique_rows_df['aclu'].values
        shank_array = unique_rows_df['shank'].values
        cluster_array = unique_rows_df['cluster'].values
        qclu_array = unique_rows_df['qclu'].values
        neuron_type_array = unique_rows_df['neuron_type'].values
        neuron_types_enum_array = np.array([neuronTypesEnum[a_type.hdfcodingClassName] for a_type in neuron_type_array]) # convert NeuronTypes to neuronTypesEnum
        n_neurons = len(aclu_array)
        
        # Open the file with h5py to add attributes to the group. The pandas.HDFStore object doesn't provide a direct way to manipulate groups as objects, as it is primarily intended to work with datasets (i.e., pandas DataFrames)
        # with h5py.File(file_path, 'r+') as f:
        with tb.open_file(file_path, mode='a') as f:
            
            # f.create_dataset(f'{key}/neuron_ids', data=self.neuron_ids)
            # f.create_dataset(f'{key}/shank_ids', data=self.shank_ids)
            group = f.create_group(key, 'neuron_identities', title='each row uniquely identifies a neuron and its various loaded, labeled, and computed properties', createparents=True)

            table = f.create_table(group, 'table', NeuronIdentityTable, "Neuron identities")

            # Serialization
            row = table.row
            for i in np.arange(n_neurons):
                ## Build the row here from aclu_array, etc
                row['neuron_uid'] = f"{session_uid}|{aclu_array[i]}"
                row['session_uid'] = session_uid  # Provide an appropriate session identifier here
                row['neuron_id'] = aclu_array[i]
                row['neuron_type'] = neuron_types_enum_array[i]
                row['shank_index'] = shank_array[i]
                row['cluster_index'] = cluster_array[i] # self.peak_channels[i]
                row['qclu'] = qclu_array[i]  # Replace with appropriate value if available                
                row.append()
                
            table.flush()
            
            # Metadata:
            # NOTE: group objects must use `_v_attrs` instead of `attrs` to set their attributes
            group._v_attrs['n_neurons'] = n_neurons
            group._v_attrs['session_uid'] = session_uid
            # group.attrs['dat_sampling_rate'] = self.sampling_rate
            # group.attrs['t_start'] = self.t_start
            # group.attrs['t_start'] = self.t_start
            # group.attrs['t_stop'] = self.t_stop
            


    @classmethod
    def build_session_pipeline_to_hdf(cls, file_path, key: str, curr_active_pipeline, debug_print=False):
        """ Saves out the entire session pipeline (corresponding for all processing on a single session) out to an HDF5 readable format. 

        """
        if debug_print:
            print(f'file_path: {file_path}')
        return curr_active_pipeline.to_hdf(file_path=file_path, key=key)



    @classmethod
    def post_compute_all_sessions_processing(cls, global_data_root_parent_path:Path, BATCH_DATE_TO_USE: str, plotting_enabled:bool):
        """ 2023-11-15 - called after batch computing all of the sessions and building the required output files. Loads them, processes them, and then plots them! 
        
        """
        # 2023-10-04 - Load Saved across-sessions-data and testing Batch-computed inst_firing_rates:
        ## Load the saved across-session results:
        inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{BATCH_DATE_TO_USE}.pkl'
        across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list = AcrossSessionsResults.load_across_sessions_data(global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=inst_fr_output_filename)
        # across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path)
        num_sessions = len(across_sessions_instantaneous_fr_dict)
        print(f'num_sessions: {num_sessions}')

        ## Load all across-session tables from the pickles:
        output_path_suffix: str = f'{BATCH_DATE_TO_USE}'
        neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.load_all_combined_tables(override_output_parent_path=global_data_root_parent_path, output_path_suffix=output_path_suffix) # output_path_suffix=f'2023-10-04-GL-Recomp'
        num_sessions = len(neuron_replay_stats_table.session_uid.unique().to_numpy())
        print(f'num_sessions: {num_sessions}')


        # Does its own additions to `long_short_fr_indicies_analysis_table` table based on the user labeled LxC/SxCs
        annotation_man = UserAnnotationsManager()
        # Hardcoded included_session_contexts:
        included_session_contexts = annotation_man.get_hardcoded_good_sessions()
        
        LxC_uids = []
        SxC_uids = []

        for a_ctxt in included_session_contexts:
            session_uid = a_ctxt.get_description(separator="|", include_property_names=False)
            session_cell_exclusivity: SessionCellExclusivityRecord = annotation_man.annotations[a_ctxt].get('session_cell_exclusivity', None)
            LxC_uids.extend([f"{session_uid}|{aclu}" for aclu in session_cell_exclusivity.LxC])
            SxC_uids.extend([f"{session_uid}|{aclu}" for aclu in session_cell_exclusivity.SxC])

        # [a_ctxt.get_description(separator="|", include_property_names=False) for a_ctxt in included_session_contexts]

        long_short_fr_indicies_analysis_table['XxC_status'] = 'Shared'
        long_short_fr_indicies_analysis_table.loc[np.isin(long_short_fr_indicies_analysis_table.neuron_uid, LxC_uids), 'XxC_status'] = 'LxC'
        long_short_fr_indicies_analysis_table.loc[np.isin(long_short_fr_indicies_analysis_table.neuron_uid, SxC_uids), 'XxC_status'] = 'SxC'

        ## 2023-10-11 - Get the long peak location
        long_short_fr_indicies_analysis_table['long_pf_peak_x'] = neuron_replay_stats_table['long_pf_peak_x']
        # long_short_fr_indicies_analysis_table

        # long_short_fr_indicies_analysis_table_filename = 'output/2023-10-07_long_short_fr_indicies_analysis_table.csv'
        long_short_fr_indicies_analysis_table_filename: str = 'output/{BATCH_DATE_TO_USE}_long_short_fr_indicies_analysis_table.csv'
        long_short_fr_indicies_analysis_table.to_csv(long_short_fr_indicies_analysis_table_filename)
        print(f'saved: {long_short_fr_indicies_analysis_table_filename}')



        # 2023-10-10 - Statistics for `across_sessions_bar_graphs`, analysing `across_session_inst_fr_computation` 
        binom_test_chance_result = pho_stats_perform_diagonal_line_binomial_test(long_short_fr_indicies_analysis_table)
        print(f'binom_test_chance_result: {binom_test_chance_result}')

        LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result = pho_stats_bar_graph_t_tests(across_session_inst_fr_computation)


        ## Plotting:
        graphics_output_dict = {}
        if plotting_enabled:
            matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
            
            long_short_fr_indicies_analysis_table.plot.scatter(x='long_pf_peak_x', y='x_frs_index', title='Pf Peak position vs. LapsFRI', ylabel='Lap FRI')
            long_short_fr_indicies_analysis_table.plot.scatter(x='long_pf_peak_x', y='y_frs_index', title='Pf Peak position vs. ReplayFRI', ylabel='Replay FRI')

            ## 2023-10-04 - Run `AcrossSessionsVisualizations` corresponding to the PhoDibaPaper2023 figures for all sessions
            ## Hacks the `PaperFigureTwo` and `InstantaneousSpikeRateGroupsComputation`
            global_multi_session_context, _out_aggregate_fig_2 = AcrossSessionsVisualizations.across_sessions_bar_graphs(across_session_inst_fr_computation, num_sessions, enable_tiny_point_labels=False, enable_hover_labels=False)

            graphics_output_dict |= AcrossSessionsVisualizations.across_sessions_firing_rate_index_figure(long_short_fr_indicies_analysis_results=long_short_fr_indicies_analysis_table, num_sessions=num_sessions, save_figure=True)

            graphics_output_dict |= AcrossSessionsVisualizations.across_sessions_long_and_short_firing_rate_replays_v_laps_figure(neuron_replay_stats_table=neuron_replay_stats_table, num_sessions=num_sessions, save_figure=True)


            # ## Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions.
            # global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.


        return graphics_output_dict
    

    # ==================================================================================================================== #
    # Old (Pre 2023-07-30 Rewrite)                                                                                         #
    # ==================================================================================================================== #

    # Across Sessions Helpers
    @classmethod
    def save_across_sessions_data(cls, across_sessions_instantaneous_fr_dict, global_data_root_parent_path:Path, inst_fr_output_filename:str='across_session_result_long_short_inst_firing_rate.pkl'):
        """ Save the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation) 
        
        Saves the `self.across_sessions_instantaneous_fr_dict`
        """
        global_batch_result_inst_fr_file_path = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
        print(f'global_batch_result_inst_fr_file_path: {global_batch_result_inst_fr_file_path}')
        # Save the all sessions instantaneous firing rate dict to the path:
        saveData(global_batch_result_inst_fr_file_path, across_sessions_instantaneous_fr_dict)

    @classmethod
    def load_across_sessions_data(cls, global_data_root_parent_path:Path, inst_fr_output_filename:str='across_session_result_long_short_inst_firing_rate.pkl'):
        """ Load the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation) 

            To correctly aggregate results across sessions, it only makes sense to combine entries at the `.cell_agg_inst_fr_list` variable and lower (as the number of cells can be added across sessions, treated as unique for each session).
        
        Usage:

            ## Load the saved across-session results:
            inst_fr_output_filename = 'long_short_inst_firing_rate_result_handlers_2023-07-12.pkl'
            across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list = AcrossSessionsResults.load_across_sessions_data(global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=inst_fr_output_filename)
            # across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path)
            num_sessions = len(across_sessions_instantaneous_fr_dict)
            print(f'num_sessions: {num_sessions}')

            ## Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions.
            global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.

            # To correctly aggregate results across sessions, it only makes sense to combine entries at the `.cell_agg_inst_fr_list` variable and lower (as the number of cells can be added across sessions, treated as unique for each session).

            ## Display the aggregate across sessions:
            _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # WARNING: we didn't save this info
            # _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline)
            # Cannot call `.compute(curr_active_pipeline=curr_active_pipeline)` like we normally would because everything is manually set.
            _out_fig_2.computation_result = across_session_inst_fr_computation
            _out_fig_2.active_identifying_session_ctx = across_session_inst_fr_computation.active_identifying_session_ctx
            # Set callback, the only self-specific property
            _out_fig_2._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)
            _out_fig_2.display(active_context=global_multi_session_context, title_modifier_fn=lambda original_title: f"{original_title} ({num_sessions} sessions)")

        """
        global_batch_result_inst_fr_file_path = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
        print(f'global_batch_result_inst_fr_file_path: {global_batch_result_inst_fr_file_path}')
        across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path)
        num_sessions = len(across_sessions_instantaneous_fr_dict)
        print(f'num_sessions: {num_sessions}')
        across_sessions_instantaneous_frs_list: List[InstantaneousSpikeRateGroupsComputation] = list(across_sessions_instantaneous_fr_dict.values())
        ## Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions.
        global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.
        # _out.cell_agg_inst_fr_list = cell_agg_firing_rates_list # .shape (n_cells,)
        across_session_inst_fr_computation = InstantaneousSpikeRateGroupsComputation()
        across_session_inst_fr_computation.active_identifying_session_ctx = global_multi_session_context 

        all_contexts_list: List[IdentifyingContext] = list(across_sessions_instantaneous_fr_dict.keys())
        assert len(all_contexts_list) > 0 # must have at least one element
        first_context = all_contexts_list[0]
        context_column_names = list(first_context.keys()) # ['format_name', 'animal', 'exper_name', 'session_name']
        expanded_context_df = pd.DataFrame.from_records([a_ctx.as_tuple() for a_ctx in all_contexts_list], columns=context_column_names)
        context_minimal_names = expanded_context_df.batch_results._build_minimal_session_identifiers_list()
        # print(f"context_minimal_names: {context_minimal_names}")
        assert len(context_minimal_names) == len(all_contexts_list)

        context_minimal_names_map = dict(zip(all_contexts_list, context_minimal_names))
        def _build_session_dep_aclu_identifier(session_context: IdentifyingContext, session_relative_aclus: np.ndarray):
            """ kdiba_pin01_one_fet11-01_12-58-54_{aclu} 
                with `context_minimal_names_map` - get tiny names like: a0s1, a0s2
            Captures: `context_minimal_names_map`
            """
            # return [f"{session_context}_{aclu}" for aclu in session_relative_aclus] # need very short version
            return [f"{context_minimal_names_map[session_context]}_{aclu}" for aclu in session_relative_aclus] # need very short version

        unique_animals = IdentifyingContext.find_unique_values(all_contexts_list)['animal'] # {'gor01', 'pin01', 'vvp01'}
        # Get number of animals to plot
        marker_list = [(5, i) for i in np.arange(len(unique_animals))] # [(5, 0), (5, 1), (5, 2)]
        scatter_props = [{'marker': mkr} for mkr in marker_list]  # Example, you should provide your own scatter properties
        scatter_props_dict = dict(zip(unique_animals, scatter_props))
        # {'pin01': {'marker': (5, 0)}, 'gor01': {'marker': (5, 1)}, 'vvp01': {'marker': (5, 2)}}
        # Pass a function that will return a set of kwargs for a given context
        def _return_scatter_props_fn(ctxt: IdentifyingContext):
            """ captures `scatter_props_dict` """
            animal_id = str(ctxt.animal)
            return scatter_props_dict[animal_id]

        LxC_aclus = np.concatenate([_build_session_dep_aclu_identifier(k, v.LxC_aclus) for k, v in across_sessions_instantaneous_fr_dict.items()])
        SxC_aclus = np.concatenate([_build_session_dep_aclu_identifier(k, v.SxC_aclus) for k, v in across_sessions_instantaneous_fr_dict.items()])

        across_session_inst_fr_computation.LxC_aclus = LxC_aclus
        across_session_inst_fr_computation.SxC_aclus = SxC_aclus
        
        ## Scatter props:
        LxC_scatter_props = [_return_scatter_props_fn(k) for k, v in across_sessions_instantaneous_fr_dict.items()]
        SxC_scatter_props = [_return_scatter_props_fn(k) for k, v in across_sessions_instantaneous_fr_dict.items()]
        
        # across_session_inst_fr_computation.LxC_scatter_props = LxC_scatter_props
        # across_session_inst_fr_computation.SxC_scatter_props = SxC_scatter_props


        # Broken as of 2023-10-03:
        across_session_inst_fr_computation.LxC_scatter_props = None
        across_session_inst_fr_computation.SxC_scatter_props = None

        # i = 0
        # across_sessions_instantaneous_frs_list[i].LxC_aclus
        # LxC_aclus = across_sessions_instantaneous_frs_list[0].LxC_ThetaDeltaPlus.LxC_aclus
        # SxC_aclus = across_sessions_instantaneous_frs_list[0].LxC_ThetaDeltaPlus.SxC_aclus

        # Note that in general LxC and SxC might have differing numbers of cells.
        across_session_inst_fr_computation.Fig2_Laps_FR = [SingleBarResult(v.mean(), v.std(), v, LxC_aclus, SxC_aclus, LxC_scatter_props, SxC_scatter_props) for v in (np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ThetaDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions) if across_sessions_instantaneous_frs_list[i].LxC_ThetaDeltaMinus is not None]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ThetaDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions) if across_sessions_instantaneous_frs_list[i].LxC_ThetaDeltaPlus is not None]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ThetaDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions) if across_sessions_instantaneous_frs_list[i].SxC_ThetaDeltaMinus is not None]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ThetaDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions) if across_sessions_instantaneous_frs_list[i].SxC_ThetaDeltaPlus is not None]))]


        across_session_inst_fr_computation.Fig2_Replay_FR = [SingleBarResult(v.mean(), v.std(), v, LxC_aclus, SxC_aclus, LxC_scatter_props, SxC_scatter_props) for v in (np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ReplayDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions) if across_sessions_instantaneous_frs_list[i].LxC_ReplayDeltaMinus is not None]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].LxC_ReplayDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions) if across_sessions_instantaneous_frs_list[i].LxC_ReplayDeltaPlus is not None]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ReplayDeltaMinus.cell_agg_inst_fr_list for i in np.arange(num_sessions) if across_sessions_instantaneous_frs_list[i].SxC_ReplayDeltaMinus is not None]),
                                                        np.concatenate([across_sessions_instantaneous_frs_list[i].SxC_ReplayDeltaPlus.cell_agg_inst_fr_list for i in np.arange(num_sessions) if across_sessions_instantaneous_frs_list[i].SxC_ReplayDeltaPlus is not None]))]

        return across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list


# ==================================================================================================================== #
# HDF5 Across File Aggregations                                                                                        #
# ==================================================================================================================== #

@define(slots=False)
class H5FileReference:
    short_name: str
    path: Path


@define(slots=False)
class ExternallyReferencedItem:
    foreign_key: str # the key in the external file that is referenced
    local_key: str # the key that will be created in the new reference table
    

@define(slots=False)
class H5FileAggregator:
    """ a class for loading and either building external links to or consolidating multiple .h5 files
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import H5FileAggregator
        session_group_keys: List[str] = [("/" + a_ctxt.get_description(separator="/", include_property_names=False)) for a_ctxt in session_identifiers] # 'kdiba/gor01/one/2006-6-08_14-26-15'
        neuron_identities_table_keys = [f"{session_group_key}/neuron_identities/table" for session_group_key in session_group_keys]
        a_loader = H5FileAggregator.init_from_file_lists(file_list=included_h5_paths, table_key_list=neuron_identities_table_keys)
        _out_table = a_loader.load_and_consolidate()
        _out_table


    """
    file_reference_list: List[H5FileReference] = field(default=Factory(list))
    table_key_list: List[str] = field(default=Factory(list))
    

    @property
    def file_short_name(self) -> list[Path]: 
        return [a_ref.short_name for a_ref in self.file_reference_list]

    @property
    def file_list(self) -> list[Path]: 
        return [a_ref.path for a_ref in self.file_reference_list]


    @classmethod
    def init_from_file_lists(cls, file_list, table_key_list=None, short_name_list=None):
        """ 
        
        
        table_key_list: only used for external linking mode, which was initially concieved of being a property of the class which was called H5ExternalLinker or something at the time.
        
        """
        if short_name_list is None:
            try:
                short_name_list = [a_file.filename for a_file in file_list]
            except AttributeError:
                # for Path inputs:
                short_name_list = [a_file.name for a_file in file_list]
                
        assert len(short_name_list) == len(file_list)
        if table_key_list is not None:
            assert len(table_key_list) == len(file_list)
        return cls(file_reference_list=[H5FileReference(short_name=a_short_name, path=a_file) for a_short_name, a_file in zip(short_name_list, file_list)], table_key_list=table_key_list)
    
    
    def load_and_consolidate(self, table_key_list=None, fail_on_exception:bool=True) -> pd.DataFrame:
        """
        Loads .h5 files and consolidates into a master table
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import H5FileReference, H5ExternalLinkBuilder

            session_short_names: List[str] = [a_ctxt.get_description(separator='_') for a_ctxt in included_session_contexts] # 'kdiba.gor01.one.2006-6-08_14-26-15'
            session_group_keys: List[str] = [("/" + a_ctxt.get_description(separator="/", include_property_names=False)) for a_ctxt in included_session_contexts] # 'kdiba/gor01/one/2006-6-08_14-26-15'
            neuron_identities_table_keys = [f"{session_group_key}/neuron_identities/table" for session_group_key in session_group_keys]
            a_loader = H5ExternalLinkBuilder.init_from_file_lists(file_list=included_h5_paths, table_key_list=neuron_identities_table_keys, short_name_list=session_short_names)
            _out_table = a_loader.load_and_consolidate()
            _out_table

        """
        table_key_list = table_key_list or self.table_key_list
        data_frames = []
        for file, table_key in zip(self.file_list, table_key_list):
            try:
                with tb.open_file(file, mode='r') as h5_file:
                        a_table = h5_file.get_node(table_key)
                        # print(f'a_table: {a_table}')
                        # for a_record in a_table
                        
                        # data_frames.append(a_table)
        #                 for table in h5_file.get_node(table_key):
        #                 # for table in h5_file.root:
                        # df = pd.DataFrame.from_records(a_table[:]) # .read()
                        df = pd.DataFrame.from_records(a_table.read()) 
                        data_frames.append(df)
            # except NoSuchNodeError:
            except BaseException as e:
                if fail_on_exception:
                    raise
                else:
                    print(f'failed for file path: {str(file)}, table_key: {table_key}. wth exception {e}. Skipping.')
                    
                        
        print(f'concatenating dataframes from {len(data_frames)} of {len(self.file_list)} files')
        master_table = pd.concat(data_frames, ignore_index=True)
        return master_table
    
        
    def build_linking_results(self, destination_file_path, referential_group_key: str = 'referential_group', table_key_list=None, destination_file_mode='w', fail_on_exception:bool=True):
        """ Creates (or overwrites) a new .h5 file at `destination_file_path` containing external links to existing files in self.file_list
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import H5FileReference, H5ExternalLinkBuilder

            session_short_names: List[str] = [a_ctxt.get_description(separator='_') for a_ctxt in included_session_contexts] # 'kdiba.gor01.one.2006-6-08_14-26-15'
            session_group_keys: List[str] = [("/" + a_ctxt.get_description(separator="/", include_property_names=False)) for a_ctxt in included_session_contexts] # 'kdiba/gor01/one/2006-6-08_14-26-15'
            neuron_identities_table_keys = [f"{session_group_key}/neuron_identities/table" for session_group_key in session_group_keys]
            a_loader = H5ExternalLinkBuilder.init_from_file_lists(file_list=included_h5_paths, table_key_list=neuron_identities_table_keys, short_name_list=session_short_names)
            # _out_table = a_loader.load_and_consolidate()
            # _out_table

            destination_file_path, external_file_links = a_loader.build_linking_results('output/test_linking_file.h5', fail_on_exception=False)
            external_file_links

        """
        table_key_list = table_key_list or self.table_key_list

        # , session_identifiers, external_h5_links
        external_file_links: Dict = {}
        with tb.open_file(destination_file_path, mode=destination_file_mode) as f: # this mode='w' is correct because it should overwrite the previous file and not append to it.
            a_referential_group: Group = f.create_group('/', referential_group_key, title='external links to all of the files in the H5ExternalLinkBuilder', createparents=True)            
            for file_short_name, file, table_key in zip(self.file_short_name, self.file_list, table_key_list):
                try:
                    with tb.open_file(file, mode='r') as h5_file:
                        a_table = h5_file.get_node(table_key)
                        # print(f'a_table: {a_table}')
                        an_external_link = f.create_external_link(where=a_referential_group, name=file_short_name, target=a_table, createparents=False)
                        # an_external_link = f.create_external_link(where=f'file:/path/to/node', name, target=f'file:{file}{table_key}', createparents=False)
                        # external_file_links.append(an_external_link)
                        external_file_links[file_short_name] = an_external_link
                # except NoSuchNodeError:
                except Exception as e:
                    if fail_on_exception:
                        raise
                    else:
                        print(f'failed for file: {file_short_name}, path: {str(file)}, table_key: {table_key}. wth exception {e}. Skipping.')                    
                        external_file_links[file_short_name] = None
                

        print(f'added {len(external_file_links)} links to file.')
        return destination_file_path, external_file_links




def check_output_h5_files(included_file_paths, minimum_good_file_size_GB:float=0.01, include_too_small_files:bool=False):
        """
        Usage:

        df = check_output_h5_files(included_file_paths=included_h5_paths)
        df
        """
        metadata = []

        for a_file in included_file_paths:
            # if not a_file.exists():
            if not isinstance(a_file, Path):
                a_file = Path(a_file).resolve()
            fetched_metadata = get_file_metadata(a_file)
            if fetched_metadata is None:
                print(f'file {a_file} does not exist. Skipping.')
            else:
                if fetched_metadata['file_size'] < minimum_good_file_size_GB:
                    print(f'WARN: file_size < {minimum_good_file_size_GB} for {a_file}!')
                    if include_too_small_files:
                        print(f'\t Continuing hesitantly.')
                        metadata.append(fetched_metadata)
                else:
                    # file size is reasonable:
                    metadata.append(fetched_metadata)
                

        # pd.options.display.float_format = '{:.2f}'.format
        df = pd.DataFrame(metadata)
        # df.style.format("{:.1f}") # suppresses scientific notation display only for this dataframe. Alternatively: pd.options.display.float_format = '{:.2f}'.format
        # df['file_size'] = df['file_size'].round(decimals=2)

        # with pd.option_context('display.float_format', lambda x: f'{x:,.3f}'):
            # print(df)
        return df


def save_filelist_to_text_file(output_paths, filelist_path: Path):
    _out_string = '\n'.join([str(a_file) for a_file in output_paths])
    print(f'{_out_string}')
    print(f'saving out to "{filelist_path}"...')
    with open(filelist_path, 'w') as f:
        f.write(_out_string)
    return _out_string, filelist_path


def build_output_filelists(filelist_save_parent_path: Path, included_session_basedirs: List[Path], BATCH_DATE_TO_USE:str, source_computer_name:str='GreatLakes', dest_computer_name:str='LabWorkstation'):
    """ 
    Usage:
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import build_output_filelists
        
        output_filelist_transfer_dict = build_output_filelists(filelist_save_parent_path=global_data_root_parent_path, included_session_basedirs=included_session_basedirs, BATCH_DATE_TO_USE=BATCH_DATE_TO_USE, dest_computer_name='LabWorkstation')
        
    """
    _dest_computers_dict = {'Apogee':Path(r'/~/W/Data/'), 'LabWorkstation':Path(r'/media/MAX/cloud/turbo/Data'), 'GreatLakes':Path(r'/nfs/turbo/umms-kdiba/Data')}
    # output_filelist_paths = []
    output_filelist_transfer_dict = {}
    
    included_h5_paths = [a_dir.joinpath('output','pipeline_results.h5').resolve() for a_dir in included_session_basedirs]
    included_pkl_output_paths = [a_dir.joinpath('loadedSessPickle.pkl').resolve() for a_dir in included_session_basedirs]
    included_global_computation_pkl_paths = [a_dir.joinpath('output','global_computation_results.pkl').resolve() for a_dir in included_session_basedirs]
    # included_global_computation_h5_paths = [a_dir.joinpath('output','global_computations.h5').resolve() for a_dir in included_session_basedirs] 

    filelist_dict = dict(zip(['pkls', 'global_pkls', 'HDF5'], (included_pkl_output_paths, included_global_computation_pkl_paths, included_h5_paths)))

    # Save output filelist:
    # h5_filelist_path = filelist_save_parent_path.joinpath(f'fileList_Greatlakes_HDF5_{BATCH_DATE_TO_USE}.txt').resolve()
    # _out_string, src_filelist_HDF5_savepath = save_filelist_to_text_file(included_h5_paths, h5_filelist_path)
    # output_filelist_paths.append(src_filelist_HDF5_savepath)
    
    # pkls_filelist_path = filelist_save_parent_path.joinpath(f'fileList_Greatlakes_pkls_{BATCH_DATE_TO_USE}.txt').resolve()
    # _out_string, src_filelist_pkls_savepath = save_filelist_to_text_file(included_pkl_output_paths, pkls_filelist_path)
    # output_filelist_paths.append(src_filelist_pkls_savepath)
    
    # global_pkls_filelist_path = filelist_save_parent_path.joinpath(f'fileList_Greatlakes_global_pkls_{BATCH_DATE_TO_USE}.txt').resolve()
    # _out_string, src_filelist_global_pkls_savepath = save_filelist_to_text_file(included_global_computation_pkl_paths, global_pkls_filelist_path)
    # output_filelist_paths.append(src_filelist_global_pkls_savepath)
    
    # source_parent_path = Path(r'/media/MAX/cloud/turbo/Data')
    # source_parent_path = Path(r'/nfs/turbo/umms-kdiba/Data')
    source_parent_path: Path = _dest_computers_dict[source_computer_name]
    # dest_parent_path = Path(r'/~/W/Data/')
    dest_parent_path: Path = _dest_computers_dict[dest_computer_name]
    # # Build the destination filelist from the source_filelist and the two paths:
    for a_filelist_name, a_source_filelist in filelist_dict.items():
        # Non-converted:
        source_filelist_path = filelist_save_parent_path.joinpath(f'fileList_{source_computer_name}_{a_filelist_name}_{BATCH_DATE_TO_USE}.txt').resolve()
        _out_string, a_src_filelist_savepath = save_filelist_to_text_file(a_source_filelist, source_filelist_path)
        # output_filelist_paths.append(a_src_filelist_savepath)
    
        ## Converted
        filelist_dest_paths = convert_filelist_to_new_parent(a_source_filelist, original_parent_path=source_parent_path, dest_parent_path=dest_parent_path)
        dest_Computer_h5_filelist_path = filelist_save_parent_path.joinpath(f'dest_fileList_{dest_computer_name}_{a_filelist_name}_{BATCH_DATE_TO_USE}.txt').resolve()
        _out_string, dest_filelist_savepath = save_filelist_to_text_file(filelist_dest_paths, dest_Computer_h5_filelist_path)
        # output_filelist_paths.append(dest_filelist_savepath)
        
        output_filelist_transfer_dict[a_src_filelist_savepath] = dest_filelist_savepath
    
    return output_filelist_transfer_dict


def copy_files_in_filelist_to_dest(filelist_text_file='fileList_GreatLakes_HDF5_2023-09-29-GL.txt', target_directory='/path/to/target/directory'):
    """ 
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import copy_files_in_filelist_to_dest
    
    copy_files_in_filelist_to_dest(filelist_text_file="/nfs/turbo/umms-kdiba/Data/fileList_GreatLakes_HDF5_2023-09-29-GL.txt", target_directory=Path('output/extracted_hdf5_files/').resolve())
    
    
    """
    # Read the file paths from the text file
    with open(filelist_text_file, 'r') as f:
        file_paths = f.readlines()

    # Remove newline characters from each line
    file_paths = [x.strip() for x in file_paths]

    # Target directory where files will be copied

    # Copy each file to the target directory
    for file_path in file_paths:
        print(f'copying {file_path} to {target_directory}...')
        shutil.copy(file_path, target_directory)
    print(f'done.')




class AcrossSessionTables:

    aliases_columns_dict = {'global_uid':'neuron_uid', 'neuron_id':'aclu'}

    @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-25 14:28', related_items=[])
    def build_custom_table(included_session_contexts, included_h5_paths, df_table_keys, drop_columns_list:Optional[List]=None, should_restore_native_column_types:bool=True):
        """
        like:
        
        df_table_keys: like [f"{session_group_key}/global_computations/jonathan_fr_analysis/neuron_replay_stats_df/table" for session_group_key in session_group_keys]
        drop_columns_list: list of columns to drop after loading is complete. e.g. ['neuron_IDX', 'has_short_pf', 'has_na', 'has_long_pf', 'index']
        
        
        """
        session_short_names: List[str] = [a_ctxt.get_description(separator='_') for a_ctxt in included_session_contexts] # 'kdiba.gor01.one.2006-6-08_14-26-15'
        # session_group_keys: List[str] = [("/" + a_ctxt.get_description(separator="/", include_property_names=False)) for a_ctxt in included_session_contexts] # 'kdiba/gor01/one/2006-6-08_14-26-15'  
        a_loader = H5FileAggregator.init_from_file_lists(file_list=included_h5_paths, short_name_list=session_short_names)
        _out_table = a_loader.load_and_consolidate(table_key_list=df_table_keys, fail_on_exception=False)
        if should_restore_native_column_types:
            _out_table = HDF_Converter.general_post_load_restore_table_as_needed(_out_table)
        
        if drop_columns_list is not None:
            # Drop columns: 'neuron_IDX', 'has_short_pf' and 3 other columns
            _out_table = _out_table.drop(columns=drop_columns_list)
            
        # try to rename the columns if needed
        _out_table.rename(columns=AcrossSessionTables.aliases_columns_dict, inplace=True)
        return _out_table
    
    @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-25 14:28', related_items=[])
    def build_neuron_replay_stats_table(included_session_contexts, included_h5_paths, **kwargs):
        """ 
        Usage:
            neuron_replay_stats_table = AcrossSessionTables.build_neuron_replay_stats_table(included_session_contexts, included_h5_paths)
            neuron_replay_stats_table
        """
        session_group_keys: List[str] = [("/" + a_ctxt.get_description(separator="/", include_property_names=False)) for a_ctxt in included_session_contexts] # 'kdiba/gor01/one/2006-6-08_14-26-15'
        neuron_replay_stats_df_table_keys = [f"{session_group_key}/global_computations/jonathan_fr_analysis/neuron_replay_stats_df/table" for session_group_key in session_group_keys]
        drop_columns_list = ['neuron_IDX', 'has_short_pf', 'has_na', 'has_long_pf', 'index']
        return AcrossSessionTables.build_custom_table(included_session_contexts, included_h5_paths, df_table_keys=neuron_replay_stats_df_table_keys, drop_columns_list=drop_columns_list, **kwargs)

    @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-25 14:28', related_items=[])
    def build_long_short_fr_indicies_analysis_table(included_session_contexts, included_h5_paths, **kwargs):
        """ 
        One row for each long/short neuron?

        Usage:
            long_short_fr_indicies_analysis_table = AcrossSessionTables.build_long_short_fr_indicies_analysis_table(included_session_contexts, included_h5_paths)
            long_short_fr_indicies_analysis_table

        """
        session_group_keys: List[str] = [("/" + a_ctxt.get_description(separator="/", include_property_names=False)) for a_ctxt in included_session_contexts] # 'kdiba/gor01/one/2006-6-08_14-26-15'
        long_short_fr_indicies_analysis_table_keys = [f"{session_group_key}/global_computations/long_short_fr_indicies_analysis/table" for session_group_key in session_group_keys]
        drop_columns_list = None # []
        long_short_fr_indicies_analysis_table = AcrossSessionTables.build_custom_table(included_session_contexts, included_h5_paths, df_table_keys=long_short_fr_indicies_analysis_table_keys, drop_columns_list=drop_columns_list, **kwargs)
        # long_short_fr_indicies_analysis_table = HDF_Converter.general_post_load_restore_table_as_needed(long_short_fr_indicies_analysis_table)
        return long_short_fr_indicies_analysis_table    

    @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-25 14:28', related_items=[])
    def build_neuron_identities_table(included_session_contexts, included_h5_paths, should_restore_native_column_types:bool=True):
        """ Extracts the neuron identities table from across the .h5 files.
        One row for each neuron.

        Usage:
            neuron_identities_table = AcrossSessionTables.build_neuron_identities_table(included_session_contexts, included_h5_paths)
            neuron_identities_table
        """
        session_group_keys: List[str] = [("/" + a_ctxt.get_description(separator="/", include_property_names=False)) for a_ctxt in included_session_contexts] # 'kdiba/gor01/one/2006-6-08_14-26-15'
        neuron_identities_table_keys = [f"{session_group_key}/neuron_identities/table" for session_group_key in session_group_keys]
        drop_columns_list = None
        neuron_identities_table = AcrossSessionTables.build_custom_table(included_session_contexts, included_h5_paths, df_table_keys=neuron_identities_table_keys, drop_columns_list=drop_columns_list, should_restore_native_column_types=should_restore_native_column_types)
        if should_restore_native_column_types:
            neuron_identities_table['session_uid'] = neuron_identities_table['session_uid'].astype(object)

        # aliases_columns_dict = {'global_uid':'neuron_uid', 'neuron_id':'aclu'}
        neuron_identities_table.rename(columns=AcrossSessionTables.aliases_columns_dict, inplace=True)
        # neuron_identities_table = HDF_Converter.general_post_load_restore_table_as_needed(neuron_identities_table)
        neuron_identities_table = neuron_identities_table[['neuron_uid', 'session_uid', 'session_datetime', 
                                    'format_name', 'animal', 'exper_name', 'session_name',
                                    'aclu', 'neuron_type', 'cluster_index', 'qclu', 'shank_index']]
        return neuron_identities_table

    @classmethod
    def write_table_to_files(cls, df, global_data_root_parent_path:Path, output_basename:str='neuron_identities_table', include_csv:bool=False, include_pkl:bool=True):
        """ 
        
        AcrossSessionTables.write_table_to_files(v, global_data_root_parent_path=global_data_root_parent_path, output_basename='a_table')
        """
        out_parent_path = global_data_root_parent_path.resolve() # = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
        out_parent_path.mkdir(parents=True, exist_ok=True)
        # print(f'global_batch_result_inst_fr_file_path: {out_parent_path}')
        # print(f'a_name: {a_name}')
        if not isinstance(output_basename, Path):
            output_basename = Path(output_basename)
        if include_csv:
            csv_out_path = out_parent_path.joinpath(output_basename.with_suffix(suffix='.csv'))
            print(f'writing {csv_out_path}.')
            df.to_csv(csv_out_path)
        if include_pkl:
            pkl_out_path = out_parent_path.joinpath(output_basename.with_suffix(suffix='.pkl'))
            print(f'writing {pkl_out_path}.')
            saveData(pkl_out_path, db=df, safe_save=False)


    @classmethod
    def load_table_from_file(cls, global_data_root_parent_path:Path, output_filename:str='neuron_identities_table', skip_on_error=False) -> pd.DataFrame:
        """ Reciprocal of  write_table_to_files
        
        v = AcrossSessionTables.load_table_from_file(global_data_root_parent_path=global_data_root_parent_path, output_filename='a_table.pkl')

        Usage:

            joined_neruon_fri_df = AcrossSessionTables.load_table_from_file(global_data_root_parent_path=global_data_root_parent_path, output_filename=f'{BATCH_DATE_TO_USE}_{output_file_prefix}_joined_neruon_fri_df')
            joined_neruon_fri_df

        """
        out_parent_path = global_data_root_parent_path.resolve() # = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
        assert out_parent_path.exists(), f"out_parent_path: '{out_parent_path}' must exist to load the tables!"
        # print(f'a_name: {a_name}')
        if not isinstance(output_filename, Path):
            output_filename = Path(output_filename)#.with_suffix(suffix='.pkl')
        pkl_out_path = out_parent_path.joinpath(output_filename)
        assert pkl_out_path.exists(), f"pkl_out_path: '{pkl_out_path}' does not exist!"
        print(f'reading {pkl_out_path}.')
        v = loadData(pkl_out_path)
        # try to rename the columns if needed
        v.rename(columns=cls.aliases_columns_dict, inplace=True)
        return v


        

    @classmethod
    def build_and_save_all_combined_tables(cls, included_session_contexts, included_h5_paths, should_restore_native_column_types:bool=True, override_output_parent_path:Optional[Path]=None, output_path_suffix:Optional[str]=None):
        """Save converted back to .h5 file, .csv file, and several others
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables

            AcrossSessionTables.build_and_save_all_combined_tables(included_session_contexts, included_h5_paths)
            included_h5_paths = [a_dir.joinpath('output','pipeline_results.h5').resolve() for a_dir in included_session_batch_progress_df['basedirs']]
        
            
            neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_and_save_all_combined_tables(included_session_contexts, included_h5_paths, output_path_suffix=f'_{BATCH_DATE_TO_USE}')
            

        """

        # Get the combined tables: 
        neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=should_restore_native_column_types)

        ## Potentially:
        # neuron_replay_stats_table = HDF_Converter.prepare_neuron_indexed_dataframe_for_hdf(neuron_replay_stats_table, active_context=curr_active_pipeline.get_session_context(), aclu_column_name=None)

        # Build the output paths:
        out_parent_path: Path = override_output_parent_path or Path('output/across_session_results')
        out_parent_path = out_parent_path.resolve()

        if output_path_suffix is not None:
            out_parent_path = out_parent_path.joinpath(output_path_suffix).resolve()

        out_parent_path.mkdir(parents=True, exist_ok=True)
        
        across_session_outputs = {'neuron_identities_table': neuron_identities_table,
        'long_short_fr_indicies_analysis_table': long_short_fr_indicies_analysis_table,
        'neuron_replay_stats_table': neuron_replay_stats_table}

        for table_name, v in across_session_outputs.items():
            table_name = Path(table_name)
            a_name = table_name.name
            print(f'a_name: {a_name}')
            cls.write_table_to_files(v, global_data_root_parent_path=out_parent_path, output_basename=table_name)
            # csv_out_path = out_parent_path.joinpath(table_name.with_suffix(suffix='.csv'))
            # print(f'writing {csv_out_path}.')
            # v.to_csv(csv_out_path)
            # pkl_out_path = out_parent_path.joinpath(table_name.with_suffix(suffix='.pkl'))
            # print(f'writing {pkl_out_path}.')
            # saveData(pkl_out_path, db=v, safe_save=False)
            # v.to_hdf(k, key=f'/{a_name}', format='table', data_columns=True)    # TypeError: objects of type ``StringArray`` are not supported in this context, sorry; supported objects are: NumPy array, record or scalar; homogeneous list or tuple, integer, float, complex or bytes
            
        return neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table
    

    @classmethod
    def load_all_combined_tables(cls, override_output_parent_path:Optional[Path]=None, output_path_suffix:Optional[str]=None):
        """Save converted back to .h5 file, .csv file, and several others
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables
 
            neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.load_all_combined_tables(override_output_parent_path=global_data_root_parent_path, output_path_suffix=f'_{BATCH_DATE_TO_USE}')
            

        """
        # Build the output paths:
        out_parent_path: Path = override_output_parent_path or Path('output/across_session_results')
        out_parent_path = out_parent_path.resolve()

        if output_path_suffix is not None:
            out_parent_path = out_parent_path.joinpath(output_path_suffix).resolve()

        # out_parent_path.mkdir(parents=True, exist_ok=True)
        assert out_parent_path.exists(), f"out_parent_path: '{out_parent_path}' must exist to load the tables!"
        
        across_session_outputs = {'neuron_identities_table': None,
        'long_short_fr_indicies_analysis_table': None,
        'neuron_replay_stats_table': None}

        _loaded_tables = []
        
        for k, v in across_session_outputs.items():
            k = Path(k)
            a_name = k.name
            print(f'a_name: {a_name}')
            # csv_out_path = out_parent_path.joinpath(k.with_suffix(suffix='.csv'))
            # print(f'loading {csv_out_path}.')
            # v.to_csv(csv_out_path)
            pkl_out_path = out_parent_path.joinpath(k.with_suffix(suffix='.pkl'))
            print(f'loading {pkl_out_path}.')
            v = loadData(pkl_out_path)
            # try to rename the columns if needed
            v.rename(columns=cls.aliases_columns_dict, inplace=True)
            _loaded_tables.append(v)


        return _loaded_tables


    


    @classmethod
    def build_all_known_tables(cls, included_session_contexts, included_h5_paths, should_restore_native_column_types:bool=True):
        """ Extracts the neuron identities table from across the .h5 files.
        One row for each neuron.

        Usage:
            
            neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=Falsee)
            
        """
        neuron_identities_table = AcrossSessionTables.build_neuron_identities_table(included_session_contexts, included_h5_paths, should_restore_native_column_types=should_restore_native_column_types)
        long_short_fr_indicies_analysis_table = AcrossSessionTables.build_long_short_fr_indicies_analysis_table(included_session_contexts, included_h5_paths, should_restore_native_column_types=should_restore_native_column_types)
        neuron_replay_stats_table = AcrossSessionTables.build_neuron_replay_stats_table(included_session_contexts, included_h5_paths, should_restore_native_column_types=should_restore_native_column_types)
        
        return neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table









# ==================================================================================================================== #
# 2024-01-27 - Across Session CSV Import and Processing                                                                #
# ==================================================================================================================== #
""" 
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import find_csv_files, find_HDF5_files, find_most_recent_files, process_csv_file

from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import find_csv_files, find_HDF5_files, find_most_recent_files, process_csv_file

"""


def find_csv_files(directory: str, recurrsive: bool=False):
    directory_path = Path(directory) # Convert string path to a Path object
    if recurrsive:
        return list(directory_path.glob('**/*.csv')) # Return a list of all .csv files in the directory and its subdirectories
    else:
        return list(directory_path.glob('*.csv')) # Return a list of all .csv files in the directory and its subdirectories
    

def find_HDF5_files(directory: str, recurrsive: bool=False):
    directory_path = Path(directory) # Convert string path to a Path object
    if recurrsive:
        return list(directory_path.glob('**/*.h5')) # Return a list of all .csv files in the directory and its subdirectories
    else:
        return list(directory_path.glob('*.h5')) # Return a list of all .h5 files in the directory and its subdirectories


from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
from attrs import define, field, Factory

@define(slots=False)
class BaseMatchParser:
    """ 
    ## Sequential Parser:
    ### Tries a series of methods to parse a filename into a variety of formats that doesn't require nested try/catch
    ### Recieves: filename: str
    """
    def try_parse(self, filename: str) -> Optional[Dict]:
        raise NotImplementedError

@define(slots=False)
class DayDateTimeParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        pattern = r"(?P<export_datetime_str>.*_\d{2}\d{2}[APMF]{2})-(?P<session_str>.*)-(?P<export_file_type>\(?.+\)?)(?:_tbin-(?P<decoding_time_bin_size_str>[^)]+))"
        match = re.match(pattern, filename)        
        if match is None:
            return None # failed
        
        parsed_output_dict = {}

        output_dict_keys = ['session_str', 'export_file_type', 'decoding_time_bin_size_str']

        # export_datetime_str, session_str, export_file_type = match.groups()
        export_datetime_str, session_str, export_file_type, decoding_time_bin_size_str = match.group('export_datetime_str'), match.group('session_str'), match.group('export_file_type'), match.group('decoding_time_bin_size_str')
        parsed_output_dict.update({k:match.group(k) for k in output_dict_keys})

        # parse the datetime from the export_datetime_str and convert it to datetime object
        export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d_%I%M%p")
        parsed_output_dict['export_datetime'] = export_datetime

        return parsed_output_dict
    

@define(slots=False)
class DayDateOnlyParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        # day_date_only_pattern = r"(.*(?:_\d{2}\d{2}[APMF]{2})?)-(.*)-(\(.+\))"
        day_date_only_pattern = r"(\d{4}-\d{2}-\d{2})-(.*)-(\(?.+\)?)" # 
        day_date_only_match = re.match(day_date_only_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'        
        if day_date_only_match is not None:
            export_datetime_str, session_str, export_file_type = day_date_only_match.groups()
            # print(export_datetime_str, session_str, export_file_type)
            # parse the datetime from the export_datetime_str and convert it to datetime object
            export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")

        match = re.match(day_date_only_pattern, filename)        
        if match is None:
            return None # failed
        
        export_datetime_str, session_str, export_file_type = day_date_only_match.groups()
        output_dict_keys = ['session_str', 'export_file_type']
        parsed_output_dict = dict(zip(output_dict_keys, [session_str, export_file_type]))
        # parse the datetime from the export_datetime_str and convert it to datetime object
        export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
        parsed_output_dict['export_datetime'] = export_datetime

        return parsed_output_dict

@define(slots=False)
class DayDateWithVariantSuffixParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        # matches '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'
        day_date_with_variant_suffix_pattern = r"(?P<export_datetime_str>\d{4}-\d{2}-\d{2})[-_]?(?P<variant_suffix>[^-_]*)[-_](?P<session_str>.+?)_(?P<export_file_type>[A-Za-z_]+)"
        match = re.match(day_date_with_variant_suffix_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26', 
        if match is None:
            return None # failed
        
        parsed_output_dict = {}
        output_dict_keys = ['session_str', 'export_file_type'] # , 'variant_suffix'
        export_datetime_str, session_str, export_file_type = match.group('export_datetime_str'), match.group('session_str'), match.group('export_file_type')
        parsed_output_dict.update({k:match.group(k) for k in output_dict_keys})
        # parse the datetime from the export_datetime_str and convert it to datetime object
        try:
            export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
            parsed_output_dict['export_datetime'] = export_datetime
        except ValueError as e:
            print(f'ERR: Could not parse date "{export_datetime_str}" of filename: "{filename}"') # 2024-01-18_GL_t_split_df
            return None # failed used to return ValueError when it couldn't parse, but we'd rather skip unparsable files

        return parsed_output_dict
    
## INPUTS: basename
@function_attributes(short_name=None, tags=['parse', 'filename'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-28 10:10', related_items=[])
def try_parse_chain(basename: str, debug_print:bool=False):
    """ tries to parse the basename with the list of parsers. 
    
    Usage:
    
        basename: str = _test_h5_filename.stem
        final_parsed_output_dict = try_parse_chain(basename=basename)
        final_parsed_output_dict

    """
    # _filename_parsers_list = (DayDateTimeParser(), DayDateWithVariantSuffixParser(), DayDateOnlyParser())
    _filename_parsers_list = (DayDateTimeParser(), DayDateOnlyParser(), DayDateWithVariantSuffixParser())
    final_parsed_output_dict = None
    for a_test_parser in _filename_parsers_list:
        a_parsed_output_dict = a_test_parser.try_parse(basename)
        if a_parsed_output_dict is not None:
            ## best parser, stop here
            if debug_print:
                print(f'got parsed output {a_test_parser} - result: {a_parsed_output_dict}, basename: {basename}')
            final_parsed_output_dict = a_parsed_output_dict
            return final_parsed_output_dict
        
    return final_parsed_output_dict

@function_attributes(short_name=None, tags=['parse'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-28 10:16', related_items=[])
def parse_filename(path: Path, debug_print:bool=False) -> Tuple[datetime, str, str]:
    """ 
    A revised version built on 2024-03-28 that uses `try_parse_chain` instead of nested for loops.

    # from the found_session_export_paths, get the most recently exported laps_csv, ripple_csv (by comparing `export_datetime`) for each session (`session_str`)
    a_export_filename: str = "2024-01-12_0420PM-kdiba_pin01_one_fet11-01_12-58-54-(laps_marginals_df).csv"
    export_datetime = "2024-01-12_0420PM"
    session_str = "kdiba_pin01_one_fet11-01_12-58-54"
    export_file_type = "(laps_marginals_df)" # .csv

    # return laps_csv, ripple_csv
    laps_csv = Path("C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(laps_marginals_df).csv").resolve()
    ripple_csv = Path("C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(ripple_marginals_df).csv").resolve()

    """
    filename: str = path.stem   # Get filename without extension    
    final_parsed_output_dict = try_parse_chain(basename=filename)

    if final_parsed_output_dict is None:
        print(f'ERR: Could not parse filename: "{filename}"') # 2024-01-18_GL_t_split_df
        return None, None, None, None # used to return ValueError when it couldn't parse, but we'd rather skip unparsable files

    export_datetime, session_str, export_file_type = final_parsed_output_dict.get('export_datetime', None), final_parsed_output_dict.get('session_str', None), final_parsed_output_dict.get('export_file_type', None)
    decoding_time_bin_size_str = final_parsed_output_dict.get('decoding_time_bin_size_str', None)

    if export_file_type is not None:
        if export_file_type[0] == '(' and export_file_type[-1] == ')':
            # Trim the brackets from the file type if they're present:
            export_file_type = export_file_type[1:-1]

    return export_datetime, session_str, export_file_type, decoding_time_bin_size_str


def _OLD_parse_filename(path: Path, debug_print:bool=False) -> Tuple[datetime, str, str]:
    """ 
    # from the found_session_export_paths, get the most recently exported laps_csv, ripple_csv (by comparing `export_datetime`) for each session (`session_str`)
    a_export_filename: str = "2024-01-12_0420PM-kdiba_pin01_one_fet11-01_12-58-54-(laps_marginals_df).csv"
    export_datetime = "2024-01-12_0420PM"
    session_str = "kdiba_pin01_one_fet11-01_12-58-54"
    export_file_type = "(laps_marginals_df)" # .csv

    # return laps_csv, ripple_csv
    laps_csv = Path("C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(laps_marginals_df).csv").resolve()
    ripple_csv = Path("C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(ripple_marginals_df).csv").resolve()

    """
    filename = path.stem   # Get filename without extension
    decoding_time_bin_size_str = None
    
    pattern = r"(?P<export_datetime_str>.*_\d{2}\d{2}[APMF]{2})-(?P<session_str>.*)-(?P<export_file_type>\(?.+\)?)(?:_tbin-(?P<decoding_time_bin_size_str>[^)]+))"
    match = re.match(pattern, filename)
    
    if match is not None:
        # export_datetime_str, session_str, export_file_type = match.groups()
        export_datetime_str, session_str, export_file_type, decoding_time_bin_size_str = match.group('export_datetime_str'), match.group('session_str'), match.group('export_file_type'), match.group('decoding_time_bin_size_str')
        # parse the datetime from the export_datetime_str and convert it to datetime object
        export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d_%I%M%p")
    else:
        if debug_print:
            print(f'did not match pattern with time.')
        # day_date_only_pattern = r"(.*(?:_\d{2}\d{2}[APMF]{2})?)-(.*)-(\(.+\))"
        day_date_only_pattern = r"(\d{4}-\d{2}-\d{2})-(.*)-(\(?.+\)?)" # 
        day_date_only_match = re.match(day_date_only_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'        
        if day_date_only_match is not None:
            export_datetime_str, session_str, export_file_type = day_date_only_match.groups()
            # print(export_datetime_str, session_str, export_file_type)
            # parse the datetime from the export_datetime_str and convert it to datetime object
            export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
        else:
            # Try H5 pattern:
            # matches '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'
            day_date_with_variant_suffix_pattern = r"(?P<export_datetime_str>\d{4}-\d{2}-\d{2})_?(?P<variant_suffix>[^-_]*)-(?P<session_str>.+?)_(?P<export_file_type>[A-Za-z_]+)"
            day_date_with_variant_suffix_match = re.match(day_date_with_variant_suffix_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26', 
            if day_date_with_variant_suffix_match is not None:
                export_datetime_str, session_str, export_file_type = day_date_with_variant_suffix_match.group('export_datetime_str'), day_date_with_variant_suffix_match.group('session_str'), day_date_with_variant_suffix_match.group('export_file_type')
                # parse the datetime from the export_datetime_str and convert it to datetime object
                try:
                    export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
                except ValueError as e:
                    print(f'ERR: Could not parse date "{export_datetime_str}" of filename: "{filename}"') # 2024-01-18_GL_t_split_df
                    return None, None, None # used to return ValueError when it couldn't parse, but we'd rather skip unparsable files
            else:
                print(f'ERR: Could not parse filename: "{filename}"') # 2024-01-18_GL_t_split_df
                return None, None, None # used to return ValueError when it couldn't parse, but we'd rather skip unparsable files

        
    if export_file_type[0] == '(' and export_file_type[-1] == ')':
        # Trim the brackets from the file type if they're present:
        export_file_type = export_file_type[1:-1]

    return export_datetime, session_str, export_file_type, decoding_time_bin_size_str



def find_most_recent_files(found_session_export_paths: List[Path], cuttoff_date:Optional[datetime]=None, debug_print: bool = False) -> Dict[str, Dict[str, Tuple[Path, datetime]]]:
    """
    Returns a dictionary representing the most recent files for each session type among a list of provided file paths.

    Parameters:
    found_session_export_paths (List[Path]): A list of Paths representing files to be checked.
    cuttoff_date (datetime): a date which all files must be newer than to be considered for inclusion. If not provided, the most recent files will be included regardless of their date.
    debug_print (bool): A flag to trigger debugging print statements within the function. Default is False.

    Returns:
    Dict[str, Dict[str, Tuple[Path, datetime]]]: A nested dictionary where the main keys represent 
    different session types. The inner dictionary's keys represent file types and values are the most recent 
    Path and datetime for this combination of session and file type.
    
    # now sessions is a dictionary where the key is the session_str and the value is another dictionary.
    # This inner dictionary's key is the file type and the value is the most recent path for this combination of session and file type
    # Thus, laps_csv and ripple_csv can be obtained from the dictionary for each session

    """
    # Function 'parse_filename' should be defined in the global scope
    parsed_paths = [(*parse_filename(p), p) for p in found_session_export_paths if (parse_filename(p)[0] is not None)] # note we append path p to the end of the tuple
    parsed_paths.sort(reverse=True)

    if debug_print:
        print(f'parsed_paths: {parsed_paths}')

    sessions = {}
    for export_datetime, session_str, file_type, decoding_time_bin_size_str, path in parsed_paths:
        if session_str not in sessions:
            sessions[session_str] = {}

        should_add: bool = False
        if (file_type not in sessions[session_str]) or (sessions[session_str][file_type][-1] < export_datetime):
            if cuttoff_date is not None:
                if (cuttoff_date <= export_datetime):
                    should_add = True
            else:
                # if there is no cutoff date, add
                should_add = True

            if should_add:
                sessions[session_str][file_type] = (path, decoding_time_bin_size_str, export_datetime)
    
    return sessions
    

def process_csv_file(file: str, session_name: str, curr_session_t_delta: Optional[float], time_col: str) -> pd.DataFrame:
    """ reads the CSV file and adds the 'session_name' column if it is missing. 
    
    """
    df = pd.read_csv(file)
    df['session_name'] = session_name 
    if curr_session_t_delta is not None:
        df['delta_aligned_start_t'] = df[time_col] - curr_session_t_delta
    return df


def export_across_session_CSVs(final_output_path: Path, TODAY_DAY_DATE:str, all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_all_scores_ripple_df, all_sessions_all_scores_laps_df=None):
    """ Exports the multi-session single CSVs after loading the CSVs for the individual sessions. Useful for plotting with RawGraphs/Orange, etc.

    Usage:
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import export_across_session_CSVs

        # TODAY_DAY_DATE: str = f"2024-03-18_Apogee"
        final_output_path = Path("../output/").resolve()

        final_csv_export_paths = export_across_session_CSVs(final_output_path=final_output_path, TODAY_DAY_DATE=TODAY_DAY_DATE,
                                                            all_sessions_laps_df=all_sessions_laps_df,  all_sessions_ripple_df=all_sessions_ripple_df,  all_sessions_laps_time_bin_df=all_sessions_laps_time_bin_df,  all_sessions_ripple_time_bin_df=all_sessions_ripple_time_bin_df, 
                                                            all_sessions_simple_pearson_laps_df=all_sessions_simple_pearson_laps_df,  all_sessions_simple_pearson_ripple_df=all_sessions_simple_pearson_ripple_df,
                                                            all_sessions_all_scores_ripple_df=all_sessions_all_scores_ripple_df,  all_sessions_all_scores_laps_df=None,
                                                            )
        final_csv_export_paths

    
    """
    # INPUTS: TODAY_DAY_DATE, final_output_path

    # final_sessions
    # {'kdiba_gor01_one_2006-6-08_14-26-15': {'ripple_marginals_df': WindowsPath('C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-17_0540PM-kdiba_gor01_one_2006-6-08_14-26-15-(ripple_marginals_df).csv'),
    #   'laps_marginals_df': WindowsPath('C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-17_0540PM-kdiba_gor01_one_2006-6-08_14-26-15-(laps_marginals_df).csv')},
    #  'kdiba_gor01_one_2006-6-09_1-22-43': {'ripple_marginals_df': WindowsPath('C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0838PM-kdiba_gor01_one_2006-6-09_1-22-43-(ripple_marginals_df).csv'),
    #   'laps_marginals_df': WindowsPath('C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0838PM-kdiba_gor01_one_2006-6-09_1-22-43-(laps_marginals_df).csv')},
    #  'kdiba_pin01_one_fet11-01_12-58-54': {'ripple_marginals_df': WindowsPath('C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(ripple_marginals_df).csv'),
    #   'laps_marginals_df': WindowsPath('C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2024-01-12_0828PM-kdiba_pin01_one_fet11-01_12-58-54-(laps_marginals_df).csv')}}

    # Save out the five dataframes to CSVs:
    across_session_output_df_prefix: str = f'AcrossSession'
    final_dfs_dict = {f"{across_session_output_df_prefix}_Laps_per-Epoch": all_sessions_laps_df, f"{across_session_output_df_prefix}_Ripple_per-Epoch": all_sessions_ripple_df,
                        f"{across_session_output_df_prefix}_Laps_per-TimeBin": all_sessions_laps_time_bin_df, f"{across_session_output_df_prefix}_Ripple_per-TimeBin": all_sessions_ripple_time_bin_df,
                        f"{across_session_output_df_prefix}_SimplePearson_Laps_per-Epoch": all_sessions_simple_pearson_laps_df, f"{across_session_output_df_prefix}_SimplePearson_Ripple_per-Epoch": all_sessions_simple_pearson_ripple_df,
                        f"{across_session_output_df_prefix}_AllScores_Ripple_per-Epoch": all_sessions_all_scores_ripple_df, #,
                        }
    
    if all_sessions_all_scores_laps_df is not None:
        final_dfs_dict.update({f"{across_session_output_df_prefix}_AllScores_Laps_per-Epoch": all_sessions_all_scores_laps_df})

    final_csv_export_paths = {}
    for a_name, a_final_df in final_dfs_dict.items():
        # save out one final DF to csv.
        out_csv_filename: str = f"{TODAY_DAY_DATE}_{a_name}.csv"

        if a_final_df is not None:    
            a_final_csv_export_path = final_output_path.joinpath(out_csv_filename).resolve()
            a_final_df.to_csv(a_final_csv_export_path) # save to CSV.
            final_csv_export_paths[a_name] = a_final_csv_export_path
        else:
            print(f'WARN: dataframe a_name: {a_name} is None, so it will not be exported to {out_csv_filename}')
            
    return final_csv_export_paths



@define(slots=False)
class AcrossSessionCSVOutputFormat:
    data_description = ["AcrossSession"]
    epoch_description = ["Laps", "Ripple"]
    granularity_description = ["per-Epoch", "per-TimeBin"]
    
    parts_names = ["export_date", "date_name", "epochs", "granularity"]
    
    def parse_filename(self, a_filename: str):
        if a_filename.endswith('.csv'):
            a_filename = a_filename.removesuffix('.csv') # drop the .csv suffix
        # split on the underscore into the parts
        parts = a_filename.split('_')
        if len(parts) == 4:
            export_date, date_name, epochs, granularity  = parts
        else:
            raise NotImplementedError(f"a_csv_filename: '{a_filename}' expected four parts but got {len(parts)} parts.\n\tparts: {parts}")
        return export_date, date_name, epochs, granularity
    


def _split_user_annotated_ripple_df(all_sessions_user_annotated_ripple_df):
    """ prints info about exported data sessions, such as the breakdown of user-annotated epochs, etc. 
    
    Usage:

        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import _split_user_annotated_ripple_df

        all_sessions_all_scores_ripple_df, (valid_ripple_df, invalid_ripple_df), (user_approved_ripple_df, user_approved_ripple_df) = _split_user_annotated_ripple_df(all_sessions_all_scores_ripple_df)



    """
    from pyphocorehelpers.indexing_helpers import partition, partition_df, partition_df_dict


    all_unique_session_names = all_sessions_user_annotated_ripple_df['session_name'].unique()

    all_sessions_user_annotated_ripple_df: pd.DataFrame = all_sessions_user_annotated_ripple_df.dropna(axis='index', subset=['is_user_annotated_epoch', 'is_valid_epoch'], inplace=False) ## Drop those missing the columns: ['is_user_annotated_epoch', 'is_valid_epoch']
    user_annotated_epoch_unique_session_names = all_sessions_user_annotated_ripple_df['session_name'].unique()
    print(f'user_annotated_epoch_unique_session_names: {user_annotated_epoch_unique_session_names}')

    unannotated_session_names = set(all_unique_session_names) - set(user_annotated_epoch_unique_session_names)
    print(f'unannotated_session_names: {unannotated_session_names}')

    ## Add 'pre_post_delta_category' helper column:
    all_sessions_user_annotated_ripple_df['pre_post_delta_category'] = 'post-delta'
    all_sessions_user_annotated_ripple_df.loc[(all_sessions_user_annotated_ripple_df['delta_aligned_start_t'] <= 0.0), 'pre_post_delta_category'] = 'pre-delta'

    _validity_partitioned_dfs = dict(zip(*partition_df(all_sessions_user_annotated_ripple_df, partitionColumn='is_valid_epoch')))
    valid_ripple_df: pd.DataFrame = _validity_partitioned_dfs[True].drop(columns=['is_valid_epoch']).reset_index(drop=True)
    invalid_ripple_df: pd.DataFrame = _validity_partitioned_dfs[False].drop(columns=['is_valid_epoch']).reset_index(drop=True)

    n_input_df_rows = np.shape(all_sessions_user_annotated_ripple_df)[0]
    n_valid_df_rows = np.shape(valid_ripple_df)[0]
    n_invalid_df_rows = np.shape(invalid_ripple_df)[0]
    n_unlabeled_df_rows = n_input_df_rows - (n_valid_df_rows + n_invalid_df_rows)

    print(f'n_input_df_rows: {n_input_df_rows}')
    print(f'\t n_valid_df_rows: {n_valid_df_rows}')
    print(f'\t n_invalid_df_rows: {n_invalid_df_rows}')
    if n_unlabeled_df_rows > 0:
        print(f'\t n_unlabeled_df_rows: {n_unlabeled_df_rows}')

    _partitioned_dfs = dict(zip(*partition_df(valid_ripple_df, partitionColumn='is_user_annotated_epoch'))) # use `valid_ripple_df` instead of the original dataframe to only get those which are valid.
    user_approved_ripple_df: pd.DataFrame = _partitioned_dfs[True].drop(columns=['is_user_annotated_epoch']).reset_index(drop=True)
    user_rejected_ripple_df: pd.DataFrame = _partitioned_dfs[False].drop(columns=['is_user_annotated_epoch']).reset_index(drop=True)

    ## Print info about user selections:
    # input_df = valid_ripple_df
    n_input_df_rows = np.shape(valid_ripple_df)[0]
    n_user_approved_df_rows = np.shape(user_approved_ripple_df)[0]
    n_user_rejected_df_rows = np.shape(user_rejected_ripple_df)[0]
    n_unlabeled_df_rows = n_input_df_rows - (n_user_approved_df_rows + n_user_rejected_df_rows)

    print(f'n_input_df_rows: {n_input_df_rows}')
    print(f'\t n_user_approved_df_rows: {n_user_approved_df_rows}')
    print(f'\t n_user_rejected_df_rows: {n_user_rejected_df_rows}')
    if n_unlabeled_df_rows > 0:
        print(f'\t n_unlabeled_df_rows: {n_unlabeled_df_rows}')

    return all_sessions_user_annotated_ripple_df, (valid_ripple_df, invalid_ripple_df), (user_approved_ripple_df, user_approved_ripple_df)












# ==================================================================================================================== #
# Visualizations                                                                                                       #
# ==================================================================================================================== #

class AcrossSessionsVisualizations:
    # 2023-07-21 - Across Sessions Aggregate Figure: __________________________________________________________________________________ #

    # _registered_output_files = {}

    @classmethod
    def output_figure(cls, final_context: IdentifyingContext, fig, write_vector_format:bool=False, write_png:bool=True, debug_print=True):
        """ outputs the figure using the provided context, replacing the pipeline's curr_active_pipeline.output_figure(...) callback which isn't usually accessible for across session figures. """
        
        def register_output_file(output_path, output_metadata=None):
            """ registers a new output file for the pipeline """
            print(f'register_output_file(output_path: {output_path}, ...)')
            # registered_output_files[output_path] = output_metadata or {}

        fig_out_man = FileOutputManager(figure_output_location=FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER, context_to_path_mode=ContextToPathMode.HIERARCHY_UNIQUE)
        active_out_figure_paths = build_and_write_to_file(fig, final_context, fig_out_man, write_vector_format=write_vector_format, write_png=write_png, register_output_file_fn=register_output_file)
        return active_out_figure_paths, final_context
        

    @classmethod
    def across_sessions_bar_graphs(cls, across_session_inst_fr_computation: Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation], num_sessions:int, save_figure=True, instantaneous_time_bin_size_seconds=0.003, **kwargs):
        """ 2023-07-21 - Across Sessions Aggregate Figure - I know this is hacked-up to use `PaperFigureTwo`'s existing plotting machinery (which was made to plot a single session) to plot something it isn't supposed to.
        Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions."""

        # num_sessions = len(across_sessions_instantaneous_fr_dict)
        print(f'num_sessions: {num_sessions}')

        global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.

        # To correctly aggregate results across sessions, it only makes sense to combine entries at the `.cell_agg_inst_fr_list` variable and lower (as the number of cells can be added across sessions, treated as unique for each session).

        ## Display the aggregate across sessions:
        _out_aggregate_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds) # WARNING: we didn't save this info
        _out_aggregate_fig_2.computation_result = across_session_inst_fr_computation
        _out_aggregate_fig_2.active_identifying_session_ctx = across_session_inst_fr_computation.active_identifying_session_ctx
        # Set callback, the only self-specific property
        # _out_fig_2._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)

        # registered_output_files = {}

        # Set callback, the only self-specific property
        _out_aggregate_fig_2._pipeline_file_callback_fn = cls.output_figure

        # Showing
        matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
        # Perform interactive Matplotlib operations with 'Qt5Agg' backend
        _fig_2_theta_out, _fig_2_replay_out = _out_aggregate_fig_2.display(active_context=global_multi_session_context, title_modifier_fn=lambda original_title: f"{original_title} ({num_sessions} sessions)", save_figure=save_figure, **kwargs)
        if save_figure:
            # _out_aggregate_fig_2.perform_save(_fig_2_theta_out)
            print(f'save_figure()!')

        return global_multi_session_context, _out_aggregate_fig_2


    @classmethod
    def across_sessions_firing_rate_index_figure(cls, long_short_fr_indicies_analysis_results, num_sessions:int, save_figure=True, **kwargs):
        """ 2023-08-24 - Across Sessions Aggregate Figure - Supposed to be the equivalent for Figure 3. 

        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsVisualizations

            neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=True)
            matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
            graphics_output_dict = AcrossSessionsVisualizations.across_sessions_firing_rate_index_figure(long_short_fr_indicies_analysis_results=long_short_fr_indicies_analysis_table, num_sessions=num_sessions)

        
        """
        # _out2 = curr_active_pipeline.display('_display_long_and_short_firing_rate_replays_v_laps', curr_active_pipeline.get_session_context(), defer_render=defer_render, save_figure=save_figure)

        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import _plot_long_short_firing_rate_indicies

        # Plot long|short firing rate index:
        x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
        x_frs_index = x_frs_index.set_axis(long_short_fr_indicies_analysis_results['neuron_uid']) # use neuron unique ID as index
        y_frs_index = y_frs_index.set_axis(long_short_fr_indicies_analysis_results['neuron_uid']) # use neuron unique ID as index

        # active_context = long_short_fr_indicies_analysis_results['active_context']
        global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.
        active_context = global_multi_session_context
        final_context = active_context.adding_context('display_fn', display_fn_name='display_long_short_laps')
        

        scatter_plot_kwargs = dict()
        if 'has_pf_color' in long_short_fr_indicies_analysis_results:
            scatter_plot_kwargs['edgecolors'] = long_short_fr_indicies_analysis_results['has_pf_color'].to_numpy() #.to_list() # edgecolors=(r, g, b, 1)
        

        fig = _plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, final_context, debug_print=True, is_centered=False, enable_hover_labels=False, enable_tiny_point_labels=False, facecolor='w', **scatter_plot_kwargs) #  markeredgewidth=1.5,
        
        def _perform_write_to_file_callback():
            active_out_figure_path, *args_L = cls.output_figure(final_context, fig)
            return (active_out_figure_path,)

        if save_figure:
            active_out_figure_paths = _perform_write_to_file_callback()
        else:
            active_out_figure_paths = []

        graphics_output_dict = MatplotlibRenderPlots(name='across_sessions_firing_rate_index_figure', figures=(fig), axes=tuple(fig.axes), plot_data={}, context=final_context, saved_figures=active_out_figure_paths)
        # graphics_output_dict['plot_data'] = {'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}            
        return graphics_output_dict
    



    @classmethod
    def across_sessions_long_and_short_firing_rate_replays_v_laps_figure(cls, neuron_replay_stats_table, num_sessions:int, save_figure=True, **kwargs):
        """ 2023-08-24 - Across Sessions Aggregate Figure - Supposed to be the equivalent for Figure 3.
        
        Based off of `pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions._plot_session_long_short_track_firing_rate_figures`
        
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsVisualizations
            
            neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=True)
            matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
            graphics_output_dict = AcrossSessionsVisualizations.across_sessions_long_and_short_firing_rate_replays_v_laps_figure(neuron_replay_stats_table=neuron_replay_stats_table, num_sessions=num_sessions)


        """

        from neuropy.utils.matplotlib_helpers import fit_both_axes
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import _plot_single_track_firing_rate_compare


        global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.
        active_context = global_multi_session_context
        final_context = active_context.adding_context('display_fn', display_fn_name='plot_single_track_firing_rate_compare')

        # (fig_L, ax_L, active_display_context_L), (fig_S, ax_S, active_display_context_S), _perform_write_to_file_callback = _plot_session_long_short_track_firing_rate_figures(owning_pipeline_reference, jonathan_firing_rate_analysis_result, defer_render=defer_render)

        ## Long Track Replay|Laps FR Figure
        neuron_replay_stats_df = neuron_replay_stats_table.dropna(subset=['long_replay_mean', 'long_non_replay_mean'], inplace=False)
        x_frs = {k:v for k,v in neuron_replay_stats_df['long_replay_mean'].items()}
        y_frs = {k:v for k,v in neuron_replay_stats_df['long_non_replay_mean'].items()}
        fig_L, ax_L, active_display_context_L = _plot_single_track_firing_rate_compare(x_frs, y_frs, active_context=final_context.adding_context_if_missing(filter_name='long'))


        ## Short Track Replay|Laps FR Figure
        neuron_replay_stats_df = neuron_replay_stats_table.dropna(subset=['short_replay_mean', 'short_non_replay_mean'], inplace=False)
        x_frs = {k:v for k,v in neuron_replay_stats_df['short_replay_mean'].items()}
        y_frs = {k:v for k,v in neuron_replay_stats_df['short_non_replay_mean'].items()}
        fig_S, ax_S, active_display_context_S = _plot_single_track_firing_rate_compare(x_frs, y_frs, active_context=final_context.adding_context_if_missing(filter_name='short'))

        ## Fit both the axes:
        fit_both_axes(ax_L, ax_S)

        def _perform_write_to_file_callback():
            active_out_figure_paths_L, *args_L = cls.output_figure(active_display_context_L, fig_L)
            active_out_figure_paths_S, *args_S = cls.output_figure(active_display_context_S, fig_S)
            return (active_out_figure_paths_L + active_out_figure_paths_S)

        if save_figure:
            active_out_figure_paths = _perform_write_to_file_callback()
        else:
            active_out_figure_paths = []

        graphics_output_dict = MatplotlibRenderPlots(name='across_sessions_long_and_short_firing_rate_replays_v_laps', figures=(fig_L, fig_S), axes=(ax_L, ax_S), context=(active_display_context_L, active_display_context_S), plot_data={'context': (active_display_context_L, active_display_context_S)}, saved_figures=active_out_figure_paths)

        return graphics_output_dict
    








# ==================================================================================================================== #
# 2024-01-29 - Across Session CSV Import and Plotting                                                                  #
# ==================================================================================================================== #
""" 

from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import plot_across_sessions_scatter_results, plot_histograms, plot_stacked_histograms

"""

import matplotlib.pyplot as plt

import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objs as go


def plotly_pre_post_delta_scatter(data_results_df: pd.DataFrame, out_scatter_fig=None, histogram_bins:int=25, px_scatter_kwargs=None,
                                   histogram_variable_name='P_Long', hist_kwargs=None,
                                   forced_range_y=[0.0, 1.0], time_delta_tuple=None):
    """ Plots a scatter plot of a variable pre/post delta, with a histogram on each end corresponding to the pre/post delta distribution
    
    px_scatter_kwargs: only used if out_scatter_fig is None
    time_delta_tuple=(earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end)


    Usage:

        import plotly.io as pio
        template: str = 'plotly_dark' # set plotl template
        pio.templates.default = template
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import plotly_pre_post_delta_scatter


        histogram_bins: int = 25

        new_laps_fig = plotly_pre_post_delta_scatter(data_results_df=deepcopy(all_sessions_laps_df), out_scatter_fig=fig_laps, histogram_bins=histogram_bins)
        new_laps_fig

    """
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objs as go

    unique_sessions = data_results_df['session_name'].unique()
    num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

    ## Extract the unique time bin sizes:
    time_bin_sizes: int = data_results_df['time_bin_size'].unique()
    num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)

    # f"Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes"
    # main_title: str = f"Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes"
    if num_unique_sessions == 1:
        # print(f'single-session mode')
        main_title: str = f"Session {px_scatter_kwargs.get('title', 'UNKNOWN')}"
    else:
        main_title: str = f"Across Sessions {px_scatter_kwargs.get('title', 'UNKNOWN')} ({num_unique_sessions} Sessions)"


    if num_unique_time_bins > 1:
        main_title = main_title + f" - {num_unique_time_bins} Time Bin Sizes"
    else:
        time_bin_size = time_bin_sizes[0]
        main_title = main_title + f" - time bin size: {time_bin_size} sec"


    
    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')


    # common_plot_kwargs = dict(color="time_bin_size")
    common_plot_kwargs = dict() # color=None
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs = hist_kwargs | dict(opacity=0.5, range_y=[0.0, 1.0], nbins=histogram_bins, barmode='overlay')
    # print(f'hist_kwargs: {hist_kwargs}')

    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]


    # ==================================================================================================================== #
    # Build Figure                                                                                                         #
    # ==================================================================================================================== #
    # creating subplots
    fig = sp.make_subplots(rows=1, cols=3, column_widths=[0.10, 0.80, 0.10], horizontal_spacing=0.01, shared_yaxes=True, column_titles=["Pre-delta", main_title, "Post-delta"])
    already_added_legend_entries = set()  # Keep track of trace names that are already added

    # Pre-Delta Histogram ________________________________________________________________________________________________ #
    # adding first histogram
    pre_delta_fig = px.histogram(pre_delta_df, y=histogram_variable_name, **common_plot_kwargs, **hist_kwargs, title="Pre-delta")
    print(f'len(pre_delta_fig.data): {len(pre_delta_fig.data)}')
    # time_bin_sizes
    for a_trace in pre_delta_fig.data:
        a_trace_name = a_trace.name
        if a_trace_name in already_added_legend_entries:
            # For already added trace categories, set showlegend to False
            a_trace.showlegend = False
        else:
            # For the first trace of each category, keep showlegend as True
            already_added_legend_entries.add(a_trace_name)
            a_trace.showlegend = True  # This is usually true by default, can be omitted

        fig.add_trace(a_trace, row=1, col=1)



    # Scatter Plot _______________________________________________________________________________________________________ #
    # adding scatter plot
    if out_scatter_fig is not None:
        for a_trace in out_scatter_fig.data:
            fig.add_trace(a_trace, row=1, col=2)
            # if forced_range_y is not None:
            #     fig.update_layout(yaxis=dict(range=forced_range_y))
    else:
        ## Create a new scatter plot:
        assert px_scatter_kwargs is not None
        out_scatter_fig = px.scatter(data_results_df, **px_scatter_kwargs)

        for i, a_trace in enumerate(out_scatter_fig.data):
            a_trace_name = a_trace.name
            if a_trace_name in already_added_legend_entries:
                # For already added trace categories, set showlegend to False
                a_trace.showlegend = False
            else:
                # For the first trace of each category, keep showlegend as True
                already_added_legend_entries.add(a_trace_name)
                a_trace.showlegend = True  # This is usually true by default, can be omitted
            
            # is_first_item: bool = (i == 0)
            # if (not is_first_item):
            #     a_trace['showlegend'] = False
                # a_trace.showlegend = False    
            # print(f'a_trace: {a_trace}')
            # a_trace = fig.add_trace(a_trace, row=1, col=2)
            fig.add_trace(a_trace, row=1, col=2)

        # if forced_range_y is not None:
        #     fig.update_layout(yaxis=dict(range=forced_range_y))


    # Post-Delta Histogram _______________________________________________________________________________________________ #
    # adding the second histogram
    post_delta_fig = px.histogram(post_delta_df, y=histogram_variable_name, **common_plot_kwargs, **hist_kwargs, title="Post-delta")

    for a_trace in post_delta_fig.data:
        a_trace_name = a_trace.name
        if a_trace_name in already_added_legend_entries:
            # For already added trace categories, set showlegend to False
            a_trace.showlegend = False
        else:
            # For the first trace of each category, keep showlegend as True
            a_trace.showlegend = True  # This is usually true by default, can be omitted
            already_added_legend_entries.add(a_trace_name)
            
        fig.add_trace(a_trace, row=1, col=3)


    # fig.update_layout(yaxis=dict(range=forced_range_y))
        

    if forced_range_y is not None:
        fig.update_layout(yaxis=dict(range=forced_range_y))

    fig.update_layout(yaxis=dict(range=forced_range_y), barmode='overlay')


    # Epoch Shapes
    if time_delta_tuple is not None:
        assert len(time_delta_tuple) == 3
        earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end = time_delta_tuple
        # Shifts the absolute times to delta-relative values, as would be needed to draw on a 'delta_aligned_start_t' axis:
        delta_relative_t_start, delta_relative_t_delta, delta_relative_t_end = np.array([earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end]) - t_delta
        _extras_output_dict = plotly_helper_add_epoch_shapes(fig, scatter_column_index=2, t_start=delta_relative_t_start, t_split=delta_relative_t_delta, t_end=delta_relative_t_end)
    else:
        _extras_output_dict = {}

    return fig


@function_attributes(short_name=None, tags=['plotly', 'helper', 'epoch'], input_requires=[], output_provides=[], uses=[], used_by=['_helper_build_figure'], creation_date='2024-03-01 13:58', related_items=[])
def plotly_helper_add_epoch_shapes(fig, scatter_column_index: int, t_start: float, t_split:float, t_end: float):
    """ adds shapes representing the epochs to the scatter plot at index scatter_column_index 
    
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plotly_helper_add_epoch_shapes
        _extras_output_dict = plotly_helper_add_epoch_shapes(fig, scatter_column_index=scatter_column, t_start=earliest_delta_aligned_t_start, t_split=t_split, t_end=latest_delta_aligned_t_end)


    """
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager

    _extras_output_dict = {}
    ## Get the track configs for the colors:
    long_short_display_config_manager = LongShortDisplayConfigManager()
    long_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.long_epoch_config.mpl_color)
    short_epoch_kwargs = dict(fillcolor=long_short_display_config_manager.short_epoch_config.mpl_color)
        
    row_column_kwargs = dict(row='all', col=scatter_column_index)

    ## new methods
    _extras_output_dict["y_zero_line"] = fig.add_hline(y=0.0, line=dict(color="rgba(0.2,0.2,0.2,.25)", width=9), **row_column_kwargs)
    vertical_divider_line = fig.add_vline(x=0.0, line=dict(color="rgba(0,0,0,.25)", width=3, ), **row_column_kwargs)

    # fig.add_hrect(y0=0.9, y1=2.6, line_width=0, fillcolor="red", opacity=0.2)

    blue_shape = fig.add_vrect(x0=t_start, x1=t_split, label=dict(text="Long", textposition="top center", font=dict(size=20, family="Times New Roman"), ), layer="below", opacity=0.5, line_width=1, **long_epoch_kwargs, **row_column_kwargs) # , fillcolor="green", opacity=0.25
    red_shape = fig.add_vrect(x0=t_split, x1=t_end, label=dict(text="Short", textposition="top center", font=dict(size=20, family="Times New Roman"), ), layer="below", opacity=0.5, line_width=1, **short_epoch_kwargs, **row_column_kwargs)

    _extras_output_dict["long_region"] = blue_shape
    _extras_output_dict["short_region"] = red_shape
    _extras_output_dict["divider_line"] = vertical_divider_line
    return _extras_output_dict


def _helper_build_figure(data_results_df: pd.DataFrame, histogram_bins:int=25, earliest_delta_aligned_t_start: float=0.0, latest_delta_aligned_t_end: float=666.0,
                                          enabled_time_bin_sizes=None, main_plot_mode: str = 'separate_row_per_session',
                                          **build_fig_kwargs):
    """ factored out of the subfunction in plot_across_sessions_scatter_results
    adds scatterplots as well
    Captures: None 
    """
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objects as go
    
    barmode='overlay'
    # barmode='stack'
    histogram_kwargs = dict(barmode=barmode)
    # px_histogram_kwargs = dict(nbins=histogram_bins, barmode='stack', opacity=0.5, range_y=[0.0, 1.0])
    scatter_title = build_fig_kwargs.pop('title', None)
    debug_print: bool = build_fig_kwargs.pop('debug_print', False)
    
    # Filter dataframe by chosen bin sizes
    if (enabled_time_bin_sizes is not None) and (len(enabled_time_bin_sizes) > 0):
        print(f'filtering data_results_df to enabled_time_bin_sizes: {enabled_time_bin_sizes}...')
        data_results_df = data_results_df[data_results_df.time_bin_size.isin(enabled_time_bin_sizes)]
        
    data_results_df = deepcopy(data_results_df)
    
    # convert time_bin_sizes column to a string so it isn't colored continuously
    data_results_df["time_bin_size"] = data_results_df["time_bin_size"].astype(str)

    
    unique_sessions = data_results_df['session_name'].unique()
    num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

    ## Extract the unique time bin sizes:
    time_bin_sizes: int = data_results_df['time_bin_size'].unique()
    num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)

    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
    
    ## Build KWARGS
    known_main_plot_modes = ['default', 'separate_facet_row_per_session', 'separate_row_per_session']
    assert main_plot_mode in known_main_plot_modes
    print(f'main_plot_mode: {main_plot_mode}')

    enable_histograms: bool = True
    enable_scatter_plot: bool = True
    enable_epoch_shading_shapes: bool = True
    px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0]} #, 'histnorm': 'probability density'
    
    if (main_plot_mode == 'default'):
        # main_plot_mode: str = 'default'
        enable_scatter_plot: bool = False
        num_cols: int = int(enable_scatter_plot) + 2 * int(enable_histograms) # 2 histograms and one scatter
        print(f'num_cols: {num_cols}')
        is_col_included = np.array([enable_histograms, enable_scatter_plot, enable_histograms])
        column_widths = list(np.array([0.1, 0.8, 0.1])[is_col_included])
        column_titles = ["Pre-delta", f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]
        
        # sp_make_subplots_kwargs = {'rows': 1, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': column_titles}
        sp_make_subplots_kwargs = {'rows': 1, 'cols': num_cols, 'column_widths': column_widths, 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': list(np.array(column_titles)[is_col_included])}
        # px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'session_name', 'size': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        
        # px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0], 'histnorm': 'probability'}
        
    elif (main_plot_mode == 'separate_facet_row_per_session'):
        # main_plot_mode: str = 'separate_facet_row_per_session'
        raise NotImplementedError(f"DOES NOT WORK")
        sp_make_subplots_kwargs = {'rows': 1, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'shared_yaxes': True, 'column_titles': ["Pre-delta",f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'title': scatter_title, 'range_y': [0.0, 1.0],
                            'facet_row': 'session_name', 'facet_row_spacing': 0.04, # 'facet_col_wrap': 2, 'facet_col_spacing': 0.04,
                            'height': (num_unique_sessions*200), 'width': 1024,
                            'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        px_histogram_kwargs = {**px_histogram_kwargs,
                                'facet_row': 'session_name', 'facet_row_spacing': 0.04, 'facet_col_wrap': 2, 'facet_col_spacing': 0.04, 'height': (num_unique_sessions*200), 'width': 1024}
        enable_histograms = False
        enable_epoch_shading_shapes = False

    elif (main_plot_mode == 'separate_row_per_session'):
        # main_plot_mode: str = 'separate_row_per_session'
        # , subplot_titles=("Plot 1", "Plot 2")
        # column_titles = ["Pre-delta", f"{scatter_title} - Across Sessions ({num_unique_sessions} Sessions) - {num_unique_time_bins} Time Bin Sizes", "Post-delta"]
        column_titles = ["Pre-delta", f"{scatter_title}", "Post-delta"]
        session_titles = [str(v) for v in unique_sessions]
        subplot_titles = []
        for a_row_title in session_titles:
            subplot_titles.extend(["Pre-delta", f"{a_row_title}", "Post-delta"])
        # subplot_titles = [["Pre-delta", f"{a_row_title}", "Post-delta"] for a_row_title in session_titles].flatten()
        
        sp_make_subplots_kwargs = {'rows': num_unique_sessions, 'cols': 3, 'column_widths': [0.1, 0.8, 0.1], 'horizontal_spacing': 0.01, 'vertical_spacing': 0.04, 'shared_yaxes': True,
                                    'column_titles': column_titles,
                                    'row_titles': session_titles,
                                    'subplot_titles': subplot_titles,
                                    }
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': 'P_Long', 'color': 'time_bin_size', 'range_y': [0.0, 1.0],
                            'height': (num_unique_sessions*200), 'width': 1024,
                            'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size'}}
        # px_histogram_kwargs = {'nbins': histogram_bins, 'barmode': barmode, 'opacity': 0.5, 'range_y': [0.0, 1.0], 'histnorm': 'probability'}
    else:
        raise ValueError(f'main_plot_mode is not a known mode: main_plot_mode: "{main_plot_mode}", known modes: known_main_plot_modes: {known_main_plot_modes}')
    

    def __sub_subfn_plot_histogram(fig, histogram_data_df, hist_title="Post-delta", row=1, col=3):
        """ captures: px_histogram_kwargs, histogram_kwargs
        
        """
        is_first_item: bool = ((row == 1) and (col == 1))
        a_hist_fig = px.histogram(histogram_data_df, y="P_Long", color="time_bin_size", **px_histogram_kwargs, title=hist_title)

        for a_trace in a_hist_fig.data:
            if debug_print:
                print(f'a_trace.legend: {a_trace.legend}, a_trace.legendgroup: {a_trace.legendgroup}, a_trace.legendgrouptitle: {a_trace.legendgrouptitle}, a_trace.showlegend: {a_trace.showlegend}, a_trace.offsetgroup: {a_trace.offsetgroup}')
            
            if (not is_first_item):
                a_trace.showlegend = False
                
            fig.add_trace(a_trace, row=row, col=col)
            fig.update_layout(yaxis=dict(range=[0.0, 1.0]), **histogram_kwargs)
            

    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]
    # creating subplots
    fig = sp.make_subplots(**sp_make_subplots_kwargs)
    next_subplot_col_idx: int = 1 
    
    # Pre-Delta Histogram ________________________________________________________________________________________________ #
    # adding first histogram
    if enable_histograms:
        histogram_col_idx: int = next_subplot_col_idx
        if (main_plot_mode == 'separate_row_per_session'):
            for a_session_i, a_session_name in enumerate(unique_sessions):              
                row_index: int = a_session_i + 1 # 1-indexed
                a_session_pre_delta_df: pd.DataFrame = pre_delta_df[pre_delta_df['session_name'] == a_session_name]
                __sub_subfn_plot_histogram(fig, histogram_data_df=a_session_pre_delta_df, hist_title="Pre-delta", row=row_index, col=histogram_col_idx)
                fig.update_yaxes(title_text=f"{a_session_name}", row=row_index, col=1)
                                
        else:
            __sub_subfn_plot_histogram(fig, histogram_data_df=pre_delta_df, hist_title="Pre-delta", row=1, col=histogram_col_idx)
        next_subplot_col_idx = next_subplot_col_idx + 1 # increment the next column

    # Scatter Plot _______________________________________________________________________________________________________ #
    if enable_scatter_plot:
        scatter_column: int = next_subplot_col_idx # default 2
        
        if (main_plot_mode == 'separate_row_per_session'):
            for a_session_i, a_session_name in enumerate(unique_sessions):              
                row_index: int = a_session_i + 1 # 1-indexed
                is_first_item: bool = ((row_index == 1) and (scatter_column == 1))
                a_session_data_results_df: pd.DataFrame = data_results_df[data_results_df['session_name'] == a_session_name]
                #  fig.add_scatter(x=a_session_data_results_df['delta_aligned_start_t'], y=a_session_data_results_df['P_Long'], row=row_index, col=2, name=a_session_name)
                scatter_fig = px.scatter(a_session_data_results_df, **px_scatter_kwargs, title=f"{a_session_name}")
                for a_trace in scatter_fig.data:
                    if (not is_first_item):
                        a_trace.showlegend = False
    
                    fig.add_trace(a_trace, row=row_index, col=scatter_column)
                    # fig.update_layout(yaxis=dict(range=[0.0, 1.0]))

                fig.update_xaxes(title_text="Delta-Relative Time (seconds)", row=row_index, col=scatter_column)
                #  fig.update_yaxes(title_text=f"{a_session_name}", row=row_index, col=scatter_column)
                fig.update_layout(yaxis=dict(range=[0.0, 1.0]))
                
            #  fig.update_xaxes(matches='x')
        
        else:
            scatter_fig = px.scatter(data_results_df, **px_scatter_kwargs)

            # for a_trace in scatter_traces:
            for a_trace in scatter_fig.data:
                # a_trace.legend = "legend"
                # a_trace['visible'] = 'legendonly'
                # a_trace['visible'] = 'legendonly' # 'legendonly', # this trace will be hidden initially
                fig.add_trace(a_trace, row=1, col=scatter_column)
                fig.update_layout(yaxis=dict(range=[0.0, 1.0]))
            
            # Update xaxis properties
            fig.update_xaxes(title_text="Delta-Relative Time (seconds)", row=1, col=scatter_column)
            
        next_subplot_col_idx = next_subplot_col_idx + 1 # increment the next column
    # else:
    #     # no scatter
    #     next_subplot_col_idx = next_subplot_col_idx
        

    # Post-Delta Histogram _______________________________________________________________________________________________ #
    # adding the second histogram
    if enable_histograms:
        histogram_col_idx: int = next_subplot_col_idx #default 3
        
        if (main_plot_mode == 'separate_row_per_session'):
            for a_session_i, a_session_name in enumerate(unique_sessions):              
                row_index: int = a_session_i + 1 # 1-indexed
                a_session_post_delta_df: pd.DataFrame = post_delta_df[post_delta_df['session_name'] == a_session_name]
                __sub_subfn_plot_histogram(fig, histogram_data_df=a_session_post_delta_df, hist_title="Post-delta", row=row_index, col=histogram_col_idx)                
        else:
            __sub_subfn_plot_histogram(fig, histogram_data_df=post_delta_df, hist_title="Post-delta", row=1, col=histogram_col_idx)
        
        next_subplot_col_idx = next_subplot_col_idx + 1 # increment the next column
        
    ## Add the delta indicator:
    if (enable_scatter_plot and enable_epoch_shading_shapes):
        
        t_split: float = 0.0



        #TODO 2024-02-02 04:36: - [ ] Should get the specific session t_start/t_end instead of using the general `earliest_delta_aligned_t_start`
        # _extras_output_dict = PlottingHelpers.helper_plotly_add_long_short_epoch_indicator_regions(fig, t_split=t_split, t_start=earliest_delta_aligned_t_start, t_end=latest_delta_aligned_t_end, build_only=True)
        # for a_shape_name, a_shape in _extras_output_dict.items():
        #     if (main_plot_mode == 'separate_row_per_session'):
        #         for a_session_i, a_session_name in enumerate(unique_sessions):    
        #             row_index: int = a_session_i + 1 # 1-indexed
        #             fig.add_shape(a_shape, name=a_shape_name, row=row_index, col=scatter_column)
        #     else:
        #         fig.add_shape(a_shape, name=a_shape_name, row=1, col=scatter_column)

        ## Inputs: fig, t_start: float, t_end: float
        _extras_output_dict = plotly_helper_add_epoch_shapes(fig, scatter_column_index=scatter_column, t_start=earliest_delta_aligned_t_start, t_split=t_split, t_end=latest_delta_aligned_t_end)


    # Update title and height
        
    if (main_plot_mode == 'separate_row_per_session'):
        row_height = 250
        required_figure_height = (num_unique_sessions*row_height)
    elif (main_plot_mode == 'separate_facet_row_per_session'):
        row_height = 200
        required_figure_height = (num_unique_sessions*row_height)
    else:
        required_figure_height = 700
        
    fig.update_layout(title_text=scatter_title, width=2048, height=required_figure_height)
    fig.update_layout(yaxis=dict(range=[0.0, 1.0]), template='plotly_dark')
    # Update y-axis range for all created figures
    fig.update_yaxes(range=[0.0, 1.0])

    # Add a footer
    fig.update_layout(
        legend_title_text='tBin Size',
        # annotations=[
        #     dict(x=0.5, y=-0.15, showarrow=False, text="Footer text here", xref="paper", yref="paper")
        # ],
        # margin=dict(b=140), # increase bottom margin to show the footer
    )
    return fig



# def plotly_plot_1D_most_likely_position_comparsions(time_window_centers, xbin, posterior): # , ax=None
#     """ 
#     Analagous to `plot_1D_most_likely_position_comparsions`
#     """
#     import plotly.graph_objects as go
    
#     # Posterior distribution heatmap:
#     assert posterior is not None

#     # print(f'time_window_centers: {time_window_centers}, posterior: {posterior}')
#     # Compute extents
#     xmin, xmax, ymin, ymax = (time_window_centers[0], time_window_centers[-1], xbin[0], xbin[-1])
#     # Create a heatmap
#     fig = go.Figure(data=go.Heatmap(
#                     z=posterior,
#                     x=time_window_centers,  y=xbin, 
#                     zmin=0, zmax=1,
#                     # colorbar=dict(title='z'),
#                     showscale=False,
#                     colorscale='Viridis', # The closest equivalent to Matplotlib 'viridis'
#                     hoverongaps = False))

#     # Update layout
#     fig.update_layout(
#         autosize=False,
#         xaxis=dict(type='linear', range=[xmin, xmax]),
#         yaxis=dict(type='linear', range=[ymin, ymax]))

#     return fig

@function_attributes(short_name=None, tags=['plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 06:04', related_items=[])
def plotly_plot_1D_most_likely_position_comparsions(time_window_centers_list, xbin, posterior_list): # , ax=None
    """ 
    Analagous to `plot_1D_most_likely_position_comparsions`
    """
    import plotly.graph_objects as go
    import plotly.subplots as sp
    # Ensure input lists are of the same length
    assert len(time_window_centers_list) == len(posterior_list)

    # Compute layout grid dimensions
    num_rows = len(time_window_centers_list)

    # Create subplots
    fig = sp.make_subplots(rows=num_rows, cols=1)

    for row_idx, (time_window_centers, posterior) in enumerate(zip(time_window_centers_list, posterior_list)):
        # Compute extents
        xmin, xmax, ymin, ymax = (time_window_centers[0], time_window_centers[-1], xbin[0], xbin[-1])
        # Add heatmap trace to subplot
        fig.add_trace(go.Heatmap(
                        z=posterior,
                        x=time_window_centers,  y=xbin, 
                        zmin=0, zmax=1,
                        # colorbar=dict(title='z'),
                        showscale=False,
                        colorscale='Viridis', # The closest equivalent to Matplotlib 'viridis'
                        hoverongaps = False),
                      row=row_idx+1, col=1)

        # Update layout for each subplot
        fig.update_xaxes(range=[xmin, xmax], row=row_idx+1, col=1)
        fig.update_yaxes(range=[ymin, ymax], row=row_idx+1, col=1)

    return fig


@function_attributes(short_name=None, tags=['plotly', 'blue_yellow'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 06:04', related_items=[])
def plot_blue_yellow_points(a_df, specific_point_list):
    """ Renders a figure containing one or more yellow-blue plots (marginals) for a given hoverred point. Used with Dash app.
    
    specific_point_list: List[Dict] - specific_point_list = [{'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03, 'epoch_idx': 0, 'delta_aligned_start_t': -713.908702568122}]
    """
    time_window_centers_list = []
    posterior_list = []

    # for a_single_epoch_row_idx, a_single_epoch_idx in enumerate(selected_epoch_idxs):
    for a_single_epoch_row_idx, a_single_custom_data_dict in enumerate(specific_point_list):
        # a_single_epoch_idx = selected_epoch_idxs[a_single_epoch_row_idx]
        a_single_epoch_idx: int = int(a_single_custom_data_dict['epoch_idx'])
        a_single_session_name: str = str(a_single_custom_data_dict['session_name'])
        a_single_time_bin_size: float = float(a_single_custom_data_dict['time_bin_size'])
        ## Get the dataframe entries:
        a_single_epoch_df = a_df.copy()
        a_single_epoch_df = a_single_epoch_df[a_single_epoch_df.epoch_idx == a_single_epoch_idx] ## filter by epoch idx
        a_single_epoch_df = a_single_epoch_df[a_single_epoch_df.session_name == a_single_session_name] ## filter by session
        a_single_epoch_df = a_single_epoch_df[a_single_epoch_df.time_bin_size == a_single_time_bin_size] ## filter by time-bin-size	

        posterior = a_single_epoch_df[['P_Long', 'P_Short']].to_numpy().T
        time_window_centers = a_single_epoch_df['delta_aligned_start_t'].to_numpy()
        xbin = np.arange(2)
        time_window_centers_list.append(time_window_centers)
        posterior_list.append(posterior)
        
        # fig = plotly_plot_1D_most_likely_position_comparsions(time_window_centers=time_window_centers, xbin=xbin, posterior=posterior)
        # fig.show()
        
    fig = plotly_plot_1D_most_likely_position_comparsions(time_window_centers_list=time_window_centers_list, xbin=xbin, posterior_list=posterior_list)
    return fig

@function_attributes(short_name=None, tags=['Dash', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 06:04', related_items=[])
def _build_dash_app(final_dfs_dict, earliest_delta_aligned_t_start: float, latest_delta_aligned_t_end: float):
    """ builds an interactive Across Sessions Dash app
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _build_dash_app
    
    app = _build_dash_app(final_dfs_dict, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end)
    """
    from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
    from dash.dash_table import DataTable, FormatTemplate
    from dash.dash_table.Format import Format, Padding

    import dash_bootstrap_components as dbc
    import pandas as pd
    from pathlib import Path
    # import plotly.express as px
    import plotly.io as pio
    template: str = 'plotly_dark' # set plotl template
    pio.templates.default = template


    ## DATA:    
    options_list = list(final_dfs_dict.keys())
    initial_option = options_list[0]
    initial_dataframe: pd.DataFrame = final_dfs_dict[initial_option].copy()
    unique_sessions: List[str] = initial_dataframe['session_name'].unique().tolist()
    num_unique_sessions: int = initial_dataframe['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    assert 'epoch_idx' in initial_dataframe.columns

    ## Extract the unique time bin sizes:
    time_bin_sizes: List[float] = initial_dataframe['time_bin_size'].unique().tolist()
    num_unique_time_bins: int = initial_dataframe.time_bin_size.nunique(dropna=True)
    print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
    enabled_time_bin_sizes = [time_bin_sizes[0], time_bin_sizes[-1]] # [0.03, 0.058, 0.10]

    ## prune to relevent columns:
    all_column_names = [
        ['P_Long', 'P_Short', 'P_LR', 'P_RL'],
        ['delta_aligned_start_t'], # 'lap_idx', 
        ['session_name'],
        ['time_bin_size'],
        ['epoch_idx'],
    ]
    all_column_names_flat = [item for sublist in all_column_names for item in sublist]
    print(f'\tall_column_names_flat: {all_column_names_flat}')
    initial_dataframe = initial_dataframe[all_column_names_flat]

    # Initialize the app
    # app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
    # Slate
    
    # # money = FormatTemplate.money(2)
    # percentage = FormatTemplate.percentage(2)
    # # percentage = FormatTemplate.deci
    # column_designators = [
    #     dict(id='a', name='delta_aligned_start_t', type='numeric', format=Format()),
    #     dict(id='a', name='session_name', type='text', format=Format()),
    #     dict(id='a', name='time_bin_size', type='numeric', format=Format(padding=Padding.yes).padding_width(9)),
    #     dict(id='a', name='P_Long', type='numeric', format=dict(specifier='05')),
    #     dict(id='a', name='P_LR', type='numeric', format=dict(specifier='05')),
    # ]

    # App layout
    app.layout = dbc.Container([
        dbc.Row([
                html.Div(children='My Custom App with Data, Graph, and Controls'),
                html.Hr()
        ]),
        dbc.Row([
            dbc.Col(dcc.RadioItems(options=options_list, value=initial_option, id='controls-and-radio-item'), width=3),
            dbc.Col(dcc.Checklist(options=time_bin_sizes, value=enabled_time_bin_sizes, id='time-bin-checkboxes', inline=True), width=3), # Add CheckboxGroup for time_bin_sizes
        ]),
        dbc.Row([
            dbc.Col(DataTable(data=initial_dataframe.to_dict('records'), page_size=16, id='tbl-datatable',
                        # columns=column_designators,
                        columns=[{"name": i, "id": i} for i in initial_dataframe.columns],
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'
                            },
                            {
                                'if': {'row_index': 'even'},
                                'backgroundColor': 'rgb(70, 70, 70)',
                                'color': 'white'
                            },
                            {
                                'if': {'column_editable': True},
                                'backgroundColor': 'rgb(100, 100, 100)',
                                'color': 'white'
                            }
                        ],
                        style_header={
                            'backgroundColor': 'rgb(30, 30, 30)',
                            'color': 'white'
                        },
                        row_selectable="multi",
                ) # end DataTable
            , align='stretch', width=3),
            dbc.Col(dcc.Graph(figure={}, id='controls-and-graph', hoverData={'points': [{'customdata': []}]},
                            ), align='end', width=9),
        ]), # end Row
        dbc.Row(dcc.Graph(figure={}, id='selected-yellow-blue-marginals-graph')),
    ]) # end Container

    # Add controls to build the interaction
    @callback(
        Output(component_id='controls-and-graph', component_property='figure'),
        [Input(component_id='controls-and-radio-item', component_property='value'),
        Input(component_id='time-bin-checkboxes', component_property='value'),
        ]
    )
    def update_graph(col_chosen, chose_bin_sizes):
        print(f'update_graph(col_chosen: {col_chosen}, chose_bin_sizes: {chose_bin_sizes})')
        data_results_df: pd.DataFrame = final_dfs_dict[col_chosen].copy()
        # Filter dataframe by chosen bin sizes
        data_results_df = data_results_df[data_results_df.time_bin_size.isin(chose_bin_sizes)]
        
        unique_sessions: List[str] = data_results_df['session_name'].unique().tolist()
        num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

        ## Extract the unique time bin sizes:
        time_bin_sizes: List[float] = data_results_df['time_bin_size'].unique().tolist()
        num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)
        print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
        enabled_time_bin_sizes = chose_bin_sizes
        fig = _helper_build_figure(data_results_df=data_results_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode='separate_row_per_session', title=f"{col_chosen}")        
        # 'delta_aligned_start_t', 'session_name', 'time_bin_size'
        tuples_data = data_results_df[['session_name', 'time_bin_size', 'epoch_idx', 'delta_aligned_start_t']].to_dict(orient='records')
        print(f'tuples_data: {tuples_data}')
        fig.update_traces(customdata=tuples_data)
        fig.update_layout(hovermode='closest') # margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        return fig


    @callback(
        Output(component_id='tbl-datatable', component_property='data'),
        [Input(component_id='controls-and-radio-item', component_property='value'),
            Input(component_id='time-bin-checkboxes', component_property='value'),
        ]
    )
    def update_datatable(col_chosen, chose_bin_sizes):
        """ captures: final_dfs_dict, all_column_names_flat
        """
        print(f'update_datatable(col_chosen: {col_chosen}, chose_bin_sizes: {chose_bin_sizes})')
        a_df = final_dfs_dict[col_chosen].copy()
        ## prune to relevent columns:
        a_df = a_df[all_column_names_flat]
        # Filter dataframe by chosen bin sizes
        a_df = a_df[a_df.time_bin_size.isin(chose_bin_sizes)]
        data = a_df.to_dict('records')
        return data

    @callback(
        Output('selected-yellow-blue-marginals-graph', 'figure'),
        [Input(component_id='controls-and-radio-item', component_property='value'),
        Input(component_id='time-bin-checkboxes', component_property='value'),
        Input(component_id='tbl-datatable', component_property='selected_rows'),
        Input(component_id='controls-and-graph', component_property='hoverData'),
        ]
    )
    def get_selected_rows(col_chosen, chose_bin_sizes, indices, hoverred_rows):
        print(f'get_selected_rows(col_chosen: {col_chosen}, chose_bin_sizes: {chose_bin_sizes}, indices: {indices}, hoverred_rows: {hoverred_rows})')
        data_results_df: pd.DataFrame = final_dfs_dict[col_chosen].copy()
        data_results_df = data_results_df[data_results_df.time_bin_size.isin(chose_bin_sizes)] # Filter dataframe by chosen bin sizes
        # ## prune to relevent columns:
        data_results_df = data_results_df[all_column_names_flat]
        
        unique_sessions: List[str] = data_results_df['session_name'].unique().tolist()
        num_unique_sessions: int = data_results_df['session_name'].nunique(dropna=True) # number of unique sessions, ignoring the NA entries

        ## Extract the unique time bin sizes:
        time_bin_sizes: List[float] = data_results_df['time_bin_size'].unique().tolist()
        num_unique_time_bins: int = data_results_df.time_bin_size.nunique(dropna=True)
        # print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bins}')
        enabled_time_bin_sizes = chose_bin_sizes

        print(f'hoverred_rows: {hoverred_rows}')
        # get_selected_rows(col_chosen: AcrossSession_Laps_per-Epoch, chose_bin_sizes: [0.03, 0.1], indices: None, hoverred_rows: {'points': [{'curveNumber': 26, 'pointNumber': 8, 'pointIndex': 8, 'x': -713.908702568122, 'y': 0.6665361938589899, 'bbox': {'x0': 1506.896, 'x1': 1512.896, 'y0': 283.62, 'y1': 289.62}, 'customdata': {'delta_aligned_start_t': -713.908702568122, 'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03}}]})
        # hoverred_rows: 
        hoverred_row_points = hoverred_rows.get('points', [])
        num_hoverred_points: int = len(hoverred_row_points)
        extracted_custom_data = [p['customdata'] for p in hoverred_row_points if (p.get('customdata', None) is not None)] # {'delta_aligned_start_t': -713.908702568122, 'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03}
        num_custom_data_hoverred_points: int = len(extracted_custom_data)

        print(f'extracted_custom_data: {extracted_custom_data}')
        # {'points': [{'curveNumber': 26, 'pointNumber': 8, 'pointIndex': 8, 'x': -713.908702568122, 'y': 0.6665361938589899, 'bbox': {'x0': 1506.896, 'x1': 1512.896, 'y0': 283.62, 'y1': 289.62}, 'customdata': {'delta_aligned_start_t': -713.908702568122, 'session_name': 'kdiba_vvp01_one_2006-4-10_12-25-50', 'time_bin_size': 0.03}}]}
            # selection empty!

        # a_df = final_dfs_dict[col_chosen].copy()
        # ## prune to relevent columns:
        # a_df = a_df[all_column_names_flat]
        # # Filter dataframe by chosen bin sizes
        # a_df = a_df[a_df.time_bin_size.isin(chose_bin_sizes)]
        # data = a_df.to_dict('records')
        if (indices is not None) and (len(indices) > 0):
            selected_rows = data_results_df.iloc[indices, :]
            print(f'\tselected_rows: {selected_rows}')
        else:
            print(f'\tselection empty!')
            
        if (extracted_custom_data is not None) and (num_custom_data_hoverred_points > 0):
            # selected_rows = data_results_df.iloc[indices, :]
            print(f'\tnum_custom_data_hoverred_points: {num_custom_data_hoverred_points}')
            fig = plot_blue_yellow_points(a_df=data_results_df.copy(), specific_point_list=extracted_custom_data)
        else:
            print(f'\thoverred points empty!')
            fig = go.Figure()

        return fig

    return app





@function_attributes(short_name=None, tags=['scatter', 'multi-session', 'plot', 'figure', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_across_sessions_scatter_results(directory: Union[Path, str], concatenated_laps_df: pd.DataFrame, concatenated_ripple_df: pd.DataFrame,
                                          earliest_delta_aligned_t_start: float=0.0, latest_delta_aligned_t_end: float=666.0,
                                          enabled_time_bin_sizes=None, main_plot_mode: str = 'separate_row_per_session',
                                          laps_title_prefix: str = f"Laps", ripple_title_prefix: str = f"Ripples",
                                          save_figures=False, figure_save_extension='.png', debug_print=False):
    """ takes the directory containing the .csv pairs that were exported by `export_marginals_df_csv`

    - Processes both ripple and laps
    - generates a single column of plots with the scatter plot in the middle flanked on both sides by the Pre/Post-delta histograms
    

    Produces and then saves figures out the the f'{directory}/figures/' subfolder

    Unknowingly captured: session_name
    
    - [ ] Truncate each session to their start/end instead of the global x bounds.
    
    
    """
    from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objects as go
    # import plotly.graph_objs as go
    
    # def _subfn_build_figure(data, **build_fig_kwargs):
    #     return go.Figure(data=data, **(dict(layout_yaxis_range=[0.0, 1.0]) | build_fig_kwargs))
    
    # def _subfn_build_figure(data_results_df: pd.DataFrame, **build_fig_kwargs):
    #     # return go.Figure(data=data, **(dict(layout_yaxis_range=[0.0, 1.0]) | build_fig_kwargs))
    #     scatter_title = build_fig_kwargs.pop('title', None) 
    #     return go.Figure(px.scatter(data_results_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size', title=scatter_title), layout_yaxis_range=[0.0, 1.0])
    
    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    if not isinstance(directory, Path):
        directory = Path(directory).resolve()
    assert directory.exists()
    print(f'plot_across_sessions_results(directory: {directory})')
    if save_figures:
        # Create a 'figures' subfolder if it doesn't exist
        figures_folder = Path(directory, 'figures')
        figures_folder.mkdir(parents=False, exist_ok=True)
        assert figures_folder.exists()
        print(f'\tfigures_folder: {file_uri_from_path(figures_folder)}')
    
    # Create an empty list to store the figures
    all_figures = []

    ## delta_t aligned:
    # Create a bubble chart for laps
    laps_num_unique_sessions: int = concatenated_laps_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    laps_num_unique_time_bins: int = concatenated_laps_df.time_bin_size.nunique(dropna=True)
    laps_title_string_suffix: str = f'{laps_num_unique_sessions} Sessions'
    laps_title: str = f"{laps_title_prefix} - {laps_title_string_suffix}"
    fig_laps = _helper_build_figure(data_results_df=concatenated_laps_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode=main_plot_mode, title=laps_title)

    # Create a bubble chart for ripples
    ripple_num_unique_sessions: int = concatenated_ripple_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    ripple_num_unique_time_bins: int = concatenated_ripple_df.time_bin_size.nunique(dropna=True)
    ripple_title_string_suffix: str = f'{ripple_num_unique_sessions} Sessions'
    ripple_title: str = f"{ripple_title_prefix} - {ripple_title_string_suffix}"
    fig_ripples = _helper_build_figure(data_results_df=concatenated_ripple_df, histogram_bins=25, earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, enabled_time_bin_sizes=enabled_time_bin_sizes, main_plot_mode=main_plot_mode, title=ripple_title)

    if save_figures:
        # Save the figures to the 'figures' subfolder
        assert figure_save_extension is not None
        if isinstance(figure_save_extension, str):
             figure_save_extension = [figure_save_extension] # a list containing only this item
        
        print(f'\tsaving figures...')
        for a_fig_save_extension in figure_save_extension:
            if a_fig_save_extension.lower() == '.html':
                 a_save_fn = lambda a_fig, a_save_name: a_fig.write_html(a_save_name)
            else:
                 a_save_fn = lambda a_fig, a_save_name: a_fig.write_image(a_save_name)
    
            fig_laps_name = Path(figures_folder, f"{laps_title_string_suffix.replace(' ', '-')}_{laps_title_prefix.lower()}_marginal{a_fig_save_extension}").resolve()
            print(f'\tsaving "{file_uri_from_path(fig_laps_name)}"...')
            a_save_fn(fig_laps, fig_laps_name)
            fig_ripple_name = Path(figures_folder, f"{ripple_title_string_suffix.replace(' ', '-')}_{ripple_title_prefix.lower()}_marginal{a_fig_save_extension}").resolve()
            print(f'\tsaving "{file_uri_from_path(fig_ripple_name)}"...')
            a_save_fn(fig_ripples, fig_ripple_name)
            

    # Append both figures to the list
    all_figures.append((fig_laps, fig_ripples))
    
    return all_figures


@function_attributes(short_name=None, tags=['histogram', 'multi-session', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_histograms(data_results_df: pd.DataFrame, data_type: str, session_spec: str, time_bin_duration_str: str, column='P_Long', **kwargs) -> None:
    """ plots a set of two histograms in subplots, split at the delta for each session.
    from PendingNotebookCode import plot_histograms
    
    """
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram #TODO 2024-01-02 12:41: - [ ] Is this where the Qt5 Import dependency Pickle complains about is coming from?
    layout = kwargs.pop('layout', 'none')
    defer_show = kwargs.pop('defer_show', False)
    
    fig = plt.figure(layout=layout, **kwargs) # layout="constrained", 
    ax_dict = fig.subplot_mosaic(
        [
            ["epochs_pre_delta", ".", "epochs_post_delta"],
        ],
        # set the height ratios between the rows
        # height_ratios=[8, 1],
        # height_ratios=[1, 1],
        # set the width ratios between the columns
        # width_ratios=[1, 8, 8, 1],
        sharey=True,
        gridspec_kw=dict(wspace=0.25, hspace=0.25) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
    )

    histogram_kwargs = dict(orientation="horizontal", bins=25)
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
    # plot pre-delta histogram
    pre_delta_df.hist(ax=ax_dict['epochs_pre_delta'], column=column, **histogram_kwargs)
    ax_dict['epochs_pre_delta'].set_title(f'{descriptor_str} - pre-$\Delta$ time bins')

    # plot post-delta histogram
    post_delta_df.hist(ax=ax_dict['epochs_post_delta'], column=column, **histogram_kwargs)
    ax_dict['epochs_post_delta'].set_title(f'{descriptor_str} - post-$\Delta$ time bins')
    if not defer_show:
        fig.show()
    return MatplotlibRenderPlots(name='plot_histograms', figures=[fig], axes=ax_dict)


@function_attributes(short_name=None, tags=['histogram', 'stacked', 'multi-session', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-29 20:47', related_items=[])
def plot_stacked_histograms(data_results_df: pd.DataFrame, data_type: str, session_spec: str, time_bin_duration_str: str, **kwargs) -> None:
    """ plots a colorful stacked histogram for each of the many time-bin sizes
    """
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram #TODO 2024-01-02 12:41: - [ ] Is this where the Qt5 Import dependency Pickle complains about is coming from?
    layout = kwargs.pop('layout', 'none')
    defer_show = kwargs.pop('defer_show', False)
    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    figure_identifier: str = f"{descriptor_str}_PrePostDelta"

    fig = plt.figure(num=figure_identifier, clear=True, figsize=(12, 2), layout=layout, **kwargs) # layout="constrained", 
    fig.suptitle(f'{descriptor_str}')
    
    ax_dict = fig.subplot_mosaic(
        [
            # ["epochs_pre_delta", ".", "epochs_post_delta"],
             ["epochs_pre_delta", "epochs_post_delta"],
        ],
        sharey=True,
        gridspec_kw=dict(wspace=0.25, hspace=0.25) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
    )
    
    histogram_kwargs = dict(orientation="horizontal", bins=25)
    
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    time_bin_sizes: int = pre_delta_df['time_bin_size'].unique()
    
    # plot pre-delta histogram:
    for time_bin_size in time_bin_sizes:
        df_tbs = pre_delta_df[pre_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(ax=ax_dict['epochs_pre_delta'], alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 
    
    ax_dict['epochs_pre_delta'].set_title(f'pre-$\Delta$ time bins')
    ax_dict['epochs_pre_delta'].legend()

    # plot post-delta histogram:
    time_bin_sizes: int = post_delta_df['time_bin_size'].unique()
    for time_bin_size in time_bin_sizes:
        df_tbs = post_delta_df[post_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(ax=ax_dict['epochs_post_delta'], alpha=0.5, label=str(time_bin_size), **histogram_kwargs) 
    
    ax_dict['epochs_post_delta'].set_title(f'post-$\Delta$ time bins')
    ax_dict['epochs_post_delta'].legend()
    
    if not defer_show:
        fig.show()
    return MatplotlibRenderPlots(name='plot_stacked_histograms', figures=[fig], axes=ax_dict)

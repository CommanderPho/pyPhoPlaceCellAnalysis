from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.General.Batch.runBatch import ConcreteSessionFolder

"""
Concerned with aggregating data (raw and computed results) across multiple sessions.
    Previously (Pre 2023-07-31) everything was designed in terms of a single session: the entire `NeuropyPipeline` object represents a single recording session - although it might slice this session several different ways and process it with several different sets of computation parameters

    All NeuropyPipelines (each session) is designed to be processed completely independently from each other with no codependencies. This enables trivial parallel processing of each session and complete encapsulation of the loogic for that session.

    As a result of this design decision, anything that aims to compute statistics aggregated across multiple sessions or combine/compare values between sessions must be implemented here.
"""


""" DESIGN DECISION/CONSTRAINT: This file should not focus on the process of directing the individual session pipeline computations (it's not a parallel processing manager) but instead on processing the data with a specified set of completed session pipelines.

"""
import sys
import re
from copy import deepcopy

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import nptyping as ND
from nptyping import NDArray
import numpy as np
import pandas as pd
from attrs import define, field, Factory
import shutil # copy_files_in_filelist_to_dest
from pyphocorehelpers.exception_helpers import CapturedException
import tables as tb
from tables import (
    Group,    StringCol,  EnumCol
)
import seaborn as sns
# from pyphocorehelpers.indexing_helpers import partition, safe_pandas_get_group
from datetime import datetime

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import  convert_filelist_to_new_parent
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.assertion_helpers import Assert

# NeuroPy (Diba Lab Python Repo) Loading
## For computation parameters:
from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update
from neuropy.core.neuron_identities import  neuronTypesEnum, NeuronIdentityTable
from neuropy.utils.mixins.HDF5_representable import HDF_Converter
from neuropy.utils.indexing_helpers import PandasHelpers
from neuropy.utils.debug_helpers import parameter_sweeps 

from pyphocorehelpers.Filesystem.metadata_helpers import  get_file_metadata
from pyphocorehelpers.assertion_helpers import Assert

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData

# from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation # for `BatchSessionCompletionHandler`, `AcrossSessionsAggregator`
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import  build_and_write_to_file
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

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


# ==================================================================================================================== #
# ScatterPlotResultsTable                                                                                              #
# ==================================================================================================================== #
class ScatterPlotResultsTable(tb.IsDescription):
    """ """
    neuron_identity = NeuronIdentityTable()

    lap_firing_rates_delta = FiringRatesDeltaTable()
    replay_firing_rates_delta = FiringRatesDeltaTable()

    active_set_membership = EnumCol(trackMembershipTypesEnum, 'neither', base='uint8')


# @pd.api.extensions.register_dataframe_accessor("inst_fr_results")
class InstantaneousFiringRatesDataframeAccessor():
    """ A Pandas pd.DataFrame representation of results from the batch processing of sessions
    # 2023-07-07
    Built from `BatchRun`

    Used for FigureTwo: the cross-session scatter plot of firing rates during laps v replays for the LxCs vs. SxCs.

    
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import InstantaneousFiringRatesDataframeAccessor

    
    """

    # _required_column_names = ['session_name', 'basedirs', 'status', 'errors']
    _required_column_names = ['context', 'basedirs', 'status', 'errors']



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
                table = file.create_table('/', 'ScatterPlotResults', ScatterPlotResultsTable)


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
    def add_results_to_inst_fr_results_table(cls, inst_fr_comps: InstantaneousSpikeRateGroupsComputation, curr_active_pipeline, common_file_path, file_mode='w') -> bool:
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
            if inst_fr_comps is None:
                ## recompute:
                inst_fr_comps = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01) # 10ms
                inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_session_context)

            ## Build the Output Dataframe:
            cell_firing_rate_summary_df: pd.DataFrame = inst_fr_comps.get_summary_dataframe() # Returns the dataframe with columns ['aclu', 'lap_delta_minus', 'lap_delta_plus', 'replay_delta_minus', 'replay_delta_plus', 'active_set_membership']

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

        except BaseException as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {e} while trying to compute the instantaneous firing rates and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
            inst_fr_comps = None
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



    @function_attributes(short_name=None, tags=['HDF5', 'csv'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-09 16:35', related_items=[])
    @classmethod
    def post_compute_all_sessions_processing(cls, global_data_root_parent_path:Path, output_path_suffix: str, plotting_enabled:bool, output_override_path=None, inst_fr_output_filename: str=None):
        """ 2023-11-15 - called after batch computing all of the sessions and building the required output files. Loads them, processes them, and then plots them!

        
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables, AcrossSessionsResults, AcrossSessionsVisualizations
        from neuropy.utils.mixins.HDF5_representable import HDF_Converter
        from pyphoplacecellanalysis.General.Batch.runBatch import BatchResultDataframeAccessor

        # output_path_suffix: str = '2024-09-26'
        # output_path_suffix: str = '2024-10-22'
        output_path_suffix: str = '2025-04-17' 
        # output_path_suffix: str = '2025-06-03'
        # output_path_suffix: str = '2024-10-04'
        # inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{output_path_suffix}.pkl'
        # inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{output_path_suffix}_0.0009.pkl' # single time bin size
        # inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{output_path_suffix}_0.0015.pkl' # single time bin size
        # inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{output_path_suffix}_0.0025.pkl' # single time bin size
        # inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{output_path_suffix}_0.025.pkl' # single time bin size
        inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{output_path_suffix}_1000.0.pkl' # single time bin size

        ## INPUTS: included_session_contexts, included_h5_paths
        neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=True)

        ## different than load_all_combined_tables, which seems to work with `long_short_fr_indicies_analysis_table`
        # graphics_output_dict |= AcrossSessionsVisualizations.across_sessions_firing_rate_index_figure(long_short_fr_indicies_analysis_results=long_short_fr_indicies_analysis_table, num_sessions=num_sessions, save_figure=True)

        ## Load all across-session tables from the pickles:
        output_path_suffix: str = f'{output_path_suffix}'
        neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.load_all_combined_tables(override_output_parent_path=collected_outputs_directory, output_path_suffix=output_path_suffix) # output_path_suffix=f'2023-10-04-GL-Recomp'
        # num_sessions = len(neuron_replay_stats_table.session_uid.unique().to_numpy())
        # print(f'num_sessions: {num_sessions}')
        num_sessions: int = len(long_short_fr_indicies_analysis_table['session_uid'].unique())
        print(f'num_sessions: {num_sessions}')

        inst_fr_output_load_filepath: Path = collected_outputs_directory.joinpath(inst_fr_output_filename).resolve() # single time bin size # non-instantaneous version
        assert inst_fr_output_load_filepath.exists()
        # inst_fr_output_filename: str = inst_fr_output_load_filepath.name
        # across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list = AcrossSessionsResults.load_across_sessions_data(global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=inst_fr_output_filename)
        across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list = AcrossSessionsResults.load_across_sessions_data(global_data_root_parent_path=inst_fr_output_load_filepath.parent, inst_fr_output_filename=inst_fr_output_filename)

        graphics_output_dict = AcrossSessionsResults.post_compute_all_sessions_processing(global_data_root_parent_path=collected_outputs_directory, output_path_suffix=output_path_suffix, plotting_enabled=False, output_override_path=Path('../../output'), inst_fr_output_filename=inst_fr_output_filename)

        num_sessions = len(across_sessions_instantaneous_fr_dict)
        print(f'num_sessions: {num_sessions}')

        # Convert byte strings to regular strings
        neuron_replay_stats_table = neuron_replay_stats_table.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        neuron_replay_stats_table

        
        """
        # 2023-10-04 - Load Saved across-sessions-data and testing Batch-computed inst_firing_rates:
        ## Load the saved across-session results:
        if inst_fr_output_filename is None:
            inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{output_path_suffix}.pkl'
        across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list = AcrossSessionsResults.load_across_sessions_data(global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=inst_fr_output_filename) ## LOADING?!?!?
        # across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path)
        num_sessions = len(across_sessions_instantaneous_fr_dict)
        print(f'num_sessions: {num_sessions}')

        ## Load all across-session tables from the pickles:
        output_path_suffix: str = f'{output_path_suffix}'
        neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.load_all_combined_tables(override_output_parent_path=global_data_root_parent_path, output_path_suffix=output_path_suffix) # output_path_suffix=f'2023-10-04-GL-Recomp'
        num_sessions = len(neuron_replay_stats_table.session_uid.unique().to_numpy())
        print(f'num_sessions: {num_sessions}')

        # Does its own additions to `long_short_fr_indicies_analysis_table` table based on the user labeled LxC/SxCs
        annotation_man = UserAnnotationsManager()
        # Hardcoded included_session_contexts:
        included_session_contexts = annotation_man.get_hardcoded_good_sessions()

        if output_override_path is None:
            output_override_path = Path('output').resolve()
            output_override_path.mkdir(parents=True, exist_ok=True)
            

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
        long_short_fr_indicies_analysis_table_filename: Path = output_override_path.joinpath(f'{output_path_suffix}_long_short_fr_indicies_analysis_table.csv')
        long_short_fr_indicies_analysis_table.to_csv(long_short_fr_indicies_analysis_table_filename)
        print(f'saved: {long_short_fr_indicies_analysis_table_filename}')



        # 2023-10-10 - Statistics for `across_sessions_bar_graphs`, analysing `across_session_inst_fr_computation`
        binom_test_chance_result = pho_stats_perform_diagonal_line_binomial_test(long_short_fr_indicies_analysis_table)
        print(f'binom_test_chance_result: {binom_test_chance_result}')

        LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result = pho_stats_bar_graph_t_tests(across_session_inst_fr_computation)


        ## Plotting:
        graphics_output_dict = {}
        if plotting_enabled:
            # matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')

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
    def load_across_sessions_data(cls, global_data_root_parent_path:Path, inst_fr_output_filename:str='across_session_result_long_short_inst_firing_rate.pkl') -> Tuple[InstantaneousSpikeRateGroupsComputation, Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation], List[InstantaneousSpikeRateGroupsComputation]]:
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

            
            ## 2024-09-04 - It looks like the values of `across_sessions_instantaneous_fr_dict` changed from `InstantaneousSpikeRateGroupsComputation` to Tuple[Context, InstantaneousSpikeRateGroupsComputation, inst_fr_t_bin_size]
            
        """
        from pyphoplacecellanalysis.General.Batch.runBatch import BatchResultDataframeAccessor
        
        global_batch_result_inst_fr_file_path = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
        print(f'global_batch_result_inst_fr_file_path: {global_batch_result_inst_fr_file_path}')
        Assert.path_exists(global_batch_result_inst_fr_file_path)
        across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path) ## LOAD THE DATA - across_sessions_instantaneous_fr_dict - looks like a dictionary with keys of float time_bin_size values and dictionaries as values -- the dictionaries are empty though
        num_sessions = len(across_sessions_instantaneous_fr_dict)
        print(f'num_sessions: {num_sessions}')
        # across_sessions_instantaneous_frs_list: List[InstantaneousSpikeRateGroupsComputation] = list(across_sessions_instantaneous_fr_dict.values())
        assert np.all([len(v) == 3 for v in across_sessions_instantaneous_fr_dict.values()]), f"expected values to be tuples Tuple[Context, InstantaneousSpikeRateGroupsComputation, inst_fr_t_bin_size] but were: {list(across_sessions_instantaneous_fr_dict.values())}" # they're actually all 

        across_sessions_instantaneous_frs_ctxts_list: List[IdentifyingContext] = [v[0] for v in across_sessions_instantaneous_fr_dict.values()]
        across_sessions_instantaneous_frs_list: List[InstantaneousSpikeRateGroupsComputation] = [v[1] for v in across_sessions_instantaneous_fr_dict.values()]        
        across_sessions_instantaneous_frs_t_bin_size_list: List[float] = [v[2] for v in across_sessions_instantaneous_fr_dict.values()]
        ## Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions.
        global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.
        # _out.cell_agg_inst_fr_list = cell_agg_firing_rates_list # .shape (n_cells,)
        across_session_inst_fr_computation = InstantaneousSpikeRateGroupsComputation()
        across_session_inst_fr_computation.active_identifying_session_ctx = global_multi_session_context

        all_contexts_list: List[IdentifyingContext] = list(across_sessions_instantaneous_fr_dict.keys())
        assert len(all_contexts_list) > 0, f"len(all_contexts_list) should be > 0 -- must have at least one element"
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

        LxC_aclus = np.concatenate([_build_session_dep_aclu_identifier(k, v.LxC_aclus) for k, v in zip(across_sessions_instantaneous_frs_ctxts_list, across_sessions_instantaneous_frs_list)])
        SxC_aclus = np.concatenate([_build_session_dep_aclu_identifier(k, v.SxC_aclus) for k, v in zip(across_sessions_instantaneous_frs_ctxts_list, across_sessions_instantaneous_frs_list)])

        across_session_inst_fr_computation.LxC_aclus = LxC_aclus
        across_session_inst_fr_computation.SxC_aclus = SxC_aclus

        ## Scatter props:
        LxC_scatter_props = [_return_scatter_props_fn(k) for k, v in zip(across_sessions_instantaneous_frs_ctxts_list, across_sessions_instantaneous_frs_list)]
        SxC_scatter_props = [_return_scatter_props_fn(k) for k, v in zip(across_sessions_instantaneous_frs_ctxts_list, across_sessions_instantaneous_frs_list)]

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


    # ==================================================================================================================================================================================================================================================================================== #
    # 2025-05-09 - Combined CSV output                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #

    @function_attributes(short_name=None, tags=['across-sessions'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-09 17:31', related_items=['AcrossSessionTables.build_neuron_replay_stats_table'])
    @classmethod
    def build_neuron_identities_df_for_CSV(cls, curr_active_pipeline) -> pd.DataFrame:
        """ Exports all available neuron information (both identity and computed in computations) for each neuron

        AcrossSessionTables.build_and_save_all_combined_tables, .build_neuron_identities_table, .build_neuron_replay_stats_table, .build_long_short_fr_indicies_analysis_table
        
        
        
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults
        
        all_neuron_stats_table: pd.DataFrame = AcrossSessionsResults.build_neuron_identities_df_for_CSV(curr_active_pipeline=curr_active_pipeline)
        
        
        """
        from neuropy.core.neuron_identities import NeuronExtendedIdentityTuple, neuronTypesEnum, NeuronIdentityTable, NeuronIdentityDataframeAccessor
        from neuropy.utils.mixins.HDF5_representable import HDF_Converter
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults # for build_neuron_identity_table_to_hdf
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import InstantaneousSpikeRateGroupsComputation
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import ExpectedVsObservedResult


        session_context = curr_active_pipeline.get_session_context() 

        # Handle long|short firing rate index:
        long_short_fr_indicies_analysis_results = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']
        x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
        active_context = long_short_fr_indicies_analysis_results['active_context']
        # Need to map keys of dict to an absolute dict value:
        sess_specific_aclus = list(x_frs_index.keys())
        session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())

        long_short_fr_indicies_analysis_results_h5_df = pd.DataFrame([(f"{session_ctxt_key}|{aclu}", session_ctxt_key, aclu, x_frs_index[aclu], y_frs_index[aclu]) for aclu in sess_specific_aclus], columns=['neuron_uid', 'session_uid', 'aclu','x_frs_index', 'y_frs_index'])
        # long_short_fr_indicies_analysis_results_h5_df.to_hdf(file_path, key=f'{a_global_computations_group_key}/long_short_fr_indicies_analysis', format='table', data_columns=True)
        ## OUTPUTS: long_short_fr_indicies_analysis_results_h5_df
        

        # long_short_post_decoding result: __________________________________________________________________________________ #
        curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
        expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
        rate_remapping_df = curr_long_short_rr.rr_df
        # Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result.Flat_epoch_time_bins_mean, expected_v_observed_result.Flat_decoder_time_bin_centers, expected_v_observed_result.num_neurons, expected_v_observed_result.num_timebins_in_epoch, expected_v_observed_result.num_total_flat_timebins, expected_v_observed_result.is_short_track_epoch, expected_v_observed_result.is_long_track_epoch, expected_v_observed_result.short_short_diff, expected_v_observed_result.long_long_diff
        ## OUTPUTS: rate_remapping_df
        

        # Rate Remapping _____________________________________________________________________________________________________ #
        rate_remapping_df: pd.DataFrame = rate_remapping_df[['laps', 'replays', 'skew', 'max_axis_distance_from_center', 'distance_from_center', 'has_considerable_remapping']]
        # rate_remapping_df.to_hdf(file_path, key=f'{a_global_computations_group_key}/rate_remapping', format='table', data_columns=True)
        rate_remapping_df = rate_remapping_df.reset_index(drop=False).neuron_identity.make_neuron_indexed_df_global(session_context, add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)


        # jonathan_firing_rate_analysis_result _______________________________________________________________________________ #
        jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis

        ## try to add extra columns if available:
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        rank_order_results = curr_active_pipeline.global_computation_results.computed_data.get('RankOrder', None) # : "RankOrderComputationsContainer"
        if rank_order_results is not None:
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            included_qclu_values: List[int] = rank_order_results.included_qclu_values
        else:        
            ## get from parameters:
            minimum_inclusion_fr_Hz: float = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
            included_qclu_values: List[int] = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
                    

        track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        


        try:
            _neuron_replay_stats_df, _all_pf2D_peaks_modified_columns = jonathan_firing_rate_analysis_result.add_peak_promenance_pf_peaks(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates)
        except (KeyError, ValueError, AttributeError) as e:
            print(f'\tfailed to `add_peak_promenance_pf_peaks`, with error e: {e}. skipping.')            
        except Exception:
            raise

        try:
            _neuron_replay_stats_df, _all_pf1D_peaks_modified_columns = jonathan_firing_rate_analysis_result.add_directional_pf_maximum_peaks(track_templates=track_templates)
        except (KeyError, ValueError, AttributeError) as e:
            print(f'\tfailed to `add_directional_pf_maximum_peaks`, with error e: {e}. skipping.')            
        except Exception:
            raise


        _neuron_replay_stats_df = deepcopy(jonathan_firing_rate_analysis_result.neuron_replay_stats_df)
        _neuron_replay_stats_df = HDF_Converter.prepare_neuron_indexed_dataframe_for_hdf(_neuron_replay_stats_df, active_context=deepcopy(session_context), aclu_column_name=None)        
        
        # jonathan_firing_rate_analysis_result.to_hdf(file_path=file_path, key=f'{a_global_computations_group_key}/jonathan_fr_analysis', active_context=session_context)
        # jonathan_firing_rate_analysis_result

        ## OUTPUTS: _neuron_replay_stats_df

        ## InstantaneousSpikeRateGroupsComputation ____________________________________________________________________________ #
        # try:
        #     inst_spike_rate_groups_result: InstantaneousSpikeRateGroupsComputation = curr_active_pipeline.global_computation_results.computed_data.long_short_inst_spike_rate_groups # = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01) # 10ms
        #     # inst_spike_rate_groups_result.compute(curr_active_pipeline=self, active_context=self.sess.get_context())
        #     # inst_spike_rate_groups_result.to_hdf(file_path, f'{a_global_computations_group_key}/inst_fr_comps') # held up by SpikeRateTrends.inst_fr_df_list  # to HDF, don't need to split it
        #     # NotImplementedError: a_field_attr: Attribute(name='LxC_aclus', default=None, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=False, metadata=mappingproxy({'tags': ['dataset'], 'serialization': {'hdf': True}, 'custom_serialization_fn': None, 'hdf_metadata': {'track_eXclusive_cells': 'LxC'}}), type=<class 'numpy.ndarray'>, converter=None, kw_only=False, inherited=False, on_setattr=None, alias='LxC_aclus') could not be serialized and _ALLOW_GLOBAL_NESTED_EXPANSION is not allowed.
            
        # except (KeyError, AttributeError) as e:
        #     print(f'long_short_inst_spike_rate_groups is missing and will be skipped. Error: {e}')
        # except NotImplementedError as e:
        #     print(f'long_short_inst_spike_rate_groups failed to save to HDF5 due to NotImplementedError issues and will be skipped. Error {e}')
        # except TypeError as e:
        #     # TypeError: Object dtype dtype('O') has no native HDF5 equivalent
        #     print(f'long_short_inst_spike_rate_groups failed to save to HDF5 due to type issues and will be skipped. This is usually caused by Python None values. Error {e}')
        # except Exception:
        #     raise

        # if not isinstance(expected_v_observed_result, ExpectedVsObservedResult):
        #     expected_v_observed_result = ExpectedVsObservedResult(**expected_v_observed_result.to_dict())

        # # expected_v_observed_result.to_hdf(file_path=file_path, key=f'{a_global_computations_group_key}/expected_v_observed_result', active_context=session_context) # 'output/test_ExpectedVsObservedResult.h5', '/expected_v_observed_result')


        ##TODO: remainder of global_computations
        # self.global_computation_results.to_hdf(file_path, key=f'{a_global_computations_group_key}')

        # AcrossSessionsResults.build_neuron_identity_table_to_hdf(file_path, key=session_group_key, spikes_df=self.sess.spikes_df, session_uid=session_uid)

        # ==================================================================================================================================================================================================================================================================================== #
        # Begin building fresh df                                                                                                                                                                                                                                                              #
        # ==================================================================================================================================================================================================================================================================================== #
        spikes_df: pd.DataFrame = get_proper_global_spikes_df(owning_pipeline_reference=curr_active_pipeline)
        unique_neuron_identities_df: pd.DataFrame = spikes_df.spikes.extract_unique_neuron_identities()
        neuron_types_enum_array = np.array([neuronTypesEnum[a_type.hdfcodingClassName] for a_type in unique_neuron_identities_df['neuron_type']]) # convert NeuronTypes to neuronTypesEnum
        unique_neuron_identities_df['neuron_type']  = neuron_types_enum_array
        unique_neuron_identities_df = unique_neuron_identities_df.neuron_identity.make_neuron_indexed_df_global(curr_active_pipeline.get_session_context(), add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)
        # unique_neuron_identities_df

        initial_num_rows: int = len(unique_neuron_identities_df)
        
        ## OUTPUT: unique_neuron_identities_df
        ## INPUTS: long_short_fr_indicies_analysis_results_h5_df, rate_remapping_df, _neuron_replay_stats_df
        

        ## Combine all unique columns from the three loaded dataframes: [neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table], into a merged df `all_neuron_stats_table`
        all_neuron_stats_table: pd.DataFrame = deepcopy(unique_neuron_identities_df)
        ## All dataframes have the same number of rows and are uniquely indexed by their 'neuron_uid' column. Add the additional columns from `long_short_fr_indicies_analysis_table` to `all_neuron_stats_table`

        ## OUTPUTS: _neuron_replay_stats_df
        ## merge in `_neuron_replay_stats_df`'s columns
        all_neuron_stats_table = pd.merge(all_neuron_stats_table, _neuron_replay_stats_df[['neuron_uid'] + [col for col in _neuron_replay_stats_df.columns if col not in all_neuron_stats_table.columns and col != 'neuron_uid']], on='neuron_uid')
        ## merge in `long_short_fr_indicies_analysis_results_h5_df`'s columns
        all_neuron_stats_table = pd.merge(all_neuron_stats_table, long_short_fr_indicies_analysis_results_h5_df[['neuron_uid'] + [col for col in long_short_fr_indicies_analysis_results_h5_df.columns if col not in all_neuron_stats_table.columns and col != 'neuron_uid']], on='neuron_uid')
        ## merge in `rate_remapping_df`'s columns
        all_neuron_stats_table = pd.merge(all_neuron_stats_table, rate_remapping_df[['neuron_uid'] + [col for col in rate_remapping_df.columns if col not in all_neuron_stats_table.columns and col != 'neuron_uid']], on='neuron_uid')

        ## check
        # assert len(all_neuron_stats_table) == initial_num_rows, f"initial_num_rows: {initial_num_rows}, len(all_neuron_stats_table): {len(all_neuron_stats_table)}"

        if len(all_neuron_stats_table) != initial_num_rows:
            print(f"WARNING: initial_num_rows: {initial_num_rows}, len(all_neuron_stats_table): {len(all_neuron_stats_table)}")

        return all_neuron_stats_table


    @function_attributes(short_name=None, tags=['across-sessions'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-09 17:31', related_items=[])
    @classmethod
    def build_neuron_identities_CSV(cls, file_path, curr_active_pipeline) -> pd.DataFrame:
        """ 
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults
        
        all_neuron_stats_table, file_path = AcrossSessionsResults.build_neuron_identities_CSV(curr_active_pipeline=curr_active_pipeline)
        
        
        """
        all_neuron_stats_table: pd.DataFrame = AcrossSessionsResults.build_neuron_identities_df_for_CSV(curr_active_pipeline=curr_active_pipeline)
        file_path = Path(file_path)
        file_path = file_path.with_suffix('.csv')
        all_neuron_stats_table.to_csv(file_path)
        return (all_neuron_stats_table, file_path)
    


    @classmethod
    def load_all_neuron_identities_CSV(cls, override_output_parent_path:Optional[Path]=None, output_path_suffix:Optional[str]=None):
        """Save converted back to .h5 file, .csv file, and several others

        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults

            all_neuron_stats_table, all_loaded_csv_dfs = AcrossSessionsResults.load_all_neuron_identities_CSV(override_output_parent_path=global_data_root_parent_path, output_path_suffix=f'_{BATCH_DATE_TO_USE}')


        """
        # Build the output paths:
        out_parent_path: Path = override_output_parent_path or Path('output/across_session_results')
        out_parent_path = out_parent_path.resolve()

        if output_path_suffix is not None:
            out_parent_path = out_parent_path.joinpath(output_path_suffix).resolve()

        # out_parent_path.mkdir(parents=True, exist_ok=True)
        assert out_parent_path.exists(), f"out_parent_path: '{out_parent_path}' must exist to load the tables!"

        # '2025-06-10_Lab_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_5.0-2006-6-07_16-40-19_neuron_replay_stats_df.csv'
        csv_file_match_glob = '*_neuron_replay_stats_df.csv'
        
        found_files = list(out_parent_path.glob(csv_file_match_glob))
        all_loaded_csv_dfs = []
        for a_found_csv in found_files:
            print(f'loading "{a_found_csv}"...')
            a_found_df: pd.DataFrame = pd.read_csv(a_found_csv)
            # try to rename the columns if needed
            # a_found_df.rename(columns=cls.aliases_columns_dict, inplace=True)
            all_loaded_csv_dfs.append(a_found_df)

        all_neuron_stats_table: pd.DataFrame = pd.concat(all_loaded_csv_dfs, axis='index')

        ## Load the exported sessions experience_ranks CSV and use it to add the ['session_experience_rank', 'session_experience_orientation_rank'] columns to the tables:
        try:
            sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv("./data/sessions_experiment_datetime_df.csv", session_uid_str_sep='|')
            all_loaded_csv_dfs = [_callback_add_df_columns(v, session_id_column_name='session_uid') for v in all_loaded_csv_dfs]
            all_neuron_stats_table: pd.DataFrame = pd.concat(all_loaded_csv_dfs, axis='index')

        except Exception as e:
            print(f'failed to load and apply the sessions rank CSV to tables. Error: {e}')
            raise
        
        return all_neuron_stats_table, all_loaded_csv_dfs
    



class ConciseSessionIdentifiers:
    """ Building and Parsing a minimal list of session identifiers
    
    
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import ConciseSessionIdentifiers
    
    
    Factored out of `BatchResultDataframeAccessor._build_minimal_session_identifiers_list(...)` on 2024-09-18
        
    """
    @function_attributes(short_name=None, tags=['concise', 'format', 'identifier'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-18 11:34', related_items=['parse_concise_abbreviated_neuron_identifying_strings'])
    @classmethod
    def _build_minimal_session_identifiers_list(cls, df: pd.DataFrame):
        """Build a list of short unique identifiers for the good sessions:
        Adds Column: ['context_minimal_name']
        
        ['a0s0', 'a0s1', 'a0s2', 'a0s3', 'a0s4', 'a0s5', 'a0s6', ... 'a2s10', 'a2s11', 'a2s12', 'a2s13', 'a2s14', 'a2s15', 'a2s16', 'a2s17', 'a2s18', 'a2s19']
        
        TODO: Critical: this 
        #TODO 2023-07-20 21:23: - [ ] This needs to only be ran on a dataframe containing all of the sessions! If it's filtered at all, the session numbers will vary depending on how it's filtered!
        
        Usage:
        
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
                animal_id = str(ctxt.animal)
                return scatter_props_dict[animal_id]

            LxC_aclus = np.concatenate([_build_session_dep_aclu_identifier(k, v.LxC_aclus) for k, v in zip(across_sessions_instantaneous_frs_ctxts_list, across_sessions_instantaneous_frs_list)])
            SxC_aclus = np.concatenate([_build_session_dep_aclu_identifier(k, v.SxC_aclus) for k, v in zip(across_sessions_instantaneous_frs_ctxts_list, across_sessions_instantaneous_frs_list)])

        
        """
        # Extract unique values for each column
        unique_format_names = df['format_name'].unique()
        unique_animals = df['animal'].unique()
        unique_exper_names = df['exper_name'].unique()
        unique_session_names = df['session_name'].unique()

        # Create mapping to shorthand notation for each column
        format_name_mapping = {name: f'f{i}' for i, name in enumerate(unique_format_names)}
        animal_mapping = {name: f'a{i}' for i, name in enumerate(unique_animals)}
        exper_name_mapping = {name: f'e{i}' for i, name in enumerate(unique_exper_names)}
        session_name_mapping = {name: f's{i}' for i, name in enumerate(unique_session_names)}

        # Create a mapping for 'session_name' within each 'animal'
        # animal_session_mapping = {animal: {session: f'{animal[0]}{i}s{j}' for j, session in enumerate(df[df['animal'] == animal]['session_name'].unique())} for i, animal in enumerate(df['animal'].unique())} # 'g0s0'
        animal_session_mapping = {animal: {session: f'{animal_mapping[animal]}s{j}' for j, session in enumerate(df[df['animal'] == animal]['session_name'].unique())} for i, animal in enumerate(df['animal'].unique())} # 'g0s0'

        # Replace original values with shorthand notation
        for animal, session_mapping in animal_session_mapping.items():
            # df.loc[df['animal'] == animal, 'session_name'] = df.loc[df['animal'] == animal, 'session_name'].replace(session_mapping)
            df.loc[df['animal'] == animal, 'context_minimal_name'] = df.loc[df['animal'] == animal, 'session_name'].replace(session_mapping)

        return df['context_minimal_name']

    @function_attributes(short_name=None, tags=['concise', 'abbreviated', 'parse'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-18 11:31', related_items=['_build_minimal_session_identifiers_list'])
    @classmethod
    def parse_concise_abbreviated_neuron_identifying_strings(cls, data: List[str]) -> pd.DataFrame:
        """ parses ['a0s0_109', 'a0s1_3', 'a0s1_29', 'a0s1_103', 'a0s3_90', 'a0s4_91', 'a0s4_95', 'a1s1_23', 'a1s2_25', 'a1s3_14', 'a1s3_30', 'a1s3_32', 'a2s0_8', 'a2s0_27', 'a2s1_27'] into ( animal_idx, session_idx, and aclu)
        """
        parsed_data: List[Tuple[int, int, int]] = []
        for entry in data:
            match = re.match(r'a(\d+)s(\d+)_(\d+)', entry)
            if match:
                animal_idx, session_idx, aclu = map(int, match.groups())
                parsed_data.append((animal_idx, session_idx, aclu))
            else:
                raise ValueError(f"String format not recognized: {entry}")
        
        parsed_aclus_df = pd.DataFrame(parsed_data, columns=['animal_idx', 'session_idx', 'aclu'])
        parsed_aclus_df['is_session_novel'] = (parsed_aclus_df['session_idx'] < 2)
        return parsed_aclus_df



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
        # master_table = pd.concat(data_frames, ignore_index=True)
        master_table = PandasHelpers.safe_concat(data_frames, ignore_index=True)
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


@function_attributes(short_name=None, tags=['copy', 'batch', 'across_session'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-04 09:05', related_items=[])
def copy_session_folder_files_to_target_dir(good_session_concrete_folders, target_dir: Path, RESULT_DATE_TO_USE: str, custom_file_types_dict: Dict, debug_print=True, dry_run: bool=False):
    """ Copes files from the individual session data folders to a shared location (such as a collected_outputs folder)
    
    Usage:
        ## INPUTS: good_session_concrete_folders, target_dir, BATCH_DATE_TO_USE, custom_file_types_dict
        from pyphoplacecellanalysis.General.Batch.runBatch import get_file_path_if_file_exists
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import copy_session_folder_files_to_target_dir

        custom_file_types_dict = {'recomputed_inst_fr_comps': (lambda a_session_folder: get_file_path_if_file_exists(a_session_folder.output_folder.joinpath(f'{RESULT_DATE_TO_USE}_recomputed_inst_fr_comps_0.0005.h5').resolve())),
                                #   'PHONEW.evt': (lambda a_session_folder: get_file_path_if_file_exists(a_session_folder.output_folder.joinpath(f'{a_session_folder.context.session_name}.PHONEW.evt').resolve())),
                                }

        # target_dir: Path = Path(global_data_root_parent_path)
        target_dir: Path = collected_outputs_directory
        moved_files_dict_files, (filelist_path, filedict_out_path) = copy_session_folder_files_to_target_dir(good_session_concrete_folders, target_dir=target_dir, RESULT_DATE_TO_USE=BATCH_DATE_TO_USE, custom_file_types_dict=custom_file_types_dict, dry_run=False)
    
    """
    from pyphoplacecellanalysis.General.Batch.runBatch import ConcreteSessionFolder, BackupMethods
    # from pyphoplacecellanalysis.General.Batch.runBatch import get_file_path_if_file_exists
    from pyphocorehelpers.Filesystem.path_helpers import save_filelist_to_text_file
    from pyphocorehelpers.Filesystem.path_helpers import copy_movedict, save_copydict_to_text_file

    def _subfn_across_session_file_output_basename_fn(session_context: Optional[IdentifyingContext], session_descr: Optional[str], basename: str, *args, separator_char: str = "_"):
        """ Captures `BATCH_DATE_TO_USE` """
        # a_session_folder.context
        if session_context is not None:
            session_descr = session_context.session_name # '2006-6-07_16-40-19'
        _filename_list = [RESULT_DATE_TO_USE, session_descr, basename]
        if len(args) > 0:
            _filename_list.extend([str(a_part) for a_part in args if a_part is not None])
        return separator_char.join(_filename_list)


    copy_file_type_dict = ConcreteSessionFolder.build_backup_copydict(good_session_concrete_folders, target_dir=target_dir, backup_mode=BackupMethods.CommonTargetDirectory, 
                                                            rename_backup_basename_fn=_subfn_across_session_file_output_basename_fn, custom_file_types_dict=custom_file_types_dict, only_include_file_types=None) # , rename_backup_suffix=BATCH_DATE_TO_USE
    # copy_file_type_dict

    ## OUTPUT: copy_file_type_dict

    ## output: moved_files_dict_files
    ## Save copied file copydict csv:
    custom_file_type_name: str = list(custom_file_types_dict.keys())[0]

    ## INPUTS: target_dir, BATCH_DATE_TO_USE
    moved_files_copydict_output_filename=f'{RESULT_DATE_TO_USE}_all_sessions_copydict_{custom_file_type_name}.csv'
    moved_files_copydict_file_path = Path(target_dir).joinpath(moved_files_copydict_output_filename).resolve() # Use Default
    print(f'moved_files_copydict_file_path: {moved_files_copydict_file_path}')

    ## Can read with:
    """

    read_moved_files_dict_files = read_copydict_from_text_file(moved_files_copydict_file_path, debug_print=False)
    read_moved_files_dict_files
    # read_moved_files_dict_files
    restore_moved_files_dict_files = invert_filedict(read_moved_files_dict_files)
    restore_moved_files_dict_files
    """
    ## Save copied file list:
    ## INPUTS: target_dir, BATCH_DATE_TO_USE
    custom_filelist_output_filename=f'{RESULT_DATE_TO_USE}_all_sessions_{custom_file_type_name}_filelist.txt'
    custom_filelist_output_file_path = Path(target_dir).joinpath(custom_filelist_output_filename).resolve() # Use Default
    print(f'custom_filelist_output_file_path: {custom_filelist_output_file_path}')

    if not dry_run:
        print(f'PERFORMING THE MOVE....')
        # perform the move:
        moved_files_dict_files = copy_movedict(copy_file_type_dict)

        _out_string, filedict_out_path = save_copydict_to_text_file(moved_files_dict_files, filelist_path=moved_files_copydict_file_path, debug_print=debug_print)
        _out_string, filelist_path = save_filelist_to_text_file(list(moved_files_dict_files.values()), filelist_path=custom_filelist_output_file_path, debug_print=debug_print) # r"W:\Data\all_sessions_h5_filelist_2024-03-28_Apogee.txt"
    

    else:
        print(f'not performing the move because dry_run==True')
        moved_files_dict_files = None
        filedict_out_path = moved_files_copydict_file_path
        filelist_path = custom_filelist_output_file_path


    ## OUTPUTS: moved_files_dict_files
    return moved_files_dict_files, (filelist_path, filedict_out_path)



@function_attributes(short_name=None, tags=['inst_fr', 'across_session'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-12 09:48', related_items=[])
def copy_session_inst_fr_data_to_across_session_pkl(RESULT_DATE_TO_USE, collected_outputs_path, instantaneous_time_bin_size_seconds_list:List[float], debug_print = False):
    """ Called on the batch processing computer (GreatLakes for example) in `ProcessBatchOutputs_*.ipynb` to collect batch-processing results from individual session folders and save `/collected_outputs/across_session_result_long_short_recomputed_inst_firing_rate_{BATCH_DATE_TO_USE}.pkl`
    
    File Outputs: `/collected_outputs/across_session_result_long_short_recomputed_inst_firing_rate_{BATCH_DATE_TO_USE}.pkl`
    
    """
    from pyphoplacecellanalysis.General.Batch.runBatch import get_file_path_if_file_exists
    from neuropy.core.user_annotations import UserAnnotationsManager
    from pyphoplacecellanalysis.General.Batch.runBatch import ConcreteSessionFolder
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData

    def _subfn_combine_across_session_inst_firing_rate_pkl(good_session_concrete_folders, collected_outputs_path, BATCH_DATE_TO_USE, custom_file_types_dict=None):
        """ 
        Creates `across_session_result_long_short_recomputed_inst_firing_rate_{BATCH_DATE_TO_USE}_{time_bin_size_str}.pkl`
        """
        if custom_file_types_dict is None:
            custom_file_types_dict = {'recomputed_inst_fr_comps': (lambda a_session_folder: get_file_path_if_file_exists(a_session_folder.output_folder.joinpath(f'{BATCH_DATE_TO_USE}_recomputed_inst_fr_comps_{time_bin_size}.pkl').resolve()))}
            
        ## build the `across_sessions_recomputed_instantaneous_fr_dict` by loading the individual results from the session folders
        across_sessions_recomputed_instantaneous_fr_dict_dict = {}

        for k, a_get_file_fn in custom_file_types_dict.items():
            ## for a single time_bin_size
            *variable_name_parts, time_bin_size_str = k.split('_')
            time_bin_size = float(time_bin_size_str)
            print(f'\ttime_bin_size: {time_bin_size}')
            across_sessions_recomputed_instantaneous_fr_dict = {}
            for a_session_folder in good_session_concrete_folders:    
                curr_pkl = a_get_file_fn(a_session_folder) ## Build the full path
                # curr_pkl = custom_file_types_dict['recomputed_inst_fr_comps'](a_session_folder)
                if curr_pkl is not None and (curr_pkl.exists()):
                    assert curr_pkl.exists()
                    print(a_session_folder)
                    print(curr_pkl)
                    try:
                        across_sessions_recomputed_instantaneous_fr_dict[a_session_folder.context] = loadData(curr_pkl) # InstantaneousSpikeRateGroupsComputation
                    except BaseException as err:
                        print(f'encountered err: {err} when trying to unpickle. Skipping.')
            # end for a_session_folder
            # OUTPUT:  across_sessions_recomputed_instantaneous_fr_dict
            num_sessions = len(across_sessions_recomputed_instantaneous_fr_dict)
            print(f'num_sessions: {num_sessions}')
            ## Outputs: across_sessions_instantaneous_fr_dict, across_sessions_recomputed_instantaneous_fr_dict
            # When done, `result_handler.across_sessions_instantaneous_fr_dict` is now equivalent to what it would have been before. It can be saved using the normal `.save_across_sessions_data(...)`

            ## Save the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation)
            across_session_result_long_short_recomputed_inst_firing_rate_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{BATCH_DATE_TO_USE}_{time_bin_size_str}.pkl'
            AcrossSessionsResults.save_across_sessions_data(across_sessions_instantaneous_fr_dict=across_sessions_recomputed_instantaneous_fr_dict, global_data_root_parent_path=collected_outputs_path.resolve(),
                                                            inst_fr_output_filename=across_session_result_long_short_recomputed_inst_firing_rate_filename)

            ## Add to the dict dict
            across_sessions_recomputed_instantaneous_fr_dict_dict[time_bin_size] = across_sessions_recomputed_instantaneous_fr_dict


        return across_sessions_recomputed_instantaneous_fr_dict_dict

    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    ## INPUTS: RESULT_DATE_TO_USE, collected_outputs_path
    # instantaneous_time_bin_size_seconds_list:List[float]=[0.0005, 0.0009, 0.0015, 0.0025, 0.025]
    custom_file_types_dict = {f'recomputed_inst_fr_comps_{time_bin_size}': (lambda a_session_folder: get_file_path_if_file_exists(a_session_folder.output_folder.joinpath(f'{RESULT_DATE_TO_USE}_recomputed_inst_fr_comps_{time_bin_size}.pkl').resolve())) for time_bin_size in instantaneous_time_bin_size_seconds_list}
    print(f'custom_file_types_dict: {custom_file_types_dict}')
    
    known_global_data_root_parent_paths = [Path(r'/nfs/turbo/umms-kdiba/Data'), Path(r'W:\Data'), Path(r'/home/halechr/cloud/turbo/Data'), Path(r'/media/halechr/MAX/Data'), Path(r'/Volumes/MoverNew/data')] # , Path(r'/home/halechr/FastData'), Path(r'/home/halechr/turbo/Data'), Path(r'W:\Data'), Path(r'/home/halechr/cloud/turbo/Data')
    global_data_root_parent_path = find_first_extant_path(known_global_data_root_parent_paths)
    assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
    # Hardcoded included_session_contexts:
    included_session_contexts = UserAnnotationsManager.get_hardcoded_good_sessions()
    good_session_concrete_folders = ConcreteSessionFolder.build_concrete_session_folders(global_data_root_parent_path, included_session_contexts)

    # Output Paths:
    target_dir: Path = collected_outputs_path
    custom_file_type_name: str = 'recomputed_inst_fr_comps'
    out_parent_path = AcrossSessionTables.make_all_combined_output_directory(override_output_parent_path=target_dir, output_path_suffix=f'{RESULT_DATE_TO_USE}/{custom_file_type_name}')

    moved_files_dict_files, (filelist_path, filedict_out_path) = copy_session_folder_files_to_target_dir(good_session_concrete_folders, target_dir=out_parent_path, RESULT_DATE_TO_USE=RESULT_DATE_TO_USE, custom_file_types_dict=custom_file_types_dict, dry_run=False)

    across_sessions_recomputed_instantaneous_fr_dict = _subfn_combine_across_session_inst_firing_rate_pkl(good_session_concrete_folders=good_session_concrete_folders, collected_outputs_path=collected_outputs_path, BATCH_DATE_TO_USE=RESULT_DATE_TO_USE, custom_file_types_dict=custom_file_types_dict)

    return across_sessions_recomputed_instantaneous_fr_dict, moved_files_dict_files, (filelist_path, filedict_out_path)



class AcrossSessionTables:

    aliases_columns_dict = {'global_uid':'neuron_uid', 'neuron_id':'aclu'}
    float_columns = ['long_pf_peak_x', 'short_pf_peak_x', 'long_LR_pf2D_peak_x', 'long_LR_pf2D_peak_y',
                            'long_RL_pf2D_peak_x', 'long_RL_pf2D_peak_y', 'short_LR_pf2D_peak_x', 'short_LR_pf2D_peak_y',
                            'short_RL_pf2D_peak_x', 'short_RL_pf2D_peak_y', 'long_LR_pf1D_peak', 'long_RL_pf1D_peak',
                            'short_LR_pf1D_peak', 'short_RL_pf1D_peak', 'peak_diff_RL_pf1D_peak', 'peak_diff_LR_pf1D_peak']


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
        if _out_table is not None:
            if should_restore_native_column_types:
                _out_table = HDF_Converter.general_post_load_restore_table_as_needed(_out_table)

            if drop_columns_list is not None:
                # Drop columns: 'neuron_IDX', 'has_short_pf' and 3 other columns
                _out_table = _out_table.drop(columns=drop_columns_list)

            # try to rename the columns if needed
            _out_table.rename(columns=AcrossSessionTables.aliases_columns_dict, inplace=True)
                        
        return _out_table

    @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-25 14:28', related_items=[])
    def build_neuron_replay_stats_table(included_session_contexts, included_h5_paths, **kwargs) -> pd.DataFrame:
        """
        Usage:
            neuron_replay_stats_table = AcrossSessionTables.build_neuron_replay_stats_table(included_session_contexts, included_h5_paths)
            neuron_replay_stats_table
        """
        session_group_keys: List[str] = [("/" + a_ctxt.get_description(separator="/", include_property_names=False)) for a_ctxt in included_session_contexts] # 'kdiba/gor01/one/2006-6-08_14-26-15'
        neuron_replay_stats_df_table_keys = [f"{session_group_key}/global_computations/jonathan_fr_analysis/neuron_replay_stats_df/table" for session_group_key in session_group_keys]
        drop_columns_list = ['neuron_IDX', 'has_short_pf', 'has_na', 'has_long_pf', 'index']
        neuron_replay_stats_table: pd.DataFrame = AcrossSessionTables.build_custom_table(included_session_contexts, included_h5_paths, df_table_keys=neuron_replay_stats_df_table_keys, drop_columns_list=drop_columns_list, **kwargs)
        # Manually convert specific columns to float64
        for col in AcrossSessionTables.float_columns:
            if col in neuron_replay_stats_table.columns:
                neuron_replay_stats_table[col] = pd.to_numeric(neuron_replay_stats_table[col], errors='coerce')
                
                
        # Specify columns to convert to float
        columns_to_convert = AcrossSessionTables.float_columns # ['long_replay_mean', 'long_non_replay_mean']
        neuron_replay_stats_table[columns_to_convert] = neuron_replay_stats_table[columns_to_convert].astype(float) # Convert specified columns to float

        return neuron_replay_stats_table


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
    def write_table_to_files(cls, df: pd.DataFrame, global_data_root_parent_path:Path, output_basename:str='neuron_identities_table', include_csv:bool=False, include_pkl:bool=True):
        """

        out_path_dict = AcrossSessionTables.write_table_to_files(v, global_data_root_parent_path=global_data_root_parent_path, output_basename='a_table')
        """
        out_parent_path = global_data_root_parent_path.resolve() # = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
        out_parent_path.mkdir(parents=True, exist_ok=True)
        # print(f'global_batch_result_inst_fr_file_path: {out_parent_path}')
        # print(f'a_name: {a_name}')
        out_path_dict = {}
        if not isinstance(output_basename, Path):
            output_basename = Path(output_basename)
        if include_csv:
            csv_out_path = out_parent_path.joinpath(output_basename.with_suffix(suffix='.csv'))
            print(f'writing {csv_out_path}')
            df.to_csv(csv_out_path)
            out_path_dict['.csv'] = csv_out_path

        if include_pkl:
            pkl_out_path = out_parent_path.joinpath(output_basename.with_suffix(suffix='.pkl'))
            print(f'writing {pkl_out_path}')
            saveData(pkl_out_path, db=df, safe_save=False)
            out_path_dict['.pkl'] = pkl_out_path

        return out_path_dict


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
        print(f'reading {pkl_out_path}')
        v = loadData(pkl_out_path)
        # try to rename the columns if needed
        v.rename(columns=cls.aliases_columns_dict, inplace=True)
        return v


    @classmethod
    def make_all_combined_output_directory(cls, override_output_parent_path:Optional[Path]=None, output_path_suffix:Optional[str]=None) -> Path:
        # Build the output paths:
        out_parent_path: Path = override_output_parent_path or Path('output/across_session_results')
        out_parent_path = out_parent_path.resolve()

        if output_path_suffix is not None:
            out_parent_path = out_parent_path.joinpath(output_path_suffix).resolve()

        out_parent_path.mkdir(parents=True, exist_ok=True)
        return out_parent_path



    @classmethod
    def build_and_save_all_combined_tables(cls, included_session_contexts, included_h5_paths, should_restore_native_column_types:bool=True, override_output_parent_path:Optional[Path]=None, output_path_suffix:Optional[str]=None, include_csv:bool=True, include_pkl:bool=True):
        """Save converted back to .h5 file, .csv file, and several others

        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables

            AcrossSessionTables.build_and_save_all_combined_tables(included_session_contexts, included_h5_paths)
            included_h5_paths = [a_dir.joinpath('output','pipeline_results.h5').resolve() for a_dir in included_session_batch_progress_df['basedirs']]


            (neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table), output_path_dicts = AcrossSessionTables.build_and_save_all_combined_tables(included_session_contexts, included_h5_paths, override_output_parent_path=override_output_parent_path, output_path_suffix=f'_{BATCH_DATE_TO_USE}')


        """

        # Get the combined tables:
        neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=should_restore_native_column_types)

        ## Potentially:
        # neuron_replay_stats_table = HDF_Converter.prepare_neuron_indexed_dataframe_for_hdf(neuron_replay_stats_table, active_context=curr_active_pipeline.get_session_context(), aclu_column_name=None)

        # Build the output paths:
        out_parent_path = cls.make_all_combined_output_directory(override_output_parent_path=override_output_parent_path, output_path_suffix=output_path_suffix)

        across_session_outputs = {'neuron_identities_table': neuron_identities_table,
        'long_short_fr_indicies_analysis_table': long_short_fr_indicies_analysis_table,
        'neuron_replay_stats_table': neuron_replay_stats_table}

        output_path_dicts = {}

        for table_name, v in across_session_outputs.items():
            table_name = Path(table_name)
            a_name = table_name.name
            print(f'table: "{a_name}"')
            output_path_dicts[a_name] = cls.write_table_to_files(v, global_data_root_parent_path=out_parent_path, output_basename=table_name, include_csv=include_csv, include_pkl=include_pkl)
            # v.to_hdf(k, key=f'/{a_name}', format='table', data_columns=True)    # TypeError: objects of type ``StringArray`` are not supported in this context, sorry; supported objects are: NumPy array, record or scalar; homogeneous list or tuple, integer, float, complex or bytes

        return (neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table), output_path_dicts


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

        ## Load the exported sessions experience_ranks CSV and use it to add the ['session_experience_rank', 'session_experience_orientation_rank'] columns to the tables:
        try:
            sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv("./data/sessions_experiment_datetime_df.csv", session_uid_str_sep='|')
            _loaded_tables = [_callback_add_df_columns(v, session_id_column_name='session_uid') for v in _loaded_tables]

        except BaseException as e:
            print(f'failed to load and apply the sessions rank CSV to tables. Error: {e}')
            raise e
        
        return _loaded_tables





    @function_attributes(short_name=None, tags=['HDF5'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-09 16:37', related_items=[])
    @classmethod
    def build_all_known_tables(cls, included_session_contexts, included_h5_paths, should_restore_native_column_types:bool=True):
        """ Extracts the neuron identities table from across the **.h5** files.
        One row for each neuron.

        Usage:

            neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=Falsee)

        """
        neuron_identities_table = AcrossSessionTables.build_neuron_identities_table(included_session_contexts, included_h5_paths, should_restore_native_column_types=should_restore_native_column_types)
        long_short_fr_indicies_analysis_table = AcrossSessionTables.build_long_short_fr_indicies_analysis_table(included_session_contexts, included_h5_paths, should_restore_native_column_types=should_restore_native_column_types)
        neuron_replay_stats_table = AcrossSessionTables.build_neuron_replay_stats_table(included_session_contexts, included_h5_paths, should_restore_native_column_types=should_restore_native_column_types)

        ## Load the exported sessions experience_ranks CSV and use it to add the ['session_experience_rank', 'session_experience_orientation_rank'] columns to the tables:
        try:
            sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv("./data/sessions_experiment_datetime_df.csv", session_uid_str_sep='|')
            neuron_identities_table = _callback_add_df_columns(neuron_identities_table, session_id_column_name='session_uid')
            neuron_replay_stats_table = _callback_add_df_columns(neuron_replay_stats_table, session_id_column_name='session_uid')
            long_short_fr_indicies_analysis_table = _callback_add_df_columns(long_short_fr_indicies_analysis_table, session_id_column_name='session_uid')

        except Exception as e:
            print(f'failed to load and apply the sessions rank CSV to tables. Error: {e}')
            raise e
        
        return neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table









# ==================================================================================================================== #
# 2024-01-27 - Across Session CSV Import and Processing                                                                #
# ==================================================================================================================== #
""" 
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import find_csv_files, find_HDF5_files, find_most_recent_files, read_and_process_csv_file

from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import find_csv_files, find_HDF5_files, find_most_recent_files, read_and_process_csv_file

"""
from typing import Dict, List, Tuple, Optional
import neuropy.utils.type_aliases as types
from attrs import define
from pyphocorehelpers.Filesystem.path_helpers import try_parse_chain, try_iterative_parse_chain # used in `parse_filename`


def build_session_t_delta(t_delta_csv_path: Path):
    """
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import build_session_t_delta
    t_delta_csv_path = collected_outputs_directory.joinpath('../2024-01-18_GL_t_split_df.csv').resolve() # GL
    # t_delta_csv_path = collected_outputs_directory.joinpath('2024-06-11_GL_t_split_df.csv').resolve()

    t_delta_df, t_delta_dict = build_session_t_delta(t_delta_csv_path=t_delta_csv_path)

    """
    assert t_delta_csv_path.exists(), f"t_split_df CSV at '{t_delta_csv_path}' does not exist!"
    ## The CSV containing the session delta time:
    t_delta_df = pd.read_csv(t_delta_csv_path, index_col=0, low_memory=False) # Assuming that your CSV file has an index column
    # adds `delta_aligned_t_start`, `delta_aligned_t_end` columns
    t_delta_df['delta_aligned_t_start'] = t_delta_df['t_start'] - t_delta_df['t_delta']
    t_delta_df['delta_aligned_t_end'] = t_delta_df['t_end'] - t_delta_df['t_delta']
    
    # computes `earliest_delta_aligned_t_start`, latest_delta_aligned_t_end
    earliest_delta_aligned_t_start: float = np.nanmin(t_delta_df['delta_aligned_t_start'])
    latest_delta_aligned_t_end: float = np.nanmax(t_delta_df['delta_aligned_t_end'])
    print(f'earliest_delta_aligned_t_start: {earliest_delta_aligned_t_start}, latest_delta_aligned_t_end: {latest_delta_aligned_t_end}')
    t_delta_dict = t_delta_df.to_dict(orient='index')
    return t_delta_df, t_delta_dict, (earliest_delta_aligned_t_start, latest_delta_aligned_t_end)

@function_attributes(short_name=None, tags=['experience_rank', 'session_order', 'csv'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-10 02:49', related_items=['find_build_and_save_sessions_experiment_datetime_df_csv'])
def load_and_apply_session_experience_rank_csv(csv_path="./data/sessions_experiment_datetime_df.csv", session_uid_str_sep: str = '|', novel_experience_rank_requirement: int= 2):
    """Load the exported sessions experience_ranks CSV and use it to add the ['session_experience_rank', 'session_experience_orientation_rank', 'is_novel_exposure'] columns to the tables:
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import load_and_apply_session_experience_rank_csv

        all_session_experiment_experience_csv_path = Path("./data/sessions_experiment_datetime_df.csv").resolve()
        Assert.path_exists(all_session_experiment_experience_csv_path)
        sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv(all_session_experiment_experience_csv_path, session_uid_str_sep='_')
        all_sessions_all_scores_ripple_df = _callback_add_df_columns(all_sessions_all_scores_ripple_df, session_id_column_name='session_name')

    """
    if isinstance(csv_path, str):
        csv_path = Path(csv_path).resolve()
        
    assert csv_path.exists(), f"csv_path: '{csv_path}' does not exist!"
    sessions_df: pd.DataFrame = pd.read_csv(csv_path, low_memory=False) # KDibaOldDataSessionFormatRegisteredClass.find_all_existing_sessions(global_data_root_parent_path=global_data_root_parent_path).sort_values(['session_datetime'])
            
    session_uid_strs = sessions_df['session_uid'].values
            
    if session_uid_str_sep != '|':
        ## replace the default separator character '|' with the user provided one
        session_uid_strs = [k.replace('|', session_uid_str_sep) for k in session_uid_strs]
        
    experience_rank_map_dict = dict(zip(session_uid_strs, sessions_df['experience_rank'].values))
    experience_orientation_rank_map_dict = dict(zip(session_uid_strs, sessions_df['experience_orientation_rank'].values))

    def _callback_add_df_columns(df, session_id_column_name: str = 'session_uid'):
        """ captures `experience_rank_map_dict`, `experience_orientation_rank_map_dict` """
        assert df is not None
        assert session_id_column_name in df.columns, f"session_id_column_name: '{session_id_column_name}' not in df.columns: {list(df.columns)}"
        df['session_experience_rank'] = df[session_id_column_name].map(experience_rank_map_dict)
        df['session_experience_orientation_rank'] = df[session_id_column_name].map(experience_orientation_rank_map_dict)
        df['is_novel_exposure'] = (df['session_experience_rank'] < novel_experience_rank_requirement)
        return df

    return sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns


@function_attributes(short_name=None, tags=['csv', 'filesystem', 'discovery'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-09 18:19', related_items=[])
def find_csv_files(directory: str, recurrsive: bool=False):
    """ 
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import find_csv_files, find_HDF5_files, find_pkl_files
    
    """
    directory_path = Path(directory) # Convert string path to a Path object
    if recurrsive:
        return list(directory_path.glob('**/*.csv')) # Return a list of all .csv files in the directory and its subdirectories
    else:
        return list(directory_path.glob('*.csv')) # Return a list of all .csv files in the directory and its subdirectories


@function_attributes(short_name=None, tags=['h5', 'filesystem', 'discovery'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-09 18:19', related_items=[])
def find_HDF5_files(directory: str, recurrsive: bool=False):
    directory_path = Path(directory) # Convert string path to a Path object
    if recurrsive:
        return list(directory_path.glob('**/*.h5')) # Return a list of all .csv files in the directory and its subdirectories
    else:
        return list(directory_path.glob('*.h5')) # Return a list of all .h5 files in the directory and its subdirectories

@function_attributes(short_name=None, tags=['pickle', 'pkl', 'filesystem', 'discovery'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-04 15:39', related_items=[])
def find_pkl_files(directory: str, recurrsive: bool=False):
    """ 
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import find_pkl_files
    
    """
    directory_path = Path(directory) # Convert string path to a Path object
    if recurrsive:
        return list(directory_path.glob('**/*.pkl')) # Return a list of all .csv files in the directory and its subdirectories
    else:
        return list(directory_path.glob('*.pkl')) # Return a list of all .csv files in the directory and its subdirectories




@function_attributes(short_name=None, tags=['parse'], input_requires=[], output_provides=[], uses=['try_parse_chain'], used_by=['find_most_recent_files'], creation_date='2024-03-28 10:16', related_items=[])
def parse_filename(path: Union[Path, str], should_print_unparsable_filenames: bool=True, debug_print:bool=False) -> Tuple[datetime, str, Optional[str], str, str]:
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

    filename = '2024-09-11_0150PM-kdiba_gor01_two_2006-6-12_16-53-46-(ripple_all_scores_merged_df)_tbin-0.058'
    

    # TESTING 2024-11-27 17:59 ___________________________________________________________________________________________ #
    ```
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import parse_filename

        test_filename_strs: Dict[str, str] = dict(date_day_only_name = "2024-11-27-kdiba_gor01_one_2006-6-12_15-55-31__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_all_scores_merged_df)_tbin-0.025",
                                                date_day_only_with_suffix_name = "2024-11-27_GL-kdiba_gor01_one_2006-6-12_15-55-31__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_all_scores_merged_df)_tbin-0.025",
        date_day_time_missing_custom_replay_name = "2024-11-27_1220PM-kdiba_gor01_one_2006-6-12_15-55-31-(laps_simple_pf_pearson_merged_df)_tbin-0.25",
        date_day_time_variant_suffix_name = "2024-11-27_1220PM_GL-kdiba_gor01_one_2006-6-12_15-55-31__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_all_scores_merged_df)_tbin-0.025",
        date_day_time_variant_suffix_missing_tbin_name = "2024-11-27_1220PM_GL-kdiba_gor01_one_2006-6-12_15-55-31__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_all_scores_merged_df)",
        )

        expected_parse_results_dict = dict(
            date_day_only_name = (datetime(2024, 11, 27, 0, 0), 'kdiba_gor01_one_2006-6-12_15-55-31', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0', 'ripple_all_scores_merged_df', '0.025'),
            date_day_only_with_suffix_name = (datetime(2024, 11, 27, 0, 0), 'kdiba_gor01_one_2006-6-12_15-55-31', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0', 'ripple_all_scores_merged_df', '0.025'),
            date_day_time_missing_custom_replay_name = (datetime(2024, 11, 27, 12, 20), 'kdiba_gor01_one_2006-6-12_15-55-31', None, 'laps_simple_pf_pearson_merged_df', '0.25'),
            date_day_time_variant_suffix_name = (datetime(2024, 11, 27, 12, 20), 'kdiba_gor01_one_2006-6-12_15-55-31', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0', 'ripple_all_scores_merged_df', '0.025'),
            date_day_time_variant_suffix_missing_tbin_name = (datetime(2024, 11, 27, 12, 20), 'kdiba_gor01_one_2006-6-12_15-55-31', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0', 'ripple_all_scores_merged_df', None)
        )

        ## perform the actual parsing:
        parse_result_dict = {k:parse_filename(v, debug_print=True) for k, v in test_filename_strs.items()}
        display(parse_result_dict)

        assert np.alltrue([(expected_parse_results_dict[k] == v) for k, v in parse_result_dict.items()]), f"not all parsed results match their expected results!"

        print('expected_parse_results_dict = dict(')
        print(',\n'.join([f"\t{k} = {str(v)}" for k, v in parse_result_dict.items()]))
        print(')\n')

    ```
    """
    if isinstance(path, str):
        filename: str = path  
    else:
        filename: str = path.stem  # Get filename without extension
    final_parsed_output_dict = try_parse_chain(basename=filename, should_print_unparsable_filenames=should_print_unparsable_filenames, debug_print=debug_print) ## previous implementation
    # final_parsed_output_dict = try_iterative_parse_chain(basename=filename)
    export_file_type = (final_parsed_output_dict or {}).get('export_file_type', None)
    session_str = (final_parsed_output_dict or {}).get('session_str', None)
    if ((export_file_type is not None) and (export_file_type in ['_withNormalComputedReplays'])) or ((session_str is not None) and (session_str in ['kdiba'])):
        ## do the new 2024-11-15 19:01 parse instead    
        print(f'Using `try_iterative_parse_chain(...)` for file "{filename}" (more modern parse method)...')
        # final_parsed_output_dict = try_parse_chain(basename=filename) ## previous implementation
        final_parsed_output_dict = try_iterative_parse_chain(basename=filename, should_print_unparsable_filenames=should_print_unparsable_filenames, debug_print=debug_print)
    # if final_parsed_output_dict is None:
    #     ## this version failed, fall-back to the older implementation
    #     final_parsed_output_dict = try_parse_chain(basename=filename) ## previous implementation
    
    if final_parsed_output_dict is None:
        if should_print_unparsable_filenames:
            print(f'ERR: Could not parse filename: "{filename}"') # 2024-01-18_GL_t_split_df
        return None, None, None, None, None # used to return ValueError when it couldn't parse, but we'd rather skip unparsable files


    export_datetime, session_str, export_file_type = final_parsed_output_dict.get('export_datetime', None), final_parsed_output_dict.get('session_str', None), final_parsed_output_dict.get('export_file_type', None)
    decoding_time_bin_size_str = final_parsed_output_dict.get('decoding_time_bin_size_str', None)
    custom_replay_name = final_parsed_output_dict.get('custom_replay_name', None)

    if export_file_type is not None:
        if export_file_type[0] == '(' and export_file_type[-1] == ')':
            # Trim the brackets from the file type if they're present:
            export_file_type = export_file_type[1:-1]

    return export_datetime, session_str, custom_replay_name, export_file_type, decoding_time_bin_size_str


top_level_parts_separators = ['-', '__']

# def get_only_most_recent_output_files(a_file_df: pd.DataFrame) -> pd.DataFrame:
#     """ returns a dataframe containing only the most recent '.err' and '.log' file for each session. 
    
#     from phoglobushelpers.compatibility_objects.Files import File, FilesystemDataType, FileList, get_only_most_recent_log_files
    
    
#     """
    

#     df = deepcopy(a_file_df)

#     required_cols = ['last_modified', 'parent_path', 'name'] # Replace with actual column names you require
#     has_required_columns = PandasHelpers.require_columns(df, required_cols, print_missing_columns=True)
#     assert has_required_columns

#     df['last_modified'] = pd.to_datetime(df['last_modified'])

#     # Separate .csv and .h5 files
#     csv_files = df[df['name'].str.endswith('.csv')]
#     h5_files = df[df['name'].str.endswith('.h5')]

#     # Get the most recent .err and .log file for each parent_path
#     most_recent_csv = csv_files.loc[csv_files.groupby('parent_path')['last_modified'].idxmax()]
#     most_recent_h5 = h5_files.loc[h5_files.groupby('parent_path')['last_modified'].idxmax()]

#     # Concatenate the results
#     most_recent_files = pd.concat([most_recent_csv, most_recent_h5]).sort_values(by=['parent_path', 'last_modified'], ascending=[True, False])
#     return most_recent_files


def get_only_most_recent_session_files(parsed_paths_df: pd.DataFrame, group_column_names=['session', 'custom_replay_name', 'file_type', 'decoding_time_bin_size_str'], sort_datetime_col_name:str='export_datetime') -> pd.DataFrame:
    """ returns a dataframe containing only the most recent '.err' and '.log' file for each session. 
    
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import get_only_most_recent_csv_sessions
    
    ['session', 'custom_replay_name', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime']

    I now have a different dataframe where I want to get the most recent (according to the 'export_datetime' column) for each of the groups, grouped by the following columns:
    ['session', 'custom_replay_name', 'file_type', 'decoding_time_bin_size_str']


    """
    df = deepcopy(parsed_paths_df)

    # required_cols = ['session', 'custom_replay_name', 'file_type', 'decoding_time_bin_size_str', 'export_datetime'] # Replace with actual column names you require
    required_cols = deepcopy(group_column_names) + [sort_datetime_col_name, ]
    has_required_columns = PandasHelpers.require_columns(df, required_cols, print_missing_columns=True)
    assert has_required_columns

    df[sort_datetime_col_name] = pd.to_datetime(df[sort_datetime_col_name])
    # Replace NaN values with empty strings
    df.fillna('', inplace=True)

    # Get the most recent entry for each group
    most_recent_entries_df: pd.DataFrame = df.loc[df.groupby(group_column_names)[sort_datetime_col_name].idxmax()]
    return most_recent_entries_df



# most_recent_only_csv_file_df = get_only_most_recent_log_files(log_file_df=all_file_df)


@function_attributes(short_name=None, tags=['recent', 'parse'], input_requires=[], output_provides=[], uses=['parse_filename'], used_by=[], creation_date='2024-04-15 09:18', related_items=['convert_to_dataframe'])
def find_most_recent_files(found_session_export_paths: List[Path], cuttoff_date:Optional[datetime]=None, debug_print: bool = False, should_fallback_to_filesystem_modification_datetime: bool=False) -> Tuple[Dict[str, Dict[str, Tuple[Path, str, datetime]]], pd.DataFrame, pd.DataFrame]:
    """
    Returns a dictionary representing the most recent files for each session type among a list of provided file paths.

    Parameters:
    found_session_export_paths (List[Path]): A list of Paths representing files to be checked.
    cuttoff_date (datetime): a date which all files must be newer than to be considered for inclusion. If not provided, the most recent files will be included regardless of their date.
    debug_print (bool): A flag to trigger debugging print statements within the function. Default is False.
    should_fallback_to_filesystem_modification_datetime (bool): If True, use file modification time when export date can't be parsed from filename. Default is False.

    Returns:
    Dict[str, Dict[str, Tuple[Path, datetime]]]: A nested dictionary where the main keys represent
    different session types. The inner dictionary's keys represent file types and values are the most recent
    Path and datetime for this combination of session and file type.

    # now sessions is a dictionary where the key is the session_str and the value is another dictionary.
    # This inner dictionary's key is the file type and the value is the most recent path for this combination of session and file type
    # Thus, laps_csv and ripple_csv can be obtained from the dictionary for each session
    """
    # Process each path to extract information or use file modification time as fallback
    parsed_paths: List[Tuple] = []

    # Original behavior: filter out files where parse_filename returns None for the first element
    if not should_fallback_to_filesystem_modification_datetime:
        parsed_paths = [(*parse_filename(p), p) for p in found_session_export_paths if (parse_filename(p)[0] is not None)]
    else:
        # Extended behavior with fallback to file modification time
        for p in found_session_export_paths:
            parsed_info = parse_filename(p)
            if parsed_info[0] is not None:
                # Use parsed datetime from filename
                parsed_paths.append((*parsed_info, p))
            else:
                # Fallback to file modification time
                try:
                    # Get file modification time
                    mtime = datetime.fromtimestamp(p.stat().st_mtime)
                    # Try to parse other info, use fallback values if needed
                    session_str = parsed_info[1] if parsed_info[1] is not None else "unknown_session"
                    custom_replay_name = parsed_info[2] if parsed_info[2] is not None else ""
                    file_type = parsed_info[3] if parsed_info[3] is not None else p.suffix.lstrip('.')
                    tbin_size = parsed_info[4] if parsed_info[4] is not None else ""
                    parsed_paths.append((mtime, session_str, custom_replay_name, file_type, tbin_size, p))
                    if debug_print:
                        print(f"Using file modification time for {p.name}: {mtime}")
                except Exception as e:
                    if debug_print:
                        print(f"Skipping file {p.name} due to error: {e}")
                    continue

    # Function that helps sort tuples by handling None values.
    def sort_key(tup):
        # Assign a boolean for each element, True if it's None, to ensure None values are sorted last.
        return (
            tup[0],                       # Sort by datetime first
            tup[1] or '',                  # Then by the session string, ensuring None becomes empty string
            tup[2] or '',                  # Then by the next string, ensuring None becomes empty string
            tup[3] or '',                  # Then by the next string, ensuring None becomes empty string
            # tup[4] or '',                  #float('-inf') if tup[4] is None else tup[3],  # Then use -inf to ensure None ends up last
            tup[-1]                         # Finally by path which should handle None by itself
        )

    # Now we sort the data using our custom sort key
    parsed_paths = sorted(parsed_paths, key=sort_key, reverse=True)

    if debug_print:
        print(f'parsed_paths: {parsed_paths}')

    tuple_column_names = ['export_datetime', 'session', 'custom_replay_name', 'file_type', 'decoding_time_bin_size_str', 'path']
    all_parsed_paths_df: pd.DataFrame = pd.DataFrame(parsed_paths, columns=tuple_column_names)

    # compare_custom_replay_name_col_name: str = 'custom_replay_name'
    compare_custom_replay_name_col_name: str = '_comparable_custom_replay_name'

    ## replace undscores with hyphens so comparisons are insensitive to whether '_' or '-' are used:
    all_parsed_paths_df['_comparable_custom_replay_name'] = all_parsed_paths_df['custom_replay_name'].str.replace('_','-') # replace both with hyphens so they match

    # Sort by columns: 'session' (ascending), 'custom_replay_name' (ascending) and 3 other columns
    all_parsed_paths_df = all_parsed_paths_df.sort_values(['session', 'file_type', compare_custom_replay_name_col_name, 'decoding_time_bin_size_str', 'export_datetime']).reset_index(drop=True)

    # get_only_most_recent_session_files _________________________________________________________________________________ #
    ## This is where we drop all but the most recent:
    filtered_parsed_paths_df = get_only_most_recent_session_files(parsed_paths_df=deepcopy(all_parsed_paths_df), group_column_names=['session', compare_custom_replay_name_col_name, 'file_type', 'decoding_time_bin_size_str']) ## `filtered_parsed_paths_df` still has it

    # Drop rows with export_datetime less than or equal to cutoff_date
    if cuttoff_date is not None:
        filtered_parsed_paths_df = filtered_parsed_paths_df[filtered_parsed_paths_df['export_datetime'] >= cuttoff_date]

    ## filtered_parsed_paths_df is new 2024-07-11 output, NOTE we don't use `filtered_parsed_paths_df` below, instead `parsed_paths`.

    sessions = {}
    for export_datetime, session_str, custom_replay_name, file_type, decoding_time_bin_size_str, path in parsed_paths:
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

    return sessions, filtered_parsed_paths_df, all_parsed_paths_df


# @function_attributes(short_name=None, tags=['DEPRICATED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-15 09:18', related_items=['find_most_recent_files'])
# def convert_to_dataframe(csv_sessions: Dict[str, Dict[str, Tuple[Path, str, datetime]]], parse_columns: List[str], debug_print:bool=False) -> pd.DataFrame:
#     """ Converts the outp[ut of `find_most_recent_files` into a dataframe.

#     from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import convert_to_dataframe

#     CSV_parse_fields = ['session', 'custom_replay_name', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime']
#     parsed_files_df: pd.DataFrame = convert_to_dataframe(csv_sessions, parse_columns=CSV_parse_fields)
#     parsed_files_df

#     CSV_parse_fields = ['session', 'custom_replay_name', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime']
#     H5_parse_fields = ['session', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime']
    
#     """
#     _output_tuples = []

#     for session_str, a_filetype_dict in csv_sessions.items():
#         if debug_print:
#             print(f'session_str: {session_str}')
#         for file_type, parse_tuple in a_filetype_dict.items():
#             if debug_print:
#                 print(f'\tfile_type: {file_type}')
#                 print(f'\t\tparse_tuple: {parse_tuple}')
#             # path, decoding_time_bin_size_str, export_datetime = parse_tuple
#             if len(parse_tuple) == len(parse_columns):
#                 _output_tuples.append((session_str, file_type, *parse_tuple))
#             else:
#                 ## don't add to the list for now
#                 print(f'\tWARNING: parse tuple skipped due to inadequazte length! required length (len(parse_columns)): {len(parse_columns)})\t actual_length: {len(parse_tuple)}')


#     return pd.DataFrame(_output_tuples, columns=parse_columns)


@function_attributes(short_name=None, tags=['csv'], input_requires=[], output_provides=[], uses=[], used_by=['_process_and_load_exported_file'], creation_date='2024-07-09 18:19', related_items=[])
def read_and_process_csv_file(file: str, session_name: str, curr_session_t_delta: Optional[float], time_col: str) -> pd.DataFrame:
    """ reads the CSV file and adds the 'session_name' column if it is missing.

    """
    df = pd.read_csv(file, na_values=['', 'nan', 'np.nan', '<NA>'], low_memory=False)
    df['session_name'] = session_name
    if curr_session_t_delta is not None:
        df['delta_aligned_start_t'] = df[time_col] - curr_session_t_delta
    return df


@function_attributes(short_name=None, tags=['csv', 'export', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-09 18:20', related_items=[])
def export_across_session_CSVs(final_output_path: Path, TODAY_DAY_DATE:str, all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df,
                                all_sessions_simple_pearson_laps_df=None, all_sessions_simple_pearson_ripple_df=None,
                                all_sessions_MultiMeasure_laps_df=None, all_sessions_MultiMeasure_ripple_df=None,                               
                                all_sessions_all_scores_ripple_df=None, all_sessions_all_scores_laps_df=None):
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
                        f"{across_session_output_df_prefix}_MultiMeasure_Laps_per-Epoch": all_sessions_MultiMeasure_laps_df, f"{across_session_output_df_prefix}_MultiMeasure_Ripple_per-Epoch": all_sessions_MultiMeasure_ripple_df,
                        f"{across_session_output_df_prefix}_AllScores_Ripple_per-Epoch": all_sessions_all_scores_ripple_df, #,
                        }
    
    # final_dfs_dict.update({
    #     f"{across_session_output_df_prefix}_MultiMeasure_Laps_per-Epoch": all_sessions_MultiMeasure_laps_df, f"{across_session_output_df_prefix}_MultiMeasure_Ripple_per-Epoch": all_sessions_MultiMeasure_ripple_df,
    # })

    if all_sessions_all_scores_laps_df is not None:
        final_dfs_dict.update({f"{across_session_output_df_prefix}_AllScores_Laps_per-Epoch": all_sessions_all_scores_laps_df})

    final_dfs_dict = {k:v for k, v in final_dfs_dict.items() if v is not None} ## filter None entries

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


    @classmethod
    def debug_print_discovered_csv_infos(cls, sessions_df, all_sessions_laps_time_bin_df, all_sessions_all_scores_ripple_df, all_sessions_laps_df, all_sessions_simple_pearson_laps_df, all_sessions_ripple_df, all_sessions_ripple_time_bin_df, **kwargs):
        """ # 🔜🔜 MAJOR: trying to get to the bottom of the basic marginal exports not loading
        """
        from IPython.display import display
        
        # Wrap each of these output statements that were copied from a notebook with the appropriate print or display function and a short description/label of each
        
        # Check for NaNs in 'time_bin_size'
        print("Number of NaNs in 'time_bin_size' of 'all_sessions_laps_time_bin_df':", 
            np.sum(all_sessions_laps_time_bin_df['time_bin_size'].isna()))
        
        # Check for entries where 'time_bin_size' equals np.nan (this will always be zero because np.nan != np.nan)
        print("Number of entries where 'time_bin_size' == np.nan in 'all_sessions_laps_time_bin_df':", 
            np.sum(all_sessions_laps_time_bin_df['time_bin_size'] == np.nan))
        
        # Number of unique 'time_bin_size' values
        print("Number of unique 'time_bin_size' in 'all_sessions_laps_time_bin_df':", 
            all_sessions_laps_time_bin_df['time_bin_size'].nunique())
        
        # Number of unique 'session_name' values
        print("Number of unique 'session_name' in 'all_sessions_laps_time_bin_df':", 
            all_sessions_laps_time_bin_df['session_name'].nunique())
        
        # Print the columns of 'sessions_df'
        print("Columns in 'sessions_df':", list(sessions_df.columns))
        
        # # Display selected columns from 'sessions_df'
        # print("Selected columns from 'sessions_df':")
        # print(sessions_df[['format_name', 'animal', 'exper_name', 'session_name']])
        
        # Ensure no duplicates in specified columns
        print("Checking for duplicates in ['format_name', 'animal', 'exper_name', 'session_name']:")
        if sessions_df[['format_name', 'animal', 'exper_name', 'session_name']].duplicated().any():
            raise ValueError("Duplicate entries found in the specified columns.")
        
        # Ensure no duplicates in 'session_uid'
        print("Checking for duplicates in 'session_uid' column:")
        if sessions_df[['session_uid']].duplicated().any():
            raise ValueError("Duplicate entries found in the 'session_uid' column.")
        
        # Create a deep copy of 'sessions_df'
        df: pd.DataFrame = deepcopy(sessions_df)
        
        # Get unique values
        unique_animals = df['animal'].unique().tolist()
        unique_exper_names = df['exper_name'].unique().tolist()
        unique_session_names = df['session_name'].unique().tolist()
        
        # Build sessions trees
        sessions_tree = {}
        sessions_number_tree = {}
        
        for animal in unique_animals:
            animal_sessions = df[df['animal'] == animal]['session_name'].unique().tolist()
            sessions_tree[animal] = animal_sessions
            sessions_number_tree[animal] = len(animal_sessions)
        
        # Display sessions number tree
        print("Sessions number tree:")
        display(sessions_number_tree)
        
        # Perform aggregation grouped on 'animal' and 'exper_name'
        sessions_count_df = sessions_df.groupby(['animal', 'exper_name']).agg(
            session_name_count=('session_name', 'count')
        ).reset_index()
        
        # Display session counts
        # print("Session counts grouped by 'animal' and 'exper_name':")
        display("Session counts grouped by 'animal' and 'exper_name':", sessions_count_df, '\n\n')
        
        # Use 'all_sessions_laps_time_bin_df' for further analysis
        a_df: pd.DataFrame = all_sessions_laps_time_bin_df
        
        # Perform aggregations grouped on 'session_name'
        n_time_bin_sizes_per_session = a_df.groupby(['session_name']).agg(
            time_bin_size_nunique=('time_bin_size', 'nunique'),
            time_bin_size_max=('time_bin_size', 'max'),
            time_bin_size_min=('time_bin_size', 'min')
        )
        print("Number of unique 'time_bin_size' per 'session_name':")
        display(n_time_bin_sizes_per_session)
        
        # Display unique 'time_bin_size' values in different DataFrames
        print("Unique 'time_bin_size' in 'all_sessions_all_scores_ripple_df':", 
            all_sessions_all_scores_ripple_df['time_bin_size'].unique())
        print("Unique 'time_bin_size' in 'all_sessions_laps_time_bin_df':", 
            all_sessions_laps_time_bin_df['time_bin_size'].unique())
        print("Unique 'time_bin_size' in 'all_sessions_laps_df':", 
            all_sessions_laps_df['time_bin_size'].unique())
        # print("Unique 'time_bin_size' in 'all_sessions_wcorr_laps_df':", 
        # 	all_sessions_wcorr_laps_df['time_bin_size'].unique())
        print("Unique 'time_bin_size' in 'all_sessions_simple_pearson_laps_df':", 
            all_sessions_simple_pearson_laps_df['time_bin_size'].unique())
        print("Unique 'time_bin_size' in 'all_sessions_ripple_df':", 
            all_sessions_ripple_df['time_bin_size'].unique())
        print("Unique 'time_bin_size' in 'all_sessions_ripple_time_bin_df':", 
            all_sessions_ripple_time_bin_df['time_bin_size'].unique())
        

        for name, df in kwargs.items():
            print(f"Unique 'time_bin_size' in '{name}':", 
            df['time_bin_size'].unique())
        
        # Aggregated counts of 'time_bin_size' per 'session_name'
        print("\nAggregated 'time_bin_size' counts per 'session_name' in 'all_sessions_laps_time_bin_df':")
        time_bin_agg_df = all_sessions_laps_time_bin_df.groupby(['session_name']).agg(
            time_bin_size_count=('time_bin_size', 'count'),
            time_bin_size_nunique=('time_bin_size', 'nunique')
        ).reset_index()
        display(time_bin_agg_df)
        
        # # Analyze a specific session
        # test_session_name: str = 'kdiba_gor01_two_2006-6-07_16-40-19'
        # _prev_loaded_df: pd.DataFrame = deepcopy(
        #     all_sessions_laps_df[all_sessions_laps_df['session_name'] == test_session_name]
        # )
        # print(f"Data for session '{test_session_name}' in 'all_sessions_laps_df':")
        # display(_prev_loaded_df)





def _split_user_annotated_ripple_df(all_sessions_user_annotated_ripple_df):
    """ prints info about exported data sessions, such as the breakdown of user-annotated epochs, etc.

    Usage:

        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import _split_user_annotated_ripple_df

        all_sessions_all_scores_ripple_df, (valid_ripple_df, invalid_ripple_df), (user_approved_ripple_df, user_rejected_ripple_df) = _split_user_annotated_ripple_df(all_sessions_all_scores_ripple_df)



    """
    from pyphocorehelpers.indexing_helpers import partition_df


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

    return all_sessions_user_annotated_ripple_df, (valid_ripple_df, invalid_ripple_df), (user_approved_ripple_df, user_rejected_ripple_df)



# ==================================================================================================================== #
# 2024-04-15 - Factor out of Across Session Point and YellowBlue Marginal CSV Exports                                  #
# ==================================================================================================================== #

from neuropy.utils.indexing_helpers import PandasHelpers
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path

@function_attributes(short_name=None, tags=['csv'], input_requires=[], output_provides=[], uses=['read_and_process_csv_file'], used_by=[], creation_date='2024-07-10 16:52', related_items=[])
def _process_and_load_exported_file(session_dict, df_file_name_key: str, loaded_dict: Dict, session_name: str, curr_session_t_delta: float, time_key: str, debug_print:bool=False) -> None:
    """ updates loaded_dict """
    try:
        file_path = session_dict[df_file_name_key]
        loaded_dict[session_name] = read_and_process_csv_file(file_path, session_name, curr_session_t_delta, time_key)
    except BaseException as e:
        if debug_print:
            print(f'session "{session_name}", df_file_name_key: "{df_file_name_key}" - did not fully work. (error "{e}". Skipping.')


def _common_cleanup_operations(a_df):
    """ post loading and concatenation across sessions dataframe cleanup """
    if a_df is None:
        return None
    ## Drop the weird 'Unnamed: 0' column:
    # Rename column 'Unnamed: 0' to 'abs_time_bin_index'
    a_df = a_df.rename(columns={'Unnamed: 0': 'abs_time_bin_index'})
    if 'abs_time_bin_index' in a_df.columns:
        # Drop column: 'abs_time_bin_index'
        a_df = a_df.drop(columns=['abs_time_bin_index'])

    # Add additional 'epoch_idx' column for compatibility:
    if 'epoch_idx' not in a_df:
        if 'lap_idx' in a_df:
            a_df['epoch_idx'] = a_df['lap_idx']
        if 'ripple_idx' in a_df:
            a_df['epoch_idx'] = a_df['ripple_idx']
    return a_df


@function_attributes(short_name=None, tags=['HDF5', 'h5', 'load'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-04 06:38', related_items=[])
def load_across_sessions_exported_h5_files(cuttoff_date: Optional[datetime] = None, collected_outputs_directory: Optional[Path]=None, known_bad_session_strs=None, debug_print: bool = False):
    """

    #TODO 2025-04-18 06:22: - [ ] Where do the returned `h5_contexts_paths_dict` come frome?
    
    
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import load_across_sessions_exported_h5_files

    

    """
    from neuropy.core.user_annotations import UserAnnotationsManager
    from pyphocorehelpers.indexing_helpers import safe_get_if_not_None
    H5_parse_fields = ['session', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime']
    # H5_parse_fields = ['session', 'decoding_time_bin_size_str', 'export_datetime']
    # H5_parse_fields = ['session', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime']
    

    if collected_outputs_directory is None:
        known_collected_outputs_paths = [Path(v).resolve() for v in [r"K:/scratch/collected_outputs", '/Users/pho/Dropbox (University of Michigan)/MED-DibaLabDropbox/Data/Pho/Outputs/output/collected_outputs', r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs',
                                                                    '/home/halechr/FastData/collected_outputs/', '/home/halechr/cloud/turbo/Data/Output/collected_outputs']]
        collected_outputs_directory = find_first_extant_path(known_collected_outputs_paths)

    assert collected_outputs_directory.exists(), f"collected_outputs_directory: {collected_outputs_directory} does not exist! Is the right computer's config commented out above?"
    # fullwidth_path_widget(scripts_output_path, file_name_label='Scripts Output Path:')
    print(f'collected_outputs_directory: {collected_outputs_directory}')

    ## Find the files:
    h5_files = find_HDF5_files(collected_outputs_directory)
    h5_sessions, h5_filtered_parsed_paths_df, all_parsed_paths_df = find_most_recent_files(found_session_export_paths=h5_files, cuttoff_date=cuttoff_date)
        
    ## INPUTS: h5_sessions, session_dict, cuttoff_date, known_bad_session_strs
    if known_bad_session_strs is None:
        known_bad_session_strs = []

    # parsed_h5_files_df: pd.DataFrame = convert_to_dataframe(h5_sessions, parse_columns=H5_parse_fields)

    if cuttoff_date is not None:
        # 'session', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime'
        h5_filtered_parsed_paths_df = h5_filtered_parsed_paths_df[h5_filtered_parsed_paths_df['export_datetime'] >= cuttoff_date]


    h5_filtered_parsed_paths_df = h5_filtered_parsed_paths_df[np.isin(h5_filtered_parsed_paths_df['session'], known_bad_session_strs, invert=True)] # drop all sessions that are in the known_bad_session_strs

    # parsed_h5_files_df: pd.DataFrame = convert_to_dataframe(final_h5_sessions)

    ## INPUTS: h5_sessions
    h5_session_names = list(h5_sessions.keys())
    good_sessions = UserAnnotationsManager.get_hardcoded_good_sessions()
    

    def _subfn_session_ctxt_to_dict_key(a_ctxt: IdentifyingContext):
        # return a_ctxt.session_name #TODO 2025-04-18 06:34: - [ ] OLD, uses session name '2006-6-08_14-26-15'
        return a_ctxt.get_description(subset_includelist=['format_name', 'animal', 'exper_name', 'session_name'], separator='_') #TODO 2025-04-18 06:34: - [ ] NEW: entire session context string like 'kdiba_gor01_one_2006-6-08_14-26-15'
    
    # [a_good_session_ctxt.get_description(subset_includelist=['format_name', 'animal', 'exper_name', 'session_name'], separator='_') for a_good_session_ctxt in good_sessions]

    # h5_session_contexts = [a_good_session_ctxt for a_good_session_ctxt in good_sessions if (a_good_session_ctxt.session_name in h5_session_names)]
    h5_session_contexts = [a_good_session_ctxt for a_good_session_ctxt in good_sessions if (_subfn_session_ctxt_to_dict_key(a_good_session_ctxt) in h5_session_names)] #TODO 2025-04-18 06:28: - [ ] Had to change to match on just session name to matching on entire string, might need to go farther and add qclu and frHZ filter

    # included_h5_paths = [a_session_dict.get('pipeline_results', None)[0] for a_sess_name, a_session_dict in h5_sessions.items()] # these are mis-ordered
    # included_h5_paths = [safe_get_if_not_None(h5_sessions[a_good_session_ctxt.session_name].get('pipeline_results', None), 0, None) for a_good_session_ctxt in h5_session_contexts]
    included_h5_paths = [safe_get_if_not_None(h5_sessions[_subfn_session_ctxt_to_dict_key(a_good_session_ctxt)].get('pipeline_results', None), 0, None) for a_good_session_ctxt in h5_session_contexts]
    assert len(included_h5_paths) == len(h5_session_contexts)

    h5_contexts_paths_dict = dict(zip(h5_session_contexts, included_h5_paths))
    return h5_filtered_parsed_paths_df, h5_contexts_paths_dict

    ## OUTPUTS: parsed_h5_files_df, h5_contexts_paths_dict
    # h5_session_contexts = list(h5_contexts_paths_dict.keys())
    # included_h5_paths = list(h5_contexts_paths_dict.values())

    ## OUTPUTS: (csv_files, csv_sessions), (h5_files, h5_sessions)




@function_attributes(short_name=None, tags=['MAIN', 'across_sessions'], input_requires=[], output_provides=[], uses=['find_csv_files', 'find_HDF5_files', 'find_most_recent_files', '_new_process_csv_files', '_common_cleanup_operations'], used_by=[], creation_date='2024-04-15 08:47', related_items=[])
def load_across_sessions_exported_files(cuttoff_date: Optional[datetime] = None, collected_outputs_directory: Optional[Path]=None, debug_print: bool = False):
    """ Discovers all exported CSV files in collected_outputs, parses their filenames to determine the most recent one for each session, and builds joined dataframes from these values.
    
    #TODO 2024-10-23 05:12: - [ ] updated to use the modern `_new_process_csv_files`-based approach
    
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import load_across_sessions_exported_files
    final_sessions, sessions_t_delta_tuple, df_results, (most_recent_parsed_csv_files_df, all_parsed_csv_files_df, csv_files, csv_sessions), (most_recent_parsed_h5_files_df, all_parsed_h5_files_df, h5_files, h5_sessions), excluded_or_outdated_files_list = load_across_sessions_exported_files(collected_outputs_directory=collected_outputs_directory, cuttoff_date=cuttoff_date, debug_print=True)
    # final_sessions, sessions_t_delta_tuple, df_results, (parsed_csv_files_df, csv_files, csv_sessions), (parsed_h5_files_df, h5_files, h5_sessions), excluded_or_outdated_files_list = load_across_sessions_exported_files(cuttoff_date=cuttoff_date, debug_print=True)
    all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_MultiMeasure_laps_df, all_sessions_MultiMeasure_ripple_df, all_sessions_all_scores_ripple_df = df_results
    t_delta_df, t_delta_dict, (earliest_delta_aligned_t_start, latest_delta_aligned_t_end) = sessions_t_delta_tuple ## UNPACK

    """
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
    from neuropy.core.user_annotations import UserAnnotationsManager

    known_bad_sessions: List[IdentifyingContext] = UserAnnotationsManager.get_hardcoded_bad_sessions()
    # bad_session_df: pd.DataFrame = pd.DataFrame.from_records([v.to_dict() for v in known_bad_sessions], columns=['format_name', 'animal', 'exper_name', 'session_name'])
    # bad_session_df

    known_bad_session_strs = [str(v.get_description()) for v in known_bad_sessions]

    ## Load across session t_delta CSV, which contains the t_delta for each session:
    if collected_outputs_directory is None:
        known_collected_outputs_paths = [Path(v).resolve() for v in ['/Users/pho/data/collected_outputs',
                                                                    '/Volumes/SwapSSD/Data/collected_outputs', r"K:/scratch/collected_outputs", '/Users/pho/Dropbox (University of Michigan)/MED-DibaLabDropbox/Data/Pho/Outputs/output/collected_outputs', r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs',
                                                                    '/home/halechr/cloud/turbo/Data/Output/collected_outputs',
                                                                    '/home/halechr/cloud/turbo/Pho/Output/collected_outputs',
                                                                    '/home/halechr/FastData/collected_outputs/',
                                                                    ]]
        collected_outputs_directory = find_first_extant_path(known_collected_outputs_paths)

    assert collected_outputs_directory.exists(), f"collected_outputs_directory: {collected_outputs_directory} does not exist! Is the right computer's config commented out above?"
    # fullwidth_path_widget(scripts_output_path, file_name_label='Scripts Output Path:')
    print(f'collected_outputs_directory: {collected_outputs_directory}')
    
    ## sessions' t_delta:
    # t_delta_csv_path = collected_outputs_directory.joinpath('../2024-01-18_GL_t_split_df.csv').resolve() # GL
    # t_delta_csv_path = collected_outputs_directory.joinpath('../2024-09-25_GL_t_split_df.csv').resolve()
    known_t_delta_csv_paths = [Path(v).resolve() for v in [collected_outputs_directory.joinpath('../2024-09-25_GL_t_split_df.csv'),
                                                            collected_outputs_directory.joinpath('2024-09-25_GL_t_split_df.csv').resolve(),
                                                            collected_outputs_directory.joinpath('../2024-01-18_GL_t_split_df.csv').resolve()
                                                            ]]
    t_delta_csv_path = find_first_extant_path(known_t_delta_csv_paths)

    # t_delta_csv_path = collected_outputs_directory.joinpath('../2024-09-25_GL_t_split_df.csv').resolve()

    Assert.path_exists(t_delta_csv_path)
    sessions_t_delta_tuple = build_session_t_delta(t_delta_csv_path=t_delta_csv_path)
    t_delta_df, t_delta_dict, (earliest_delta_aligned_t_start, latest_delta_aligned_t_end) = sessions_t_delta_tuple ## UNPACK
    # 
    

    ## Find the files:
    csv_files = find_csv_files(collected_outputs_directory)
    h5_files = find_HDF5_files(collected_outputs_directory)

    csv_sessions, most_recent_parsed_csv_files_df, all_parsed_csv_files_df  = find_most_recent_files(found_session_export_paths=csv_files, cuttoff_date=cuttoff_date) # #TODO 2024-09-27 02:01: - [ ] Note `csv_sessions` is unused, replaced by `parsed_csv_files_df`
    h5_sessions, most_recent_parsed_h5_files_df, all_parsed_h5_files_df = find_most_recent_files(found_session_export_paths=h5_files, cuttoff_date=cuttoff_date)

    ## OUTPUTS: (csv_files, csv_sessions), (h5_files, h5_sessions)

    # # #TODO 2024-03-02 12:12: - [ ] Could add weighted correlation if there is a dataframe for that and it's computed:
    # _df_raw_variable_names = ['simple_pf_pearson_merged_df', 'weighted_corr_merged_df']
    # _df_variables_names = ['laps_weighted_corr_merged_df', 'ripple_weighted_corr_merged_df', 'laps_simple_pf_pearson_merged_df', 'ripple_simple_pf_pearson_merged_df']

    # # # tbin_values_dict = {'laps': self.laps_decoding_time_bin_size, 'ripple': self.ripple_decoding_time_bin_size}
    # time_col_name_dict = {'laps': 'lap_start_t', 'ripple': 'ripple_start_t'} ## default should be 't_bin_center'

    # fold older files:
    # {'laps_marginals_df': 'lap_start_t', 'ripple_marginals_df': 'ripple_start_t', 'laps_time_bin_marginals_df':'t_bin_center', 'ripple_time_bin_marginals_df':'t_bin_center'}

    # csv_sessions
    # Extract each of the separate files from the sessions:

    # final_sessions: Dict[types.session_str, Dict[str, Path]] = {}
    if cuttoff_date is not None:
        final_sessions: Dict[types.session_str, Dict[str, Path]] = {session_str:{file_type:a_path for file_type, (a_path, an_decoding_time_bin_size_str, an_export_datetime) in session_dict.items() if (an_export_datetime >= cuttoff_date)}
                                                                                                for session_str, session_dict in csv_sessions.items() }
    else:
        # no cutoff recency date:
        final_sessions: Dict[types.session_str, Dict[str, Path]] = {session_str:{file_type:a_path for file_type, (a_path, an_decoding_time_bin_size_str, an_export_datetime) in session_dict.items()}
                                                                                                for session_str, session_dict in csv_sessions.items()}


    # 2024-10-23 - modern `_new_process_csv_files` method ________________________________________________________________ #
    all_session_experiment_experience_csv_path = find_first_extant_path(path_list=[Path("./EXTERNAL/sessions_experiment_datetime_df.csv").resolve(),
                                        Path("../sessions_experiment_datetime_df.csv").resolve(),
                                        # Path('EXTERNAL/PhoDibaPaper2024Book/data/neptune').resolve(),
                                        # Path('/Users/pho/repo/Pho Secondary Workspace/Spike3DEnv/Spike3DWorkEnv/Spike3D/EXTERNAL/PhoDibaPaper2024Book/data').resolve(),
        ])
        
    Assert.path_exists(all_session_experiment_experience_csv_path)
    ## NEW `parsed_csv_files_df1-based approach 2024-07-11 - 
    ## INPUTS: parsed_csv_files_df
    dict_results, df_results, excluded_or_outdated_files_list = _new_process_csv_files(parsed_csv_files_df=most_recent_parsed_csv_files_df, t_delta_dict=t_delta_dict, cuttoff_date=cuttoff_date, known_bad_session_strs=known_bad_session_strs, all_session_experiment_experience_csv_path=all_session_experiment_experience_csv_path, debug_print=False) # , known_bad_session_strs=known_bad_session_strs
    (final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict, final_sessions_loaded_merged_complete_epoch_stats_df_dict, *final_sessions_loaded_extra_df_dicts_list) = dict_results
    (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df, all_sessions_merged_complete_epoch_stats_df, *final_sessions_loaded_extra_dfs_list) = df_results

    # Old (pre 2024-10-22) method ________________________________________________________________________________________ #
    ## Build across_sessions join dataframes:
    # Add 'epoch_idx' column for compatibility:
    if all_sessions_laps_df is not None:
        all_sessions_laps_df['epoch_idx'] = all_sessions_laps_df['lap_idx']
    if all_sessions_ripple_df is not None:
        all_sessions_ripple_df['epoch_idx'] = all_sessions_ripple_df['ripple_idx']


    # dfs_list = (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df)
    for a_df in df_results:
        if a_df is not None:
            if 'time_bin_size' not in a_df:
                print('Uh-oh! time_bin_size is missing! This must be old exports!')
                print(f'\tTry to determine the time_bin_size from the filenames: {csv_sessions}')
                raise DeprecationWarning('Uh-oh! time_bin_size is missing! This must be old exports!')
                raise NotImplementedError('NO LOONGER DOING THIS')
                ## manual correction UwU
                # time_bin_size: float = 0.025
                # print(f'WARNING! MANUAL OVERRIDE TIME BIN SIZE SET: time_bin_size = {time_bin_size}. Assigning to dataframes....')
                # a_df['time_bin_size'] = time_bin_size
            else:
                # Filter rows based on column: 'time_bin_size'
                a_df = a_df[a_df['time_bin_size'].notna()]
    
    df_results = [_common_cleanup_operations(a_df) for a_df in df_results]
    ## Unpack again:
    (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df, all_sessions_merged_complete_epoch_stats_df, *final_sessions_loaded_extra_dfs_list) = df_results

    all_sessions_MultiMeasure_laps_df: pd.DataFrame = DecoderDecodedEpochsResult.merge_decoded_epochs_result_dfs(all_sessions_simple_pearson_laps_df, all_sessions_wcorr_laps_df, should_drop_directional_columns=False, start_t_idx_name='delta_aligned_start_t')
    all_sessions_MultiMeasure_ripple_df: pd.DataFrame = DecoderDecodedEpochsResult.merge_decoded_epochs_result_dfs(all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_ripple_df, should_drop_directional_columns=False, start_t_idx_name='ripple_start_t')

    # all_sessions_laps_time_bin_df # 601845 rows × 9 column
    ## Re-pack
    # df_results = (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df)
    df_results = (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_MultiMeasure_laps_df, all_sessions_MultiMeasure_ripple_df, all_sessions_all_scores_ripple_df, all_sessions_merged_complete_epoch_stats_df, *final_sessions_loaded_extra_dfs_list)
    # t_delta_df, t_delta_dict, (earliest_delta_aligned_t_start, latest_delta_aligned_t_end) = sessions_t_delta_tuple ## UNPACK
    ## OUTPUTS: final_sessions: Dict[types.session_str, Dict[str, Path]], all_sessions_all_scores_ripple_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df
    # return final_sessions, (all_sessions_all_scores_ripple_df, all_sessions_MultiMeasure_laps_df, all_sessions_MultiMeasure_ripple_df), (csv_files, csv_sessions), (h5_files, h5_sessions)
    return final_sessions, sessions_t_delta_tuple, df_results, (most_recent_parsed_csv_files_df, all_parsed_csv_files_df, csv_files, csv_sessions), (most_recent_parsed_h5_files_df, all_parsed_h5_files_df, h5_files, h5_sessions), excluded_or_outdated_files_list



def try_convert_to_float(a_str: str, default_val: float = np.nan) -> float:
    """ allows trying to parse to a float, and fallsback to a default value if it fails.
    """
    try:
        return float(a_str)
    except:
        return default_val
    
## INPUTS: an_active_df, all_sessions_all_scores_df, a_time_column_names = 'ripple_start_t'
@function_attributes(short_name=None, tags=['IMPORTANT', 'missing-columns'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-23 18:10', related_items=[])
def recover_user_annotation_and_is_valid_columns(an_active_df, all_sessions_all_scores_df, a_time_column_names:str='ripple_start_t'):
    """ Gets the proper 'is_user_annotated_epoch' and 'is_valid_epoch' columns for the epochs passed in 'an_active_df' evaluated with different time_bin_sizes
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import recover_user_annotation_and_is_valid_columns
        
        ## epoch-based ones:
        all_sessions_ripple_df = recover_user_annotation_and_is_valid_columns(all_sessions_ripple_df, all_sessions_all_scores_df=all_sessions_all_scores_df, a_time_column_names='ripple_start_t')
        all_sessions_ripple_df

        # ## can't do the time_bin ones yet because it doesn't have 'ripple_start_t' to match on:
        # an_active_df = all_sessions_ripple_time_bin_df
        # a_time_column_names = 'delta_aligned_start_t'
        # all_sessions_all_scores_df = all_sessions_all_scores_ripple_df
    """
    from neuropy.utils.misc import numpyify_array
    from neuropy.core.user_annotations import UserAnnotationsManager
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

    # ## METHOD 0:
    # annotations_man = UserAnnotationsManager()
    # user_annotations = annotations_man.get_user_annotations()

    # # [k.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys()) for k, v in user_annotations.items()]

    # ## recover/build the annotation contexts to find the annotations:
    # recovered_session_contexts = [IdentifyingContext(**dict(zip(IdentifyingContext._get_session_context_keys(), k.split('_', maxsplit=3)))) for k in an_active_df.session_name.unique()]

    # epochs_name = 'ripple'

    # _out_any_good_selected_epoch_times = []

    # for a_ctxt in recovered_session_contexts:    
    #     loaded_selections_context_dict = {a_name:a_ctxt.adding_context_if_missing(display_fn_name='DecodedEpochSlices', epochs=epochs_name, decoder=a_name, user_annotation='selections') for a_name in ('long_LR','long_RL','short_LR','short_RL')}
    #     decoder_user_selected_epoch_times_dict = {a_name:numpyify_array(user_annotations.get(a_selections_ctx, [])) for a_name, a_selections_ctx in loaded_selections_context_dict.items()}
    #     _out_any_good_selected_epoch_times.extend(decoder_user_selected_epoch_times_dict.values())

    # # Find epochs that are present in any of the decoders:
    # concatenated_selected_epoch_times = np.concatenate([v for v in _out_any_good_selected_epoch_times if (np.size(v) > 0)], axis=0)
    # any_good_selected_epoch_times: NDArray = np.unique(concatenated_selected_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
    # print(f'METHOD 0: any_good_selected_epoch_times: {np.shape(any_good_selected_epoch_times)}')

    # `is_user_annotated` ________________________________________________________________________________________________ #
    # did_update_user_annotation_col = DecoderDecodedEpochsResult.try_add_is_user_annotated_epoch_column(an_active_df, any_good_selected_epoch_times=any_good_selected_epoch_times, t_column_names=[a_time_column_names,])
    failed_to_find_columns = True
    
    if all_sessions_all_scores_df is not None:
        ## Option 2 - the `all_sessions_all_scores_ripple_df` df column-based approach. get only the valid rows from the `all_sessions_all_scores_ripple_df` df
        
        try:
            any_good_selected_epoch_times: NDArray = all_sessions_all_scores_df[all_sessions_all_scores_df['is_user_annotated_epoch']][['start', 'stop']].to_numpy()
            any_good_selected_epoch_times = np.unique(any_good_selected_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
            # print(f'METHOD 1: any_good_selected_epoch_times: {np.shape(any_good_selected_epoch_times)}') # interesting: difference of 1: (436, 2) v. (435, 2) 

            did_update_user_annotation_col = DecoderDecodedEpochsResult.try_add_is_user_annotated_epoch_column(an_active_df, any_good_selected_epoch_times=any_good_selected_epoch_times, t_column_names=[a_time_column_names,])

            # `is_valid_epoch` ___________________________________________________________________________________________________ #
            # get only the valid rows from the `all_sessions_all_scores_ripple_df` df
            any_good_is_valid_epoch_times: NDArray = all_sessions_all_scores_df[all_sessions_all_scores_df['is_valid_epoch']][['start', 'stop']].to_numpy()
            any_good_is_valid_epoch_times = np.unique(any_good_is_valid_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
            did_update_is_valid = DecoderDecodedEpochsResult.try_add_is_valid_epoch_column(an_active_df, any_good_selected_epoch_times=any_good_is_valid_epoch_times, t_column_names=[a_time_column_names,])
            
        except BaseException as err:
            print(f"failed to find proper 'is_user_annotated_epoch' and 'is_valid_epoch' columns for the epochs passed with error: {err}. Skipping.")
            failed_to_find_columns = True
            # raise err

    else:
        failed_to_find_columns = True

    if failed_to_find_columns:
        print(f'WARNING: no `all_sessions_all_scores_df` to get "is_valid_epoch" and "is_user_annotated_epoch" from!\n\tSetting "is_valid_epoch" all to True and "is_user_annotated_Epoch" all to False.')
        an_active_df['is_valid_epoch'] = True # all True
        an_active_df['is_user_annotated_epoch'] = False # all False

    ## OUTPUTS: an_active_df
    return an_active_df


@function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=['_new_process_csv_files'], creation_date='2024-11-05 05:13', related_items=[])
def _concat_all_dicts_to_dfs(final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict,
                              final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict, final_sessions_loaded_merged_complete_epoch_stats_df_dict):
    """ 
    
    all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df = _concat_all_dicts_to_dfs(final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict)
    all_sessions_ripple_df

    """
    ## Build across_sessions join dataframes:
    all_sessions_laps_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_laps_dict.values()), axis='index', ignore_index=True)
    all_sessions_ripple_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_ripple_dict.values()), axis='index', ignore_index=True)
    # Add 'epoch_idx' column for compatibility:
    if all_sessions_laps_df is not None:
        all_sessions_laps_df['epoch_idx'] = all_sessions_laps_df['lap_idx']
    if all_sessions_ripple_df is not None:
        all_sessions_ripple_df['epoch_idx'] = all_sessions_ripple_df['ripple_idx']

    # *_time_bin marginals:
    all_sessions_laps_time_bin_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_laps_time_bin_dict.values()), axis='index', ignore_index=True)
    all_sessions_ripple_time_bin_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_ripple_time_bin_dict.values()), axis='index', ignore_index=True)

    # NEW ________________________________________________________________________________________________________________ #
    all_sessions_simple_pearson_laps_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_simple_pearson_laps_dict.values()), axis='index', ignore_index=True)
    all_sessions_simple_pearson_ripple_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_simple_pearson_ripple_dict.values()), axis='index', ignore_index=True)

    if len(final_sessions_loaded_laps_wcorr_dict) > 0:
        all_sessions_wcorr_laps_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_laps_wcorr_dict.values()), axis='index', ignore_index=True)
    else:
        all_sessions_wcorr_laps_df = None # empty df would be better

    all_sessions_wcorr_ripple_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_ripple_wcorr_dict.values()), axis='index', ignore_index=True)

    # `*_all_scores_*`: __________________________________________________________________________________________________ #
    # test_df_list = list(final_sessions_loaded_ripple_all_scores_dict.values())
    # [np.shape(df) for df in test_df_list]
    # {k:np.shape(df) for k, df in final_sessions_loaded_ripple_all_scores_dict.items()}
    # all_sessions_all_score_laps_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_laps_all_scores_dict.values()), axis='index', ignore_index=True)
    all_sessions_all_scores_ripple_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_ripple_all_scores_dict.values()), axis='index', ignore_index=True)

    all_sessions_merged_complete_epoch_stats_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_merged_complete_epoch_stats_df_dict.values()), axis='index', ignore_index=True)

    # dfs_list = (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df)
    dfs_dict = dict(zip(('all_sessions_laps_df', 'all_sessions_ripple_df', 'all_sessions_laps_time_bin_df', 'all_sessions_ripple_time_bin_df'), (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df)))

    # for a_df in dfs_list:
    for a_df_name, a_df in dfs_dict.items():
        if a_df is not None:
            if 'time_bin_size' not in a_df:
                print(f'Uh-oh! time_bin_size is missing for "{a_df_name}"! This must be old exports!')
                # print(f'\tTry to determine the time_bin_size from the filenames: {csv_sessions}')
                ## manual correction UwU
                time_bin_size: float = 0.025
                print(f'\tWARNING! MANUAL OVERRIDE TIME BIN SIZE SET: time_bin_size = {time_bin_size}. Assigning to dataframes....')
                a_df['time_bin_size'] = time_bin_size
            else:
                # Filter rows based on column: 'time_bin_size'
                a_df = a_df[a_df['time_bin_size'].notna()]

    all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df = [_common_cleanup_operations(a_df) for a_df in (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df)]
    all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df = [_common_cleanup_operations(a_df) for a_df in (all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df)]

    # all_sessions_all_score_laps_df, all_sessions_all_scores_ripple_df = [_common_cleanup_operations(a_df) for a_df in (all_sessions_all_score_laps_df, all_sessions_all_scores_ripple_df)]
    # all_sessions_all_score_laps_df = _common_cleanup_operations(all_sessions_all_score_laps_df)
    all_sessions_all_scores_ripple_df = _common_cleanup_operations(all_sessions_all_scores_ripple_df)
    all_sessions_merged_complete_epoch_stats_df = _common_cleanup_operations(all_sessions_merged_complete_epoch_stats_df)

    all_sessions_simple_pearson_laps_df: pd.DataFrame = DecoderDecodedEpochsResult.merge_decoded_epochs_result_dfs(all_sessions_simple_pearson_laps_df, all_sessions_wcorr_laps_df, should_drop_directional_columns=False, start_t_idx_name='delta_aligned_start_t')
    all_sessions_simple_pearson_ripple_df: pd.DataFrame = DecoderDecodedEpochsResult.merge_decoded_epochs_result_dfs(all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_ripple_df, should_drop_directional_columns=False, start_t_idx_name='ripple_start_t')
    # all_sessions_laps_time_bin_df # 601845 rows × 9 column

    ## epoch-based ones:
    if all_sessions_ripple_df is not None:
        all_sessions_ripple_df = recover_user_annotation_and_is_valid_columns(all_sessions_ripple_df, all_sessions_all_scores_df=all_sessions_all_scores_ripple_df, a_time_column_names='ripple_start_t')

    return (
        all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df, all_sessions_merged_complete_epoch_stats_df
    )


def _concat_custom_dict_to_df(final_sessions_loaded_df_dict):
    """ 
    
    all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df = _concat_custom_dict_to_df(final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict)
    all_sessions_ripple_df

    """
    all_sessions_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_df_dict.values()), axis='index', ignore_index=True)

    # all_sessions_all_score_laps_df, all_sessions_all_scores_ripple_df = [_common_cleanup_operations(a_df) for a_df in (all_sessions_all_score_laps_df, all_sessions_all_scores_ripple_df)]
    # all_sessions_all_score_laps_df = _common_cleanup_operations(all_sessions_all_score_laps_df)
    all_sessions_df = _common_cleanup_operations(all_sessions_df)

    return (
        all_sessions_df
    )


def _subfn_new_df_process_and_load_exported_file(file_path, loaded_dict: Dict, session_name: str, curr_session_t_delta: float, time_key: str, debug_print:bool=False, **additional_columns) -> bool:
    try:
        # loaded_dict[session_name] = read_and_process_csv_file(file_path, session_name, curr_session_t_delta, time_key)
        df = pd.read_csv(file_path, na_values=['', 'nan', 'np.nan', '<NA>'], low_memory=False) # `low_memory=False` tells pandas to use more memory to correctly infer data types.
        df['session_name'] = session_name
        if curr_session_t_delta is not None:
            df['delta_aligned_start_t'] = df[time_key] - curr_session_t_delta

        # loaded_dict_key = session_name # old way, session only
        loaded_dict_key = [session_name]
        for k, v in additional_columns.items():
            df[k] = v
            loaded_dict_key.append(str(v))

        loaded_dict_key = tuple(loaded_dict_key)

        ## update dict:
        loaded_dict[loaded_dict_key] = df ## it's being overwritten here
        return True

    except Exception as e:
        if debug_print:
            print(f'session "{session_name}", file_path: "{file_path}" - did not fully work. (error "{e}". Skipping.')
        return False



@function_attributes(short_name=None, tags=['csv'], input_requires=[], output_provides=[], uses=['_new_df_process_and_load_exported_file', 'recover_user_annotation_and_is_valid_columns', '_concat_all_dicts_to_dfs', 'load_and_apply_session_experience_rank_csv'], used_by=[], creation_date='2024-07-11 17:11', related_items=[])
def _new_process_csv_files(parsed_csv_files_df: pd.DataFrame, t_delta_dict: Dict, cuttoff_date=None, known_bad_session_strs=[], all_session_experiment_experience_csv_path:Path=None, debug_print=False):
    """  NEW `parsed_csv_files_df1-based approach 2024-07-11 
    # We now have the correct and parsed filepaths for each of the exported .csvs, now we need to actually load them and concatenate them toggether across sessions.
    # Extract each of the separate files from the sessions:

    
    DOCUMENTATION ON NaNs
    `*(ripple_marginals_df)_tbin-0.025.csv` - has the time_bin_size in the filename, but no 'time_bin_size' column 
    Conclusion, only look at the non-basic exported .CSVs, NOT `basic_marginals_file_types = ['laps_marginals_df', 'ripple_marginals_df', 'laps_time_bin_marginals_df', 'ripple_time_bin_marginals_df',]`
        - the basic ones don't have a `time_bin_size` column, and only seem to be exported for tbin=0.025 based on the filename. Strangely these basics values don't even seem to be computed for other time bins.
        - 
        
        - # strangely ripple_simple_pf_pearson_merged_df and ripple_all_scores_merged_df for the same session at the same time_bin_size can have different numbers of rows... I for sure don't get this. I'm guessing it has to do with NaNs or something?
        
        
    Usage:  
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import _new_process_csv_files
        ## NEW `parsed_csv_files_df1-based approach 2024-07-11 - 
        ## INPUTS: parsed_csv_files_df
        dict_results, df_results = _new_process_csv_files(parsed_csv_files_df=parsed_csv_files_df, t_delta_dict=t_delta_dict, cuttoff_date=cuttoff_date, known_bad_session_strs=known_bad_session_strs, debug_print=False) # , known_bad_session_strs=known_bad_session_strs
        (final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict) = dict_results
        (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df) = df_results

    """
    # final_sessions: Dict[types.session_str, Dict[str, Path]] = {}
    final_sessions_loaded_laps_dict = {}
    final_sessions_loaded_ripple_dict = {}
    final_sessions_loaded_laps_time_bin_dict = {}
    final_sessions_loaded_ripple_time_bin_dict = {}

    final_sessions_loaded_simple_pearson_laps_dict = {}
    final_sessions_loaded_simple_pearson_ripple_dict = {}

    # stupid reudndant but compatible method
    final_sessions_loaded_laps_wcorr_dict = {}
    final_sessions_loaded_ripple_wcorr_dict = {}

    final_sessions_loaded_laps_all_scores_dict = {}
    final_sessions_loaded_ripple_all_scores_dict = {}
    final_sessions_loaded_merged_complete_epoch_stats_df_dict = {}
    

    final_sessions_loaded_extra_df_dict: Dict[str, Dict] = {'ripple_WCorrShuffle_df': {},
                                           }
    final_sessions_loaded_extra_dfs: Dict[str, pd.DataFrame] = {'ripple_WCorrShuffle_df': None,
                                      }
    
    basic_marginals_file_types = ['laps_marginals_df', 'ripple_marginals_df', 'laps_time_bin_marginals_df', 'ripple_time_bin_marginals_df',]
    extended_file_types_list = ['laps_simple_pf_pearson_merged_df', 'ripple_simple_pf_pearson_merged_df', 'laps_weighted_corr_merged_df', 'ripple_weighted_corr_merged_df', 'laps_all_scores_merged_df', 'ripple_all_scores_merged_df', 'merged_complete_epoch_stats_df']

    sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv("./data/sessions_experiment_datetime_df.csv")
    
    # Sort by columns: 'session' (ascending), 'custom_replay_name' (ascending) and 3 other columns
    parsed_csv_files_df = parsed_csv_files_df.sort_values(['session', 'file_type', 'custom_replay_name', 'decoding_time_bin_size_str', 'export_datetime'], ascending=[True, True, True, True, False]).reset_index(drop=True) # ensures all are sorted ascending except for export_datetime, which are sorted decending so the first value is the most recent.

    excluded_or_outdated_files_list = []
    for index, row in parsed_csv_files_df.iterrows():
        session_str = str(row['session'])
        if session_str in known_bad_session_strs:
            if debug_print:
                print(f'Skipping "{session_str}" because it is in known_bad_session_strs.')
            excluded_or_outdated_files_list.append(row['path']) ## mark file for exclusion/removal
            continue
        
        custom_replay_name = row['custom_replay_name']
        file_type = row['file_type']
        decoding_time_bin_size_str = row['decoding_time_bin_size_str'] #TODO 2024-09-25 09:49: - [ ] Looks like `row['decoding_time_bin_size_str']` must be not set for the basic dataframes
    
        ## Setup:
        common_additional_columns_dict = {'custom_replay_name': custom_replay_name} # skip ['time_bin_size'] override for `common_additional_columns_dict`, so we can use it for 'laps_marginals_df', 'laps_time_bin_marginals_df', ... which already have good time_bin_size columns
        additional_columns_dict = deepcopy(common_additional_columns_dict)
        additional_columns_dict['time_bin_size'] = try_convert_to_float(decoding_time_bin_size_str, default_val=np.nan) #TODO 2024-10-24 18:53: - [ ] NaN time_bin_sizes can get introduced here, if the string can't be parsed into a float.

        export_datetime = row['export_datetime']
        if cuttoff_date is not None:
            if export_datetime < cuttoff_date:
                if debug_print:
                    print(f'Skipping "{session_str}" because export_datetime = "{export_datetime}" is less than cuttoff_date = "{cuttoff_date}".')
                excluded_or_outdated_files_list.append(row['path']) ## mark file for exclusion/removal
                continue
            
        path = Path(row['path']).resolve()
        assert path.exists()

        ## load the CSV
        session_name = str(session_str)  # Extract session name from the filename
        if debug_print:
            print(f'processing session_name: {session_name}')
        curr_session_t_delta = t_delta_dict.get(session_name, {}).get('t_delta', None)
        if curr_session_t_delta is None:
            if debug_print:
                print(f'WARN: curr_session_t_delta is None for session_str = "{session_str}"') # fails for 'kdiba_gor01_one_2006-6-09_1-22-43_None'

        
        
        # basic_marginals_file_types_dicts_list = [final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict]
        #TODO 2024-09-27 09:43: - [ ] the 4 basic marginals return two tuples ('session_id_str', 'custom_replay_name') while all the others return 3-tuples ('session_id_str', 'custom_replay_name', time_bin_size)
        ## Basic marginals: final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict
        _is_file_valid = True # shouldn't mark unknown files as invalid
        # extended_file_types_dicts_list = [final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict, final_sessions_loaded_merged_complete_epoch_stats_df_dict]

        if file_type in basic_marginals_file_types:
            # # Invalid files will have the form:
            # ## 2024-10-24 19:42 - actually ANY tbin value is bad for these basic files.
            # known_bad_file_patterns = ["(ripple_marginals_df)_tbin-0.025.csv",
            #     "(ripple_time_bin_marginals_df)_tbin-0.025.csv",
            #     "(laps_time_bin_marginals_df)_tbin-0.25.csv",
            #     "(laps_marginals_df)_tbin-0.25.csv",
            # ]            
            file_name: str = Path(row['path']).stem
            if file_name.find('_tbin-') != -1:
                # found an invalid export from the old  `compute_and_export_marginals_dfs_completion_function`
                _is_file_valid = False
            
        if _is_file_valid:
            # Process each file type with its corresponding details
            if file_type == 'laps_marginals_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_laps_dict, session_name, curr_session_t_delta, time_key='lap_start_t', **common_additional_columns_dict)
            elif file_type == 'ripple_marginals_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_ripple_dict, session_name, curr_session_t_delta, time_key='ripple_start_t', **common_additional_columns_dict)
            elif file_type == 'laps_time_bin_marginals_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_laps_time_bin_dict, session_name, curr_session_t_delta, time_key='t_bin_center', **common_additional_columns_dict)
            elif file_type == 'ripple_time_bin_marginals_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_ripple_time_bin_dict, session_name, curr_session_t_delta, time_key='t_bin_center', **common_additional_columns_dict)
            elif file_type == 'laps_simple_pf_pearson_merged_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_simple_pearson_laps_dict, session_name, curr_session_t_delta, time_key='lap_start_t', **additional_columns_dict)
            elif file_type == 'ripple_simple_pf_pearson_merged_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_simple_pearson_ripple_dict, session_name, curr_session_t_delta, time_key='ripple_start_t', **additional_columns_dict)
            elif file_type == 'laps_weighted_corr_merged_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_laps_wcorr_dict, session_name, curr_session_t_delta, time_key='lap_start_t', **additional_columns_dict)
            elif file_type == 'ripple_weighted_corr_merged_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_ripple_wcorr_dict, session_name, curr_session_t_delta, time_key='ripple_start_t', **additional_columns_dict)
            elif file_type == 'laps_all_scores_merged_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_laps_all_scores_dict, session_name, curr_session_t_delta, time_key='lap_start_t', **additional_columns_dict)
            elif file_type == 'ripple_all_scores_merged_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_ripple_all_scores_dict, session_name, curr_session_t_delta, time_key='ripple_start_t', **additional_columns_dict)
            elif file_type == 'merged_complete_epoch_stats_df':
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_merged_complete_epoch_stats_df_dict, session_name, curr_session_t_delta, time_key='start', **additional_columns_dict)				
            elif file_type in extended_file_types_list:
                ## main extended check, tests against all the known filetypes in `extended_file_types_list` so we don't have to add a new `elif` case for each new filetype`
                _curr_dict = final_sessions_loaded_extra_df_dict[file_type]
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, _curr_dict, session_name, curr_session_t_delta, time_key='start', **additional_columns_dict)
                ## update when done:
                final_sessions_loaded_extra_df_dict[file_type] = _curr_dict      
            elif file_type == 'ripple_WCorrShuffle_df':
                _curr_dict = final_sessions_loaded_extra_df_dict[file_type]
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, _curr_dict, session_name, curr_session_t_delta, time_key='start', **additional_columns_dict)
                ## update when done:
                final_sessions_loaded_extra_df_dict[file_type] = _curr_dict
            elif file_type == 'FAT':
                # 2025-03-01 - a bulk export .CSV format where all results are stacked vertically and fields duplicated as needed (no attentioned paid to disk
                # #TODO 2025-04-04 10:11: - [ ] FAT is a legit exception and shouldn't belong in `extended_file_types_list`, as it has both time bin and epoch centered items
                _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_laps_time_bin_dict, session_name, curr_session_t_delta, time_key='t_bin_center', **common_additional_columns_dict) ## adds to the time bins 
                #TODO 2025-04-04 10:12: - [ ] Why does it add to `final_sessions_loaded_laps_time_bin_dict`? this seems wrong
                



            else:
                print(f'WARN: File type "{file_type}" for filename "{path.name}" not implemented.')
                # _is_file_valid = False # shouldn't mark unknown files as invalid
                # File type neuron_replay_stats_df not implemented.
                # File type laps_marginals_df not implemented.
                # continue


        if (not _is_file_valid):
            excluded_or_outdated_files_list.append(row['path']) ## mark file for exclusion/removal
        else:
            if file_type in basic_marginals_file_types:
                if debug_print:
                    print(f'file_type: {file_type} -- "{path.name}" is valid and has been loaded.')
            # np.any([v[0] for k, v in final_sessions_loaded_laps_dict.items()])
            
        ## END FOR    
    if debug_print:
        print(f'done with main processing loop')
    ## OUTPUTS: final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict,
    # #TODO 2024-09-27 09:48: - [ ] ERROR: the basic dataframes are now missing their 'time_bin_size' columns somehow!!
    ## Build across_sessions join dataframes:
    all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df, all_sessions_merged_complete_epoch_stats_df = _concat_all_dicts_to_dfs(final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict,
                                                                                                                                                                                                                                                                                                                                                                   final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict,
                                                                                                                                                                                                                                                                                                                                                                   final_sessions_loaded_merged_complete_epoch_stats_df_dict)
    

    # ==================================================================================================================== #
    # Update Metadata in each dataframe (`*df.attrs.*`)                                                                    #
    # ==================================================================================================================== #

    ## Set the metadata for the default-built dataframes as well:
    basic_marginals_file_types_dfs_list = [all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df]
    for file_type, v in zip(basic_marginals_file_types, basic_marginals_file_types_dfs_list):
        if v is not None:
            ## udpate the metadata:
            v.attrs.update(**dict(file_type=deepcopy(file_type)))
            

    extended_file_types_dfs_list = [all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, None, all_sessions_all_scores_ripple_df, all_sessions_merged_complete_epoch_stats_df]
    for file_type, v in zip(extended_file_types_list, extended_file_types_dfs_list):
        if v is not None:
            ## udpate the metadata:
            v.attrs.update(**dict(file_type=deepcopy(file_type)))


    final_sessions_loaded_extra_dfs = {}
    for file_type, v in final_sessions_loaded_extra_df_dict.items():
        # final_sessions_loaded_extra_dfs[k] = _concat_all_dicts_to_dfs(v)
        final_sessions_loaded_extra_dfs[file_type] = _concat_custom_dict_to_df(v)
        ## udpate the metadata:
        if final_sessions_loaded_extra_dfs[file_type] is not None:
            final_sessions_loaded_extra_dfs[file_type].attrs.update(**dict(file_type=deepcopy(file_type)))
        # final_sessions_loaded_extra_dfs[file_type].metadata = final_sessions_loaded_extra_dfs[file_type].get('metadata', {})
        # final_sessions_loaded_extra_dfs[file_type].metadata.update(**{'file_type': file_type, })
        

    final_sessions_loaded_extra_df_dicts_list = list(final_sessions_loaded_extra_df_dict.values())        
    final_sessions_loaded_extra_dfs_list = list(final_sessions_loaded_extra_dfs.values())
    
        
    ## #TODO 2024-09-25 09:42: - [X] FIXED: Unfortunately both `all_sessions_laps_*df` dataframes have all missing values for their ['time_bin_size'] column.
    ## - [X] FIXED: ALL of the 4 basic dataframes and their input dicts (for both laps and ripples) lack a valid time_bin_size, while all of the derived additional computation dfs have them.
     
    ## OUTPUTS: all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df
    try:
        if all_session_experiment_experience_csv_path is None:
            all_session_experiment_experience_csv_path = Path("./data/sessions_experiment_datetime_df.csv").resolve()
            Assert.path_exists(all_session_experiment_experience_csv_path)
        df_results = (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df, all_sessions_merged_complete_epoch_stats_df, *final_sessions_loaded_extra_dfs_list)
        sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv(all_session_experiment_experience_csv_path, session_uid_str_sep='_')
        # all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df = [_callback_add_df_columns(a_df, session_id_column_name='session_name') for a_df in (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df)]
        # df_results = [_callback_add_df_columns(a_df, session_id_column_name='session_name') for a_df in df_results if (a_df is not None)]
        df_results = [_callback_add_df_columns(a_df, session_id_column_name='session_name') if (a_df is not None) else None for a_df in df_results ]
        all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df, all_sessions_merged_complete_epoch_stats_df, *final_sessions_loaded_extra_dfs_list = df_results # unpack again
    except Exception as e:
        print(f'WARNING: failed to add the session_experience_rank CSV results to the dataframes. Failed with error: {e}. Skipping and continuing.') 
        # raise e
        
    return (
        (final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict, final_sessions_loaded_merged_complete_epoch_stats_df_dict, *final_sessions_loaded_extra_df_dicts_list),
        (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df, all_sessions_merged_complete_epoch_stats_df, *final_sessions_loaded_extra_dfs_list),
        excluded_or_outdated_files_list,
    )



@metadata_attributes(short_name=None, tags=['archive', 'filesystem', 'greatlakes', 'useful', 'delete'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
class OldFileArchiver:
    """ Cleans up and deletes old files and temporary pickles to recover disk space.
    
    """
    @function_attributes(short_name=None, tags=['archive', 'cleanup', 'filesystem'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-02 11:30', related_items=[])
    @classmethod
    def archive_old_files(cls, collected_outputs_directory: Path, excluded_or_outdated_files_list: List[Path], is_dry_run: bool=False):
        """ moves old files that didn't meet the inclusion criteria into an archive directory.
        
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import OldFileArchiver
        archive_folder = OldFileArchiver.archive_old_files(collected_outputs_directory=collected_outputs_directory, excluded_or_outdated_files_list=excluded_or_outdated_files_list, is_dry_run=True)
        """
        ## INPUTS: collected_outputs_directory, excluded_or_outdated_files_list
        
        assert collected_outputs_directory.exists(), f"collected_outputs_directory: {collected_outputs_directory} does not exist! Is the right computer's config commented out above?"
        # fullwidth_path_widget(scripts_output_path, file_name_label='Scripts Output Path:')
        print(f'collected_outputs_directory: "{collected_outputs_directory}"')

        # Create a 'figures' subfolder if it doesn't exist
        archive_folder: Path = collected_outputs_directory.joinpath('OLD').resolve()
        archive_folder.mkdir(parents=False, exist_ok=True)
        assert archive_folder.exists()
        print(f'\tarchive_folder: "{archive_folder}"')

        for a_file_path in excluded_or_outdated_files_list:
            ## move the file to the archive folder
            # a_file_path.stem
            print(f'copying "{a_file_path}" to "{archive_folder}"...')
            if (not is_dry_run):
                shutil.move(a_file_path, archive_folder)

        return archive_folder


    @function_attributes(short_name=None, tags=['filesystem', 'clean', 'delete', 'temporary'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-06 04:27', related_items=[])
    @classmethod
    def remove_backup_files_from_session_data_folders(cls, good_session_concrete_folders: List["ConcreteSessionFolder"],
                                                   always_delete_patterns=None, conditional_delete_patterns=None, conditional_delete_files_cutoff_date: Optional[datetime.datetime]=None, is_dryrun: bool = True, debug_print:bool=False):
        """ cleans up the temporary pickle files that are created during the session data loading process to recover space.
        
        cutoff_date: files older than this date will be removed regardless of whether they are temporary or not!!
            conditional_delete_files_cutoff_date = datetime(2023, 9, 1)  # Example cutoff date
        
        Usage:
            from pyphoplacecellanalysis.General.Batch.runBatch import ConcreteSessionFolder, BackupMethods
            from neuropy.core.user_annotations import UserAnnotationsManager

            
            included_session_contexts: List[IdentifyingContext] = UserAnnotationsManager.get_all_known_sessions()
            known_global_data_root_parent_paths = [Path('/Users/pho/data'), Path(r'/nfs/turbo/umms-kdiba/Data'), Path(r'/media/halechr/MAX/Data'), Path(r'W:/Data'), Path(r'/home/halechr/FastData'), Path(r'/Volumes/MoverNew/data')] # Path(r'/home/halechr/cloud/turbo/Data'), , Path(r'/nfs/turbo/umms-kdiba/Data'), Path(r'/home/halechr/turbo/Data'), 
            global_data_root_parent_path = find_first_extant_path(known_global_data_root_parent_paths)
            assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
            print(f'global_data_root_parent_path: {global_data_root_parent_path}')
            good_session_concrete_folders = ConcreteSessionFolder.build_concrete_session_folders(global_data_root_parent_path, included_session_contexts)
            deleted_file_list = OldFileArchiver.remove_backup_files_from_session_data_folders(good_session_concrete_folders=good_session_concrete_folders, is_dryrun=True)
            deleted_file_list

        """	
        # a_folder = good_session_concrete_folders[0]
        # a_folder.session_pickle
        # backup-20240911134307-loadedSessPickle.pkl.bak'
        deleted_file_list = []

        # Convert cutoff_date to timestamp for comparison, if provided
        if conditional_delete_files_cutoff_date is not None:
            if conditional_delete_files_cutoff_date.tzinfo is None:
                # Assume local timezone if none provided
                cutoff_timestamp = conditional_delete_files_cutoff_date.timestamp()
            else:
                # Convert to UTC timestamp
                cutoff_timestamp = conditional_delete_files_cutoff_date.astimezone(datetime.timezone.utc).timestamp()
        else:
            cutoff_timestamp = None
            

        for a_folder in good_session_concrete_folders:
            # Define patterns to always delete
            if always_delete_patterns is None:
                always_delete_patterns = {
                    '.pkltmp': '*.pkltmp',
                    '.pkl.bak': '*.pkl.bak',
                }
            
            # Define patterns to conditionally delete
            if conditional_delete_patterns is None:
                conditional_delete_patterns = {
                    # '.pkl': '*.pkl',
                    'pipeline.pkl': '*loadedSessPickle*.pkl',
                    'global.pkl': '*global_computation_results*.pkl',
                }
            
            # Function to collect files based on patterns and cutoff_date
            def collect_files(folder: Path, patterns: dict, condition: str = 'always') -> List[Path]:
                """ captures debug_print, cutoff_timestamp, and is_dryrun from the outer function """
                files = []
                for desc, pattern in patterns.items():
                    matched_files = list(folder.glob(pattern))
                    if debug_print:
                        print(f'\tLooking for {desc} files with pattern "{pattern}": Found {len(matched_files)} files.')
                    for file_path in matched_files:
                        if condition == 'conditional' and cutoff_timestamp is not None:
                            try:
                                file_mtime = file_path.stat().st_mtime
                            except OSError as e:
                                print(f'\t\tError accessing file "{file_path}": {e}')
                                continue
                            if file_mtime >= cutoff_timestamp:
                                if debug_print:
                                    print(f'\t\tSkipping "{file_path.as_posix()}" (modified on {datetime.fromtimestamp(file_mtime)}) - newer than cutoff.')
                                continue
                        elif condition == 'conditional' and cutoff_timestamp is None:
                            # If conditional pattern but no cutoff_date provided, skip deletion
                            if debug_print:
                                print(f'\t\tNo cutoff_date provided. Skipping "{file_path.as_posix()}".')
                            continue
                        files.append(file_path)
                return files

            # Collect files to always delete from session folder
            session_always_files = collect_files(a_folder.path, always_delete_patterns, condition='always')
            if debug_print:
                print(f'\tSession folder: {len(session_always_files)} files marked for always deletion.')

            # Collect files to conditionally delete from session folder
            session_conditional_files = collect_files(a_folder.path, conditional_delete_patterns, condition='conditional')
            if debug_print:
                print(f'\tSession folder: {len(session_conditional_files)} *.pkl files marked for conditional deletion.')

            # Collect files to always delete from output folder
            output_always_files = collect_files(a_folder.output_folder, always_delete_patterns, condition='always')
            if debug_print:
                print(f'\tOutput folder: {len(output_always_files)} files marked for always deletion.')

            # Collect files to conditionally delete from output folder
            output_conditional_files = collect_files(a_folder.output_folder, conditional_delete_patterns, condition='conditional')
            if debug_print:
                print(f'\tOutput folder: {len(output_conditional_files)} *.pkl files marked for conditional deletion.')

            # Combine all files to delete
            all_files_to_delete = session_always_files + session_conditional_files + output_always_files + output_conditional_files

            for a_path in all_files_to_delete:
                print(f'\tDeleting "{a_path.as_posix()}"...')
                if not is_dryrun:
                    try:
                        a_path.unlink()
                        print(f'\t\tDeleted "{a_path.as_posix()}".')
                    except OSError as e:
                        print(f'\t\tFailed to delete "{a_path.as_posix()}": {e}')
                        continue
                deleted_file_list.append(a_path)
                        
            print(f'\tDone processing folder "{a_folder.path.as_posix()}".')

        print(f'Total files {"would be " if is_dryrun else ""}deleted: {len(deleted_file_list)}')
        return deleted_file_list




# ==================================================================================================================== #
# Visualizations                                                                                                       #
# ==================================================================================================================== #

@metadata_attributes(short_name=None, tags=['across-session', 'visualizations', 'figure', 'output'], input_requires=[], output_provides=[], uses=['PaperFigureTwo'], used_by=[], creation_date='2023-07-21 00:00', related_items=[])
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
    @function_attributes(short_name=None, tags=['across-session', 'figure', 'matplotlib', 'figure-3'], input_requires=[], output_provides=[], uses=['_plot_long_short_firing_rate_indicies'], used_by=[], creation_date='2023-08-24 00:00', related_items=[])
    def across_sessions_firing_rate_index_figure(cls, long_short_fr_indicies_analysis_results: pd.DataFrame, num_sessions:int, save_figure=True, include_axes_lines:bool=True, **kwargs):
        """ 2023-08-24 - Across Sessions Aggregate Figure - Supposed to be the equivalent for Figure 3.

        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsVisualizations

            neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=True)
            matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
            graphics_output_dict = AcrossSessionsVisualizations.across_sessions_firing_rate_index_figure(long_short_fr_indicies_analysis_results=long_short_fr_indicies_analysis_table, num_sessions=num_sessions)


        """
        # _out2 = curr_active_pipeline.display('_display_long_and_short_firing_rate_replays_v_laps', curr_active_pipeline.get_session_context(), defer_render=defer_render, save_figure=save_figure)
        from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import _plot_long_short_firing_rate_indicies

        # Plot long|short firing rate index:
        x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
        x_frs_index = x_frs_index.set_axis(long_short_fr_indicies_analysis_results['neuron_uid']) # use neuron unique ID as index
        y_frs_index = y_frs_index.set_axis(long_short_fr_indicies_analysis_results['neuron_uid']) # use neuron unique ID as index


        # get number of points above vs. below the diagnonal
        is_above_diagonal = (y_frs_index > x_frs_index)
        num_above_diagonal = np.sum(is_above_diagonal)
        is_below_count = np.sum(y_frs_index < x_frs_index)
        total_num_points = np.shape(y_frs_index)[0]
        percent_below_diagonal = float(is_below_count) / float(total_num_points)
        percent_above_diagonal = float(num_above_diagonal) / float(total_num_points)
        
        print(f'num_above_diagonal/total_num_points: {num_above_diagonal}/{total_num_points}')
        print(f'percent_below_diagonal: {percent_below_diagonal * 100}%')
        print(f'percent_above_diagonal: {percent_above_diagonal * 100}%')
        

        # active_context = long_short_fr_indicies_analysis_results['active_context']
        global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.
        active_context = global_multi_session_context
        final_context = active_context.adding_context('display_fn', display_fn_name='across_sessions_firing_rate_index')
        final_context = DisplaySpecifyingIdentifyingContext.init_from_context(final_context, display_dict={})

        scatter_plot_kwargs = dict(zorder=5)
        scatter_plot_kwargs['point_colors'] = '#33333333'
        if 'has_pf_color' in long_short_fr_indicies_analysis_results:
            scatter_plot_kwargs['edgecolors'] = long_short_fr_indicies_analysis_results['has_pf_color'].to_numpy() #.to_list() # edgecolors=(r, g, b, 1)


        fig, ax, scatter_plot = _plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, final_context, debug_print=True, is_centered=False, enable_hover_labels=False, enable_tiny_point_labels=False, facecolor='w', include_axes_lines=include_axes_lines, **scatter_plot_kwargs) #  markeredgewidth=1.5,
        
        def _perform_write_to_file_callback():
            active_out_figure_path, *args_L = cls.output_figure(final_context, fig)
            return (active_out_figure_path,)

        if save_figure:
            active_out_figure_paths = _perform_write_to_file_callback()
        else:
            active_out_figure_paths = []

        graphics_output_dict = MatplotlibRenderPlots(name='across_sessions_firing_rate_index_figure', figures=(fig, ), axes=tuple(fig.axes), plot_data={'scatter_plot': scatter_plot}, context=final_context, saved_figures=active_out_figure_paths)
        # graphics_output_dict['plot_data'] = {'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}
        return graphics_output_dict

    @classmethod
    @function_attributes(short_name=None, tags=['across-session', 'figure', 'matplotlib', 'figure-3'], input_requires=[], output_provides=[], uses=['_plot_single_track_firing_rate_compare'], used_by=[], creation_date='2023-08-24 00:00', related_items=[])
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

        common_scatter_kwargs = dict(point_colors='#33333333')
        
        ## Long Track Replay|Laps FR Figure
        neuron_replay_stats_df = neuron_replay_stats_table.dropna(subset=['long_replay_mean', 'long_non_replay_mean'], inplace=False)
        x_frs = {k:v for k,v in neuron_replay_stats_df['long_non_replay_mean'].items()} 
        y_frs = {k:v for k,v in neuron_replay_stats_df['long_replay_mean'].items()}
        fig_L, ax_L, active_display_context_L = _plot_single_track_firing_rate_compare(x_frs, y_frs, active_context=final_context.adding_context_if_missing(filter_name='long'), **common_scatter_kwargs)

        ## Short Track Replay|Laps FR Figure
        neuron_replay_stats_df = neuron_replay_stats_table.dropna(subset=['short_replay_mean', 'short_non_replay_mean'], inplace=False)
        x_frs = {k:v for k,v in neuron_replay_stats_df['short_non_replay_mean'].items()} 
        y_frs = {k:v for k,v in neuron_replay_stats_df['short_replay_mean'].items()}
        fig_S, ax_S, active_display_context_S = _plot_single_track_firing_rate_compare(x_frs, y_frs, active_context=final_context.adding_context_if_missing(filter_name='short'), **common_scatter_kwargs)

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



class ExportValueNameCleaner:
    """ 
    Usage:
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import ExportValueNameCleaner
        new_name_list = ExportValueNameCleaner.clean_all(name_list=all_sessions_MultiMeasure_ripple_df['custom_replay_name'].unique().tolist())
        new_name_list

    """
    @staticmethod
    def reorder_ensuring_qclu_before_frateThresh(name: str) -> str:
        pattern = r'^(.*?)-?(frateThresh_[^-\s]+|qclu_\[[^\]]+\])-?(frateThresh_[^-\s]+|qclu_\[[^\]]+\])(.*)$'
        match = re.match(pattern, name)
        if match:
            prefix = match.group(1)
            comp1 = match.group(2)
            comp2 = match.group(3)
            suffix = match.group(4)
            # Determine which component is 'qclu_' and which is 'frateThresh_'
            if comp1.startswith('qclu_'):
                qclu_comp = comp1
                frate_comp = comp2
            else:
                qclu_comp = comp2
                frate_comp = comp1
            new_name = f"{prefix}-{qclu_comp}-{frate_comp}{suffix}"
        else:
            # Does not match expected pattern, return as is
            new_name = name
        return new_name


    @staticmethod
    def remove_redundant_suffix(name: str) -> str:
        """ 
        malformed: 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]' - > 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0normal_computed-frateThresh_1.0'

        """
        data_str: str = 'withNormalComputedReplays'
        # Split the string at '_withNormalComputedReplays'
        # parts = name.split('_withNormalComputedReplays')
        parts = name.split(data_str)
        # print(f'parts: {parts}')
        valid_parts = parts[:2]
        valid_str: str = data_str.join(valid_parts).rstrip('-_')
        # print(f'valid_str: {valid_str}')

        # # Collect all components from each part
        # components = []
        # for part in valid_parts:
        #     # Remove leading/trailing hyphens and split into components
        #     comps = part.strip('-').split('-')
        #     components.extend(comps)
        # # Remove duplicates while preserving order
        # seen = set()
        # unique_components = []
        # for comp in components:
        #     if comp not in seen:
        #         seen.add(comp)
        #         unique_components.append(comp)
        # # Reconstruct the string from unique components
        # new_name = '-'.join(unique_components)
        return valid_str
    

    @classmethod
    def clean_all(cls, name_list: List[str], return_only_unique:bool=False, debug_print=False):
        """ 
        name_list=all_sessions_MultiMeasure_ripple_df['custom_replay_name'].unique().tolist()
        """
        if debug_print:
            print(name_list)

        new_name_list = [cls.remove_redundant_suffix(replay_name) for replay_name in name_list]
        if debug_print:
            print(new_name_list)

        new_name_list = [cls.reorder_ensuring_qclu_before_frateThresh(replay_name) for replay_name in new_name_list]

        new_name_list = [v for v in new_name_list if (v.find('normal_computed') == -1)] # drop entries with 'normal_computed', which is a malformed string
        # ['', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0', 'withNormalComputedReplays-qclu_[1, 2]-frateThresh_5.0', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0']

        if return_only_unique:
            new_name_list = list(dict.fromkeys(new_name_list))

        if debug_print:
            print(new_name_list) # ['', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0', 'withNormalComputedReplays-qclu_[1, 2]-frateThresh_5.0', 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0']
        return new_name_list
    


class AcrossSessionHelpers:
    """ contains helper methods for loading and using AcrossSession outputs
    
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionHelpers
    
    
    """
    @function_attributes(short_name=None, tags=['merge', 'correlation'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-18 11:33', related_items=[])
    @classmethod
    def _subfn_perform_add_merged_complete_epoch_stats_df(cls, a_paired_main_ripple_df: pd.DataFrame, an_all_sessions_merged_complete_epoch_stats_df: pd.DataFrame,
                                                          comparison_must_match_column_names=None, comparison_must_match_non_temporal_column_names=None, desired_transfer_col_names=None,
                                                          ):
        """ adds the columns in `desired_transfer_col_names` to the dataframe `a_paired_main_ripple_df`

        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionHelpers
        an_all_sessions_merged_complete_epoch_stats_df = AcrossSessionHelpers._subfn_perform_add_merged_complete_epoch_stats_df(a_paired_main_ripple_df=deepcopy(all_sessions_all_scores_ripple_df), an_all_sessions_merged_complete_epoch_stats_df=deepcopy(all_sessions_merged_complete_epoch_stats_df))
        an_all_sessions_merged_complete_epoch_stats_df

        """
        # from pyphocorehelpers.DataStructure.data_structure_builders import cartesian_product
        from neuropy.utils.indexing_helpers import NumpyHelpers
        
        if comparison_must_match_column_names is None:
            comparison_must_match_column_names = ['custom_replay_name', 'session_name', 'start', 'stop']
            
        if comparison_must_match_non_temporal_column_names is None:
            comparison_must_match_non_temporal_column_names = ['custom_replay_name', 'session_name']
            
        if desired_transfer_col_names is None:
            desired_transfer_col_names = ['Long_BestDir_quantile', 'Short_BestDir_quantile', 'best_overall_quantile']
            

        # an_all_sessions_merged_complete_epoch_stats_df['best_overall_quantile']
        len(an_all_sessions_merged_complete_epoch_stats_df)
        
        if (len(desired_transfer_col_names)==3) and (desired_transfer_col_names[-1] not in an_all_sessions_merged_complete_epoch_stats_df.columns):
            # e.g. ['Long_BestDir_quantile', 'Short_BestDir_quantile', 'best_overall_quantile']
            derived_column_name: str = desired_transfer_col_names[-1] # derived_column_name -- e.g. 'best_overall_quantile'
            an_all_sessions_merged_complete_epoch_stats_df[derived_column_name] = np.nanmax(an_all_sessions_merged_complete_epoch_stats_df[desired_transfer_col_names[:-1]], axis=1)
            
        # if 'best_overall_quantile' not in an_all_sessions_merged_complete_epoch_stats_df.columns:
        #     an_all_sessions_merged_complete_epoch_stats_df['best_overall_quantile'] = np.nanmax(an_all_sessions_merged_complete_epoch_stats_df[['Long_BestDir_quantile', 'Short_BestDir_quantile']], axis=1)

        # assert (a_paired_main_ripple_df['ripple_start_t'] == a_paired_main_ripple_df['start']).all()
        # a_paired_main_ripple_df[['start', 'stop']]

        # # Shared columns for comparison
        # comparison_must_match_column_names = ['custom_replay_name', 'session_name', 'start', 'stop']
        # comparison_must_match_non_temporal_column_names = ['custom_replay_name', 'session_name']
        # desired_transfer_col_names = ['Long_BestDir_quantile', 'Short_BestDir_quantile', 'best_overall_quantile']
        
        target_df = deepcopy(a_paired_main_ripple_df) # 'df1'
        source_df = deepcopy(an_all_sessions_merged_complete_epoch_stats_df) # 'df2'

        # _preferred_form_session_names = deepcopy(df1['session_name'].unique()) ## the one with underscores like "kdiba_gor01_one_2006-6-08_14-26-15"
        target_df['session_name'] = target_df['session_name'].str.replace('_','-') # replace both with hyphens so they match
        source_df['session_name'] = source_df['session_name'].str.replace('_','-') # replace both with hyphens so they match

        ## Clean up!
        target_df['custom_replay_name'] = ExportValueNameCleaner.clean_all(name_list=target_df['custom_replay_name'].to_list(), return_only_unique=False)
        source_df['custom_replay_name'] = ExportValueNameCleaner.clean_all(name_list=source_df['custom_replay_name'].to_list(), return_only_unique=False)

        target_df['custom_replay_name'] = target_df['custom_replay_name'].str.replace('_','-') # replace both with hyphens so they match
        source_df['custom_replay_name'] = source_df['custom_replay_name'].str.replace('_','-') # replace both with hyphens so they match
        
        # df1['custom_replay_name'] = df1['custom_replay_name'].str.replace('withNormalComputedReplays-qclu-[1, 2, 4, 6, 7, 9]-frateThresh-1.0-withNormalComputedReplays-frateThresh-1.0-qclu-[1, 2, 4, 6, 7, 9]',
        #                                                                   'withNormalComputedReplays-qclu-[1, 2, 4, 6, 7, 9]-frateThresh-1.0') ## drop invalid duplicated versions
        

        # _compare_form_session_names = deepcopy(df1['session_name'].unique())
        # _compare_to_preferred_session_name_map = dict(zip(_compare_form_session_names, _preferred_form_session_names)) # used to replace desired session names when done

        df1_unique_values_dict = {k:deepcopy(target_df[k].unique().tolist()) for k in comparison_must_match_non_temporal_column_names}
        df2_unique_values_dict = {k:deepcopy(source_df[k].unique().tolist()) for k in comparison_must_match_non_temporal_column_names}

        combined_shared_unique_values_dict = {}
        for k in comparison_must_match_non_temporal_column_names:
            combined_shared_unique_values_dict[k] = tuple(set(df1_unique_values_dict[k]).intersection(df2_unique_values_dict[k]))
            

        # new_name_list = ExportValueNameCleaner.clean_all(name_list=all_sessions_MultiMeasure_ripple_df['custom_replay_name'].unique().tolist())
        # new_name_list
        # combined_shared_unique_value_tuples, param_sweep_option_n_values = parameter_sweeps(smooth=[(None, None), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)], grid_bin=[(1,1),(5,5),(10,10)])
        combined_shared_unique_value_dicts, param_sweep_option_n_values = parameter_sweeps(**combined_shared_unique_values_dict)

        for a_values_dict in combined_shared_unique_value_dicts:
            # a_values_dict
            df1_matches_all_conditions = NumpyHelpers.logical_generic(np.logical_and, *[target_df[k] == v for k, v in a_values_dict.items()])
            df2_matches_all_conditions = NumpyHelpers.logical_generic(np.logical_and, *[source_df[k] == v for k, v in a_values_dict.items()])
            active_df1 = target_df[df1_matches_all_conditions][comparison_must_match_column_names]
            active_df2 = source_df[df2_matches_all_conditions][comparison_must_match_column_names + desired_transfer_col_names]
            # (active_df1.shape, active_df2.shape)
            ## drop duplicates    
            # active_df1 = active_df1.drop_duplicates(ignore_index=True) # df1 should have duplicates for each time_bin_size
            active_df2 = active_df2.drop_duplicates(ignore_index=True) # df2 should have no duplicates. 
            active_df2 = active_df2.drop_duplicates(ignore_index=True, subset=['start', 'stop'])
            # (active_df1.shape, active_df2.shape)
            ## for each start, stop in `active_df2`, find matching values in active_df1
            for a_row in active_df2.itertuples(index=True):
                is_df1_row_matching = NumpyHelpers.logical_generic(np.logical_and, *[(active_df1['start'] == a_row.start), (active_df1['stop'] == a_row.stop)])
                _df1_row_matching_indicies = active_df1[is_df1_row_matching].index ## get the indicies
                ## perform the update inline:
                for an_update_col_name in desired_transfer_col_names:
                    target_df.loc[_df1_row_matching_indicies, an_update_col_name] = getattr(a_row, an_update_col_name) # a_row.Long_BestDir_quantile
                    a_paired_main_ripple_df.loc[_df1_row_matching_indicies, an_update_col_name] = getattr(a_row, an_update_col_name) # a_row.Long_BestDir_quantile

                # # matching_df1_update_tuples.append([_df1_row_matching_indicies, a_row.Long_BestDir_quantile, a_row.Short_BestDir_quantile])
                
            # unique_start_stop = active_df2[['start', 'stop']].unique()
            # unique_start_stop
            
        # a_paired_main_ripple_df = df1
        return a_paired_main_ripple_df
    

    @function_attributes(short_name=None, tags=['merge', 'time_bin', 'correlation'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-18 11:33', related_items=[])
    @classmethod
    def _subfn_perform_add_stats_to_time_bins_df(cls, a_paired_epoch_stats_df: pd.DataFrame, an_epoch_time_bin_df: pd.DataFrame,
                                                          comparison_must_match_column_names=None, comparison_must_match_non_temporal_column_names=None, desired_transfer_col_names=None,
                                                          ):
        """ adds the columns `desired_transfer_col_names` from `a_paired_epoch_stats_df` to the dataframe `an_epoch_time_bin_df`

        
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionHelpers

        an_epoch_time_bin_df = AcrossSessionHelpers._subfn_perform_add_stats_to_time_bins_df(a_paired_epoch_stats_df=deepcopy(all_sessions_MultiMeasure_ripple_df),
             an_epoch_time_bin_df=deepcopy(all_sessions_ripple_time_bin_df))
        an_epoch_time_bin_df
        
        # all_sessions_ripple_time_bin_df = an_epoch_time_bin_df ## assign

        """
        # from pyphocorehelpers.DataStructure.data_structure_builders import cartesian_product
        from neuropy.utils.indexing_helpers import NumpyHelpers
        
        if comparison_must_match_column_names is None:
            # epoch_idx_col_name: str = 'epoch_idx'
            # epoch_idx_col_name: str = 'label' # for 'all_sessions_all_scores_ripple_df', doesn't quite work
            epoch_idx_col_name: str = 'ripple_idx' # for 'all_sessions_MultiMeasure_ripple_df',
            comparison_must_match_column_names = ['session_name', 'custom_replay_name', 'time_bin_size', epoch_idx_col_name]
            
        if comparison_must_match_non_temporal_column_names is None:
            comparison_must_match_non_temporal_column_names = comparison_must_match_column_names
            
        if desired_transfer_col_names is None:
            desired_transfer_col_names = ['Long_BestDir_quantile', 'Short_BestDir_quantile', 'best_overall_quantile', 'wcorr_z_long', 'wcorr_z_short', 'best_overall_wcorr_z']
            
        ## add the corresponding column to `an_epoch_time_bin_df` if it is missing
        epoch_idx_match_col_name: str = comparison_must_match_column_names[-1]
        if (epoch_idx_match_col_name not in an_epoch_time_bin_df) and ('parent_epoch_label' in an_epoch_time_bin_df):
            ## make the needed epoch index column in time_bin
            an_epoch_time_bin_df[epoch_idx_match_col_name] = an_epoch_time_bin_df['parent_epoch_label']
        
        if (len(desired_transfer_col_names)==3) and (desired_transfer_col_names[-1] not in an_epoch_time_bin_df.columns):
            # e.g. ['Long_BestDir_quantile', 'Short_BestDir_quantile', 'best_overall_quantile']
            derived_column_name: str = desired_transfer_col_names[-1] # derived_column_name -- e.g. 'best_overall_quantile'
            an_epoch_time_bin_df[derived_column_name] = np.nanmax(an_epoch_time_bin_df[desired_transfer_col_names[:-1]], axis=1)

        source_df = deepcopy(a_paired_epoch_stats_df) ## "df1"
        target_df = deepcopy(an_epoch_time_bin_df)

        # _preferred_form_session_names = deepcopy(df1['session_name'].unique()) ## the one with underscores like "kdiba_gor01_one_2006-6-08_14-26-15"
        target_df['session_name'] = target_df['session_name'].str.replace('_','-') # replace both with hyphens so they match
        source_df['session_name'] = source_df['session_name'].str.replace('_','-') # replace both with hyphens so they match

        ## Clean up!
        target_df['custom_replay_name'] = ExportValueNameCleaner.clean_all(name_list=target_df['custom_replay_name'].to_list(), return_only_unique=False)
        source_df['custom_replay_name'] = ExportValueNameCleaner.clean_all(name_list=source_df['custom_replay_name'].to_list(), return_only_unique=False)

        target_df['custom_replay_name'] = target_df['custom_replay_name'].str.replace('_','-') # replace both with hyphens so they match
        source_df['custom_replay_name'] = source_df['custom_replay_name'].str.replace('_','-') # replace both with hyphens so they match
        
        # df1['custom_replay_name'] = df1['custom_replay_name'].str.replace('withNormalComputedReplays-qclu-[1, 2, 4, 6, 7, 9]-frateThresh-1.0-withNormalComputedReplays-frateThresh-1.0-qclu-[1, 2, 4, 6, 7, 9]',
        #                                                                   'withNormalComputedReplays-qclu-[1, 2, 4, 6, 7, 9]-frateThresh-1.0') ## drop invalid duplicated versions
        

        # _compare_form_session_names = deepcopy(df1['session_name'].unique())
        # _compare_to_preferred_session_name_map = dict(zip(_compare_form_session_names, _preferred_form_session_names)) # used to replace desired session names when done

        df1_unique_values_dict = {k:deepcopy(target_df[k].unique().tolist()) for k in comparison_must_match_non_temporal_column_names}
        df2_unique_values_dict = {k:deepcopy(source_df[k].unique().tolist()) for k in comparison_must_match_non_temporal_column_names}

        combined_shared_unique_values_dict = {}
        for k in comparison_must_match_non_temporal_column_names:
            combined_shared_unique_values_dict[k] = tuple(set(df1_unique_values_dict[k]).intersection(df2_unique_values_dict[k]))
            
        # new_name_list = ExportValueNameCleaner.clean_all(name_list=all_sessions_MultiMeasure_ripple_df['custom_replay_name'].unique().tolist())
        # new_name_list
        combined_shared_unique_value_dicts, param_sweep_option_n_values = parameter_sweeps(**combined_shared_unique_values_dict)

        # start_idx_name: str = 'start'
        start_idx_name: str = 'ripple_idx'
        
        for a_values_dict in combined_shared_unique_value_dicts:
            # a_values_dict
            target_df_matches_all_conditions = NumpyHelpers.logical_generic(np.logical_and, *[target_df[k] == v for k, v in a_values_dict.items()])
            source_df_matches_all_conditions = NumpyHelpers.logical_generic(np.logical_and, *[source_df[k] == v for k, v in a_values_dict.items()])
            active_target_df = target_df[target_df_matches_all_conditions][comparison_must_match_column_names]
            active_source_df = source_df[source_df_matches_all_conditions][comparison_must_match_column_names + desired_transfer_col_names]
            ## drop duplicates    
            # active_df1 = active_df1.drop_duplicates(ignore_index=True) # df1 should have duplicates for each time_bin_size
            active_source_df = active_source_df.drop_duplicates(ignore_index=True) # df2 should have no duplicates. 
            active_source_df = active_source_df.drop_duplicates(ignore_index=True, subset=[start_idx_name, ])
            ## for each start, stop in `active_df2`, find matching values in active_df1
            for a_row in active_source_df.itertuples(index=True):
                is_target_df_row_matching = NumpyHelpers.logical_generic(np.logical_and, *[(active_target_df[start_idx_name] == a_row.ripple_idx), ])
                _target_df_row_matching_indicies = active_target_df[is_target_df_row_matching].index ## get the indicies
                ## perform the update inline:
                for an_update_col_name in desired_transfer_col_names:
                    target_df.loc[_target_df_row_matching_indicies, an_update_col_name] = getattr(a_row, an_update_col_name) # a_row.Long_BestDir_quantile
                    an_epoch_time_bin_df.loc[_target_df_row_matching_indicies, an_update_col_name] = getattr(a_row, an_update_col_name) # a_row.Long_BestDir_quantile

                # # matching_df1_update_tuples.append([_df1_row_matching_indicies, a_row.Long_BestDir_quantile, a_row.Short_BestDir_quantile])
                

    
        return an_epoch_time_bin_df
    

    @function_attributes(short_name=None, tags=['filesystem', 'copy', 'file', 'session_folder'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-19 06:02', related_items=[])
    @classmethod
    def _copy_exported_files_from_session_folder_to_collected_outputs(cls, target_dir: Path, BATCH_DATE_TO_USE = '2024-11-19', cuttoff_date = datetime(2024, 11, 16), is_dry_run:bool=True, debug_print:bool=False, custom_file_globs_dict=None, find_most_recent_files_kwargs=None):
        """ Extracts the output files (.pkl, .h5, etc) from the individual session folders to the target folder

        
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionHelpers

        copy_dict, moved_files_dict_files = AcrossSessionHelpers._copy_exported_files_from_session_folder_to_collected_outputs(BATCH_DATE_TO_USE='2024-11-19', cuttoff_date=datetime(2024, 11, 16), target_dir=collected_outputs_directory, is_dry_run=False)
        copy_dict

        ## Specify which files to match:    
        copy_dict, moved_files_dict_files = AcrossSessionHelpers._copy_exported_files_from_session_folder_to_collected_outputs(BATCH_DATE_TO_USE='2024-11-19', cuttoff_date=datetime(2024, 11, 16), target_dir=collected_outputs_directory, custom_file_globs_dict={
           'pkl': '*.pkl',
           'csv': '*ripple_WCorrShuffle_df*.csv',
           'h5': '*.h5',
        }, is_dry_run=False)
        copy_dict

        """
        # from pyphoplacecellanalysis.General.Batch.runBatch import get_file_path_if_file_exists
        # from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import copy_session_folder_files_to_target_dir
        from pyphocorehelpers.Filesystem.path_helpers import copy_movedict
        from neuropy.core.user_annotations import UserAnnotationsManager
        from pyphoplacecellanalysis.General.Batch.runBatch import ConcreteSessionFolder, BackupMethods
        # from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import find_csv_files, find_HDF5_files, find_pkl_files
        
        included_session_contexts = UserAnnotationsManager.get_hardcoded_good_sessions()
        
        known_global_data_root_parent_paths = [Path('/Users/pho/data'), Path(r'/nfs/turbo/umms-kdiba/Data'), Path(r'/media/halechr/MAX/Data'), Path(r'W:/Data'), Path(r'/home/halechr/FastData'), Path(r'/Volumes/MoverNew/data')] # Path(r'/home/halechr/cloud/turbo/Data'), , Path(r'/nfs/turbo/umms-kdiba/Data'), Path(r'/home/halechr/turbo/Data'), 
        global_data_root_parent_path = find_first_extant_path(known_global_data_root_parent_paths)
        assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
        good_session_concrete_folders = ConcreteSessionFolder.build_concrete_session_folders(global_data_root_parent_path, included_session_contexts)
        session_basedirs_dict: Dict[IdentifyingContext, Path] = {a_session_folder.context:a_session_folder.path for a_session_folder in good_session_concrete_folders}

        # excluded_session_keys = ['kdiba_pin01_one_fet11-01_12-58-54', 'kdiba_gor01_one_2006-6-08_14-26-15', 'kdiba_gor01_two_2006-6-07_16-40-19']
        # excluded_session_contexts = [IdentifyingContext(**dict(zip(IdentifyingContext._get_session_context_keys(), v.split('_', maxsplit=3)))) for v in excluded_session_keys]

        """ 
        /nfs/turbo/umms-kdiba/Data/KDIBA/gor01/one/2006-6-12_15-55-31/output/2024-11-18_1130PM-kdiba_gor01_one_2006-6-12_15-55-31__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0-(ripple_WCorrShuffle_df)_tbin-0.025.csv
        /nfs/turbo/umms-kdiba/Data/KDIBA/gor01/one/2006-6-12_15-55-31/output/2024-11-18_1130PM_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0_standalone_all_shuffles_wcorr_array.mat
        /nfs/turbo/umms-kdiba/Data/KDIBA/gor01/one/2006-6-12_15-55-31/output/2024-11-18_1130PM_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0_standalone_wcorr_ripple_shuffle_data_only_1028.pkl
        """
        # included_session_keys = ['kdiba_gor01_one_2006-6-09_1-22-43',]
        # included_session_contexts = [IdentifyingContext(**dict(zip(IdentifyingContext._get_session_context_keys(), v.split('_', maxsplit=3)))) for v in included_session_keys]

        all_found_parsed_csv_files_df_dict = {}

        # all_found_pkl_files_dict = {}
        # all_found_pipeline_pkl_files_dict = {}
        # all_found_global_pkl_files_dict = {}
        # all_found_pipeline_h5_files_dict = {}
        
        if custom_file_globs_dict is None:
            custom_file_globs_dict = {
            #    'pkl': '*.pkl',
               'csv': '*.csv',
            #    'h5': '*.h5',
            }

        if find_most_recent_files_kwargs is None:
            find_most_recent_files_kwargs = dict(should_fallback_to_filesystem_modification_datetime=True)

        copy_dict = {}
        # moved_dict = {}

        # scripts_output_path
        for a_good_session_concrete_folder, a_session_basedir in zip(good_session_concrete_folders, session_basedirs_dict):
            if debug_print:
                print(f'a_good_session_concrete_folder: "{a_good_session_concrete_folder}", a_session_basedir: "{a_session_basedir}"')
                
            should_skip_session: bool = True
                
            a_session_output_folder = a_good_session_concrete_folder.output_folder
            
            if a_good_session_concrete_folder.context in included_session_contexts:
                should_skip_session = False
            else:
                should_skip_session = True
                
            # if a_good_session_concrete_folder.context in excluded_session_contexts:
            #     should_skip_session = True
            if should_skip_session:
                if debug_print:
                    print(f'skipping excluded session: {a_good_session_concrete_folder.context}')
            else:
                ## 
                # csv_files = find_csv_files(a_session_output_folder)
                # csv_sessions, parsed_csv_files_df  = find_most_recent_files(found_session_export_paths=csv_files, cuttoff_date=cuttoff_date) ## this is not working, as it's missing older files with different custom_replay
                # parsed_csv_files_df['filename'] = parsed_csv_files_df['path'].map(lambda x: x.name)
                # parsed_csv_files_df['desired_filepath'] = parsed_csv_files_df['path'].map(lambda x: target_dir.joinpath(x.name))
                # parsed_csv_files_df = parsed_csv_files_df[parsed_csv_files_df['file_type'] == 'ripple_WCorrShuffle_df'] ## only consider 'ripple_WCorrShuffle_df' files
                # all_found_parsed_csv_files_df_dict[a_session_basedir] = deepcopy(parsed_csv_files_df)

                # _temp_desired_copy_dict = dict(zip(parsed_csv_files_df['path'].to_list(), parsed_csv_files_df['desired_filepath'].to_list()))
                # # custom_file_types_dict = dict(zip(parsed_csv_files_df['filename'].to_list(), parsed_csv_files_df['path'].to_list()))
                # copy_dict.update(_temp_desired_copy_dict) 
                
                all_found_custom_glob_files_dict = {}
                all_found_custom_glob_parsed_files_df_dict = {}
                
                for k, a_glob in custom_file_globs_dict.items():
                    all_found_custom_glob_files_dict[k] = list(a_session_output_folder.glob(a_glob))
                    _, a_most_recent_parsed_files_df, all_parsed_files_df  = find_most_recent_files(found_session_export_paths=all_found_custom_glob_files_dict[k], cuttoff_date=cuttoff_date, **find_most_recent_files_kwargs)
                    a_most_recent_parsed_files_df['filename'] = a_most_recent_parsed_files_df['path'].map(lambda x: x.name)
                    a_most_recent_parsed_files_df['desired_filepath'] = a_most_recent_parsed_files_df['path'].map(lambda x: target_dir.joinpath(x.name))
                    all_found_custom_glob_parsed_files_df_dict[k] = a_most_recent_parsed_files_df

                    # a_parsed_files_df = a_parsed_files_df[a_parsed_files_df['file_type'] == 'ripple_WCorrShuffle_df'] ## only consider 'ripple_WCorrShuffle_df' files
                    all_found_parsed_csv_files_df_dict[a_session_basedir] = deepcopy(a_most_recent_parsed_files_df)

                    _temp_desired_copy_dict = dict(zip(a_most_recent_parsed_files_df['path'].to_list(), a_most_recent_parsed_files_df['desired_filepath'].to_list()))
                    
                    # custom_file_types_dict = dict(zip(parsed_csv_files_df['filename'].to_list(), parsed_csv_files_df['path'].to_list()))
                    
                    ## this is okay/safe because the keys are absolute paths so each session folder will add unique entries
                    copy_dict.update(_temp_desired_copy_dict) 
                
                # # display(parsed_csv_files_df)
                # all_found_global_pkl_files_dict[a_session_basedir] = list(a_session_output_folder.glob('global_computation_results*.pkl'))
                
                # for a_global_file in all_found_global_pkl_files_dict[a_session_basedir]:
                #     ## iterate through the found global files:
                #     target_file = a_good_session_concrete_folder.global_computation_result_pickle.with_name(a_global_file.name)
                #     copy_dict[a_global_file] = target_file
                #     # if not is_dryrun:
                #     ## perform the move/copy
                #     was_success = try_perform_move(src_file=a_global_file, target_file=target_file, is_dryrun=is_dryrun)
                #     if was_success:
                #         moved_dict[a_file] = target_file
                # all_found_pipeline_pkl_files_dict[a_session_basedir] = list(a_session_output_folder.glob('loadedSessPickle*.pkl'))
                # for a_file in all_found_pipeline_pkl_files_dict[a_session_basedir]:
                #     ## iterate through the found global files:
                #     target_file = a_good_session_concrete_folder.session_pickle.with_name(a_file.name)
                #     copy_dict[a_file] = target_file
                #     # if not is_dryrun:
                #     ## perform the move/copy
                #     was_success = try_perform_move(src_file=a_file, target_file=target_file, is_dryrun=is_dryrun)
                #     if was_success:
                #         moved_dict[a_file] = target_file
                # all_found_pipeline_h5_files_dict[a_session_basedir] = list(a_session_output_folder.glob('loadedSessPickle*.h5'))
                # for a_file in all_found_pipeline_h5_files_dict[a_session_basedir]:
                #     ## iterate through the found global files:
                #     target_file = a_good_session_concrete_folder.pipeline_results_h5.with_name(a_file.name)
                #     copy_dict[a_file] = target_file
                #     # if not is_dryrun:
                #     ## perform the move/copy
                #     was_success = try_perform_move(src_file=a_file, target_file=target_file, is_dryrun=is_dryrun)
                #     if was_success:
                #         moved_dict[a_file] = target_file
                # all_found_pkl_files_dict[a_session_basedir] = find_pkl_files(a_session_output_folder)

        if not is_dry_run:
            copied_files_dict_files = copy_movedict(copy_dict)
            print(F'copied: {len(copied_files_dict_files)} files!')
            
        else:
            print(f'is_dry_run == True, so not actually copying...')
            copied_files_dict_files = None
        return copy_dict, copied_files_dict_files




# ==================================================================================================================== #
# 2024-11-15 - `across_session_identity` dataframe helper                                                              #
# ==================================================================================================================== #

@pd.api.extensions.register_dataframe_accessor("across_session_identity")
class AcrossSessionIdentityDataframeAccessor:
    """ Describes a dataframe with at least a neuron_id (aclu) column. Provides functionality regarding building globally (across-sessions) unique neuron identifiers.
    

    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor


    """
   
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('neuron_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"object must be a pandas Dataframe but is of type: {type(obj)}!\nobj: {obj}")


    @classmethod
    def perform_add_session_df_columns(cls, df: pd.DataFrame, session_name: str, time_bin_size: float=None, custom_replay_source: Optional[str]=None,
                               t_start: Optional[float]=None, curr_session_t_delta: Optional[float]=None, t_end: Optional[float]=None, time_col: str=None, end_time_col_name: Optional[str]=None, **kwargs) -> pd.DataFrame:
        """ adds session-specific information to the marginal dataframes 
    
        Added Columns: ['session_name', 'time_bin_size', 'delta_aligned_start_t', 'pre_post_delta_category', 'maze_id']

        Usage:
            # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
            session_name: str = curr_active_pipeline.session_name
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
            df = DecoderDecodedEpochsResult.add_session_df_columns(df, session_name=session_name, time_bin_size=None, curr_session_t_delta=t_delta, time_col='ripple_start_t')
            
            a_ripple_df = DecoderDecodedEpochsResult.add_session_df_columns(a_ripple_df, session_name=session_name, time_bin_size=None, curr_session_t_delta=t_delta, time_col='ripple_start_t')
    
        """
        from neuropy.core.epoch import EpochsAccessor
        from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

        should_raise_exception_on_fail: bool = kwargs.pop('should_raise_exception_on_fail', False)
        

        df['session_name'] = session_name
        
        if custom_replay_source is not None:
            df['custom_replay_source'] = custom_replay_source

        if time_bin_size is not None:
            df['time_bin_size'] = np.full((len(df), ), time_bin_size)
        if curr_session_t_delta is not None:
            if time_col is None:
                # time_col = 'start' # 'ripple_start_t' for ripples, etc
                time_col: str = TimeColumnAliasesProtocol.find_first_extant_suitable_columns_name(df, col_connonical_name='start', required_columns_synonym_dict={"start":{'begin','start_t','ripple_start_t'}, "stop":['end','stop_t']}, should_raise_exception_on_fail=should_raise_exception_on_fail)
                
            if end_time_col_name is None:
                end_time_col_name: str = TimeColumnAliasesProtocol.find_first_extant_suitable_columns_name(df, col_connonical_name='stop', required_columns_synonym_dict={"start":{'begin','start_t','ripple_start_t'}, "stop":['end','stop_t']}, should_raise_exception_on_fail=should_raise_exception_on_fail)

            if time_col is not None:
                df['delta_aligned_start_t'] = df[time_col] - curr_session_t_delta
                ## Add 'pre_post_delta_category' helper column:
                df['pre_post_delta_category'] = 'post-delta'
                df.loc[(df['delta_aligned_start_t'] <= 0.0), 'pre_post_delta_category'] = 'pre-delta'
                if (t_start is not None) and (t_end is not None) and (end_time_col_name is not None):
                    try:
                        df = EpochsAccessor.add_maze_id_if_needed(epochs_df=df, t_start=t_start, t_delta=curr_session_t_delta, t_end=t_end, start_time_col_name=time_col, end_time_col_name=end_time_col_name) # Adds Columns: ['maze_id']
                    except (AttributeError, KeyError) as e:
                        print(f'could not add the "maze_id" column to the dataframe (err: {e})\n\tlikely because it lacks valid "t_start" or "t_end" columns. df.columns: {list(df.columns)}. Skipping.')
                        if should_raise_exception_on_fail:
                            raise
                    except Exception as e:
                        raise e

        return df




    def add_session_df_columns(self, session_name: str, time_bin_size: float=None, custom_replay_source: Optional[str]=None,
                               t_start: Optional[float]=None, curr_session_t_delta: Optional[float]=None, t_end: Optional[float]=None, time_col: str=None, end_time_col_name: Optional[str]=None, **kwargs) -> pd.DataFrame:
        """         
        Added Columns: ['session_name', 'time_bin_size', 'delta_aligned_start_t', 'pre_post_delta_category', 'maze_id']

        Usage:
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor

            # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
            session_name: str = curr_active_pipeline.session_name
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
            df = df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=None, curr_session_t_delta=t_delta, time_col='ripple_start_t')
            a_ripple_df = a_ripple_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=None, curr_session_t_delta=t_delta, time_col='ripple_start_t')
    
        """
        df: pd.DataFrame = deepcopy(self._obj)
        return self.perform_add_session_df_columns(df=df, session_name=session_name, time_bin_size=time_bin_size, custom_replay_source=custom_replay_source, t_start=t_start, curr_session_t_delta=curr_session_t_delta, t_end=t_end, time_col=time_col, end_time_col_name=end_time_col_name, **kwargs)


    def split_session_key_col_to_fmt_animal_exper_cols(self, session_key_col: str = 'session_name') -> pd.DataFrame:
        """ Split 'session_name' to the individual columns:
            adds columns ['format_name', 'animal', 'exper_name', 'session_name'] based on 'session_name'
            
            Usage: 
                from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor
                
                df = df.across_session_identity.split_session_key_col_to_fmt_animal_exper_cols(session_key_col='session_name')
            
        """
        df: pd.DataFrame = deepcopy(self._obj)
        _added_columns = []
        if session_key_col in df:
            if 'format_name' not in df.columns:
                df['format_name'] = df[session_key_col].map(lambda x: x.split('_', maxsplit=3)[0]) ## add animal name
            if 'animal' not in df.columns:
                df['animal'] = df[session_key_col].map(lambda x: x.split('_', maxsplit=3)[1]) ## add animal name
                ## strip the '01' suffix from each
            if 'exper_name' not in df.columns:
                df['exper_name'] = df[session_key_col].map(lambda x: x.split('_', maxsplit=3)[2]) # not needed
            if (('session_name' not in df.columns) and ('session_name' != session_key_col)):
                df['session_name'] = df[session_key_col].map(lambda x: x.split('_', maxsplit=3)[-1]) # not needed
                
        return df



@function_attributes(short_name=None, tags=['df', 'merge', 'concatenate', 'FAT', 'single_fat_df'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-13 08:26', related_items=[])
class SingleFatDataframe:
    """ instead of multiple output files for different results and using filenames for transmitting context, all information is added (duplicated for each row) to a single frame with most columns present.
    
    This only makes sense for dataframes with the same number of rows, as there would be too much duplication for epochs AND epoch_time_bins, for example.
    
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import SingleFatDataframe
    
    
    """
    to_filename_conversion_dict = {'compute_diba_quiescent_style_replay_events':'_withNewComputedReplays', 'diba_evt_file':'_withNewKamranExportedReplays', 'initial_loaded': '_withOldestImportedReplays', 'normal_computed': '_withNormalComputedReplays'}
    

    @classmethod
    def build_fat_df(cls, dfs_dict: Dict[IdentifyingContext, pd.DataFrame], additional_common_context: Optional[IdentifyingContext]=None) -> pd.DataFrame:
        """ builds a single FAT_df from a dict of identities and their corresponding dfs. Adds all of the index keys as columns, and all of their values a duplicated along all rows of the coresponding df.
        Then stacks them into a single, FAT dataframe.

        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import SingleFatDataframe
           
        """
        from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
        # from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import to_filename_conversion_dict
        
        FAT_df_list: List[pd.DataFrame] = []
    
        df_col_names_dict: Dict[IdentifyingContext, List[str]] = {k:list(v.columns) for k, v in dfs_dict.items()}
        for a_df_context, a_df in dfs_dict.items():
            ## In a single_FAT frame, we add columns with the context value for all entries in the dataframe.
            for a_ctxt_key, a_ctxt_value in a_df_context.to_dict().items():
                a_df[a_ctxt_key] = a_ctxt_value
                
            if additional_common_context is not None:
                for a_ctxt_key, a_ctxt_value in additional_common_context.to_dict().items():
                    ## need to handle lists
                    # if (ctxt.get('epochs_source', None) is not None) and (len(str(ctxt.get('epochs_source', None))) > 0) and ('epochs_source' not in subset_excludelist):
                    #     custom_suffix_string_parts.append(to_filename_conversion_dict[ctxt.get('epochs_source', None)])
                    # if (ctxt.get('included_qclu_values', None) is not None) and (len(str(ctxt.get('included_qclu_values', None))) > 0) and ('included_qclu_values' not in subset_excludelist):
                    #     custom_suffix_string_parts.append(f"qclu_{ctxt.get('included_qclu_values', None)}")
                    # if (ctxt.get('minimum_inclusion_fr_Hz', None) is not None) and (len(str(ctxt.get('minimum_inclusion_fr_Hz', None))) > 0) and ('minimum_inclusion_fr_Hz' not in subset_excludelist):
                    #     custom_suffix_string_parts.append(f"frateThresh_{ctxt.get('minimum_inclusion_fr_Hz', None):.1f}")

                    # specially_formatted_key_names_dict = {'epochs_source':'', 'included_qclu_values':f"qclu_", 'minimum_inclusion_fr_Hz':f"frateThresh_"}
                    # specially_formatted_values_dict = {'epochs_source':SingleFatDataframe.to_filename_conversion_dict.get(a_ctxt_value, f'{a_ctxt_value}'), 'included_qclu_values':f"{a_ctxt_value}", 'minimum_inclusion_fr_Hz':f"{a_ctxt_value:.1f}"}
                    
                    _default_formatter_fn = lambda v: f'{v}'
                    specially_formatted_values_dict = {'epochs_source': lambda v: SingleFatDataframe.to_filename_conversion_dict.get(v, f'{v}'), 'included_qclu_values': (lambda v: f"{v}"), 'minimum_inclusion_fr_Hz': (lambda v: f"{v:.1f}")}
                    


                    a_ctxt_value_formatter_fn = specially_formatted_values_dict.get(a_ctxt_key, _default_formatter_fn)
                    a_ctxt_value_str: str = a_ctxt_value_formatter_fn(a_ctxt_value)

                    # a_ctxt_value_str: str = specially_formatted_values_dict.get(a_ctxt_value, f'{a_ctxt_value}')

                    a_df[a_ctxt_key] = a_ctxt_value_str ## need to turn this into a flat string ValueError: Length of values (6) does not match length of index (19102)
                
            # time_col = 'start' # 'ripple_start_t' for ripples, etc
            extant_time_col: str = TimeColumnAliasesProtocol.find_first_extant_suitable_columns_name(a_df, col_connonical_name='t_bin_center', required_columns_synonym_dict={"t_bin_center":{'lap_start_t','ripple_start_t','start_t','start', 't'}}, should_raise_exception_on_fail=True)
            if extant_time_col != 't_bin_center':
                a_df['t_bin_center'] = deepcopy(a_df[extant_time_col])
            assert 't_bin_center' in a_df
            # assert np.all(np.logical_not(a_df.isna()))
            a_df['is_t_bin_center_fake'] = (extant_time_col != 't_bin_center') ## #TODO 2025-03-27 18:31: - [ ] If it's not of t_bin data_grain, it's fake, and just used for temporary calculations
            
            FAT_df_list.append(a_df)
        # end for a_df_name, a_df
        fat_df: pd.DataFrame = pd.concat(FAT_df_list, ignore_index=True)
        return fat_df
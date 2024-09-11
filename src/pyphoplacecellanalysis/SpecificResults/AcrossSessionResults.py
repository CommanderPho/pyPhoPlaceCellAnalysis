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
from typing import List, Dict, Optional,  Tuple
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



    @classmethod
    def post_compute_all_sessions_processing(cls, global_data_root_parent_path:Path, output_path_suffix: str, plotting_enabled:bool, output_override_path=None):
        """ 2023-11-15 - called after batch computing all of the sessions and building the required output files. Loads them, processes them, and then plots them!

        """
        # 2023-10-04 - Load Saved across-sessions-data and testing Batch-computed inst_firing_rates:
        ## Load the saved across-session results:
        inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{output_path_suffix}.pkl'
        across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list = AcrossSessionsResults.load_across_sessions_data(global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=inst_fr_output_filename)
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
        across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path)
        num_sessions = len(across_sessions_instantaneous_fr_dict)
        print(f'num_sessions: {num_sessions}')
        # across_sessions_instantaneous_frs_list: List[InstantaneousSpikeRateGroupsComputation] = list(across_sessions_instantaneous_fr_dict.values())
        assert np.all([len(v) == 3 for v in across_sessions_instantaneous_fr_dict.values()]), f"expected values to be tuples Tuple[Context, InstantaneousSpikeRateGroupsComputation, inst_fr_t_bin_size] but were: {list(across_sessions_instantaneous_fr_dict.values())}"
        
        across_sessions_instantaneous_frs_ctxts_list: List[IdentifyingContext] = [v[0] for v in across_sessions_instantaneous_fr_dict.values()]
        across_sessions_instantaneous_frs_list: List[InstantaneousSpikeRateGroupsComputation] = [v[1] for v in across_sessions_instantaneous_fr_dict.values()]        
        across_sessions_instantaneous_frs_t_bin_size_list: List[float] = [v[2] for v in across_sessions_instantaneous_fr_dict.values()]
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


@function_attributes(short_name=None, tags=['copy', 'batch', 'across_session'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-04 09:05', related_items=[])
def copy_session_folder_files_to_target_dir(good_session_concrete_folders, target_dir: Path, RESULT_DATE_TO_USE: str, custom_file_types_dict: Dict, debug_print=True, dry_run: bool=False):
    """ 
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
            sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv("./EXTERNAL/sessions_experiment_datetime_df.csv", session_uid_str_sep='|')
            _loaded_tables = [_callback_add_df_columns(v, session_id_column_name='session_uid') for v in _loaded_tables]

        except BaseException as e:
            print(f'failed to load and apply the sessions rank CSV to tables. Error: {e}')
            raise e
        
        return _loaded_tables





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
            sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv("./EXTERNAL/sessions_experiment_datetime_df.csv", session_uid_str_sep='|')
            neuron_identities_table = _callback_add_df_columns(neuron_identities_table, session_id_column_name='session_uid')
            neuron_replay_stats_table = _callback_add_df_columns(neuron_replay_stats_table, session_id_column_name='session_uid')
            long_short_fr_indicies_analysis_table = _callback_add_df_columns(long_short_fr_indicies_analysis_table, session_id_column_name='session_uid')

        except BaseException as e:
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
from pyphocorehelpers.Filesystem.path_helpers import try_parse_chain # used in `parse_filename`


def build_session_t_delta(t_delta_csv_path: Path):
    """
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import build_session_t_delta
    t_delta_csv_path = collected_outputs_directory.joinpath('../2024-01-18_GL_t_split_df.csv').resolve() # GL
    # t_delta_csv_path = collected_outputs_directory.joinpath('2024-06-11_GL_t_split_df.csv').resolve()

    t_delta_df, t_delta_dict = build_session_t_delta(t_delta_csv_path=t_delta_csv_path)

    """
    assert t_delta_csv_path.exists(), f"t_split_df CSV at '{t_delta_csv_path}' does not exist!"
    ## The CSV containing the session delta time:
    t_delta_df = pd.read_csv(t_delta_csv_path, index_col=0) # Assuming that your CSV file has an index column
    # adds `delta_aligned_t_start`, `delta_aligned_t_end` columns
    t_delta_df['delta_aligned_t_start'] = t_delta_df['t_start'] - t_delta_df['t_delta']
    t_delta_df['delta_aligned_t_end'] = t_delta_df['t_end'] - t_delta_df['t_delta']
    
    # computes `earliest_delta_aligned_t_start`, latest_delta_aligned_t_end
    earliest_delta_aligned_t_start: float = np.nanmin(t_delta_df['delta_aligned_t_start'])
    latest_delta_aligned_t_end: float = np.nanmax(t_delta_df['delta_aligned_t_end'])
    print(f'earliest_delta_aligned_t_start: {earliest_delta_aligned_t_start}, latest_delta_aligned_t_end: {latest_delta_aligned_t_end}')
    t_delta_dict = t_delta_df.to_dict(orient='index')
    return t_delta_df, t_delta_dict, (earliest_delta_aligned_t_start, latest_delta_aligned_t_end)

@function_attributes(short_name=None, tags=['experience_rank', 'session_order', 'csv'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-10 02:49', related_items=[])
def load_and_apply_session_experience_rank_csv(csv_path="./EXTERNAL/sessions_experiment_datetime_df.csv", session_uid_str_sep: str = '|', novel_experience_rank_requirement: int= 2):
    """Load the exported sessions experience_ranks CSV and use it to add the ['session_experience_rank', 'session_experience_orientation_rank', 'is_novel_exposure'] columns to the tables:
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import load_and_apply_session_experience_rank_csv

        all_session_experiment_experience_csv_path = Path("./EXTERNAL/sessions_experiment_datetime_df.csv").resolve()
        Assert.path_exists(all_session_experiment_experience_csv_path)
        sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv(all_session_experiment_experience_csv_path, session_uid_str_sep='_')
        all_sessions_all_scores_ripple_df = _callback_add_df_columns(all_sessions_all_scores_ripple_df, session_id_column_name='session_name')

    """
    if isinstance(csv_path, str):
        csv_path = Path(csv_path).resolve()
        
    assert csv_path.exists(), f"csv_path: '{csv_path}' does not exist!"
    sessions_df: pd.DataFrame = pd.read_csv(csv_path) # KDibaOldDataSessionFormatRegisteredClass.find_all_existing_sessions(global_data_root_parent_path=global_data_root_parent_path).sort_values(['session_datetime'])
            
    session_uid_strs = sessions_df['session_uid'].values
            
    if session_uid_str_sep != '|':
        ## replace the default separator character '|' with the user provided one
        session_uid_strs = [k.replace('|', session_uid_str_sep) for k in session_uid_strs]
        
    experience_rank_map_dict = dict(zip(session_uid_strs, sessions_df['experience_rank'].values))
    experience_orientation_rank_map_dict = dict(zip(session_uid_strs, sessions_df['experience_orientation_rank'].values))

    def _callback_add_df_columns(df, session_id_column_name: str = 'session_uid'):
        """ captures `experience_rank_map_dict`, `experience_orientation_rank_map_dict` """
        assert session_id_column_name in df.columns, f"session_id_column_name: '{session_id_column_name}' not in df.columns: {list(df.columns)}"
        df['session_experience_rank'] = df[session_id_column_name].map(experience_rank_map_dict)
        df['session_experience_orientation_rank'] = df[session_id_column_name].map(experience_orientation_rank_map_dict)
        df['is_novel_exposure'] = (df['session_experience_rank'] < novel_experience_rank_requirement)
        return df

    return sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns


@function_attributes(short_name=None, tags=['csv', 'filesystem', 'discovery'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-09 18:19', related_items=[])
def find_csv_files(directory: str, recurrsive: bool=False):
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




@function_attributes(short_name=None, tags=['parse'], input_requires=[], output_provides=[], uses=['try_parse_chain'], used_by=['find_most_recent_files'], creation_date='2024-03-28 10:16', related_items=[])
def parse_filename(path: Path, debug_print:bool=False) -> Tuple[datetime, str, Optional[str], str, str]:
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


def get_only_most_recent_csv_sessions(parsed_paths_df: pd.DataFrame) -> pd.DataFrame:
    """ returns a dataframe containing only the most recent '.err' and '.log' file for each session. 
    
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import get_only_most_recent_csv_sessions
    
    ['session', 'custom_replay_name', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime']

    I now have a different dataframe where I want to get the most recent (according to the 'export_datetime' column) for each of the groups, grouped by the following columns:
    ['session', 'custom_replay_name', 'file_type', 'decoding_time_bin_size_str']


    """
    df = deepcopy(parsed_paths_df)

    required_cols = ['session', 'custom_replay_name', 'file_type', 'decoding_time_bin_size_str', 'export_datetime'] # Replace with actual column names you require
    has_required_columns = PandasHelpers.require_columns(df, required_cols, print_missing_columns=True)
    assert has_required_columns

    df['export_datetime'] = pd.to_datetime(df['export_datetime'])
    # Replace NaN values with empty strings
    df.fillna('', inplace=True)

    # Get the most recent entry for each group
    most_recent_entries_df: pd.DataFrame = df.loc[df.groupby(['session', 'custom_replay_name', 'file_type', 'decoding_time_bin_size_str'])['export_datetime'].idxmax()]
    return most_recent_entries_df



# most_recent_only_csv_file_df = get_only_most_recent_log_files(log_file_df=all_file_df)


@function_attributes(short_name=None, tags=['recent', 'parse'], input_requires=[], output_provides=[], uses=['parse_filename'], used_by=[], creation_date='2024-04-15 09:18', related_items=['convert_to_dataframe'])
def find_most_recent_files(found_session_export_paths: List[Path], cuttoff_date:Optional[datetime]=None, debug_print: bool = False) -> Tuple[Dict[str, Dict[str, Tuple[Path, str, datetime]]], pd.DataFrame]:
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
    parsed_paths: List[Tuple] = [(*parse_filename(p), p) for p in found_session_export_paths if (parse_filename(p)[0] is not None)] # note we append path p to the end of the tuple

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
    # parsed_paths.sort(key=lambda x: (x[3] is not None, x), reverse=True)
    # parsed_paths.sort(reverse=True) # old way

    if debug_print:
        print(f'parsed_paths: {parsed_paths}')

    tuple_column_names = ['export_datetime', 'session', 'custom_replay_name', 'file_type', 'decoding_time_bin_size_str', 'path']
    parsed_paths_df: pd.DataFrame = pd.DataFrame(parsed_paths, columns=tuple_column_names)
    # Sort by columns: 'session' (ascending), 'custom_replay_name' (ascending) and 3 other columns
    parsed_paths_df = parsed_paths_df.sort_values(['session', 'file_type', 'custom_replay_name', 'decoding_time_bin_size_str', 'export_datetime']).reset_index(drop=True)

    ## This is where we drop all but the most recent:
    filtered_parsed_paths_df = get_only_most_recent_csv_sessions(parsed_paths_df=deepcopy(parsed_paths_df))

    # Drop rows with export_datetime less than or equal to cutoff_date
    if cuttoff_date is not None:
        filtered_parsed_paths_df = filtered_parsed_paths_df[filtered_parsed_paths_df['export_datetime'] > cuttoff_date]

    ## filtered_parsed_paths_df is new 2024-07-11 output

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


    ## Post-process to get session info
    # split_char: str = '__'
    # for session_str, outer_dict in sessions.items():
    #     for file_type, inner_dict in outer_dict.items():
    #         ## Parse
    #         _split_columns = session_str.split(split_char)
    #         if len(_split_columns) > 1:
    #             _session_names.append(_split_columns[0])
    #             _accrued_replay_epoch_names.append(_split_columns[-1])
    #         else:
    #             _session_names.append(_split_columns[0])
                # _accrued_replay_epoch_names.append('')

    return sessions, filtered_parsed_paths_df


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
    df = pd.read_csv(file, na_values=['', 'nan', 'np.nan', '<NA>'])
    df['session_name'] = session_name
    if curr_session_t_delta is not None:
        df['delta_aligned_start_t'] = df[time_col] - curr_session_t_delta
    return df


@function_attributes(short_name=None, tags=['csv', 'export', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-09 18:20', related_items=[])
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

        all_sessions_all_scores_ripple_df, (valid_ripple_df, invalid_ripple_df), (user_approved_ripple_df, user_rejected_ripple_df) = _split_user_annotated_ripple_df(all_sessions_all_scores_ripple_df)



    """
    from pyphocorehelpers.indexing_helpers import  partition_df


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
def load_across_sessions_exported_h5_files(collected_outputs_directory=None, cuttoff_date: Optional[datetime] = None, known_bad_session_strs=None, debug_print: bool = False):
    """

    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import load_across_sessions_exported_h5_files


    """
    from neuropy.core.user_annotations import UserAnnotationsManager

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
    h5_sessions, h5_filtered_parsed_paths_df = find_most_recent_files(found_session_export_paths=h5_files)
        
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
    h5_session_contexts = [a_good_session_ctxt for a_good_session_ctxt in good_sessions if (a_good_session_ctxt.session_name in h5_session_names)]

    # included_h5_paths = [a_session_dict.get('pipeline_results', None)[0] for a_sess_name, a_session_dict in h5_sessions.items()] # these are mis-ordered
    included_h5_paths = [h5_sessions[a_good_session_ctxt.session_name].get('pipeline_results', None)[0] for a_good_session_ctxt in h5_session_contexts]
    assert len(included_h5_paths) == len(h5_session_contexts)

    h5_contexts_paths_dict = dict(zip(h5_session_contexts, included_h5_paths))
    return h5_filtered_parsed_paths_df, h5_contexts_paths_dict

    ## OUTPUTS: parsed_h5_files_df, h5_contexts_paths_dict
    # h5_session_contexts = list(h5_contexts_paths_dict.keys())
    # included_h5_paths = list(h5_contexts_paths_dict.values())

    ## OUTPUTS: (csv_files, csv_sessions), (h5_files, h5_sessions)




@function_attributes(short_name=None, tags=['across_sessions'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-15 08:47', related_items=[])
def load_across_sessions_exported_files(cuttoff_date: Optional[datetime] = None, debug_print: bool = False):
    """

    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import load_across_sessions_exported_files


    """
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

    ## Load across session t_delta CSV, which contains the t_delta for each session:

    # cuttoff_date = datetime(2024, 3, 16)
    # cuttoff_date = None

    # t_delta_csv_path = Path(r'C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\output\collected_outputs\2024-01-18_GL_t_split_df.csv').resolve() # Apogee
    # t_delta_csv_path = Path('/home/halechr/cloud/turbo/Data/Output/collected_outputs/2024-01-18_GL_t_split_df.csv').resolve() # GL

    # collected_outputs_directory = '/home/halechr/FastData/collected_outputs/'
    # collected_outputs_directory = r'C:\Users\pho\Desktop\collected_outputs'
    # collected_outputs_directory = r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs' # APOGEE
    # collected_outputs_directory = '/home/halechr/cloud/turbo/Data/Output/collected_outputs' # GL

    known_collected_outputs_paths = [Path(v).resolve() for v in [r"K:/scratch/collected_outputs", '/Users/pho/Dropbox (University of Michigan)/MED-DibaLabDropbox/Data/Pho/Outputs/output/collected_outputs', r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs',
                                                                '/home/halechr/FastData/collected_outputs/', '/home/halechr/cloud/turbo/Data/Output/collected_outputs']]
    collected_outputs_directory = find_first_extant_path(known_collected_outputs_paths)
    assert collected_outputs_directory.exists(), f"collected_outputs_directory: {collected_outputs_directory} does not exist! Is the right computer's config commented out above?"
    # fullwidth_path_widget(scripts_output_path, file_name_label='Scripts Output Path:')
    print(f'collected_outputs_directory: {collected_outputs_directory}')

    t_delta_csv_path = collected_outputs_directory.joinpath('2024-01-18_GL_t_split_df.csv').resolve() # GL
    assert t_delta_csv_path.exists()

    ## Find the files:
    csv_files = find_csv_files(collected_outputs_directory)
    h5_files = find_HDF5_files(collected_outputs_directory)

    csv_sessions = find_most_recent_files(found_session_export_paths=csv_files, cuttoff_date=cuttoff_date)
    h5_sessions = find_most_recent_files(found_session_export_paths=h5_files)


    ## OUTPUTS: (csv_files, csv_sessions), (h5_files, h5_sessions)

    ## The CSV containing the session delta time:
    t_delta_df = pd.read_csv(t_delta_csv_path, index_col=0, na_values=['', 'nan', 'np.nan', '<NA>']) # Assuming that your CSV file has an index column
    # adds `delta_aligned_t_start`, `delta_aligned_t_end` columns
    t_delta_df['delta_aligned_t_start'] = t_delta_df['t_start'] - t_delta_df['t_delta']
    t_delta_df['delta_aligned_t_end'] = t_delta_df['t_end'] - t_delta_df['t_delta']

    # computes `earliest_delta_aligned_t_start`, latest_delta_aligned_t_end
    earliest_delta_aligned_t_start: float = np.nanmin(t_delta_df['delta_aligned_t_start'])
    latest_delta_aligned_t_end: float = np.nanmax(t_delta_df['delta_aligned_t_end'])
    print(f'earliest_delta_aligned_t_start: {earliest_delta_aligned_t_start}, latest_delta_aligned_t_end: {latest_delta_aligned_t_end}')
    t_delta_dict = t_delta_df.to_dict(orient='index')
    # t_delta_df

    # #TODO 2024-03-02 12:12: - [ ] Could add weighted correlation if there is a dataframe for that and it's computed:
    _df_raw_variable_names = ['simple_pf_pearson_merged_df', 'weighted_corr_merged_df']
    _df_variables_names = ['laps_weighted_corr_merged_df', 'ripple_weighted_corr_merged_df', 'laps_simple_pf_pearson_merged_df', 'ripple_simple_pf_pearson_merged_df']

    # # tbin_values_dict = {'laps': self.laps_decoding_time_bin_size, 'ripple': self.ripple_decoding_time_bin_size}
    time_col_name_dict = {'laps': 'lap_start_t', 'ripple': 'ripple_start_t'} ## default should be 't_bin_center'

    # fold older files:
    # {'laps_marginals_df': 'lap_start_t', 'ripple_marginals_df': 'ripple_start_t', 'laps_time_bin_marginals_df':'t_bin_center', 'ripple_time_bin_marginals_df':'t_bin_center'}


    # csv_sessions
    # Extract each of the separate files from the sessions:

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

    if cuttoff_date is not None:
        final_sessions: Dict[types.session_str, Dict[str, Path]] = {session_str:{file_type:a_path for file_type, (a_path, an_decoding_time_bin_size_str, an_export_datetime) in session_dict.items() if (an_export_datetime >= cuttoff_date)}
                                                                                                for session_str, session_dict in csv_sessions.items() }
    else:
        # no cutoff recency date:
        final_sessions: Dict[types.session_str, Dict[str, Path]] = {session_str:{file_type:a_path for file_type, (a_path, an_decoding_time_bin_size_str, an_export_datetime) in session_dict.items()}
                                                                                                for session_str, session_dict in csv_sessions.items()}


    for session_str, session_dict in final_sessions.items():
        session_name = str(session_str)  # Extract session name from the filename
        if debug_print:
            print(f'processing session_name: {session_name}')
        curr_session_t_delta = t_delta_dict.get(session_name, {}).get('t_delta', None)
        if curr_session_t_delta is None:
            print(f'WARN: curr_session_t_delta is None for session_str = "{session_str}"')

        # Process each file type with its corresponding details
        _process_and_load_exported_file(session_dict, 'laps_marginals_df', final_sessions_loaded_laps_dict, session_str, curr_session_t_delta, 'lap_start_t')
        _process_and_load_exported_file(session_dict, 'ripple_marginals_df', final_sessions_loaded_ripple_dict, session_str, curr_session_t_delta, 'ripple_start_t')
        _process_and_load_exported_file(session_dict, 'laps_time_bin_marginals_df', final_sessions_loaded_laps_time_bin_dict, session_str, curr_session_t_delta, 't_bin_center')
        _process_and_load_exported_file(session_dict, 'ripple_time_bin_marginals_df', final_sessions_loaded_ripple_time_bin_dict, session_str, curr_session_t_delta, 't_bin_center')
        _process_and_load_exported_file(session_dict, 'laps_simple_pf_pearson_merged_df', final_sessions_loaded_simple_pearson_laps_dict, session_str, curr_session_t_delta, 'lap_start_t')
        _process_and_load_exported_file(session_dict, 'ripple_simple_pf_pearson_merged_df', final_sessions_loaded_simple_pearson_ripple_dict, session_str, curr_session_t_delta, 'ripple_start_t')
        _process_and_load_exported_file(session_dict, 'laps_weighted_corr_merged_df', final_sessions_loaded_laps_wcorr_dict, session_str, curr_session_t_delta, 'lap_start_t')
        _process_and_load_exported_file(session_dict, 'ripple_weighted_corr_merged_df', final_sessions_loaded_ripple_wcorr_dict, session_str, curr_session_t_delta, 'ripple_start_t')

        # process_and_load_file(session_dict, 'laps_all_scores_merged_df', final_sessions_loaded_laps_all_scores_dict, session_str, curr_session_t_delta, 'lap_start_t')
        _process_and_load_exported_file(session_dict, 'ripple_all_scores_merged_df', final_sessions_loaded_ripple_all_scores_dict, session_str, curr_session_t_delta, 'ripple_start_t')



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
    # all_sessions_all_score_laps_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_laps_all_scores_dict.values()), axis='index', ignore_index=True)
    all_sessions_all_scores_ripple_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_ripple_all_scores_dict.values()), axis='index', ignore_index=True)

    dfs_list = (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df)
    for a_df in dfs_list:
        if a_df is not None:
            if 'time_bin_size' not in a_df:
                print('Uh-oh! time_bin_size is missing! This must be old exports!')
                print(f'\tTry to determine the time_bin_size from the filenames: {csv_sessions}')
                ## manual correction UwU
                time_bin_size: float = 0.025
                print(f'WARNING! MANUAL OVERRIDE TIME BIN SIZE SET: time_bin_size = {time_bin_size}. Assigning to dataframes....')
                a_df['time_bin_size'] = time_bin_size
            else:
                # Filter rows based on column: 'time_bin_size'
                a_df = a_df[a_df['time_bin_size'].notna()]


    # if 'time_bin_size' not in all_sessions_laps_df:
    #     print('Uh-oh! time_bin_size is missing! This must be old exports!')
    #     print(f'\tTry to determine the time_bin_size from the filenames: {csv_sessions}')
    #     ## manual correction UwU
    #     time_bin_size: float = 0.025
    #     print(f'WARNING! MANUAL OVERRIDE TIME BIN SIZE SET: time_bin_size = {time_bin_size}. Assigning to dataframes....')
    #     all_sessions_laps_df['time_bin_size'] = time_bin_size
    #     all_sessions_ripple_df['time_bin_size'] = time_bin_size
    #     all_sessions_laps_time_bin_df['time_bin_size'] = time_bin_size
    #     all_sessions_ripple_time_bin_df['time_bin_size'] = time_bin_size
    #     print(f'\tdone.')
    # else:
    #     # Filter rows based on column: 'time_bin_size'
    #     all_sessions_laps_df = all_sessions_laps_df[all_sessions_laps_df['time_bin_size'].notna()]
    #     all_sessions_ripple_df = all_sessions_ripple_df[all_sessions_ripple_df['time_bin_size'].notna()]
    #     all_sessions_laps_time_bin_df = all_sessions_laps_time_bin_df[all_sessions_laps_time_bin_df['time_bin_size'].notna()]
    #     all_sessions_ripple_time_bin_df = all_sessions_ripple_time_bin_df[all_sessions_ripple_time_bin_df['time_bin_size'].notna()]

    all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df = [_common_cleanup_operations(a_df) for a_df in (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df)]
    all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df = [_common_cleanup_operations(a_df) for a_df in (all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df)]

    # all_sessions_all_score_laps_df, all_sessions_all_scores_ripple_df = [_common_cleanup_operations(a_df) for a_df in (all_sessions_all_score_laps_df, all_sessions_all_scores_ripple_df)]
    # all_sessions_all_score_laps_df = _common_cleanup_operations(all_sessions_all_score_laps_df)
    all_sessions_all_scores_ripple_df = _common_cleanup_operations(all_sessions_all_scores_ripple_df)

    all_sessions_simple_pearson_laps_df: pd.DataFrame = DecoderDecodedEpochsResult.merge_decoded_epochs_result_dfs(all_sessions_simple_pearson_laps_df, all_sessions_wcorr_laps_df, should_drop_directional_columns=False, start_t_idx_name='delta_aligned_start_t')
    all_sessions_simple_pearson_ripple_df: pd.DataFrame = DecoderDecodedEpochsResult.merge_decoded_epochs_result_dfs(all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_ripple_df, should_drop_directional_columns=False, start_t_idx_name='ripple_start_t')

    # all_sessions_laps_time_bin_df # 601845 rows × 9 column


    ## OUTPUTS: final_sessions: Dict[types.session_str, Dict[str, Path]], all_sessions_all_scores_ripple_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df
    return final_sessions, (all_sessions_all_scores_ripple_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df), (csv_files, csv_sessions), (h5_files, h5_sessions)



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


def _concat_all_dicts_to_dfs(final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict):
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
    # all_sessions_all_score_laps_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_laps_all_scores_dict.values()), axis='index', ignore_index=True)
    all_sessions_all_scores_ripple_df: pd.DataFrame = PandasHelpers.safe_concat(list(final_sessions_loaded_ripple_all_scores_dict.values()), axis='index', ignore_index=True)

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

    all_sessions_simple_pearson_laps_df: pd.DataFrame = DecoderDecodedEpochsResult.merge_decoded_epochs_result_dfs(all_sessions_simple_pearson_laps_df, all_sessions_wcorr_laps_df, should_drop_directional_columns=False, start_t_idx_name='delta_aligned_start_t')
    all_sessions_simple_pearson_ripple_df: pd.DataFrame = DecoderDecodedEpochsResult.merge_decoded_epochs_result_dfs(all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_ripple_df, should_drop_directional_columns=False, start_t_idx_name='ripple_start_t')
    # all_sessions_laps_time_bin_df # 601845 rows × 9 column

    ## epoch-based ones:
    if all_sessions_ripple_df is not None:
        all_sessions_ripple_df = recover_user_annotation_and_is_valid_columns(all_sessions_ripple_df, all_sessions_all_scores_df=all_sessions_all_scores_ripple_df, a_time_column_names='ripple_start_t')

    return (
        all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df
    )


@function_attributes(short_name=None, tags=['csv', 'OLD'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-11 18:06', related_items=[])
def _old_process_csv_files(csv_sessions, t_delta_dict, cuttoff_date=None, known_bad_session_strs=[], debug_print=False):
    """ OLD (pre 2024-07-11) method for processing CSV files.

    Usage:
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import _old_process_csv_files
        
        ## OLD (pre 2024-07-11):
        ## INPUTS: csv_sessions
        dict_results, df_results = _old_process_csv_files(csv_sessions=csv_sessions, t_delta_dict=t_delta_dict, cuttoff_date=cuttoff_date, known_bad_session_strs=known_bad_session_strs, debug_print=False)
        (final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict) = dict_results
        (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df) = df_results

    """
    # csv_sessions
    # We now have the correct and parsed filepaths for each of the exported .csvs, now we need to actually load them and concatenate them toggether across sessions.
    # Extract each of the separate files from the sessions:
    ## 
    ## INPUTS: csv_sessions
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

    if cuttoff_date is not None:
        final_sessions: Dict[types.session_str, Dict[str, Path]] = {session_str:{file_type:a_path for file_type, (a_path, an_decoding_time_bin_size_str, an_export_datetime) in session_dict.items() if ((an_export_datetime >= cuttoff_date) and (session_str not in known_bad_session_strs))}
                                                                                                for session_str, session_dict in csv_sessions.items() }
    else:
        # no cutoff recency date:
        final_sessions: Dict[types.session_str, Dict[str, Path]] = {session_str:{file_type:a_path for file_type, (a_path, an_decoding_time_bin_size_str, an_export_datetime) in session_dict.items() if (session_str not in known_bad_session_strs)}
                                                                                                for session_str, session_dict in csv_sessions.items()}

    for session_str, session_dict in final_sessions.items():
        session_name = str(session_str)  # Extract session name from the filename
        if debug_print:
            print(f'processing session_name: {session_name}')
        curr_session_t_delta = t_delta_dict.get(session_name, {}).get('t_delta', None)
        if curr_session_t_delta is None:
            print(f'WARN: curr_session_t_delta is None for session_str = "{session_str}"')

        # Process each file type with its corresponding details
        _process_and_load_exported_file(session_dict, 'laps_marginals_df', final_sessions_loaded_laps_dict, session_str, curr_session_t_delta, 'lap_start_t')
        _process_and_load_exported_file(session_dict, 'ripple_marginals_df', final_sessions_loaded_ripple_dict, session_str, curr_session_t_delta, 'ripple_start_t')
        _process_and_load_exported_file(session_dict, 'laps_time_bin_marginals_df', final_sessions_loaded_laps_time_bin_dict, session_str, curr_session_t_delta, 't_bin_center')
        _process_and_load_exported_file(session_dict, 'ripple_time_bin_marginals_df', final_sessions_loaded_ripple_time_bin_dict, session_str, curr_session_t_delta, 't_bin_center')
        _process_and_load_exported_file(session_dict, 'laps_simple_pf_pearson_merged_df', final_sessions_loaded_simple_pearson_laps_dict, session_str, curr_session_t_delta, 'lap_start_t')
        _process_and_load_exported_file(session_dict, 'ripple_simple_pf_pearson_merged_df', final_sessions_loaded_simple_pearson_ripple_dict, session_str, curr_session_t_delta, 'ripple_start_t')
        _process_and_load_exported_file(session_dict, 'laps_weighted_corr_merged_df', final_sessions_loaded_laps_wcorr_dict, session_str, curr_session_t_delta, 'lap_start_t')
        _process_and_load_exported_file(session_dict, 'ripple_weighted_corr_merged_df', final_sessions_loaded_ripple_wcorr_dict, session_str, curr_session_t_delta, 'ripple_start_t')
        
        # _process_and_load_exported_file(session_dict, 'laps_all_scores_merged_df', final_sessions_loaded_laps_all_scores_dict, session_str, curr_session_t_delta, 'lap_start_t')
        _process_and_load_exported_file(session_dict, 'ripple_all_scores_merged_df', final_sessions_loaded_ripple_all_scores_dict, session_str, curr_session_t_delta, 'ripple_start_t')

    ## OUTPUTS: final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict,
       
    # return (final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict)
    
    ## Build across_sessions join dataframes:
    all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df = _concat_all_dicts_to_dfs(final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict)
    ## OUTPUTS: all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df
    
    return (
        (final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict),
        (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df),
    )



def _subfn_new_df_process_and_load_exported_file(file_path, loaded_dict: Dict, session_name: str, curr_session_t_delta: float, time_key: str, debug_print:bool=False, **additional_columns) -> bool:
    try:
        # loaded_dict[session_name] = read_and_process_csv_file(file_path, session_name, curr_session_t_delta, time_key)
        df = pd.read_csv(file_path, na_values=['', 'nan', 'np.nan', '<NA>'])
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

    except BaseException as e:
        if debug_print:
            print(f'session "{session_name}", file_path: "{file_path}" - did not fully work. (error "{e}". Skipping.')
        return False

@function_attributes(short_name=None, tags=['csv'], input_requires=[], output_provides=[], uses=['_new_df_process_and_load_exported_file', 'recover_user_annotation_and_is_valid_columns'], used_by=[], creation_date='2024-07-11 17:11', related_items=[])
def _new_process_csv_files(parsed_csv_files_df: pd.DataFrame, t_delta_dict: Dict, cuttoff_date=None, known_bad_session_strs=[], all_session_experiment_experience_csv_path:Path=None, debug_print=False):
    """  NEW `parsed_csv_files_df1-based approach 2024-07-11 
    # We now have the correct and parsed filepaths for each of the exported .csvs, now we need to actually load them and concatenate them toggether across sessions.
    # Extract each of the separate files from the sessions:

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
    
    sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv("./EXTERNAL/sessions_experiment_datetime_df.csv")
    
    # Sort by columns: 'session' (ascending), 'custom_replay_name' (ascending) and 3 other columns
    parsed_csv_files_df = parsed_csv_files_df.sort_values(['session', 'file_type', 'custom_replay_name', 'decoding_time_bin_size_str', 'export_datetime'], ascending=[True, True, True, True, False]).reset_index(drop=True) # ensures all are sorted ascending except for export_datetime, which are sorted decending so the first value is the most recent.

    excluded_or_outdated_files_list = []
    for index, row in parsed_csv_files_df.iterrows():
        session_str = str(row['session'])
        if session_str in known_bad_session_strs:
            print(f'Skipping "{session_str}" because it is in known_bad_session_strs.')
            excluded_or_outdated_files_list.append(row['path']) ## mark file for exclusion/removal
            continue
        
        custom_replay_name = row['custom_replay_name']
        file_type = row['file_type']
        decoding_time_bin_size_str = row['decoding_time_bin_size_str']
    
        ## Setup:
        additional_columns_dict = {'custom_replay_name': custom_replay_name}
        additional_columns_dict['time_bin_size'] = try_convert_to_float(decoding_time_bin_size_str, default_val=np.nan)

        export_datetime = row['export_datetime']
        if cuttoff_date is not None:
            if export_datetime < cuttoff_date:
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
            print(f'WARN: curr_session_t_delta is None for session_str = "{session_str}"')

        _is_file_valid = True # shouldn't mark unknown files as invalid
        # Process each file type with its corresponding details
        if file_type == 'laps_marginals_df':
            _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_laps_dict, session_name, curr_session_t_delta, time_key='lap_start_t', **additional_columns_dict)
        elif file_type == 'ripple_marginals_df':
            _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_ripple_dict, session_name, curr_session_t_delta, time_key='ripple_start_t', **additional_columns_dict)
        elif file_type == 'laps_time_bin_marginals_df':
            _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_laps_time_bin_dict, session_name, curr_session_t_delta, time_key='t_bin_center', **additional_columns_dict)
        elif file_type == 'ripple_time_bin_marginals_df':
            _is_file_valid = _subfn_new_df_process_and_load_exported_file(path, final_sessions_loaded_ripple_time_bin_dict, session_name, curr_session_t_delta, time_key='t_bin_center', **additional_columns_dict)
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
        else:
            print(f'File type {file_type} not implemented.')
            # _is_file_valid = False # shouldn't mark unknown files as invalid
            # File type neuron_replay_stats_df not implemented.
            # File type laps_marginals_df not implemented.
            # continue
            
        if (not _is_file_valid):
            excluded_or_outdated_files_list.append(row['path']) ## mark file for exclusion/removal
            
        ## END FOR    

    ## OUTPUTS: final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict,
    
    ## Build across_sessions join dataframes:
    all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df = _concat_all_dicts_to_dfs(final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict)
    ## OUTPUTS: all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df
    try:
        if all_session_experiment_experience_csv_path is None:
            all_session_experiment_experience_csv_path = Path("./EXTERNAL/sessions_experiment_datetime_df.csv").resolve()
            Assert.path_exists(all_session_experiment_experience_csv_path)
        df_results = (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df)
        sessions_df, (experience_rank_map_dict, experience_orientation_rank_map_dict), _callback_add_df_columns = load_and_apply_session_experience_rank_csv(all_session_experiment_experience_csv_path, session_uid_str_sep='_')
        # all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df = [_callback_add_df_columns(a_df, session_id_column_name='session_name') for a_df in (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df)]
        df_results = [_callback_add_df_columns(a_df, session_id_column_name='session_name') for a_df in df_results]
        all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df = df_results # unpack again
    except BaseException as e:
        print(f'WARNING: failed to add the session_experience_rank CSV results to the dataframes. Failed with error: {e}. Skipping and continuing.') 
        # raise e
    
    return (
        (final_sessions_loaded_laps_dict, final_sessions_loaded_ripple_dict, final_sessions_loaded_laps_time_bin_dict, final_sessions_loaded_ripple_time_bin_dict, final_sessions_loaded_simple_pearson_laps_dict, final_sessions_loaded_simple_pearson_ripple_dict, final_sessions_loaded_laps_wcorr_dict, final_sessions_loaded_ripple_wcorr_dict, final_sessions_loaded_laps_all_scores_dict, final_sessions_loaded_ripple_all_scores_dict),
        (all_sessions_laps_df, all_sessions_ripple_df, all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df, all_sessions_simple_pearson_laps_df, all_sessions_simple_pearson_ripple_df, all_sessions_wcorr_laps_df, all_sessions_wcorr_ripple_df, all_sessions_all_scores_ripple_df),
        excluded_or_outdated_files_list,
    )

@function_attributes(short_name=None, tags=['archive', 'cleanup', 'filesystem'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-02 11:30', related_items=[])
def archive_old_files(collected_outputs_directory: Path, excluded_or_outdated_files_list: List[Path], is_dry_run: bool=False):
    """ moves old files that didn't meet the inclusion criteria into an archive directory.
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

# ==================================================================================================================== #
# Visualizations                                                                                                       #
# ==================================================================================================================== #

@metadata_attributes(short_name=None, tags=['across-session', 'visualizations', 'figure', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-07-21 00:00', related_items=[])
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










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
from pathlib import Path
from typing import List, Dict, Optional,  Tuple
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
from pyphocorehelpers.function_helpers import function_attributes

# NeuroPy (Diba Lab Python Repo) Loading
## For computation parameters:
from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update
from neuropy.core.neuron_identities import  neuronTypesEnum, NeuronIdentityTable
from neuropy.utils.mixins.HDF5_representable import HDF_Converter

from pyphocorehelpers.Filesystem.metadata_helpers import  get_file_metadata


from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData

# from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation # for `BatchSessionCompletionHandler`, `AcrossSessionsAggregator`
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import  build_and_write_to_file

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
        assert global_batch_result_inst_fr_file_path.exists()
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
        out_parent_path: Path = override_output_parent_path or Path('output/across_session_results')
        out_parent_path = out_parent_path.resolve()

        if output_path_suffix is not None:
            out_parent_path = out_parent_path.joinpath(output_path_suffix).resolve()

        out_parent_path.mkdir(parents=True, exist_ok=True)

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


from typing import Dict, List, Tuple, Optional
import neuropy.utils.type_aliases as types
from attrs import define
from pyphocorehelpers.Filesystem.path_helpers import try_parse_chain # used in `parse_filename`


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


@function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-15 09:18', related_items=['convert_to_dataframe'])
def find_most_recent_files(found_session_export_paths: List[Path], cuttoff_date:Optional[datetime]=None, debug_print: bool = False) -> Dict[str, Dict[str, Tuple[Path, str, datetime]]]:
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

    # Function that helps sort tuples by handling None values.
    def sort_key(tup):
        # Assign a boolean for each element, True if it's None, to ensure None values are sorted last.
        return (
            tup[0],                       # Sort by datetime first
            tup[1] or '',                  # Then by the string, ensuring None becomes empty string
            tup[2] or '',                  # Then by the next string, ensuring None becomes empty string
            tup[2] or '', #float('-inf') if tup[3] is None else tup[3],  # Then use -inf to ensure None ends up last
            tup[4]                         # Finally by path which should handle None by itself
        )

    # Now we sort the data using our custom sort key
    parsed_paths = sorted(parsed_paths, key=sort_key, reverse=True)

    # parsed_paths.sort(key=lambda x: (x[3] is not None, x), reverse=True)
    # parsed_paths.sort(reverse=True) # old way

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


@function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-15 09:18', related_items=['find_most_recent_files'])
def convert_to_dataframe(csv_sessions: Dict[str, Dict[str, Tuple[Path, str, datetime]]], debug_print:bool=False) -> pd.DataFrame:
    """ Converts the outp[ut of `find_most_recent_files` into a dataframe.

    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import convert_to_dataframe

    parsed_files_df: pd.DataFrame = convert_to_dataframe(csv_sessions)
    parsed_files_df

    """
    _output_tuples = []

    for session_str, a_filetype_dict in csv_sessions.items():
        if debug_print:
            print(f'session_str: {session_str}')
        for file_type, parse_tuple in a_filetype_dict.items():
            if debug_print:
                print(f'\tfile_type: {file_type}')
                print(f'\t\tparse_tuple: {parse_tuple}')
            # path, decoding_time_bin_size_str, export_datetime = parse_tuple
            _output_tuples.append((session_str, file_type, *parse_tuple))

    return pd.DataFrame(_output_tuples, columns=['session', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime'])
    # parsed_files_df


def process_csv_file(file: str, session_name: str, curr_session_t_delta: Optional[float], time_col: str) -> pd.DataFrame:
    """ reads the CSV file and adds the 'session_name' column if it is missing.

    """
    df = pd.read_csv(file, na_values=['', 'nan', 'np.nan', '<NA>'])
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


def _process_and_load_exported_file(session_dict, df_file_name_key: str, loaded_dict: Dict, session_name: str, curr_session_t_delta: float, time_key: str) -> None:
    """ updates loaded_dict """
    try:
        file_path = session_dict[df_file_name_key]
        loaded_dict[session_name] = process_csv_file(file_path, session_name, curr_session_t_delta, time_key)
    except BaseException as e:
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

def load_across_sessions_exported_h5_files(collected_outputs_directory=None, cuttoff_date: Optional[datetime] = None, known_bad_session_strs=None, debug_print: bool = False):
    """

    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import load_across_sessions_exported_h5_files


    """
    from neuropy.core.user_annotations import UserAnnotationsManager

    if collected_outputs_directory is None:
        known_collected_outputs_paths = [Path(v).resolve() for v in [r"K:/scratch/collected_outputs", '/Users/pho/Dropbox (University of Michigan)/MED-DibaLabDropbox/Data/Pho/Outputs/output/collected_outputs', r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs',
                                                                    '/home/halechr/FastData/collected_outputs/', '/home/halechr/cloud/turbo/Data/Output/collected_outputs']]
        collected_outputs_directory = find_first_extant_path(known_collected_outputs_paths)

    assert collected_outputs_directory.exists(), f"collected_outputs_directory: {collected_outputs_directory} does not exist! Is the right computer's config commented out above?"
    # fullwidth_path_widget(scripts_output_path, file_name_label='Scripts Output Path:')
    print(f'collected_outputs_directory: {collected_outputs_directory}')

    ## Find the files:
    h5_files = find_HDF5_files(collected_outputs_directory)
    h5_sessions = find_most_recent_files(found_session_export_paths=h5_files)

    ## INPUTS: h5_sessions, session_dict, cuttoff_date, known_bad_session_strs
    if known_bad_session_strs is None:
        known_bad_session_strs = []

    parsed_h5_files_df: pd.DataFrame = convert_to_dataframe(h5_sessions)

    if cuttoff_date is not None:
        # 'session', 'file_type', 'path', 'decoding_time_bin_size_str', 'export_datetime'
        parsed_h5_files_df = parsed_h5_files_df[parsed_h5_files_df['export_datetime'] >= cuttoff_date]


    parsed_h5_files_df = parsed_h5_files_df[np.isin(parsed_h5_files_df['session'], known_bad_session_strs, invert=True)] # drop all sessions that are in the known_bad_session_strs

    # parsed_h5_files_df: pd.DataFrame = convert_to_dataframe(final_h5_sessions)

    ## INPUTS: h5_sessions
    h5_session_names = list(h5_sessions.keys())
    good_sessions = UserAnnotationsManager.get_hardcoded_good_sessions()
    h5_session_contexts = [a_good_session_ctxt for a_good_session_ctxt in good_sessions if (a_good_session_ctxt.session_name in h5_session_names)]

    # included_h5_paths = [a_session_dict.get('pipeline_results', None)[0] for a_sess_name, a_session_dict in h5_sessions.items()] # these are mis-ordered
    included_h5_paths = [h5_sessions[a_good_session_ctxt.session_name].get('pipeline_results', None)[0] for a_good_session_ctxt in h5_session_contexts]
    assert len(included_h5_paths) == len(h5_session_contexts)

    h5_contexts_paths_dict = dict(zip(h5_session_contexts, included_h5_paths))
    return parsed_h5_files_df, h5_contexts_paths_dict

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
    def across_sessions_firing_rate_index_figure(cls, long_short_fr_indicies_analysis_results: pd.DataFrame, num_sessions:int, save_figure=True, **kwargs):
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

        # active_context = long_short_fr_indicies_analysis_results['active_context']
        global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.
        active_context = global_multi_session_context
        final_context = active_context.adding_context('display_fn', display_fn_name='across_sessions_firing_rate_index')
        final_context = DisplaySpecifyingIdentifyingContext.init_from_context(final_context, display_dict={})

        scatter_plot_kwargs = dict()
        if 'has_pf_color' in long_short_fr_indicies_analysis_results:
            scatter_plot_kwargs['edgecolors'] = long_short_fr_indicies_analysis_results['has_pf_color'].to_numpy() #.to_list() # edgecolors=(r, g, b, 1)


        fig, ax, scatter_plot = _plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, final_context, debug_print=True, is_centered=False, enable_hover_labels=False, enable_tiny_point_labels=False, facecolor='w', **scatter_plot_kwargs) #  markeredgewidth=1.5,

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










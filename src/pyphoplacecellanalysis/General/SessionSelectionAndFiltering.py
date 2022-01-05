## Idea: Build-up Filters for filtering the DataFrame sub-objects of my new DataFrame-based DataSession.
""" Types of filters
flat_spikes_df_inclusion: for a single filter/selector config, this is a pd.DataFrame or pd.Series object the same height as sess.spikes_df that specifies whether a given spike (corresponding to a row in the df) is included or not. The column name can be the name of the filter critiera.
    a filter excluding neurons (such as filtering by (neuron_id == aclu), (neuron_type == cell_type))
    a filter excluding time ranges (such as filtering by (epochs == maze), (lap_id == lap_id), or raw time ranges.

position_df_inclusion: for a single filter/selector config, this is a pd.DataFrame or pd.Series object the same height as sess.pos_df that specifies whether a given position datapoint (corresponding to a row in the df) is included or not.
    a filter excluding time ranges (such as filtering by (epochs == maze), (lap_id == lap_id), or raw time ranges.
    a filter excluding based on velocities, speeds, position values, etc.
"""
# NeuroPy (Diba Lab Python Repo) Loading
# import importlib
import sys

import numpy as np

try:
    from neuropy import core

    # importlib.reload(core)
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print(
        "neuropy module not found, adding directory to sys.path. \n >> Updated sys.path."
    )
    from neuropy import core

from neuropy.core.flattened_spiketrains import FlattenedSpiketrains
from neuropy.core.neurons import Neurons, NeuronType
from neuropy.core.position import Position  # , PositionAccessor
from neuropy.core.session.dataSession import DataSession



""" See sess.laps.as_epoch_obj()
lap_specific_epochs = sess.laps.as_epoch_obj()
any_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(len(sess.laps.lap_id))])
even_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(0, len(sess.laps.lap_id), 2)])
odd_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(1, len(sess.laps.lap_id), 2)])
  

"""


## Efficiently filter by cell type and desired ids
def batch_filter_session(sess, position, spikes_df, epochs):
    """a workaround to efficiently filter DataSession objects by epochs and cell_type (currently hardcoded Pyramidal) that works around the issue with deepcopy(...) on DataSessions filled with Bapun's data."""
    position.compute_higher_order_derivatives()
    pos_df = (
        position.compute_smoothed_position_info()
    )  ## Smooth the velocity curve to apply meaningful logic to it
    pos_df = position.to_dataframe().copy()
    pos_df.position.speed
    # pos_df = position.to_dataframe().copy() # 159 ms ± 3.95 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    spk_df = (
        spikes_df.copy()
    )  # 949 ms ± 62.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    spk_df.spikes.set_time_variable_name("t_seconds")
    # spk_df.spikes.time_variable_name # 't_seconds'
    # print(f'spk_df.spikes.time_variable_name: {spk_df.spikes.time_variable_name}')

    # filtering:

    # Filter by multiple conditions:
    # Perform general pre-filtering for neuron type and constraining between the ranges:
    # included_neuron_type = 'pyramidal'
    included_neuron_type = NeuronType.PYRAMIDAL
    filtered_spikes_df = spk_df.query(
        "@epochs.t_start <= `t_seconds` <= @epochs.t_stop and `cell_type` == @included_neuron_type"
    )  # 272 ms, 393 ms, Wall time: 183 ms
    # filtered_spikes_df = spk_df.query("@epochs.t_start <= `t_seconds` <= @epochs.t_stop and `aclu` in @filtered_spikes_df.spikes.neuron_ids") # 272 ms, 393 ms, Wall time: 183 ms
    # filtered_spikes_df = spk_df.query("@epochs.t_start <= `t_seconds` <= @epochs.t_stop and `aclu` in @spk_df.spikes.neuron_ids and `cell_type` == @included_neuron_type" ) # 272 ms, 393 ms, Wall time: 183 ms
    # filtered_spikes_df = spk_df.query("@epochs.t_start <= `t_seconds` <= @epochs.t_stop and `aclu` in @filtered_spikes_df.spikes.neuron_ids and `cell_type` == @included_neuron_type" ) # 272 ms, 393 ms, Wall time: 183 ms
    # filtered_spikes_df = spk_df.query("`t_seconds` >= @epochs.starts and `t_seconds` <= @epochs.stops")
    # filtered_spikes_df = spk_df.query("`t_seconds` >= @epochs.starts and `t_seconds` <= @epochs.stops and `aclu` in @filtered_spikes_df.spikes.neuron_ids")

    filtered_spikes_df = filtered_spikes_df.spikes.time_sliced(
        epochs.starts, epochs.stops
    )  # 168 ms ± 4.5 ms per loop (mean ± std. dev. of 7 runs, 10 loops each), Wall time: 20.4 ms
    filtered_pos_df = pos_df.position.time_sliced(
        epochs.starts, epochs.stops
    )  # 27.6 ms ± 2.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    ## Old Separate way:
    # if epochs is not None:
    #     # filter the spikes_df:
    #     filtered_spikes_df = spk_df.spikes.time_sliced(epochs.starts, epochs.stops)
    #     # filter the pos_df:
    #     filtered_pos_df = pos_df.position.time_sliced(epochs.starts, epochs.stops) # 5378 rows × 18 columns
    # else:
    #     # if no epochs filtering, set the filtered objects to be sliced by the available range of the position data (given by position.t_start, position.t_stop)
    #     filtered_spikes_df = spk_df.spikes.time_sliced(position.t_start, position.t_stop)
    #     filtered_pos_df = pos_df.position.time_sliced(position.t_start, position.t_stop)

    # Debug print output:
    print(
        f"Spikes Dataframe: \nnp.shape(sess.spikes_df): {np.shape(sess.spikes_df)}"
    )  # (16318817, 10)
    print(
        f"np.shape(filtered_spikes_df): {np.shape(filtered_spikes_df)}"
    )  # (1236690, 10)

    print(
        f"Position Dataframe: \nnp.shape(sess.position.to_dataframe()): {np.shape(sess.position.to_dataframe())}"
    )  # (2538347, 12)
    print(f"np.shape(filtered_pos_df):{np.shape(filtered_pos_df)}")  # (174000, 5)

    # Once filtering is done, apply the grouping:
    Neurons.initialize_missing_spikes_df_columns(filtered_spikes_df)

    # Group by the aclu (cluster indicator) column
    # cell_grouped_spikes_df = filtered_spikes_df.groupby(['aclu']) # Wall time: 15.6 ms
    # cell_spikes_dfs = [cell_grouped_spikes_df.get_group(a_neuron_id) for a_neuron_id in filtered_spikes_df.spikes.neuron_ids] # a list of dataframes for each neuron_id, Wall time: 723 ms

    # Finally, make the filtered session:
    neurons_obj = Neurons.from_dataframe(
        filtered_spikes_df,
        sess.recinfo.dat_sampling_rate,
        time_variable_name=spk_df.spikes.time_variable_name,
    )
    # neurons_obj = None # Wait, it doesn't even set a neurons object and yet it all still works!!
    # Doesn't mess with laps, probegroup
    filtered_sess = DataSession(
        sess.config,
        filePrefix=sess.filePrefix,
        recinfo=sess.recinfo,
        eegfile=sess.eegfile,
        datfile=sess.datfile,
        neurons=neurons_obj,
        probegroup=sess.probegroup,
        position=Position(filtered_pos_df, metadata=sess.position.metadata),
        paradigm=epochs,
        ripple=sess.ripple,
        mua=sess.mua,
        laps=sess.laps,
        flattened_spiketrains=FlattenedSpiketrains(
            filtered_spikes_df,
            time_variable_name=spk_df.spikes.time_variable_name,
            t_start=epochs.t_start,
            metadata=sess.flattened_spiketrains.metadata,
        ),
    )  # 15.6 ms

    return filtered_sess

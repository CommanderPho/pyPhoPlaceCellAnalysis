from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Dict, Callable, List, Optional, Tuple, Union, Any, Sequence, Literal, Set
from attrs import define, field
import nptyping as ND
from nptyping import NDArray
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib as mpl
import matplotlib.patches as mpatches # used for plot_epoch_track_assignments
from flexitext import flexitext ## flexitext version

from neuropy.utils.mixins.enum_helpers import ExtendedEnum # used in TrackAssignmentState
from neuropy.core.epoch import Epoch, ensure_dataframe
from neuropy.core.user_annotations import UserAnnotationsManager, SessionCellExclusivityRecord
from neuropy.utils.indexing_helpers import PandasHelpers

from pyphocorehelpers.mixins.key_value_hashable import KeyValueHashableObject
from pyphocorehelpers.indexing_helpers import partition # needed by `AssigningEpochs` to partition the dataframe by aclus

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences

## Laps Stuff:
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass

from neuropy.utils.result_context import IdentifyingContext, providing_context, DisplaySpecifyingIdentifyingContext
from neuropy.core.user_annotations import UserAnnotationsManager
from neuropy.utils.mixins.indexing_helpers import UnpackableMixin, pop_dict_subset

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import determine_long_short_pf1D_indicies_sort_by_peak
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from neuropy.utils.matplotlib_helpers import FormattedFigureText


from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots, BatchPhoJonathanFiguresHelper
from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import interactive_good_epoch_selections

import pyphoplacecellanalysis.External.pyqtgraph as pg # pyqtgraph
import matplotlib.pyplot as plt

from pyphoplacecellanalysis.SpecificResults.fourthYearPresentation import fig_surprise_results, fig_remapping_cells
from pyphocorehelpers.indexing_helpers import list_of_dicts_to_dict_of_lists
from neuropy.utils.mixins.indexing_helpers import UnpackableMixin

from attrs import make_class, asdict, astuple, define, field, Factory  # Import the attrs library
import ipywidgets as widgets
from IPython.display import display
from copy import deepcopy
from IPython import get_ipython  # Import get_ipython to access the user namespace

# Testing:


_bak_rcParams = mpl.rcParams.copy()
# mpl.rcParams['toolbar'] = 'None' # disable toolbars
# %matplotlib qt




from copy import deepcopy
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from neuropy.utils.mixins.indexing_helpers import get_dict_subset


# ==================================================================================================================================================================================================================================================================================== #
# 2026-10-20 - OpenAI Figure 4 Stats                                                                                                                                                                                                                                                   #
# ==================================================================================================================================================================================================================================================================================== #

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import statsmodels.api as sm

@metadata_attributes(short_name=None, tags=['stats', 'fig4', 'hierarchical-bootstrap', 'bootstrap', 'permutation', 'sampling'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-20 10:12', related_items=[])
class PrePostDeltaDistributionComparison:
    """Compare pre/post-delta P_Short distributions using hierarchical resampling.

    Pass `sampling_unit` to the constructor. Cluster modes ignore parent-epoch identity.

    - `'block'`: moving blocks of laps within each pre/post run (default). Time bins
      within a lap are never resampled independently.
    - `'animal'` / `'session'`: copy each session's pre and post values intact. Do not
      group by lap or parent epoch. The optional p-value is a sign-flip test of the
      signed mean difference, not Wasserstein distance. Session-level tests treat
      sessions as exchangeable and therefore ignore within-animal dependence.


    #TODO 2026-10-27 19:32: - [ ] NOTE: IMPORTANT!! Much of this class isn't used because we decided against doing block-level shuffles

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PrePostDeltaDistributionComparison


    """

    def __init__(self, filtered_all_sessions_laps_time_bin_df, value_col='P_Short',
                 condition_col='pre_post_delta_category', animal_col='animal',
                 session_col='session_name', lap_col='parent_epoch_id',
                 lap_order_col='parent_epoch_start_t',
                 pre_label='pre-delta', post_label='post-delta',
                 block_length_laps=5, n_bootstrap=1000, random_state=None,
                 sampling_unit='block'):

        valid_sampling_units = ('block', 'animal', 'session')
        if sampling_unit not in valid_sampling_units:
            raise ValueError(
                f'sampling_unit must be one of {valid_sampling_units}, got {sampling_unit!r}.'
            )
        if n_bootstrap < 1:
            raise ValueError('n_bootstrap must be >= 1.')
        if sampling_unit == 'block' and block_length_laps < 1:
            raise ValueError('block_length_laps must be >= 1.')

        self.value_col = value_col
        self.condition_col = condition_col
        self.animal_col = animal_col
        self.session_col = session_col
        self.lap_col = lap_col
        self.lap_order_col = lap_order_col
        self.group_values = (pre_label, post_label)
        self.block_length_laps = int(block_length_laps)
        self.n_bootstrap = int(n_bootstrap)
        self.sampling_unit = sampling_unit
        self.rng = np.random.default_rng(random_state)

        required_cols = [value_col, condition_col, animal_col, session_col]
        if sampling_unit == 'block':
            required_cols = required_cols + [lap_col, lap_order_col]
        missing_cols = set(required_cols).difference(filtered_all_sessions_laps_time_bin_df.columns)
        if missing_cols:
            raise ValueError(f'Missing required columns: {sorted(missing_cols)}')

        data = filtered_all_sessions_laps_time_bin_df.loc[
            filtered_all_sessions_laps_time_bin_df[condition_col].isin(self.group_values),
            required_cols
        ].dropna(subset=required_cols).copy().reset_index(drop=True)

        if data.empty or set(data[condition_col].unique()) != set(self.group_values):
            raise ValueError(f'Data must contain both groups: {self.group_values}')

        data[value_col] = pd.to_numeric(data[value_col], errors='raise')

        self._y = data[value_col].to_numpy(dtype=float)
        if not np.isfinite(self._y).all() or np.any((self._y < 0) | (self._y > 1)):
            raise ValueError(f'{value_col} must contain finite probabilities in [0, 1].')

        data['_condition_code'] = (data[condition_col] == post_label).astype(np.int8)
        data['_animal_code'], self._animals = pd.factorize(data[animal_col], sort=False)
        self._data = data
        self._row_condition_code = data['_condition_code'].to_numpy()
        self._animal_code_rows = data['_animal_code'].to_numpy()
        self._n_animals = len(self._animals)

        for animal_code in range(self._n_animals):
            animal_mask = self._animal_code_rows == animal_code
            if set(self._row_condition_code[animal_mask]) != {0, 1}:
                raise ValueError(
                    f'Animal {self._animals[animal_code]!r} lacks one condition.'
                )
        ## END for animal_code in range(self._n_animals)....

        self._lap_table = None
        self._values_by_lap = None
        self._positions_by_lap = None
        self._runs = None
        self._values_by_session = None
        self._sessions_by_profile = {}

        if sampling_unit == 'block':
            self._build_block_lap_structures(data, animal_col, session_col, lap_col, lap_order_col)
        else:
            self._build_cluster_session_structures(data, session_col)


    def _build_block_lap_structures(self, data, animal_col, session_col, lap_col, lap_order_col):
        """Group rows into laps and chronological pre/post runs for moving-block resampling."""
        data[lap_order_col] = pd.to_numeric(data[lap_order_col], errors='raise')
        data['_lap_id'] = data.groupby(
            [animal_col, session_col, lap_col], sort=False, dropna=False
        ).ngroup().astype(np.int64)

        # Verify that each lap has exactly one condition and one chronological position.
        self._lap_table = data.groupby('_lap_id', sort=False).agg(
            _animal_code=('_animal_code', 'first'),
            _session=(session_col, 'first'),
            _condition_code=('_condition_code', 'first'),
            _n_conditions=('_condition_code', 'nunique'),
            _lap_order=(lap_order_col, 'first'),
            _n_lap_order_values=(lap_order_col, 'nunique'),
        ).reset_index()

        if self._lap_table['_n_conditions'].ne(1).any():
            raise ValueError('At least one lap contains both pre- and post-delta labels.')
        if self._lap_table['_n_lap_order_values'].ne(1).any():
            raise ValueError(f'{lap_order_col} must be constant within each lap.')

        # Cache every complete lap; no time bins are independently resampled.
        self._values_by_lap = {}
        self._positions_by_lap = {}

        for lap_id, positions in data.groupby('_lap_id', sort=False).indices.items():
            positions = np.asarray(positions, dtype=np.int64)
            self._values_by_lap[int(lap_id)] = self._y[positions]
            self._positions_by_lap[int(lap_id)] = positions
        ## END for lap_id, positions in data.groupby('_lap_id', sort=False).indices.items()....

        # Organize chronologically contiguous runs of laps by animal and session.
        # Separate runs ensure bootstrap blocks never cross a pre/post boundary.
        self._runs = {}

        for (animal_code, session), session_laps in self._lap_table.groupby(
                ['_animal_code', '_session'], sort=False):

            session_laps = session_laps.sort_values('_lap_order', kind='stable')
            runs = []
            current_condition = None
            current_lap_ids = []

            for lap_id, condition in session_laps[
                    ['_lap_id', '_condition_code']].itertuples(index=False, name=None):

                condition = int(condition)

                if current_condition is None or condition != current_condition:
                    if current_lap_ids:
                        runs.append({
                            'condition': current_condition,
                            'lap_ids': current_lap_ids,
                        })
                    current_condition = condition
                    current_lap_ids = []

                current_lap_ids.append(int(lap_id))
            ## END for lap_id, condition in session_laps[['_lap_id', '_condition_code']].itertuples....

            if current_lap_ids:
                runs.append({'condition': current_condition, 'lap_ids': current_lap_ids})

            self._runs.setdefault(int(animal_code), {})[session] = runs
        ## END for (animal_code, session), session_laps in self._lap_table.groupby....

        for animal_code, sessions in self._runs.items():
            profiles = {}
            for session, runs in sessions.items():
                profile = tuple(sorted({run['condition'] for run in runs}))
                profiles.setdefault(profile, []).append(session)
            ## END for session, runs in sessions.items()....
            self._sessions_by_profile[animal_code] = profiles
        ## END for animal_code, sessions in self._runs.items()....


    def _build_cluster_session_structures(self, data, session_col):
        """Cache pre/post values per session without grouping by parent epoch."""
        self._values_by_session = {}

        for (animal_code, session), idx in data.groupby(
                ['_animal_code', session_col], sort=False, dropna=False
        ).indices.items():
            idx = np.asarray(idx, dtype=np.int64)
            cond = self._row_condition_code[idx]
            y = self._y[idx]
            values = {}
            if np.any(cond == 0):
                values[0] = y[cond == 0]
            if np.any(cond == 1):
                values[1] = y[cond == 1]
            self._values_by_session.setdefault(int(animal_code), {})[session] = values
        ## END for (animal_code, session), idx in data.groupby....

        for animal_code, sessions in self._values_by_session.items():
            profiles = {}
            for session, values in sessions.items():
                profile = tuple(sorted(values.keys()))
                profiles.setdefault(profile, []).append(session)
            ## END for session, values in sessions.items()....
            self._sessions_by_profile[animal_code] = profiles
        ## END for animal_code, sessions in self._values_by_session.items()....


    def _distance_for_labels(self, row_labels, row_mask=None):
        """Wasserstein-1 distance, weighting each animal equally."""
        if row_mask is None:
            row_mask = np.ones(len(self._y), dtype=bool)

        pre_mask = row_mask & (row_labels == 0)
        post_mask = row_mask & (row_labels == 1)

        if not pre_mask.any() or not post_mask.any():
            return np.nan

        pre_codes = self._animal_code_rows[pre_mask]
        post_codes = self._animal_code_rows[post_mask]
        pre_counts = np.bincount(pre_codes, minlength=self._n_animals)
        post_counts = np.bincount(post_codes, minlength=self._n_animals)

        if np.any(pre_counts == 0) or np.any(post_counts == 0):
            return np.nan

        return float(wasserstein_distance(
            self._y[pre_mask], self._y[post_mask],
            u_weights=1.0 / pre_counts[pre_codes],
            v_weights=1.0 / post_counts[post_codes],
        ))


    def _moving_block_resample_lap_ids(self, lap_ids):
        """Resample overlapping blocks of adjacent laps within one uninterrupted run."""
        n_laps = len(lap_ids)
        block_length = min(self.block_length_laps, n_laps)

        sampled_lap_ids = []
        while len(sampled_lap_ids) < n_laps:
            start = self.rng.integers(0, n_laps - block_length + 1)
            sampled_lap_ids.extend(lap_ids[start:start + block_length])

        return sampled_lap_ids[:n_laps]

    @staticmethod
    def _distance_for_bootstrap_samples(pre_by_animal, post_by_animal):
        """Wasserstein-1 distance with equal total weight per resampled animal."""
        pre = np.concatenate(pre_by_animal)
        post = np.concatenate(post_by_animal)

        pre_weights = np.concatenate([
            np.full(values.size, 1.0 / values.size) for values in pre_by_animal
        ])
        post_weights = np.concatenate([
            np.full(values.size, 1.0 / values.size) for values in post_by_animal
        ])

        return float(wasserstein_distance(
            pre, post, u_weights=pre_weights, v_weights=post_weights
        ))


    def _hierarchical_bootstrap(self, resample_within_run=True, mean_difference_unit=None):
        """Resample animals, sessions within animals, and optionally contiguous lap blocks.

        resample_within_run True draws overlapping lap blocks within each run. False copies
        each session's cached pre/post values. mean_difference_unit `'animal'` or
        `'session'` also returns a 95% CI for the mean of that unit's signed mean differences.
        """
        bootstrap_distances = []
        bootstrap_mean_diffs = []

        for _ in range(self.n_bootstrap):
            sampled_animals = self.rng.choice(
                self._n_animals, size=self._n_animals, replace=True
            )
            pre_by_animal, post_by_animal = [], []
            animal_mean_diffs = []
            session_mean_diffs = []

            for animal_code in sampled_animals:
                animal_code = int(animal_code)
                animal_values = {0: [], 1: []}
                sampled_sessions = []

                for sessions in self._sessions_by_profile[animal_code].values():
                    sampled_sessions.extend(
                        self.rng.choice(sessions, size=len(sessions), replace=True).tolist()
                    )
                ## END for sessions in self._sessions_by_profile[animal_code].values()....

                for session in sampled_sessions:
                    if resample_within_run:
                        session_values = {0: [], 1: []}
                        for run in self._runs[animal_code][session]:
                            sampled_lap_ids = self._moving_block_resample_lap_ids(
                                run['lap_ids']
                            )
                            run_values = [
                                self._values_by_lap[lap_id] for lap_id in sampled_lap_ids
                            ]
                            session_values[run['condition']].extend(run_values)
                            animal_values[run['condition']].extend(run_values)
                        ## END for run in self._runs[animal_code][session]....

                        if (
                            mean_difference_unit == 'session'
                            and session_values[0]
                            and session_values[1]
                        ):
                            pre_session = np.concatenate(session_values[0])
                            post_session = np.concatenate(session_values[1])
                            session_mean_diffs.append(
                                post_session.mean() - pre_session.mean()
                            )
                    else:
                        cached = self._values_by_session[animal_code][session]
                        for condition, values in cached.items():
                            animal_values[condition].append(values)
                        ## END for condition, values in cached.items()....

                        if (
                            mean_difference_unit == 'session'
                            and 0 in cached and 1 in cached
                            and cached[0].size and cached[1].size
                        ):
                            session_mean_diffs.append(
                                cached[1].mean() - cached[0].mean()
                            )
                ## END for session in sampled_sessions....

                pre_concat = np.concatenate(animal_values[0])
                post_concat = np.concatenate(animal_values[1])
                pre_by_animal.append(pre_concat)
                post_by_animal.append(post_concat)
                if mean_difference_unit == 'animal':
                    animal_mean_diffs.append(post_concat.mean() - pre_concat.mean())
            ## END for animal_code in sampled_animals....

            bootstrap_distances.append(
                self._distance_for_bootstrap_samples(pre_by_animal, post_by_animal)
            )
            if mean_difference_unit == 'animal' and animal_mean_diffs:
                bootstrap_mean_diffs.append(float(np.mean(animal_mean_diffs)))
            elif mean_difference_unit == 'session' and session_mean_diffs:
                bootstrap_mean_diffs.append(float(np.mean(session_mean_diffs)))
        ## END for _ in range(self.n_bootstrap)....

        wasserstein_ci = tuple(np.quantile(bootstrap_distances, [0.025, 0.975]))
        if mean_difference_unit is None or not bootstrap_mean_diffs:
            mean_difference_ci = None
        else:
            mean_difference_ci = tuple(
                np.quantile(bootstrap_mean_diffs, [0.025, 0.975])
            )

        return {
            'wasserstein_1_bootstrap_95ci': wasserstein_ci,
            'mean_difference_bootstrap_95ci': mean_difference_ci,
        }


    @staticmethod
    def _effect_statistics(pre, post):
        """Mean, median, and Wasserstein summaries for one paired pre/post sample."""
        return {
            'n_bins_pre': int(pre.size),
            'n_bins_post': int(post.size),
            'mean_pre': float(pre.mean()),
            'mean_post': float(post.mean()),
            'mean_difference_post_minus_pre': float(post.mean() - pre.mean()),
            'median_pre': float(np.median(pre)),
            'median_post': float(np.median(post)),
            'median_difference_post_minus_pre': float(np.median(post) - np.median(pre)),
            'wasserstein_1': float(wasserstein_distance(pre, post)),
        }


    def _session_effect_rows(self):
        """One paired-effect row per session that contains both pre and post bins."""
        rows = []

        for animal_code, sessions in self._values_by_session.items():
            for session, values in sessions.items():
                if 0 not in values or 1 not in values:
                    continue
                pre = values[0]
                post = values[1]
                if pre.size == 0 or post.size == 0:
                    continue
                row = {
                    'animal': self._animals[int(animal_code)],
                    'session': session,
                }
                row.update(self._effect_statistics(pre, post))
                rows.append(row)
            ## END for session, values in sessions.items()....
        ## END for animal_code, sessions in self._values_by_session.items()....

        return pd.DataFrame(rows)


    def _paired_sign_permutation_test(self, effects, n_permutations, sampling_unit):
        """Two-sided sign-flip test that the mean of paired unit effects is zero.

        Does not assume lap-block exchangeability. Session-level tests treat sessions as
        exchangeable and therefore ignore within-animal dependence.
        """
        if n_permutations < 1:
            raise ValueError('n_permutations must be >= 1.')

        effects = np.asarray(effects, dtype=float)
        n_units = int(effects.size)
        if n_units < 1:
            return {
                'p_value': None,
                'observed_mean_difference_post_minus_pre': None,
                'null_mean_differences': None,
                'effect': 'mean_difference_post_minus_pre',
                'sampling_unit': sampling_unit,
                'n_units': n_units,
                'exact': False,
                'method': None,
                'status': 'Not testable: no paired units with both conditions.',
            }

        observed = float(effects.mean())
        abs_observed = abs(observed)
        n_exact = 1 << n_units
        use_exact = (n_units <= 20) and (n_exact <= n_permutations)

        if use_exact:
            sign_bits = (
                (np.arange(n_exact, dtype=np.uint32)[:, None] >> np.arange(n_units)) & 1
            )
            signs = 1.0 - 2.0 * sign_bits.astype(float)
            null_means = signs @ effects / n_units
            p_value = float(np.count_nonzero(np.abs(null_means) >= abs_observed) / n_exact)
            method = 'exact'
        else:
            null_means = np.empty(n_permutations, dtype=float)
            sign_choices = np.array([1.0, -1.0])

            for i in range(n_permutations):
                signs = self.rng.choice(sign_choices, size=n_units)
                null_means[i] = float(np.mean(signs * effects))
            ## END for i in range(n_permutations)....

            p_value = (
                1 + np.count_nonzero(np.abs(null_means) >= abs_observed)
            ) / (n_permutations + 1)
            method = 'monte_carlo'

        if sampling_unit == 'session':
            unit_status = (
                'Session-level tests treat sessions as exchangeable and ignore '
                'within-animal dependence.'
            )
        else:
            unit_status = (
                'Animal-level tests treat animals as independent experimental units.'
            )

        return {
            'p_value': float(p_value),
            'observed_mean_difference_post_minus_pre': observed,
            'null_mean_differences': null_means,
            'effect': 'mean_difference_post_minus_pre',
            'sampling_unit': sampling_unit,
            'n_units': n_units,
            'exact': bool(use_exact),
            'method': method,
            'status': (
                'ok; paired cluster sign-flip test of the mean difference. '
                'Does not assume lap-block exchangeability. ' + unit_status
            ),
        }


    def _block_permutation_test(self, n_permutations):
        """Approximate null test by permuting labels of equal-length lap blocks within sessions.

        Requires approximate stationarity and exchangeability of sufficiently separated
        blocks under the null. Incomplete blocks and sessions without both conditions
        represented by complete blocks are omitted from this test.
        """
        if n_permutations < 1:
            raise ValueError('n_permutations must be >= 1.')

        block_length = self.block_length_laps
        blocks_by_session = {}

        for animal_code, sessions in self._runs.items():
            for session, runs in sessions.items():
                blocks = []

                for run in runs:
                    lap_ids = run['lap_ids']

                    # Use complete, non-overlapping blocks so all permuted blocks
                    # contain the same number of consecutive laps.
                    for start in range(0, len(lap_ids) - block_length + 1, block_length):
                        block_lap_ids = lap_ids[start:start + block_length]
                        positions = np.concatenate([
                            self._positions_by_lap[lap_id] for lap_id in block_lap_ids
                        ])

                        blocks.append({
                            'condition': run['condition'],
                            'lap_ids': block_lap_ids,
                            'positions': positions,
                        })

                if {block['condition'] for block in blocks} == {0, 1}:
                    blocks_by_session[(animal_code, session)] = blocks

        eligible_animals = {animal_code for animal_code, _ in blocks_by_session}
        if eligible_animals != set(range(self._n_animals)):
            missing_animals = [
                self._animals[a] for a in range(self._n_animals)
                if a not in eligible_animals
            ]
            return {
                'p_value': None,
                'null_distances': None,
                'status': (
                    'Not testable with this block length: at least one animal has no '
                    'session containing complete blocks from both conditions. '
                    f'Affected animals: {missing_animals}'
                ),
                'block_length_laps': block_length,
                'n_sessions_used': len(blocks_by_session),
            }

        row_mask = np.zeros(len(self._y), dtype=bool)
        observed_labels = np.full(len(self._y), -1, dtype=np.int8)
        used_laps = set()
        n_blocks_pre = 0
        n_blocks_post = 0

        for blocks in blocks_by_session.values():
            for block in blocks:
                positions = block['positions']
                row_mask[positions] = True
                observed_labels[positions] = block['condition']
                used_laps.update(block['lap_ids'])

                if block['condition'] == 0:
                    n_blocks_pre += 1
                else:
                    n_blocks_post += 1

        observed_distance = self._distance_for_labels(observed_labels, row_mask)
        null_distances = np.empty(n_permutations, dtype=float)

        for i in range(n_permutations):
            permuted_labels = np.full(len(self._y), -1, dtype=np.int8)

            for blocks in blocks_by_session.values():
                permuted_conditions = self.rng.permutation(
                    np.array([block['condition'] for block in blocks], dtype=np.int8)
                )

                for block, condition in zip(blocks, permuted_conditions):
                    permuted_labels[block['positions']] = condition

            null_distances[i] = self._distance_for_labels(permuted_labels, row_mask)

        p_value = (
            1 + np.count_nonzero(null_distances >= observed_distance)
        ) / (n_permutations + 1)

        return {
            'p_value': float(p_value),
            'observed_wasserstein_1': observed_distance,
            'null_distances': null_distances,
            'status': 'ok; approximate block-exchangeability test',
            'block_length_laps': block_length,
            'n_sessions_used': len(blocks_by_session),
            'n_laps_used': len(used_laps),
            'n_bins_used': int(row_mask.sum()),
            'n_blocks_pre': n_blocks_pre,
            'n_blocks_post': n_blocks_post,
        }


    def run(self, perform_block_permutation_test=False, n_permutations=2000, sampling_unit=None, perform_ordinal_mode_model: bool = True, ordinal_mode_kwargs: Optional[Dict] = None) -> Dict:
        """Compute the effect, bootstrap interval, and optional permutation test.

        Uses `sampling_unit` from the constructor. Passing it here is optional and must
        match. `'block'` resamples moving lap blocks and, if requested, permutes
        within-session lap-block labels of Wasserstein-1. `'animal'` and `'session'` copy
        each session's pre/post values intact and, if requested, run a paired sign-flip
        test of the signed mean difference. Cluster modes do not group by parent epoch.
        Session-level tests treat sessions as exchangeable and therefore ignore
        within-animal dependence.

        When `perform_ordinal_mode_model` is True (default), also fits a session-level
        proportional-odds model of low/mid/high `P_Short` and stores it under
        `ordinal_mode_model`.
        """
        if sampling_unit is None:
            sampling_unit = self.sampling_unit
        elif sampling_unit != self.sampling_unit:
            raise ValueError(
                f'sampling_unit={sampling_unit!r} does not match constructor '
                f'sampling_unit={self.sampling_unit!r}. Pass sampling_unit to __init__.'
            )

        labels = self._row_condition_code
        resample_within_run = sampling_unit == 'block'
        mean_difference_unit = None if sampling_unit == 'block' else sampling_unit
        bootstrap_result = self._hierarchical_bootstrap(
            resample_within_run=resample_within_run,
            mean_difference_unit=mean_difference_unit,
        )
        animal_rows = []

        for animal_code, animal in enumerate(self._animals):
            animal_mask = self._animal_code_rows == animal_code
            pre = self._y[animal_mask & (labels == 0)]
            post = self._y[animal_mask & (labels == 1)]
            row = {'animal': animal}
            row.update(self._effect_statistics(pre, post))
            animal_rows.append(row)
        ## END for animal_code, animal in enumerate(self._animals)....

        animal_summary = pd.DataFrame(animal_rows)
        session_summary = (
            self._session_effect_rows() if sampling_unit == 'session' else None
        )

        permutation_results = None
        p_value_effect = None
        if perform_block_permutation_test:
            if sampling_unit == 'block':
                permutation_results = self._block_permutation_test(n_permutations)
                p_value_effect = 'wasserstein_1'
            elif sampling_unit == 'animal':
                permutation_results = self._paired_sign_permutation_test(
                    animal_summary['mean_difference_post_minus_pre'].to_numpy(),
                    n_permutations,
                    sampling_unit='animal',
                )
                p_value_effect = 'mean_difference_post_minus_pre'
            else:
                permutation_results = self._paired_sign_permutation_test(
                    session_summary['mean_difference_post_minus_pre'].to_numpy()
                    if not session_summary.empty else np.array([], dtype=float),
                    n_permutations,
                    sampling_unit='session',
                )
                p_value_effect = 'mean_difference_post_minus_pre'

        bin_counts = self._data['_condition_code'].value_counts()
        n_sessions = int(
            self._data[['_animal_code', self.session_col]].drop_duplicates().shape[0]
        )
        if self._lap_table is None:
            n_laps_pre = None
            n_laps_post = None
        else:
            lap_counts = self._lap_table['_condition_code'].value_counts()
            n_laps_pre = int(lap_counts.get(0, 0))
            n_laps_post = int(lap_counts.get(1, 0))
        results = {
            'comparison': f'{self.group_values[0]} vs {self.group_values[1]}',
            'sampling_unit': sampling_unit,
            'p_value_effect': p_value_effect,
            'wasserstein_1': self._distance_for_labels(labels),
            'wasserstein_1_bootstrap_95ci': bootstrap_result['wasserstein_1_bootstrap_95ci'],
            'mean_difference_post_minus_pre': float(
                animal_summary['mean_difference_post_minus_pre'].mean()
            ),
            'median_difference_post_minus_pre': float(
                animal_summary['median_difference_post_minus_pre'].mean()
            ),
            'p_value': (
                permutation_results['p_value'] if permutation_results is not None else None
            ),
            'permutation_results': permutation_results,
            'n_animals': self._n_animals,
            'n_sessions': n_sessions,
            'n_laps_pre': n_laps_pre,
            'n_laps_post': n_laps_post,
            'n_bins_pre': int(bin_counts.get(0, 0)),
            'n_bins_post': int(bin_counts.get(1, 0)),
            'block_length_laps': self.block_length_laps,
            'animal_summary': animal_summary,
        }
        if sampling_unit != 'block':
            results['mean_difference_bootstrap_95ci'] = bootstrap_result[
                'mean_difference_bootstrap_95ci'
            ]
        if session_summary is not None:
            results['session_summary'] = session_summary

        if perform_ordinal_mode_model:
            _ordinal_kwargs = dict(value_col=self.value_col, condition_col=self.condition_col, animal_col=self.animal_col, session_col=self.session_col, pre_label=self.group_values[0], post_label=self.group_values[1], n_bootstrap=self.n_bootstrap, n_permutations=n_permutations, random_state=int(self.rng.integers(0, 2**31 - 1)))
            _ordinal_kwargs.update(ordinal_mode_kwargs or {})
            try:
                results['ordinal_mode_model'] = self.compute_ordinal_mode_shift(self._data, **_ordinal_kwargs)
            except (ValueError, np.linalg.LinAlgError) as e:
                results['ordinal_mode_model'] = {'status': f'failed: {e}'}

        return results


    # ==================================================================================================================================================================================================================================================================================== #
    # 2026-10-27 - Relevant Only to 'block' resampling, which IS NOT DONE because blocks of laps are autocorrelated!                                                                                                                                                                       #
    # ==================================================================================================================================================================================================================================================================================== #
    @function_attributes(short_name=None, tags=['_DEPRICATED', 'block-only', 'NOT_USED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-27 19:32', related_items=[])
    @classmethod
    def estimate_block_length_lap_t_bins_from_autocorrelation(cls, filtered_all_sessions_laps_time_bin_df, value_col='P_Short', condition_col='pre_post_delta_category', animal_col='animal', session_col='session_name', lap_col='parent_epoch_id', lap_order_col='parent_epoch_start_t', pre_label='pre-delta', post_label='post-delta', threshold=0.5, acf_threshold=0.1, consecutive_lags=3, min_run_laps=12, max_lag=None, recommendation_quantile=0.90, plot=False):
        """Estimate a candidate moving-block length from lap-summary autocorrelation.

        The estimate is in laps, compatible with block_length_laps in
        PrePostDeltaDistributionComparison. This is a diagnostic heuristic,
        not a guarantee that a permutation test is valid.
        """
        import numpy as np
        import pandas as pd

        required_cols = [
            value_col, condition_col, animal_col, session_col,
            lap_col, lap_order_col,
        ]
        missing_cols = set(required_cols).difference(
            filtered_all_sessions_laps_time_bin_df.columns
        )
        if missing_cols:
            raise ValueError(f'Missing required columns: {sorted(missing_cols)}')
        if acf_threshold <= 0 or consecutive_lags < 1 or min_run_laps < 4:
            raise ValueError('Invalid ACF threshold or lag/run settings.')
        if not 0 < recommendation_quantile <= 1:
            raise ValueError('recommendation_quantile must be in (0, 1].')

        data = filtered_all_sessions_laps_time_bin_df[required_cols].copy()
        data[value_col] = pd.to_numeric(data[value_col], errors='raise')
        data[lap_order_col] = pd.to_numeric(data[lap_order_col], errors='raise')
        data = data.dropna(subset=[
            animal_col, session_col, lap_col, lap_order_col,
        ])

        group_cols = [animal_col, session_col, lap_col]

        def _single_value(series):
            if series.nunique(dropna=False) != 1:
                raise ValueError(
                    'Each lap must have exactly one condition and one chronological '
                    'start time. Check the lap and condition columns.'
                )
            return series.iloc[0]

        lap_table = (
            data.groupby(group_cols, sort=False, dropna=False)
            .agg(
                condition=(condition_col, _single_value),
                lap_order=(lap_order_col, _single_value),
            )
            .reset_index()
        )

        # Summarize each lap, preserving all time bins for the original distribution
        # analysis while estimating serial dependence at the lap level.
        feature_names = [
            'lap_mean', 'lap_median', 'lap_q25', 'lap_q75', 'lap_fraction_above_threshold',
        ]
        value_data = data.loc[
            data[condition_col].isin([pre_label, post_label]) &
            data[value_col].notna()
        ].copy()

        summary = (
            value_data.groupby(group_cols, sort=False, dropna=False)[value_col]
            .agg(
                lap_mean='mean',
                lap_median='median',
                lap_q25=lambda x: x.quantile(0.25),
                lap_q75=lambda x: x.quantile(0.75),
                lap_fraction_above_threshold=lambda x: (x > threshold).mean(),
            )
            .reset_index()
        )

        lap_table = lap_table.merge(
            summary, on=group_cols, how='left', validate='one_to_one'
        )

        # Split chronologically ordered laps into uninterrupted runs. Other
        # condition labels and pre/post transitions terminate a run.
        runs = []
        for (animal, session), session_laps in lap_table.groupby(
            [animal_col, session_col], sort=False, dropna=False
        ):
            session_laps = session_laps.sort_values('lap_order', kind='stable')

            current_condition = None
            current_run = []

            for row in session_laps.to_dict('records'):
                condition = row['condition']
                eligible = pd.notna(condition) and condition in (pre_label, post_label)

                if current_run and (
                    not eligible or condition != current_condition
                ):
                    runs.append({
                        'animal': animal,
                        'session': session,
                        'condition': current_condition,
                        'rows': current_run,
                    })
                    current_run = []
                    current_condition = None

                if eligible:
                    if not current_run:
                        current_condition = condition
                    current_run.append(row)

            if current_run:
                runs.append({
                    'animal': animal,
                    'session': session,
                    'condition': current_condition,
                    'rows': current_run,
                })

        acf_records = []
        run_records = []

        for run_id, run in enumerate(runs):
            n_laps = len(run['rows'])
            record = {
                'run_id': run_id,
                'animal': run['animal'],
                'session': run['session'],
                'condition': run['condition'],
                'n_laps': n_laps,
                'decorrelation_lag_laps': np.nan,
                'censored': False,
                'status': 'ok',
            }

            if n_laps < min_run_laps:
                record['status'] = 'too_short'
                run_records.append(record)
                continue

            run_block_lengths = []
            run_has_censored_feature = False

            current_max_lag = (
                min(25, n_laps // 3)
                if max_lag is None
                else min(int(max_lag), n_laps // 3)
            )

            if current_max_lag < consecutive_lags:
                record['status'] = 'too_short_for_acf'
                run_records.append(record)
                continue

            for feature in feature_names:
                x = np.asarray(
                    [row[feature] for row in run['rows']], dtype=float
                )

                if not np.isfinite(x).all() or np.var(x) <= np.finfo(float).eps:
                    continue

                x = x - x.mean()
                denominator = np.dot(x, x)
                acf_values = {}

                for lag in range(1, current_max_lag + 1):
                    # Biased sample ACF; the same denominator is used at each lag.
                    rho = float(np.dot(x[:-lag], x[lag:]) / denominator)
                    acf_values[lag] = rho

                    acf_records.append({
                        'run_id': run_id,
                        'animal': run['animal'],
                        'session': run['session'],
                        'condition': run['condition'],
                        'feature': feature,
                        'lag': lag,
                        'acf': rho,
                    })

                decorrelation_lag = None
                for start_lag in range(
                    1, current_max_lag - consecutive_lags + 2
                ):
                    if all(
                        abs(acf_values[lag]) <= acf_threshold
                        for lag in range(start_lag, start_lag + consecutive_lags)
                    ):
                        decorrelation_lag = start_lag
                        break

                if decorrelation_lag is None:
                    # Dependence did not fall below the selected threshold
                    # within the range examined.
                    run_has_censored_feature = True
                    run_block_lengths.append(current_max_lag + 1)
                else:
                    run_block_lengths.append(decorrelation_lag)

            if not run_block_lengths:
                record['status'] = 'no_variable_features'
            else:
                record['decorrelation_lag_laps'] = max(run_block_lengths)
                record['censored'] = run_has_censored_feature
                if run_has_censored_feature:
                    record['status'] = 'dependence_not_resolved'

            run_records.append(record)

        acf_df = pd.DataFrame(acf_records)
        run_diagnostics = pd.DataFrame(run_records)

        informative_runs = run_diagnostics.loc[
            run_diagnostics['decorrelation_lag_laps'].notna()
        ]

        provisional_block_length_laps = None
        suggested_block_length_laps = None
        censored_runs = 0

        if not informative_runs.empty:
            provisional_block_length_laps = max(
                1,
                int(np.ceil(informative_runs['decorrelation_lag_laps'].quantile(
                    recommendation_quantile, interpolation='higher'
                ))),
            )
            censored_runs = int(informative_runs['censored'].sum())

            # Do not claim a finite recommendation if any informative run
            # remains correlated through its maximum examined lag.
            if censored_runs == 0:
                suggested_block_length_laps = provisional_block_length_laps

        if not acf_df.empty:
            acf_df['abs_acf'] = acf_df['acf'].abs()
            acf_summary = (
                acf_df.groupby('lag')['abs_acf']
                .agg(
                    median_abs_acf='median',
                    q90_abs_acf=lambda x: x.quantile(0.90),
                    n_series='size',
                )
                .reset_index()
            )
        else:
            acf_summary = pd.DataFrame(
                columns=['lag', 'median_abs_acf', 'q90_abs_acf', 'n_series']
            )

        fig = ax = None
        if plot and not acf_summary.empty:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                acf_summary['lag'], acf_summary['median_abs_acf'],
                label='Median absolute ACF',
            )
            ax.plot(
                acf_summary['lag'], acf_summary['q90_abs_acf'],
                label='90th percentile absolute ACF',
            )
            ax.axhline(
                acf_threshold, color='black', linestyle='--',
                label=f'Chosen ACF threshold ({acf_threshold:g})',
            )
            ax.set(
                xlabel='Lag (laps)',
                ylabel='Absolute autocorrelation',
                title='Autocorrelation of within-run lap summaries',
            )
            ax.legend()
            fig.tight_layout()

        return {
            'suggested_block_length_laps': suggested_block_length_laps,
            'provisional_block_length_laps': provisional_block_length_laps,
            'acf_threshold': acf_threshold,
            'consecutive_lags': consecutive_lags,
            'recommendation_quantile': recommendation_quantile,
            'n_runs': len(run_diagnostics),
            'n_informative_runs': len(informative_runs),
            'n_censored_runs': censored_runs,
            'run_diagnostics': run_diagnostics,
            'acf_by_run': acf_df,
            'acf_summary': acf_summary,
            'figure': fig,
            'axes': ax,
        }


    @function_attributes(short_name=None, tags=['_DEPRICATED', 'block-only', 'NOT_USED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-27 19:32', related_items=[])
    @classmethod
    def estimate_block_length_laps_from_autocorrelation(cls, filtered_all_sessions_laps_df, value_col='P_Short', condition_col='pre_post_delta_category', animal_col='animal', session_col='session_name', lap_col=None, lap_order_col=None, pre_label='pre-delta', post_label='post-delta', acf_threshold=0.1, consecutive_lags=3, min_run_laps=12, max_lag=None, recommendation_quantile=0.90, plot=False):
        """Recommend a moving-block length (in laps) from residual lap-to-lap autocorrelation.

        Within a lap, time bins are structurally dependent: they come from one continuous run.
        Consecutive laps are not. Every lap is separated from the next by a non-lap period
        in which the animal stops and receives reward. Across-lap dependence is therefore not
        implied by the task; it is an empirical question (slow behavioral state, strategy, or
        representational drift). `PrePostDeltaDistributionComparison` always resamples whole
        laps, never individual time bins. This function asks whether those laps still need to
        be grouped into longer moving blocks.

        The input must already be one row per lap. Within each animal/session, laps are sorted
        by `lap_order_col` and split into pre-delta or post-delta sequences. "Run" here means
        an ordered sequence of the same condition with no missing scores, not continuous
        running: adjacent laps in a run still have a reward/stop interval between them. A
        missing score, a non-pre/post label, or a pre/post transition ends the current run so
        the ACF is never computed across a condition change or a missing lap.

        For each run long enough to support an ACF, the biased sample autocorrelation of
        `value_col` is computed at successive lap lags. A run's decorrelation lag is the first
        lag at which `consecutive_lags` consecutive |ACF| values all fall at or below
        `acf_threshold`. If dependence is already gone at lag 1, iid resampling of laps
        (`block_length_laps=1`) is supported. If the series is still correlated at the largest
        lag examined, that run is marked censored.

        The recommendation is a high quantile (default 90th percentile) of those per-run
        decorrelation lags, in units of laps, for use as `block_length_laps`. A finite
        `suggested_block_length_laps` is returned only when no informative run is censored;
        otherwise only `provisional_block_length_laps` is filled. This is a diagnostic
        heuristic, not a guarantee that a subsequent permutation or bootstrap test is valid.

        Use `estimate_block_length_lap_t_bins_from_autocorrelation` instead when the table is
        one row per time bin and each lap must first be summarized.
        """

        import numpy as np
        import pandas as pd

        required_cols = [value_col, condition_col, animal_col, session_col]
        missing_cols = set(required_cols).difference(filtered_all_sessions_laps_df.columns)
        if missing_cols:
            raise ValueError(f'Missing required columns: {sorted(missing_cols)}')

        # Detect common lap ID and chronological-order columns when unspecified.
        if lap_col is None:
            lap_col = next((
                col for col in ['lap_idx', 'parent_epoch_id', 'epoch_id']
                if col in filtered_all_sessions_laps_df.columns
                and filtered_all_sessions_laps_df[col].notna().any()
            ), None)

        if lap_order_col is None:
            lap_order_col = next((
                col for col in ['lap_start_t', 'parent_epoch_start_t', 'epoch_start_t', 'start']
                if col in filtered_all_sessions_laps_df.columns
                and filtered_all_sessions_laps_df[col].notna().any()
            ), None)

        if lap_col is None or lap_order_col is None:
            raise ValueError(
                'Could not identify lap_col or lap_order_col. '
                'Specify both explicitly.'
            )

        data = filtered_all_sessions_laps_df[
            required_cols + [lap_col, lap_order_col]
        ].copy()

        data[value_col] = pd.to_numeric(data[value_col], errors='raise')
        data[lap_order_col] = pd.to_numeric(data[lap_order_col], errors='raise')

        data = data.dropna(subset=[
            animal_col, session_col, lap_col, lap_order_col,
        ])

        valid_values = data[value_col].dropna().to_numpy(dtype=float)
        if not np.isfinite(valid_values).all() or np.any(
            (valid_values < 0) | (valid_values > 1)
        ):
            raise ValueError(f'{value_col} must contain finite probabilities in [0, 1].')

        # This version expects exactly one row per lap.
        group_cols = [animal_col, session_col, lap_col]
        if data.duplicated(group_cols).any():
            raise ValueError(
                'Expected one row per lap. Duplicate animal/session/lap combinations found.'
            )

        # A row with a missing score or a different condition breaks a run;
        # do not silently join observations across that gap.
        runs = []
        for (animal, session), session_laps in data.groupby(
            [animal_col, session_col], sort=False
        ):
            session_laps = session_laps.sort_values(lap_order_col, kind='stable')

            current_condition = None
            current_rows = []

            for row in session_laps.to_dict('records'):
                condition = row[condition_col]
                score = row[value_col]

                eligible = (
                    pd.notna(condition)
                    and condition in (pre_label, post_label)
                    and pd.notna(score)
                    and np.isfinite(score)
                )

                if current_rows and (
                    not eligible or condition != current_condition
                ):
                    runs.append({
                        'animal': animal,
                        'session': session,
                        'condition': current_condition,
                        'rows': current_rows,
                    })
                    current_rows = []
                    current_condition = None

                if eligible:
                    if not current_rows:
                        current_condition = condition
                    current_rows.append(row)

            if current_rows:
                runs.append({
                    'animal': animal,
                    'session': session,
                    'condition': current_condition,
                    'rows': current_rows,
                })

        acf_records = []
        run_records = []

        for run_id, run in enumerate(runs):
            x = np.asarray(
                [row[value_col] for row in run['rows']], dtype=float
            )
            n_laps = len(x)

            record = {
                'run_id': run_id,
                'animal': run['animal'],
                'session': run['session'],
                'condition': run['condition'],
                'n_laps': n_laps,
                'decorrelation_lag_laps': np.nan,
                'censored': False,
                'status': 'ok',
            }

            if n_laps < min_run_laps:
                record['status'] = 'too_short'
                run_records.append(record)
                continue

            current_max_lag = (
                min(25, n_laps // 3)
                if max_lag is None
                else min(int(max_lag), n_laps // 3)
            )

            if current_max_lag < consecutive_lags:
                record['status'] = 'too_short_for_acf'
                run_records.append(record)
                continue

            x = x - x.mean()
            denominator = np.dot(x, x)

            if denominator <= np.finfo(float).eps:
                record['status'] = 'constant_series'
                run_records.append(record)
                continue

            acf_values = {}
            for lag in range(1, current_max_lag + 1):
                rho = float(np.dot(x[:-lag], x[lag:]) / denominator)
                acf_values[lag] = rho

                acf_records.append({
                    'run_id': run_id,
                    'animal': run['animal'],
                    'session': run['session'],
                    'condition': run['condition'],
                    'lag': lag,
                    'acf': rho,
                })

            # First lag followed by consecutive_lags values below the threshold.
            decorrelation_lag = None
            for start_lag in range(
                1, current_max_lag - consecutive_lags + 2
            ):
                if all(
                    abs(acf_values[lag]) <= acf_threshold
                    for lag in range(start_lag, start_lag + consecutive_lags)
                ):
                    decorrelation_lag = start_lag
                    break

            if decorrelation_lag is None:
                record['decorrelation_lag_laps'] = current_max_lag + 1
                record['censored'] = True
                record['status'] = 'dependence_not_resolved'
            else:
                record['decorrelation_lag_laps'] = decorrelation_lag

            run_records.append(record)

        acf_df = pd.DataFrame(acf_records)
        run_diagnostics = pd.DataFrame(run_records)

        informative_runs = run_diagnostics.loc[
            run_diagnostics['decorrelation_lag_laps'].notna()
        ]

        provisional_block_length_laps = None
        suggested_block_length_laps = None

        if not informative_runs.empty:
            provisional_block_length_laps = max(
                1,
                int(np.ceil(informative_runs['decorrelation_lag_laps'].quantile(
                    recommendation_quantile, interpolation='higher'
                ))),
            )

            if not informative_runs['censored'].any():
                suggested_block_length_laps = provisional_block_length_laps

        if not acf_df.empty:
            acf_df['abs_acf'] = acf_df['acf'].abs()
            acf_summary = acf_df.groupby('lag')['abs_acf'].agg(
                median_abs_acf='median',
                q90_abs_acf=lambda x: x.quantile(0.90),
                n_series='size',
            ).reset_index()
        else:
            acf_summary = pd.DataFrame(
                columns=['lag', 'median_abs_acf', 'q90_abs_acf', 'n_series']
            )

        fig = ax = None
        if plot and not acf_summary.empty:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                acf_summary['lag'], acf_summary['median_abs_acf'],
                label='Median absolute ACF',
            )
            ax.plot(
                acf_summary['lag'], acf_summary['q90_abs_acf'],
                label='90th percentile absolute ACF',
            )
            ax.axhline(
                acf_threshold, color='black', linestyle='--',
                label=f'ACF threshold = {acf_threshold:g}',
            )
            ax.set(
                xlabel='Lag (laps)',
                ylabel='Absolute autocorrelation',
                title='Autocorrelation of lap-level P_Short',
            )
            ax.legend()
            fig.tight_layout()

        return {
            'lap_col': lap_col,
            'lap_order_col': lap_order_col,
            'suggested_block_length_laps': suggested_block_length_laps,
            'provisional_block_length_laps': provisional_block_length_laps,
            'acf_threshold': acf_threshold,
            'consecutive_lags': consecutive_lags,
            'recommendation_quantile': recommendation_quantile,
            'n_runs': len(run_diagnostics),
            'n_informative_runs': len(informative_runs),
            'n_censored_runs': int(informative_runs['censored'].sum()),
            'run_diagnostics': run_diagnostics,
            'acf_by_run': acf_df,
            'acf_summary': acf_summary,
            'figure': fig,
            'axes': ax,
        }



    # ==================================================================================================================================================================================================================================================================================== #
    # 2026-10-26 - Interrupted Timeseries Analysis (ITA) Stats                                                                                                                                                                                                                             #
    # ==================================================================================================================================================================================================================================================================================== #
    ## INPUTS: pre_df, post_df
    @function_attributes(short_name=None, tags=['Figure 4/S6/S8 Stats', 'stats', 'figure4', 'interrupted-timeseries-analysis', 'timeseries'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-26 14:02', related_items=[])
    @classmethod
    def stats_interrupted_timeseries_analysis(cls, pre_df: pd.DataFrame, post_df: pd.DataFrame, should_plot: bool = True, variable_name: str = 'P_Short', time_variable_name: str = 'delta_aligned_start_t', debug_print: bool=True):
        """ Applies Interrupted Timeseries Analysis (ITA) to the pre/post delta data, treating delta as the intervention

        Lopez Bernal et al., “Interrupted Time Series Regression for the Evaluation of Public Health Interventions.”

        Usage:

            time_variable_name: str = 'delta_aligned_start_t'
            variable_name = df_filter.active_plot_variable_name  # usually 'P_Short'
            df = deepcopy(df_filter.active_plot_df).dropna(subset=[variable_name])
            pre_df  = df[df[time_variable_name] <= 0]
            post_df = df[df[time_variable_name] > 0]
            (its_model, its_tuple) = stats_interrupted_timeseries_analysis(pre_df, post_df, should_plot=True, 
                                        variable_name=variable_name, time_variable_name=time_variable_name,
                                        )
            intercept, pre_slope, level_change, slope_change, post_slope = its_tuple


        """
        # Combine pre- and post-intervention observations
        pre_df = pre_df.copy()
        post_df = post_df.copy()

        pre_df['intervention'] = 0
        post_df['intervention'] = 1

        its_df = pd.concat([pre_df, post_df], ignore_index=True)
        its_df = its_df.sort_values(time_variable_name)

        # Corrected ITS interaction: (T - T0) * intervention, where T0 = 0
        its_df['time_interaction'] = (its_df[time_variable_name] * its_df['intervention'])

        X = sm.add_constant(its_df[[time_variable_name, 'intervention', 'time_interaction']])

        # Fit the model
        its_model = sm.OLS(its_df[variable_name], X).fit()
        if debug_print:
            print(its_model.summary())

        intercept = its_model.params['const']
        pre_slope = its_model.params[time_variable_name]
        level_change = its_model.params['intervention']
        slope_change = its_model.params['time_interaction']

        post_slope = pre_slope + slope_change
        if debug_print:
            print(f"Pre-intervention slope: {pre_slope:.6f}")
            print(f"Immediate level change: {level_change:.6f}")
            print(f"Slope change: {slope_change:.6f}")
            print(f"Post-intervention slope: {post_slope:.6f}")

        outputs = (its_model, (intercept, pre_slope, level_change, slope_change, post_slope))

        if should_plot:
            import matplotlib.pyplot as plt

            fig = plt.figure(num='Interrupted Time Series stats model', clear=True)
            plt.scatter(pre_df[time_variable_name], pre_df[variable_name], label='Pre-intervention', alpha=0.6)
            plt.scatter(post_df[time_variable_name], post_df[variable_name], label='Post-intervention', alpha=0.6)

            # Plot each fitted segment separately
            pre_mask = its_df['intervention'] == 0
            post_mask = its_df['intervention'] == 1

            plt.plot(its_df.loc[pre_mask, time_variable_name], its_model.predict(X.loc[pre_mask]), color='red', label='Segmented regression')

            plt.plot(its_df.loc[post_mask, time_variable_name], its_model.predict(X.loc[post_mask]), color='red')

            plt.axvline(0.0, color='black', linestyle='--', label='Intervention')
            plt.xlabel(time_variable_name)
            plt.ylabel(variable_name)
            plt.legend()
            plt.show()

        return outputs


    @function_attributes(short_name=None, tags=['UNUSED', 'Figure 4/S6/S8 Stats', 'stats', 'figure4'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-26 15:35', related_items=[])
    @classmethod
    def linear_fit(cls, a_df: pd.DataFrame, variable_name: str = 'P_Short', time_variable_name: str = 'delta_aligned_start_t'):
        """ a simple linear fit to the pre/post period df separately.

        Usage:
            pre_df, pre_tuple = linear_fit(a_df=pre_df)
            post_df, post_tuple = linear_fit(a_df=post_df)

            pre_slope, pre_intercept, pre_r_value, pre_p_value, pre_std_err = pre_tuple
            post_slope, post_intercept, post_r_value, post_p_value, post_std_err = post_tuple

            pre_slope, post_slope


        """
        from scipy.stats import linregress
        x = a_df[time_variable_name]
        y = a_df[variable_name]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        a_df['best_fit'] = intercept + (slope * x)
        return (a_df, (slope, intercept, r_value, p_value, std_err))


    @function_attributes(short_name=None, tags=['UNUSED', 'Figure 4/S6/S8 Stats', 'main', 'stats'], input_requires=[], output_provides=[], uses=['cls.stats_interrupted_timeseries_analysis'], used_by=[], creation_date='2026-10-28 07:36', related_items=[])
    @classmethod
    def run_interrupted_timeseries_analysis(cls, df_filter, variable_name: str = 'P_Short', time_variable_name: str = 'delta_aligned_start_t', **kwargs):
        time_variable_name: str = 'delta_aligned_start_t'
        variable_name = df_filter.active_plot_variable_name  # usually 'P_Short'
        df = deepcopy(df_filter.active_plot_df).dropna(subset=[variable_name])
        pre_df  = df[df[time_variable_name] <= 0]
        post_df = df[df[time_variable_name] > 0]
        (its_model, its_tuple) = cls.stats_interrupted_timeseries_analysis(pre_df, post_df, should_plot=True, 
                                    variable_name=variable_name, time_variable_name=time_variable_name,
                                    )
        intercept, pre_slope, level_change, slope_change, post_slope = its_tuple

        # OLS Regression Results                            
        # ==============================================================================
        # Dep. Variable:                P_Short   R-squared:                       0.056
        # Model:                            OLS   Adj. R-squared:                  0.055
        # Method:                 Least Squares   F-statistic:                     63.22
        # Date:                Tue, 27 Oct 2026   Prob (F-statistic):           1.03e-39
        # Time:                        03:43:24   Log-Likelihood:                -1573.6
        # No. Observations:                3229   AIC:                             3155.
        # Df Residuals:                    3225   BIC:                             3179.
        # Df Model:                           3                                         
        # Covariance Type:            nonrobust                                         
        # =========================================================================================
        #                             coef    std err          t      P>|t|      [0.025      0.975]
        # -----------------------------------------------------------------------------------------
        # const                     0.3753      0.017     22.461      0.000       0.343       0.408
        # delta_aligned_start_t     0.0001   2.18e-05      5.140      0.000    6.92e-05       0.000
        # intervention              0.0820      0.024      3.446      0.001       0.035       0.129
        # time_interaction      -6.823e-05   3.32e-05     -2.053      0.040      -0.000   -3.05e-06
        # ==============================================================================
        # Omnibus:                   145406.152   Durbin-Watson:                   1.952
        # Prob(Omnibus):                  0.000   Jarque-Bera (JB):              336.695
        # Skew:                           0.415   Prob(JB):                     7.72e-74
        # Kurtosis:                       1.653   Cond. No.                     3.15e+03
        # ==============================================================================

        # Notes:
        # [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
        # [2] The condition number is large, 3.15e+03. This might indicate that there are
        # strong multicollinearity or other numerical problems.
        # Pre-intervention slope: 0.000112
        # Immediate level change: 0.082021
        # Slope change: -0.000068
        # Post-intervention slope: 0.000044
        return (its_model, its_tuple)





    @function_attributes(short_name=None, tags=['stats', 'fig4', 'P_Short', 'mode-ratio'], input_requires=[], output_provides=[], uses=[], used_by=['compute_ordinal_mode_shift'], creation_date='2026-10-28 07:30', related_items=[])
    @classmethod
    def compute_mode_ratio(cls, df: pd.DataFrame, value_col='P_Short', condition_col='pre_post_delta_category', low_threshold: float = 0.25, high_threshold: float = 0.75) -> pd.DataFrame:
        """ computes the ratio of >25% quantile and <75% quantile (the histogram's modes)

        For each `condition_col` group, counts occupancy of the high mode (`value_col` > `high_threshold`, default 0.75)
        versus the low mode (`value_col` < `low_threshold`, default 0.25). `mode_ratio` is n_high / n_low.
        Values in [`low_threshold`, `high_threshold`] are the ambiguous middle and are excluded from the ratio.

        P_Short is typically U-shaped on [0, 1]; this ratio is Short-preferring mode occupancy over Long-preferring mode occupancy.

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PrePostDeltaDistributionComparison

            mode_ratio_df = PrePostDeltaDistributionComparison.compute_mode_ratio(filtered_all_sessions_laps_time_bin_df)

        """
        if value_col not in df.columns:
            raise ValueError(f'Missing required column: {value_col!r}')
        if condition_col not in df.columns:
            raise ValueError(f'Missing required column: {condition_col!r}')
        if not (0.0 <= low_threshold < high_threshold <= 1.0):
            raise ValueError('Require 0 <= low_threshold < high_threshold <= 1.')

        work = df[[condition_col, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors='coerce')
        work = work.dropna(subset=[value_col])

        rows = []
        for condition, group in work.groupby(condition_col, dropna=False, sort=False):
            values = group[value_col].to_numpy(dtype=float)
            n = int(values.size)
            n_low = int(np.sum(values < low_threshold))
            n_high = int(np.sum(values > high_threshold))
            n_mid = int(n - n_low - n_high)
            rows.append({
                condition_col: condition,
                'n': n,
                'n_low': n_low,
                'n_high': n_high,
                'n_mid': n_mid,
                'frac_low': (n_low / n) if n > 0 else np.nan,
                'frac_high': (n_high / n) if n > 0 else np.nan,
                'mode_ratio': (n_high / n_low) if n_low > 0 else np.nan,
                'low_threshold': low_threshold,
                'high_threshold': high_threshold,
            })
        ## END for condition, group in work.groupby(condition_col, dropna=False, sort=False)...

        if not rows:
            return pd.DataFrame(columns=[condition_col, 'n', 'n_low', 'n_high', 'n_mid', 'frac_low', 'frac_high', 'mode_ratio', 'low_threshold', 'high_threshold'])
        return pd.DataFrame(rows)


    @function_attributes(short_name=None, tags=['stats', 'fig4', 'P_Short', 'ordinal', 'proportional-odds', 'mode-ratio', 'AI'], input_requires=[], output_provides=[], uses=['OrderedModel', 'compute_mode_ratio'], used_by=['run'], creation_date='2026-10-28 08:00', related_items=['compute_mode_ratio'])
    @classmethod
    def compute_ordinal_mode_shift(cls, df: pd.DataFrame, value_col='P_Short', condition_col='pre_post_delta_category', animal_col='animal', session_col='session_name',
                                   pre_label='pre-delta', post_label='post-delta', low_threshold: float = 0.25, high_threshold: float = 0.75,
                                   distr: str = 'logit', min_bins_per_condition: int = 10, n_bootstrap: int = 1000, n_permutations: int = 2000, random_state=None) -> Dict:
        """Session-level proportional-odds model of low/mid/high P_Short on pre vs post-delta.

        Bins `value_col` into an ordered category: low (`< low_threshold`), mid, high (`> high_threshold`).
        Each session is fit with `OrderedModel(category ~ is_post)`. The headline effect is the unweighted
        mean of session slopes `beta_s` (cumulative log-odds shift toward higher P_Short). `exp(beta)` is
        the proportional odds ratio.

        Inference treats sessions as units nested in animals, matching `sampling_unit='session'`:
        a sign-flip test on `beta_s` and a hierarchical bootstrap (animals, then sessions within animal).
        Time-bin dependence is not modeled as independent Bernoulli trials.

        The proportional-odds assumption is that the post-delta shift is the same for the low-to-mid and
        mid-to-high cutpoints; check `session_summary` category fractions. With few animals, an animal-level
        sign-flip has a high p-value floor (0.25 for n=3), so the test is at the session level.

        A pooled `OrderedModel` with session fixed effects is reported as `pooled_beta_session_fe` with a
        naive SE flagged as anti-conservative. Threshold sensitivity is left to the caller via `low_threshold`
        / `high_threshold`.

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PrePostDeltaDistributionComparison

            ordinal_results = PrePostDeltaDistributionComparison.compute_ordinal_mode_shift(filtered_all_sessions_laps_time_bin_df)
        """
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        required_cols = [value_col, condition_col, animal_col, session_col]
        missing_cols = set(required_cols).difference(df.columns)
        if missing_cols:
            raise ValueError(f'Missing required columns: {sorted(missing_cols)}')
        if not (0.0 <= low_threshold < high_threshold <= 1.0):
            raise ValueError('Require 0 <= low_threshold < high_threshold <= 1.')
        if min_bins_per_condition < 1:
            raise ValueError('min_bins_per_condition must be >= 1.')
        if n_bootstrap < 1:
            raise ValueError('n_bootstrap must be >= 1.')
        if n_permutations < 1:
            raise ValueError('n_permutations must be >= 1.')

        work = df.loc[df[condition_col].isin([pre_label, post_label]), required_cols].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors='coerce')
        work = work.dropna(subset=required_cols).reset_index(drop=True)
        if work.empty or set(work[condition_col].unique()) != {pre_label, post_label}:
            raise ValueError(f'Data must contain both groups: {(pre_label, post_label)}')

        category_levels = ('low', 'mid', 'high')
        work['category'] = np.select(
            [work[value_col] < low_threshold, work[value_col] > high_threshold],
            ['low', 'high'],
            default='mid',
        )
        work['category'] = pd.Categorical(work['category'], categories=list(category_levels), ordered=True)
        work['is_post'] = (work[condition_col] == post_label).astype(float)
        work['_session_key'] = work[animal_col].astype(str) + '|' + work[session_col].astype(str)

        def _cat_counts(series: pd.Series) -> Dict[str, int]:
            counts = series.value_counts()
            return {name: int(counts.get(name, 0)) for name in category_levels}

        def _fit_is_post_slope(endog: pd.Series, is_post: pd.Series) -> Tuple[float, float, bool]:
            observed = [name for name in category_levels if int((endog == name).sum()) > 0]
            endog_series = pd.Series(pd.Categorical(np.asarray(endog), categories=observed, ordered=True), name='category')
            exog = pd.DataFrame({'is_post': np.asarray(is_post, dtype=np.float64)})
            model = OrderedModel(endog_series, exog, distr=distr)
            fit = model.fit(method='bfgs', disp=False)
            converged = bool(fit.mle_retvals.get('converged', False))
            return float(fit.params['is_post']), float(fit.bse['is_post']), converged

        session_rows = []
        for (animal, session), sess in work.groupby([animal_col, session_col], dropna=False, sort=False):
            pre_mask = sess[condition_col] == pre_label
            post_mask = sess[condition_col] == post_label
            n_pre = int(pre_mask.sum())
            n_post = int(post_mask.sum())
            pre_counts = _cat_counts(sess.loc[pre_mask, 'category'])
            post_counts = _cat_counts(sess.loc[post_mask, 'category'])
            n_cats = int(sess['category'].nunique(dropna=True))
            row = {
                'animal': animal,
                'session': session,
                'n_pre': n_pre,
                'n_post': n_post,
                'n_low_pre': pre_counts['low'],
                'n_mid_pre': pre_counts['mid'],
                'n_high_pre': pre_counts['high'],
                'n_low_post': post_counts['low'],
                'n_mid_post': post_counts['mid'],
                'n_high_post': post_counts['high'],
                'frac_low_pre': (pre_counts['low'] / n_pre) if n_pre > 0 else np.nan,
                'frac_mid_pre': (pre_counts['mid'] / n_pre) if n_pre > 0 else np.nan,
                'frac_high_pre': (pre_counts['high'] / n_pre) if n_pre > 0 else np.nan,
                'frac_low_post': (post_counts['low'] / n_post) if n_post > 0 else np.nan,
                'frac_mid_post': (post_counts['mid'] / n_post) if n_post > 0 else np.nan,
                'frac_high_post': (post_counts['high'] / n_post) if n_post > 0 else np.nan,
                'beta': np.nan,
                'se': np.nan,
                'converged': False,
                'quasi_separated': False,
                'used_for_inference': False,
                'status': 'ok',
            }
            if n_pre < min_bins_per_condition or n_post < min_bins_per_condition:
                row['status'] = (
                    f'too_few_bins: n_pre={n_pre}, n_post={n_post}, '
                    f'min_bins_per_condition={min_bins_per_condition}'
                )
                session_rows.append(row)
                continue
            if n_cats < 2:
                row['status'] = f'too_few_categories: n_categories={n_cats}'
                session_rows.append(row)
                continue
            try:
                beta, se, converged = _fit_is_post_slope(sess['category'], sess['is_post'])
            except Exception as e:
                row['status'] = f'fit_failed: {e}'
                session_rows.append(row)
                continue
            quasi_separated = bool(np.isfinite(beta) and abs(beta) > 20.0)
            row['beta'] = beta
            row['se'] = se
            row['converged'] = converged
            row['quasi_separated'] = quasi_separated
            if not converged:
                row['status'] = 'not_converged'
            elif quasi_separated:
                row['status'] = 'quasi_separated'
            else:
                row['used_for_inference'] = True
                row['status'] = 'ok'
            session_rows.append(row)
        ## END for (animal, session), sess in work.groupby([animal_col, session_col], dropna=False, sort=False)....

        session_summary = pd.DataFrame(session_rows)
        if session_summary.empty:
            valid = session_summary
        else:
            valid = session_summary.loc[session_summary['used_for_inference']].copy()
        n_sessions_used = int(len(valid))
        n_sessions_skipped = int(len(session_summary) - n_sessions_used)
        n_animals_used = int(valid['animal'].nunique()) if n_sessions_used > 0 else 0
        rng = np.random.default_rng(random_state)

        if n_sessions_used > 0:
            animal_summary = (
                valid.groupby('animal', sort=False, dropna=False)['beta']
                .mean()
                .rename('mean_beta_post_minus_pre')
                .reset_index()
            )
        else:
            animal_summary = pd.DataFrame(columns=['animal', 'mean_beta_post_minus_pre'])

        mean_beta = float(valid['beta'].mean()) if n_sessions_used > 0 else np.nan
        odds_ratio = float(np.exp(mean_beta)) if np.isfinite(mean_beta) else np.nan

        if n_sessions_used < 1:
            p_value = None
            p_value_method = None
            beta_ci = None
            odds_ratio_ci = None
            infer_status = 'Not testable: no eligible sessions.'
        else:
            effects = valid['beta'].to_numpy(dtype=float)
            n_units = int(effects.size)
            observed = float(effects.mean())
            abs_observed = abs(observed)
            n_exact = 1 << n_units
            use_exact = (n_units <= 20) and (n_exact <= n_permutations)
            if use_exact:
                sign_bits = (
                    (np.arange(n_exact, dtype=np.uint32)[:, None] >> np.arange(n_units)) & 1
                )
                signs = 1.0 - 2.0 * sign_bits.astype(float)
                null_means = signs @ effects / n_units
                p_value = float(np.count_nonzero(np.abs(null_means) >= abs_observed) / n_exact)
                p_value_method = 'exact'
            else:
                null_means = np.empty(n_permutations, dtype=float)
                sign_choices = np.array([1.0, -1.0])
                for i in range(n_permutations):
                    signs = rng.choice(sign_choices, size=n_units)
                    null_means[i] = float(np.mean(signs * effects))
                ## END for i in range(n_permutations)....

                p_value = (
                    1 + np.count_nonzero(np.abs(null_means) >= abs_observed)
                ) / (n_permutations + 1)
                p_value_method = 'monte_carlo'

            betas_by_animal = {
                animal: group['beta'].to_numpy(dtype=float)
                for animal, group in valid.groupby('animal', sort=False, dropna=False)
            }
            animals = list(betas_by_animal.keys())
            n_animals_boot = len(animals)
            bootstrap_means = np.empty(n_bootstrap, dtype=float)
            for i in range(n_bootstrap):
                sampled_animals = rng.choice(animals, size=n_animals_boot, replace=True)
                sampled_betas = []
                for animal in sampled_animals:
                    animal_betas = betas_by_animal[animal]
                    sampled_betas.extend(
                        rng.choice(animal_betas, size=len(animal_betas), replace=True).tolist()
                    )
                ## END for animal in sampled_animals....

                bootstrap_means[i] = float(np.mean(sampled_betas))
            ## END for i in range(n_bootstrap)....

            beta_ci = tuple(np.quantile(bootstrap_means, [0.025, 0.975]))
            odds_ratio_ci = tuple(float(np.exp(v)) for v in beta_ci)
            infer_status = (
                'ok; session-level proportional-odds (ordinal logit) of low/mid/high P_Short. '
                'Sign-flip p-value and hierarchical bootstrap treat sessions as units nested in animals. '
                'Does not treat time bins as independent. '
                'Session-level tests treat sessions as exchangeable and ignore within-animal dependence. '
                'Proportional-odds assumes the same post-delta shift for low-to-mid and mid-to-high cutpoints.'
            )

        pooled_beta = np.nan
        pooled_se = np.nan
        try:
            observed = [name for name in category_levels if int((work['category'] == name).sum()) > 0]
            if len(observed) < 2:
                pooled_status = 'failed: fewer than 2 categories in pooled data.'
            else:
                session_dummies = pd.get_dummies(work['_session_key'], drop_first=True, dtype=np.float64)
                pooled_exog = pd.concat([work[['is_post']].astype(np.float64), session_dummies], axis=1)
                pooled_endog = pd.Series(pd.Categorical(work['category'].to_numpy(), categories=observed, ordered=True), name='category')
                pooled_model = OrderedModel(pooled_endog, pooled_exog, distr=distr)
                pooled_fit = pooled_model.fit(method='bfgs', disp=False)
                pooled_beta = float(pooled_fit.params['is_post'])
                pooled_se = float(pooled_fit.bse['is_post'])
                pooled_status = 'ok; naive SE is anti-conservative because time bins are dependent.'
        except Exception as e:
            pooled_status = f'failed: {e}'

        frac_rows = []
        for condition, group in work.groupby(condition_col, dropna=False, sort=False):
            n = int(len(group))
            counts = _cat_counts(group['category'])
            frac_rows.append({
                condition_col: condition,
                'n': n,
                'frac_low': (counts['low'] / n) if n > 0 else np.nan,
                'frac_mid': (counts['mid'] / n) if n > 0 else np.nan,
                'frac_high': (counts['high'] / n) if n > 0 else np.nan,
            })
        ## END for condition, group in work.groupby(condition_col, dropna=False, sort=False)....

        category_fractions = pd.DataFrame(frac_rows)
        mode_ratio_df = cls.compute_mode_ratio(
            work, value_col=value_col, condition_col=condition_col,
            low_threshold=low_threshold, high_threshold=high_threshold,
        )

        return {
            'mean_beta_post_minus_pre': mean_beta,
            'odds_ratio': odds_ratio,
            'beta_bootstrap_95ci': beta_ci,
            'odds_ratio_bootstrap_95ci': odds_ratio_ci,
            'p_value': p_value,
            'p_value_method': p_value_method,
            'n_sessions_used': n_sessions_used,
            'n_sessions_skipped': n_sessions_skipped,
            'n_animals_used': n_animals_used,
            'animal_summary': animal_summary,
            'session_summary': session_summary,
            'pooled_beta_session_fe': pooled_beta,
            'pooled_se_naive': pooled_se,
            'pooled_status': pooled_status,
            'category_fractions': category_fractions,
            'mode_ratio': mode_ratio_df,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'distr': distr,
            'status': infer_status,
        }


    @function_attributes(short_name=None, tags=['stats', 'fig4', 'P_Short', 'figure2', 'summary', 'bar'], input_requires=[], output_provides=[], uses=['PaperFigureTwo.create_plot', 'add_significance_bars'], used_by=[], creation_date='2026-10-28 08:32', related_items=['compute_ordinal_mode_shift'])
    @classmethod
    def plot_fig2_style_summary_four_bar(cls, laps_results, pbe_results,
                                         pre_col: str = 'mean_pre', post_col: str = 'mean_post', animal_col: str = 'animal',
                                         laps_p_value: Optional[float] = None, pbe_p_value: Optional[float] = None,
                                         ylabel: str = 'Session-mean $P_{Short}$', title: str = 'Decoded $P_{Short}$',
                                         group_labels: Tuple[str, str] = ('Laps', 'PBEs'), prepare_for_publication: bool = True,
                                         defer_show: bool = False, save_figure: bool = False, export_folder: Optional[Path] = None,
                                         fig_file_name: str = 'figure4_summary_four_bar', active_context: Optional[IdentifyingContext] = None,
                                         enable_paired_lines: bool = True, debug_print: bool = False, **kwargs) -> Dict[str, Any]:
        """Plot session-level P_Short means as one Figure-2-style 4-bar panel (Laps-pre, Laps-post, PBE-pre, PBE-post).

        `laps_results` / `pbe_results` may be `run()` result dicts (must include `session_summary`) or `session_summary` DataFrames.
        Significance brackets use the permutation `p_value` from each `run()` dict (or explicit `laps_p_value` / `pbe_p_value`).

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PrePostDeltaDistributionComparison

            out = PrePostDeltaDistributionComparison.plot_fig2_style_summary_four_bar(
                pre_post_delta_stats_results_dict['_perTimeBin_Laps'],
                pre_post_delta_stats_results_dict['_perTimeBin_PBEs'],
            )
            out['fig']
        """
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PaperFigureTwo, add_significance_bars

        def _coerce_session_summary(obj, explicit_p_value, name: str) -> Tuple[pd.DataFrame, Optional[float]]:
            if isinstance(obj, pd.DataFrame):
                summary = obj
                p_value = explicit_p_value
            elif isinstance(obj, dict):
                if ('session_summary' not in obj) or (obj['session_summary'] is None):
                    raise ValueError(f'{name} is missing session_summary; pass sampling_unit="session" to run().')
                summary = obj['session_summary']
                p_value = explicit_p_value if explicit_p_value is not None else obj.get('p_value')
            else:
                raise TypeError(f'{name} must be a run() result dict or a session_summary DataFrame, got {type(obj)!r}.')
            if not isinstance(summary, pd.DataFrame) or summary.empty:
                raise ValueError(f'{name} session_summary is empty.')
            missing_cols = [c for c in (pre_col, post_col) if c not in summary.columns]
            if missing_cols:
                raise ValueError(f'{name} session_summary is missing columns: {missing_cols}.')
            work = summary.dropna(subset=[pre_col, post_col]).copy().reset_index(drop=True)
            if work.empty:
                raise ValueError(f'{name} session_summary has no finite {pre_col}/{post_col} rows.')
            return work, p_value

        laps_session_summary, laps_p_value = _coerce_session_summary(laps_results, laps_p_value, 'laps_results')
        pbe_session_summary, pbe_p_value = _coerce_session_summary(pbe_results, pbe_p_value, 'pbe_results')

        y_values = [
            laps_session_summary[pre_col].to_numpy(dtype=float),
            laps_session_summary[post_col].to_numpy(dtype=float),
            pbe_session_summary[pre_col].to_numpy(dtype=float),
            pbe_session_summary[post_col].to_numpy(dtype=float),
        ]
        x_labels = [
            f'{group_labels[0]} pre-$\\Delta$',
            f'{group_labels[0]} post-$\\Delta$',
            f'{group_labels[1]} pre-$\\Delta$',
            f'{group_labels[1]} post-$\\Delta$',
        ]

        has_animal: bool = (animal_col in laps_session_summary.columns) and (animal_col in pbe_session_summary.columns)
        filled_markers = ['o', '^', '8', 's', 'p', 'd', 'P', 'X']
        if has_animal:
            unique_animals = pd.unique(pd.concat([laps_session_summary[animal_col], pbe_session_summary[animal_col]], ignore_index=True))
            marker_mapping = {animal: filled_markers[i % len(filled_markers)] for i, animal in enumerate(unique_animals)}
        else:
            marker_mapping = {}

        def _bar_scatter_props(summary_df: pd.DataFrame):
            if not has_animal:
                return {}
            return [{'marker': marker_mapping.get(a, 'o')} for a in summary_df[animal_col].to_numpy()]

        scatter_props = [
            _bar_scatter_props(laps_session_summary),
            _bar_scatter_props(laps_session_summary),
            _bar_scatter_props(pbe_session_summary),
            _bar_scatter_props(pbe_session_summary),
        ]
        if active_context is None:
            active_context = IdentifyingContext(format_name='kdiba', fig='4', display_fn_name='pre_post_delta_summary_bars')

        if debug_print:
            print(f'laps n_sessions={len(laps_session_summary)}, pbe n_sessions={len(pbe_session_summary)}, laps_p_value={laps_p_value}, pbe_p_value={pbe_p_value}')

        render = PaperFigureTwo.create_plot(
            x_labels, y_values, scatter_props, ylabel, title, 'fig4_summary_four_bar', active_context, defer_show,
            prepare_for_publication=prepare_for_publication,
            publication_group_labels=group_labels, metric_name='Session Means',
            title_subtitle=f'for {group_labels[0]} and {group_labels[1]}', xlabel='',
            **kwargs,
        )
        fig = render.fig
        ax = render.ax
        x_values_list, plotted_y_values = render.plot_data

        if enable_paired_lines:
            for pair_i in (0, 2):
                for x0, y0, x1, y1 in zip(x_values_list[pair_i], plotted_y_values[pair_i], x_values_list[pair_i + 1], plotted_y_values[pair_i + 1]):
                    ax.plot([x0, x1], [y0, y1], color='gray', alpha=0.3, zorder=1, linewidth=0.8)
                ## END for x0, y0, x1, y1 in zip(x_values_list[pair_i], plotted_y_values[pair_i], x_values_list[pair_i + 1], plotted_y_values[pair_i + 1])...

            ## END for pair_i in (0, 2)...

        y_max = float(np.nanmax(np.concatenate(y_values)))
        if laps_p_value is not None:
            add_significance_bars(ax, laps_p_value, x1=0, x2=1, y=y_max)
        if pbe_p_value is not None:
            add_significance_bars(ax, pbe_p_value, x1=2, x2=3, y=y_max)

        saved_path = None
        if save_figure:
            if export_folder is None:
                raise ValueError('export_folder is required when save_figure=True.')
            export_folder = Path(export_folder)
            export_folder.mkdir(parents=True, exist_ok=True)
            saved_path = export_folder / f'{fig_file_name}.pdf'
            fig.savefig(saved_path)

        return dict(
            fig=fig,
            ax=ax,
            render=render,
            laps_session_summary=laps_session_summary,
            pbe_session_summary=pbe_session_summary,
            y_values=y_values,
            laps_p_value=laps_p_value,
            pbe_p_value=pbe_p_value,
            saved_path=saved_path,
        )


# ==================================================================================================================================================================================================================================================================================== #
# 2026-10-26 - Factor out Figure S8                                                                                                                                                                                                                                                    #
# ==================================================================================================================================================================================================================================================================================== #
import plotly.graph_objects as go
import plotly.io as pio
from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import PlotlyFigureContainer
from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers, text_sizes_px
from pyphocorehelpers.plotting.media_output_helpers import PDFHelpers, SVGHelpers ## for outputs


@metadata_attributes(short_name=None, tags=['figure-s8', 's8', 'figure-4', 'fig4', 'figure-s6', 's6', 'figure', 'publication'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-26 07:57', related_items=[])
class FigureS8:
    """ Used to output Figure 4, Figure S6, Figure S8

        Usage:
            from pathlib import Path
            import plotly.graph_objects as go
            import plotly.io as pio
            from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import PlotlyFigureContainer
            from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers, text_sizes_px
            from pyphocorehelpers.plotting.media_output_helpers import PDFHelpers, SVGHelpers ## for outputs

            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import FigureS8

            ## INPUTS: earliest_delta_aligned_t_start, latest_delta_aligned_t_end
            plotly_helpers.earliest_delta_aligned_t_start = earliest_delta_aligned_t_start
            plotly_helpers.latest_delta_aligned_t_end = latest_delta_aligned_t_end
            # figures_parent_folder: Path = Path(r'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/Pho Diba Paper 2023/Figures/UneditedSources/Figure 4 - P_Short Scatter over Time/2026-10-23').resolve()
            figures_parent_folder: Path = Path(r'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/Pho Diba Paper 2023/Figures/UneditedSources/Figure S8 - Effect of Heuristic-filtering P_Short Scatter over Time/Inputs/2026-10-23').resolve()
            figures_parent_folder.mkdir(exist_ok=True)
            _out = FigureS8.main(df_filter=df_filter, plotly_helpers=plotly_helpers, additional_filter_predicates=additional_filter_predicates, figures_parent_folder=figures_parent_folder, split_by_column='animal')


    """
    plotly_helpers: PlotlyHelpers = None
    PUB_FIG_SIZE: Dict = dict(width=792, height=320)
    figures_parent_folder: Path = None
    _transparent_bgcolor: str = "rgba(0,0,0,0)"  # Kaleido PDF ignores this (always white); SVG+cairosvg preserves it.

    _heuristic_filter_color_map: Dict[str, str] = {
        'False': 'rgba(170,170,170,0.45)',
        'True': 'black',
    }
    _PREFERRED_ANIMAL_ORDER = ('gor01', 'vvp01', 'pin01')


    @classmethod
    def figures_intermediates_parent_folder(cls) -> Path:
        """Where temporary/intermediate figures are written."""
        if cls.figures_parent_folder is None:
            return None
        else:
            out_path = cls.figures_parent_folder.joinpath('_TEMP')
            out_path.mkdir(exist_ok=True)
            return out_path


    @classmethod
    def _post_hoc_format_figure(cls, a_fig: go.Figure, showlegend: bool = True) -> go.Figure:
        """Force publication fonts onto the figure object (templates are only fallbacks)."""
        layout_kwargs = dict(
            template=cls.template,
            **cls.PUB_FIG_SIZE,
            paper_bgcolor=cls._transparent_bgcolor,
            plot_bgcolor=cls._transparent_bgcolor,
            showlegend=showlegend,
            font=dict(**cls._pub_font, size=text_sizes_px["insets"]),
            title_font=dict(**cls._pub_font, size=text_sizes_px["titles"]),
            legend_font=dict(**cls._pub_font, size=text_sizes_px["insets"]),
            legend_title_font=dict(**cls._pub_font, size=text_sizes_px["insets"]),
            margin=dict(l=50, r=40, t=36, b=50),
        )
        if showlegend:
            # Horizontal legend below the x-axis titles — paper (0.99, 0.99) sat on the right histogram
            layout_kwargs['legend'] = dict(title_text='Heuristic-Passing', orientation='h', x=0.5, y=-0.28, xanchor='center', yanchor='top', bgcolor=cls._transparent_bgcolor, borderwidth=0, itemsizing='constant', tracegroupgap=0)
            layout_kwargs['margin'] = dict(l=50, r=40, t=36, b=88)
        a_fig = a_fig.update_layout(**layout_kwargs)
        a_fig = a_fig.update_xaxes(
            title_font=dict(**cls._pub_font, size=text_sizes_px["axes"]),
            tickfont=dict(**cls._pub_font, size=text_sizes_px["axes"]),
            linewidth=1,
        )
        a_fig = a_fig.update_yaxes(
            title_font=dict(**cls._pub_font, size=text_sizes_px["axes"]),
            tickfont=dict(**cls._pub_font, size=text_sizes_px["axes"]),
            linewidth=1,
        )
        # Do not bump all annotations to title size — Pre-Δ / Post-Δ / axis labels are annotations and become unreadable.
        return a_fig


    # def _save_publication_svg(a_fig: go.Figure, a_fig_context, figures_parent_folder: Path, basename: Optional[str] = None) -> Path:
    #     if basename is None:
    #         basename = a_fig_context.get_description()
    #     image_save_path = figures_parent_folder.joinpath(f"{basename}.svg")
    #     a_fig.write_image(image_save_path, engine="kaleido", format="svg")
    #     return image_save_path

    @classmethod
    def _write_transparent_pdf_from_svg(cls, svg_path: Path, pdf_path: Path) -> Path:
        """Kaleido `write_image(..., format='pdf')` composites onto an opaque white canvas. Convert a transparent SVG instead."""
        import cairosvg
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        return pdf_path


    @classmethod
    def _save_publication_figs(cls, a_fig: go.Figure, a_fig_context, figures_parent_folder: Path, export_suffix: str, basename: Optional[str] = None, _out_paths: Optional[Dict]=None,
                                should_export_svg: bool = True, should_export_pdf: Optional[bool] = None,
                            ) -> Path:
        if basename is None:
            basename = a_fig_context.get_description()

        if _out_paths is None:
            _out_paths = {}
        if should_export_pdf is None:
            should_export_pdf = (not should_export_svg)  # SVG mode: skip PDF; PDF mode: skip SVG unless both requested

        export_fig = deepcopy(a_fig)
        export_fig.update_layout(paper_bgcolor=cls._transparent_bgcolor, plot_bgcolor=cls._transparent_bgcolor)

        svg_output_folder = figures_parent_folder
        if should_export_pdf and (not should_export_svg):
            intermediates_folder = cls.figures_intermediates_parent_folder()
            if intermediates_folder is not None:
                svg_output_folder = intermediates_folder  # SVG is only a conversion source for the PDF
        svg_save_path = svg_output_folder.joinpath(f'{basename}.svg')
        if should_export_svg or should_export_pdf:
            export_fig.write_image(svg_save_path, format="svg")  # Kaleido SVG does honor transparent bg
        if should_export_svg:
            _out_paths[f'{export_suffix}/svg'] = svg_save_path

        if should_export_pdf:
            pdf_save_path = figures_parent_folder.joinpath(f'{basename}.pdf')
            cls._write_transparent_pdf_from_svg(svg_save_path, pdf_save_path)
            _out_paths[f'{export_suffix}/pdf'] = pdf_save_path

        return _out_paths


    @classmethod
    def _overlay_heuristic_filter_rows(cls, df: pd.DataFrame, additional_filter_predicates) -> pd.DataFrame:
        """Stack unfiltered (`False`) and heuristic-passing (`True`) copies so both appear on one figure."""
        passing_mask = additional_filter_predicates['is_heuristic_passing'](df).astype(bool)
        df_all = deepcopy(df)
        df_all['heuristic_filter'] = 'False'
        df_pass = deepcopy(df[passing_mask])
        df_pass['heuristic_filter'] = 'True'
        out_df: pd.DataFrame = pd.concat([df_all, df_pass], ignore_index=True)
        if 'data_context' in df.attrs:
            out_df.attrs['data_context'] = deepcopy(df.attrs['data_context'])
        return out_df

    # ==================================================================================================================================================================================================================================================================================== #
    # Split On Animal Helpers                                                                                                                                                                                                                                                              #
    # ==================================================================================================================================================================================================================================================================================== #
    @function_attributes(short_name=None, tags=['private', 'split-on-animal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-26 14:39', related_items=[])
    @classmethod
    def _copy_df_preserving_context(cls, df: pd.DataFrame) -> pd.DataFrame:
        out_df: pd.DataFrame = deepcopy(df)
        if 'data_context' in df.attrs:
            out_df.attrs['data_context'] = deepcopy(df.attrs['data_context'])
        return out_df


    @function_attributes(short_name=None, tags=['private', 'split-on-animal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-26 14:39', related_items=[])
    @classmethod
    def _sorted_split_values(cls, values, split_by_column: str):
        unique_values = [str(v) for v in values]
        if split_by_column == 'animal':
            preferred = [v for v in cls._PREFERRED_ANIMAL_ORDER if v in unique_values]
            rest = sorted([v for v in unique_values if v not in preferred], key=str)
            return preferred + rest
        return sorted(unique_values, key=str)


    @function_attributes(short_name=None, tags=['private', 'split-on-animal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-26 14:39', related_items=[])
    @classmethod
    def _iter_split_plot_groups(cls, df: pd.DataFrame, split_by_column: Optional[str]):
        """Yield `(group_key, group_df)`. `group_key` is None when not splitting."""
        if split_by_column is None:
            yield None, df
            return
        if split_by_column not in df.columns:
            print(f'WARN: split_by_column={split_by_column!r} not in dataframe columns; plotting pooled')
            yield None, df
            return
        unique_values = cls._sorted_split_values(df[split_by_column].dropna().unique(), split_by_column=split_by_column)
        for a_value in unique_values:
            group_df = cls._copy_df_preserving_context(df[df[split_by_column].astype(str) == a_value])
            yield a_value, group_df
        ## END for a_value in unique_values...


    @function_attributes(short_name=None, tags=['private', 'split-on-animal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-26 14:39', related_items=[])
    @classmethod
    def _plot_formatted_pre_post_scatter(cls, plot_df: pd.DataFrame, data_context, plot_kwargs: Dict, showlegend: bool):
        new_fig, new_fig_context, *_ = cls.plotly_helpers._perform_plot_pre_post_delta_scatter(
            data_context=data_context,
            concatenated_ripple_df=plot_df,
            **plot_kwargs,
        )
        new_fig = cls._post_hoc_format_figure(new_fig, showlegend=showlegend)
        new_fig.update_layout(**cls.PUB_FIG_SIZE)
        return new_fig, new_fig_context


    # ==================================================================================================================================================================================================================================================================================== #
    # Pre/post-delta stats insets                                                                                                                                                                                                                                                          #
    # ==================================================================================================================================================================================================================================================================================== #
    _STATS_ANNOTATION_LABEL_COLOR: Dict[str, str] = {
        'False': '#aaaaaa',  # solid gray; overlay map alpha is too faint for text
        'True': 'black',
    }


    @classmethod
    def _format_pre_post_delta_stats_text(cls, stats: Dict) -> str:
        """Compact Δμ / CI / p inset text for a session-level pre/post comparison."""
        d = stats['mean_difference_post_minus_pre']
        ci = stats.get('mean_difference_bootstrap_95ci')
        p = stats.get('p_value')
        n_sessions = stats.get('n_sessions')
        if ci is None:
            ci_text: str = ''
        else:
            lo, hi = ci
            ci_text = f'  [{lo:.2f}, {hi:.2f}]'
        if p is None:
            p_text: str = 'p = n/a'
        elif p < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p:.3f}'
        n_text: str = f'   (n = {n_sessions} sessions)' if n_sessions is not None else ''
        return f'Δμ = {d:.2f}{ci_text}<br>{p_text}{n_text}'


    @classmethod
    def _add_pre_post_delta_stats_annotation(cls, fig: go.Figure, stats_by_label: Dict[str, Dict]) -> go.Figure:
        """Add one scatter-panel inset. Empty label = pooled; False/True = heuristic overlay."""
        lines = []
        for label, stats in stats_by_label.items():
            if stats is None:
                continue
            body: str = cls._format_pre_post_delta_stats_text(stats)
            if label:
                color: str = cls._STATS_ANNOTATION_LABEL_COLOR.get(label, 'black')
                body_lines = body.split('<br>')
                body_lines[0] = f'{label}: {body_lines[0]}'
                inner: str = '<br>'.join(body_lines)
                lines.append(f'<span style="color:{color}">{inner}</span>')
            else:
                lines.append(body)
        ## END for label, stats in stats_by_label.items()...

        if not lines:
            return fig
        fig.add_annotation(
            text='<br>'.join(lines),
            xref='x2 domain', yref='y2 domain', x=0.99, y=0.98,
            xanchor='right', yanchor='top', showarrow=False, align='right',
            font=dict(family='Arial', size=text_sizes_px['insets'], color='black'),
            bgcolor='rgba(255,255,255,0.75)', borderpad=3,
            name='pre_post_delta_stats',
        )
        return fig


    @function_attributes(short_name=None, tags=['private', 'stats'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-28 07:48', related_items=[])
    @classmethod
    def _compute_pre_post_delta_stats(cls, df: pd.DataFrame, class_kwargs: Dict, run_kwargs: Dict) -> Tuple[Optional[PrePostDeltaDistributionComparison], Optional[Dict]]:
        """Run PrePostDeltaDistributionComparison; skip a subset that lacks both pre and post."""
        try:
            comparison = PrePostDeltaDistributionComparison(df, **class_kwargs)
            results = comparison.run(**run_kwargs)
            return comparison, results
        except ValueError as e:
            print(f'WARN: pre/post-delta stats skipped: {e}')
            return None, None


    ## INPUTS: df_filter, plotly_helpers, additional_filter_predicates, figures_parent_folder
    @function_attributes(short_name=None, tags=['private', 'secondary', 'figure', 'stats'], input_requires=[], output_provides=[], uses=['cls._compute_pre_post_delta_stats'], used_by=[], creation_date='2026-10-28 07:47', related_items=[])
    @classmethod
    def _perform_plot_iteratively_constrained(cls, base_df: pd.DataFrame, additional_filter_predicates, figures_parent_folder: Path, export_suffix: str, title_suffix: str='', constraint_keys=("is_long_duration", "is_track_body"), should_save: bool = True,
                                                    should_export_initially_filtered: bool = False, should_export_iteratively_constrainted: bool = True, should_overlay_heuristic_filter: bool = False, split_by_column: Optional[str] = None,
                                                    should_compute_stats: bool = False, enable_add_stats_to_figures: bool = False, stats_class_kwargs: Optional[Dict] = None, stats_run_kwargs: Optional[Dict] = None,
        ):
        """Plot the constrained pre/post-delta scatter. Also does stats when `should_compute_stats == True`

        `split_by_column='animal'` plots one 1x3 row per animal and stacks those SVGs into `*_by_animal.svg`.
        PX `facet_row` cannot do this: the helper always copies traces into a fixed 1x3 figure.
        """
        predicates = {k: additional_filter_predicates[k] for k in constraint_keys}
        plot_kind: str = 'PBEs' if ('PBEs' in export_suffix) else 'Laps'
        main_title: str = f"Short-track Decoded Likelihood during {plot_kind}" # keep the center subplot title short so it does not collide with Pre-Δ / Post-Δ
        should_compute_or_annotate_stats: bool = bool(should_compute_stats or enable_add_stats_to_figures)
        if stats_class_kwargs is None:
            stats_class_kwargs = {}
        if stats_run_kwargs is None:
            stats_run_kwargs = {}

        _plot_kwargs = dict(
            is_publication_ready_figure=True,
            px_scatter_kwargs={"size_max": 2.0},
            should_save=False,  # save AFTER font/size fix (library saves too early)
            main_title=main_title,
            # epoch_label_mode='annotation_above', #TODO 2026-09-24 18:23: - [ ] where the labels sit
        )
        if should_overlay_heuristic_filter:
            _heuristic_color_kwargs = dict(color='heuristic_filter', color_discrete_map=cls._heuristic_filter_color_map, category_orders={'heuristic_filter': ['False', 'True']}, color_discrete_sequence=[cls._heuristic_filter_color_map['False'], cls._heuristic_filter_color_map['True']])
            _plot_kwargs['px_scatter_kwargs'] = {"size_max": 2.0, **_heuristic_color_kwargs}
            _plot_kwargs['hist_kwargs'] = deepcopy(_heuristic_color_kwargs)
            _plot_kwargs['common_plot_kwargs'] = deepcopy(_heuristic_color_kwargs)

        _out_figs_dict = {}
        _out_paths_dict = {}
        _out_stats_dict = {}

        def _export_stage(k: str, source_df: pd.DataFrame, title_prefix: str, basename: Optional[str]):
            """Plot pooled, or one 1x3 row per `split_by_column` value then stack those SVGs vertically."""
            group_figs = {}
            group_svg_paths = []
            group_pdf_paths = []
            if k not in _out_paths_dict:
                _out_paths_dict[k] = {}

            for group_key, group_df in cls._iter_split_plot_groups(source_df, split_by_column):
                plot_df = cls._overlay_heuristic_filter_rows(group_df, additional_filter_predicates) if should_overlay_heuristic_filter else group_df
                context_kwargs = dict(title_prefix=title_prefix, filter=k)
                if group_key is not None:
                    context_kwargs[split_by_column] = group_key
                    print(f"\t{k} {split_by_column}={group_key}: {len(group_df)} rows")
                data_context = deepcopy(source_df.attrs["data_context"]).overwriting_context(**context_kwargs)
                active_plot_kwargs = deepcopy(_plot_kwargs)
                if group_key is not None:
                    active_plot_kwargs['main_title'] = f"{main_title} | {group_key}"
                new_fig, new_fig_context = cls._plot_formatted_pre_post_scatter(plot_df=plot_df, data_context=data_context, plot_kwargs=active_plot_kwargs, showlegend=should_overlay_heuristic_filter)
                if should_compute_or_annotate_stats and (k == 'Final'):
                    stats_by_label = {}
                    comparison_all, results_all = cls._compute_pre_post_delta_stats(group_df, stats_class_kwargs, stats_run_kwargs)
                    group_stats = {'all': (comparison_all, results_all)}
                    if results_all is not None:
                        stats_by_label['False' if should_overlay_heuristic_filter else ''] = results_all
                    if should_overlay_heuristic_filter:
                        passing_mask = additional_filter_predicates['is_heuristic_passing'](group_df).astype(bool)
                        comparison_pass, results_pass = cls._compute_pre_post_delta_stats(group_df[passing_mask], stats_class_kwargs, stats_run_kwargs)
                        group_stats['heuristic_passing'] = (comparison_pass, results_pass)
                        if results_pass is not None:
                            stats_by_label['True'] = results_pass
                    _out_stats_dict.setdefault(k, {})[group_key] = group_stats
                    if enable_add_stats_to_figures and stats_by_label:
                        new_fig = cls._add_pre_post_delta_stats_annotation(new_fig, stats_by_label)
                fig_key = group_key if group_key is not None else k
                group_figs[fig_key] = new_fig
                if should_save:
                    if basename is None:
                        save_basename = None
                    elif group_key is not None:
                        save_basename = f"{basename}_{group_key}"
                    else:
                        save_basename = basename
                    save_folder = figures_parent_folder
                    if group_key is not None:
                        intermediates_folder = cls.figures_intermediates_parent_folder()
                        if intermediates_folder is not None:
                            save_folder = intermediates_folder  # pre-concatenated per-group SVGs/PDFs
                    saved_paths = cls._save_publication_figs(new_fig, new_fig_context, save_folder, export_suffix=export_suffix, basename=save_basename, _out_paths={})
                    if group_key is not None:
                        if f'{export_suffix}/svg' in saved_paths:
                            _out_paths_dict[k][f'{export_suffix}/svg/{group_key}'] = saved_paths[f'{export_suffix}/svg']
                            group_svg_paths.append(saved_paths[f'{export_suffix}/svg'])
                        if f'{export_suffix}/pdf' in saved_paths:
                            _out_paths_dict[k][f'{export_suffix}/pdf/{group_key}'] = saved_paths[f'{export_suffix}/pdf']
                            group_pdf_paths.append(saved_paths[f'{export_suffix}/pdf'])
                    else:
                        _out_paths_dict[k] = _out_paths_dict[k] | saved_paths
                ## END if should_save...
            ## END for group_key, group_df in cls._iter_split_plot_groups(source_df, split_by_column)...

            if (split_by_column is not None) and (len(group_figs) > 0):
                _out_figs_dict[k] = group_figs
                stacked_basename = basename if basename is not None else f'ScatterOverTime_{export_suffix}_{k}'
                if should_save and (len(group_svg_paths) > 0):
                    stacked_svg_path = figures_parent_folder.joinpath(f'{stacked_basename}_by_{split_by_column}.svg')
                    SVGHelpers.concatenate_svgs_vertically(svg_paths=group_svg_paths, output_path=stacked_svg_path, padding=0)
                    _out_paths_dict[k][f'{export_suffix}/svg'] = stacked_svg_path
                    print(f'\tstacked {split_by_column} rows: "{stacked_svg_path.as_posix()}"')
                if should_save and (len(group_pdf_paths) > 1):
                    stacked_pdf_path = figures_parent_folder.joinpath(f'{stacked_basename}_by_{split_by_column}.pdf')
                    PDFHelpers.concatenate_pdfs_vertically(pdf_paths=group_pdf_paths, output_path=stacked_pdf_path, padding=0)
                    _out_paths_dict[k][f'{export_suffix}/pdf'] = stacked_pdf_path
            else:
                _out_figs_dict[k] = next(iter(group_figs.values())) if group_figs else None


        # Unconstrained ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        k = "Initial"
        if should_export_initially_filtered:
            _export_stage(k, base_df, title_prefix="Initial", basename=f'ScatterOverTime_{export_suffix}_{k}')

        print(f"after {k}: {len(base_df)} rows")

        ## Constrained

        active = deepcopy(base_df)
        for k in constraint_keys:
            mask = predicates[k](active).astype(bool)
            active = deepcopy(active[mask])
            print(f"\tafter {k}: {len(active)} rows")

            if should_export_iteratively_constrainted:
                _export_stage(k, active, title_prefix=f"FILTERED | {k}", basename=None)
        ## END for k in constraint_keys...

        if should_overlay_heuristic_filter:
            n_heuristic_pass: int = int(additional_filter_predicates['is_heuristic_passing'](active).astype(bool).sum())
            print(f"heuristic passing: {n_heuristic_pass}/{len(active)} rows")

        ## Get the FINAL Figure
        # FINAL ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        k = "Final"
        _export_stage(k, active, title_prefix="Final", basename=f'ScatterOverTime_{export_suffix}')

        print(f"after {k}: {len(base_df)} rows")

        return _out_figs_dict, _out_paths_dict, _out_stats_dict



    @function_attributes(short_name=None, tags=['main', 'figure', 'output'], input_requires=[], output_provides=[], uses=['cls._perform_plot_iteratively_constrained'], used_by=[], creation_date='2026-10-28 07:44', related_items=[])
    @classmethod
    def main(cls, df_filter, plotly_helpers: PlotlyHelpers, additional_filter_predicates: Dict, figures_parent_folder: Path, active_filter_constraints: Optional[Dict]=None, should_concatenate_figures: bool=False, enable_heuristic_filtered_outputs: bool = False, split_by_column: Optional[str] = None,
            should_compute_stats: bool = True, enable_add_stats_to_figures: bool = True, debug_print: bool = True,
        ):
        """ Performs the main plotting and adds optional stats


            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import FigureS8

            ## INPUTS: earliest_delta_aligned_t_start, latest_delta_aligned_t_end
            plotly_helpers.earliest_delta_aligned_t_start = earliest_delta_aligned_t_start
            plotly_helpers.latest_delta_aligned_t_end = latest_delta_aligned_t_end
            # figures_parent_folder: Path = Path(r'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/Pho Diba Paper 2023/Figures/UneditedSources/Figure 4 - P_Short Scatter over Time/2026-10-23').resolve()
            figures_parent_folder: Path = Path(r'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/Pho Diba Paper 2023/Figures/UneditedSources/Figure S8 - Effect of Heuristic-filtering P_Short Scatter over Time - perEpoch/Inputs/2026-10-27').resolve() # Figure S8
            enable_heuristic_filtered_outputs: bool = True

            # figures_parent_folder: Path = Path(r'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/Pho Diba Paper 2023/Figures/UneditedSources/Figure 4 - P_Short Scatter over Time/2026-10-27').resolve() ## Figure 4
            # enable_heuristic_filtered_outputs: bool = False

            figures_parent_folder.mkdir(exist_ok=True)
            should_compute_stats: bool = True
            enable_add_stats_to_figures: bool = True
            _out = FigureS8.main(df_filter=df_filter, plotly_helpers=plotly_helpers, additional_filter_predicates=additional_filter_predicates, figures_parent_folder=figures_parent_folder, should_concatenate_figures=False, enable_heuristic_filtered_outputs=enable_heuristic_filtered_outputs, split_by_column=None, should_compute_stats=should_compute_stats, enable_add_stats_to_figures=enable_add_stats_to_figures)


            if should_compute_stats:
                _out_figs_dict_dict_dict, _out_paths_dict, _out_stats_tuple = _out
                pre_post_delta_stats_distribution_comparison_dict, pre_post_delta_stats_results_dict, _stats_params_kwargs = _out_stats_tuple
                _stats_output_keys = _stats_params_kwargs['output_keys']
                # if should_compute_stats:
                {k:get_dict_subset(v, _stats_output_keys) for k, v in pre_post_delta_stats_results_dict.items()}


        """
        cls.plotly_helpers = plotly_helpers

        # --- publication export setup -------------------------------------------------
        cls.plotly_helpers.export_png = False
        cls.plotly_helpers.export_html = False
        cls.plotly_helpers.is_publication = True

        # Match publication canvas used by _build_plot_callback (not the 1800x480 default)
        # plotly_helpers.resolution_multiplier = 792.0 / 1800.0  # -> ~792 x ~211; or set explicitly below ## plotly_helpers.resolution_multiplier: 0.44
        cls.plotly_helpers.resolution_multiplier = 1.0
        print(f'plotly_helpers.resolution_multiplier: {cls.plotly_helpers.resolution_multiplier}')

        cls.template = cls.plotly_helpers.update_plotly_template(is_dark_mode=False, is_publication=True)
        pio.templates.default = cls.template

        cls.PUB_FIG_SIZE = dict(width=792, height=320) # override PlotlyHelpers.fig_size_kwargs for export -- actual = (594, 240)
        # PUB_FIG_SIZE = dict(width=594, height=240) ## actual = (446, 179.4512)
        # PUB_FIG_SIZE = dict(width=(594+148), height=(240+60)) ## corrections: (148, 60), PUB_FIG_SIZE: {'width': 742, 'height': 300}, actual = (557, 224.5)
        print(f'PUB_FIG_SIZE: {cls.PUB_FIG_SIZE}') 
        # PUB_FIG_SIZE = dict() ## NO OVERRIDES

        cls._pub_font = dict(family="Arial", color="black")

        cls.figures_parent_folder = deepcopy(figures_parent_folder)

        # ## INPUTS: df_filter, additional_filter_predicates, figures_parent_folder
        # export_suffix: str = "Laps"
        # base_df = deepcopy(df_filter.original_df_dict["all_sessions_laps_time_bin_df"])

        # heuristic_filter_requirements = {
        #     # 'mld_mseq_len_ignoring_intrusions_and_repeats': 4,
        #     'mld_mseq_len_ignoring_intrusions': 4,
        #     'mld_mseq_tcov': 0.1
        # }
        # # ['mld_mseq_len_ignoring_intrusions_and_repeats', 'mld_mseq_tcov']

        # def _heuristic_col(df, col_name: str) -> str:
        #     """ gets the appropriate heuristic column name for the per-time-bin or per-epoch df columns """
        #     parent_col: str = f'parent_epoch_{col_name}'
        #     if parent_col in df.columns:
        #         return parent_col
        #     return col_name  # per-epoch: HIGHEST_*; missing → KeyError, same as today

        # additional_filter_predicates.update({
        #     'is_heuristic_passing': (lambda df: np.logical_and(
        #         (df[_heuristic_col(df, 'HIGHEST_mseq_len_ignoring_intrusions')] >= heuristic_filter_requirements['mld_mseq_len_ignoring_intrusions']),
        #         (df[_heuristic_col(df, 'HIGHEST_mseq_tcov')] >= heuristic_filter_requirements['mld_mseq_tcov']),
        #     ).astype(bool)),
        # })

        # # keep same column constraints as the widget (replay / tbin / etc.)
        # base_df = base_df.pho.constrain_df_cols(**active_filter_constraints)
        # _out_figs_dict, _out_paths_dict = _perform_plot_iteratively_constrained(
        #     base_df=base_df,
        #     additional_filter_predicates=additional_filter_predicates,
        # )
        # _out_figs_dict["Unfiltered"]

        if active_filter_constraints is None:
            # active_filter_constraints = df_filter.get_df_filter_active_constraint_dict() ## live/dynamic
            active_filter_constraints = {'trained_compute_epochs': 'laps',
                'decoder_identifier': 'pseudo2D',
                'masked_time_bin_fill_type': ('dropped',),
                'custom_replay_name': 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_2.0',
                'time_bin_size': (0.075,),
            }


        # ## INPUTS: df_filter, additional_filter_predicates, 

        # base_constraint_keys = []
        # base_constraint_keys = ["is_long_duration"]

        # output_suffix: str = '_perTimeBin'
        # base_df_dict = {'Laps': deepcopy(df_filter.original_df_dict['all_sessions_laps_time_bin_df']).pho.constrain_df_cols(**active_filter_constraints), # keep same column constraints as the widget (replay / tbin / etc.)
        #     'PBEs': deepcopy(df_filter.original_df_dict['all_sessions_ripple_time_bin_df']).pho.constrain_df_cols(**active_filter_constraints), # keep same column constraints as the widget (replay / tbin / etc.)
        # }
        # base_constraint_keys.append("is_track_body")
        # # constraint_keys = ("is_long_duration", "is_track_body")


        # # # ['all_sessions_ripple_df', 'all_sessions_laps_df']
        # output_suffix: str = '_perEpochs'
        # base_df_dict = {'Laps': deepcopy(df_filter.original_df_dict['all_sessions_laps_df']).pho.constrain_df_cols(**active_filter_constraints), # keep same column constraints as the widget (replay / tbin / etc.)
        #     'PBEs': deepcopy(df_filter.original_df_dict['all_sessions_ripple_df']).pho.constrain_df_cols(**active_filter_constraints), # keep same column constraints as the widget (replay / tbin / etc.)
        # }
        should_drop_constrained_columns: bool = False

        base_df_dict_dict = {'_perTimeBin':  {'Laps': deepcopy(df_filter.original_df_dict['all_sessions_laps_time_bin_df']).pho.constrain_df_cols(should_drop_constrained_columns=should_drop_constrained_columns, **active_filter_constraints), # keep same column constraints as the widget (replay / tbin / etc.)
                'PBEs': deepcopy(df_filter.original_df_dict['all_sessions_ripple_time_bin_df']).pho.constrain_df_cols(should_drop_constrained_columns=should_drop_constrained_columns, **active_filter_constraints), # keep same column constraints as the widget (replay / tbin / etc.)
            },
        '_perEpochs': {'Laps': deepcopy(df_filter.original_df_dict['all_sessions_laps_df']).pho.constrain_df_cols(should_drop_constrained_columns=should_drop_constrained_columns, **active_filter_constraints), # keep same column constraints as the widget (replay / tbin / etc.)
                'PBEs': deepcopy(df_filter.original_df_dict['all_sessions_ripple_df']).pho.constrain_df_cols(should_drop_constrained_columns=should_drop_constrained_columns, **active_filter_constraints), # keep same column constraints as the widget (replay / tbin / etc.)
            },
        }

        # Add Heuristic Columns: _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        base_df_dict_dict['_perEpochs']['PBEs'] = base_df_dict_dict['_perEpochs']['PBEs'].pho_mld.adding_mld_columns(overwrite_existing=True)
        base_df_dict_dict['_perEpochs']['Laps'] = base_df_dict_dict['_perEpochs']['Laps'].pho_mld.adding_mld_columns(overwrite_existing=True)

        ## Add the 'parent_epoch_*' prefixed heuristic columns to per-time_bin df from the per-epoch one:
        columns_to_add = ['mld_mseq_tcov', 'mld_mseq_len_ignoring_intrusions', 'HIGHEST_mseq_tcov', 'HIGHEST_mseq_len_ignoring_intrusions']
        base_df_dict_dict['_perTimeBin']['PBEs'] = MostLikelyDecoderDataframeAccessor.add_parent_epoch_heuristic_columns_to_time_bin_df(base_df_dict_dict['_perTimeBin']['PBEs'], base_df_dict_dict['_perEpochs']['PBEs'], columns_to_add=columns_to_add)
        base_df_dict_dict['_perTimeBin']['Laps'] = MostLikelyDecoderDataframeAccessor.add_parent_epoch_heuristic_columns_to_time_bin_df(base_df_dict_dict['_perTimeBin']['Laps'], base_df_dict_dict['_perEpochs']['Laps'], columns_to_add=columns_to_add)

        base_constraint_keys_dict = {'_perTimeBin':  ("is_long_duration", "is_track_body"),
            '_perEpochs': ("is_long_duration", ),
        }


        _out_figs_dict_dict_dict = {}
        _out_paths_dict = {}

        # Stats ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        _stats_class_kwargs = dict(n_bootstrap=10000, random_state=1337, sampling_unit='session')
        _stats_run_kwargs = dict(perform_block_permutation_test=True,
                            # sampling_unit='animal',  # or 'session'
                            sampling_unit='session',
                        )
        _stats_output_keys = ['wasserstein_1', 'wasserstein_1_bootstrap_95ci', 'mean_difference_post_minus_pre', 'mean_difference_bootstrap_95ci', 'p_value']

        _stats_params_kwargs = {'inputs_class': _stats_class_kwargs, 'inputs_run': _stats_run_kwargs, 'output_keys': _stats_output_keys}

        pre_post_delta_stats_distribution_comparison_dict: Dict[str, PrePostDeltaDistributionComparison] = {}
        pre_post_delta_stats_results_dict: Dict[str, Dict] = {}

        ## INPUTS: (pre_post_delta_stats_distribution_comparison_dict, pre_post_delta_stats_results_dict, _stats_params_kwargs)

        ## INPUTS: base_df_dict_dict, base_constraint_keys_dict
        for a_grainularity, base_df_dict in base_df_dict_dict.items(): # a_grainularity: grainularity_options = ['_perTimeBin', '_perEpochs']
            ## output_suffix: ['Laps', 'PBEs']
            ## same for laps/PBEs
            base_constraint_keys = list(deepcopy(base_constraint_keys_dict[a_grainularity])) # ["is_long_duration"] — heuristic is overlaid on PBEs only, not dropped

            ## separate for laps/PBEs
            constraint_keys = {'Laps': deepcopy(base_constraint_keys),
                'PBEs': deepcopy(base_constraint_keys),
            }

            _out_figs_dict_dict = {}
            _out_paths = {}

            for an_epoch_name, base_df in base_df_dict.items(): # an_epoch_name: ['Laps', 'PBEs']
                ## export_suffix: ['_perTimeBin', '_perEpochs']
                if isinstance(constraint_keys, dict):
                    ## separate for laps/PBEs
                    a_constraint_keys = constraint_keys[an_epoch_name] 

                else:
                    a_constraint_keys = deepcopy(constraint_keys)


                
                should_overlay_heuristic_filter: bool = (enable_heuristic_filtered_outputs and an_epoch_name.startswith('PBEs'))

                title_suffix: str = ' (heuristic filtered)' if should_overlay_heuristic_filter else ''

                epoch_kind: str = an_epoch_name  # 'Laps' or 'PBEs' — keep un-suffixed for stats keys
                if not should_concatenate_figures:
                    an_epoch_name = f"{an_epoch_name}{a_grainularity}"

                _out_figs_dict_dict[an_epoch_name], _out_paths[an_epoch_name], a_stage_stats_dict = cls._perform_plot_iteratively_constrained(base_df=base_df, additional_filter_predicates=additional_filter_predicates, figures_parent_folder=figures_parent_folder,
                                                                                                                should_export_initially_filtered = False,
                                                                                                                should_export_iteratively_constrainted = False,
                                                                                                                export_suffix=an_epoch_name,
                                                                                                                constraint_keys=a_constraint_keys,
                                                                                                                title_suffix=title_suffix,
                                                                                                                should_overlay_heuristic_filter=should_overlay_heuristic_filter,
                                                                                                                split_by_column=split_by_column,
                                                                                                                should_compute_stats=should_compute_stats, enable_add_stats_to_figures=enable_add_stats_to_figures,
                                                                                                                stats_class_kwargs=_stats_class_kwargs, stats_run_kwargs=_stats_run_kwargs,
                                                                                                            )

                # Flatten Final stats (computed on the plotted Final-constrained group_df, not unconstrained base_df)
                # g = filtered_all_sessions_ripple_time_bin_df.groupby(['animal', 'session_name', 'parent_epoch_id'], dropna=False)['pre_post_delta_category']
                # bad = g.nunique()
                # bad = bad[bad > 1]
                # print(bad.head())
                # print('n mixed groups', len(bad), 'n nan parent_epoch_id', filtered_all_sessions_ripple_time_bin_df['parent_epoch_id'].isna().sum())
                if should_compute_stats or enable_add_stats_to_figures:
                    active_key: str = '_'.join([a_grainularity, epoch_kind])  # e.g. '_perTimeBin_Laps'
                    for group_key, group_stats in a_stage_stats_dict.get('Final', {}).items():
                        result_key: str = active_key if group_key is None else f'{active_key}_{group_key}'
                        comparison_all, results_all = group_stats.get('all', (None, None))
                        if results_all is not None:
                            pre_post_delta_stats_distribution_comparison_dict[result_key] = comparison_all
                            pre_post_delta_stats_results_dict[result_key] = results_all
                        comparison_pass, results_pass = group_stats.get('heuristic_passing', (None, None))
                        if results_pass is not None:
                            pass_key: str = f'{result_key}_heuristic_passing'
                            pre_post_delta_stats_distribution_comparison_dict[pass_key] = comparison_pass
                            pre_post_delta_stats_results_dict[pass_key] = results_pass
                        if debug_print:
                            keys_to_print = []
                            if results_all is not None:
                                keys_to_print.append(result_key)
                            if results_pass is not None:
                                keys_to_print.append(f'{result_key}_heuristic_passing')
                            for printed_key in keys_to_print:
                                print(f'for "{printed_key}":')
                                print({
                                    k: v for k, v in pre_post_delta_stats_results_dict[printed_key].items()
                                    if k not in ('animal_summary', 'permutation_results', 'session_summary')
                                })
                            ## END for printed_key in keys_to_print...

                    ## END for group_key, group_stats in a_stage_stats_dict.get('Final', {}).items()...

            ## END for export_suffix, base_df in base_df_dict.items()...

            # _out_paths
            # _out_figs_dict_dict['Laps']['Final']
            # _out_figs_dict_dict['PBEs']['Final']

            _out_figs_dict_dict_dict[an_epoch_name] = _out_figs_dict_dict
            _out_paths_dict[an_epoch_name] = _out_paths


            if should_concatenate_figures:
                for k in list(_out_paths['Laps'].keys()):
                    # k: str = 'Final' or 'Initial'
                    pdf_paths = [
                        _out_paths['Laps'][k]['Laps/pdf'],
                        _out_paths['PBEs'][k]['PBEs/pdf'],
                    ]

                    _final_fig4_pdf = PDFHelpers.concatenate_pdfs_vertically(
                        pdf_paths=pdf_paths,
                        output_path=figures_parent_folder.joinpath(f'fig4_Laps_PBEs_{k}{a_grainularity}.pdf'),
                        padding=0,  # points; use e.g. 10 if you want a gap
                    )
                    
                    print(f'\t{k}: "{_final_fig4_pdf.as_posix()}"')
                ## END for k in list(_out_paths['Laps'].keys())...

        ## END for output_suffix, base_df_dict in base_df_dict_dict.items()...

        # if should_compute_stats:
        #     {k:get_dict_subset(v, _stats_output_keys) for k, v in pre_post_delta_stats_results_dict.items()}

        return _out_figs_dict_dict_dict, _out_paths_dict, (pre_post_delta_stats_distribution_comparison_dict, pre_post_delta_stats_results_dict, _stats_params_kwargs)



# ==================================================================================================================================================================================================================================================================================== #
# 2026-10-27 - Figure 4/S6/S8 Stats                                                                                                                                                                                                                                                    #
# ==================================================================================================================================================================================================================================================================================== #



# Old Bootstrapped Extrema Bins ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #

# from copy import deepcopy
# from typing import Dict

# import numpy as np
# import pandas as pd
# from neuropy.utils.indexing_helpers import transpose_nested_dict
# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import (
#     SeriesLetter, AnimalName,
#     compute_hist_variables, fig4_hist_stats_pho,
#     plot_fisher_z, plot_earthmovers, plot_kldivergence,
#     plot_by_animal_split_histogram_figure, plot_by_session_split_histogram_figure,
# )

# ## INPUTS: df_filter
# laps_df = deepcopy(df_filter.filtered_df_dict['filtered_all_sessions_laps_time_bin_df'])
# ripple_df = deepcopy(df_filter.filtered_df_dict['filtered_all_sessions_ripple_time_bin_df'])

# variable_name: str = df_filter.active_plot_variable_name  # usually 'P_Short'
# nbins: int = 11
# bin_edges = np.linspace(0.0, 1.0, nbins + 1)
# a_time_bin_size: float = float(np.atleast_1d(df_filter.time_bin_size)[0])  # or hardcode 0.075


# def _split_pre_post(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
#     df = df.dropna(subset=[variable_name, 'delta_aligned_start_t'], how='any')
#     pre_df = df[df['delta_aligned_start_t'] <= 0].copy()
#     post_df = df[df['delta_aligned_start_t'] > 0].copy()
#     return pre_df, post_df


# laps_pre_df, laps_post_df = _split_pre_post(laps_df)
# ripple_pre_df, ripple_post_df = _split_pre_post(ripple_df)

# hist_samples_df_dict: Dict[SeriesLetter, pd.DataFrame] = {
#     'a': laps_pre_df,      # PreLaps
#     'b': laps_post_df,     # PostLaps
#     'c': ripple_pre_df,    # PrePBEs
#     'd': ripple_post_df,   # PostPBEs
# }

# hist_counts_dict = {
#     k: np.histogram(df[variable_name].to_numpy(), bins=bin_edges)[0]
#     for k, df in hist_samples_df_dict.items()
# }
# hist_count_extremas_only_dict = {k: np.array([v[0], v[-1]]) for k, v in hist_counts_dict.items()}
# hist_count_extrema_ratios = {k: (float(v[0]) / float(v[-1])) for k, v in hist_counts_dict.items()}
# hist_densities_dict = {k: (v.astype(float) / np.sum(v)) for k, v in hist_counts_dict.items()}

# ## OUTPUTS: hist_counts_dict, hist_densities_dict, hist_count_extremas_only_dict, hist_count_extrema_ratios
# hist_counts_dict
# hist_count_extremas_only_dict

# ## Per-animal bags (Animal -> Letter -> DF)
# hist_animal_split_samples_df_dict: Dict[SeriesLetter, Dict[AnimalName, pd.DataFrame]] = {
#     a_letter: a_df.pho.partition_df_dict('animal') for a_letter, a_df in hist_samples_df_dict.items()
# }
# hist_animal_split_samples_df_dict: Dict[AnimalName, Dict[SeriesLetter, pd.DataFrame]] = transpose_nested_dict(hist_animal_split_samples_df_dict)

# animal_bags: Dict[AnimalName, Dict[SeriesLetter, np.ndarray]] = {}
# for an_animal_name, a_hist_samples_df_dict in hist_animal_split_samples_df_dict.items():
#     _counts, hist_count_extremas_only_dict, _ratios, _dens, (bin_edges, bin_values) = compute_hist_variables(hist_samples_df_dict=a_hist_samples_df_dict)
#     animal_bags[an_animal_name] = deepcopy(hist_count_extremas_only_dict)
# ## END for an_animal_name, a_hist_samples_df_dict in hist_animal_split_samples_df_dict.items()...

# # OUTPUTS: animal_bags
# animal_bags

# # animal_bags: {'gor01': {'a': array([6436,  198], dtype=int64),
# #   'b': array([ 748, 3032], dtype=int64),
# #   'c': array([431, 165], dtype=int64),
# #   'd': array([401, 402], dtype=int64)},
# #  'pin01': {'a': array([1178,   72], dtype=int64),
# #   'b': array([ 88, 470], dtype=int64),
# #   'c': array([329,  74], dtype=int64),
# #   'd': array([131,  91], dtype=int64)},
# #  'vvp01': {'a': array([430,  23], dtype=int64),
# #   'b': array([ 76, 206], dtype=int64),
# #   'c': array([37, 10], dtype=int64),
# #   'd': array([11,  8], dtype=int64)}}


# # np.sum(sim_diffs_dict['KLDIV'] <= 0.0) / num_bootstrap_samples   # fraction with non-positive change
# # # plus exploratory: stats.ttest_1samp(...)
# # plot_earthmovers(sim_diffs_dict['EARTHMOVERS'], ...)
# # plot_kldivergence(sim_diffs_dict['KLDIV'], ...)



# ==================================================================================================================================================================================================================================================================================== #
# 2026-10-26 - Animal/Session but not epoch Hierarchical Bootstrapping                                                                                                                                                                                                                 #
# ==================================================================================================================================================================================================================================================================================== #

# from neuropy.utils.indexing_helpers import NeuroPyDataframeAccessor


# end_kwargs = dict(trained_compute_epochs='laps')
# # end_kwargs = dict(trained_compute_epochs='non_pbe')

# filtered_all_sessions_laps_time_bin_df = deepcopy(df_filter.filtered_df_dict['filtered_all_sessions_laps_time_bin_df']) # deepcopy(all_sessions_laps_time_bin_df)
# filtered_all_sessions_laps_time_bin_df: pd.DataFrame =  filtered_all_sessions_laps_time_bin_df.neuropy.constrain_df_cols(decoder_identifier='pseudo2D', masked_time_bin_fill_type='dropped', **end_kwargs) # long_RL=0, short_LR=0, short_RL=0, known_named_decoding_epochs_type='laps'
# filtered_all_sessions_laps_time_bin_df

# filtered_all_sessions_ripple_time_bin_df = deepcopy(df_filter.filtered_df_dict['filtered_all_sessions_ripple_time_bin_df']) # deepcopy(all_sessions_ripple_time_bin_df)
# filtered_all_sessions_ripple_time_bin_df: pd.DataFrame = filtered_all_sessions_ripple_time_bin_df.neuropy.constrain_df_cols(decoder_identifier='pseudo2D', masked_time_bin_fill_type='dropped', **end_kwargs) # long_RL=0, short_LR=0, short_RL=0, known_named_decoding_epochs_type='laps'
# filtered_all_sessions_ripple_time_bin_df


# ## INPUTS: all_sessions_laps_time_bin_df, all_sessions_ripple_time_bin_df
# filtered_all_sessions_laps_df = deepcopy(df_filter.filtered_df_dict['filtered_all_sessions_laps_df']) # deepcopy(all_sessions_laps_time_bin_df)
# filtered_all_sessions_laps_df: pd.DataFrame =  filtered_all_sessions_laps_df.neuropy.constrain_df_cols(decoder_identifier='pseudo2D', masked_time_bin_fill_type='dropped', **end_kwargs) # long_RL=0, short_LR=0, short_RL=0, known_named_decoding_epochs_type='laps'
# filtered_all_sessions_laps_df

# filtered_all_sessions_ripple_df = deepcopy(df_filter.filtered_df_dict['filtered_all_sessions_ripple_df']) # deepcopy(all_sessions_ripple_time_bin_df)
# filtered_all_sessions_ripple_df: pd.DataFrame = filtered_all_sessions_ripple_df.neuropy.constrain_df_cols(decoder_identifier='pseudo2D', masked_time_bin_fill_type='dropped', **end_kwargs) # long_RL=0, short_LR=0, short_RL=0, known_named_decoding_epochs_type='laps'
# filtered_all_sessions_ripple_df


# # { 
# #        'filtered_all_sessions_ripple_df': deepcopy(df_filter.filtered_all_sessions_ripple_df),
# #         'filtered_all_sessions_ripple_time_bin_df': deepcopy(df_filter.filtered_all_sessions_ripple_time_bin_df),
# #         'filtered_all_sessions_laps_df': deepcopy(df_filter.filtered_all_sessions_laps_df),
# #         'filtered_all_sessions_laps_time_bin_df' : deepcopy(df_filter.filtered_all_sessions_laps_time_bin_df),
# # }


# ## OUTPUTS: filtered_all_sessions_laps_time_bin_df, filtered_all_sessions_ripple_time_bin_df, (filtered_all_sessions_laps_df, filtered_all_sessions_ripple_df)


# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PrePostDeltaDistributionComparison
# from neuropy.utils.mixins.indexing_helpers import get_dict_subset

# pre_post_delta_stats_distribution_comparison_dict = {}
# pre_post_delta_stats_results_dict = {}

# _class_kwargs = dict(n_bootstrap=10000, random_state=1337, sampling_unit='session')
# _run_kwargs = dict(perform_block_permutation_test=True,
#                     # sampling_unit='animal',  # or 'session'
#                     sampling_unit='session',
#                 )

# # grainularity_options = {f'by-time-bin': f'_perTimeBin', 'by-epoch':'_perEpochs'}
# grainularity_options = ['_perTimeBin', '_perEpochs']
# _filter_dfs_dict = {'_perEpochs': {'Laps': filtered_all_sessions_laps_df, 'PBEs': filtered_all_sessions_ripple_df},
#                     '_perTimeBin': {'Laps': filtered_all_sessions_laps_time_bin_df, 'PBEs': filtered_all_sessions_ripple_time_bin_df}}

# for a_grainularity, a_dict in _filter_dfs_dict.items():

#     for an_epoch_name, df in a_dict.items():
#         # for an_epoch_name, df in {'Laps': filtered_all_sessions_laps_time_bin_df, 'PBEs': filtered_all_sessions_ripple_time_bin_df}.items():
#         # for an_epoch_name, df in {'Laps': filtered_all_sessions_laps_df, 'PBEs': filtered_all_sessions_ripple_df}.items():
#         # active_key: str = an_epoch_name

#         active_key: str = '_'.join([a_grainularity, an_epoch_name])
#         # g = filtered_all_sessions_ripple_time_bin_df.groupby(['animal', 'session_name', 'parent_epoch_id'], dropna=False)['pre_post_delta_category']
#         # bad = g.nunique()
#         # bad = bad[bad > 1]
#         # print(bad.head())
#         # print('n mixed groups', len(bad), 'n nan parent_epoch_id', filtered_all_sessions_ripple_time_bin_df['parent_epoch_id'].isna().sum())

#         pre_post_delta_stats_distribution_comparison_dict[active_key] = PrePostDeltaDistributionComparison(df, **_class_kwargs)
#         pre_post_delta_stats_results_dict[active_key] = pre_post_delta_stats_distribution_comparison_dict[active_key].run(**_run_kwargs)

#         print({
#             k: v for k, v in pre_post_delta_stats_results_dict[active_key].items()
#             if k not in ('animal_summary', 'permutation_results', 'session_summary')
#         })

#         # display(pre_post_delta_results_dict[an_epoch_name]['animal_summary'])
#         # pre_post_delta_results_dict[an_epoch_name]['session_summary']

#     ## END for an_epoch_name, df in ...

# ## END for a_grainularity, a_dict in _filter_dfs_dict.items()...

# ## OUTPUTS: pre_post_delta_distribution_comparison_dict, pre_post_delta_results_dict
# # {'comparison': 'pre-delta vs post-delta', 'sampling_unit': 'session', 'p_value_effect': 'mean_difference_post_minus_pre', 'wasserstein_1': 0.34990765382605804, 'wasserstein_1_bootstrap_95ci': (0.2892773195070399, 0.4116581170350606), 'mean_difference_post_minus_pre': 0.34990765382605865, 'median_difference_post_minus_pre': 0.5671199119523261, 'p_value': 0.0009995002498750624, 'n_animals': 3, 'n_sessions': 12, 'n_laps_pre': None, 'n_laps_post': None, 'n_bins_pre': 252, 'n_bins_post': 218, 'block_length_laps': 5, 'mean_difference_bootstrap_95ci': (0.2894704007111565, 0.411809699351879)}
# # {'comparison': 'pre-delta vs post-delta', 'sampling_unit': 'session', 'p_value_effect': 'mean_difference_post_minus_pre', 'wasserstein_1': 0.2289006107063542, 'wasserstein_1_bootstrap_95ci': (0.13084789106492914, 0.31555707604088457), 'mean_difference_post_minus_pre': 0.22890061070575576, 'median_difference_post_minus_pre': 0.49047066443769377, 'p_value': 0.0014992503748125937, 'n_animals': 3, 'n_sessions': 12, 'n_laps_pre': None, 'n_laps_post': None, 'n_bins_pre': 494, 'n_bins_post': 480, 'block_length_laps': 5, 'mean_difference_bootstrap_95ci': (0.08607401931188968, 0.2986915300711381)}
# # {'comparison': 'pre-delta vs post-delta', 'sampling_unit': 'session', 'p_value_effect': 'mean_difference_post_minus_pre', 'wasserstein_1': 0.46730290285742065, 'wasserstein_1_bootstrap_95ci': (0.34951288392966806, 0.559555311766802), 'mean_difference_post_minus_pre': 0.4673029028574142, 'median_difference_post_minus_pre': 0.731446287686231, 'p_value': 0.0009995002498750624, 'n_animals': 3, 'n_sessions': 12, 'n_laps_pre': None, 'n_laps_post': None, 'n_bins_pre': 12271, 'n_bins_post': 8330, 'block_length_laps': 5, 'mean_difference_bootstrap_95ci': (0.3497991770372044, 0.5664152666835826)}
# # {'comparison': 'pre-delta vs post-delta', 'sampling_unit': 'session', 'p_value_effect': 'mean_difference_post_minus_pre', 'wasserstein_1': 0.1506443390419763, 'wasserstein_1_bootstrap_95ci': (0.09257872542339074, 0.19319929389259136), 'mean_difference_post_minus_pre': 0.15064433904014948, 'median_difference_post_minus_pre': 0.25815120846333883, 'p_value': 0.001999000499750125, 'n_animals': 3, 'n_sessions': 12, 'n_laps_pre': None, 'n_laps_post': None, 'n_bins_pre': 1614, 'n_bins_post': 1615, 'block_length_laps': 5, 'mean_difference_bootstrap_95ci': (0.09447508799250623, 0.18319471501909232)}


# _stats_output_keys = ['wasserstein_1', 'wasserstein_1_bootstrap_95ci', 'mean_difference_post_minus_pre', 'mean_difference_bootstrap_95ci', 'p_value']
# {k:get_dict_subset(v, _stats_output_keys) for k, v in pre_post_delta_stats_results_dict.items()}

# # {'_perEpochs_Laps': {'wasserstein_1': 0.34990765382605804,
# #   'wasserstein_1_bootstrap_95ci': (0.2892773195070399, 0.4116581170350606),
# #   'mean_difference_post_minus_pre': 0.34990765382605865,
# #   'mean_difference_bootstrap_95ci': (0.2894704007111565, 0.411809699351879),
# #   'p_value': 0.0009995002498750624},
# #  '_perEpochs_PBEs': {'wasserstein_1': 0.2289006107063542,
# #   'wasserstein_1_bootstrap_95ci': (0.13084789106492914, 0.31555707604088457),
# #   'mean_difference_post_minus_pre': 0.22890061070575576,
# #   'mean_difference_bootstrap_95ci': (0.08607401931188968, 0.2986915300711381),
# #   'p_value': 0.0014992503748125937},
# #  'by-time-bin_Laps': {'wasserstein_1': 0.46730290285742065,
# #   'wasserstein_1_bootstrap_95ci': (0.34951288392966806, 0.559555311766802),
# #   'mean_difference_post_minus_pre': 0.4673029028574142,
# #   'mean_difference_bootstrap_95ci': (0.3497991770372044, 0.5664152666835826),
# #   'p_value': 0.0009995002498750624},
# #  'by-time-bin_PBEs': {'wasserstein_1': 0.1506443390419763,
# #   'wasserstein_1_bootstrap_95ci': (0.09257872542339074, 0.19319929389259136),
# #   'mean_difference_post_minus_pre': 0.15064433904014948,
# #   'mean_difference_bootstrap_95ci': (0.09447508799250623, 0.18319471501909232),
# #   'p_value': 0.001999000499750125}}





# ==================================================================================================================================================================================================================================================================================== #
# 2026-10-23 - MostLikelyDecoder Column Creation                                                                                                                                                                                                                                       #
# ==================================================================================================================================================================================================================================================================================== #

DECODER_NAMES: Tuple[str, str, str, str] = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
DEFAULT_MLD_SCORE_COL_NAMES: Tuple[str, ...] = (
    'mseq_len_ratio_ignoring_intrusions_and_repeats',
    'mseq_len_ignoring_intrusions_and_repeats',
    'mseq_len_ignoring_intrusions',
    'mseq_tcov',
)


@pd.api.extensions.register_dataframe_accessor("pho_mld")
class MostLikelyDecoderDataframeAccessor:
    """ Adds most-likely-decoder (mld) index/name and `mld_{score}` columns on per-epoch DFs such as `all_sessions_ripple_df`.

    Most-likely decoder is argmax over the four decoder posterior columns, in this order:
        ('long_LR', 'long_RL', 'short_LR', 'short_RL')  # `mld_decoder_index` is pandas Int64, values 0..3 (NA if all four probs are missing/zero)
    Falls back to ('P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL'), then P_LR*P_Long / P_RL*P_Short.

    Also adds independent Long/Short direction winners (LR vs RL within each track).
    This is not the old `long_best_*` helper: that used one shared `P_LR` vs `P_RL` for both tracks. These Long/Short columns argmax within each track on its own.

    | Column | Meaning |
    |---|---|
    | `mld_decoder_index` | 0..3 among all four decoders |
    | `mld_{score}` | that one decoder's score |
    | `mld_long_decoder_index` | **Int64** 0=LR, 1=RL within Long |
    | `mld_long_decoder_name` | `'long_LR'` or `'long_RL'` |
    | `mld_long_{score}` | Long's LR/RL winner's score |
    | `mld_short_decoder_index` | **Int64** 0=LR, 1=RL within Short |
    | `mld_short_decoder_name` | `'short_LR'` or `'short_RL'` |
    | `mld_short_{score}` | Short's LR/RL winner's score |
    | `HIGHEST_{score}` | max of `{score}` across all four decoders (ignores mld; NaN only if all four are NaN) |

    Prompt:

        # Write a dataframe accessor that adds in the following to per_epoch dfs such as 'all_sessions_ripple_df'
        ## Add in, if missing:

        # 1. the most likely decoder ("mld") index based on ('long_LR', 'long_RL', 'short_LR', 'short_RL')

        # 2. the 'mld_' versions for ('mseq_len_ratio_ignoring_intrusions_and_repeats', 'mseq_len_ignoring_intrusions_and_repeats', 'mseq_tcov')

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import MostLikelyDecoderDataframeAccessor

        all_sessions_ripple_df = all_sessions_ripple_df.pho_mld.adding_mld_columns()
        all_sessions_ripple_df[['mld_decoder_index', 'mld_decoder_name', 'mld_mseq_tcov', 'mld_long_decoder_index', 'mld_long_mseq_tcov', 'mld_short_decoder_index', 'mld_short_mseq_tcov']]
    """
    decoder_names: Tuple[str, str, str, str] = DECODER_NAMES
    default_score_col_names: Tuple[str, ...] = DEFAULT_MLD_SCORE_COL_NAMES
    p_decoder_col_names: Tuple[str, str, str, str] = ('P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL')
    long_decoder_names: Tuple[str, str] = ('long_LR', 'long_RL')
    short_decoder_names: Tuple[str, str] = ('short_LR', 'short_RL')


    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._df = pandas_obj


    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"object must be a pandas Dataframe but is of type: {type(obj)}!\nobj: {obj}")


    @classmethod
    def resolve_decoder_probability_columns(cls, df: pd.DataFrame) -> List[str]:
        if all((c in df.columns) for c in cls.decoder_names):
            return list(cls.decoder_names)
        if all((c in df.columns) for c in cls.p_decoder_col_names):
            return list(cls.p_decoder_col_names)
        needed = ('P_LR', 'P_RL', 'P_Long', 'P_Short')
        if all((c in df.columns) for c in needed):
            return list(needed)
        raise KeyError(f'Could not find decoder probability columns. Need {cls.decoder_names}, {cls.p_decoder_col_names}, or {needed}. Have: {list(df.columns)}')


    @classmethod
    def _build_decoder_probability_matrix(cls, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        prob_cols: List[str] = cls.resolve_decoder_probability_columns(df)
        if tuple(prob_cols) == ('P_LR', 'P_RL', 'P_Long', 'P_Short'):
            p_lr, p_rl, p_long, p_short = (df[c].to_numpy(dtype=float) for c in prob_cols)
            probs = np.column_stack((p_lr * p_long, p_rl * p_long, p_lr * p_short, p_rl * p_short))
            return probs, list(cls.p_decoder_col_names)
        return df[prob_cols].to_numpy(dtype=float), prob_cols


    @classmethod
    def _argmax_index_and_names(cls, probs: np.ndarray, name_options: Sequence[str]) -> Tuple[pd.Series, np.ndarray]:
        """ Argmax along axis=1; returns pandas Int64 index (NA if a row is all-missing/zero) and the corresponding names. """
        n_rows: int = np.shape(probs)[0]
        is_invalid = np.isnan(probs).all(axis=1) | (np.nansum(np.abs(probs), axis=1) <= 0.0)
        valid = np.logical_not(is_invalid)
        mld_idx = pd.Series(pd.NA, index=np.arange(n_rows), dtype='Int64')
        if np.any(valid):
            mld_idx.loc[valid] = np.nanargmax(probs[valid, :], axis=1).astype(int)
        mld_names = np.full((n_rows,), '', dtype=object)
        if np.any(valid):
            mld_names[valid] = np.asarray(list(name_options))[mld_idx.loc[valid].to_numpy(dtype=int)]
        return mld_idx, mld_names


    @classmethod
    def compute_mld_decoder_index(cls, df: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
        """ Returns 4-way integer index (0..3, pandas Int64 with NA) and decoder-name array; NA/'' where all four probs are missing or zero. """
        probs, _prob_cols = cls._build_decoder_probability_matrix(df)
        return cls._argmax_index_and_names(probs, cls.decoder_names)


    @classmethod
    def compute_mld_long_short_direction_indices(cls, df: pd.DataFrame) -> Tuple[pd.Series, np.ndarray, pd.Series, np.ndarray]:
        """ Independent LR-vs-RL argmax within Long and within Short (directions need not match).

        Returns:
            mld_long_decoder_index: Int64 0=LR, 1=RL (NA if both long probs missing/zero)
            mld_long_decoder_name: 'long_LR' | 'long_RL' | ''
            mld_short_decoder_index: Int64 0=LR, 1=RL
            mld_short_decoder_name: 'short_LR' | 'short_RL' | ''
        """
        probs, _prob_cols = cls._build_decoder_probability_matrix(df)
        long_idx, long_names = cls._argmax_index_and_names(probs[:, 0:2], cls.long_decoder_names)
        short_idx, short_names = cls._argmax_index_and_names(probs[:, 2:4], cls.short_decoder_names)
        return long_idx, long_names, short_idx, short_names


    @classmethod
    def take_score_for_mld_index(cls, df: pd.DataFrame, score_col_name: str, mld_idx: Union[np.ndarray, pd.Series, pd.api.extensions.ExtensionArray], decoder_names: Optional[Sequence[str]] = None) -> np.ndarray:
        if decoder_names is None:
            decoder_names = cls.decoder_names
        decoder_score_cols: List[str] = [f'{score_col_name}_{a_decoder_name}' for a_decoder_name in decoder_names]
        missing = [c for c in decoder_score_cols if c not in df.columns]
        if len(missing) > 0:
            raise KeyError(f'missing decoder-suffixed columns for "{score_col_name}": {missing}')
        score_mat = df[decoder_score_cols].to_numpy(dtype=float)
        n_rows: int = np.shape(score_mat)[0]
        mld_idx = pd.array(mld_idx, dtype='Int64')
        out = np.full((n_rows,), np.nan, dtype=float)
        valid = np.asarray(pd.notna(mld_idx), dtype=bool)
        if np.any(valid):
            row_i = np.arange(n_rows)[valid]
            col_i = np.asarray(mld_idx[valid], dtype=int)
            out[valid] = score_mat[row_i, col_i]
        return out


    @classmethod
    def take_highest_score_across_decoders(cls, df: pd.DataFrame, score_col_name: str, decoder_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """ Row-wise max of `{score_col_name}_{decoder}` over all decoders, independent of which decoder is most likely. NaN only where every decoder is NaN. """
        if decoder_names is None:
            decoder_names = cls.decoder_names
        decoder_score_cols: List[str] = [f'{score_col_name}_{a_decoder_name}' for a_decoder_name in decoder_names]
        missing = [c for c in decoder_score_cols if c not in df.columns]
        if len(missing) > 0:
            raise KeyError(f'missing decoder-suffixed columns for "{score_col_name}": {missing}')
        return df[decoder_score_cols].astype(float).max(axis=1, skipna=True).to_numpy(dtype=float)


    @classmethod
    def add_mld_columns(cls, df: pd.DataFrame, score_col_names: Optional[Sequence[str]] = None, overwrite_existing: bool = False) -> pd.DataFrame:
        """ Adds, if missing: 4-way `mld_*` columns, independent Long/Short direction `mld_long_*` / `mld_short_*` columns, and `HIGHEST_*` (max over all four decoders) columns. """
        if score_col_names is None:
            score_col_names = cls.default_score_col_names
        out_df: pd.DataFrame = deepcopy(df)
        need_four_way: bool = overwrite_existing or ('mld_decoder_index' not in out_df.columns) or ('mld_decoder_name' not in out_df.columns)
        if need_four_way:
            mld_idx, mld_names = cls.compute_mld_decoder_index(out_df)
            if overwrite_existing or ('mld_decoder_index' not in out_df.columns):
                out_df['mld_decoder_index'] = pd.array(mld_idx, dtype='Int64')
            if overwrite_existing or ('mld_decoder_name' not in out_df.columns):
                out_df['mld_decoder_name'] = mld_names
        else:
            mld_idx = pd.array(out_df['mld_decoder_index'], dtype='Int64')
        need_long_short: bool = overwrite_existing or any((c not in out_df.columns) for c in ('mld_long_decoder_index', 'mld_long_decoder_name', 'mld_short_decoder_index', 'mld_short_decoder_name'))
        if need_long_short:
            long_idx, long_names, short_idx, short_names = cls.compute_mld_long_short_direction_indices(out_df)
            if overwrite_existing or ('mld_long_decoder_index' not in out_df.columns):
                out_df['mld_long_decoder_index'] = pd.array(long_idx, dtype='Int64')
            if overwrite_existing or ('mld_long_decoder_name' not in out_df.columns):
                out_df['mld_long_decoder_name'] = long_names
            if overwrite_existing or ('mld_short_decoder_index' not in out_df.columns):
                out_df['mld_short_decoder_index'] = pd.array(short_idx, dtype='Int64')
            if overwrite_existing or ('mld_short_decoder_name' not in out_df.columns):
                out_df['mld_short_decoder_name'] = short_names
        else:
            long_idx = pd.array(out_df['mld_long_decoder_index'], dtype='Int64')
            short_idx = pd.array(out_df['mld_short_decoder_index'], dtype='Int64')
        score_specs = (('mld', mld_idx, cls.decoder_names), ('mld_long', long_idx, cls.long_decoder_names), ('mld_short', short_idx, cls.short_decoder_names))
        for a_score_col_name in score_col_names:
            for a_prefix, an_idx, a_decoder_names in score_specs:
                out_col: str = f'{a_prefix}_{a_score_col_name}'
                if (not overwrite_existing) and (out_col in out_df.columns):
                    continue
                out_df[out_col] = cls.take_score_for_mld_index(out_df, score_col_name=a_score_col_name, mld_idx=an_idx, decoder_names=a_decoder_names)
            ## END for a_prefix, an_idx, a_decoder_names in score_specs...
            highest_col: str = f'HIGHEST_{a_score_col_name}'
            if overwrite_existing or (highest_col not in out_df.columns):
                out_df[highest_col] = cls.take_highest_score_across_decoders(out_df, score_col_name=a_score_col_name)
        ## END for a_score_col_name in score_col_names...
        return out_df


    def adding_mld_columns(self, score_col_names: Optional[Sequence[str]] = None, overwrite_existing: bool = False) -> pd.DataFrame:
        return self.add_mld_columns(df=self._df, score_col_names=score_col_names, overwrite_existing=overwrite_existing)


    @classmethod
    def add_parent_epoch_heuristic_columns_to_time_bin_df(cls, an_epoch_time_bin_df: pd.DataFrame, a_paired_epoch_df: pd.DataFrame, columns_to_add: Optional[List[str]] = None) -> pd.DataFrame:
        """ Left-merge selected per-epoch heuristic columns onto a per-time-bin DF as `parent_epoch_{col}`.

        Does not modify `a_paired_epoch_df`. Does not rewrite stored `session_name` / `custom_replay_name` on the time-bin DF;
        `_` vs `-` is normalized only on temporary join-key copies that are dropped after the merge.
        """
        if columns_to_add is None:
            columns_to_add = ['mld_mseq_tcov', 'mld_mseq_len_ignoring_intrusions', 'HIGHEST_mseq_tcov', 'HIGHEST_mseq_len_ignoring_intrusions']
        missing_src = [c for c in columns_to_add if c not in a_paired_epoch_df.columns]
        if len(missing_src) > 0:
            raise KeyError(f'a_paired_epoch_df missing columns_to_add: {missing_src}')
        epoch_src_keys = ['session_name', 'custom_replay_name', 'time_bin_size', 'label']
        missing_join = [c for c in epoch_src_keys if c not in a_paired_epoch_df.columns]
        if len(missing_join) > 0:
            raise KeyError(f'a_paired_epoch_df missing join keys: {missing_join}')
        missing_tbin = [c for c in ['session_name', 'custom_replay_name', 'time_bin_size'] if c not in an_epoch_time_bin_df.columns]
        if len(missing_tbin) > 0:
            raise KeyError(f'an_epoch_time_bin_df missing join keys: {missing_tbin}')
        out_df: pd.DataFrame = deepcopy(an_epoch_time_bin_df)
        if 'parent_epoch_label' not in out_df.columns:
            if 'ripple_idx' in out_df.columns:
                out_df['parent_epoch_label'] = out_df['ripple_idx']
            else:
                raise KeyError("an_epoch_time_bin_df needs 'parent_epoch_label' (or 'ripple_idx') to join to a_paired_epoch_df['label']")
        dest_rename = {c: f'parent_epoch_{c}' for c in columns_to_add}
        already_present = [dest for dest in dest_rename.values() if dest in out_df.columns]
        cols_needed = [c for c, dest in dest_rename.items() if dest not in already_present]
        if len(cols_needed) == 0:
            return out_df
        tmp_session_col: str = '__join_session_name'
        tmp_replay_col: str = '__join_custom_replay_name'
        right = deepcopy(a_paired_epoch_df.loc[:, epoch_src_keys + cols_needed])
        right[tmp_session_col] = right['session_name'].astype(str).str.replace('_', '-', regex=False)
        right[tmp_replay_col] = right['custom_replay_name'].astype(str).str.replace('_', '-', regex=False)
        out_df[tmp_session_col] = out_df['session_name'].astype(str).str.replace('_', '-', regex=False)
        out_df[tmp_replay_col] = out_df['custom_replay_name'].astype(str).str.replace('_', '-', regex=False)
        right = right.drop(columns=['session_name', 'custom_replay_name'])  # keep left's original identity columns
        right = right.drop_duplicates(subset=[tmp_session_col, tmp_replay_col, 'time_bin_size', 'label'], keep='first')
        right = right.rename(columns={'label': 'parent_epoch_label', **{c: dest_rename[c] for c in cols_needed}})
        merge_on = [tmp_session_col, tmp_replay_col, 'time_bin_size', 'parent_epoch_label']
        n_before: int = len(out_df)
        out_df = PandasHelpers.safe_merge(out_df, right, how='left', on=merge_on)
        assert len(out_df) == n_before, f'merge duplicated rows: {n_before} -> {len(out_df)}'
        out_df = out_df.drop(columns=[tmp_session_col, tmp_replay_col])
        return out_df


# ==================================================================================================================================================================================================================================================================================== #
# 2026-10-22 - Extract out Figure 5 Meas v. Decoded Occupancy code                                                                                                                                                                                                                     #
# ==================================================================================================================================================================================================================================================================================== #
from pyphoplacecellanalysis.Pho2D.data_exporting import LoadedPosteriorContainer
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance
from neuropy.core.user_annotations import UserAnnotationsManager
import neuropy.utils.type_aliases as types

loaded_track_limits = {'long_xlim': np.array([59.0774, 228.69]),
 'long_unit_xlim': np.array([0.205294, 0.794698]),
 'short_xlim': np.array([94.0156, 193.757]),
 'short_unit_xlim': np.array([0.326704, 0.673304]),
 'long_ylim': np.array([138.164, 146.12]),
 'long_unit_ylim': np.array([0.48012, 0.507766]),
 'short_ylim': np.array([138.021, 146.263]),
 'short_unit_ylim': np.array([0.479622, 0.508264])
}

long_track_inst, short_track_inst = LinearTrackInstance.init_LS_tracks_from_loaded_track_limits(loaded_track_limits=loaded_track_limits)




@metadata_attributes(short_name=None, tags=['figure-5', 'fig-5', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-22 09:32', related_items=['MeasuredVsDecodedOccupancy', 'perform_plot_occupancy'])
class Figure5:
    """ Plots Measured v. Decoded occupancy from h5 batch outputs

    Related:
        from neuropy.plotting.placemaps import plot_placefield_occupancy, perform_plot_occupancy
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MeasuredVsDecodedOccupancy

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import Figure5
        from neuropy.utils.indexing_helpers import NeuroPyDataframeAccessor


        # ==================================================================================================================================================================================================================================================================================== #
        # HDF5 (H5) Posterior Import from Batch                                                                                                                                                                                                                                                #
        # ==================================================================================================================================================================================================================================================================================== #
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import H5FileAggregator
        from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
        from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting, LoadedPosteriorContainer
        from neuropy.utils.indexing_helpers import flatten, flatten_dict
        from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import Figure5

        # ## INPUTS: parsed_h5_files_df
        decoded_posteriors_parsed_h5_files_df = parsed_h5_files_df[parsed_h5_files_df['file_type'] == 'decoded_posteriors']
        # print(decoded_posteriors_parsed_h5_files_df['custom_replay_name'].unique())
        # # matching_custom_replay_name_str: str = "withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]"
        # # matching_custom_replay_name_str: str = "withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]"
        # matching_custom_replay_name_str: str = 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0'
        # decoded_posteriors_parsed_h5_files_df = decoded_posteriors_parsed_h5_files_df[decoded_posteriors_parsed_h5_files_df['custom_replay_name'] == matching_custom_replay_name_str]
        # decoded_posteriors_parsed_h5_files_df
        decoded_posteriors_h5_files = [Path(v.as_posix()).resolve() for v in decoded_posteriors_parsed_h5_files_df['path'].to_list()]
        # decoded_posteriors_h5_files
        ## OUTPUTS: decoded_posteriors_h5_files
        decoded_posteriors_h5_files

        # # a_posterior_hdf5_path = Path("C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs/2025-08-14_0330PM-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(decoded_posteriors)_tbin-0.05.h5").resolve()
        # a_posterior_hdf5_path = Path("K:/scratch/collected_outputs/2025-08-14_0340PM-kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_0.2-(decoded_posteriors)_tbin-0.05.h5").resolve()

        # decoded_posteriors_h5_files = [a_posterior_hdf5_path]


        # 2026-10-20 7pm: With Laps __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        dataset_type_fields = ['p_x_given_n', 'p_x_given_n_grey', 'most_likely_positions', 'most_likely_position_indicies', 'time_bin_edges', 't_bin_centers']
        decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']

        def _build_epoch_field_dict(_out_dict, epochs_name: str):
            ''' extract both exported PBE and Laps posteriors from h5 posterior '''
            missing = [dn for dn in decoder_names if (dn not in _out_dict) or (epochs_name not in _out_dict[dn])]
            if missing:
                raise KeyError(f"H5 is missing '{epochs_name}' for decoders {missing}. Regenerated post-export files should have both 'laps' and 'ripple'.")
            epoch_only = {dn: _out_dict[dn][epochs_name] for dn in decoder_names}
            return {
                field: {dn: list(epoch_only[dn][field]) for dn in decoder_names}
                for field in dataset_type_fields
            }

        all_sessions_hdf_load_path_to_sess_key_dict: Dict[Path, str] = {}
        all_sessions_exported_posteriors_raw_dict = {}
        all_sessions_exported_ripple_data_only_dict = {}
        all_sessions_exported_laps_data_only_dict = {}

        for hdf_load_path in decoded_posteriors_h5_files:
            _out_dict, (session_key_parts, custom_replay_parts) = PosteriorExporting.load_decoded_posteriors_from_HDF5(load_path=Path(hdf_load_path).resolve(), debug_print=False) # _out_dict: Dict[types.DecoderName, Dict[KnownEpochsName, Dict]]
            session_key = '/' + '/'.join(session_key_parts)  # e.g. '/kdiba/gor01/two/2006-6-12_16-53-46'
            all_sessions_hdf_load_path_to_sess_key_dict[hdf_load_path] = session_key
            all_sessions_exported_posteriors_raw_dict[session_key] = _out_dict
            all_sessions_exported_ripple_data_only_dict[session_key] = _build_epoch_field_dict(_out_dict, 'ripple')
            all_sessions_exported_laps_data_only_dict[session_key] = _build_epoch_field_dict(_out_dict, 'laps')


        ## OUTPUTS: all_sessions_exported_posteriors_raw_dict, all_sessions_exported_ripple_data_only_dict, all_sessions_exported_laps_data_only_dict

        # all_sessions_exported_posteriors_dict, all_sessions_exported_posteriors_data_only_dict = LoadedPosteriorContainer.load_batch_hdf5_exports(exported_posterior_data_h5_files=decoded_posteriors_h5_files)
        ## OUTPUTS: all_sessions_exported_posteriors_dict, all_sessions_exported_posteriors_data_only_dict

        # Decoded ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #


        ## OUTPUTS: full_epoch_occupancy_seconds_dict_dict, full_epoch_occupancy_fractional_dict_dict
        # INPUTS: sessions_t_delta_tuple
        t_delta_df, t_delta_dict, (earliest_delta_aligned_t_start, latest_delta_aligned_t_end) = sessions_t_delta_tuple ## UNPACK

        time_bin_size: float = 0.025 # 25ms
        use_most_likely_pos_only_pseudo_occupancy: bool = True ## when `use_most_likely_pos_only_pseudo_occupancy == True`, only the most probable position from the most probable decoder is used.
        split_by_session: bool = False ## True: keep occupancy per session (no cross-session sum) so `main(..., split_by_session=True)` opens one 2x3 figure per session
        full_epoch_occupancy_seconds_dict_dict, full_epoch_occupancy_fractional_dict_dict = Figure5.compute_decoded_occupancy(all_sessions_exported_posteriors_dict=all_sessions_exported_posteriors_dict,
                    all_sessions_exported_ripple_data_only_dict=all_sessions_exported_ripple_data_only_dict, all_sessions_exported_laps_data_only_dict=all_sessions_exported_laps_data_only_dict,
                    t_delta_dict=t_delta_dict, time_bin_size=time_bin_size, use_most_likely_pos_only_pseudo_occupancy=use_most_likely_pos_only_pseudo_occupancy, split_by_session=split_by_session)


        # Measured ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        measured_x_col: str = 'binned_x_meas'

        # 'x_meas' ia only non-NA for `masked_time_bin_fill_type='nan_filled'`
        end_kwargs = dict(trained_compute_epochs='laps', custom_replay_name='withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_2.0')


        ## INPUTS: filtered_all_sessions_laps_time_bin_df
        active_filtered_all_sessions_laps_time_bin_df: pd.DataFrame = deepcopy(all_sessions_laps_time_bin_df) # deepcopy(filtered_all_sessions_laps_time_bin_df)
        # active_filtered_all_sessions_laps_time_bin_df = deepcopy(df_filter.filtered_df_dict['filtered_all_sessions_laps_time_bin_df']) # deepcopy(all_sessions_laps_time_bin_df)
        active_filtered_all_sessions_laps_time_bin_df = active_filtered_all_sessions_laps_time_bin_df[active_filtered_all_sessions_laps_time_bin_df[measured_x_col].notna()]
        active_filtered_all_sessions_laps_time_bin_df =  active_filtered_all_sessions_laps_time_bin_df.neuropy.constrain_df_cols(decoder_identifier='pseudo2D', masked_time_bin_fill_type='nan_filled', **end_kwargs)
        # active_filtered_all_sessions_laps_time_bin_df

        active_filtered_all_sessions_ripple_time_bin_df: pd.DataFrame = deepcopy(all_sessions_ripple_time_bin_df) # deepcopy(filtered_all_sessions_laps_time_bin_df)
        # active_filtered_all_sessions_laps_time_bin_df = deepcopy(df_filter.filtered_df_dict['filtered_all_sessions_ripple_time_bin_df']) # deepcopy(all_sessions_laps_time_bin_df)
        active_filtered_all_sessions_ripple_time_bin_df = active_filtered_all_sessions_ripple_time_bin_df[active_filtered_all_sessions_ripple_time_bin_df[measured_x_col].notna()]
        active_filtered_all_sessions_ripple_time_bin_df =  active_filtered_all_sessions_ripple_time_bin_df.neuropy.constrain_df_cols(decoder_identifier='pseudo2D', masked_time_bin_fill_type='nan_filled', **end_kwargs)
        # active_filtered_all_sessions_ripple_time_bin_df

        active_measured_positions_dfs_dict = {'laps': active_filtered_all_sessions_laps_time_bin_df, 'PBEs': active_filtered_all_sessions_ripple_time_bin_df}
        for epoch_name, df in active_measured_positions_dfs_dict.items():
            assert 'pre_post_delta_category' in df

        ## OUTPUTS: active_measured_positions_dfs_dict


        ## INPUTS: full_epoch_occupancy_seconds_dict_dict, full_epoch_occupancy_fractional_dict_dict, time_bin_size, measured_x_col: str = 'binned_x_meas'
        _fig5_outs = Figure5.main(full_epoch_occupancy_seconds_dict_dict=full_epoch_occupancy_seconds_dict_dict, full_epoch_occupancy_fractional_dict_dict=full_epoch_occupancy_fractional_dict_dict,
                                                                        active_measured_positions_dfs_dict=active_measured_positions_dfs_dict,
                                                                        time_bin_size=time_bin_size, measured_x_col = 'binned_x_meas',
                                                                        split_by_session=split_by_session,
                                                                    )

    """
    # @function_attributes(short_name=None, tags=['occupancy', 'fig-5'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-21 07:17', related_items=[])
    @classmethod
    def _process_result(cls, p_x_given_n_dict: Dict[types.DecoderName, List[NDArray]], t_bin_centers_flat: Optional[NDArray]=None, curr_session_t_delta: Optional[float]=None, 
            use_all_decoders_simultaneously_occupancy: bool = True, use_most_likely_pos_only_pseudo_occupancy: bool = False, 
            debug_print: bool = False):
        """ compute occupancy by summing posteriors over time. Does not handle most_likely_positions.

        When `use_all_decoders_simultaneously_occupancy == True`: don't pick a most-likely decoder per bin. Keep occupancy decoder-split: for each decoder, optionally collapse to its MAP position, then sum over time. Returns {'pre-delta'|'post-delta'|'all': {decoder_name: (n_pos,)}} when t_delta is provided.

        When `use_all_decoders_simultaneously_occupancy == False`: jointly normalize to P(x, d | t), pick the most-likely decoder per time bin, and occupancy-sum that decoder's 1D posterior.

        """
        # p_x_given_n_dict
        # t_bin_centers
        if debug_print:
                print(f'type(p_x_given_n_dict): {type(p_x_given_n_dict)}')

        assert p_x_given_n_dict is not None
        decoder_names = list(p_x_given_n_dict.keys()) ## #TODO 2026-10-22 13:11: - [ ] I get it, they truely are independent 1D decoders and not Pseudo2D
        p_x_given_n_flat = np.stack([np.concatenate(p_x_given_n_dict[a_decoder_name], axis=-1) for a_decoder_name in decoder_names])  # .shape (4, 59, 639) - (n_decoders, n_pos_bins, n_time_bins) -- comes in as 4 independent 1D decoders, not a pseduo2D. Summing each time bin over all pos gives ~4.0, not 1.0
        n_decoders, n_pos, n_t = np.shape(p_x_given_n_flat)
        assert p_x_given_n_flat.ndim == 3, f"p_x_given_n_flat.ndim: {p_x_given_n_flat.ndim} != 3, expected (n_decoders, n_pos, n_t)"
        assert n_decoders == len(decoder_names), f"n_decoders: {n_decoders} != len(decoder_names): {len(decoder_names)} ({decoder_names})"
        assert (n_decoders > 0) and (n_pos > 0) and (n_t > 0), f"empty posterior: np.shape(p_x_given_n_flat): {np.shape(p_x_given_n_flat)}"

        if (curr_session_t_delta is not None) and (t_bin_centers_flat is not None):
            ## split pre/post delta
            n_t_bins: int = len(t_bin_centers_flat)
            is_pre_delta_t_bin = (t_bin_centers_flat < curr_session_t_delta)
            assert n_t_bins == n_t, f"n_t_bins: {n_t_bins} != n_t: {n_t}"
        else:
            n_t_bins = n_t
            is_pre_delta_t_bin = None

        if use_all_decoders_simultaneously_occupancy:
            if debug_print:
                print(f'use_all_decoders_simultaneously_occupancy: TRUE')
            assert (is_pre_delta_t_bin is not None)
            # _temp_occupancy: a per_decoder_occupancy_dict ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            _temp_occupancy = {
                'pre-delta': {}, 'post-delta': {}, 'all': {},
            }
            for decoder_name, p_x_given_n in p_x_given_n_dict.items():
                # np.nansum(p_x_given_n, axis=-1)
                active_p_x_given_n_flat = np.concatenate(p_x_given_n, axis=-1)
                assert n_t_bins == np.shape(active_p_x_given_n_flat)[-1], f"n_t_bins: {n_t_bins} != np.shape(active_p_x_given_n_flat)[-1]: {np.shape(active_p_x_given_n_flat)[-1]}"
                _pos_mass_t = np.nansum(active_p_x_given_n_flat, axis=0)
                is_valid_t = np.isfinite(_pos_mass_t) & (_pos_mass_t > 0)  # drop empty/NaN time bins from occupancy

                if use_most_likely_pos_only_pseudo_occupancy:
                    most_likely_pos_idxs = np.nanargmax(active_p_x_given_n_flat, axis=0) ## most likely positions for this decoder for each t_bin
                    _active_map_only = np.zeros_like(active_p_x_given_n_flat)
                    t_idx = np.arange(n_t_bins)
                    _active_map_only[most_likely_pos_idxs, t_idx] = 1.0
                    _active_map_only[:, ~is_valid_t] = 0.0
                    active_p_x_given_n_flat = _active_map_only

                _temp_occupancy['pre-delta'][decoder_name] = np.nansum(active_p_x_given_n_flat[:, is_pre_delta_t_bin & is_valid_t], axis=-1)
                _temp_occupancy['post-delta'][decoder_name] = np.nansum(active_p_x_given_n_flat[:, (~is_pre_delta_t_bin) & is_valid_t], axis=-1)
                _temp_occupancy['all'][decoder_name] = np.nansum(active_p_x_given_n_flat[:, is_valid_t], axis=-1)

                # else:
                #     _temp_occupancy = {
                #                     'pre-delta': np.nansum(p_x_given_n[:, is_pre_delta & is_valid_t], axis=-1),
                #                     'post-delta': np.nansum(p_x_given_n[:, (~is_pre_delta) & is_valid_t], axis=-1),
                #                     'all': np.nansum(p_x_given_n[:, is_valid_t], axis=-1),
                #                 }
            ## END for decoder_name, p_x_given_n in p_x_given_n_dict.items()....

        else:
            ## independently-normalized 1D: each decoder sums to ~1 over pos, so stacked (decoder, pos) mass is ~n_decoders per time bin (not 1.0)
            _per_decoder_pos_sum = np.nansum(p_x_given_n_flat, axis=1)  # (n_decoders, n_t)
            _input_joint_mass = np.nansum(p_x_given_n_flat, axis=(0, 1))  # (n_t,)
            _decoder_has_mass = np.isfinite(_per_decoder_pos_sum) & (_per_decoder_pos_sum > 0)
            assert np.any(_decoder_has_mass), f"p_x_given_n_flat has no finite posterior mass; np.shape(p_x_given_n_flat): {np.shape(p_x_given_n_flat)}"
            assert np.all(np.isclose(_per_decoder_pos_sum[_decoder_has_mass], 1.0, rtol=1e-3, atol=1e-3)), f"independent 1D posteriors must sum to 1 over pos (axis=1); got min={np.nanmin(_per_decoder_pos_sum[_decoder_has_mass])}, max={np.nanmax(_per_decoder_pos_sum[_decoder_has_mass])}"
            _all_decoders_have_mass_t = np.all(_decoder_has_mass, axis=0)
            if np.any(_all_decoders_have_mass_t):
                assert np.all(np.isclose(_input_joint_mass[_all_decoders_have_mass_t], float(n_decoders), rtol=1e-3, atol=1e-3)), f"stacked independent 1D posteriors must sum over (decoder, pos) to n_decoders ({n_decoders}), not 1.0 (that would already be a joint Pseudo2D); got min={np.nanmin(_input_joint_mass[_all_decoders_have_mass_t])}, max={np.nanmax(_input_joint_mass[_all_decoders_have_mass_t])}"
            # np.nansum(p_x_given_n_flat, axis=1) #.shape -- all ones because it's normalized to each decoder independently, have to undo this normalization (like we would when building the Pseudo2D representation from 4 separate decoded probabilities
            # _decoder_rel_renormalized_p_x_given_n_flat = p_x_given_n_flat.copy() # deepcopy(p_x_given_n_flat)
            # _decoder_rel_renormalized_p_x_given_n_flat = _decoder_rel_renormalized_p_x_given_n_flat / np.nansum(_decoder_rel_renormalized_p_x_given_n_flat, axis=0, keepdims=True) #.shape - (59, 639) -- sum over each decoder to find out the likelihood of each
            # # decoder_rel_renormalized_p_x_given_n_flat.shape # (4, 59, 639)
            # # np.nansum(decoder_rel_renormalized_p_x_given_n_flat, axis=1).shape #.shape -- (4, 639)
            # most_likely_decoder_idx = np.argmax(np.nanmax(_decoder_rel_renormalized_p_x_given_n_flat, axis=1), axis=0) #.shape -- (639,) - (n_time_bins, ) # SyntaxError('invalid syntax', ('<string>', 2, 48, 'p_x_given_n_flat_from_most_likely_decoder_flat = np.squeeze(_decoder_rel_renormalized_p_x_given_n_flat[most_likely_decoder_idx, :, :]) # .shape (59, 639) - (n_pos_bins, n_time_bins)\n'))
            if debug_print:
                print(f'np.shape(p_x_given_n_flat): {np.shape(p_x_given_n_flat)}')

            ## Pseudo2D-style joint P(x, d | t): normalize over (decoder, pos) per time bin, not P(d | x, t) via axis=0
            _decoder_rel_renormalized_p_x_given_n_flat = p_x_given_n_flat.copy() # deepcopy(p_x_given_n_flat)
            _joint_denom = np.nansum(_decoder_rel_renormalized_p_x_given_n_flat, axis=(0, 1), keepdims=True) #.shape (1, 1, n_time_bins) -- sum over decoder AND pos so each time bin totals ~1.0
            _joint_denom = np.where(_joint_denom == 0, np.nan, _joint_denom)
            _decoder_rel_renormalized_p_x_given_n_flat = _decoder_rel_renormalized_p_x_given_n_flat / _joint_denom #.shape (4, 59, 639) - (n_decoders, n_pos_bins, n_time_bins)
            is_valid_t = np.isfinite(_joint_denom).reshape(-1)  # drop empty/NaN time bins from occupancy
            assert np.shape(_decoder_rel_renormalized_p_x_given_n_flat) == (n_decoders, n_pos, n_t), f"np.shape(_decoder_rel_renormalized_p_x_given_n_flat): {np.shape(_decoder_rel_renormalized_p_x_given_n_flat)} != {(n_decoders, n_pos, n_t)}"
            _joint_mass = np.nansum(_decoder_rel_renormalized_p_x_given_n_flat, axis=(0, 1))  # (n_t,)
            assert np.any(is_valid_t), f"no valid time bins after joint normalization; n_t: {n_t}"
            assert np.all(np.isclose(_joint_mass[is_valid_t], 1.0, rtol=1e-3, atol=1e-3)), f"joint P(x, d | t) must sum to 1 over (decoder, pos) per time bin; got min={np.nanmin(_joint_mass[is_valid_t])}, max={np.nanmax(_joint_mass[is_valid_t])}"
            # decoder_rel_renormalized_p_x_given_n_flat.shape # (4, 59, 639)
            if debug_print:
                print(f'np.shape(_decoder_rel_renormalized_p_x_given_n_flat): {np.shape(_decoder_rel_renormalized_p_x_given_n_flat)}')
                print(f'np.nanmean(P_d): {np.nanmean(np.nansum(_decoder_rel_renormalized_p_x_given_n_flat, axis=1), axis=1)}')  # ~uniform if 1D posteriors were independently normalized

            # np.nansum(decoder_rel_renormalized_p_x_given_n_flat, axis=1).shape #.shape -- (4, 639)
            # most_likely_decoder_idx = np.argmax(np.nansum(p_x_given_n_flat, axis=1), axis=0) #.shape -- (639,) - (n_time_bins, ) # SyntaxError('invalid syntax', ('<string>', 2, 48, 'p_x_given_n_flat_from_most_likely_decoder_flat = np.squeeze(_decoder_rel_renormalized_p_x_given_n_flat[most_likely_decoder_idx, :, :]) # .shape (59, 639) - (n_pos_bins, n_time_bins)\n'))
            most_likely_decoder_idx = np.atleast_1d(np.squeeze(np.argmax(np.nansum(_decoder_rel_renormalized_p_x_given_n_flat, axis=1), axis=0))) #.shape -- (639,) - (n_time_bins, ) - sum over all positions first
            assert np.shape(most_likely_decoder_idx) == (n_t,), f"np.shape(most_likely_decoder_idx): {np.shape(most_likely_decoder_idx)} != (n_t,): {(n_t,)}"
            assert np.all((most_likely_decoder_idx >= 0) & (most_likely_decoder_idx < n_decoders)), f"most_likely_decoder_idx out of range [0, {n_decoders}); unique: {np.unique(most_likely_decoder_idx)}"
            if debug_print:
                print(f'np.shape(most_likely_decoder_idx): {np.shape(most_likely_decoder_idx)}')

            ## occupancy from the original 1D posterior of d*, then renormalize over position (not from P(d | x))
            p_x_given_n_flat_from_most_likely_decoder_flat = np.vstack([np.atleast_1d(p_x_given_n_flat[a_most_likely_idx, :, i]) for i, a_most_likely_idx in enumerate(most_likely_decoder_idx)]).T # .shape (59, 639) - (n_pos_bins, n_time_bins)
            # p_x_given_n_flat_from_most_likely_decoder_flat = np.squeeze(_decoder_rel_renormalized_p_x_given_n_flat[most_likely_decoder_idx, :, :]) # .shape (59, 639) - (n_pos_bins, n_time_bins)
            _winner_pos_denom = np.nansum(p_x_given_n_flat_from_most_likely_decoder_flat, axis=0, keepdims=True)
            _winner_pos_denom = np.where(_winner_pos_denom == 0, np.nan, _winner_pos_denom)
            p_x_given_n_flat_from_most_likely_decoder_flat = p_x_given_n_flat_from_most_likely_decoder_flat / _winner_pos_denom
            assert np.shape(p_x_given_n_flat_from_most_likely_decoder_flat) == (n_pos, n_t), f"np.shape(p_x_given_n_flat_from_most_likely_decoder_flat): {np.shape(p_x_given_n_flat_from_most_likely_decoder_flat)} != {(n_pos, n_t)}"
            _winner_pos_sum = np.nansum(p_x_given_n_flat_from_most_likely_decoder_flat, axis=0)  # (n_t,)
            _winner_valid_t = is_valid_t & np.isfinite(_winner_pos_sum) & (_winner_pos_sum > 0)
            assert np.any(_winner_valid_t), f"winner 1D posterior has no finite mass on valid time bins; n_valid_t: {np.sum(is_valid_t)}"
            assert np.all(np.isclose(_winner_pos_sum[_winner_valid_t], 1.0, rtol=1e-3, atol=1e-3)), f"winner 1D posterior must sum to 1 over pos per time bin; got min={np.nanmin(_winner_pos_sum[_winner_valid_t])}, max={np.nanmax(_winner_pos_sum[_winner_valid_t])}"
            if debug_print:
                print(f'np.shape(p_x_given_n_flat_from_most_likely_decoder_flat): {np.shape(p_x_given_n_flat_from_most_likely_decoder_flat)}')

            if (is_pre_delta_t_bin is not None):
                ## split pre/post delta
                assert n_t_bins == np.shape(p_x_given_n_flat_from_most_likely_decoder_flat)[-1], f"n_t_bins: {n_t_bins} != np.shape(p_x_given_n_flat_from_most_likely_decoder_flat)[-1]: {np.shape(p_x_given_n_flat_from_most_likely_decoder_flat)[-1]}"

                if use_most_likely_pos_only_pseudo_occupancy:
                    most_likely_pos_idxs = np.nanargmax(p_x_given_n_flat_from_most_likely_decoder_flat, axis=0) ## most likely positions for each t_bin
                    active_p_x_given_n_flat_from_most_likely_decoder_flat = np.zeros_like(p_x_given_n_flat_from_most_likely_decoder_flat)
                    # active_p_x_given_n_flat_from_most_likely_decoder_flat[most_likely_pos_idxs] = 1.0 ## wrong, sets entire line
                    t_idx = np.arange(n_t_bins)
                    active_p_x_given_n_flat_from_most_likely_decoder_flat[most_likely_pos_idxs, t_idx] = 1.0
                    active_p_x_given_n_flat_from_most_likely_decoder_flat[:, ~is_valid_t] = 0.0

                else:
                    active_p_x_given_n_flat_from_most_likely_decoder_flat = p_x_given_n_flat_from_most_likely_decoder_flat

                # _temp_occupancy = np.nansum(p_x_given_n_flat_from_most_likely_decoder_flat, axis=-1)
                _temp_occupancy = {
                                    'pre-delta': np.nansum(active_p_x_given_n_flat_from_most_likely_decoder_flat[:, is_pre_delta_t_bin & is_valid_t], axis=-1),
                                    'post-delta': np.nansum(active_p_x_given_n_flat_from_most_likely_decoder_flat[:, (~is_pre_delta_t_bin) & is_valid_t], axis=-1),
                                    'all': np.nansum(active_p_x_given_n_flat_from_most_likely_decoder_flat[:, is_valid_t], axis=-1),
                                }
            else:
                ## old method of not using a dict
                ## sum up these posteriors to get the REALIO:
                _temp_occupancy = np.nansum(p_x_given_n_flat_from_most_likely_decoder_flat[:, is_valid_t], axis=-1)
                if debug_print:
                    print(f'np.shape(_temp_occupancy): {np.shape(_temp_occupancy)}')

        return _temp_occupancy


    @classmethod
    def _process_most_likely_positions(cls, most_likely_positions_dict: Dict[types.DecoderName, List[NDArray]], p_x_given_n_dict: Dict[types.DecoderName, List[NDArray]], debug_print: bool = False) -> NDArray:
        """ Pick the most-likely decoder per time bin from independently-normalized 1D posteriors, then return that decoder's most_likely_positions.

        Occupancy is handled separately by `_process_result`.
        """
        assert p_x_given_n_dict is not None
        assert most_likely_positions_dict is not None
        decoder_names = list(p_x_given_n_dict.keys()) ## #TODO 2026-10-22 13:11: - [ ] I get it, they truely are independent 1D decoders and not Pseudo2D
        p_x_given_n_flat = np.stack([np.concatenate(p_x_given_n_dict[a_decoder_name], axis=-1) for a_decoder_name in decoder_names])  # .shape (4, 59, 639) - (n_decoders, n_pos_bins, n_time_bins)
        n_decoders, n_pos, n_t = np.shape(p_x_given_n_flat)
        assert p_x_given_n_flat.ndim == 3, f"p_x_given_n_flat.ndim: {p_x_given_n_flat.ndim} != 3, expected (n_decoders, n_pos, n_t)"
        assert n_decoders == len(decoder_names), f"n_decoders: {n_decoders} != len(decoder_names): {len(decoder_names)} ({decoder_names})"
        assert (n_decoders > 0) and (n_pos > 0) and (n_t > 0), f"empty posterior: np.shape(p_x_given_n_flat): {np.shape(p_x_given_n_flat)}"

        ## independently-normalized 1D: each decoder sums to ~1 over pos, so stacked (decoder, pos) mass is ~n_decoders per time bin (not 1.0)
        _per_decoder_pos_sum = np.nansum(p_x_given_n_flat, axis=1)  # (n_decoders, n_t)
        _input_joint_mass = np.nansum(p_x_given_n_flat, axis=(0, 1))  # (n_t,)
        _decoder_has_mass = np.isfinite(_per_decoder_pos_sum) & (_per_decoder_pos_sum > 0)
        assert np.any(_decoder_has_mass), f"p_x_given_n_flat has no finite posterior mass; np.shape(p_x_given_n_flat): {np.shape(p_x_given_n_flat)}"
        assert np.all(np.isclose(_per_decoder_pos_sum[_decoder_has_mass], 1.0, rtol=1e-3, atol=1e-3)), f"independent 1D posteriors must sum to 1 over pos (axis=1); got min={np.nanmin(_per_decoder_pos_sum[_decoder_has_mass])}, max={np.nanmax(_per_decoder_pos_sum[_decoder_has_mass])}"
        _all_decoders_have_mass_t = np.all(_decoder_has_mass, axis=0)
        if np.any(_all_decoders_have_mass_t):
            assert np.all(np.isclose(_input_joint_mass[_all_decoders_have_mass_t], float(n_decoders), rtol=1e-3, atol=1e-3)), f"stacked independent 1D posteriors must sum over (decoder, pos) to n_decoders ({n_decoders}), not 1.0 (that would already be a joint Pseudo2D); got min={np.nanmin(_input_joint_mass[_all_decoders_have_mass_t])}, max={np.nanmax(_input_joint_mass[_all_decoders_have_mass_t])}"
        if debug_print:
            print(f'np.shape(p_x_given_n_flat): {np.shape(p_x_given_n_flat)}')

        ## Pseudo2D-style joint P(x, d | t): normalize over (decoder, pos) per time bin, not P(d | x, t) via axis=0
        _decoder_rel_renormalized_p_x_given_n_flat = p_x_given_n_flat.copy() # deepcopy(p_x_given_n_flat)
        _joint_denom = np.nansum(_decoder_rel_renormalized_p_x_given_n_flat, axis=(0, 1), keepdims=True) #.shape (1, 1, n_time_bins) -- sum over decoder AND pos so each time bin totals ~1.0
        _joint_denom = np.where(_joint_denom == 0, np.nan, _joint_denom)
        _decoder_rel_renormalized_p_x_given_n_flat = _decoder_rel_renormalized_p_x_given_n_flat / _joint_denom #.shape (4, 59, 639) - (n_decoders, n_pos_bins, n_time_bins)
        assert np.shape(_decoder_rel_renormalized_p_x_given_n_flat) == (n_decoders, n_pos, n_t), f"np.shape(_decoder_rel_renormalized_p_x_given_n_flat): {np.shape(_decoder_rel_renormalized_p_x_given_n_flat)} != {(n_decoders, n_pos, n_t)}"
        most_likely_decoder_idx = np.atleast_1d(np.squeeze(np.argmax(np.nansum(_decoder_rel_renormalized_p_x_given_n_flat, axis=1), axis=0))) #.shape -- (639,) - (n_time_bins, ) - sum over all positions first
        assert np.shape(most_likely_decoder_idx) == (n_t,), f"np.shape(most_likely_decoder_idx): {np.shape(most_likely_decoder_idx)} != (n_t,): {(n_t,)}"
        assert np.all((most_likely_decoder_idx >= 0) & (most_likely_decoder_idx < n_decoders)), f"most_likely_decoder_idx out of range [0, {n_decoders}); unique: {np.unique(most_likely_decoder_idx)}"
        if debug_print:
            print(f'np.shape(most_likely_decoder_idx): {np.shape(most_likely_decoder_idx)}')

        most_likely_positions_flat = np.stack([np.concatenate(most_likely_positions_dict[a_decoder_name]) for a_decoder_name in decoder_names]) # (4, 639) - (n_decoders, n_time_bins)
        most_likely_positions_from_most_likely_decoder_flat = np.concatenate([np.atleast_1d(most_likely_positions_flat[a_most_likely_idx, i]) for i, a_most_likely_idx in enumerate(most_likely_decoder_idx)]) ## one for each time bin #.shape (639,)
        return most_likely_positions_from_most_likely_decoder_flat


    @classmethod
    def compute_decoded_occupancy(cls, all_sessions_exported_posteriors_dict, all_sessions_exported_ripple_data_only_dict, all_sessions_exported_laps_data_only_dict, t_delta_dict, 
            time_bin_size: float = 0.025, use_all_decoders_simultaneously_occupancy: bool = True, use_most_likely_pos_only_pseudo_occupancy: bool = False, split_by_session: bool = False, **kwargs):
        ## INPUTS: time_bin_size, 
        # a_posterior_container: LoadedPosteriorContainer = all_sessions_exported_posteriors_dict['/kdiba/gor01/two/2006-6-12_16-53-46']

        _temp_out_occupancy_result_dict = {'laps': {}, 'PBEs': {}}  # session_name -> occupancy dict; `.values()` still works with `_sum_period_occupancy`

        most_likely_positions_from_most_likely_decoder_classification_df = []
        for a_sess_path, a_posterior_container in all_sessions_exported_posteriors_dict.items():
            # a_sess_name: '/kdiba/vvp01/two/2006-4-10_12-58-3'
            # print(f'a_sess_path: "{a_sess_path}"')
            a_sess_ctxt: IdentifyingContext = IdentifyingContext.try_init_from_session_key(a_sess_path.removeprefix('/'), separator='/')
            if '/' in a_sess_ctxt.session_name:
                print(f'WARN: / detected in a_sess_ctxt.session_name: a_sess_ctxt: {a_sess_ctxt}\n\ta_sess_path: "{a_sess_path}"')
                _old_session_name: str = a_sess_ctxt.session_name
                session_name, *extra_parameters = _old_session_name.split('/')
                session_name
                print(f'\textra_parameters: {extra_parameters}, _old_session_name: "{_old_session_name}" -> new_session_name: "{session_name}"')
                a_sess_ctxt = a_sess_ctxt.overwriting_context(session_name=session_name)

            a_sess_global_uid = a_sess_ctxt.get_description_as_session_global_uid() # kdiba|gor01|one|2006-6-08_14-26-15
            a_session_key: str = a_sess_path # all_sessions_hdf_load_path_to_sess_key_dict[a_sess_path]
            a_sess_name: str = a_sess_ctxt.get_description(separator='_') # 'kdiba_gor01_one_2006-6-08_14-26-15' -- matches the active_plot_df

            curr_session_t_delta: float = t_delta_dict.get(a_sess_name, {}).get('t_delta', None)
            assert curr_session_t_delta is not None, f"curr_session_t_delta is None for context: {a_sess_ctxt}!"
            # Handle both Laps and PBEs:
            # _a_sess_laps_PBEs_dict = {'laps': a_posterior_container.ripple_data_field_dict, 'PBEs': a_posterior_container.ripple_data_field_dict}
            _a_sess_laps_PBEs_dict = {'laps': all_sessions_exported_laps_data_only_dict.get(a_session_key, None), 'PBEs': all_sessions_exported_ripple_data_only_dict.get(a_session_key, None)}

            for an_epoch_name, a_result in _a_sess_laps_PBEs_dict.items():
                if a_result is None:
                    print(F'WARN: "{an_epoch_name}" is None for a_sess_path: "{a_sess_path}". Skipping.')
                    continue
                else:
                    ## build occupancy by summing over the 'most_likely_positions':
                    t_bin_centers_flat: NDArray = np.concatenate(list(a_result['t_bin_centers'].values())[0]) ## they're all the same for each decoder, so just get the first decoder's values
                    # print(f'np.shape(t_bin_centers_flat): {np.shape(t_bin_centers_flat)}, t_bin_centers_flat: {t_bin_centers_flat}')
                    _temp_occupancy = cls._process_result(p_x_given_n_dict=a_result['p_x_given_n'], t_bin_centers_flat=t_bin_centers_flat, curr_session_t_delta=curr_session_t_delta, use_most_likely_pos_only_pseudo_occupancy=use_most_likely_pos_only_pseudo_occupancy, use_all_decoders_simultaneously_occupancy=use_all_decoders_simultaneously_occupancy, **kwargs)
                    _temp_out_occupancy_result_dict[an_epoch_name][a_sess_name] = _temp_occupancy
            ## END for an_epoch....

            # list(a_posterior_container.ripple_data_field_dict.keys()) # list(a_posterior_container.ripple_data_field_dict.keys()): ['p_x_given_n', 'p_x_given_n_grey', 'most_likely_positions', 'most_likely_position_indicies', 'time_bin_edges', 't_bin_centers']

            most_likely_positions_dict: Dict[types.DecoderName, List[NDArray]] = a_posterior_container.ripple_data_field_dict['most_likely_positions']
            p_x_given_n_dict: Dict[types.DecoderName, List[NDArray]] = a_posterior_container.ripple_data_field_dict['p_x_given_n']
            t_bin_centers: List[NDArray] = list(a_posterior_container.ripple_data_field_dict['t_bin_centers'].values())[0] ## they're all the same for each decoder, so just get the first decoder's values
            t_bin_centers_flat: NDArray = np.concatenate(t_bin_centers)
            n_time_bins: int = len(t_bin_centers_flat)

            most_likely_positions_from_most_likely_decoder_flat = cls._process_most_likely_positions(most_likely_positions_dict=most_likely_positions_dict, p_x_given_n_dict=p_x_given_n_dict)
            
            a_most_likely_positions_from_most_likely_decoder_classification_df: pd.DataFrame = long_track_inst.build_x_position_classification_df(x_arr=most_likely_positions_from_most_likely_decoder_flat)
            ## Add in the other important columns
            a_most_likely_positions_from_most_likely_decoder_classification_df['t_bin_center'] = deepcopy(t_bin_centers_flat)
            a_most_likely_positions_from_most_likely_decoder_classification_df = a_most_likely_positions_from_most_likely_decoder_classification_df.across_session_identity.add_session_df_columns(session_name=a_sess_global_uid, curr_session_t_delta=curr_session_t_delta, time_col='t_bin_center')
            a_most_likely_positions_from_most_likely_decoder_classification_df['session_uid'] = a_sess_global_uid
            a_most_likely_positions_from_most_likely_decoder_classification_df = a_most_likely_positions_from_most_likely_decoder_classification_df.across_session_identity.split_session_key_col_to_fmt_animal_exper_cols(session_key_col='session_uid', separator='|')
            # a_most_likely_positions_from_most_likely_decoder_classification_df['session_name'] = a_sess_ctxt.session_name # '2006-6-08_14-26-15' -- makes logical sense but isn't what the other df does
            a_most_likely_positions_from_most_likely_decoder_classification_df['session_name'] = a_sess_name # 'kdiba_gor01_one_2006-6-08_14-26-15' -- matches the active_plot_df
            
            # Always apply the transformation
            # a_most_likely_positions_from_most_likely_decoder_classification_df = a_most_likely_positions_from_most_likely_decoder_classification_df.across_session_identity.split_session_key_col_to_fmt_animal_exper_cols(session_key_col='session_name')
            # FAT_df = FAT_df.across_session_identity.split_session_key_col_to_fmt_animal_exper_cols(session_key_col='session_name')

            most_likely_positions_from_most_likely_decoder_classification_df.append(a_most_likely_positions_from_most_likely_decoder_classification_df)
            # FINAL OUTPUT: most_likely_positions_from_most_likely_decoder_flat
        ## END for a_sess_path, a_posterior_container in all_sessions_exported_posteriors_dict.items()....

        most_likely_positions_from_most_likely_decoder_classification_df: pd.DataFrame = pd.concat(most_likely_positions_from_most_likely_decoder_classification_df)
        most_likely_positions_from_most_likely_decoder_classification_df

        ## OUTPUTS: most_likely_positions_from_most_likely_decoder_classification_df, _temp_out_occupancy_result_dict

        def _sum_period_occupancy(v_list, pre_post_delta_period, scale: float = 1.0):
            """Sum occupancy across sessions. Decoder-split values stay `{decoder_name: (n_pos,)}`; otherwise a single `(n_pos,)` vector."""
            sample = v_list[0][pre_post_delta_period]
            if isinstance(sample, dict):
                decoder_names = list(sample.keys())
                return {decoder_name: (np.nansum(np.vstack([v[pre_post_delta_period][decoder_name] for v in v_list]), axis=0) * scale) for decoder_name in decoder_names}
            return np.nansum(np.vstack([v[pre_post_delta_period] for v in v_list]), axis=0) * scale


        def _scale_period_occupancy(v, pre_post_delta_period, scale: float = 1.0):
            """Scale one session's occupancy for a period. Decoder-split values stay `{decoder_name: (n_pos,)}`."""
            sample = v[pre_post_delta_period]
            if isinstance(sample, dict):
                return {decoder_name: (arr * scale) for decoder_name, arr in sample.items()}
            return sample * scale


        def _normalize_occupancy(v):
            if isinstance(v, dict):
                return {decoder_name: (arr / np.nansum(arr)) for decoder_name, arr in v.items()}
            occ_sum = np.nansum(v)
            if occ_sum == 0:
                return v
            return v / occ_sum


        ## Decoded Occupancies:
        laps_pbe_epoch_names = ['pre-delta', 'post-delta', 'all']
        full_epoch_occupancy_seconds_dict_dict = {}
        full_epoch_occupancy_fractional_dict_dict = {}

        if split_by_session:
            session_names = []
            for laps_pbe_epoch_name, sess_dict in _temp_out_occupancy_result_dict.items():
                for session_name in sess_dict.keys():
                    if session_name not in session_names:
                        session_names.append(session_name)
                ## END for session_name in sess_dict.keys()....
            ## END for laps_pbe_epoch_name, sess_dict in _temp_out_occupancy_result_dict.items()....

            for session_name in session_names:
                full_epoch_occupancy_seconds_dict_dict[session_name] = {}
                full_epoch_occupancy_fractional_dict_dict[session_name] = {}
                for pre_post_delta_period in laps_pbe_epoch_names:
                    seconds_dict = {}
                    for laps_pbe_epoch_name, sess_dict in _temp_out_occupancy_result_dict.items():
                        if session_name not in sess_dict:
                            continue
                        seconds_dict[laps_pbe_epoch_name] = _scale_period_occupancy(sess_dict[session_name], pre_post_delta_period, scale=float(time_bin_size))
                    ## END for laps_pbe_epoch_name, sess_dict in _temp_out_occupancy_result_dict.items()....
                    full_epoch_occupancy_seconds_dict_dict[session_name][pre_post_delta_period] = seconds_dict
                    full_epoch_occupancy_fractional_dict_dict[session_name][pre_post_delta_period] = {laps_pbe_epoch_name: _normalize_occupancy(v) for laps_pbe_epoch_name, v in seconds_dict.items()}
                ## END for pre_post_delta_period in laps_pbe_epoch_names....
            ## END for session_name in session_names....
        else:
            for pre_post_delta_period in laps_pbe_epoch_names:
                # pre_post_delta_period = 'pre-delta'
                full_epoch_occupancy_n_time_bins_dict = {laps_pbe_epoch_name: _sum_period_occupancy(list(v_dict.values()), pre_post_delta_period, scale=1.0) for laps_pbe_epoch_name, v_dict in _temp_out_occupancy_result_dict.items() if len(v_dict) > 0} # .shape vstack: {'laps': (11, 59), 'PBEs': (11, 59)}, in units of # time bins; decoder-split: {'laps': {decoder: (59,)}}
                # full_epoch_occupancy_n_time_bins_dict
                full_epoch_occupancy_seconds_dict_dict[pre_post_delta_period] = {laps_pbe_epoch_name: _sum_period_occupancy(list(v_dict.values()), pre_post_delta_period, scale=float(time_bin_size)) for laps_pbe_epoch_name, v_dict in _temp_out_occupancy_result_dict.items() if len(v_dict) > 0} # {laps_pbe_epoch_name:{pre_post_delta_period:(v * float(time_bin_size)) for pre_post_delta_period, v in v_dict.items()} for laps_pbe_epoch_name, v_dict in full_epoch_occupancy_n_time_bins_dict.items()} #{k:v*time_bin_size for k, v in full_epoch_occupancy_n_time_bins_dict.items()} ## in units of seconds
                # full_epoch_occupancy_seconds_dict
                full_epoch_occupancy_fractional_dict_dict[pre_post_delta_period] = {laps_pbe_epoch_name: _normalize_occupancy(v) for laps_pbe_epoch_name, v in full_epoch_occupancy_seconds_dict_dict[pre_post_delta_period].items()}
                # full_epoch_occupancy_fractional_dict = {k:v/np.nansum(v) for k, v in full_epoch_occupancy_n_time_bins_dict.items()} ## normalize over all positions
                # full_epoch_occupancy_fractional_dict
            ## END for pre_post_delta_period in laps_pbe_epoch_names....

        ## OUTPUTS: full_epoch_occupancy_seconds_dict_dict, full_epoch_occupancy_fractional_dict_dict

        return (full_epoch_occupancy_seconds_dict_dict, full_epoch_occupancy_fractional_dict_dict)


    @classmethod
    def main(cls, full_epoch_occupancy_seconds_dict_dict, 
                active_measured_positions_dfs_dict,
                time_bin_size, measured_x_col: str = 'binned_x_meas', 
                enable_figure: bool = True, num=f'Figure 5 - measured vs decoded', plot_pos_bin_axes: bool = False, should_max_normalize: bool = True,
                split_by_session: bool = False, pdf_export_path: Optional[Union[str, Path]] = None,
                **kwargs,
        ):
        """Plots measured vs decoded occupancy.

        When `split_by_session=True`, `full_epoch_occupancy_seconds_dict_dict` must be session-nested (from `compute_decoded_occupancy(..., split_by_session=True)`).
        Returns one 2x3 figure per session (keyed dicts), plus session-keyed measured occupancy dicts.
        Also writes per-session PDFs and a single vertically stacked PDF (`pdf_export_path`, default `output/Figure5_measured_vs_decoded_by_session.pdf`).
        """
        if enable_figure:
            import matplotlib.pyplot as plt
            from neuropy.plotting.placemaps import perform_plot_occupancy

        ## INPUTS: full_epoch_occupancy_seconds_dict_dict, full_epoch_occupancy_fractional_dict_dict, time_bin_size, measured_x_col: str = 'binned_x_meas'
        pre_post_all_epoch_names = ['pre-delta', 'post-delta', 'all']
        _PERIOD_KEYS = set(pre_post_all_epoch_names)
        decoder_color_dict: Dict[types.DecoderName, str] = {'long_LR': '#0099ff', 'long_RL': '#1b115a', 'short_LR': '#f51616', 'short_RL': '#891111'}

        def _is_session_nested_occupancy(occupancy_dict_dict) -> bool:
            return (len(occupancy_dict_dict) > 0) and set(occupancy_dict_dict.keys()).isdisjoint(_PERIOD_KEYS)

        def _n_pos_from_occupancy(occupancy):
            if isinstance(occupancy, dict):
                return len(next(iter(occupancy.values())))  # 59; decoder-split {decoder_name: (n_pos,)}
            return len(occupancy)  # 59; matches decoded

        def occupancy_vector_from_binned_x(df, measured_x_col=measured_x_col, xbin_bin_ids=None):
            """One occupancy vector (seconds) from FAT time-bin rows."""
            sub = df.dropna(subset=[measured_x_col])
            bin_ids = sub[measured_x_col].astype(int)
            if 'time_bin_size' in sub.columns:
                occ_seconds = sub.groupby(bin_ids)['time_bin_size'].sum()
            else:
                occ_seconds = bin_ids.value_counts() * float(time_bin_size)
            occ_seconds = occ_seconds.reindex(xbin_bin_ids, fill_value=0.0).to_numpy(dtype=float)
            return occ_seconds

        def _build_measured_occupancy_dicts(decoded_seconds_dict_dict, measured_dfs_dict, xbin_bin_ids):
            """Build MEASURED position occupancies matching decoded `{pre-post: {laps|PBEs: vec}}` shape."""
            measured_epoch_occupancy_seconds_dict_dict = {}
            measured_epoch_occupancy_fractional_dict_dict = {}
            # --- convert to Figure 5 occupancy dicts (laps only; no PBEs in this FAT slice) ---
            ## This is about building MEASURED position occupancies:
            laps_pbe_epoch_names = list(decoded_seconds_dict_dict['pre-delta'].keys()) # ['laps', 'PBEs']
            for a_lap_pbe_epoch_name in laps_pbe_epoch_names:
                measured_df = measured_dfs_dict[a_lap_pbe_epoch_name]
                if (measured_df is None) or (len(measured_df) == 0):
                    empty_occ = np.zeros((len(xbin_bin_ids),), dtype=float)
                    for pre_post in pre_post_all_epoch_names:
                        if pre_post not in measured_epoch_occupancy_seconds_dict_dict:
                            measured_epoch_occupancy_seconds_dict_dict[pre_post] = {a_lap_pbe_epoch_name: empty_occ}
                        else:
                            measured_epoch_occupancy_seconds_dict_dict[pre_post].update(**{a_lap_pbe_epoch_name: empty_occ})
                        if pre_post not in measured_epoch_occupancy_fractional_dict_dict:
                            measured_epoch_occupancy_fractional_dict_dict[pre_post] = {a_lap_pbe_epoch_name: empty_occ}
                        else:
                            measured_epoch_occupancy_fractional_dict_dict[pre_post].update(**{a_lap_pbe_epoch_name: empty_occ})
                    ## END for pre_post in pre_post_all_epoch_names....
                    continue

                df_by_pre_post = measured_df.pho.partition_df_dict('pre_post_delta_category')
                df_by_pre_post['all'] = deepcopy(measured_df)

                for pre_post in pre_post_all_epoch_names:
                    occ_s = occupancy_vector_from_binned_x(df_by_pre_post.get(pre_post, measured_df.iloc[0:0]), xbin_bin_ids=xbin_bin_ids)
                    occ_sum = np.nansum(occ_s)
                    occ_frac = (occ_s / occ_sum) if (occ_sum > 0) else occ_s
                    if pre_post not in measured_epoch_occupancy_seconds_dict_dict:
                        measured_epoch_occupancy_seconds_dict_dict[pre_post] = {a_lap_pbe_epoch_name: occ_s}
                    else:
                        measured_epoch_occupancy_seconds_dict_dict[pre_post].update(**{a_lap_pbe_epoch_name: occ_s})

                    if pre_post not in measured_epoch_occupancy_fractional_dict_dict:
                        measured_epoch_occupancy_fractional_dict_dict[pre_post] = {a_lap_pbe_epoch_name: occ_frac}
                    else:
                        measured_epoch_occupancy_fractional_dict_dict[pre_post].update(**{a_lap_pbe_epoch_name: occ_frac})
                ## END for pre_post in pre_post_all_epoch_names....
            ## END for a_lap_pbe_epoch_name in laps_pbe_epoch_names....
            return measured_epoch_occupancy_seconds_dict_dict, measured_epoch_occupancy_fractional_dict_dict

        def _plot_occupancy_mosaic(fig, decoded_seconds_dict_dict, measured_seconds_dict_dict):
            """2x3 laps/PBEs × pre/post/all mosaic on `fig`."""
            # ax_dict = fig.subplot_mosaic([['laps_pre-delta', 'laps_post-delta'], ['PBEs_pre-delta', 'PBEs_post-delta']], sharex=True, sharey=True)
            ax_dict = fig.subplot_mosaic([['laps_pre-delta', 'laps_post-delta', 'laps_all'], ['PBEs_pre-delta', 'PBEs_post-delta', 'PBEs_all']], sharex=True, sharey=True)
            # _plot_kwargs = dict(plot_pos_bin_axes=False, should_max_normalize=False)
            _plot_kwargs = dict(plot_pos_bin_axes=plot_pos_bin_axes, should_max_normalize=should_max_normalize)

            def _color_last_occupancy_stairs(ax, color):
                """`perform_plot_occupancy` drops `color` via `overriding_dict_with`; color the last stairs patch instead. Also drops the hardcoded `hatch='//'`."""
                patch = ax.patches[-1]
                patch.set_hatch('')
                patch.set_edgecolor(color)
                patch.set_facecolor('none')

            legend_handles = None
            legend_labels = None
            for lap_PBEs_plus_pre_post, ax in ax_dict.items():
                a_lap_pbe_epoch_name, pre_post = lap_PBEs_plus_pre_post.split('_')
                decoded_active = decoded_seconds_dict_dict.get(pre_post, {}).get(a_lap_pbe_epoch_name)
                measured_active = measured_seconds_dict_dict.get(pre_post, {}).get(a_lap_pbe_epoch_name)
                if (decoded_active is None) and (measured_active is None):
                    ax.set_title(f"{pre_post} | {a_lap_pbe_epoch_name}  (no data)")
                    continue

                # xbin_edges = np.linspace(loaded_track_limits['long_xlim'][0], loaded_track_limits['long_xlim'][1], n_pos + 1) ## I think always using 'long*' here is why both tracks are stretched to the long track length, but it would make since because they both use the same pos bins right? Oh, the bins aren't right from the bounds alone.
                # xbin_centers = 0.5 * (xbin_edges[:-1] + xbin_edges[1:])
                xbin_edges = None
                xbin_centers = None

                # decoded_active = full_epoch_occupancy_fractional_dict_dict[pre_post][a_lap_pbe_epoch_name]
                # measured_active = measured_epoch_occupancy_fractional_dict_dict[pre_post][a_lap_pbe_epoch_name]
                if measured_active is None:
                    n_pos = _n_pos_from_occupancy(decoded_active)
                    measured_active = np.zeros((n_pos,), dtype=float)

                measured_total_s: float = np.nansum(measured_active)
                perform_plot_occupancy(measured_active, xbin_centers=xbin_centers, xbin_edges=xbin_edges, fig=fig, ax=ax,
                                    label='measured', **_plot_kwargs)
                _color_last_occupancy_stairs(ax, 'black')
                if decoded_active is None:
                    decoded_total_s: float = 0.0
                    _decoded_legend_labels = ['decoded']
                elif isinstance(decoded_active, dict):
                    decoded_total_s: float = float(np.nansum([np.nansum(arr) for arr in decoded_active.values()]))
                    _decoded_legend_labels = [f'decoded[{decoder_name}]' for decoder_name in decoded_active.keys()]
                    for decoder_name, decoder_occ in decoded_active.items():
                        perform_plot_occupancy(decoder_occ, xbin_centers=xbin_centers, xbin_edges=xbin_edges, fig=fig, ax=ax,
                                            label=f'decoded[{decoder_name}]', **_plot_kwargs, alpha=0.9)
                        _color_last_occupancy_stairs(ax, decoder_color_dict[decoder_name])
                    ## END for decoder_name, decoder_occ in decoded_active.items()....

                else:
                    decoded_total_s: float = np.nansum(decoded_active)
                    _decoded_legend_labels = ['decoded']
                    perform_plot_occupancy(decoded_active, xbin_centers=xbin_centers, xbin_edges=xbin_edges, fig=fig, ax=ax,
                                        label='decoded', **_plot_kwargs, alpha=0.9)
                    _color_last_occupancy_stairs(ax, 'C0')
                ax.set_title(f"{pre_post} | {a_lap_pbe_epoch_name}  (decoded={decoded_total_s:0.1f}s, measured={measured_total_s:0.1f}s)")
                if legend_handles is None:
                    legend_handles, _ignored_labels = ax.get_legend_handles_labels()
                    legend_labels = ['measured'] + _decoded_legend_labels  # perform_plot_occupancy hardcodes stairs label='1D Placefield Occupancy'
            ## END for lap_PBEs_plus_pre_post, ax in ax_dict.items()....

            if legend_handles is not None:
                fig.legend(legend_handles, legend_labels, loc='outside right', frameon=True)
            return ax_dict

        is_session_nested: bool = _is_session_nested_occupancy(full_epoch_occupancy_seconds_dict_dict)
        if split_by_session:
            assert is_session_nested, "split_by_session=True requires session-nested occupancy from compute_decoded_occupancy(..., split_by_session=True)"
            session_names = list(full_epoch_occupancy_seconds_dict_dict.keys())
            assert len(session_names) > 0, "split_by_session=True but occupancy dict is empty"

            measured_session_names = set()
            for a_lap_pbe_epoch_name, measured_df in active_measured_positions_dfs_dict.items():
                assert 'session_name' in measured_df.columns, f"measured df '{a_lap_pbe_epoch_name}' is missing 'session_name' needed for split_by_session"
                measured_session_names.update(measured_df['session_name'].dropna().unique().tolist())
            ## END for a_lap_pbe_epoch_name, measured_df in active_measured_positions_dfs_dict.items()....

            decoded_session_names = set(session_names)
            _missing_measured = decoded_session_names.difference(measured_session_names)
            _extra_measured = measured_session_names.difference(decoded_session_names)
            if len(_missing_measured) > 0:
                print(f'WARN: split_by_session: decoded sessions with no measured overlap: {_missing_measured}')
            if len(_extra_measured) > 0:
                print(f'WARN: split_by_session: measured sessions with no decoded occupancy: {_extra_measured}')

            measured_epoch_occupancy_seconds_dict_dict = {}
            measured_epoch_occupancy_fractional_dict_dict = {}
            session_figs = {}
            session_ax_dicts = {}

            for session_name in session_names:
                sess_decoded_seconds = full_epoch_occupancy_seconds_dict_dict[session_name]
                if 'pre-delta' not in sess_decoded_seconds:
                    print(f'WARN: skip session "{session_name}" — no decoded occupancy.')
                    continue
                _laps_pre_occupancy = sess_decoded_seconds['pre-delta'].get('laps')
                if _laps_pre_occupancy is None:
                    _first_epoch = next(iter(sess_decoded_seconds['pre-delta'].values()))
                    n_pos = _n_pos_from_occupancy(_first_epoch)
                else:
                    n_pos = _n_pos_from_occupancy(_laps_pre_occupancy)  # 59; matches decoded
                xbin_bin_ids = np.arange(n_pos) + 1  # binned_x_meas is 1-indexed

                sess_measured_dfs_dict = {}
                for a_lap_pbe_epoch_name, measured_df in active_measured_positions_dfs_dict.items():
                    sess_measured_dfs_dict[a_lap_pbe_epoch_name] = measured_df[measured_df['session_name'] == session_name]
                ## END for a_lap_pbe_epoch_name, measured_df in active_measured_positions_dfs_dict.items()....

                sess_measured_seconds, sess_measured_frac = _build_measured_occupancy_dicts(sess_decoded_seconds, sess_measured_dfs_dict, xbin_bin_ids)
                measured_epoch_occupancy_seconds_dict_dict[session_name] = sess_measured_seconds
                measured_epoch_occupancy_fractional_dict_dict[session_name] = sess_measured_frac

                if enable_figure:
                    sess_fig = plt.figure(layout='constrained', figsize=[18, 4], dpi=220, clear=True, num=f'{num} | {session_name}')
                    ax_dict = _plot_occupancy_mosaic(sess_fig, sess_decoded_seconds, sess_measured_seconds)
                    sess_fig.suptitle(f'Figure 5 — measured vs decoded occupancy | {session_name}')
                    session_figs[session_name] = sess_fig
                    session_ax_dicts[session_name] = ax_dict
            ## END for session_name in session_names....

            if not enable_figure:
                session_figs = None
                session_ax_dicts = None

            merged_pdf_path = None
            if enable_figure and (session_figs is not None) and (len(session_figs) > 0):
                from pyphocorehelpers.plotting.media_output_helpers import PDFHelpers
                if pdf_export_path is None:
                    pdf_export_path = Path('output') / 'Figure5_measured_vs_decoded_by_session.pdf'
                merged_pdf_path = Path(pdf_export_path).resolve()
                merged_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                per_session_pdf_dir = merged_pdf_path.parent.joinpath(merged_pdf_path.stem)
                per_session_pdf_dir.mkdir(parents=True, exist_ok=True)
                per_session_pdf_paths = []
                with mpl.rc_context({'savefig.transparent': True, 'ps.fonttype': 42, 'pdf.fonttype': 42}):
                    for session_name, sess_fig in session_figs.items():
                        sess_stem: str = str(session_name).replace('/', '_').replace('\\', '_').replace('|', '_').replace(':', '_')
                        sess_pdf_path = per_session_pdf_dir.joinpath(f'{sess_stem}.pdf')
                        sess_fig.savefig(sess_pdf_path, format='pdf', bbox_inches='tight', dpi=220)
                        per_session_pdf_paths.append(sess_pdf_path)
                    ## END for session_name, sess_fig in session_figs.items()....
                merged_pdf_path = PDFHelpers.concatenate_pdfs_vertically(per_session_pdf_paths, merged_pdf_path)

            return (session_figs, session_ax_dicts), (measured_epoch_occupancy_seconds_dict_dict, measured_epoch_occupancy_fractional_dict_dict), merged_pdf_path

        assert (not is_session_nested), "session-nested occupancy requires split_by_session=True"

        _laps_pre_occupancy = full_epoch_occupancy_seconds_dict_dict['pre-delta']['laps']
        n_pos = _n_pos_from_occupancy(_laps_pre_occupancy)
        xbin_bin_ids = np.arange(n_pos) + 1  # binned_x_meas is 1-indexed

        ## INPUTS: active_measured_positions_dfs_dict
        measured_epoch_occupancy_seconds_dict_dict, measured_epoch_occupancy_fractional_dict_dict = _build_measured_occupancy_dicts(full_epoch_occupancy_seconds_dict_dict, active_measured_positions_dfs_dict, xbin_bin_ids)

        if enable_figure:
            ## single figure mode
            fig = plt.figure(layout='constrained', figsize=[18, 4], dpi=220, clear=True, num=num)
            ax_dict = _plot_occupancy_mosaic(fig, full_epoch_occupancy_seconds_dict_dict, measured_epoch_occupancy_seconds_dict_dict)
            plt.suptitle(f'Figure 5 — measured x vs decoded occupancy')
            # _out_figs_dict[a_lap_pbe_epoch_name] = (fig, ax_dict)
        else:
            fig = None
            ax_dict = None

        return (fig, ax_dict), (measured_epoch_occupancy_seconds_dict_dict, measured_epoch_occupancy_fractional_dict_dict), None


# ==================================================================================================================== #
# 2023-06-29 - Build Properly sorted ratemaps with potentially missing entries                                         #
# ==================================================================================================================== #

from neuropy.plotting.ratemaps import _help_plot_ratemap_neuronIDs # required for build_shared_sorted_neuronIDs

@function_attributes(short_name=None, tags=['active', 'sort', 'neuron_ids'], input_requires=[], output_provides=[], uses=['_help_plot_ratemap_neuronIDs'], used_by=[], creation_date='2023-06-29 07:46', related_items=[])
def build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind):
    """ 
    
    `sort_ind` should be the indicies to sort `included_unit_neuron_IDs`.
        `included_unit_neuron_IDs` may include entries not present in `ratemap`.
    
    Usage:
        ratemap = long_pf1D.ratemap
        included_unit_neuron_IDs = EITHER_subset.track_exclusive_aclus
        rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=new_all_aclus_sort_indicies.copy())
    """
    if not isinstance(sort_ind, np.ndarray):
        sort_ind = np.array(sort_ind)
    assert np.size(included_unit_neuron_IDs) == np.size(sort_ind), f"`sort_ind` should be the indicies to sort `included_unit_neuron_IDs`."

    #TODO 2023-06-29 06:50: - [ ] SOOO CLOSE. This is the right way to do it... the way that's done in `neuropy.plotting.ratemaps.plot_ratemap_1D`, but because i'm trying to plot the ratemaps as a heatmap I need to fill the missing entries with appropriately sized np.nans or something.
    active_maps, title_substring, included_unit_indicies = _help_plot_ratemap_neuronIDs(ratemap, included_unit_neuron_IDs=included_unit_neuron_IDs, debug_print=True)
    n_neurons = len(included_unit_indicies) # n_neurons includes Non-active neurons without a placefield if they're provided in included_unit_indicies.
    if not isinstance(included_unit_indicies, np.ndarray):
        included_unit_indicies = np.array(included_unit_indicies)
    included_unit_indicies

    needed_empty_map_shape = np.shape(active_maps)[1:]

    sorted_included_unit_indicies = included_unit_indicies[sort_ind]
    rediculous_final_sorted_all_included_pfmap = []
    rediculous_final_sorted_all_included_neuron_ID = []

    for i, curr_included_unit_index in enumerate(sorted_included_unit_indicies):
        # `curr_included_unit_index` is either an index into the `included_unit_neuron_IDs` array or None
        ### Three things must be considered for each "row" of the plot: 1. the pfmap curve values, 2. the cell id label displayed to the left of the row, 3. the color which is used for the row.
        if curr_included_unit_index is not None:
            # valid neuron ID, access like normal
            pfmap = active_maps[curr_included_unit_index]
            # normal (non-shared mode)
            curr_ratemap_relative_neuron_IDX = curr_included_unit_index
            curr_neuron_ID = ratemap.neuron_ids[curr_ratemap_relative_neuron_IDX]
            
        else:
            # invalid neuron ID, generate blank entry
            curr_ratemap_relative_neuron_IDX = None # This neuron_ID doesn't correspond to a neuron_IDX in the current ratemap, so we'll mark this value as None
            assert included_unit_neuron_IDs is not None
            curr_neuron_ID = included_unit_neuron_IDs[sort_ind[i]]

            # pfmap = np.zeros((np.shape(active_maps)[1],)) # fully allocated new array of zeros
            pfmap = np.zeros(needed_empty_map_shape) # fully allocated new array of zeros
            
        rediculous_final_sorted_all_included_pfmap.append(pfmap)
        rediculous_final_sorted_all_included_neuron_ID.append(curr_neuron_ID)

    rediculous_final_sorted_all_included_neuron_ID = np.array(rediculous_final_sorted_all_included_neuron_ID)
    
    rediculous_final_sorted_all_included_pfmap = np.vstack(rediculous_final_sorted_all_included_pfmap)
    # rediculous_final_sorted_all_included_pfmap.shape # (68, 117)
    return rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap

# ==================================================================================================================== #
# 2023-06-27 - Paper Figure 1 Code                                                                                     #
# ==================================================================================================================== #


# ==================================================================================================================== #
# 2023-07-14 - Iterative Epoch Track Assignment                                                                        #
# ==================================================================================================================== #
class TrackAssignmentState(ExtendedEnum):
    """Docstring for TrackAssignmentState."""
    UNASSIGNED = "unassigned"
    LONG_TRACK = "long_track"
    SHORT_TRACK = "short_track"
    NEITHER = "neither"
    
    @property
    def displayColor(self):
        return self.displayColorList()[self]
    
    @classmethod
    def displayColorList(cls):
        return cls.build_member_value_dict(['grey', 'blue', 'red', 'black'])
    
    # track_assignment = epoch['track_assignment'].decision.name
    # # Set the color of the epoch based on the track assignment
    # if track_assignment == 'UNASSIGNED':
    #     color = 'grey'
    # elif track_assignment == 'LONG_TRACK':
    #     color = 'blue'
    # elif track_assignment == 'SHORT_TRACK':
    #     color = 'red'
    # elif track_assignment == 'NEITHER':
    #     color = 'black'

@define(slots=False, frozen=True)
class TrackAssignmentDecision(KeyValueHashableObject):
    """ represents the decision made to assign a particular epoch to a track (long or short) or to neither track"""
    decision: TrackAssignmentState
    confidence: float # a value between 0 and 1

@define(slots=False)
class AssigningEpochs:
    """ class responsible for iteratively applying various criteria to assign epochs to a particular track starting with the unassigned set of all epochs (in this analysis replay events) """
    filter_epochs_df: pd.DataFrame

    @property
    def unassigned_epochs_df(self):
        """A convenience accessor for only the unassigned epochs remaining in the self.filter_epochs_df."""
        is_unassigned = np.array([v.decision == TrackAssignmentState.UNASSIGNED for v in self.filter_epochs_df.track_assignment])
        return self.filter_epochs_df[is_unassigned]
    @unassigned_epochs_df.setter
    def unassigned_epochs_df(self, value):
        self.filter_epochs_df[np.array([v.decision == TrackAssignmentState.UNASSIGNED for v in self.filter_epochs_df.track_assignment])] = value
        

    def _subfn_find_example_epochs(self, spikes_df: pd.DataFrame, exclusive_aclus, included_neuron_ids=None):
        """ aims to find epochs for use as examples in Figure 1:
            1. Contain spikes from aclus that are exclusive to one of the two tracks (either short or long)
            2. Are labeled as "good replays" by my user annotations
            
            Adds: 'contains_one_exclusive_aclu' column to filter_epochs_df
            
            Usage:
                from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _find_example_epochs
                # included_neuron_ids = deepcopy(exclusive_aclus)
                # included_neuron_ids = None
                included_neuron_ids = EITHER_subset.track_exclusive_aclus
                spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df).spikes.sliced_by_neuron_type('pyr')
                # filter_epochs = deepcopy(curr_active_pipeline.sess.replay)
                # spikes_df = deepcopy(long_results_obj.spikes_df) # LeaveOneOutDecodingAnalysisResult
                # spikes_df[np.isin(spikes_df.aclu, included_neuron_ids)]
                filter_epochs = deepcopy(long_results_obj.active_filter_epochs)

                filter_epoch_spikes_df, filter_epochs_df = _find_example_epochs(spikes_df, filter_epochs, EITHER_subset.track_exclusive_aclus, included_neuron_ids=included_neuron_ids)

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _prepare_spikes_df_from_filter_epochs

        if included_neuron_ids is None:
            included_neuron_ids = spikes_df.spikes.neuron_ids
            
        filter_epoch_spikes_df = deepcopy(spikes_df)
        filter_epoch_spikes_df = _prepare_spikes_df_from_filter_epochs(filter_epoch_spikes_df, filter_epochs=self.filter_epochs_df, included_neuron_ids=included_neuron_ids, epoch_id_key_name='replay_epoch_id', debug_print=False) # replay_epoch_id

        unique_replay_epoch_id_values, epoch_split_spikes_df_list = partition(filter_epoch_spikes_df, 'replay_epoch_id')
        # filter_epoch_spikes_df.groupby('replay_epoch_id')
        # unique_replay_epoch_id_values # (198,)
        epoch_spikes_unique_aclus_list = [] # a list of each aclu that fires at least once in the epoch
        epoch_contains_any_exclusive_aclus = []

        for an_epoch_id, epoch_specific_spikes_df in zip(unique_replay_epoch_id_values, epoch_split_spikes_df_list):
            # Loop through the epochs
            # find epochs containing at least one in `exclusive_aclus`:
            epoch_spikes_unique_aclus = epoch_specific_spikes_df.aclu.unique()
            epoch_spikes_unique_aclus_list.append(epoch_spikes_unique_aclus)
            # epoch_spikes_unique_aclus # finds lists of all the active aclus in the given epoch
            if len(epoch_spikes_unique_aclus) > 0:	
                epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
            else:
                epoch_contains_any_exclusive_aclus.append(False)
            # epoch_contains_any_exclusive_aclus # returns True if the epoch contains one ore more spikes from the search list `exclusive_aclus`

        assert len(epoch_contains_any_exclusive_aclus) == np.shape(self.filter_epochs_df)[0]
        self.filter_epochs_df['contains_one_exclusive_aclu'] = epoch_contains_any_exclusive_aclus # adds the appropriate column to the filter_epochs_df
        self.filter_epochs_df['active_unique_aclus'] = epoch_spikes_unique_aclus_list
        return filter_epoch_spikes_df, self.filter_epochs_df


    def determine_if_contains_active_set_exclusive_cells(self, spikes_df: pd.DataFrame, exclusive_aclus, short_exclusive, long_exclusive, included_neuron_ids=None):
        """ NOTE: also updates `filter_epoch_spikes_df` which is returned
    
        adds ['contains_one_exclusive_aclu', 'active_unique_aclus', 'has_SHORT_exclusive_aclu', 'has_LONG_exclusive_aclu'] columns to `filter_epochs_df`
        """
        filter_epoch_spikes_df, self.filter_epochs_df = self._subfn_find_example_epochs(spikes_df, exclusive_aclus=exclusive_aclus, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'
        # epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
        self.filter_epochs_df['has_SHORT_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, short_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in self.filter_epochs_df['active_unique_aclus']]
        self.filter_epochs_df['has_LONG_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, long_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in self.filter_epochs_df['active_unique_aclus']]
        return filter_epoch_spikes_df, self.filter_epochs_df


    @function_attributes(short_name=None, tags=['user-annotations','selections'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-04 15:30', related_items=[])
    def filter_by_user_selections(self, curr_active_pipeline, allow_interactive_selection:bool = False):
        """Get the manual user annotations to determine the good replays for both long/short decoding
        
        Adds the ['long_is_user_included', 'short_is_user_included'] columns to the `filter_epochs_df` DataFrame
        """
        user_annotation_man = UserAnnotationsManager()
        user_annotations = user_annotation_man.get_user_annotations()

        final_context_L = curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='long_results_obj')
        final_context_S = curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='short_results_obj')
        # _out_pagination_controller.params.active_identifying_figure_ctx.adding_context(None,  user_annotation="selections")
        selections_context_L = final_context_L.adding_context(None,  user_annotation="selections")
        selections_context_S = final_context_S.adding_context(None,  user_annotation="selections")
        
        ## try to get the user annotations for this session:
        try:
            selection_idxs_L = user_annotations[selections_context_L]
            selection_idxs_S = user_annotations[selections_context_S]
        except KeyError as e:
            if allow_interactive_selection:
                print(f'user annotations <good replay selections> are not found. Creating them interactively...')
                user_annotations = interactive_good_epoch_selections(annotations_man=user_annotation_man, curr_active_pipeline=curr_active_pipeline)  # perform interactive selection. Should block here.
                selection_idxs_L = user_annotations[selections_context_L]
                selection_idxs_S = user_annotations[selections_context_S]
            else:
                print(f'interactive annotation is not permitted. Failing.')
                raise e
        except Exception as e:
            print('Unhandled exception: {e}')
            raise

        # for updating the filter_epochs_df (`filter_epochs_df`) from the selections:
        self.filter_epochs_df['long_is_user_included'] = np.isin(self.filter_epochs_df.index, selection_idxs_L)
        self.filter_epochs_df['short_is_user_included'] = np.isin(self.filter_epochs_df.index, selection_idxs_S)


    def _debug_print_assignment_statuses(self, debug_print=True):
        is_unassigned = np.array([v.decision == TrackAssignmentState.UNASSIGNED for v in self.filter_epochs_df.track_assignment])
        is_disregarded = np.array([v.decision == TrackAssignmentState.NEITHER for v in self.filter_epochs_df.track_assignment])
        num_unassigned = np.sum(is_unassigned)
        num_disregarded = np.sum(is_disregarded)
        num_assigned = len(self.filter_epochs_df) - (num_unassigned + num_disregarded)
        if debug_print:
            print(f'num_unassigned: {num_unassigned}, num_disregarded: {num_disregarded}, num_assigned: {num_assigned}')
        return num_unassigned, num_disregarded, num_assigned
    

    def _subfn_plot_epoch_track_assignments(self, axis_idx:int, fig=None, axs=None, defer_render=False):
        """ Plots a figure that represents each epoch as a little box that can be colored in based on the track assignment: grey for Unassigned, blue for Long, red for Short, black for Neither.

        Args:
        - assigning_epochs_obj: an object of type AssigningEpochs

        Returns:
        - None
        """
        def _subfn_setup_each_individual_row_axis(ax, is_bottom_row=False):
            """ captures self 
            """
            # Set the x and y limits of the axis
            ax.set_xlim([0, len(self.filter_epochs_df)])
            ax.set_ylim([0, 1])
            
            # Add vertical grid lines
            for i in range(len(self.filter_epochs_df)):
                ax.axvline(x=i, color='white', linewidth=0.5)

            # Set the title and axis labels
            if is_bottom_row:
                ax.set_title('Replay Track Assignments')
                ax.set_xlabel('Replay Epoch Index')
                
            ax.set_ylabel('') # Track Assignment

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            return ax            

        # Create a figure and axis object if not provided
        if fig is None or axs is None:
            fig, axs = plt.subplots(ncols=1, nrows=5, sharex=True, sharey=True, figsize=(20, 10))
            
            # Set up each individual row axis
            for ax in axs:
                ax = _subfn_setup_each_individual_row_axis(ax, is_bottom_row=(ax == axs[-1]))
            
            # Add legend (only to the bottom axis)
            grey_patch = mpatches.Patch(color='grey', label='Unassigned')
            blue_patch = mpatches.Patch(color='blue', label='Long')
            red_patch = mpatches.Patch(color='red', label='Short')
            black_patch = mpatches.Patch(color='black', label='Neither')
            plt.legend(handles=[grey_patch, blue_patch, red_patch, black_patch], loc='lower center', ncol=4)


        # Get the axis object
        ax = axs[axis_idx]

        # Iterate over each epoch in the filter_epochs_df
        for i, epoch in self.filter_epochs_df.iterrows():
            # Get the track assignment of the epoch
            # track_assignment = epoch['track_assignment'].decision.name
            
            track_assignment = epoch['track_assignment'].decision
            track_assignment_color = track_assignment.displayColor

            # # Set the color of the epoch based on the track assignment
            # if track_assignment == 'UNASSIGNED':
            #     color = 'grey'
            # elif track_assignment == 'LONG_TRACK':
            #     color = 'blue'
            # elif track_assignment == 'SHORT_TRACK':
            #     color = 'red'
            # elif track_assignment == 'NEITHER':
            #     color = 'black'

            # Draw a rectangle for the epoch
            ax.add_patch(plt.Rectangle((i, len(ax.get_yticks())), 1, 1, color=track_assignment_color))

        # Show the plot
        if not defer_render:
            plt.show()
            
        return fig, axs
    

    @classmethod
    def main_plot_iterative_epoch_track_assignments(cls, curr_active_pipeline, defer_render=False):
        """ 2023-06-27 - Test how many can actually be sorted by active_set criteria:
        
        """
        ## First filter by only those user-included replays (in the future this won't be done)

        # from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import TrackAssignmentDecision, TrackAssignmentState, AssigningEpochs
        (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)

        ## Initialize to unassigned
        axis_idx:int = 0
        assigning_epochs_obj.filter_epochs_df['track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.UNASSIGNED, 0.0)
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, defer_render=True)

        ## User Assignment:
        # # Partition based on whether the user included the epoch in the long or short track in the user-included epochs:
        # is_user_exclusive_L = np.logical_and(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included']))
        # is_user_exclusive_S = np.logical_and(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included']))
        # is_user_unassigned_in_both_epochs = np.logical_and(np.logical_not(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included']), np.logical_not(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included'])) # the user said it was bad in both epochs, so assign it to neither with high confidence

        # # NOTE: be sure to assign to the filter_epochs_df, not the unassigned_epochs_df, because the unassigned_epochs_df is a subset of the filter_epochs_df, and we want to assign to the filter_epochs_df so that the unassigned_epochs_df will be a subset of the filter_epochs_df:
        # # assign the user_exclusive_L to long_track:
        # assigning_epochs_obj.filter_epochs_df.loc[is_user_exclusive_L, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.LONG_TRACK, 1.0)
        # # assign the user_exclusive_S to short_track:
        # assigning_epochs_obj.filter_epochs_df.loc[is_user_exclusive_S, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.SHORT_TRACK, 1.0)
        # # assign the user_unassigned_in_both_epochs to neither:
        # assigning_epochs_obj.filter_epochs_df.loc[is_user_unassigned_in_both_epochs, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.NEITHER, 1.0)

        # assigning_epochs_obj.filter_epochs_df[is_user_exclusive_S]
        # assigning_epochs_obj.filter_epochs_df[is_user_exclusive_L]
        axis_idx = axis_idx + 1
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, fig=fig, axs=axs, defer_render=True)

        unassigned_epochs_df = assigning_epochs_obj.unassigned_epochs_df
        # Filter based on the active_set cells (LxC, SxC):
        unassigned_epochs_df.loc[np.logical_and(unassigned_epochs_df['has_LONG_exclusive_aclu'], np.logical_not(unassigned_epochs_df['has_SHORT_exclusive_aclu'])), 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.LONG_TRACK, 0.85)
        assigning_epochs_obj._debug_print_assignment_statuses()
        # assign the user_exclusive_S to short_track:
        unassigned_epochs_df.loc[np.logical_and(np.logical_not(unassigned_epochs_df['has_LONG_exclusive_aclu']), unassigned_epochs_df['has_SHORT_exclusive_aclu']), 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.SHORT_TRACK, 0.85)

        # Re assign
        assigning_epochs_obj.unassigned_epochs_df = unassigned_epochs_df

        axis_idx = axis_idx + 1
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, fig=fig, axs=axs, defer_render=True)
        
        if not defer_render:
            fig.show() 
            
        return fig, axs, assigning_epochs_obj


# ==================================================================================================================== #
# 2023-07-14 - Paper Figure 1                                                                                          #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['FIGURE1', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 21:40', related_items=[])
def PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline, allow_interactive_good_epochs_selection=False, debug_print=False):
    """ 
    
    # general approach copied from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations._epoch_unit_avg_firing_rates`
    
    # requires: selection_idxs_L, selection_idxs_S
        - Uses the `long_short_decoding_analyses` global result to access `long_results_obj.active_filter_epochs`:
        - Uses the `JonathanFiringRateAnalysisResult` global result to get info about the long/short placefields:
    
    long_results_obj, short_results_obj 
    """
    ## Use the `long_short_decoding_analyses` global result to access `long_results_obj.active_filter_epochs`:
    curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global

    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _prepare_spikes_df_from_filter_epochs

    ## Use the `JonathanFiringRateAnalysisResult` to get info about the long/short placefields:
    jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis
    neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset = jonathan_firing_rate_analysis_result.get_cell_track_partitions(frs_index_inclusion_magnitude=0.5)

    assigning_epochs_obj = AssigningEpochs(filter_epochs_df=ensure_dataframe(deepcopy(long_results_obj.active_filter_epochs)))

    # included_neuron_ids = deepcopy(exclusive_aclus)
    # included_neuron_ids = None
    included_neuron_ids = EITHER_subset.track_exclusive_aclus
    spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df).spikes.sliced_by_neuron_type('pyr')
    # filter_epochs_df = ensure_dataframe(deepcopy(long_results_obj.active_filter_epochs))


    ## TODO: need to usee the actual LxC/SxCs (hand-picked) instead of the ones based on placefields.
    filter_epoch_spikes_df, filter_epochs_df = assigning_epochs_obj.determine_if_contains_active_set_exclusive_cells(spikes_df, exclusive_aclus=EITHER_subset.track_exclusive_aclus, short_exclusive=short_exclusive, long_exclusive=long_exclusive, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'    

    # filter_epoch_spikes_df, filter_epochs_df = assigning_epochs_obj._find_example_epochs(spikes_df, EITHER_subset.track_exclusive_aclus, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'
    # # epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
    # filter_epochs_df['has_SHORT_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, short_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in filter_epochs_df['active_unique_aclus']]
    # filter_epochs_df['has_LONG_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, long_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in filter_epochs_df['active_unique_aclus']]

    # Get the manual user annotations to determine the good replays for both long/short decoding:
    # assigning_epochs_obj.filter_by_user_selections(curr_active_pipeline=curr_active_pipeline, allow_interactive_selection=allow_interactive_good_epochs_selection)

    #### Finally, get only the epochs that meet the criteria:

    # # only include those with one or more exclusive aclu:
    # considered_filter_epochs_df = filter_epochs_df[filter_epochs_df.contains_one_exclusive_aclu].copy()
    # # require inclusion for long or short:
    # considered_filter_epochs_df = considered_filter_epochs_df[np.logical_xor(filter_epochs_df['long_is_user_included'], filter_epochs_df['short_is_user_included'])]

    # Get separate long-side/short-side candidate replays:
    if 'long_is_user_included' not in filter_epochs_df:
        print(f'WARNING: PAPER_FIGURE_figure_1_add_replay_epoch_rasters(...): no user-assigned manually labeled replay epochs. Reeturning all epochs.')
        epochs_df_L = filter_epochs_df[(filter_epochs_df['has_LONG_exclusive_aclu'])].copy() # replay not considered good by user for short decoding, but it is for long decoding. Finally, has at least one LONG exclusive ACLU.
        epochs_df_S = filter_epochs_df[(filter_epochs_df['has_SHORT_exclusive_aclu'])].copy()  # replay not considered good by user for long decoding, but it is for short decoding. Finally, has at least one SHORT exclusive ACLU.
    else:
        epochs_df_L = filter_epochs_df[(filter_epochs_df['has_LONG_exclusive_aclu'] & filter_epochs_df['long_is_user_included'])].copy() # replay not considered good by user for short decoding, but it is for long decoding. Finally, has at least one LONG exclusive ACLU.
        epochs_df_S = filter_epochs_df[(filter_epochs_df['has_SHORT_exclusive_aclu'] & filter_epochs_df['short_is_user_included'])].copy()  # replay not considered good by user for long decoding, but it is for short decoding. Finally, has at least one SHORT exclusive ACLU.


    # epochs_df_L = filter_epochs_df[(filter_epochs_df['has_LONG_exclusive_aclu'] & filter_epochs_df['long_is_user_included'])].copy() # replay not considered good by user for short decoding, but it is for long decoding. Finally, has at least one LONG exclusive ACLU.
    # epochs_df_S = filter_epochs_df[(filter_epochs_df['has_SHORT_exclusive_aclu'] & filter_epochs_df['short_is_user_included'])].copy()  # replay not considered good by user for long decoding, but it is for short decoding. Finally, has at least one SHORT exclusive ACLU.

    # Common for all rasters:
    new_all_aclus_sort_indicies = determine_long_short_pf1D_indicies_sort_by_peak(curr_active_pipeline=curr_active_pipeline, curr_any_context_neurons=EITHER_subset.track_exclusive_aclus)

    # Build one spikes_df for Long and Short:
    filter_epoch_spikes_df_L, filter_epoch_spikes_df_S = [_prepare_spikes_df_from_filter_epochs(filter_epoch_spikes_df, filter_epochs=an_epochs_df, included_neuron_ids=EITHER_subset.track_exclusive_aclus, epoch_id_key_name='replay_epoch_id', debug_print=False) for an_epochs_df in (epochs_df_L, epochs_df_S)]    

    # requires epochs_df_L, epochs_df_S from `PAPER_FIGURE_figure_1_add_replay_epoch_rasters`
    # requires epochs_df_L, epochs_df_S from `PAPER_FIGURE_figure_1_add_replay_epoch_rasters`
    # get the good epoch indicies from epoch_df_L.Index
    good_example_epoch_indicies_L = epochs_df_L.index.to_numpy()
    # good_epoch_indicies_L = np.array([15, 49])
    good_example_epoch_indicies_S = epochs_df_S.index.to_numpy()
    # good_epoch_indicies_S = np.array([ 31,  49,  55,  68,  70,  71,  73,  77,  78,  89,  94, 100, 104, 105, 111, 114, 117, 118, 122, 123, 131])
    if debug_print:
        print(f'good_example_epoch_indicies_L: {good_example_epoch_indicies_L}')
        print(f'good_example_epoch_indicies_S: {good_example_epoch_indicies_S}')


    return (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj


@function_attributes(short_name=None, tags=['FINAL', 'publication', 'figure', 'combined'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 14:33', related_items=[])
def PAPER_FIGURE_figure_1_full(curr_active_pipeline, defer_show=False, save_figure=True, should_plot_pf1d_compare=True, should_plot_example_rasters=False, should_plot_stacked_epoch_slices=False, should_plot_pho_jonathan_figures=True, show_only_refined_cells=True, **kwargs):
    """ 
    
    show_only_refined_cells: bool - added 2023-09-28 to output LxC and SxC values "refined" by their firing rate index.

    
    Usage:
        pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics, fig_1c_figures_out_dict = PAPER_FIGURE_figure_1_full(curr_active_pipeline) # did not display the pf1
    
    """
    save_figure_kwargs = dict(write_vector_format=kwargs.pop('write_vector_format', False), write_png=kwargs.pop('write_png', True)) | pop_dict_subset(kwargs, ['bbox_inches', 'pad_inches'])
    prepare_for_publication: bool = kwargs.pop('prepare_for_publication', False)


    ## long_short_decoding_analyses:
    curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
    ## Extract variables from results object:
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global

    # (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_short_long_constrained_decoders(curr_active_pipeline, recalculate_anyway=True)
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_computation_config, short_computation_config, global_computation_config = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_pf1D, short_pf1D, global_pf1D = long_results.pf1D, short_results.pf1D, global_results.pf1D
    long_pf2D, short_pf2D, global_pf2D = long_results.pf2D, short_results.pf2D, global_results.pf2D
    decoding_time_bin_size = long_one_step_decoder_1D.time_bin_size # 1.0/30.0 # 0.03333333333333333

    ## Get global 'long_short_post_decoding' results:
    # curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
    # expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
    # rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
    # Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result.Flat_epoch_time_bins_mean, expected_v_observed_result.Flat_decoder_time_bin_centers, expected_v_observed_result.num_neurons, expected_v_observed_result.num_timebins_in_epoch, expected_v_observed_result.num_total_flat_timebins, expected_v_observed_result.is_short_track_epoch, expected_v_observed_result.is_long_track_epoch, expected_v_observed_result.short_short_diff, expected_v_observed_result.long_long_diff
    # ==================================================================================================================== #
    # Show 1D Placefields for both Short and Long (top half of the figure)                                                 #
    # ==================================================================================================================== #
    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

    if should_plot_pf1d_compare:
        long_single_cell_pfmap_processing_fn = None
        short_single_cell_pfmap_processing_fn = None
        # sort_idx = None
        # sort_idx = sort_ind
        # sort_idx = sort_indicies
        # sort_idx = new_all_aclus_sort_indicies
        sort_idx = 'peak_long'
        # sort_idx = 'bad'

        pf1d_compare_graphics = curr_active_pipeline.display('_display_long_short_pf1D_comparison', active_identifying_session_ctx, single_figure=False, debug_print=False, fignum='Short v Long pf1D Comparison',
                                        long_kwargs={'sortby': sort_idx, 'single_cell_pfmap_processing_fn': long_single_cell_pfmap_processing_fn},
                                        short_kwargs={'sortby': sort_idx, 'single_cell_pfmap_processing_fn': short_single_cell_pfmap_processing_fn, 'curve_hatch_style': {'hatch':'///', 'edgecolor':'k'}},
                                        shared_kwargs={'cmap': 'hsv'},
                                        save_figure=save_figure,
                                        defer_render=defer_show
                                        )
                                        
    else:
        pf1d_compare_graphics = None


    # ==================================================================================================================== #
    # Show Example Replay Epochs containing the long or short only cells                                                                  #
    # ==================================================================================================================== #

    (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)


    if should_plot_example_rasters:
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multiple_raster_plot
        # unit_colors_list = None # default rainbow of colors for the raster plots
        neuron_qcolors_list = [pg.mkColor('black') for aclu in EITHER_subset.track_exclusive_aclus] # solid green for all
        unit_colors_list = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=True)

        # Copy and modify the colors for the cells that are long/short exclusive:
        unit_colors_list_L = deepcopy(unit_colors_list)
        is_L_exclusive = np.isin(EITHER_subset.track_exclusive_aclus, long_exclusive.track_exclusive_aclus) # get long exclusive
        unit_colors_list_L[0, is_L_exclusive] = 255 # [1.0, 0.0, 0.0, 1.0]
        unit_colors_list_L[1, is_L_exclusive] = 0.0
        unit_colors_list_L[2, is_L_exclusive] = 0.0
        
        unit_colors_list_S = deepcopy(unit_colors_list)
        is_S_exclusive = np.isin(EITHER_subset.track_exclusive_aclus, short_exclusive.track_exclusive_aclus) # get short exclusive
        unit_colors_list_S[0, is_S_exclusive] = 0.0 # [1.0, 0.0, 0.0, 1.0]
        unit_colors_list_S[1, is_S_exclusive] = 0.0
        unit_colors_list_S[2, is_S_exclusive] = 255.0
        
        
        ## Build scatterplot args:    
        # Common Tick Label
        vtick = pg.QtGui.QPainterPath()

        # Thicker Tick Label:
        tick_width = 0.25
        # tick_width = 10.0
        half_tick_width = 0.5 * tick_width
        vtick.moveTo(-half_tick_width, -0.5)
        vtick.addRect(-half_tick_width, -0.5, tick_width, 1.0) # x, y, width, height
        pen = {'color': 'white', 'width': 1}
        override_scatter_plot_kwargs = dict(pxMode=True, symbol=vtick, size=6, pen=pen)

        # Not sure if needed:
        # filter_epoch_spikes_df_L.spikes.rebuild_fragile_linear_neuron_IDXs()

        example_epoch_rasters_L = plot_multiple_raster_plot(epochs_df_L, filter_epoch_spikes_df_L, included_neuron_ids=EITHER_subset.track_exclusive_aclus, unit_sort_order=new_all_aclus_sort_indicies, unit_colors_list=unit_colors_list_L, scatter_plot_kwargs=override_scatter_plot_kwargs,
                                            epoch_id_key_name='replay_epoch_id', scatter_app_name="Long Decoded Example Replays", defer_show=defer_show,
                                            active_context=curr_active_pipeline.build_display_context_for_session('plot_multiple_raster_plot', fig=1, track='long', epoch='example_replays'))
        app_L, win_L, plots_L, plots_data_L = example_epoch_rasters_L
        if save_figure:
            curr_active_pipeline.output_figure(plots_data_L.active_context, win_L) 

        example_epoch_rasters_S = plot_multiple_raster_plot(epochs_df_S, filter_epoch_spikes_df_S, included_neuron_ids=EITHER_subset.track_exclusive_aclus, unit_sort_order=new_all_aclus_sort_indicies, unit_colors_list=unit_colors_list_S, scatter_plot_kwargs=override_scatter_plot_kwargs,
                                                                        epoch_id_key_name='replay_epoch_id', scatter_app_name="Short Decoded Example Replays", defer_show=defer_show,
                                                                        active_context=curr_active_pipeline.build_display_context_for_session('plot_multiple_raster_plot', fig=1, track='short', epoch='example_replays'))
        app_S, win_S, plots_S, plots_data_S = example_epoch_rasters_S
        if save_figure:
            curr_active_pipeline.output_figure(plots_data_S.active_context, win_S)
    else:
        example_epoch_rasters_L, example_epoch_rasters_S = None, None
        

    ## Stacked Epoch Plot: this is the slowest part of the figure by far as it renders in a scrollable pyqtgraph window.
    if should_plot_stacked_epoch_slices:
        example_stacked_epoch_graphics = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', defer_render=defer_show, save_figure=save_figure)
    else: 
        example_stacked_epoch_graphics = None


    # ==================================================================================================================== #
    # Fig 1c) 2023-07-14 - LxC and SxC PhoJonathanSession plots                                                            #
    # ==================================================================================================================== #

    ## Get global 'jonathan_firing_rate_analysis' results:
    curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
    neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis.neuron_replay_stats_df, curr_jonathan_firing_rate_analysis.rdf.rdf, curr_jonathan_firing_rate_analysis.rdf.aclu_to_idx, curr_jonathan_firing_rate_analysis.irdf.irdf
    
    if should_plot_pho_jonathan_figures:
        if show_only_refined_cells:
            fig_1c_figures_out_dict = BatchPhoJonathanFiguresHelper.run(curr_active_pipeline, neuron_replay_stats_df, included_unit_neuron_IDs=XOR_subset.get_refined_track_exclusive_aclus(), n_max_page_rows=20, **save_figure_kwargs, show_only_refined_cells=show_only_refined_cells, disable_top_row=True) # write_vector_format=False, write_png=save_figure
        else:
            fig_1c_figures_out_dict = BatchPhoJonathanFiguresHelper.run(curr_active_pipeline, neuron_replay_stats_df, included_unit_neuron_IDs=XOR_subset.track_exclusive_aclus, n_max_page_rows=20, **save_figure_kwargs, disable_top_row=True) # active_out_figures_dict: {IdentifyingContext<('kdiba', 'gor01', 'two', '2006-6-07_16-40-19', 'BatchPhoJonathanReplayFRC', 'long_only', '(12,21,48)')>: <Figure size 1920x660 with 12 Axes>, IdentifyingContext<('kdiba', 'gor01', 'two', '2006-6-07_16-40-19', 'BatchPhoJonathanReplayFRC', 'short_only', '(18,19,65)')>: <Figure size 1920x660 with 12 Axes>}
    else:
        fig_1c_figures_out_dict = None

    return pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics, fig_1c_figures_out_dict

# ==================================================================================================================== #
# 2023-06-26 - Paper Figure 2 Code                                                                                     #
# ==================================================================================================================== #
# Shows the LxC/SxC metrics and firing rate indicies

@function_attributes(short_name=None, tags=['significance_bars', 'fig2', 'statistics'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-29 18:12', related_items=[])
def add_significance_bars(ax, p_value, x1, x2, y, significance_level:float=0.05, y_offset: Optional[float] = None, text_additional_y_offset: Optional[float] = None, ns_fontsize: float = 10, y_offset_frac: float = 0.08):
    """Add significance bars to a boxplot.

    When `y_offset` / `text_additional_y_offset` are None, they scale with |y|
    (`y_offset_frac * |y|` and `0.02 * |y|`) so participation (0–1) and Hz modes both get proportional headroom.


    Usage:

        # Add significance bars between groups
        ax = _fig_2_output_dict['theta'].ax
        _out_ann_AB = add_significance_bars(ax, LxC_Laps_T_result.pvalue, 0, 1, 6)
        # _out_ann_BC = add_significance_bars(ax, SxC_Laps_T_result.pvalue, 1, 2, 6)
        _out_ann_CD = add_significance_bars(ax, SxC_Laps_T_result.pvalue, 2, 3, 6)

    """
      # Offset for the bar above y
    y_scale: float = max(abs(float(y)), 1e-12)
    if y_offset is None:
        y_offset = y_offset_frac * y_scale
    if text_additional_y_offset is None:
        text_additional_y_offset = 0.02 * y_scale
    _out_ann = {}
    _out_ann['bar'] = ax.plot([x1, x1, x2, x2], [y, (y + y_offset), (y + y_offset), y], color='black')
    
    # Annotate with asterisks based on p-value
    sig_astrisks_str: str = 'N.S.'
    if p_value < significance_level:
        sig_astrisks_str: str = '*'
    if p_value < significance_level / 10:
        sig_astrisks_str += '*'
    if p_value < significance_level / 100:
        sig_astrisks_str += '*'
    ## OUTPUT:
    label_y: float = float(y + y_offset + text_additional_y_offset)
    if sig_astrisks_str:
        label_fontsize = ns_fontsize if (sig_astrisks_str == 'N.S.') else 16
        _out_ann['label'] = ax.text(((x1 + x2) / 2), label_y, sig_astrisks_str, fontsize=label_fontsize, ha='center')

    # create_plot may have pinned ylim before overlays; expand top so the bracket/label are not clipped (keep floor at 0)
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(0.0, max(float(_ymax), label_y * 1.08))

    return _out_ann



# Instantaneous versions:
@metadata_attributes(short_name=None, tags=['debugging'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-08-06 13:22', related_items=[])
class DebuggingHelpers:
    """ 
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import DebuggingHelpers
    
    """
    @function_attributes(short_name=None, tags=['participation', 'figure2', 'figure3', 'fixup'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-08-06 05:15', related_items=[])
    @classmethod
    def _compute_participation_and_n_spike_fr_stats(cls, across_session_inst_fr_computation_dict: Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation], debug_print: bool = False, **kwargs):
        """ 
        Usage:
        
            _out_new_dfs = DebuggingHelpers._compute_participation_and_n_spike_fr_stats(across_session_inst_fr_computation_dict=across_session_inst_fr_computation_dict)
            _out_new_dfs

        """
        from neuropy.core.neuron_identities import NeuronIdentityDataframeAccessor
        
        _out_new_dfs = []
        for a_session_ctxt, a_session_InstantaneousSpikeRateGroupsComputation in across_session_inst_fr_computation_dict.items():
            session_uid: str = a_session_ctxt.get_description_as_session_global_uid()
            print(f'processing session_uid: "{session_uid}"...')
        
            active_ACLUS = deepcopy(a_session_InstantaneousSpikeRateGroupsComputation.AnyC_aclus)
            active_neuron_UIDs: List[str] = [f"{session_uid}|{aclu}" for aclu in active_ACLUS]
            
            a_sess_pre_post_delta_result_list = [a_session_InstantaneousSpikeRateGroupsComputation.AnyC_ThetaDeltaMinus, a_session_InstantaneousSpikeRateGroupsComputation.AnyC_ThetaDeltaPlus, a_session_InstantaneousSpikeRateGroupsComputation.AnyC_ReplayDeltaMinus, a_session_InstantaneousSpikeRateGroupsComputation.AnyC_ReplayDeltaPlus]
            a_sess_pre_post_delta_result_dict = dict(zip(['ThetaDeltaMinus', 'ThetaDeltaPlus', 'ReplayDeltaMinus', 'ReplayDeltaPlus'], a_sess_pre_post_delta_result_list))

            _out_dict = {'aclu': active_ACLUS, 'neuron_uid': deepcopy(active_neuron_UIDs)} # , 'ThetaDeltaMinus': None, 'ThetaDeltaPlus': None, 'ReplayDeltaMinus': None, 'ReplayDeltaPlus': None
            # for a_pre_post_period_result in a_sess_pre_post_delta_result_list:
            for a_period_name, a_pre_post_period_result in a_sess_pre_post_delta_result_dict.items():
                an_inst_fr_list = a_pre_post_period_result.epoch_agg_inst_fr_list # (N_EPOCHS, N_ACLUS) in period
                # an_inst_fr_list = np.squeeze(a_pre_post_period_result.epoch_agg_inst_fr_list[:, target_aclu_idx]) # (N_EPOCHS) in period

                LR_an_inst_fr_list = an_inst_fr_list[::2, :] ## all aclus
                RL_an_inst_fr_list = an_inst_fr_list[1::2, :] ## all aclus

                a_period_directional_inst_fr_list = [LR_an_inst_fr_list, RL_an_inst_fr_list, an_inst_fr_list] # LR, RL, ALL
                a_period_epoch_agg_frs_list = np.vstack([np.nanmean(a_fr_list, axis=0) for a_fr_list in a_period_directional_inst_fr_list]) ## average over epochs, output (3, N_ACLUS)
                
                a_period_epoch_agg_fr: float = np.nanmax(a_period_epoch_agg_frs_list, axis=0) ## get the highest fr in any the LR/RL/ALL only
                if debug_print:
                    print(f'an_inst_fr_list.shape: {np.shape(an_inst_fr_list)}') # (41, N_ACLUS)
                    print(f'LR_an_inst_fr_list.shape: {np.shape(LR_an_inst_fr_list)}') # (n_LR_epochs, N_ACLUS)
                    print(f'RL_an_inst_fr_list.shape: {np.shape(RL_an_inst_fr_list)}') # (n_RL_epochs, N_ACLUS)
                    print(f'a_period_epoch_agg_frs_list.shape: {np.shape(a_period_epoch_agg_frs_list)}')
                    print(f'a_period_epoch_agg_fr.shape: {np.shape(a_period_epoch_agg_fr)}') # (20,)
                # a_period_epoch_agg_fr
                ## OVERWRITE a_pre_post_period_result
                # a_pre_post_period_result.cell_agg_inst_fr_list = deepcopy(a_period_epoch_agg_fr)
                a_result_col_name: str = 'fr'
                _out_dict[f"{a_period_name}_{a_result_col_name}"] = a_period_epoch_agg_fr
                
                a_result_col_name: str = 'n_participating_epochs'
                # has_epoch_participation: NDArray = np.vstack([(v.T[0].to_numpy() > 0.0) for v in a_pre_post_period_result.spike_counts_df_list]) # has_epoch_participation # .shape # (39, 20) - (n_epochs, n_aclus)
                # n_participating_epochs: NDArray = has_epoch_participation.sum(axis=0) # .shape (N_ACLUS)
                
                # assert len(a_pre_post_period_result.included_neuron_ids) == len(n_participating_epochs), f"len(a_pre_post_period_result.included_neuron_ids): {len(a_pre_post_period_result.included_neuron_ids)} != len(n_participating_epochs): {len(n_participating_epochs)}"
                # assert len(a_pre_post_period_result.included_neuron_ids) == len(active_neuron_UIDs), f"len(a_pre_post_period_result.included_neuron_ids): {len(a_pre_post_period_result.included_neuron_ids)} != len(active_neuron_UIDs): {len(active_neuron_UIDs)}"
                
                # input_df = input_df.neuron_identity.make_neuron_indexed_df_global(a_session_ctxt, add_expanded_session_context_keys=False, add_extended_aclu_identity_columns=False)
                # n_participating_epochs_dict = dict(zip(active_neuron_UIDs, n_participating_epochs))
            
                n_participating_epochs_dict, n_participating_epochs, has_epoch_participation, per_aclu_additional_properties_dict, skip_column_names = a_pre_post_period_result.compute_participation_stats(a_session_ctxt=a_session_ctxt, should_update_self=False, **kwargs)

                _out_dict[f"{a_period_name}_{a_result_col_name}"] = n_participating_epochs
                for k, v in per_aclu_additional_properties_dict.items():
                    if k not in skip_column_names:
                        _out_dict[f"{a_period_name}_{k}"] = deepcopy(v) 
                    
                
            _out_temp_df: pd.DataFrame = pd.DataFrame(_out_dict)
            _out_new_dfs.append(_out_temp_df)
            # _out_temp_df
        # an_inst_fr_list.shape: (41, 20)
        # LR_an_inst_fr_list.shape: (21, 20)
        # RL_an_inst_fr_list.shape: (20, 20)
        # a_period_epoch_agg_frs_list.shape: (3, 20)
        # a_period_epoch_agg_fr.shape: (20,)
        _out_new_dfs: pd.DataFrame = pd.concat(_out_new_dfs)
        return _out_new_dfs




    @classmethod
    def _compute_active_set_membership_from_participation(cls, across_session_inst_fr_computation_df: pd.DataFrame, all_neuron_stats_table: pd.DataFrame, 
                                                          contra_period_max_participating_epochs_ratio: float = 0.6,
                                                          dominant_period_min_participating_epochs_ratio: float = 0.3,
                                                          cell_LS_eXclusivity_threshold: Optional[float] = 0.75,
                                                          debug_print: bool = False):
        """ 
        Sets columns: ['active_set_membership_from_participation']
        
        Usage:
            contra_period_max_participating_epochs_ratio: float = 0.6 ## don't allow more than 0.2 stability for the opposite period
            dominant_period_min_participating_epochs_ratio: float = 0.3 ## allow this to be low and we'll control for this by requiring stability. how much stability is required for the cell's dominant period - #TODO 2025-08-06 08:20: - [ ] Note this has to be less than 0.5 because of directionality
            
            _out_new_dfs = DebuggingHelpers._compute_participation_and_n_spike_fr_stats(across_session_inst_fr_computation_dict=across_session_inst_fr_computation_dict)
            _out_new_dfs

        """
        ## Add XxC Constraints from `ratio_participating_epochs` columns
        variable_name: str = 'ratio_participating_epochs'

        # subplot_name: str = 'Laps'
        # # x_var_name: str = f'ThetaDeltaMinus_n_participating_epochs'
        # # y_var_name: str = f'ThetaDeltaPlus_n_participating_epochs'
        x_var_name: str = f'ThetaDeltaMinus_{variable_name}'
        y_var_name: str = f'ThetaDeltaPlus_{variable_name}'


        ## NOTE: uses the values in `all_neuron_stats_table` but actually updates (adding columns ['active_set_membership_from_stability']) to both `all_neuron_stats_table` AND `across_session_inst_fr_computation_df`
        ## INPUTS: all_neuron_stats_table, across_session_inst_fr_computation_df
        ## UPDATES: all_neuron_stats_table, across_session_inst_fr_computation_df, 
         ## don't allow more than 0.2 stability for the opposite period
        # dominant_period_min_participating_epochs_ratio: float = 0.3 ## allow this to be low and we'll control for this by requiring stability. how much stability is required for the cell's dominant period - #TODO 2025-08-06 08:20: - [ ] Note this has to be less than 0.5 because of directionality

        # participation_df = all_neuron_stats_table
        ## find effectively zero (unstable/no-pf) cells:    
        is_not_participating_long = (across_session_inst_fr_computation_df[x_var_name] <= contra_period_max_participating_epochs_ratio)
        is_not_participating_short = (across_session_inst_fr_computation_df[y_var_name] <= contra_period_max_participating_epochs_ratio)

        ## NOTE that `stable_*` is not the negation of `unstable_*` because they require a minimum `one_point_stability`
        is_participating_long = (across_session_inst_fr_computation_df[x_var_name] >= dominant_period_min_participating_epochs_ratio)
        is_participating_short = (across_session_inst_fr_computation_df[y_var_name] >= dominant_period_min_participating_epochs_ratio)

        is_participation_LdC = np.logical_and(is_participating_long, is_not_participating_short)
        is_participation_SdC = np.logical_and(is_participating_short, is_not_participating_long)

        # all_neuron_stats_table.loc[is_participation_LdC, 'neuron_uid']

        # stability_LdC_neuron_uids: List[str] = across_session_inst_fr_computation_df['neuron_uid'][is_participation_LdC].tolist()
        # stability_SdC_neuron_uids: List[str] = across_session_inst_fr_computation_df['neuron_uid'][is_participation_SdC].tolist()

        ## Adds the 'active_set_membership_from_stability' col directly to this df
        across_session_inst_fr_computation_df['active_set_membership_from_participation'] = 'AnyC'
        across_session_inst_fr_computation_df.loc[is_participation_LdC, 'active_set_membership_from_participation'] = 'LxC'
        across_session_inst_fr_computation_df.loc[is_participation_SdC, 'active_set_membership_from_participation'] = 'SxC'

        # participation_df

        ## Map new column to other dataframes:
        neuron_uid_to_active_set_membership_from_participation_dict = deepcopy(across_session_inst_fr_computation_df[['neuron_uid', 'active_set_membership_from_participation']]).set_index('neuron_uid', drop=True).to_dict()['active_set_membership_from_participation']

        across_session_inst_fr_computation_df['active_set_membership_from_participation'] = across_session_inst_fr_computation_df['neuron_uid'].map(lambda x: neuron_uid_to_active_set_membership_from_participation_dict.get(x, 'none')) ## Actually apply changes to `across_session_inst_fr_computation_df`
        if all_neuron_stats_table is not None:
            all_neuron_stats_table['active_set_membership_from_participation'] = all_neuron_stats_table['neuron_uid'].map(lambda x: neuron_uid_to_active_set_membership_from_participation_dict.get(x, 'none')) ## Actually apply changes to `all_neuron_stats_table`


        # Adds 'active_set_membership_from_n_spikes' _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        if cell_LS_eXclusivity_threshold is not None:
            # 'active_set_membership_from_n_spikes': New measure 2025-08-06 09:07 
            all_neuron_stats_table['is_n_spikes_LxC'] = all_neuron_stats_table['pct_long_n_spikes'] >= cell_LS_eXclusivity_threshold
            all_neuron_stats_table['is_n_spikes_SxC'] = all_neuron_stats_table['pct_short_n_spikes'] >= cell_LS_eXclusivity_threshold

            ## Adds the 'active_set_membership_from_stability' col directly to this df
            all_neuron_stats_table['active_set_membership_from_n_spikes'] = 'AnyC'
            all_neuron_stats_table.loc[all_neuron_stats_table['is_n_spikes_LxC'], 'active_set_membership_from_n_spikes'] = 'LxC'
            all_neuron_stats_table.loc[all_neuron_stats_table['is_n_spikes_SxC'], 'active_set_membership_from_n_spikes'] = 'SxC'

            # participation_df

            ## Map new column to other dataframes:
            neuron_uid_to_active_set_membership_from_n_spikes_dict = deepcopy(all_neuron_stats_table[['neuron_uid', 'active_set_membership_from_n_spikes']]).set_index('neuron_uid', drop=True).to_dict()['active_set_membership_from_n_spikes']

            across_session_inst_fr_computation_df['active_set_membership_from_n_spikes'] = across_session_inst_fr_computation_df['neuron_uid'].map(lambda x: neuron_uid_to_active_set_membership_from_n_spikes_dict.get(x, 'none')) ## Actually apply changes to `across_session_inst_fr_computation_df`
            all_neuron_stats_table['active_set_membership_from_n_spikes'] = all_neuron_stats_table['neuron_uid'].map(lambda x: neuron_uid_to_active_set_membership_from_n_spikes_dict.get(x, 'none')) ## Actually apply changes to `all_neuron_stats_table`


        ## OUTPUTS: across_session_inst_fr_computation_df
        return across_session_inst_fr_computation_df, all_neuron_stats_table, (is_not_participating_long, is_not_participating_short, is_participating_long, is_participating_short, is_participation_LdC, is_participation_SdC)



    @classmethod
    def _plot_debug_scatter(cls, _out_temp_df: pd.DataFrame, variable_name: str = 'ratio_participating_epochs', subplot_name: str = 'Laps',
                            x_var_name: str = f'ThetaDeltaMinus_', y_var_name: str = f'ThetaDeltaPlus_', active_set_membership_variable_col_name: Optional[str]=None):
        """ Inputs:


        Usage:        
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import DebuggingHelpers

            ## INPUTS: _out_temp_df
            _out_temp_df: pd.DataFrame = deepcopy(across_session_inst_fr_computation_df)

            shared_kwargs = dict(active_set_membership_variable_col_name='active_set_membership_from_stability')
            # variable_name: str = 'n_participating_epochs'
            variable_name: str = 'ratio_participating_epochs'

            subplot_name: str = 'Laps'
            x_var_name: str = f'ThetaDeltaMinus_{variable_name}'
            y_var_name: str = f'ThetaDeltaPlus_{variable_name}'
            _out_plot_ax_laps = DebuggingHelpers._plot_debug_scatter(_out_temp_df=_out_temp_df, variable_name=variable_name, subplot_name=subplot_name, x_var_name=x_var_name, y_var_name=y_var_name, **shared_kwargs)

            subplot_name: str = 'PBEs'
            x_var_name: str = f'ReplayDeltaMinus_{variable_name}'
            y_var_name: str = f'ReplayDeltaPlus_{variable_name}'
            _out_plot_ax_PBEs = DebuggingHelpers._plot_debug_scatter(_out_temp_df=_out_temp_df, variable_name=variable_name, subplot_name=subplot_name, x_var_name=x_var_name, y_var_name=y_var_name, **shared_kwargs)


        """
        

        ## Plot the result
        scatter_props_kwargs = dict(alpha=0.9)
        if (active_set_membership_variable_col_name is None) or (active_set_membership_variable_col_name not in _out_temp_df.columns):
            # _out_plot_ax = _out_temp_df.plot.scatter(x='ThetaDeltaMinus', y='ThetaDeltaPlus')
            _out_plot_ax = _out_temp_df.plot.scatter(x=x_var_name, y=y_var_name, **scatter_props_kwargs)
        else:
            fig, _out_plot_ax = plt.subplots(num=f'XdC_diagnostics - {variable_name} - {subplot_name}', clear=True)
            x_values = _out_temp_df[x_var_name].to_numpy()
            y_values = _out_temp_df[y_var_name].to_numpy()

            # Convert categorical values to integers or colors
            categories = _out_temp_df[active_set_membership_variable_col_name].astype('category')
            color_values = categories.cat.codes  # numeric mapping for color
            XxC_cat_to_marker_size_dict = {0: 1.0, 1: 15.0, 2: 15.0, 3: 1.0}
            scatter_props_kwargs['s'] = color_values.map(XxC_cat_to_marker_size_dict)

            # _out_plot_ax = _out_temp_df.plot.scatter(x='ThetaDeltaMinus_n_participating_epochs', y='ThetaDeltaPlus_n_participating_epochs', c=color_values, **scatter_props_kwargs)
            scatter_plot = _out_plot_ax.scatter(x_values, y_values, c=color_values, **scatter_props_kwargs)
            ## Only need legend if we're coloring by active set indentity:
            handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=scatter_plot.cmap(scatter_plot.norm(code))) for code in range(len(categories.cat.categories))]
            _out_plot_ax.legend(handles, categories.cat.categories, title='stability_class', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                    #    loc='upper center', ncol=len(categories.cat.categories),
                        ) # , bbox_to_anchor=(0.5, -0.15)

        plt.title(f'XdC_diagnostics - {variable_name} - {subplot_name}')
        # plt.suptitle(f'Stability Group Diagnostics:\n(max_contra={contra_period_max_permitted_stability}, min_dom={dominant_period_min_stability})')
        # plots_dict = {'scatter': scatter_plot}
            
        # Set the x and y axes to standard limits for easy visual comparison across sessions
        # _out_plot_ax.set_xlim([-1.1, 1.1])
        # _out_plot_ax.set_ylim([-1.1, 1.1])

        # Add y=x diagonal line:
        diagonal_y_equals_x_line_kwargs = dict(linestyle='--', color='gray', label='y=x')
        _out_plot_ax.plot(_out_plot_ax.get_xlim(), _out_plot_ax.get_ylim(), **diagonal_y_equals_x_line_kwargs)
        # Set the x and y axes to standard limits for easy visual comparison across sessions
        # _out_plot_ax.set_xlim([-1.1, 1.1])
        # _out_plot_ax.set_ylim([-1.1, 1.1])

        _out_plot_ax.set_xlabel(x_var_name)
        _out_plot_ax.set_ylabel(y_var_name)
        plt.tight_layout()

        return _out_plot_ax


# @overwriting_display_context(
@metadata_attributes(short_name=None, tags=['figure_2', 'paper', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-26 21:36', related_items=[])
@define(slots=False, repr=False)
class PaperFigureTwo(SerializedAttributesAllowBlockSpecifyingClass):
    """ Plots the LxC/SxC bar graph containing full instantaneous firing rate computations for both Long and Short epochs
    
    Usage:
        _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # 10ms
        _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline)
        _out_fig_2.display()

    """

    instantaneous_time_bin_size_seconds: float = 0.01 # 10ms
    
    computation_result: InstantaneousSpikeRateGroupsComputation = field(init=False)
    active_identifying_session_ctx: IdentifyingContext = field(init=False)

    _pipeline_file_callback_fn: Callable = field(init=False, repr=False, default=None) # this callback is 2737.983306 MB!!


    @classmethod
    def get_bar_colors(cls):
        # return [(1.0, 0, 0, 1), (0.65, 0, 0, 1), (0, 0, 0.65, 1), (0, 0, 1.0, 1)] # corresponding colors
        # return [(0, 0, 1.0, 1), (0, 0, 0.65, 1), (0.65, 0, 0, 1), (1.0, 0, 0, 1)] # long=Blue, short=Red -- dimmer -- 4 separate colors (2 shades of red and 2 shades of blue)
        return [(0, 0, 0.65, 1), (0, 0, 0.65, 1), (0.65, 0, 0, 1), (0.65, 0, 0, 1)] # long=Blue, short=Red -- dimmer -- only 2 colors




    def compute(self, curr_active_pipeline, **kwargs):
        """ full instantaneous computations for both Long and Short epochs:
        
        Can access via:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PaperFigureTwo

            _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # 10ms
            _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
            out_inst_fr_comps = _out_fig_2.computation_result
            LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus = _out_inst_fr_comps.LxC_ReplayDeltaMinus, _out_inst_fr_comps.LxC_ReplayDeltaPlus, _out_inst_fr_comps.SxC_ReplayDeltaMinus, _out_inst_fr_comps.SxC_ReplayDeltaPlus
            LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus = _out_inst_fr_comps.LxC_ThetaDeltaMinus, _out_inst_fr_comps.LxC_ThetaDeltaPlus, _out_inst_fr_comps.SxC_ThetaDeltaMinus, _out_inst_fr_comps.SxC_ThetaDeltaPlus
        
        """
        self.computation_result = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        self.computation_result.compute(curr_active_pipeline, **kwargs)
        self.active_identifying_session_ctx = self.computation_result.active_identifying_session_ctx

        # Set callback, the only self-specific property
        self._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)
        
    
    @classmethod
    def _coerce_axis_label(cls, label) -> str:
        """ Unwrap accidental 1-tuples/lists from trailing-comma assignments; return a plain str. """
        if isinstance(label, (tuple, list)) and (len(label) == 1):
            label = label[0]
        return str(label)


    @classmethod
    def _metric_name_from_ylabel(cls, ylabel: str) -> str:
        """ Derive the flexitext title metric phrase from the y-axis label. """
        ylabel_lower: str = cls._coerce_axis_label(ylabel).lower()
        if 'participat' in ylabel_lower:
            return 'Participation Fractions'
        if 'firing rate' in ylabel_lower:
            return 'Firing Rates'
        return 'Firing Rates'


    @classmethod
    def _build_formatted_title_string(cls, epochs_name, prepare_for_publication: bool = False, metric_name: str = 'Firing Rates', subtitle: Optional[str] = None) -> str:
        """ buidls the two line colored string that is passed into `flexitext`.
        """
        if not prepare_for_publication:
            title_font_size: int = 22
            subtitle_font_size: int = 14
        else:
            title_font_size: int = 9*2
            subtitle_font_size: int = 7*2

        if subtitle is None:
            subtitle = ('for the '
                        "<color:royalblue, weight:bold>Long</>/<color:crimson, weight:bold>Short</> dominant Cells on each track")

        return (f"<size:{title_font_size}><weight:bold>{epochs_name}</> {metric_name}\n"
                f"<size:{subtitle_font_size}>{subtitle}</></>"
                )


    @classmethod
    def _apply_publication_grouped_xticklabels(cls, ax, group_labels: Tuple[str, str] = ('LdCs', 'SdCs')):
        """ Two-tier publication x-axis: pre-/post-$\\Delta$ under each bar, group labels under each pair. """
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels([r'pre-$\Delta$', r'post-$\Delta$', r'pre-$\Delta$', r'post-$\Delta$'])
        ax.tick_params(axis='x', length=0, pad=2)
        ax.set_xlabel('')

        # Group labels centered under bars 0–1 and 2–3 (matplotlib multilevel-tick pattern)
        sec = ax.secondary_xaxis(location=0)
        sec.set_xticks([0.5, 2.5], labels=[f'\n\n{group_labels[0]}', f'\n\n{group_labels[1]}'])
        sec.tick_params('x', length=0)
        sec.spines['bottom'].set_visible(False)
        return sec


    @classmethod
    def create_plot(cls, x_labels, y_values, scatter_props, ylabel, title, fig_name, active_context, defer_show, title_modifier=None, prepare_for_publication: bool = False, **kwargs):
        """ Main plotting function, called by both Laps and PBE subfigures
        
        prepare_for_publication: if True, no context footer is added; also skips title_modifier (e.g. '(N sessions)') and uses publication epoch titles
        """
        if title_modifier and (not prepare_for_publication):
            title = title_modifier(title)

        ylabel = cls._coerce_axis_label(ylabel)

        # fig, ax, bars, scatter_plots, title_text_obj, footer_text_obj, plots_data = cls.create_bar_plot(x_labels, y_values, scatter_props, active_context, ylabel, title)
        text_formatter = FormattedFigureText()
        # Axes top from kwargs (display default 0.78); title sits slightly higher so sig labels clear the subtitle
        top_margin: float = float(kwargs.pop('top_margin', 0.78))
        left_margin: float = float(kwargs.pop('left_margin', text_formatter.left_margin))
        bottom_margin: float = float(kwargs.pop('bottom_margin', text_formatter.bottom_margin))
        title_top_margin: float = float(kwargs.pop('title_top_margin', top_margin + 0.02))
        publication_group_labels = kwargs.pop('publication_group_labels', ('LdCs', 'SdCs'))
        title_subtitle = kwargs.pop('title_subtitle', None)
        metric_name_override = kwargs.pop('metric_name', None)
        xlabel: str = kwargs.pop('xlabel', 'Neuron Track Dominance Classification')
        text_formatter.top_margin = top_margin
        text_formatter.left_margin = left_margin
        text_formatter.bottom_margin = bottom_margin
        x = np.arange(len(x_labels))
        width = 0.3

        fig, ax = plt.subplots()
        text_formatter.setup_margins(fig)

        bars = ax.bar(x, [np.mean(yi) for yi in y_values], yerr=[np.std(yi) for yi in y_values], capsize=5, width=width, tick_label=x_labels, color=(0, 0, 0, 0), edgecolor=cls.get_bar_colors())

        scatter_plots = []
        x_values_list = []
        
        if hasattr(cls, 'scatter_props_fn'):
            # get individual scatter_props_Fn
            _output = cls.scatter_props_fn(active_context)
            print(f'_output: {_output}')


        for i in range(len(x)):
            x_values = (x[i] + np.random.random(y_values[i].size) * width - width / 2)
            scatter_props_kwargs: Dict = list_of_dicts_to_dict_of_lists(scatter_props[i])
            had_custom_marker_handling = False
            user_markers = scatter_props_kwargs.get('marker', None)
            if user_markers is not None:
                # if marker is just a scalar we can do the normal (flat) ax.scatter(...) command. Otherwise we have to loop.
                if isinstance(user_markers, (list, tuple, np.ndarray)):
                    ## have to loop. 
                    user_markers = scatter_props_kwargs.pop('marker') # remove from `scatter_props_kwargs`
                    scatter_plot = [] # scatter plot is just going to be a list here, hope that's okay.
                    for point_index, a_marker in enumerate(user_markers):
                        a_sub_scatter_plot = ax.scatter(x_values[point_index], y_values[i][point_index], color=cls.get_bar_colors()[i], marker=user_markers[point_index], **scatter_props_kwargs) # the np.random part is to spread the points out along the x-axis within their bar so they're visible and don't overlap.
                        scatter_plot.append(a_sub_scatter_plot)
                    ## END for point_index, a_marker in enumerate(user_markers)...

                    had_custom_marker_handling = True # indicate that this was handled in this manner

            if not had_custom_marker_handling:
                # No special marker or just a scalar specific marker value, no need to loop
                scatter_plot = ax.scatter(x_values, y_values[i], color=cls.get_bar_colors()[i], **scatter_props_kwargs) # the np.random part is to spread the points out along the x-axis within their bar so they're visible and don't overlap.
    
            scatter_plots.append(scatter_plot)
            x_values_list.append(x_values)
        ## END for i in range(len(x))...

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # Hide the right and top spines (box components)
        ax.spines[['right', 'top']].set_visible(False)

        metric_name: str = metric_name_override if metric_name_override is not None else cls._metric_name_from_ylabel(ylabel)
        _display_fn_name: str = 'Particp_bar_graphs' if (metric_name == 'Participation Fractions') else 'inst_FR_bar_graphs'
        active_context = active_context.overwriting_context(display_fn_name=_display_fn_name)
        title_text_obj = flexitext(text_formatter.left_margin, title_top_margin,
                                cls._build_formatted_title_string(epochs_name=title, prepare_for_publication=prepare_for_publication, metric_name=metric_name, subtitle=title_subtitle), va="bottom", xycoords="figure fraction")
        text_objects = {'title': title_text_obj}
        
        if (not prepare_for_publication):
            footer_text_obj = flexitext(text_formatter.left_margin * 0.1, text_formatter.bottom_margin * 0.25,
                                        text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction") ## How did that get added? and when?
            text_objects['footer'] = footer_text_obj
            

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        if prepare_for_publication:
            text_objects['publication_xgroup_axis'] = cls._apply_publication_grouped_xticklabels(ax, group_labels=publication_group_labels)

        # Firing rates / participation fractions are non-negative; keep axis floor at 0
        _ymin, _ymax = ax.get_ylim()
        ax.set_ylim(0.0, max(float(_ymax), 0.0))

        plot_data = (x_values_list, y_values)
        
        if not defer_show:
            plt.show()

        return MatplotlibRenderPlots(name=fig_name, figures=[fig], axes=[ax], context=active_context,
                                    plot_objects={'bars': bars, 'scatter_plots': scatter_plots, 'text_objects': text_objects},
                                    plot_data=plot_data)


    @classmethod
    def _resolve_AnyC_bar_results(cls, computation_result: InstantaneousSpikeRateGroupsComputation) -> Tuple[Optional[List[SingleBarResult]], Optional[List[SingleBarResult]]]:
        """ Return (Fig2_Laps_AnyC_FR, Fig2_Replay_AnyC_FR) from shell fields, or derive from live AnyC_* trends. """
        laps_anyc = getattr(computation_result, 'Fig2_Laps_AnyC_FR', None)
        replay_anyc = getattr(computation_result, 'Fig2_Replay_AnyC_FR', None)
        if (laps_anyc is not None) and (replay_anyc is not None):
            return laps_anyc, replay_anyc

        anyc_theta_minus = getattr(computation_result, 'AnyC_ThetaDeltaMinus', None)
        anyc_theta_plus = getattr(computation_result, 'AnyC_ThetaDeltaPlus', None)
        anyc_replay_minus = getattr(computation_result, 'AnyC_ReplayDeltaMinus', None)
        anyc_replay_plus = getattr(computation_result, 'AnyC_ReplayDeltaPlus', None)
        anyc_aclus = getattr(computation_result, 'AnyC_aclus', None)
        if (anyc_aclus is None) or (anyc_theta_minus is None) or (anyc_theta_plus is None):
            return None, None

        LxC_set = set(np.asarray(getattr(computation_result, 'LxC_aclus', np.array([]))).tolist())
        SxC_set = set(np.asarray(getattr(computation_result, 'SxC_aclus', np.array([]))).tolist())
        exclusive_set = LxC_set | SxC_set
        anyc_aclus_arr = np.asarray(anyc_aclus)
        keep_mask = np.array([(aclu not in exclusive_set) for aclu in anyc_aclus_arr])
        filtered_aclus = anyc_aclus_arr[keep_mask]

        def _bar_from_trend(trend) -> SingleBarResult:
            if trend is None:
                return SingleBarResult(None, None, np.array([], dtype=float), filtered_aclus, filtered_aclus, None, None)
            vals = np.asarray(trend.cell_agg_inst_fr_list)
            if len(vals) == len(keep_mask):
                vals = vals[keep_mask]
            mean_v = float(np.nanmean(vals)) if len(vals) > 0 else None
            std_v = float(np.nanstd(vals)) if len(vals) > 0 else None
            return SingleBarResult(mean_v, std_v, vals, filtered_aclus, filtered_aclus, None, None)

        Fig2_Laps_AnyC_FR: List[SingleBarResult] = [_bar_from_trend(anyc_theta_minus), _bar_from_trend(anyc_theta_plus)]
        if (anyc_replay_minus is not None) and (anyc_replay_plus is not None):
            Fig2_Replay_AnyC_FR: List[SingleBarResult] = [_bar_from_trend(anyc_replay_minus), _bar_from_trend(anyc_replay_plus)]
        else:
            Fig2_Replay_AnyC_FR = None
        return Fig2_Laps_AnyC_FR, Fig2_Replay_AnyC_FR


    @classmethod
    def add_non_XdC_background_scatter(cls, ax, anyc_bar_results: List[SingleBarResult], width: float = 0.3, n_main_bars: int = 4, **kwargs):
        """ Scatter AnyC (non-XdC) Δ−/Δ+ points faintly behind the main XdC 4-bar figure.

        Mapping by epoch column:
            AnyC Δ− -> bars 0 and 2 (LxC Δ−, SxC Δ−)
            AnyC Δ+ -> bars 1 and 3 (LxC Δ+, SxC Δ+)

        Y-limits are restored after plotting so AnyC extremes do not expand the axis.
        """
        if (anyc_bar_results is None) or (len(anyc_bar_results) < 2):
            return None

        y_minus = np.asarray(anyc_bar_results[0].values) if (anyc_bar_results[0] is not None and anyc_bar_results[0].values is not None) else np.array([], dtype=float)
        y_plus = np.asarray(anyc_bar_results[1].values) if (anyc_bar_results[1] is not None and anyc_bar_results[1].values is not None) else np.array([], dtype=float)
        if (len(y_minus) == 0) and (len(y_plus) == 0):
            return None

        ylim0 = ax.get_ylim()
        # Ensure existing XdC artists stay above the faint AnyC series
        for artist in list(ax.patches) + list(ax.collections):
            try:
                if artist.get_zorder() < 2:
                    artist.set_zorder(2)
            except Exception:
                pass
        ## END for artist in list(ax.patches) + list(ax.collections)...

        scatter_artists = []
        x = np.arange(n_main_bars)
        for i in range(n_main_bars):
            y_i = y_minus if (i in (0, 2)) else y_plus
            if len(y_i) == 0:
                continue
            x_values = (x[i] + np.random.random(y_i.size) * width - width / 2)
            sc = ax.scatter(x_values, y_i, color=(0.5, 0.5, 0.5), alpha=0.50, s=20, zorder=1, marker='.', linewidths=0)
            scatter_artists.append(sc)
        ## END for i in range(n_main_bars)...

        ax.set_ylim(0.0, max(float(ylim0[1]), 0.0))
        return scatter_artists if len(scatter_artists) > 0 else None


    @classmethod
    def add_non_XdC_inset(cls, ax, anyc_bar_results: List[SingleBarResult], **kwargs):
        """ Deprecated alias for `add_non_XdC_background_scatter` (kept for notebook autoreload / old callers). """
        return cls.add_non_XdC_background_scatter(ax, anyc_bar_results, **kwargs)


    # @providing_context(fig='F2', frs='Laps')
    @classmethod
    def fig_2_Theta_FR_matplotlib(cls, Fig2_Laps_FR, defer_show=False, active_context=None, title_modifier=None, prepare_for_publication: bool = False, **kwargs) -> MatplotlibRenderPlots:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PhoPublicationFigureHelper
        assert active_context is not None
        
        delta_minus_str: str = kwargs.get('delta_minus_str', PhoPublicationFigureHelper.delta_minus_str) # 't < \\Delta'        
        delta_plus_str: str = kwargs.get('delta_plus_str', PhoPublicationFigureHelper.delta_plus_str) # 't > \\Delta'

        var_name:str = 'Laps'
        active_context = active_context.adding_context_if_missing(variable=var_name) # title='Laps'
        # Laps_fr_label: str = '\\theta'
        Laps_fr_label: str = '\\text{Laps}'

        x_labels = [f'$L d C$\t${Laps_fr_label}' + '_{' + delta_minus_str + '}$', f'$L d C$\t${Laps_fr_label}' + '_{' + delta_plus_str + '}$', f'$S d C$\t${Laps_fr_label}' + '_{' + delta_minus_str + '}$', f'$S d C$\t${Laps_fr_label}' + '_{' + delta_plus_str + '}$']
        # LxC vs SxC bars have different n cells — keep a list (np.array would raise on inhomogeneous shapes)
        all_data_points = [np.asarray(v.values) for v in Fig2_Laps_FR]
        # all_scatter_props =  Fig2_Laps_FR[0].LxC_scatter_props + Fig2_Laps_FR[1].LxC_scatter_props + Fig2_Laps_FR[2].SxC_scatter_props + Fig2_Laps_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 

        if Fig2_Laps_FR[0].LxC_scatter_props is not None:
            # all_scatter_props =  Fig2_Laps_FR[0].LxC_scatter_props + Fig2_Laps_FR[1].LxC_scatter_props + Fig2_Laps_FR[2].SxC_scatter_props + Fig2_Laps_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 
            all_scatter_props =  [Fig2_Laps_FR[0].LxC_scatter_props, Fig2_Laps_FR[1].LxC_scatter_props, Fig2_Laps_FR[2].SxC_scatter_props, Fig2_Laps_FR[3].SxC_scatter_props]

        else:
            all_scatter_props = [{}, {}, {}, {}]

        all_scatter_props = [{}, {}, {}, {}] # override, 2023-10-03

        laps_y_axis_label: str = cls._coerce_axis_label(kwargs.get('laps_y_axis_label', 'Lap-averaged Firing Rates (Hz)'))
        epochs_title: str = 'Lap' if prepare_for_publication else 'Lap ($\\theta$)'

        return cls.create_plot(x_labels, all_data_points, all_scatter_props, laps_y_axis_label, epochs_title, 'fig_2_Theta_FR_matplotlib', active_context=active_context, defer_show=defer_show, title_modifier=title_modifier, prepare_for_publication=prepare_for_publication, **kwargs)

    # @providing_context(fig='F2', frs='Replay')
    @classmethod
    def fig_2_Replay_FR_matplotlib(cls, Fig2_Replay_FR, defer_show=False, active_context=None, title_modifier=None, prepare_for_publication: bool = False, **kwargs) -> MatplotlibRenderPlots:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PhoPublicationFigureHelper

        assert active_context is not None
        
        delta_minus_str: str = kwargs.get('delta_minus_str', PhoPublicationFigureHelper.delta_minus_str) # 't < \\Delta'        
        delta_plus_str: str = kwargs.get('delta_plus_str', PhoPublicationFigureHelper.delta_plus_str) # 't > \\Delta'
        
        var_name:str = 'PBE'
        active_context = active_context.adding_context_if_missing(variable=var_name) # title='Laps'

        pbe_y_axis_label: str = cls._coerce_axis_label(kwargs.get('pbe_y_axis_label', 'PBE-averaged Firing Rates (Hz)'))

        # PBE_fr_label: str = 'R'
        PBE_fr_label: str = '\\text{PBE}'

        x_labels = [f'$L d C$\t${PBE_fr_label}' + '_{' + delta_minus_str + '}$', f'$L d C$\t${PBE_fr_label}' + '_{' + delta_plus_str + '}$', f'$S d C$\t${PBE_fr_label}' + '_{' + delta_minus_str + '}$', f'$S d C$\t${PBE_fr_label}' + '_{' +  delta_plus_str + '}$']
        assert len(Fig2_Replay_FR) == 4
        # LxC vs SxC bars have different n cells — keep a list (np.array would raise on inhomogeneous shapes)
        all_data_points = [np.asarray(v.values) for v in Fig2_Replay_FR]
        # all_scatter_props = Fig2_Replay_FR[0].LxC_scatter_props + Fig2_Replay_FR[1].LxC_scatter_props + Fig2_Replay_FR[2].SxC_scatter_props + Fig2_Replay_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway.

        if Fig2_Replay_FR[0].LxC_scatter_props is not None:
            # all_scatter_props =  Fig2_Laps_FR[0].LxC_scatter_props + Fig2_Laps_FR[1].LxC_scatter_props + Fig2_Laps_FR[2].SxC_scatter_props + Fig2_Laps_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 
            all_scatter_props =  [Fig2_Replay_FR[0].LxC_scatter_props, Fig2_Replay_FR[1].LxC_scatter_props, Fig2_Replay_FR[2].SxC_scatter_props, Fig2_Replay_FR[3].SxC_scatter_props]
        else:
            all_scatter_props = [{}, {}, {}, {}]
            
        all_scatter_props = [{}, {}, {}, {}] # override, 2023-10-03
        # label_list = [LxC_aclus, LxC_aclus, SxC_aclus, SxC_aclus]
        return cls.create_plot(x_labels, all_data_points, all_scatter_props, pbe_y_axis_label, 'PBE', 'fig_2_Replay_FR_matplotlib', active_context=active_context, defer_show=defer_show, title_modifier=title_modifier, prepare_for_publication=prepare_for_publication, **kwargs)


    @providing_context(fig='2', display_fn_name='inst_FR_bar_graphs')
    def display(self, defer_show=False, save_figure=True, enable_tiny_point_labels=True, enable_hover_labels=False, enabled_point_connection_lines=True, enable_stats_overlays:bool=True, active_context=None, title_modifier_fn=None, top_margin=0.78, left_margin=0.090, bottom_margin=0.150, prepare_for_publication: bool = False, debug_print: bool = False, **kwargs):
        """ 
        
        title_modifier: lambda original_title: f"{original_title} (all sessions)"

        Values come from:
        `self.computation_result.Fig2_Laps_FR`
        `self.computation_result.Fig2_Replay_FR`

        When `include_non_XdC_cells_as_inset=True`, also plots a faint AnyC (non-XdC)
        scatter series behind the main XdC bars from `Fig2_*_AnyC_FR` or live `AnyC_*` trends
        (AnyC Δ− under bars 0/2, AnyC Δ+ under bars 1/3; XdC y-limits preserved).
        """
        
        
        # Get the provided context or use the session context:
        active_context = active_context if active_context is not None else self.active_identifying_session_ctx
        title_modifier = title_modifier_fn if title_modifier_fn is not None else (lambda original_title: original_title)

        include_non_XdC_cells_as_inset: bool = kwargs.pop('include_non_XdC_cells_as_inset', False)

        _fig_2_theta_out = self.fig_2_Theta_FR_matplotlib(self.computation_result.Fig2_Laps_FR, defer_show=defer_show,
                                                        active_context=active_context, top_margin=top_margin,
                                                        left_margin=left_margin, bottom_margin=bottom_margin,
                                                        title_modifier=title_modifier, prepare_for_publication=prepare_for_publication, debug_print=debug_print, **kwargs)

        _fig_2_replay_out = self.fig_2_Replay_FR_matplotlib(self.computation_result.Fig2_Replay_FR, defer_show=defer_show,
                                                            active_context=active_context, top_margin=top_margin,
                                                            left_margin=left_margin, bottom_margin=bottom_margin,
                                                            title_modifier=title_modifier, prepare_for_publication=prepare_for_publication, debug_print=debug_print, **kwargs)


        if (enable_hover_labels or enable_tiny_point_labels or enabled_point_connection_lines):
            LxC_aclus = self.computation_result.LxC_aclus
            SxC_aclus = self.computation_result.SxC_aclus
            _fig_2_theta_out = self.add_optional_aclu_labels(_fig_2_theta_out, LxC_aclus, SxC_aclus, enable_tiny_point_labels=enable_tiny_point_labels, enable_hover_labels=enable_hover_labels, enabled_point_connection_lines=enabled_point_connection_lines, debug_print=debug_print)
            _fig_2_replay_out = self.add_optional_aclu_labels(_fig_2_replay_out, LxC_aclus, SxC_aclus, enable_tiny_point_labels=enable_tiny_point_labels, enable_hover_labels=enable_hover_labels, enabled_point_connection_lines=enabled_point_connection_lines, debug_print=debug_print)
        


        if enable_stats_overlays:
            ## Compute and add the bar graph stats and significance bars
            ## Add stats from t-tests (check?)
            ## INPUTS: across_session_inst_fr_computation_shell_obj
            LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result = pho_stats_bar_graph_t_tests(self.computation_result)
            bar_graph_stats_str: str = format_bar_graph_t_test_pvalues_str(LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result)
            print(bar_graph_stats_str)
            bar_graph_t_results = (LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result)
            # Stash on the returned figure containers (picked up by across_sessions_bar_graphs output dict)
            _fig_2_theta_out['bar_graph_stats_str'] = bar_graph_stats_str
            _fig_2_theta_out['bar_graph_t_results'] = bar_graph_t_results
            _fig_2_replay_out['bar_graph_stats_str'] = bar_graph_stats_str
            _fig_2_replay_out['bar_graph_t_results'] = bar_graph_t_results

            ## get the max datapoint so we can position the signifiance indicator bars above them:
            max_laps_y: float = float(np.nanmax(np.concatenate([v.values for v in self.computation_result.Fig2_Laps_FR])))
            max_PBEs_y: float = float(np.nanmax(np.concatenate([v.values for v in self.computation_result.Fig2_Replay_FR])))
            
            # 'Fig2_Laps_FR'

            # LxC_Laps_T_result: TtestResult(statistic=6.712433589492588, pvalue=5.2823734983320806e-05, df=10)
            # SxC_Laps_T_result: TtestResult(statistic=-4.339172228913557, pvalue=0.002481149212058544, df=8)
            # LxC_Replay_T_result: TtestResult(statistic=-1.1080728076872137, pvalue=0.29376948620786025, df=10)
            # SxC_Replay_T_result: TtestResult(statistic=-2.952880002807658, pvalue=0.018344744614346115, df=8)

            # LxC_Laps_T_result: TtestResult(statistic=4.91664380764766, pvalue=8.326747683573472e-05, df=20)
            # SxC_Laps_T_result: TtestResult(statistic=-4.836261942292889, pvalue=0.00011472243045619965, df=19)
            # LxC_Replay_T_result: TtestResult(statistic=0.6019865878499693, pvalue=0.5539469660448748, df=20)
            # SxC_Replay_T_result: TtestResult(statistic=-4.580715643685985, pvalue=0.00020418641915182124, df=19)

            # Add significance bars between groups
            # ax = _fig_2_output_dict['theta'].ax
            ax = _fig_2_theta_out.ax
            # ax = _fig_2_theta_out.axes[0] # one shared axis per figure
            _out_ann_AB = add_significance_bars(ax, LxC_Laps_T_result.pvalue, x1=0, x2=1, y=max_laps_y)
            # _out_ann_BC = add_significance_bars(ax, SxC_Laps_T_result.pvalue, 1, 2, 6)
            _out_ann_CD = add_significance_bars(ax, SxC_Laps_T_result.pvalue, x1=2, x2=3, y=max_laps_y)

            # _out_ann_BD = add_significance_bars(ax, SxC_Laps_T_result.pvalue, 0.5, 2.5, 8)

            # _out_ann_AC = add_significance_bars(ax, 0.005, 0, 2, 15)


            # Add significance bars between groups
            # ax = _fig_2_output_dict['replay'].ax
            ax = _fig_2_replay_out.ax
            _out_ann_AB = add_significance_bars(ax, LxC_Replay_T_result.pvalue, x1=0, x2=1, y=max_PBEs_y)
            # _out_ann_BC = add_significance_bars(ax, SxC_Laps_T_result.pvalue, 1, 2, 6)
            _out_ann_CD = add_significance_bars(ax, SxC_Replay_T_result.pvalue, x1=2, x2=3, y=max_PBEs_y)


        if include_non_XdC_cells_as_inset:
            Fig2_Laps_AnyC_FR, Fig2_Replay_AnyC_FR = self._resolve_AnyC_bar_results(self.computation_result)
            if Fig2_Laps_AnyC_FR is not None:
                anyc_scatters_theta = self.add_non_XdC_background_scatter(_fig_2_theta_out.ax, Fig2_Laps_AnyC_FR)
                if anyc_scatters_theta is not None:
                    if getattr(_fig_2_theta_out, 'plot_objects', None) is None:
                        _fig_2_theta_out.plot_objects = {}
                    _fig_2_theta_out.plot_objects['anyc_background_scatter'] = anyc_scatters_theta
            if Fig2_Replay_AnyC_FR is not None:
                anyc_scatters_replay = self.add_non_XdC_background_scatter(_fig_2_replay_out.ax, Fig2_Replay_AnyC_FR)
                if anyc_scatters_replay is not None:
                    if getattr(_fig_2_replay_out, 'plot_objects', None) is None:
                        _fig_2_replay_out.plot_objects = {}
                    _fig_2_replay_out.plot_objects['anyc_background_scatter'] = anyc_scatters_replay


        # Keep y-axis floor at 0 after overlays (stats bars / AnyC) may have re-autoscaled
        for _fig_out in [_fig_2_theta_out, _fig_2_replay_out]:
            _ymin, _ymax = _fig_out.ax.get_ylim()
            _fig_out.ax.set_ylim(0.0, max(float(_ymax), 0.0))
        ## END for _fig_out in [_fig_2_theta_out, _fig_2_replay_out]...

        def _perform_write_to_file_callback():
            ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
            return (self.perform_save(_fig_2_theta_out.context, _fig_2_theta_out.figures[0]), 
                    self.perform_save(_fig_2_replay_out.context, _fig_2_replay_out.figures[0]))

        if save_figure:
            _fig_2_theta_out['saved_figures'], _fig_2_replay_out['saved_figures'] = _perform_write_to_file_callback()
        else:
            _fig_2_theta_out['saved_figures'], _fig_2_replay_out['saved_figures'] = [], []

        # Merge the two (_fig_2_theta_out | _fig_2_replay_out)
        return (_fig_2_theta_out, _fig_2_replay_out)


    def perform_save(self, *args, **kwargs):
        """ used to save the figure without needing a hard reference to curr_active_pipeline """
        assert self._pipeline_file_callback_fn is not None
        return self._pipeline_file_callback_fn(*args, **kwargs) # call the saved callback

    @classmethod
    def add_optional_aclu_labels(cls, a_fig_container, LxC_aclus, SxC_aclus, enable_hover_labels=True, enable_tiny_point_labels=True, enabled_point_connection_lines=True, debug_print: bool = False, **kwargs):
        """ Adds disambiguating labels to each of the scatterplot points. Important for specifying which ACLU is plotted.


        Parameters:
            enable_hover_labels = True # add interactive point hover labels using mplcursors
            enable_tiny_point_labels = True # add static tiny aclu labels beside each point

        Uses:
            a_fig_container['plot_objects']['scatter_plots']
            a_fig_container['plot_data']
            
        Modifies a_fig_container, adding:
            a_fig_container['plot_objects']['tiny_annotation_labels']
            a_fig_container['plot_objects']['hover_label_objects']
        """
        if enable_hover_labels:
            import mplcursors # for hover tooltips that specify the aclu of the selected point

        # LxC_aclus = _out_fig_2.computation_result.LxC_aclus
        # SxC_aclus = _out_fig_2.computation_result.SxC_aclus
        label_list = [LxC_aclus, LxC_aclus, SxC_aclus, SxC_aclus]
            
        fig = a_fig_container.figures[0]
        ax = a_fig_container.axes[0] # one shared axis per figure
        plot_data = a_fig_container['plot_data']
        x_values_list, y_values_list = plot_data # four lists of x and y values,

        assert len(x_values_list) == 4
        assert len(y_values_list) == 4
        
        # Loop through the four bars in the container:
        if enable_tiny_point_labels:
            a_fig_container['plot_objects']['tiny_annotation_labels'] = []
        if enable_hover_labels:
            a_fig_container['plot_objects']['hover_label_objects'] = []
        if enabled_point_connection_lines:
            a_fig_container['plot_objects']['point_connection_lines'] = {} # aclu:(start_point, end_point)
            # start_points = []
            # end_points = []
            # Draw arrows from the first set of points to the second set _________________________________________________________ #
            # arrowprops_kwargs = dict(arrowstyle="->", alpha=0.6)
            # arrowprops_kwargs = dict(arrowstyle="simple", alpha=0.7)
            arrowprops_kwargs = dict(arrowstyle="fancy, head_length=0.25, head_width=0.25, tail_width=0.05", alpha=0.4)
            a_fig_container['plot_objects']['long_to_short_arrow'] = {}

        if (enable_hover_labels or enable_tiny_point_labels or enabled_point_connection_lines):
            
            for bar_idx, a_scatter_plot, x_values, y_values, active_labels in zip(np.arange(4), a_fig_container['plot_objects']['scatter_plots'], x_values_list, y_values_list, label_list): # four scatter plots ( one for each group/bar)
                point_hover_labels = [f'{i}' for i in active_labels] # point_hover_labels will be added as tooltip annotations to the datapoints
                assert len(x_values) == len(y_values) and len(x_values) == len(point_hover_labels), f"len(x_values): {len(x_values)}, len(y_values): {len(y_values)}, len(point_hover_labels): {len(point_hover_labels)}"
                # add static tiny labels beside each point
                if (enable_tiny_point_labels or enabled_point_connection_lines):
                    if enable_tiny_point_labels:
                        temp_annotation_labels_list = []
                    for i, (x, y, label) in enumerate(zip(x_values, y_values, point_hover_labels)):
                        # print(f'{i}, (x, y, label): ({x}, {y}, {label})')
                        if enable_tiny_point_labels:
                            annotation_item = ax.annotate(label, (x, y), textcoords="offset points", xytext=(2,2), ha='left', va='bottom', fontsize=8) # , color=rect.get_facecolor()
                            temp_annotation_labels_list.append(annotation_item)
                        if enabled_point_connection_lines:
                            if str(label) not in a_fig_container['plot_objects']['point_connection_lines']:
                                a_fig_container['plot_objects']['point_connection_lines'][str(label)] = [] ## create
                            a_fig_container['plot_objects']['point_connection_lines'][str(label)].append((x, y))
    
                    if enable_tiny_point_labels:
                        a_fig_container['plot_objects']['tiny_annotation_labels'].append(temp_annotation_labels_list)
                    
                # add hover labels:
                # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
                # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib/21654635#21654635
                # add hover labels using mplcursors
                if enable_hover_labels:
                    hover_label_obj = mplcursors.cursor(a_scatter_plot, hover=True).connect("add", lambda sel: sel.annotation.set_text(point_hover_labels[sel.index]))
                    a_fig_container['plot_objects']['hover_label_objects'].append(hover_label_obj)


        if enabled_point_connection_lines:
            for a_label, a_point_list in a_fig_container['plot_objects']['point_connection_lines'].items():
                if debug_print:
                    print(f'a_label: {a_label}, a_point_list: {a_point_list}')
                assert len(a_point_list) == 2,f"len(a_point_list) should be 2, but it is {len(a_point_list)}"
                (start_x, start_y), (end_x, end_y) = a_point_list
                a_fig_container['plot_objects']['long_to_short_arrow'][a_label] = ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y), arrowprops=dict(**arrowprops_kwargs, color='black'), label=str(label))
                # plt.plot(x,y,zorder=1) 

        return a_fig_container

# ==================================================================================================================== #
# 2023-06-26 Figure 3                                                                                                  #
""" Firing rate index, Long|Short firing rate figures 

        Renders 3 Subfigures:
            a) Shows the firing rate index between the long and short track computed for two different periods: the laps along the x-axis and the replays along the y-axis.
            b) The ratio of lap to replay firing rate on the long track.
            c) The ratio of lap to replay firing rate on the short track
"""
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['figure_3', 'paper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-27 03:03', related_items=[])
def PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=False, save_figure=True):
    """ Firing rate index, Long|Short firing rate figures 

        Renders 3 Subfigures:
            a) Shows the firing rate index between the long and short track computed for two different periods: the laps along the x-axis and the replays along the y-axis.
            b) The ratio of lap to replay firing rate on the long track.
            c) The ratio of lap to replay firing rate on the short track.
    
    from PendingNotebookCode import PAPER_FIGURE_figure_3
    _out, _out2 = PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=False, save_figure=True)
    
    """
    _out = curr_active_pipeline.display('_display_short_long_firing_rate_index_comparison', curr_active_pipeline.get_session_context(), defer_render=defer_render, save_figure=save_figure)
    _out2 = curr_active_pipeline.display('_display_long_and_short_firing_rate_replays_v_laps', curr_active_pipeline.get_session_context(), defer_render=defer_render, save_figure=save_figure)

    return (_out, _out2)



# ==================================================================================================================== #
# Statistical Tests                                                                                                    #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['stats', 'binomial', 'FRI'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-10 16:55', related_items=[])
def pho_stats_perform_diagonal_line_binomial_test(long_short_fr_indicies_analysis_table, x_col_name='x_frs_index', y_col_name='y_frs_index', extra_required_columns=['neuron_uid']):
    """ Performs a binomial test to see if the number of entries above/below the y=x diagnoal were greater than would be expected by chance.

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_perform_diagonal_line_binomial_test

        binom_test_chance_result = pho_stats_perform_diagonal_line_binomial_test(long_short_fr_indicies_analysis_table)
        binom_test_chance_result

    """
    # Drop column: 'index'
    # long_short_fr_indicies_analysis_table = long_short_fr_indicies_analysis_table.drop(columns=['index'])
    # Drop rows with missing data in columns: 'x_frs_index', 'y_frs_index', 'neuron_uid'
    # extra_required_columns = ['neuron_uid']
    long_short_fr_indicies_analysis_table = long_short_fr_indicies_analysis_table.dropna(subset=[x_col_name, y_col_name, *extra_required_columns])

    ## Find the values above/below the main y=x diagonal:
    x_minus_y_diff = (long_short_fr_indicies_analysis_table[x_col_name] - long_short_fr_indicies_analysis_table[y_col_name])
    assert np.sum(np.logical_not(np.isfinite(x_minus_y_diff))) == 0, f"ERROR: contains {np.sum(np.logical_not(np.isfinite(x_minus_y_diff)))} non-finite values"
    ## Find the counts for each:
    n_total = len(x_minus_y_diff) # 856
    n_below_diagonal = np.sum((0.0 > x_minus_y_diff)) # 365
    n_above_diagonal = np.sum((0.0 < x_minus_y_diff)) # 487
    n_exact_on_diagonal = np.sum((0.0 == x_minus_y_diff))
    print(f'n_total: {n_total}, n_above_diagonal: {n_above_diagonal}, n_exact_on_diagonal: {n_exact_on_diagonal}, n_below_diagonal: {n_below_diagonal}')
    assert (n_above_diagonal + n_below_diagonal + n_exact_on_diagonal) == n_total, f"they don't add up!" 
    binom_test_chance_result = stats.binomtest(n_above_diagonal, n=n_total, p=0.5) # p=0.5 random assignment on each trial, n=n_total trials
    return binom_test_chance_result



# Define a simple attrs class that's better than a namedtuple and unpackable:
LinearRegressionOutput = make_class("LinearRegressionOutput", {k:field() for k in ("reg_x", "reg_y", "slope", "intercept", "regression_line", "model", "r_value")}, bases=(UnpackableMixin, object,))


def _linear_regression_output_plot(self, ax, r_text_kwargs=None, **kwargs):
    """Plot regression line and R-value label. Line kwargs default to red/0.7/0.5; R text defaults to black/8pt. Pass r_text_kwargs to override the R label."""
    line_kwargs = dict(color='red', alpha=0.7, lw=0.5) | kwargs
    _r_text_kwargs = dict(fontsize=8, color='black', alpha=0.9, verticalalignment='top') | (r_text_kwargs or {})
    line_artists = ax.plot(self.reg_x, self.regression_line, label=f'Regression Line\ny={self.slope:.2f}x + {self.intercept:.2f}', **line_kwargs)
    ax.text(0.02, 0.98, f'R = {self.r_value:.3f}', transform=ax.transAxes, **_r_text_kwargs)
    return line_artists[0]


LinearRegressionOutput.plot = _linear_regression_output_plot

@function_attributes(short_name=None, tags=['stats', 'regression'], input_requires=[], output_provides=[], uses=['LinearRegressionOutput'], used_by=[], creation_date='2025-07-04 03:11', related_items=[])
def pho_stats_linear_regression(reg_x: NDArray, reg_y: NDArray) -> "LinearRegressionOutput":
    """ 
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_linear_regression, LinearRegressionOutput
    
    _lin_reg = pho_stats_linear_regression(reg_x, reg_y)
    _lin_reg
    """
    import statsmodels.api as sm
    
    if not isinstance(reg_x, NDArray):
        reg_x = np.array(reg_x)
    if not isinstance(reg_y, NDArray):
        reg_y = np.array(reg_y)
        
    # Fit linear regression using numpy.polyfit
    reg_slope, reg_intercept = np.polyfit(reg_x, reg_y, 1)
    # Compute regression line
    regression_line = reg_slope * reg_x + reg_intercept
    

    # 2025-07-18 08:18 Added to find p-value: https://www.statology.org/statsmodels-linear-regression-p-value/
    #add constant to predictor variables
    x_lin_model = deepcopy(reg_x)
    y_lin_model = deepcopy(reg_y)
    x_lin_model = sm.add_constant(x_lin_model)
    #fit linear regression model
    model = sm.OLS(y_lin_model, x_lin_model).fit()
    #view model summary
    # print(model.summary())
    # #extract p-values for all predictor variables
    # for x in np.arange(2):
    #     print(model.pvalues[x])

    # Calculate R value (correlation coefficient)
    r_value = np.corrcoef(reg_x, reg_y)[0, 1]
    
    return LinearRegressionOutput(**{'reg_x': reg_x, 'reg_y': reg_y, 'slope': reg_slope, 'intercept': reg_intercept, 'regression_line': regression_line, 'model': model, 'r_value': r_value})


def pho_stats_paired_t_test(values1, values2):
    """ Paired (Dependent) T-Test of means

    degrees of freedom (dof): n -1

    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_paired_t_test


    """
    assert len(values1) == len(values2), f"this is supposed to be a paired t-test so the number of samples in values1 should equal values2!! but {np.shape(values1)} and {np.shape(values2)}"
    # # Manual Calculation:
    # n_samples = len(values1) # sample_size (number of neurons)
    # out_numerator = (np.mean(values1) - np.mean(values2))
    # out_denom = np.std(values1 - values2) / np.sqrt(n_samples)
    # T = out_numerator/out_denom
    T_result = stats.ttest_rel(values1, values2)
    # T_value = T_result.statistic
    return T_result


@function_attributes(short_name=None, tags=['stats', 'bar'], input_requires=[], output_provides=[], uses=['pho_stats_paired_t_test'], used_by=[], creation_date='2023-10-10 16:54', related_items=[])
def pho_stats_bar_graph_t_tests(across_session_inst_fr_computation):
    """ performs the statistical tests for the bar-graphs 

    Usage:
    
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_bar_graph_t_tests

        LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result = pho_stats_bar_graph_t_tests(across_session_inst_fr_computation)

        max_laps_y = np.nanmax([v.values for v in across_session_inst_fr_computation.Fig2_Laps_FR])
        max_PBEs_y = np.nanmax([v.values for v in across_session_inst_fr_computation.Fig2_Replay_FR])

    """
    ## Laps Bar Graph Statistics:
    LxC_Laps_T_result = pho_stats_paired_t_test(across_session_inst_fr_computation.Fig2_Laps_FR[0].values, across_session_inst_fr_computation.Fig2_Laps_FR[1].values)
    SxC_Laps_T_result = pho_stats_paired_t_test(across_session_inst_fr_computation.Fig2_Laps_FR[2].values, across_session_inst_fr_computation.Fig2_Laps_FR[3].values)
    print(f'LxC_Laps_T_result: {LxC_Laps_T_result}') # LxC_Laps_T_result: TtestResult(statistic=13.925882964152734, pvalue=2.3158087721181958e-10, df=16)
    print(f'SxC_Laps_T_result: {SxC_Laps_T_result}') # SxC_Laps_T_result: TtestResult(statistic=-12.402705609994197, pvalue=8.279901167065766e-08, df=11)

    ## Replay Bar Graph Statistics
    LxC_Replay_T_result = pho_stats_paired_t_test(across_session_inst_fr_computation.Fig2_Replay_FR[0].values, across_session_inst_fr_computation.Fig2_Replay_FR[1].values)
    SxC_Replay_T_result = pho_stats_paired_t_test(across_session_inst_fr_computation.Fig2_Replay_FR[2].values, across_session_inst_fr_computation.Fig2_Replay_FR[3].values)
    print(f'LxC_Replay_T_result: {LxC_Replay_T_result}') # LxC_Replay_T_result: TtestResult(statistic=-0.44250837706672874, pvalue=0.6640450004297094, df=16) # LxC_Replay_T_result is NOT p<0.05 significant (pvalue=0.6640450004297094)
    print(f'SxC_Replay_T_result: {SxC_Replay_T_result}') # SxC_Replay_T_result: TtestResult(statistic=-3.6555017036343607, pvalue=0.0037841961453242896, df=11) # SxC_Replay_T_result IS p<0.05 significant (pvalue=0.0037841961453242896)
    
    return LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result


@function_attributes(short_name=None, tags=['stats', 'bar', 'format'], input_requires=[], output_provides=[], uses=[], used_by=['PaperFigureTwo.display'], creation_date='2026-03-26 08:20', related_items=[])
def format_bar_graph_t_test_pvalues_str(LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result, include_num_cells: bool = True) -> str:
    """ Format Fig2 bar-graph paired t-test p-values for notebook/publication paste.

    Returns e.g. '(Laps: LdC p=1.25×10^-13,  SdC p=8.65×10^-9; PBEs: LdC p=0.0618, SdC p=0.000635)'
        or '25 LdC, 17 SdC cells; (Laps: LdC p=1.25×10^-13,  SdC p=8.65×10^-9; PBEs: LdC p=0.0618, SdC p=0.000635)' if (include_num_cells == True)

    Very small p-values use MS Word UnicodeMath-friendly scientific notation (`a×10^b`),
    which converts to a proper equation when pasted into Insert → Equation.

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_bar_graph_t_tests, format_bar_graph_t_test_pvalues_str
        LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result = pho_stats_bar_graph_t_tests(across_session_inst_fr_computation)
        print(format_bar_graph_t_test_pvalues_str(LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result))
    """
    def _fmt_p_scientific_word(p: float) -> str:
        """e.g. 1.25e-13 -> '1.25×10^-13' (Word equation linear / UnicodeMath)."""
        mantissa_str, exp_str = f'{p:.2e}'.split('e')
        return f'{mantissa_str}×10^{int(exp_str)}'

    def _fmt_p(p: float) -> str:
        if p < 1e-4:
            return _fmt_p_scientific_word(p)
        if p < 1e-3:
            return f'{p:.6f}'.rstrip('0').rstrip('.')
        return f'{p:.3g}'

    _out_str: str = (f'(Laps: LdC p={_fmt_p(float(LxC_Laps_T_result.pvalue))},  SdC p={_fmt_p(float(SxC_Laps_T_result.pvalue))}; '
            f'PBEs: LdC p={_fmt_p(float(LxC_Replay_T_result.pvalue))}, SdC p={_fmt_p(float(SxC_Replay_T_result.pvalue))})')

    if include_num_cells:
        ## creates output string like '25 LdC, 17 SdC cells; (Laps: LdC p=1.25×10^-13,  SdC p=8.65×10^-9; PBEs: LdC p=0.0618, SdC p=0.000635)'
        n_LdC, n_SdC = [v.df for v in (LxC_Laps_T_result, SxC_Laps_T_result)]
        num_cells_str: str = f'{n_LdC} LdC, {n_SdC} SdC cells'
        _out_str = '; '.join([num_cells_str, _out_str])

    return _out_str

@function_attributes(short_name=None, tags=['epoch', 'pbe'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-11 14:13', related_items=[])
def build_derived_epochs_dicts(owning_pipeline_reference):
    """ builds three dictionaries containing all of the Epoch objects for {global, long, short}. Contains epochs that don't normally exist on the session object
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import build_derived_epochs_dicts
        all_epochs, long_only_all_epochs, short_only_all_epochs = build_derived_epochs_dicts(curr_active_pipeline)
    
    """	
    
    long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
    long_epoch_obj, short_epoch_obj = [Epoch(owning_pipeline_reference.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name)) for an_epoch_name in [long_epoch_name, short_epoch_name]]

    ## Do all epoch computations on the original session. When done, should have: ['pbe', 'replay', 'laps', 'non_running_periods', 'non_replay_periods'
    # dictionary of all epoch objects across the session
    all_epochs: Dict[str,Epoch] = dict(
        laps = owning_pipeline_reference.sess.laps.as_epoch_obj(),
        pbe = owning_pipeline_reference.sess.pbe,
        replay = Epoch(owning_pipeline_reference.sess.replay),
        non_running_periods = Epoch.from_PortionInterval(owning_pipeline_reference.sess.laps.as_epoch_obj().to_PortionInterval().complement()),
        non_replay_periods = Epoch(Epoch.from_PortionInterval(owning_pipeline_reference.sess.replay.epochs.to_PortionInterval().complement()).time_slice(t_start=long_epoch_obj.t_start, t_stop=short_epoch_obj.t_stop).to_dataframe()[:-1]),  #[:-1] # any period except the replay ones, drop the infinite last entry
    )

    # Split into long/short only periods:
    long_only_all_epochs: Dict[str,Epoch] = {k:v.time_slice(t_start=long_epoch_obj.t_start, t_stop=long_epoch_obj.t_stop) for k,v in all_epochs.items()}
    short_only_all_epochs: Dict[str,Epoch] = {k:v.time_slice(t_start=short_epoch_obj.t_start, t_stop=short_epoch_obj.t_stop) for k,v in all_epochs.items()}

    return all_epochs, long_only_all_epochs, short_only_all_epochs


@function_attributes(short_name=None, tags=['inst_fr', 'spike_rate_Trends'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-11 14:13', related_items=[])
def add_extra_spike_rate_trends(curr_active_pipeline) -> InstantaneousSpikeRateGroupsComputation:
    """ independent of all other FR computations. Builds inst spike rate groups for the PBEs. 
    
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import add_extra_spike_rate_trends


    """
    temp = InstantaneousSpikeRateGroupsComputation()
    temp.active_identifying_session_ctx = curr_active_pipeline.sess.get_context()

    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # only uses global_session

    ## Add additional Epochs:
    all_epochs, long_only_all_epochs, short_only_all_epochs = build_derived_epochs_dicts(curr_active_pipeline)

    ## Manual User-annotation mode:
    annotation_man: UserAnnotationsManager = UserAnnotationsManager()
    session_cell_exclusivity: SessionCellExclusivityRecord = annotation_man.annotations[temp.active_identifying_session_ctx].get('session_cell_exclusivity', None)
    if session_cell_exclusivity is not None:
        print(f'setting LxC_aclus/SxC_aclus from user annotation.')
        temp.LxC_aclus = session_cell_exclusivity.LxC
        temp.SxC_aclus = session_cell_exclusivity.SxC
    else:
        print(f'WARN: no user annotation for session_cell_exclusivity')

    are_LxC_empty: bool = (len(temp.LxC_aclus) == 0)
    are_SxC_empty: bool = (len(temp.SxC_aclus) == 0)

    temp.LxC_PBEsDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_only_all_epochs['pbe'], included_neuron_ids=temp.LxC_aclus, instantaneous_time_bin_size_seconds=temp.instantaneous_time_bin_size_seconds)
    temp.LxC_PBEsDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_only_all_epochs['pbe'], included_neuron_ids=temp.LxC_aclus, instantaneous_time_bin_size_seconds=temp.instantaneous_time_bin_size_seconds)
    temp.SxC_PBEsDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_only_all_epochs['pbe'], included_neuron_ids=temp.SxC_aclus, instantaneous_time_bin_size_seconds=temp.instantaneous_time_bin_size_seconds)
    temp.SxC_PBEsDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_only_all_epochs['pbe'], included_neuron_ids=temp.SxC_aclus, instantaneous_time_bin_size_seconds=temp.instantaneous_time_bin_size_seconds)

    # Note that in general LxC and SxC might have differing numbers of cells.
    if (are_LxC_empty or are_SxC_empty):
        temp.Fig2_PBEs_FR: list[SingleBarResult] = []
        for v in (temp.LxC_PBEsDeltaMinus, temp.LxC_PBEsDeltaPlus, temp.SxC_PBEsDeltaMinus, temp.SxC_PBEsDeltaPlus):
            if v is not None:
                temp.Fig2_PBEs_FR.append(SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, temp.LxC_aclus, temp.SxC_aclus, None, None))
            else:
                temp.Fig2_PBEs_FR.append(SingleBarResult(None, None, np.array([], dtype=float), temp.LxC_aclus, temp.SxC_aclus, None, None))
    else:
        temp.Fig2_PBEs_FR: list[SingleBarResult] = [SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, temp.LxC_aclus, temp.SxC_aclus, None, None) for v in (temp.LxC_PBEsDeltaMinus, temp.LxC_PBEsDeltaPlus, temp.SxC_PBEsDeltaMinus, temp.SxC_PBEsDeltaPlus)]

    return temp



# ==================================================================================================================== #
# 2024-09-05 Firing Rate Regression Lines                                                                              #
# ==================================================================================================================== #
from sklearn.linear_model import LinearRegression

class LongShortFRRegression:
    """
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import LongShortFRRegression
    import matplotlib
    from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update

    _bak = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')

    (model_L, model_S), (fig, ax) = LongShortFRRegression.compute_laps_replay_regression(neuron_replay_stats_table)
    (model_L, model_S), (fig, ax) = LongShortFRRegression.compute_laps_replay_opposite_regression(neuron_replay_stats_table)

    """
    @classmethod
    def compute_regression_line(cls, X, y, name: str, ax=None):
        from neuropy.utils.matplotlib_helpers import build_or_reuse_figure

        # Fit the regression model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Plot the results
        if ax is None:
            # fig = build_or_reuse_figure(fignum=kwargs.pop('fignum', None), fig=kwargs.pop('fig', None), fig_idx=kwargs.pop('fig_idx', 0), figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True) # , clear=True
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
            
        # plt.figure(clear=True)
        ax.scatter(X, y, label=f'Data {name}', alpha=0.7) # , color='blue'
        ax.plot(X, y_pred, label=f'Regression {name}', alpha=0.7) # , color='red'
        plt.xlabel('Non-Replay Mean')
        plt.ylabel('Replay Mean')
        plt.legend()
        # ax.show()

        # Retrieve the slope
        slope = model.coef_[0]
        
        # Print the slope
        print("Slope:", slope)
        
        return model, (fig, ax)

    @classmethod
    def compute_laps_replay_regression(cls, neuron_replay_stats_table: pd.DataFrame, long_replay_mean_col: str = 'long_replay_mean', long_non_replay_mean_col: str = 'long_non_replay_mean', short_replay_mean_col: str = 'short_replay_mean', short_non_replay_mean_col: str = 'short_non_replay_mean') -> tuple:
        """ computes the laps vs. replay 
        
        """        
        # Specify columns to convert to float
        columns_to_convert = [long_replay_mean_col, long_non_replay_mean_col, short_replay_mean_col, short_non_replay_mean_col]

        # Convert specified columns to float
        neuron_replay_stats_table[columns_to_convert] = neuron_replay_stats_table[columns_to_convert].astype(float)

        # Long Track Replay|Laps FR Figure
        cleaned_df = neuron_replay_stats_table.dropna(subset=[long_replay_mean_col, long_non_replay_mean_col], inplace=False, how='any')
        np.sum(cleaned_df[[long_replay_mean_col, long_non_replay_mean_col, short_replay_mean_col, short_non_replay_mean_col]].isna())

        X = cleaned_df[[long_non_replay_mean_col]].to_numpy().astype(float)
        y = cleaned_df[long_replay_mean_col].to_numpy().astype(float)

        assert np.sum(np.isnan(X)) == 0
        assert np.sum(np.isnan(y)) == 0

        model_L, (fig, ax) = cls.compute_regression_line(X, y, name='long')

        # Short Track Replay|Laps FR Figure
        cleaned_df = neuron_replay_stats_table.dropna(subset=[short_replay_mean_col, short_non_replay_mean_col], inplace=False, how='any')[[short_replay_mean_col, short_non_replay_mean_col]]
        cleaned_df[[short_replay_mean_col, short_non_replay_mean_col]] = cleaned_df[[short_replay_mean_col, short_non_replay_mean_col]].astype(float)
        cleaned_df = cleaned_df.dropna(subset=[short_replay_mean_col, short_non_replay_mean_col], inplace=False, how='any')

        X = cleaned_df[[short_non_replay_mean_col]].to_numpy().astype(float)
        y = cleaned_df[short_replay_mean_col].to_numpy().astype(float)

        assert np.sum(np.isnan(X)) == 0
        assert np.sum(np.isnan(y)) == 0

        model_S, (fig, ax) = cls.compute_regression_line(X, y, name='short', ax=ax)
        ax.set_aspect(1.)
        plt.title('Non-PBE/PBE Firing Rate Gain Factor Regression for Long+Short')

        slopes_string = '\n'.join([f"{a_name}: {model.coef_[0]:.4f}" for a_name, model in dict(zip(('Long', 'Short'), (model_L, model_S))).items()])
        
        ax.text(0.05, 0.95, f'Slope:\n{slopes_string}', transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        
        fig.show()

        return (model_L, model_S), (fig, ax)




    @classmethod
    def compute_laps_replay_opposite_regression(cls, neuron_replay_stats_table):
        ## INPUTS: neuron_replay_stats_table

        ## Long Track Replay|Laps FR Figure
        # Specify columns to convert to float
        columns_to_convert = ['long_replay_mean', 'long_non_replay_mean', 'short_replay_mean', 'short_non_replay_mean']

        # Convert specified columns to float
        neuron_replay_stats_table[columns_to_convert] = neuron_replay_stats_table[columns_to_convert].astype(float)



        # first oness ________________________________________________________________________________________________________ #
        cleaned_df = neuron_replay_stats_table.dropna(subset=['long_replay_mean', 'short_non_replay_mean'], inplace=False, how='any')
        # Drop rows with NaN values
        # cleaned_df = neuron_replay_stats_df.dropna(subset=['long_replay_mean', 'long_non_replay_mean'])
        # np.sum(cleaned_df[['long_replay_mean', 'long_non_replay_mean', 'short_replay_mean', 'short_non_replay_mean']].isna())

        # Extract data
        # X = neuron_replay_stats_df['long_replay_mean'].values.astype(float)
        # X = cleaned_df[['long_replay_mean']].values.astype(float)
        # y = cleaned_df['long_non_replay_mean'].values.astype(float)

        X = cleaned_df[['long_replay_mean']].to_numpy().astype(float)
        y = cleaned_df['short_non_replay_mean'].to_numpy().astype(float)

        # X.dtype
        assert np.sum(np.isnan(X)) == 0
        assert np.sum(np.isnan(y)) == 0

        # X.shape
        # y.shape
        
        model_L, (fig, ax) = cls.compute_regression_line(X, y, name='long') # Slope: 0.3690231851911259

        ## Short Track Replay|Laps FR Figure
        # cleaned_df.dtypes
        cleaned_df = neuron_replay_stats_table.dropna(subset=['short_replay_mean', 'long_non_replay_mean'], inplace=False, how='any')[['short_replay_mean', 'long_non_replay_mean']]
        cleaned_df[['short_replay_mean', 'long_non_replay_mean']] = cleaned_df[['short_replay_mean', 'long_non_replay_mean']].astype(float)
        # cleaned_df.dtypes
        cleaned_df = cleaned_df.dropna(subset=['short_replay_mean', 'long_non_replay_mean'], inplace=False, how='any')
        # cleaned_df
        # cleaned_df.dtypes

        X = cleaned_df[['short_replay_mean']].to_numpy().astype(float)
        y = cleaned_df['long_non_replay_mean'].to_numpy().astype(float)

        assert np.sum(np.isnan(X)) == 0
        assert np.sum(np.isnan(y)) == 0
        # fig_S, ax_S, active_display_context_S = _plot_single_track_firing_rate_compare(x_frs, y_frs, active_context=final_context.adding_context_if_missing(filter_name='short'), **common_scatter_kwargs)

        model_S, (fig, ax) = cls.compute_regression_line(X, y, name='short', ax=ax) # Slope: 0.3690231851911259
        # Set aspect of the main Axes.
        ax.set_aspect(1.)
        plt.title('Long/Short Opposite Gain Factor Regression')
        
            # Retrieve the slope
        slopes_string = '\n'.join([f"{a_name}: {model.coef_[0]:.4f}" for a_name, model in dict(zip(('Long', 'Short'), (model_L, model_S))).items()])
        
            # Add slope text to the plot
        ax.text(0.05, 0.95, f'Slope:\n{slopes_string}', transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        
        
        fig.show()
        # Slope: 0.3374416368231605

        return (model_L, model_S), (fig, ax)









# ==================================================================================================================== #
# MAIN RUN FUNCTION TO GENERATE ALL FIGURES                                                                            #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['export', 'output', 'images', 'final', 'figure', 'complete'], input_requires=[], output_provides=[], uses=['PAPER_FIGURE_figure_1_full', 'PaperFigureTwo', 'PAPER_FIGURE_figure_3', 'fig_remapping_cells', 'batch_perform_all_plots'], used_by=[], creation_date='2024-08-29 18:07', related_items=[])
def main_complete_figure_generations(curr_active_pipeline, enable_default_neptune_plots:bool=True, save_figures_only:bool=False, save_figure=True):
    """ main run function to generate all figures
    
        Usage 1:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import main_complete_figure_generations
            main_complete_figure_generations(curr_active_pipeline)

        Usage 2:        
            from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.BatchCompletionHandler import BatchSessionCompletionHandler
            from neuropy.utils.matplotlib_helpers import matplotlib_file_only
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import main_complete_figure_generations

            # self.try_complete_figure_generation_to_file(curr_active_pipeline, enable_default_neptune_plots=self.should_generate_all_plots)
            fail_on_exception: bool = False
            enable_default_neptune_plots = False
            completed_good: bool = False

            try:
                ## To file only:
                with matplotlib_file_only():
                    # Perform non-interactive Matplotlib operations with 'AGG' backend
                    # neptuner = batch_perform_all_plots(curr_active_pipeline, enable_neptune=True, neptuner=None)
                    main_complete_figure_generations(curr_active_pipeline, enable_default_neptune_plots=enable_default_neptune_plots, save_figures_only=True, save_figure=True)

                curr_active_pipeline.clear_display_outputs()
                curr_active_pipeline.clear_registered_output_files()
                completed_good = True # completed successfully (without raising an error at least).

            except Exception as e:
                completed_good =  False
                raise

            ## OUTPUTS: completed_good
            print(f'completed_good: {completed_good}')

        
    
    """

    if save_figures_only:
        defer_show = True
    else:
        defer_show = False
        
    curr_active_pipeline.reload_default_display_functions()

    # ==================================================================================================================== #
    # Perform `batch_perform_all_plots`                                                                                    #
    # ==================================================================================================================== #
    if enable_default_neptune_plots:
        neptuner = batch_perform_all_plots(curr_active_pipeline, enable_neptune=True)

    # ==================================================================================================================== #
    # Extract Relevent Specific Data Needed for Figure Display                                                             #
    # ==================================================================================================================== #
    ## long_short_decoding_analyses:
    curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
    ## Extract variables from results object:
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global

    # (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_short_long_constrained_decoders(curr_active_pipeline, recalculate_anyway=True)
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_computation_config, short_computation_config, global_computation_config = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_pf1D, short_pf1D, global_pf1D = long_results.pf1D, short_results.pf1D, global_results.pf1D
    long_pf2D, short_pf2D, global_pf2D = long_results.pf2D, short_results.pf2D, global_results.pf2D
    decoding_time_bin_size = long_one_step_decoder_1D.time_bin_size # 1.0/30.0 # 0.03333333333333333

    ## Get global 'long_short_post_decoding' results:
    curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
    expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
    rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
    Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result.Flat_epoch_time_bins_mean, expected_v_observed_result.Flat_decoder_time_bin_centers, expected_v_observed_result.num_neurons, expected_v_observed_result.num_timebins_in_epoch, expected_v_observed_result.num_total_flat_timebins, expected_v_observed_result.is_short_track_epoch, expected_v_observed_result.is_long_track_epoch, expected_v_observed_result.short_short_diff, expected_v_observed_result.long_long_diff
    jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis    
    (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)


    # ==================================================================================================================== #
    # Figure 1) pf1D Ratemaps, Active set, etc                                                                             #
    # ==================================================================================================================== #
    _out_fig_1 = PAPER_FIGURE_figure_1_full(curr_active_pipeline, defer_show=defer_show, save_figure=save_figure) # did not display the pf1


    # Critical new code: Not used anyhwere
    # ratemap = long_pf1D.ratemap
    # included_unit_neuron_IDs = EITHER_subset.track_exclusive_aclus
    # included_unit_neuron_IDs = EITHER_subset.get_refined_track_exclusive_aclus() # 2023-09-28 - "Refined"
    # included_aclus_sort_indicies = deepcopy(new_all_aclus_sort_indicies[np.isin(new_all_aclus_sort_indicies, included_unit_neuron_IDs)])
    # # rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=new_all_aclus_sort_indicies.copy())
    # rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=included_aclus_sort_indicies.copy())

    # ==================================================================================================================== #
    # Figure 2) Firing Rate Bar Graphs                                                                                     #
    # ==================================================================================================================== #


    # Instantaneous versions:

    _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.005) # 10ms
    _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
    _out_fig_2.display(defer_show=defer_show, save_figure=save_figure, active_context=curr_active_pipeline.sess.get_context())


    # ==================================================================================================================== #
    # Figure 3) `PAPER_FIGURE_figure_3`: Firing Rate Index and Long/Short Firing Rate Replays v. Laps                      #
    # ==================================================================================================================== #
    _out_fig_3_a, _out_fig_3_b = PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=defer_show, save_figure=save_figure)

    # # ==================================================================================================================== #
    # # HELPERS: Interactive Components                                                                                      #
    # # ==================================================================================================================== #
    # from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_2D
    # # fig, ax, rect_selector, set_extents, reset_extents = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True)

    # grid_bin_bounds = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True, should_apply_updates_to_pipeline=False)
    # print(f'grid_bin_bounds: {grid_bin_bounds}')
    # print(f"Add this to `specific_session_override_dict`:\n\n{curr_active_pipeline.get_session_context().get_initialization_code_string()}:dict(grid_bin_bounds=({(grid_bin_bounds[0], grid_bin_bounds[1]), (grid_bin_bounds[2], grid_bin_bounds[3])})),\n")

    # # ==================================================================================================================== #
    # # DEBUGGING:                                                                                                           #
    # # ==================================================================================================================== #
    # ### Testing `plot_kourosh_activity_style_figure` for debugging:
    # from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import plot_kourosh_activity_style_figure

    # # plot_aclus = EITHER_subset.track_exclusive_aclus.copy()
    # plot_aclus = EITHER_subset.track_exclusive_aclus[new_all_aclus_sort_indicies].copy()
    # _out_A = plot_kourosh_activity_style_figure(long_results_obj, long_session, plot_aclus, unit_sort_order=new_all_aclus_sort_indicies, epoch_idx=13, callout_epoch_IDXs=None, skip_rendering_callouts=False)
    # app, win, plots, plots_data = _out_A


    # ==================================================================================================================== #
    # 2023-09-26 - Presentation Figures:                                                                                   #
    # ==================================================================================================================== #
    
    graphics_output_dict = {}

    try:
        # 2023-09-21 - Plot All
        graphics_output_dict = graphics_output_dict | fig_remapping_cells(curr_active_pipeline)
    except Exception:
        print(f'plotting `fig_remapping_cells(...)` failed. Continuing.') 


    try:
        # 2023-09-21 - Plot All
        graphics_outputs_list = fig_surprise_results(curr_active_pipeline)
    except Exception:
        print(f'plotting `fig_surprise_results(...)` failed. Continuing.')
        

    # Unwrapping:
    # pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics = _out_fig_1
    if not save_figures_only:
        # only in active display mode is there something to return:
        return (_out_fig_1, _out_fig_2, _out_fig_3_a, _out_fig_3_b)
    
    # plots




# ==================================================================================================================== #
# Plotting Helpers 2024-10-29                                                                                          #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['ACTIVE'], input_requires=[], output_provides=[], uses=['plot_pre_scatter_post_matplotlib'], used_by=[], creation_date='2025-08-11 13:36', related_items=[])
def _perform_matplotlib_SINGLE_SERIES_pre_post_scatter(grainularity_desc: str, epochs_df: pd.DataFrame, is_dark_mode: bool=False, legend_groups_to_solo=None, legend_groups_to_hide: Optional[List[str]]=None, time_column: str='t_start', value_column: str='P_Short', context_prefix=None):
    """ plots the stacked histograms for both laps and ripples, with optional scatterplot showing values over time
    
    Aims to replace the plotly version `_perform_plot_pre_post_delta_scatter` by implementing the same figure in MATPLOTLIB for easier export to publication
    
    
    Parameters:
    -----------
    grainularity_desc : str
        Description of the granularity
    laps_df : pd.DataFrame
        DataFrame containing laps data
    ripple_df : pd.DataFrame  
        DataFrame containing ripple data
    is_dark_mode : bool, optional
        Whether to use dark mode styling
    legend_groups_to_solo : optional
        Groups to show exclusively
    legend_groups_to_hide : List[str], optional
        Groups to hide from display
    include_scatterplot : bool, optional
        Whether to include a scatterplot between histograms showing values over time
    time_column : str, optional
        Column name for time values (default: 't_start')
    value_column : str, optional
        Column name for values to plot (default: 'P_Short')
    
    Returns:
    --------
    tuple
        Returns histogram outputs and optionally scatterplot figure
    
    Usage:
    ------
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import _perform_matplotlib_pre_post_scatter
    
    
    """
    from flexitext import flexitext
    from neuropy.utils.matplotlib_helpers import MatplotlibFigureExtractors, FormattedFigureText ## flexitext version
    from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import plot_pre_scatter_post_matplotlib
    
    # if include_scatterplot:
    #     from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import plot_pre_scatter_post_matplotlib
    #     _active_plot_fn = plot_pre_scatter_post_matplotlib
    # else:
    #     from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import plot_histograms_across_sessions, plot_stacked_histograms        
    #     _active_plot_fn = plot_stacked_histograms

    if (legend_groups_to_solo is None) and (legend_groups_to_hide is None):
        ## if neither are provided, hide none (show all)
        legend_groups_to_hide = []

    variable_name: str = value_column
    y_baseline_level: float = 0.5 # for P(short), etc
    y_ylims = (0, 1)

    # y_baseline_level: float = 0.0 # for wcorr, etc
    # y_ylims = (-1, 1)

    if is_dark_mode:
        baseline_kwargs = dict(color=(0.8,0.8,0.8,.75), linewidth=2)
    else:
        baseline_kwargs = dict(color=(0.2,0.2,0.2,.75), linewidth=2)
        
    def _subfn_update_stacked_post_plot(histogram_out):
        """ captures: y_baseline_level, y_ylims """
        # a_fig = histogram_out.figures[0]
        a_fig = histogram_out.fig
        # for k, ax in histogram_out.axes.items():
        for k, ax in histogram_out.ax_dict.items():
            ## this works for scatter as well
            _tmp_line = ax.axhline(y_baseline_level, **baseline_kwargs) # draw baseline line (horizontally)
            ax.set_ylim(*y_ylims)
            
            if a_fig is None:
                a_fig = ax.get_figure() # recover figure from axes when container.fig is unset
                assert a_fig is not None
                histogram_out.fig = a_fig ## assign a_fig

        ## add flexitext text:
        extracted_fig_titles_dict = MatplotlibFigureExtractors.extract_titles(fig=a_fig)
        suptitle: str = extracted_fig_titles_dict.get('suptitle', None) # 'Laps (by-time-bin)|2 Sessions|5 tbin sizes'
        subtitle_string = None

        # Clear the normal text:
        a_fig.suptitle('')
        # for k, ax in a_histogram_out.axes.items():
        # 	ax.set_title('')

        text_formatter = FormattedFigureText.init_from_margins() # top_margin=0.8
        text_formatter.setup_margins(a_fig)
        # active_config = deepcopy(self.config)
        # active_config.float_precision = 1

        # subtitle_string = '\n'.join([f'{active_config.str_for_display(is_2D)}'])
        full_title_str: str = f'<size:22><weight:bold>{suptitle}</></>'
        if (subtitle_string is not None) and (len(subtitle_string) > 0):
            full_title_str += f'\n<size:10>{subtitle_string}</>'

        # header_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, full_title_str, va="bottom", xycoords="figure fraction") # 0.90
        header_text_obj = flexitext(text_formatter.left_margin, 0.89, full_title_str, va="bottom", xycoords="figure fraction")
        return {'header_text_obj': header_text_obj}
            

    common_stacked_hist_kwargs = dict(figsize=(12, 3), column_name=variable_name)
    scatter_kwargs = dict(s=0.25)

    compare_decimal_precision_ndigits: int = 3
    if (legend_groups_to_solo is not None):
        assert legend_groups_to_hide is None, f"cannot provide both legend_groups_to_solo and legend_groups_to_hide"
        legend_groups_to_solo = np.array([round(float(v), ndigits=compare_decimal_precision_ndigits) for v in legend_groups_to_solo]) # convert to float
        epochs_df = deepcopy(epochs_df)
        epochs_df = epochs_df[np.isin(epochs_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_solo)]		

    elif (legend_groups_to_hide is not None):
        legend_groups_to_hide = np.array([round(float(v), ndigits=compare_decimal_precision_ndigits) for v in legend_groups_to_hide]) # convert to float
        epochs_df = deepcopy(epochs_df)
        epochs_df = epochs_df[np.isin(epochs_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_hide, invert=True)]		

    # Add scatterplot if requested
    scatter_fig = None

    # You can use it like this:
    num_unique_sessions: int = epochs_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    num_unique_time_bins: int = epochs_df.time_bin_size.nunique(dropna=True)

    epoch_identifer: str = 'EPOCHS'
    if context_prefix is not None:
        data_type = f'{context_prefix}'
    else:
        data_type = ''
    data_type = f'{data_type}{epoch_identifer} ({grainularity_desc})'

    _epochs_histogram_out = plot_pre_scatter_post_matplotlib(epochs_df, data_type=data_type, session_spec=f'{num_unique_sessions} Sessions', time_bin_duration_str=f"{num_unique_time_bins} tbin sizes", time_bin_column_name=time_column, scatter_kwargs=scatter_kwargs, **common_stacked_hist_kwargs)
    _epochs_flexitext_dict = _subfn_update_stacked_post_plot(_epochs_histogram_out)
    # fig_to_clipboard(_ripple_histogram_out.figures[0], bbox_inches='tight')

    return _epochs_histogram_out




@function_attributes(short_name=None, tags=['matplotlib', 'good', 'stacked-hist', 'scatter', 'ACTIVE', 'publication'], input_requires=[], output_provides=[], uses=['plot_pre_scatter_post_matplotlib'], used_by=[], creation_date='2024-07-29 00:00', related_items=['_perform_plot_pre_post_delta_scatter'])
def _perform_matplotlib_pre_post_scatter(grainularity_desc: str, laps_df: pd.DataFrame, ripple_df: pd.DataFrame, is_dark_mode: bool=False, legend_groups_to_solo=None, legend_groups_to_hide: Optional[List[str]]=None, include_scatterplot: bool=True, time_column: str='t_start', value_column: str='P_Short'):
    """ plots the stacked histograms for both laps and ripples, with optional scatterplot showing values over time
    
    Aims to replace the plotly version `_perform_plot_pre_post_delta_scatter` by implementing the same figure in MATPLOTLIB for easier export to publication
    
    
    Parameters:
    -----------
    grainularity_desc : str
        Description of the granularity
    laps_df : pd.DataFrame
        DataFrame containing laps data
    ripple_df : pd.DataFrame  
        DataFrame containing ripple data
    is_dark_mode : bool, optional
        Whether to use dark mode styling
    legend_groups_to_solo : optional
        Groups to show exclusively
    legend_groups_to_hide : List[str], optional
        Groups to hide from display
    include_scatterplot : bool, optional
        Whether to include a scatterplot between histograms showing values over time
    time_column : str, optional
        Column name for time values (default: 't_start')
    value_column : str, optional
        Column name for values to plot (default: 'P_Short')
    
    Returns:
    --------
    tuple
        Returns histogram outputs and optionally scatterplot figure
    
    Usage:
    ------
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import _perform_matplotlib_pre_post_scatter
    
    
    """
    from flexitext import flexitext
    from neuropy.utils.matplotlib_helpers import MatplotlibFigureExtractors, FormattedFigureText ## flexitext version
    
    if include_scatterplot:
        from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import plot_pre_scatter_post_matplotlib
        _active_plot_fn = plot_pre_scatter_post_matplotlib
    else:
        from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import plot_histograms_across_sessions, plot_stacked_histograms        
        _active_plot_fn = plot_stacked_histograms

    if (legend_groups_to_solo is None) and (legend_groups_to_hide is None):
        ## if neither are provided, hide none (show all)
        legend_groups_to_hide = []

    variable_name: str = value_column
    y_baseline_level: float = 0.5 # for P(short), etc
    y_ylims = (0, 1)

    # y_baseline_level: float = 0.0 # for wcorr, etc
    # y_ylims = (-1, 1)

    if is_dark_mode:
        baseline_kwargs = dict(color=(0.8,0.8,0.8,.75), linewidth=2)
    else:
        baseline_kwargs = dict(color=(0.2,0.2,0.2,.75), linewidth=2)
        
    def _subfn_update_stacked_hist_post_plot(histogram_out):
        """ captures: y_baseline_level, y_ylims """
        # a_fig = histogram_out.figures[0]
        a_fig = histogram_out.fig

        # for k, ax in histogram_out.axes.items():
        for k, ax in histogram_out.ax_dict.items():
            ## this works for scatter as well
            _tmp_line = ax.axhline(y_baseline_level, **baseline_kwargs) # draw baseline line (horizontally)
            ax.set_ylim(*y_ylims)
            
            if a_fig is None:
                a_fig = ax.get_figure() # recover figure from axes when container.fig is unset
                assert a_fig is not None
                histogram_out.fig = a_fig ## assign a_fig
        ## END for k, ax in histogram_out.ax_dict...

        ## add flexitext text:
        extracted_fig_titles_dict = MatplotlibFigureExtractors.extract_titles(fig=a_fig)
        suptitle: str = extracted_fig_titles_dict.get('suptitle', None) # 'Laps (by-time-bin)|2 Sessions|5 tbin sizes'
        subtitle_string = None

        # Clear the normal text:
        a_fig.suptitle('')
        # for k, ax in a_histogram_out.axes.items():
        # 	ax.set_title('')

        text_formatter = FormattedFigureText.init_from_margins() # top_margin=0.8
        text_formatter.setup_margins(a_fig)
        # active_config = deepcopy(self.config)
        # active_config.float_precision = 1

        # subtitle_string = '\n'.join([f'{active_config.str_for_display(is_2D)}'])
        full_title_str: str = f'<size:22><weight:bold>{suptitle}</></>'
        if (subtitle_string is not None) and (len(subtitle_string) > 0):
            full_title_str += f'\n<size:10>{subtitle_string}</>'

        # header_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, full_title_str, va="bottom", xycoords="figure fraction") # 0.90
        header_text_obj = flexitext(text_formatter.left_margin, 0.89, full_title_str, va="bottom", xycoords="figure fraction")
        return {'header_text_obj': header_text_obj}
            


    common_stacked_hist_kwargs = dict(figsize=(12, 3), column_name=variable_name)

    compare_decimal_precision_ndigits: int = 3
    if (legend_groups_to_solo is not None):
        assert legend_groups_to_hide is None, f"cannot provide both legend_groups_to_solo and legend_groups_to_hide"
        legend_groups_to_solo = np.array([round(float(v), ndigits=compare_decimal_precision_ndigits) for v in legend_groups_to_solo]) # convert to float
        laps_df = deepcopy(laps_df)
        ripple_df = deepcopy(ripple_df)
        # laps_df = laps_df[np.isin(laps_df.time_bin_size.astype(float), legend_groups_to_hide)]
        # ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float), legend_groups_to_hide)]
        laps_df = laps_df[np.isin(laps_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_solo)]
        ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_solo)]		

    elif (legend_groups_to_hide is not None):
        legend_groups_to_hide = np.array([round(float(v), ndigits=compare_decimal_precision_ndigits) for v in legend_groups_to_hide]) # convert to float
        laps_df = deepcopy(laps_df)
        ripple_df = deepcopy(ripple_df)
        # laps_df = laps_df[np.isin(laps_df.time_bin_size.astype(float), legend_groups_to_hide)]
        # ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float), legend_groups_to_hide)]
        laps_df = laps_df[np.isin(laps_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_hide, invert=True)]
        ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_hide, invert=True)]		
        # laps_df = laps_df[np.isclose(laps_df.time_bin_size.astype(float), legend_groups_to_hide)]
        # ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float), legend_groups_to_hide)]
        

    # Add scatterplot if requested
    scatter_fig = None

    # You can use it like this:
    num_unique_sessions: int = laps_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    num_unique_time_bins: int = laps_df.time_bin_size.nunique(dropna=True)
    _laps_histogram_out = _active_plot_fn(laps_df, data_type=f'Laps ({grainularity_desc})', session_spec=f'{num_unique_sessions} Sessions', time_bin_duration_str=f"{num_unique_time_bins} tbin sizes", **common_stacked_hist_kwargs)
    _laps_flexitext_dict = _subfn_update_stacked_hist_post_plot(_laps_histogram_out)
    # fig_to_clipboard(_laps_histogram_out.figures[0], bbox_inches='tight')

    num_unique_sessions: int = ripple_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    num_unique_time_bins: int = ripple_df.time_bin_size.nunique(dropna=True)
    _ripple_histogram_out = _active_plot_fn(ripple_df, data_type=f'PBEs ({grainularity_desc})', session_spec=f'{num_unique_sessions} Sessions', time_bin_duration_str=f"{num_unique_time_bins} tbin sizes", **common_stacked_hist_kwargs)
    _ripple_flexitext_dict = _subfn_update_stacked_hist_post_plot(_ripple_histogram_out)
    # fig_to_clipboard(_ripple_histogram_out.figures[0], bbox_inches='tight')

    # Add scatterplot if requested
    # scatter_fig = None
    # if include_scatterplot:
    #     import matplotlib.pyplot as plt
        
    #     # Combine both dataframes for scatterplot
    #     combined_df = pd.concat([
    #         laps_df.assign(data_type='Laps'),
    #         ripple_df.assign(data_type='PBEs')
    #     ], ignore_index=True)
        
    #     if len(combined_df) > 0 and time_column in combined_df.columns and value_column in combined_df.columns:
    #         # Create figure with 3 subplots: hist1, scatter, hist2
    #         scatter_fig, (ax1, ax_scatter, ax2) = plt.subplots(1, 3, figsize=(15, 5))
            
    #         # Plot scatterplot in middle
    #         colors = {'Laps': 'blue', 'PBEs': 'red'}
    #         for data_type in combined_df['data_type'].unique():
    #             subset = combined_df[combined_df['data_type'] == data_type]
    #             ax_scatter.scatter(subset[time_column], subset[value_column], 
    #                              c=colors.get(data_type, 'gray'), 
    #                              alpha=0.6, label=data_type)
            
    #         ax_scatter.axhline(y_baseline_level, **baseline_kwargs)
    #         ax_scatter.set_ylim(*y_ylims)
    #         ax_scatter.set_xlabel(f'Time ({time_column})')
    #         ax_scatter.set_ylabel(value_column)
    #         ax_scatter.set_title(f'{value_column} vs {time_column}')
    #         ax_scatter.legend()
    #         ax_scatter.grid(True, alpha=0.3)
            
    #         # Hide the side plots for now (they could contain histograms if needed)
    #         ax1.set_visible(False)
    #         ax2.set_visible(False)
            
    #         plt.tight_layout()

    return _laps_histogram_out, _ripple_histogram_out


# Backward compatibility alias
def _perform_dual_hist_plot(grainularity_desc: str, laps_df: pd.DataFrame, ripple_df: pd.DataFrame, is_dark_mode: bool=False, legend_groups_to_solo=None, legend_groups_to_hide: Optional[List[str]]=None):
    """Backward compatibility wrapper for _perform_matplotlib_pre_post_scatter without scatterplot"""
    laps_out, ripple_out = _perform_matplotlib_pre_post_scatter(
        grainularity_desc=grainularity_desc, 
        laps_df=laps_df, 
        ripple_df=ripple_df, 
        is_dark_mode=is_dark_mode, 
        legend_groups_to_solo=legend_groups_to_solo, 
        legend_groups_to_hide=legend_groups_to_hide, 
        include_scatterplot=False
    )
    return laps_out, ripple_out


@function_attributes(short_name=None, tags=['MAIN', 'CRITICAL', 'FINAL', 'plotly', 'scatter', 'histogram', 'publication'], input_requires=[], output_provides=[], uses=['plotly_pre_post_delta_scatter'], used_by=[], creation_date='2024-10-23 20:04', related_items=['_perform_matplotlib_pre_post_scatter'])
def _helper_perform_plot_pre_post_delta_scatter(data_context: IdentifyingContext, concatenated_ripple_df: pd.DataFrame, time_delta_tuple: Tuple[float, float, float], fig_size_kwargs: Dict, save_plotly: Callable, is_dark_mode: bool=False, enable_custom_widget_buttons:bool=True,
                                          extant_figure=None, custom_output_widget=None, legend_groups_to_hide: Optional[List[str]]=None, should_save: bool = True, variable_name = 'P_Short', y_baseline_level: float = 0.5, additional_fig_layout_kwargs: Dict=None, is_publication_ready_figure: bool=False, histogram_bins: int = 11, should_set_hist_same_magnitude_axes: bool = True, **kwargs):
    """ plots the stacked histograms for both laps and ripples
    2025-07-29 - Created ALTERNATIVE <MATPLOTLIB> function: `_perform_matplotlib_pre_post_scatter` for publication to avoid the Plotly exporting headaches
    
    should_set_hist_same_magnitude_axes: normalize height/magnitude of both histograms to the same value


    Usage:
        from functools import partial
        from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import plotly_pre_post_delta_scatter
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import _perform_plot_pre_post_delta_scatter

        is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=False)

        _perform_plot_pre_post_delta_scatter = partial(
            _perform_plot_pre_post_delta_scatter,
            time_delta_tuple=(earliest_delta_aligned_t_start, 0.0, latest_delta_aligned_t_end),
            fig_size_kwargs=fig_size_kwargs,
            is_dark_mode=is_dark_mode,
            save_plotly=save_plotly,
            legend_groups_to_hide=['0.03', '0.044', '0.05'],
        )

        new_fig, new_fig_context, _extras_output_dict, figure_out_paths = _perform_plot_pre_post_delta_scatter(data_context=IdentifyingContext(epochs_name='laps', data_grain='per_epoch', title_prefix="Lap Per Epoch", dataframe_name='laps_df'), concatenated_ripple_df=deepcopy(all_sessions_laps_df))


    """
    from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers
    from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import plotly_helper_save_figures, _helper_build_figure, plotly_pre_post_delta_scatter, add_copy_save_action_buttons
    
    is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=is_dark_mode)
    
    if data_context is None:
        data_context = concatenated_ripple_df.attrs.get('data_context', None)
        assert data_context is not None, f"could not get context from dataframe's df.attrs.data_context either."

    if additional_fig_layout_kwargs is None:
        additional_fig_layout_kwargs = {}        

    num_sessions = 1

    num_events: int = len(concatenated_ripple_df)
    # print(f'num_events: {num_events}')
    data_context.overwriting_context(n_events=num_events) # adds 'n_events' context
    # .025 .03 .044 .05 .058
    # Define the legend groups you want to hide on startup
    if legend_groups_to_hide is None:
        legend_groups_to_hide = []
        
    # legend_groups_to_hide = kwargs.pop('legend_groups_to_hide', [])  # ['0.03', '0.044', '0.05'] '0.025', , '0.058'

    # y_baseline_level: float = 0.5 # for P(short), etc
    # y_baseline_level: float = 0.0 # for wcorr, etc

    # px_scatter_kwargs __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    # px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'color':"is_user_annotated_epoch", 'title': f"'{variable_name}'"} # , 'color': 'time_bin_size', 'range_y': [-1.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size', 'is_user_annotated_epoch':'user_sel'}
    px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'title': f"{data_context.get_description(subset_includelist=['dataframe_name', 'title_prefix', 'num_events'], separator=' - ')} - '{variable_name}'"} # , 'color': 'time_bin_size', 'range_y': [-1.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size', 'is_user_annotated_epoch':'user_sel'}
    if is_publication_ready_figure:
        print(f'is_publication_ready_figure: omitting "color" from px_scatter_kwargs')
        # if 'dummy_column_for_color' not in concatenated_ripple_df.columns:
        #     concatenated_ripple_df['dummy_column_for_color'] = '#000000'            
        # px_scatter_kwargs['color'] = "dummy_column_for_color"
        px_scatter_kwargs['color'] = "time_bin_size"
        px_scatter_kwargs['color_discrete_sequence'] = ['black']
        # px_scatter_kwargs.pop('color')
    else:
        px_scatter_kwargs['color'] = "time_bin_size"
    
    # Controls scatterplot point size
    if not is_publication_ready_figure:
        if 'dummy_column_for_size' not in concatenated_ripple_df.columns:
            concatenated_ripple_df['dummy_column_for_size'] = 2.0
                        
        px_scatter_kwargs['size'] = "dummy_column_for_size"
        px_scatter_kwargs.setdefault('size_max', 5) # don't override tho
        
    else:
        # concatenated_ripple_df['dummy_column_for_size'] = 0.5 ## These points are TINY
        concatenated_ripple_df['dummy_column_for_size'] = 0.9 ## These points are TINY
        px_scatter_kwargs['size'] = "dummy_column_for_size"
        px_scatter_kwargs.setdefault('size_max', 1) # don't override tho
        # px_scatter_kwargs
    
    # px_scatter_kwargs['marker'] = dict(line=dict(width=0))
    # Remove the white border around scatter points by setting line width to 0
    # px_scatter_kwargs['line_width'] = 0  # <---- Correct way for Plotly Express

    # px_scatter_kwargs.update(dict(marginal_x="histogram", marginal_y="rug"))

    # hist_kwargs ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    hist_kwargs = dict()

    if is_publication_ready_figure:
        print(f'is_publication_ready_figure: omitting "color" from px_scatter_kwargs')
        # assert 'dummy_column_for_color' in concatenated_ripple_df.columns, f"it was supposed to be added in the above px_scatter... related block. How did this not happen?"
        # hist_kwargs['color'] = "dummy_column_for_color"
        hist_kwargs['color'] = "time_bin_size"
        hist_kwargs['color_discrete_sequence'] = ['black']
    else:
        hist_kwargs['color'] = "time_bin_size"
        
    
    # hist_kwargs = dict(color="is_user_annotated_epoch") # , histnorm='probability density'
    # hist_kwargs.pop('color')
    
    px_scatter_kwargs = px_scatter_kwargs | kwargs.pop('px_scatter_kwargs', {})
    hist_kwargs = hist_kwargs | kwargs.pop('hist_kwargs', {})

    figure_sup_huge_title_text = kwargs.pop('figure_sup_huge_title_text', None)
    if (figure_sup_huge_title_text is None) and (not is_publication_ready_figure):
        figure_sup_huge_title_text: str = data_context.get_description(subset_includelist=['epochs_name', 'data_grain', 'dataframe_name', 'n_events'], separator=' | ')
        if data_context.has_keys(keys_list=['custom_suffix']):
            custom_suffix_description: str = data_context.get_description(subset_includelist=['custom_suffix'])
            figure_sup_huge_title_text = figure_sup_huge_title_text + f'\n{custom_suffix_description}'
    else:
        print(f'WARN: figure_sup_huge_title_text is provided and not None, figure_sup_huge_title_text: "{figure_sup_huge_title_text}". Overriding.')


    # filter_context = df_filter.filter_context # IdentifyingContext(time_bin_sizes=df_filter.time_bin_size, custom_suffix=df_filter.replay_name)
    
    figure_footer_text = kwargs.pop('figure_sup_huge_title_text', None)
    if (figure_footer_text is None) and (not is_publication_ready_figure):
        figure_footer_text = data_context.get_description(separator='|', subset_excludelist=['time_bin_sizes'])
    else:
        print(f'WARN: figure_footer_text is provided and not None, figure_footer_text: "{figure_footer_text}". Overriding.')

    new_fig, new_fig_context = plotly_pre_post_delta_scatter(data_results_df=concatenated_ripple_df, data_context=data_context,
                                                              extant_figure=extant_figure,
                            out_scatter_fig=None, histogram_bins=histogram_bins,
                            px_scatter_kwargs=px_scatter_kwargs, histogram_variable_name=variable_name, hist_kwargs=hist_kwargs, forced_range_y=None,
                            time_delta_tuple=time_delta_tuple, legend_title_text=None, is_dark_mode=is_dark_mode,
                            # figure_sup_huge_title_text=data_context.get_description(subset_excludelist=['title_prefix'], separator=' | '),
                            figure_sup_huge_title_text=figure_sup_huge_title_text, figure_footer_text=figure_footer_text,
                            is_publication_ready_figure=is_publication_ready_figure,
                            should_set_hist_same_magnitude_axes=should_set_hist_same_magnitude_axes,
                            **kwargs,
    )

    new_fig = new_fig.update_layout(fig_size_kwargs)

    if (additional_fig_layout_kwargs is not None) and len(additional_fig_layout_kwargs) > 0:        
        new_fig = new_fig.update_layout(fig_size_kwargs)

    if legend_groups_to_hide is not None:
        # Collect all unique legend groups you want to hide
        hidden_groups = set(legend_groups_to_hide)
        # Iterate over traces and hide those in the specified legend groups
        for trace in new_fig.data:
            if trace.legendgroup in hidden_groups:
                trace.visible = 'legendonly'
            
    # new_fig_laps.show()

    _extras_output_dict = {}
    if is_dark_mode:
        _extras_output_dict["y_mid_line"] = new_fig.add_hline(y=y_baseline_level, line=dict(color="rgba(0.8,0.8,0.8,.75)", width=2), row='all', col='all')
    else:
        _extras_output_dict["y_mid_line"] = new_fig.add_hline(y=y_baseline_level, line=dict(color="rgba(0.2,0.2,0.2,.75)", width=2), row='all', col='all')


    # fig_to_clipboard(new_fig_laps, **fig_size_kwargs)
    new_fig_context = new_fig_context.adding_context_if_missing(**data_context.get_subset(subset_includelist=['epochs_name', 'data_grain']).to_dict(), num_sessions=num_sessions, plot_type='scatter+hist', comparison='pre-post-delta', variable_name=variable_name)
    if should_save:
        figure_out_paths = save_plotly(a_fig=new_fig, a_fig_context=new_fig_context)
    else:
        figure_out_paths = None
        
    if enable_custom_widget_buttons:
        # _extras_output_dict['out_widget'] = add_copy_save_action_buttons(new_fig)
        # _extras_output_dict['out_container_widget'], _extras_output_dict['out_widget'] = add_copy_save_action_buttons(new_fig, output_widget=custom_output_widget)
        _extras_output_dict['custom_widget_buttons'] = add_copy_save_action_buttons(new_fig)
        
    # Auto-hide legend for publication figures
    if is_publication_ready_figure:
        new_fig = new_fig.update_layout(showlegend=False)

    return new_fig, new_fig_context, _extras_output_dict, figure_out_paths


# ==================================================================================================================== #
# SpecificPrePostDeltaScatter                                                                                          #
# ==================================================================================================================== #

from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import plotly_pre_post_delta_scatter
from pyphocorehelpers.Filesystem.path_helpers import sanitize_filename_for_Windows


@metadata_attributes(short_name=None, tags=['plotly', 'scatter', 'notebook', 'figure', 'outputs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-08 15:10', related_items=[])
class SpecificPrePostDeltaScatter:
    """ not yet finished.
    """
    _generic_kwargs = {'fig_size_kwargs': None, 'is_dark_mode': False, 'histogram_bins': 25, 'num_sessions': 1}
    
    @classmethod
    def _pre_post_delta_scatter_laps_per_time_bin(cls, all_sessions_laps_time_bin_df, t_delta_df, t_delta_dict, earliest_delta_aligned_t_start, latest_delta_aligned_t_end, **kwargs):
        histogram_bins = 25
        num_sessions = 1

        # ==================================================================================================================== #
        # Laps                                                                                                                 #
        # ==================================================================================================================== #
        # all_sessions_laps_time_bin_df
        # all_sessions_simple_pearson_laps_df

        # Define the legend groups you want to hide on startup
        # legend_groups_to_hide = ['0.030', '0.044', '0.050', '0.058'] # '0.025', 
        legend_groups_to_hide = kwargs.pop('legend_groups_to_hide', []) # ['0.030', '0.044', '0.050', '0.058'] 

        # data_context = IdentifyingContext(epochs_name='laps', data_grain='per_epoch', title_prefix="Lap Per Epoch")
        # concatenated_ripple_df = deepcopy(all_sessions_laps_df)
        data_context = IdentifyingContext(epochs_name='laps', data_grain='per_time_bin', title_prefix="Lap Individual Time Bins")
        concatenated_ripple_df = deepcopy(all_sessions_laps_time_bin_df)

        # data_context = IdentifyingContext(epochs_name='PBE', data_grain='per_epoch', title_prefix="PBE Per Epoch")
        # concatenated_ripple_df = deepcopy(all_sessions_ripple_df)
        # data_context = IdentifyingContext(epochs_name='PBE', data_grain='per_time_bin', title_prefix="PBE Individual Time Bins")
        # concatenated_ripple_df = deepcopy(all_sessions_ripple_time_bin_df)

        # concatenated_ripple_df = deepcopy(all_sessions_simple_pearson_laps_df) # ['P_LR', 'P_RL', 'P_Long', 'P_Short', 'ripple_idx', 'ripple_start_t', 'P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL', 'most_likely_decoder_index', 'start', 'stop', 'label', 'duration', 'long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr', 'best_decoder_index', 'session_name', 'time_bin_size', 'delta_aligned_start_t', 'is_user_annotated_epoch', 'is_valid_epoch', 'custom_replay_name', 'epoch_idx', 'long_best_pf_peak_x_pearsonr', 'short_best_pf_peak_x_pearsonr', 'wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL', 'long_best_wcorr', 'short_best_wcorr', 'wcorr_abs_diff', 'pearsonr_abs_diff']
        # concatenated_ripple_df = deepcopy(all_sessions_laps_time_bin_df) # ['P_LR', 'P_RL', 'P_Long', 'P_Short', 'ripple_idx', 'ripple_start_t', 'P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL', 'most_likely_decoder_index', 'start', 'stop', 'label', 'duration', 'long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr', 'best_decoder_index', 'session_name', 'time_bin_size', 'delta_aligned_start_t', 'is_user_annotated_epoch', 'is_valid_epoch', 'custom_replay_name', 'epoch_idx', 'long_best_pf_peak_x_pearsonr', 'short_best_pf_peak_x_pearsonr', 'wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL', 'long_best_wcorr', 'short_best_wcorr', 'wcorr_abs_diff', 'pearsonr_abs_diff']
        # print(f'concatenated_ripple_df.columns: {list(concatenated_ripple_df.columns)}')
        # concatenated_ripple_df

        # variable_name = 'P_Long'
        variable_name = 'P_Short' # Shows expected effect - short-only replay prior to delta and then split replays post-delta
        # variable_name = 'P_LR'

        y_baseline_level: float = 0.5 # for P(short), etc
        # y_baseline_level: float = 0.0 # for wcorr, etc

        # px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'color':"is_user_annotated_epoch", 'title': f"'{variable_name}'"} # , 'color': 'time_bin_size', 'range_y': [-1.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size', 'is_user_annotated_epoch':'user_sel'}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'title': f"{data_context.get_description(subset_includelist=['title_prefix'])} - '{variable_name}'"} # , 'color': 'time_bin_size', 'range_y': [-1.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size', 'is_user_annotated_epoch':'user_sel'}
        px_scatter_kwargs['color'] = "time_bin_size"
        # px_scatter_kwargs.pop('color')

        concatenated_ripple_df['dummy_column_for_size'] = 1.0
        px_scatter_kwargs['size'] = "dummy_column_for_size"
        px_scatter_kwargs['size_max'] = 3
        # px_scatter_kwargs['marker'] = dict(line=dict(width=0))
        # Remove the white border around scatter points by setting line width to 0
        # px_scatter_kwargs['line_width'] = 0  # <---- Correct way for Plotly Express

        # px_scatter_kwargs.update(dict(marginal_x="histogram", marginal_y="rug"))

        hist_kwargs = dict(color="time_bin_size")
        # hist_kwargs = dict(color="is_user_annotated_epoch") # , histnorm='probability density'
        # hist_kwargs.pop('color')
        new_fig_laps, new_fig_laps_context = plotly_pre_post_delta_scatter(data_results_df=concatenated_ripple_df, out_scatter_fig=None, histogram_bins=histogram_bins,
                                px_scatter_kwargs=px_scatter_kwargs, histogram_variable_name=variable_name, hist_kwargs=hist_kwargs, forced_range_y=None,
                                time_delta_tuple=(earliest_delta_aligned_t_start, 0.0, latest_delta_aligned_t_end), legend_title_text=None, is_dark_mode=is_dark_mode)


        new_fig_laps = new_fig_laps.update_layout(fig_size_kwargs)

        if legend_groups_to_hide is not None:
            # Collect all unique legend groups you want to hide
            hidden_groups = set(legend_groups_to_hide)

            # Iterate over traces and hide those in the specified legend groups
            for trace in new_fig_laps.data:
                if trace.legendgroup in hidden_groups:
                    trace.visible = 'legendonly'
                
        # new_fig_laps.show()

        _extras_output_dict = {}
        if is_dark_mode:
            _extras_output_dict["y_mid_line"] = new_fig_laps.add_hline(y=y_baseline_level, line=dict(color="rgba(0.8,0.8,0.8,.75)", width=2), row='all', col='all')
        else:
            _extras_output_dict["y_mid_line"] = new_fig_laps.add_hline(y=y_baseline_level, line=dict(color="rgba(0.2,0.2,0.2,.75)", width=2), row='all', col='all')


        # fig_to_clipboard(new_fig_laps, **fig_size_kwargs)
        new_fig_laps_context = new_fig_laps_context.adding_context_if_missing(**data_context.get_subset(subset_includelist=['epochs_name', 'data_grain']).to_dict(), num_sessions=num_sessions, plot_type='scatter+hist', comparison='pre-post-delta', variable_name=variable_name)
        # figure_out_paths = save_plotly(a_fig=new_fig_laps, a_fig_context=new_fig_laps_context)
        new_fig_laps



@function_attributes(short_name=None, tags=['across-session', 'time_bin_size'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-15 00:00', related_items=[])
def build_single_time_bin_size_dfs(all_sessions_all_scores_epochs_df, all_sessions_epochs_df, all_sessions_epochs_time_bin_df, target_time_bin_size: float, included_columns = ['delta_aligned_start_t', 'is_user_annotated_epoch', 'is_valid_epoch']):
    """ Filters the epochs dataframe down to a single time_bin_size specified by `target_time_bin_size`. 
     
     from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import build_single_time_bin_size_dfs
      
     History: 2024-07-15 - Factored out of Across Session Notebook
    """
    from neuropy.utils.indexing_helpers import PandasHelpers

    print(f'all_sessions_ripple_df.time_bin_size.unique(): {all_sessions_epochs_df.time_bin_size.unique()}')
    single_time_bin_size_all_sessions_epochs_df = deepcopy(all_sessions_epochs_df[np.isclose(all_sessions_epochs_df['time_bin_size'], target_time_bin_size)])
    print(f'np.shape(single_time_bin_size_all_sessions_ripple_df): {np.shape(single_time_bin_size_all_sessions_epochs_df)}')

    print(f'all_sessions_ripple_time_bin_df.time_bin_size.unique(): {all_sessions_epochs_time_bin_df.time_bin_size.unique()}')
    single_time_bin_size_all_sessions_epochs_time_bin_df = deepcopy(all_sessions_epochs_time_bin_df[np.isclose(all_sessions_epochs_time_bin_df['time_bin_size'], target_time_bin_size)])
    print(f'np.shape(single_time_bin_size_all_sessions_ripple_time_bin_df): {np.shape(single_time_bin_size_all_sessions_epochs_time_bin_df)}')

    # single_time_bin_size_all_sessions_ripple_time_bin_df
    # single_time_bin_size_all_sessions_ripple_df # has ['ripple_start_t']
    # all_sessions_all_scores_ripple_df

    ## recover the important columns (user-annotation, epoch validity) from the newer `all_sessions_all_scores_ripple_df` for use in 'single_time_bin_size_all_sessions_ripple_df'
    all_sessions_all_scores_epochs_df['delta_aligned_start_t'] = all_sessions_all_scores_epochs_df['delta_aligned_start_t'].astype(float)
    single_time_bin_size_all_sessions_epochs_df['delta_aligned_start_t'] = single_time_bin_size_all_sessions_epochs_df['delta_aligned_start_t'].astype(float)

    # Added 'delta_aligned_start_t' for the merge
    single_time_bin_size_all_sessions_epochs_df = PandasHelpers.add_explicit_dataframe_columns_from_lookup_df(single_time_bin_size_all_sessions_epochs_df, all_sessions_all_scores_epochs_df[included_columns], join_column_name='delta_aligned_start_t')
    single_time_bin_size_all_sessions_epochs_df.sort_values(by=['delta_aligned_start_t'], inplace=True) # Need to re-sort by timestamps once done
    single_time_bin_size_all_sessions_epochs_df

    single_time_bin_size_all_sessions_epochs_time_bin_df = PandasHelpers.add_explicit_dataframe_columns_from_lookup_df(single_time_bin_size_all_sessions_epochs_time_bin_df, all_sessions_all_scores_epochs_df[included_columns], join_column_name='delta_aligned_start_t')
    single_time_bin_size_all_sessions_epochs_time_bin_df.sort_values(by=['t_bin_center'], inplace=True) # Need to re-sort by timestamps once done
    
    ## Add plotly helper columns:
    for a_df in (all_sessions_all_scores_epochs_df, all_sessions_epochs_df, all_sessions_epochs_time_bin_df, single_time_bin_size_all_sessions_epochs_df, single_time_bin_size_all_sessions_epochs_time_bin_df):
        a_df['pre_post_delta_category'] = 'post-delta'
        a_df['pre_post_delta_category'][a_df['delta_aligned_start_t'] < 0.0] = 'pre-delta'

    ## OUTPUTS: single_time_bin_size_all_sessions_ripple_df, single_time_bin_size_all_sessions_ripple_time_bin_df
    return single_time_bin_size_all_sessions_epochs_df, single_time_bin_size_all_sessions_epochs_time_bin_df


from functools import partial
from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers
import plotly.io as pio
import plotly.graph_objects as go
from ipydatagrid import Expr, DataGrid, TextRenderer, BarRenderer # for use in DataFrameFilter
from IPython.display import display, Javascript
import base64
import solara # `pip install "solara[assets]`
import json
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from pyphocorehelpers.gui.Jupyter.simple_widgets import CheckBoxListWidget
from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
from pyphocorehelpers.assertion_helpers import Assert
from attrs import define, field, Factory
from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorPlottingDatasource, LoadedPosteriorContainer
import threading

def _build_solera_file_download_widget(fig, filename="figure-image.png", label="Save Figure", get_png_bytes=None):
    """ 
    _file_download_widget = _build_solera_file_download_widget(fig=self.figure_widget, filename="figure-image.png")
    _file_download_widget = _build_solera_file_download_widget(fig=self.figure_widget, get_png_bytes=self._get_export_png_bytes)
    
    """
    mime_type = "image/png"
    if get_png_bytes is None:
        def _default_get_png_bytes():
            return pio.to_image(fig, format='png')
        get_png_bytes = _default_get_png_bytes
    return solara.FileDownload.widget(data=get_png_bytes, filename=filename, label=label, mime_type=mime_type, )

@custom_define(slots=False, eq=False)
class DataframeFilterPredicates(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    is_enabled: bool = serialized_attribute_field(default=True)
    


from enum import Enum, auto

class InitialSelectionModeEnum(Enum):
    """Defines how the initial selection of a widget should be set (none selected, all selected, first selected
    
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import InitialSelectionModeEnum
    
    
    , initial_selection_mode: InitialSelectionModeEnum=InitialSelectionModeEnum.NO_SELECTION, custom_selection: Optional[List]=None
    
    
        # Set initial widget selection/selections ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        initial_selection = []
        if (initial_selection_mode.value == InitialSelectionModeEnum.NO_SELECTION.value):
            initial_selection = []
        elif (initial_selection_mode.value == InitialSelectionModeEnum.FIRST_SELECTED.value):
            if len(a_col_values_options) > 0:
                initial_selection = [a_col_values_options[0]]
        elif (initial_selection_mode.value == InitialSelectionModeEnum.CUSTOM_SELECTED.value):
            assert custom_selection is not None, f"with this mode you must provide a custom_selection"
            initial_selection = deepcopy(custom_selection)
        elif (initial_selection_mode.value == InitialSelectionModeEnum.ALL_SELECTED.value):
            initial_selection = deepcopy(a_col_values_options)
        else:
            raise NotImplementedError(f'{initial_selection_mode} is not VALID')

        a_widget.value = initial_selection
        

    
    """
    NO_SELECTION = auto()
    FIRST_SELECTED = auto()
    CUSTOM_SELECTED = auto()
    ALL_SELECTED = auto()

    def __str__(self):
        return self.name

    @classmethod
    def list_values(cls):
        """Returns a list of all enum values"""
        return list(cls)

    @classmethod
    def list_names(cls):
        """Returns a list of all enum names"""
        return [e.name for e in cls]


@custom_define(slots=False, eq=False)
class DataFrameFilter(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ Involves an interactive ipywidget, and handles interactive filtering of dataframes by presenting a jupyter widget interface.
    
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import DataFrameFilter


        assert 'best_overall_quantile' in all_sessions_all_scores_ripple_df
        assert 'best_overall_quantile' in all_sessions_MultiMeasure_ripple_df

        min_wcorr_threshold: float = 0.7
        high_pearsonr_threshold: float = 0.9
        high_shuffle_score_threshold: float = 0.9
        high_shuffle_wcorr_z_score_threshold: float = 0.9

        additional_filter_predicates = {
            'high_wcorr': (lambda df: np.any((df[['long_best_wcorr', 'short_best_wcorr']].abs() > min_wcorr_threshold), axis=1)),
            'user_selected': lambda df: np.all((df[['is_user_annotated_epoch', 'is_valid_epoch']]), axis=1),
            'high_pearsonr_corr': (lambda df: np.any((df[['long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr']].abs() > high_pearsonr_threshold), axis=1)),
            'high_shuffle_percentile_score': (lambda df: (df['best_overall_quantile'].abs() > high_shuffle_score_threshold)),
            'high_shuffle_wcorr_z_score': (lambda df: (df['best_overall_wcorr_z'].abs() > high_shuffle_wcorr_z_score_threshold)),
        }

        ## ensure that the qclu is always before the frateThresh, reversing them if needed:
        replay_name: str = 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0'
        # time_bin_size: float = 0.025
        time_bin_size: Tuple[float] = (0.025, 0.058)

        _build_filter_changed_plotly_plotting_callback_fn = DataFrameFilter._build_plot_callback(earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, save_plotly=save_plotly, resolution_multiplier=resolution_multiplier)

        df_filter: DataFrameFilter = DataFrameFilter(
            all_sessions_ripple_df=all_sessions_ripple_df,
            all_sessions_ripple_time_bin_df=all_sessions_ripple_time_bin_df,
            all_sessions_MultiMeasure_ripple_df=all_sessions_MultiMeasure_ripple_df,
            all_sessions_all_scores_ripple_df=all_sessions_all_scores_ripple_df,
            all_sessions_laps_df=all_sessions_laps_df,
            all_sessions_laps_time_bin_df=all_sessions_laps_time_bin_df,
            all_sessions_MultiMeasure_laps_df=all_sessions_MultiMeasure_laps_df,
            additional_filter_predicates=additional_filter_predicates,
            initially_is_checked={'high_wcorr': True, 'user_selected': True},
            on_filtered_dataframes_changed_callback_fns={'build_filter_changed_plotly_plotting_callback_fn': _build_filter_changed_plotly_plotting_callback_fn},
            active_plot_df_name='filtered_all_sessions_all_scores_ripple_df',
        )

        # Set initial values: ________________________________________________________________________________________________ #
        df_filter.replay_name = replay_name # 'withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]'
        # df_filter.time_bin_size = (time_bin_size, )
        df_filter.time_bin_size = time_bin_size
        # df_filter.time_bin_size = (0.025, 0.058)
        df_filter.update_filters()
        df_filter.display()

    """
    ## Original DataFrames passed during initialization
    _original_df_dict: Dict[str, Optional[pd.DataFrame]] = serialized_field()
    # Filtered DataFrames
    _filtered_df_dict: Dict[str, Optional[pd.DataFrame]] = serialized_field(default=Factory(dict))
    

    active_plot_df_name: str = serialized_attribute_field(default='filtered_all_sessions_all_scores_ripple_df')
    active_plot_variable_name: str = serialized_attribute_field(default='P_Short')
    active_plot_fn_kwargs: Dict = serialized_field(default=Factory(dict))
    # Add filename attribute
    filename: str = serialized_attribute_field(init=False, default='figure.png')
    default_output_folder: Optional[Path] = non_serialized_field(default=None) # preferred folder for later programmatic saves; browser download uses basename only
    include_date_prefix: bool = non_serialized_field(default=True) # when True, prefix download basename with YYYY-MM-DD_
    include_export_controls_and_table: bool = non_serialized_field(default=False) # when True, Copy/Save PNG includes filter-control + stats-table bands around the figure

    additional_filter_predicates = non_serialized_field(default=Factory(dict)) # a list of boolean predicates to be applied as filters
    initially_is_checked: Optional[Dict[str, bool]] = non_serialized_field(default=None) # optional {predicate_name: True/False} to pre-check predicate checkboxes at init; unknown keys ignored
    on_filtered_dataframes_changed_callback_fns = non_serialized_field(default=Factory(dict)) # a list of callables that will be called when the filters are changed. 
    
    selected_points = non_serialized_field(default=Factory(dict))

    step_by_step_predicate_filtered_row_counts_dict: Dict[str, Dict[str, int]] = non_serialized_field(default=Factory(dict)) # a Dict of Dict showing the number of entries after each update


    # Widgets (will be initialized in __attrs_post_init__) _______________________________________________________________ #
    replay_name_widget = non_serialized_field(init=False)
    time_bin_size_widget = non_serialized_field(init=False)
    active_filter_predicate_selector_widget: CheckBoxListWidget = non_serialized_field(init=False)
    active_plot_df_name_selector_widget = non_serialized_field(init=False)
    active_plot_variable_name_widget = non_serialized_field(init=False)
    custom_dynamic_filter_widgets_list: List = non_serialized_field(init=False, metadata={'desc': 'stores references to the widgets with knowledge of which properties to update to filter the dataframe.'})
    
    is_figure_widget_mode: bool = non_serialized_field(default=True)

    output_widget: widgets.Output = non_serialized_field(init=False)
    figure_widget: Optional[go.FigureWidget] = non_serialized_field(init=False, default=None)
    table_widget: Optional[DataGrid] = non_serialized_field(init=False, default=None)

    # Add button widgets as class attributes
    button_copy: Optional[widgets.Button] = non_serialized_field(init=False, default=None)
    button_download: Optional[widgets.CoreWidget] = non_serialized_field(init=False, default=None)
    filename_label: Optional[widgets.Label] = non_serialized_field(init=False, default=None)
    # Add Output widget for JavaScript execution
    # js_output: widgets.Output = non_serialized_field(init=False)
    hover_posterior_preview_figure_widget: Optional[go.FigureWidget] = non_serialized_field(init=False, default=None)
    _hover_posterior_preview_box: Optional[widgets.Box] = non_serialized_field(init=False, default=None)
    # hover_posterior_data: LoadedPosteriorContainer = non_serialized_field()
    hover_posterior_data: Optional[PosteriorPlottingDatasource] = non_serialized_field(default=None)

    debounce_timer: Optional[threading.Timer] = non_serialized_field(default=None) 
    debounce_delay_ms: int = non_serialized_field(default=300)  # milliseconds
    _update_lock: threading.RLock = non_serialized_field(default=Factory(threading.RLock))
    _hover_posterior_debounce_timer: Optional[threading.Timer] = non_serialized_field(default=None)
    hover_posterior_debounce_delay_ms: int = non_serialized_field(default=75)
    _hover_posterior_pending_payload: Optional[Dict] = non_serialized_field(default=None)
    _last_hover_posterior_key: Optional[tuple] = non_serialized_field(default=None)
    _hover_posterior_customdata_cols: List[str] = non_serialized_field(default=Factory(list))

    _HOVER_POSTERIOR_DECODER_NAMES = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
    _HOVER_POSTERIOR_CUSTOMDATA_COLS = ['session_name', 'format_name', 'animal', 'exper_name', 'custom_replay_name', 'time_bin_size', 'start']
    _HOVER_POSTERIOR_START_COL_CANDIDATES = ['start', 'parent_epoch_start_t', 'ripple_start_t', 'lap_start_t']


    # Begin Properties ___________________________________________________________________________________________________ #
    @property
    def preferred_output_path(self) -> Optional[Path]:
        """Folder + constructed basename when default_output_folder is set; else None."""
        if self.default_output_folder is None:
            return None
        return Path(self.default_output_folder) / self.filename


    @property
    def replay_name(self) -> str:
        """The replay_name property."""
        return self.replay_name_widget.value
    @replay_name.setter
    def replay_name(self, value):
        self.replay_name_widget.value = value

    @property
    def time_bin_size(self) -> str:
        """The time_bin_size property."""
        return self.time_bin_size_widget.value
    @time_bin_size.setter
    def time_bin_size(self, value):
        # Combine all DataFrames to get unique options
        # combined_df = self.all_sessions_ripple_df.append(self.all_sessions_laps_df, ignore_index=True)
        # replay_name_options = sorted(combined_df['custom_replay_name'].unique())
        # time_bin_size_options = sorted(combined_df['time_bin_size'].unique())

        # Set default initial_time_bin_sizes if not provided
        if value is None:
            # Default to selecting all options or the first option
            # initial_time_bin_sizes = (time_bin_size_options[0],)
            value = tuple() # empty tuple
            pass
        else:
            # Ensure initial_time_bin_sizes is a tuple
            if isinstance(value, (float, int)):
                value = (value,)
            elif isinstance(value, list):
                value = tuple(value)
            elif isinstance(value, tuple):
                pass  # already a tuple
            else:
                raise ValueError("initial_time_bin_sizes must be a float, int, list, or tuple")

        try:
            self.time_bin_size_widget.value = value
        except Exception as e:
            print(f'value: {value} is no in the list of valid options: {self.time_bin_size_widget.options}')
            raise e
        

    
    @property
    def filter_context(self) -> IdentifyingContext:
        """The time_bin_size property."""
        return IdentifyingContext(time_bin_sizes=self.time_bin_size, custom_suffix=self.replay_name)
    

    @property
    def original_df_dict(self) -> Dict[str, pd.DataFrame]:
        """The original_df_list property."""
        return {k:v for k, v in self._original_df_dict.items() if (v is not None)}
            

    @property
    def filtered_df_dict(self) -> Dict[str, pd.DataFrame]:
        """The original_df_list property."""
        return {k:v for k, v in self._filtered_df_dict.items() if (v is not None)}


    @property
    def original_df_names(self) -> List[str]:
        return sorted(list(self.original_df_dict.keys()))
    
    @property
    def filtered_df_names(self) -> List[str]:
        return sorted(list(self.filtered_df_dict.keys()))
    
    @property
    def filtered_size_info_df(self) -> pd.DataFrame:
        """The size of the filtered dataframes with the current filters."""
        n_records_tuples = [(name, len(df), len(self.original_df_dict.get(name.removeprefix('filtered_'), df))) for name, df in self.filtered_df_dict.items() if (df is not None)]
        return pd.DataFrame(n_records_tuples, columns=['df_name', 'n_elements', 'n_total'])


    @property
    def plot_variable_name_options(self) -> List[str]:
        return sorted(['P_Short', 'best_overall_quantile', 'best_overall_wcorr_z'])
    

    @property
    def active_plot_df(self) -> pd.DataFrame:
        """The selected filtered dataframe to use with the plot."""
        assert self.active_plot_df_name in list(self.filtered_df_dict.keys()), f"self.active_plot_df_name: '{self.active_plot_df_name}' not in list(self.filtered_df_dict.keys()): {list(self.filtered_df_dict.keys())}"
        return self.filtered_df_dict[self.active_plot_df_name]




    # ==================================================================================================================== #
    # Initializers                                                                                                         #
    # ==================================================================================================================== #
    def __attrs_post_init__(self):
        # This method runs after the generated __init__
        self._init_filtered_dataframes()
        if self.hover_posterior_data is not None and (not isinstance(self.hover_posterior_data, PosteriorPlottingDatasource)):
            self.hover_posterior_data = PosteriorPlottingDatasource.from_loaded_containers(self.hover_posterior_data)

        self._setup_widgets()
        # Initial filtering with default widget values
        if getattr(self, 'time_bin_size_widget', None) is not None:
            self.update_filtered_dataframes(self.replay_name_widget.value, self.time_bin_size_widget.value)
        # else:
        #     self.update_filtered_dataframes(self.replay_name_widget.value)

        # Button Widget Initialize ___________________________________________________________________________________________ #
        # Set up the buttons after figure_widget is created
        if self.is_figure_widget_mode:
            self._setup_widgets_buttons()


        ## set time_bin_size selections to the first value by default
        initial_selection = self.set_initial_selection(a_widget=self.time_bin_size_widget, initial_selection_mode=InitialSelectionModeEnum.FIRST_SELECTED)


    
    @classmethod
    def set_initial_selection(cls, a_widget, initial_selection_mode: InitialSelectionModeEnum=InitialSelectionModeEnum.FIRST_SELECTED, custom_initial_selections: Optional[Union[List, Any]]=None):
        # Set initial widget selection/selections ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        import ipywidgets as widgets        

        a_col_values_options = deepcopy(a_widget.options)
        allows_multiple_selection: bool = (isinstance(a_widget, (widgets.SelectMultiple, )))
        

        initial_selection = []
        if (initial_selection_mode.value == InitialSelectionModeEnum.NO_SELECTION.value):
            if allows_multiple_selection:
                initial_selection = []
            else:
                initial_selection = ''
        elif (initial_selection_mode.value == InitialSelectionModeEnum.FIRST_SELECTED.value):
            if len(a_col_values_options) > 0:
                if allows_multiple_selection:
                    initial_selection = [a_col_values_options[0]]
                else:
                    initial_selection = a_col_values_options[0]
        elif (initial_selection_mode.value == InitialSelectionModeEnum.CUSTOM_SELECTED.value):
            assert custom_initial_selections is not None, f"with this mode you must provide a custom_selection"
            initial_selection = deepcopy(custom_initial_selections)
        elif (initial_selection_mode.value == InitialSelectionModeEnum.ALL_SELECTED.value):
            if allows_multiple_selection:
                initial_selection = deepcopy(a_col_values_options)
            else:
                raise ValueError(f'{initial_selection_mode} is not VALID - `InitialSelectionModeEnum.ALL_SELECTED` mode is not allowed for widgets that do not permit multiple selections (such as widgets.Dropdown). type(a_widget): {type(a_widget)}')
        else:
            raise NotImplementedError(f'{initial_selection_mode} is not VALID')

        a_widget.value = initial_selection
        return initial_selection
    


    @function_attributes(short_name=None, tags=['filter', 'dynamic', 'ui', 'widget'], input_requires=[], output_provides=[], uses=['._rebuild_predicate_widget()'], used_by=[], creation_date='2025-03-27 14:05', related_items=[])
    def build_extra_dropdown_widget(self, a_name: str = 'replay_name', df_col_name: str = 'custom_replay_name', a_widget_label: str = 'Replay Name:', initial_selection_mode: InitialSelectionModeEnum=InitialSelectionModeEnum.FIRST_SELECTED, custom_initial_selection: Optional[Any]=None) -> widgets.Dropdown:
        """ adds a new dropdown widget to refine the active points, triggers `self._on_widget_change` when a selection is made

        Works by adding the widget as property of this instance, and imposing the widget's selection criteria by adding a custom predicate to `self.additional_filter_predicates`. 
        The predicate selector widget is then rebuilt by calling `self._rebuild_predicate_widget(...)`
        
        
        Usage:        
            df_filter.build_extra_control_widget(a_name='trained_compute_epochs', df_col_name='trained_compute_epochs', a_widget_label='TrainedComputeEpochs :')
            df_filter.build_extra_control_widget(a_name='pfND_ndim', df_col_name='pfND_ndim', a_widget_label='pfND_ndim:')
            df_filter.build_extra_control_widget(a_name='decoder_identifier', df_col_name='decoder_identifier', a_widget_label='decoder_identifier:')
            df_filter.build_extra_control_widget(a_name='data_grain', df_col_name='data_grain', a_widget_label='data_grain:')
            df_filter.build_extra_control_widget(a_name='masked_time_bin_fill_type', df_col_name='masked_time_bin_fill_type', a_widget_label='masked_time_bin_fill_type:')
            df_filter.build_extra_control_widget(a_name='known_named_decoding_epochs_type', df_col_name='known_named_decoding_epochs_type', a_widget_label='known_named_decoding_epochs_type:')

        """
        import ipywidgets as widgets
        from traitlets import Dict as TraitDict  # Import the Dict traitlet

        a_widget_name: str = f"{a_name}_widget" # replay_name_widget
        extant_widget = getattr(self, a_widget_name, None)
        if extant_widget is not None:
            raise NotImplementedError(f'Not sure what to do with extant widgets yet! a_widget_label: "{a_widget_label}".')

        ## Create and add new widget:
        a_col_values_options = sorted(self.active_plot_df[df_col_name].astype(str).unique())    
        a_widget = widgets.Dropdown(
                    options=a_col_values_options,
                    description=a_widget_label,
                    disabled=False,
                    layout=widgets.Layout(width='500px'),
                    style={'description_width': 'initial'},
                    # metadata={"desc": "build_extra_control_widget", "df_col_name": df_col_name, "name": a_name, "widget_name": a_widget_name, }, ## DOES NOT WORK
                )
        
        ## add_traits (does work)
        a_widget.metadata = TraitDict({  # Use the Dict traitlet here
            "desc": "build_extra_dropdown_widget",
            "df_col_name": df_col_name,
            "name": a_name,
            "widget_name": a_widget_name,
        })
        
        # Build the appropriate filter predicate to go along with the custom control _________________________________________ #
        self.additional_filter_predicates.update({
            f'{a_widget_name}': (lambda df: (df[df_col_name].astype(str) == str(a_widget.value))),
        })

        ## INPUTS: self.custom_dynamic_filter_widgets_list
        ## Add to output widgets list
        self.custom_dynamic_filter_widgets_list.append(a_widget)
        setattr(self, a_widget_name, a_widget) # self.replay_name_widget
        ## INPUTS: self.custom_dynamic_filter_widgets_dict, self.custom_dynamic_filter_widgets_property_map

        ## Update the predicate enabled selection widget and default to enabling this predicate:
        self._rebuild_predicate_widget(initially_is_checked={f'{a_widget_name}':True})

        # Set up observers to handle changes in widget values
        a_widget.observe(self._on_widget_change, names='value')

        # Set initial widget selection/selections ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        if custom_initial_selection is not None:
            initial_selection_mode = InitialSelectionModeEnum.CUSTOM_SELECTED
        initial_selection = self.set_initial_selection(a_widget=a_widget, initial_selection_mode=initial_selection_mode, custom_initial_selections=custom_initial_selection)

        return a_widget


    @function_attributes(short_name=None, tags=['filter', 'dynamic', 'ui', 'widget'], input_requires=[], output_provides=[], uses=['._rebuild_predicate_widget()'], used_by=[], creation_date='2025-03-27 14:05', related_items=[])
    def build_extra_selectMultiple_widget(self, a_name: str = 'replay_name', df_col_name: str = 'custom_replay_name', a_widget_label: str = 'Replay Name:', initial_selection_mode: InitialSelectionModeEnum=InitialSelectionModeEnum.FIRST_SELECTED, custom_initial_selections: Optional[List]=None) -> widgets.SelectMultiple:
        """ adds a new dropdown widget to refine the active points, triggers `self._on_widget_change` when a selection is made

        Works by adding the widget as property of this instance, and imposing the widget's selection criteria by adding a custom predicate to `self.additional_filter_predicates`. 
        The predicate selector widget is then rebuilt by calling `self._rebuild_predicate_widget(...)`
        
        
        Usage:        
            _a_time_bin_size_widget = df_filter.build_extra_selectMultiple_widget(a_name='time_bin_size', df_col_name='time_bin_size', a_widget_label='Time Bin Size:')

        """
        import ipywidgets as widgets
        from traitlets import Dict as TraitDict  # Import the Dict traitlet

        a_widget_name: str = f"{a_name}_widget" # replay_name_widget
        extant_widget = getattr(self, a_widget_name, None)
        if extant_widget is not None:
            raise NotImplementedError(f'Not sure what to do with extant widgets yet! a_widget_label: "{a_widget_label}".')

        ## Create and add new widget:
        a_col_values_options = sorted(self.active_plot_df[df_col_name].astype(str).unique())    
        a_widget = widgets.SelectMultiple(
            options=a_col_values_options,
            description=a_widget_label,
            disabled=False,
            layout=widgets.Layout(width='300px'),
            style={'description_width': 'initial'},
        )


        ## add_traits (does work)
        a_widget.metadata = TraitDict({  # Use the Dict traitlet here
            "desc": "build_extra_selectMultiple_widget",
            "df_col_name": df_col_name,
            "name": a_name,
            "widget_name": a_widget_name,
        })
        
        # Build the appropriate filter predicate to go along with the custom control _________________________________________ #
        # [str(v) for v in a_widget.value]
        self.additional_filter_predicates.update({
            f'{a_widget_name}': (lambda df: (df[df_col_name].astype(str).isin([str(v) for v in a_widget.value]))),
        })

        ## INPUTS: self.custom_dynamic_filter_widgets_list
        ## Add to output widgets list
        self.custom_dynamic_filter_widgets_list.append(a_widget)
        setattr(self, a_widget_name, a_widget) # self.replay_name_widget
        ## INPUTS: self.custom_dynamic_filter_widgets_dict, self.custom_dynamic_filter_widgets_property_map

        ## Update the predicate enabled selection widget and default to enabling this predicate:
        self._rebuild_predicate_widget(initially_is_checked={f'{a_widget_name}':True})

        # Set up observers to handle changes in widget values
        a_widget.observe(self._on_widget_change, names='value')


        # Set initial widget selection/selections ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        initial_selection = self.set_initial_selection(a_widget=a_widget, initial_selection_mode=initial_selection_mode, custom_initial_selections=custom_initial_selections)
        
        return a_widget


    @function_attributes(short_name=None, tags=['private', 'widget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-27 12:18', related_items=[])
    def _setup_widgets(self):
        """ builds the interactive widgets

        When a predicate control is changed, `self._on_widget_change` is called to update the plot and other widgets
        Only in `self.is_figure_widget_mode` == True (default), the main plotly figure widget `self.figure_widget` is built by calling `PlotlyFigureContainer._helper_build_pre_post_delta_figure_if_needed(...)`

        The number of rows information in each df is stored in `self.filtered_size_info_df` and displayed in `self.table_widget`


        """
        import plotly.subplots as sp
        from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import PlotlyFigureContainer
        
        self.custom_dynamic_filter_widgets_list = [] ## start with an empty list of additional filter widgets

        # Extract unique options for the widgets
        replay_name_options = sorted(self.active_plot_df['custom_replay_name'].astype(str).unique())
        time_bin_size_options = sorted(self.active_plot_df['time_bin_size'].unique())
        
        self.output_widget = widgets.Output(layout=widgets.Layout(width='100%', # min_width='200px', height='100px',
                                                                  border='1px solid black'),
                                                                  ) #  {'border': '1px solid black'}
        
        # Create dropdown widgets with adjusted layout and style
        self.replay_name_widget = widgets.Dropdown(
            options=replay_name_options,
            description='Replay Name:',
            disabled=False,
            layout=widgets.Layout(width='600px'),
            style={'description_width': 'initial'}
        )
        
        self.active_plot_df_name_selector_widget = widgets.Dropdown(
            options=sorted(self.filtered_df_names),
            description='Plot df Name:',
            disabled=False,
            layout=widgets.Layout(width='400px'),
            style={'description_width': 'initial'}
        )
        self.active_plot_df_name_selector_widget.value = self.active_plot_df_name
        
        self.active_plot_variable_name_widget = widgets.Dropdown(
            options=sorted(self.plot_variable_name_options),
            description='Plot Variable Name:',
            disabled=False,
            layout=widgets.Layout(width='300px'),
            style={'description_width': 'initial'}
        )
        self.active_plot_variable_name_widget.value = self.active_plot_variable_name
        
        predicate_keys = list(self.additional_filter_predicates.keys())
        initial_checked = self.initially_is_checked or {}
        options_dict = {k: bool(initial_checked.get(k, False)) for k in predicate_keys}
        self.active_filter_predicate_selector_widget = CheckBoxListWidget(options_list=options_dict)
            # description='Filter Predicates:',
            # disabled=False,
        # )
        self.active_filter_predicate_selector_widget.observe(self._on_widget_change, names='value')

        # Use SelectMultiple widget for time_bin_size
        self.time_bin_size_widget = widgets.SelectMultiple(
            options=time_bin_size_options,
            description='Time Bin Size:',
            disabled=False,
            layout=widgets.Layout(width='220px', height='100px'),
            style={'description_width': 'initial'},
        )
        
        # self.time_bin_size_widget = self.build_extra_selectMultiple_widget(a_name='time_bin_size', df_col_name='time_bin_size', a_widget_label='Time Bin Size:', initial_selection_mode=InitialSelectionModeEnum.FIRST_SELECTED)


        if self.is_figure_widget_mode:
            # If true, the interative plotly figure is rendered in a widget, otherwise only the filter selections are allowed.

            self.figure_widget, did_create_new_figure = PlotlyFigureContainer._helper_build_pre_post_delta_figure_if_needed(extant_figure=None, use_latex_labels=False, main_title=self.active_plot_df_name.removeprefix('filtered_'), figure_class=go.FigureWidget)
            self.figure_widget.layout.dragmode = 'select'
            # self.figure_widget.layout.dragmode = 'lasso'
            
            self._setup_hover_posterior_preview_widget()
        else:
            print(f'self.is_figure_widget_mode is False, so no plotly widget will be displayed')            



        # Set up observers to handle changes in widget values
        self.replay_name_widget.observe(self._on_widget_change, names='value')
        self.time_bin_size_widget.observe(self._on_widget_change, names='value')
        # self.active_filter_predicate_selector_widget.observe(self._on_widget_change, names='value')
        self.active_plot_df_name_selector_widget.observe(self._on_widget_change, names='value')
        self.active_plot_variable_name_widget.observe(self._on_widget_change, names='value')
        
        ## the table at the bottom that shows the active number of filtered points
        self.table_widget = DataGrid(self.filtered_size_info_df,
                                base_row_size=15, base_column_size=300, horizontal_stripes=True,
                                auto_fit_columns=False,
                                # auto_fit_columns=True,
                                # auto_fit_params={"area": "all", "padding": 30, "numCols": None},  # use "row-header" to target only the key/index column
                                #  renderers=renderers,
                                )
        self._update_table_widget_column_widths()


    def _setup_hover_posterior_preview_widget(self):
        """Build a 4-decoder stacked heatmap FigureWidget in a fixed-width layout pane.

        FigureWidget.layout is Plotly layout, not ipywidgets CSS, so the figure must sit inside a real
        Box/HBox with pixel width. Do not host it in Output+display() — that view never paints here.
        """
        decoder_names = list(self._HOVER_POSTERIOR_DECODER_NAMES)
        placeholder = np.zeros((8, 8), dtype=float)
        fig = go.FigureWidget().set_subplots(rows=len(decoder_names), cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=decoder_names)
        for i, decoder_name in enumerate(decoder_names):
            fig.add_heatmap(z=placeholder, showscale=False, name=decoder_name, colorscale='Gray', row=i + 1, col=1)
        ## END for i, decoder_name in enumerate(decoder_names)....

        has_data: bool = (self.hover_posterior_data is not None)
        title_text: str = 'Hover an epoch to preview posterior' if has_data else 'No posterior data loaded'
        fig.update_layout(autosize=False, margin=dict(l=50, r=10, t=40, b=20), width=620, height=280, title=dict(text=title_text, font=dict(size=12)), dragmode=False, hovermode=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        self.hover_posterior_preview_figure_widget = fig
        caption = widgets.HTML('<b>Hover posterior preview</b>')
        # Pixel-sized wrapper so DataGrid cannot collapse this pane; FigureWidget is a direct child (same as scatter).
        self._hover_posterior_preview_box = widgets.VBox(
            [caption, fig],
            layout=widgets.Layout(width='640px', min_width='640px', max_width='640px', height='320px', flex='0 0 640px', border='1px solid #bbbbbb', overflow='hidden', padding='4px'),
        )


    def update_preview_heatmap(self, images_by_decoder: Optional[Union[Dict[str, NDArray], NDArray]] = None, title: Optional[str] = None):
        """Replace the four decoder heatmap `z` arrays in the bottom preview widget (does not accumulate traces)."""
        fig = self.hover_posterior_preview_figure_widget
        if fig is None:
            return
        decoder_names = list(self._HOVER_POSTERIOR_DECODER_NAMES)
        if images_by_decoder is not None and (not isinstance(images_by_decoder, dict)):
            images_by_decoder = {decoder_names[0]: np.asarray(images_by_decoder)}
        with fig.batch_update():
            if title is not None:
                fig.layout.title.text = title
                if images_by_decoder is None:
                    return
                for i, decoder_name in enumerate(decoder_names):
                    img = images_by_decoder.get(decoder_name)
                    if (img is None) or (i >= len(fig.data)):
                        continue
                    arr = np.asarray(img)
                    if arr.ndim == 3:
                        arr = np.nanmean(arr, axis=-1)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    fig.data[i].z = arr
                ## END for i, decoder_name in enumerate(decoder_names)....


    def _update_table_widget_column_widths(self, non_index_col_default_width: int = 160, index_col_default_width: int = 200):
        """
        Updates (overwrites) `self.table_widget.column_widths` to update the displayed widths of the df column

        """
        table_widget_col_names: List[str] = list(set(['key', *self.table_widget.data.columns.tolist()]))
        # print(table_widget_col_names) # ['key', 'n_predicate_true_rows', 'n_remaining_rows', 'n_pred_filtered_rows', 'n_cum_filtered_rows']
        original_col_widths_dict = {}
        if (self.table_widget.column_widths is not None):
            original_col_widths_dict = deepcopy(self.table_widget.column_widths)

        table_widget_col_widths_dict: Dict[str, int] = dict(zip(table_widget_col_names, (len(table_widget_col_names) * [non_index_col_default_width])))
        table_widget_col_widths_dict['key'] = index_col_default_width ## good size for the key column
        # OUTPUTS: table_widget_col_widths_dict
        self.table_widget.column_widths = table_widget_col_widths_dict ## need to update the dict next


    # ==================================================================================================================================================================================================================================================================================== #
    # Figure/Table/Controls Export to Clipboard and Image                                                                                                                                                                                                                                  #
    # ==================================================================================================================================================================================================================================================================================== #
    @staticmethod
    def _format_export_control_value(value) -> str:
        """Human-readable widget/constraint values for export bands (unwrap single-item sequences)."""
        if isinstance(value, (tuple, list)):
            if len(value) == 1:
                return str(value[0])
            return ", ".join([str(v) for v in value])
        return str(value)


    @staticmethod
    def _resolve_export_controls_font() -> str:
        """Prefer system Arial (matches Plotly); else bundled DejaVuSans."""
        from pathlib import Path
        for candidate in (Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/Arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
            if candidate.is_file():
                return str(candidate)
        ## END for candidate in (Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/Arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))....
        return 'DejaVuSans.ttf'


    def _get_export_png_bytes(self) -> bytes:
        """PNG of the Plotly figure; optionally composite filter controls (top) + stats table (bottom) when `include_export_controls_and_table` is True."""
        import io
        from pyphocorehelpers.plotting.media_output_helpers import figure_to_pil_image, ImageOperationsAndEffects

        width = self.figure_widget.layout.width
        height = self.figure_widget.layout.height
        to_image_kwargs = {}
        if width is not None:
            to_image_kwargs['width'] = width
        if height is not None:
            to_image_kwargs['height'] = height

        img = figure_to_pil_image(self.figure_widget, format='png', **to_image_kwargs)
        assert img is not None, "figure_to_pil_image returned None"
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        if self.include_export_controls_and_table:
            # Control summary from extant filter constraint dict + plot/predicate widgets
            _fmt = self._format_export_control_value
            control_rows = [f"{k}: {_fmt(v)}" for k, v in self.get_df_filter_active_constraint_dict().items()]
            if getattr(self, 'active_filter_predicate_selector_widget', None) is not None:
                control_rows.append(f"filter_predicates: {_fmt(self.active_filter_predicate_selector_widget.value)}")
            if getattr(self, 'active_plot_df_name_selector_widget', None) is not None:
                control_rows.append(f"plot_df: {_fmt(self.active_plot_df_name_selector_widget.value)}")
            if getattr(self, 'active_plot_variable_name_widget', None) is not None:
                control_rows.append(f"plot_variable: {_fmt(self.active_plot_variable_name_widget.value)}")
            controls_text = "\n".join(control_rows) if control_rows else "(no filter controls)"

            controls_font = self._resolve_export_controls_font()
            img = ImageOperationsAndEffects.add_boxed_adjacent_label(img, controls_text, image_edge='top', text_color=(0, 0, 0), background_color=(255, 255, 255, 255), font=controls_font, font_size=14, relative_padding=0.008, spacing=4)

            if (self.table_widget is not None) and (getattr(self.table_widget, 'data', None) is not None):
                table_df = self.table_widget.data
            else:
                table_df = self.filtered_size_info_df
            table_text = table_df.to_string() if table_df is not None else "(no table)"
            img = ImageOperationsAndEffects.add_boxed_adjacent_label(img, table_text, image_edge='bottom', text_color=(0, 0, 0), background_color=(255, 255, 255, 255), font='FreeMono.ttf', font_size=12, relative_padding=0.008, spacing=4)

        with io.BytesIO() as buf:
            img.save(buf, format='PNG')
            return buf.getvalue()


    def _setup_widgets_buttons(self):
        """Sets up the copy and download buttons."""
        self.button_copy = widgets.Button(description="Copy to Clipboard", icon='copy')
        # self.button_download = widgets.Button(description="Download Image", icon='save')
        self.button_download = _build_solera_file_download_widget(fig=self.figure_widget, filename="figure-image.png", get_png_bytes=self._get_export_png_bytes)
        
        # @solara.component
        # def Page():
        #     def get_data():
        #         # I run in a thread, so I can do some heavy processing
        #         time.sleep(3)
        #         # I only get called when the download is requested
        #         return "This is the content of the file"
        #     solara.FileDownload(get_data, "solara-lazy-download.txt")
            

        self.filename_label = widgets.Label()

        def _subfn_on_copy_button_click(b):
            # Convert the dashboard composite (controls + figure + table) to a PNG image
            width = self.figure_widget.layout.width
            height = self.figure_widget.layout.height
            print(f"Width: {width}, Height: {height}")

            png_bytes = self._get_export_png_bytes()
            encoded_image = base64.b64encode(png_bytes).decode('utf-8')

            # JavaScript code to copy the image to the clipboard using the canvas element
            js_code = f'''
                const img = new Image();
                img.src = 'data:image/png;base64,{encoded_image}';
                img.onload = function() {{
                    const canvas = document.createElement('canvas');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    canvas.toBlob(function(blob) {{
                        const item = new ClipboardItem({{ 'image/png': blob }});
                        navigator.clipboard.write([item]).then(function() {{
                            console.log('Image copied to clipboard');
                        }}).catch(function(error) {{
                            console.error('Error copying image to clipboard: ', error);
                        }});
                    }});
                }};
            '''

            display(Javascript(js_code))

        self.button_copy.on_click(_subfn_on_copy_button_click)

        ## Finish setup:
        self.on_widget_update_filename()  # Initialize filename and label
        # Set up observers for figure changes
        # self.figure_widget.layout.on_change(self.on_fig_layout_change, 'title', 'meta')
        # Initialize the Output widget for JavaScript execution
        # self.js_output = widgets.Output()


        # self.button_copy.on_click(self._subfn_on_copy_button_click)
        # self.button_download.on_click(self.on_download_button_click)

    # ==================================================================================================================== #
    # Widget Update Functions                                                                                              #
    # ==================================================================================================================== #
    def on_widget_update_filename(self):
        """Updates the filename and label based on the figure's title or metadata."""
        fig = self.figure_widget
        if fig is not None:
            preferred_filename = fig.layout.meta.get('preferred_filename') if fig.layout.meta else None
            if preferred_filename:
                stem = preferred_filename
            else:
                title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else "figure"
                stem = title.replace(' ', '_')

            if self.include_date_prefix:
                date_prefix = f"{date.today().strftime('%Y-%m-%d')}_"
                # Avoid double-prefixing if stem already starts with YYYY-MM-DD_
                already_prefixed = (len(stem) >= 11 and stem[4] == '-' and stem[7] == '-' and stem[10] == '_'
                                    and stem[:4].isdigit() and stem[5:7].isdigit() and stem[8:10].isdigit())
                if not already_prefixed:
                    stem = f"{date_prefix}{stem}"

            self.filename = f"{stem}.png"
            self.filename_label.value = stem

            ## Sync the live Solara download widget filename trait
            if self.button_download is not None:
                self.button_download.filename = self.filename
        
    def on_fig_layout_change(self, layout, *args):
        """Callback for when the figure's layout changes."""
        self.on_widget_update_filename()


    @function_attributes(short_name=None, tags=['MAIN', 'update', 'filter'], input_requires=[], output_provides=[], uses=['_debounced_update'], used_by=[], creation_date='2025-03-27 12:20', related_items=[])
    def _on_widget_change(self, change):
        """ this is the main update function that is called whenever an observed widget's value changes to update the filtered dataframe.
        Updates the bound variables from the widget's new value!

        Calls `self._debounced_update` after debounce timer fires to actually perform the update

        """
        import threading
        
        # Cancel any pending timer
        if self.debounce_timer:
            self.debounce_timer.cancel()
 
        # Set a new timer
        self.debounce_timer = threading.Timer(
            (self.debounce_delay_ms/1000.0), 
            self._debounced_update,
            [self.replay_name_widget.value, self.time_bin_size_widget.value, self.active_plot_df_name_selector_widget.value, self.active_plot_variable_name_widget.value]
        )
        self.debounce_timer.start()
        
        # Show immediate feedback
        with self.output_widget:
            print("Filter update scheduled...")


    @function_attributes(short_name=None, tags=['update', 'debounce', 'efficiency'], input_requires=[], output_provides=[], uses=['.update_filtered_dataframes', '.on_widget_update_filename'], used_by=['_on_widget_change'], creation_date='2025-05-02 06:23', related_items=[])
    def _debounced_update(self, replay_name, time_bin_sizes, active_plot_df_name, active_plot_variable_name):
        """Called after debounce delay
        To actually perform the main updates via: [`self.update_filtered_dataframes(...)`, `self.on_widget_update_filename()`]
        """
        ## do simple updates:
        # active_plot_df_name = self.active_plot_df_name_selector_widget.value
        self.active_plot_df_name = active_plot_df_name # self.active_plot_df_name_selector_widget.value
        self.active_plot_variable_name = active_plot_variable_name # self.active_plot_variable_name_widget.value             

        self.update_filtered_dataframes(replay_name, time_bin_sizes)
        self.on_widget_update_filename()
        


    @function_attributes(short_name=None, tags=['predicate', 'controls', 'filter'], input_requires=[], output_provides=[], uses=[], used_by=['.build_extra_control_widget'], creation_date='2025-03-27 14:05', related_items=[])
    def _rebuild_predicate_widget(self, initially_is_checked: Optional[Dict[str, bool]]=None):
        """ remove existing if needed, then rebuild """
        if hasattr(self, 'active_filter_predicate_selector_widget'):
            ## get existing is_checked values
            existing_options_dict = deepcopy(self.active_filter_predicate_selector_widget.options_dict)
            self.active_filter_predicate_selector_widget.unobserve(self._on_widget_change, names='value') ## stop observing changes
            self.active_filter_predicate_selector_widget.close()
            delattr(self, 'active_filter_predicate_selector_widget')
        else:
            existing_options_dict = {}

        if initially_is_checked is not None:
            existing_options_dict.update(initially_is_checked)
            
        # options_list = list(self.additional_filter_predicates.keys())
        new_options_dict = {k:existing_options_dict.get(k, False) for k in list(self.additional_filter_predicates.keys())} ## use the existing key value if there was one (user's selection), else default to False
        self.active_filter_predicate_selector_widget = CheckBoxListWidget(options_list=new_options_dict) ## create widget
        self.active_filter_predicate_selector_widget.observe(self._on_widget_change, names='value') ## observe changes

    @classmethod
    def _extract_widget_metadata(cls, a_widget) -> Dict:
        try:
            return a_widget.metadata.__dict__['default_args'][0]
        except Exception as e:
            raise


    @classmethod
    def _build_pho_constrain_df_code_str_from_filter_widget_constrain_dict(cls, filter_widget_constrain_dict: Dict) -> str:
        code_str: str = f'df = deepcopy(all_sessions_laps_time_bin_df).pho.constrain_df_cols('

        _kwargs_str: str = ', '.join([f'{k}="{v}"' for k, v in filter_widget_constrain_dict.items()])

        code_str = f'{code_str}{_kwargs_str})'
        return code_str






    @function_attributes(short_name=None, tags=['programmatic', 'code-generation', 'constrain_df'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-08-29 08:26', related_items=[])
    def get_df_filter_active_constraint_dict(self) -> Dict:
        """ Extracts the filter parameters (which can be passed to `df.pho.constrain(**filter_widget_constrain_dict)`) from the active filter config.


        Usage:

            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import DataFrameFilter

            filter_widget_constrain_dict = df_filter.get_df_filter_active_constraint_dict()
            filter_widget_constrain_dict

            filtered_df = df.pho.constrain(**filter_widget_constrain_dict)
            


        >> {'trained_compute_epochs': 'laps',
            'decoder_identifier': 'pseudo2D',
            'masked_time_bin_fill_type': ('dropped',),
            'replay_name': 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_2.0',
            'time_bin_size': (0.06,)}

        """
        from traitlets import Dict as TraitDict  # Import the Dict traitlet


        # all_widgets_list = [
        #         self.replay_name_widget, 
        #         self.time_bin_size_widget, 
        #         self.active_filter_predicate_selector_widget,
        #         self.active_plot_df_name_selector_widget, 
        #         self.active_plot_variable_name_widget
        #     ])

        #TODO 2025-08-29 08:24: - [ ] Hardcoded defn of the df col names to assign metadata to the auto-created widgets (which don't have any):
        all_widgets_dict = {
                self.replay_name_widget:{'df_col_name':'custom_replay_name'}, 
                self.time_bin_size_widget:{'df_col_name':'time_bin_size'}, 
                # self.active_plot_df_name_selector_widget, 
                # self.active_plot_variable_name_widget
        }
        
        # self.active_filter_predicate_selector_widget,
        
        for a_widget, a_widget_metadata_dict in all_widgets_dict.items():
            ## add_traits (does work)
            if (not hasattr(a_widget, 'metadata')) or (a_widget.metadata is None):
                ## adds the metadata if needed
                a_widget.metadata = TraitDict(a_widget_metadata_dict) 
        
        # [self.replay_name_widget.value, self.time_bin_size_widget.value, self.active_plot_df_name_selector_widget.value, self.active_plot_variable_name_widget.value]


        filter_widget_constrain_dict = {self._extract_widget_metadata(a_widget)['df_col_name']:a_widget.value for a_widget in self.custom_dynamic_filter_widgets_list}
        
        # for a_widget in all_widgets_list:
        # for a_widget in all_widgets_list:
        for a_widget, a_widget_metadata_dict in all_widgets_dict.items():   

            a_widget_metadata = self._extract_widget_metadata(a_widget)
            filter_widget_constrain_dict[a_widget_metadata['df_col_name']] = a_widget.value 
            # {'trained_compute_epochs': 'laps',
            # 'decoder_identifier': 'pseudo2D',
            # 'masked_time_bin_fill_type': ('nan_filled',)}

        return filter_widget_constrain_dict


    def _iter_column_filter_dimensions(self) -> List[Dict[str, Any]]:
        """Ordered column-filter dimensions from replay/time_bin/custom widgets.

        Returns list of dicts with keys: name, df_col_name, options, active_value, is_multi, widget_predicate_name
        """
        from traitlets import Dict as TraitDict  # Import the Dict traitlet

        # Ensure built-in widgets have metadata (same as get_df_filter_active_constraint_dict)
        builtin_widgets_meta = {
            self.replay_name_widget: {'df_col_name': 'custom_replay_name'},
            self.time_bin_size_widget: {'df_col_name': 'time_bin_size'},
        }
        for a_widget, a_widget_metadata_dict in builtin_widgets_meta.items():
            if (not hasattr(a_widget, 'metadata')) or (a_widget.metadata is None):
                a_widget.metadata = TraitDict(a_widget_metadata_dict)
        ## END for a_widget, a_widget_metadata_dict in builtin_widgets_meta.items()...

        ordered_widgets: List[Any] = [self.replay_name_widget, self.time_bin_size_widget, *list(self.custom_dynamic_filter_widgets_list)]
        dims: List[Dict[str, Any]] = []
        for a_widget in ordered_widgets:
            meta = self._extract_widget_metadata(a_widget)
            df_col_name: str = meta['df_col_name']
            is_multi: bool = isinstance(getattr(a_widget, 'value', None), (tuple, list))
            options = list(getattr(a_widget, 'options', []) or [])
            widget_name: Optional[str] = meta.get('widget_name', None)
            if widget_name is None:
                # Built-ins: map to the predicate names used in update_filtered_dataframes
                if df_col_name == 'custom_replay_name':
                    widget_predicate_name = 'replay_name'
                elif df_col_name == 'time_bin_size':
                    widget_predicate_name = 'time_bin_size'
                else:
                    widget_predicate_name = f'{df_col_name}_widget'
            else:
                widget_predicate_name = widget_name

            dims.append({
                'name': df_col_name,
                'df_col_name': df_col_name,
                'options': options,
                'active_value': a_widget.value,
                'is_multi': is_multi,
                'widget_predicate_name': widget_predicate_name,
                'always_applied': df_col_name in ('custom_replay_name', 'time_bin_size'),
            })
        ## END for a_widget in ordered_widgets...

        return dims


    @classmethod
    def _resolve_expand_fixed_vars(cls, expand_vars: Optional[Sequence[str]], fixed_vars: Optional[Sequence[str]], all_dim_names: Sequence[str]) -> Tuple[Set[str], Set[str]]:
        """Resolve which column dims branch vs stay pinned. Raises if both lists are provided or names are unknown."""
        all_dim_set: Set[str] = set(all_dim_names)
        has_expand = expand_vars is not None
        has_fixed = fixed_vars is not None
        if has_expand and has_fixed:
            raise ValueError('Pass only one of expand_vars or fixed_vars (not both).')

        def _validate_names(names: Sequence[str], arg_name: str) -> Set[str]:
            name_set = set(names)
            unknown = name_set - all_dim_set
            if len(unknown) > 0:
                raise ValueError(f'Unknown {arg_name} names: {sorted(unknown)}. Known dimensions: {list(all_dim_names)}')
            return name_set

        if has_expand:
            assert expand_vars is not None
            expand_set = _validate_names(list(expand_vars), 'expand_vars')
            fixed_set = all_dim_set - expand_set
        elif has_fixed:
            assert fixed_vars is not None
            fixed_set = _validate_names(list(fixed_vars), 'fixed_vars')
            expand_set = all_dim_set - fixed_set
        else:
            expand_set = set()
            fixed_set = set(all_dim_set)

        return expand_set, fixed_set


    @classmethod
    def _column_value_mask(cls, df: pd.DataFrame, df_col_name: str, value: Any, is_multi_active: bool = False) -> NDArray:
        """Boolean mask for rows matching a column filter value (widget-compatible str compare / isin)."""
        col_as_str = df[df_col_name].astype(str)
        if is_multi_active or isinstance(value, (list, tuple, set)):
            return col_as_str.isin([str(v) for v in value]).to_numpy()
        return (col_as_str == str(value)).to_numpy()


    def _build_filter_impact_flow_graph(self, df: pd.DataFrame, expand_set: Set[str], include_boolean_predicates: bool = True) -> Dict[str, Any]:
        """Walk filter dims with mask-based counts; returns nodes/links (and funnel stages when single-path).

        PURE w.r.t. filter widget / original_df_dict state (does not mutate live frames).
        """
        dims = self._iter_column_filter_dimensions()
        enabled_filter_predicate_list = list(getattr(self.active_filter_predicate_selector_widget, 'value', ()) or ())
        n_unfiltered: int = len(df)
        root_mask = np.ones(n_unfiltered, dtype=bool)

        node_labels: List[str] = [f'unfiltered ({n_unfiltered})']
        node_masks: List[NDArray] = [root_mask]
        link_sources: List[int] = []
        link_targets: List[int] = []
        link_values: List[int] = []
        # Funnel / active-path stages: only meaningful when expand_set is empty
        stage_labels: List[str] = [node_labels[0]]
        stage_counts: List[int] = [n_unfiltered]
        frontier: List[int] = [0]

        for dim in dims:
            # Custom column widgets only apply when their predicate is enabled; builtins always apply
            if (not dim.get('always_applied', False)) and (dim['widget_predicate_name'] not in enabled_filter_predicate_list):
                continue
            dim_name: str = dim['name']
            df_col_name: str = dim['df_col_name']
            is_expand: bool = dim_name in expand_set
            if is_expand:
                branch_values: List[Any] = list(dim['options'])
                use_multi_match = False
            else:
                branch_values = [dim['active_value']]
                use_multi_match = bool(dim['is_multi'])

            new_frontier: List[int] = []
            for parent_idx in frontier:
                parent_mask = node_masks[parent_idx]
                for v in branch_values:
                    child_mask = parent_mask & self._column_value_mask(df, df_col_name, v, is_multi_active=use_multi_match)
                    n_remaining: int = int(np.sum(child_mask))
                    if n_remaining == 0:
                        continue
                    if use_multi_match or isinstance(v, (list, tuple, set)):
                        value_label = ','.join([str(x) for x in v])
                    else:
                        value_label = str(v)
                    child_label = f'{dim_name}={value_label} ({n_remaining})'
                    child_idx = len(node_labels)
                    node_labels.append(child_label)
                    node_masks.append(child_mask)
                    link_sources.append(parent_idx)
                    link_targets.append(child_idx)
                    link_values.append(n_remaining)
                    new_frontier.append(child_idx)
                    if (not is_expand) and (len(expand_set) == 0):
                        stage_labels.append(child_label)
                        stage_counts.append(n_remaining)
                ## END for v in branch_values...
            ## END for parent_idx in frontier...

            frontier = new_frontier
        ## END for dim in dims...

        if include_boolean_predicates:
            widget_predicate_names = {d['widget_predicate_name'] for d in dims}
            for a_predicate_name, a_predicate_fn in self.additional_filter_predicates.items():
                if a_predicate_name not in enabled_filter_predicate_list:
                    continue
                if a_predicate_name in widget_predicate_names:
                    continue
                if a_predicate_name.endswith('_widget'):
                    continue

                new_frontier = []
                for parent_idx in frontier:
                    parent_mask = node_masks[parent_idx]
                    try:
                        is_predicate_true = np.asarray(a_predicate_fn(df)).astype(bool)
                    except Exception:
                        continue
                    child_mask = parent_mask & is_predicate_true
                    n_remaining = int(np.sum(child_mask))
                    if n_remaining == 0:
                        continue
                    child_label = f'{a_predicate_name} ({n_remaining})'
                    child_idx = len(node_labels)
                    node_labels.append(child_label)
                    node_masks.append(child_mask)
                    link_sources.append(parent_idx)
                    link_targets.append(child_idx)
                    link_values.append(n_remaining)
                    new_frontier.append(child_idx)
                    if len(expand_set) == 0:
                        stage_labels.append(child_label)
                        stage_counts.append(n_remaining)
                ## END for parent_idx in frontier...

                frontier = new_frontier
            ## END for a_predicate_name, a_predicate_fn in self.additional_filter_predicates.items()...

        return {
            'node_labels': node_labels,
            'link_sources': link_sources,
            'link_targets': link_targets,
            'link_values': link_values,
            'stage_labels': stage_labels,
            'stage_counts': stage_counts,
            'expand_set': expand_set,
            'n_unfiltered': n_unfiltered,
        }


    @function_attributes(short_name=None, tags=['filter', 'sankey', 'funnel', 'visualization'], input_requires=[], output_provides=[], uses=['_iter_column_filter_dimensions', '_resolve_expand_fixed_vars', '_build_filter_impact_flow_graph'], used_by=[], creation_date='2026-09-14 13:50', related_items=[])
    def build_filter_impact_flow_figure(self, df_name: Optional[str] = None, expand_vars: Optional[Sequence[str]] = None, fixed_vars: Optional[Sequence[str]] = None, chart: Literal['sankey', 'funnel'] = 'sankey', include_boolean_predicates: bool = True) -> go.Figure:
        """Build a Plotly Sankey (branching) or funnel (active path) of remaining row counts across filter dimensions.

        - expand_vars: those column dims branch over all options; others fixed to active widget values.
        - fixed_vars: those stay fixed; all other column dims expand.
        - neither: all column dims fixed (single active path).
        - both: ValueError.

        Usage:
            fig = df_filter.build_filter_impact_flow_figure(expand_vars=['trained_compute_epochs'], chart='sankey')
            fig.show()
            fig = df_filter.build_filter_impact_flow_figure(chart='funnel')
        """
        if df_name is None:
            df_name = self.active_plot_df_name.removeprefix('filtered_')
        df = self.original_df_dict.get(df_name, None)
        if df is None:
            raise KeyError(f'df_name="{df_name}" not in original_df_dict keys: {list(self.original_df_dict.keys())}')

        dims = self._iter_column_filter_dimensions()
        all_dim_names = [d['name'] for d in dims]
        expand_set, _fixed_set = self._resolve_expand_fixed_vars(expand_vars=expand_vars, fixed_vars=fixed_vars, all_dim_names=all_dim_names)

        if (chart == 'funnel') and (len(expand_set) > 0):
            raise ValueError(f"chart='funnel' requires an empty expand set (active path only). Got expand_set={sorted(expand_set)}. Use chart='sankey' or omit expand_vars / pass all dims via fixed_vars.")

        graph = self._build_filter_impact_flow_graph(df=df, expand_set=expand_set, include_boolean_predicates=include_boolean_predicates)

        if chart == 'funnel':
            fig = go.Figure(go.Funnel(y=graph['stage_labels'], x=graph['stage_counts'], textinfo='value+percent initial'))
            fig.update_layout(title=f'Filter impact funnel — {df_name}')
            return fig

        if chart != 'sankey':
            raise ValueError(f"chart must be 'sankey' or 'funnel', got {chart!r}")

        fig = go.Figure(data=[go.Sankey(
            node=dict(label=graph['node_labels'], pad=15, thickness=18),
            link=dict(source=graph['link_sources'], target=graph['link_targets'], value=graph['link_values']),
        )])
        fig.update_layout(title=f'Filter impact Sankey — {df_name}', font=dict(size=11))
        return fig


    def display(self):
        """Displays the widgets."""
        # Wrap Plotly widget in an Output widget for layout management
        # plotly_hover_output = widgets.Output()
        # with plotly_hover_output:
        #     display(self.hover_posterior_preview_figure_widget)
            
        # Arrange your widgets as needed
        out_list = [
            widgets.HBox([
                self.replay_name_widget, 
                self.time_bin_size_widget, 
                self.active_filter_predicate_selector_widget,
                # self.table_widget
            ], #layout=widgets.Layout(width='100%'),
            ),
            widgets.HBox([*self.custom_dynamic_filter_widgets_list], #layout=widgets.Layout(width='100%'),
            ),
            widgets.HBox([
                self.active_plot_df_name_selector_widget, 
                self.active_plot_variable_name_widget
            ]),
        ]

        if self.figure_widget is not None:
            out_list.append(self.figure_widget)
            out_list.append(widgets.HBox([self.button_copy, self.button_download, self.filename_label],
                        #   layout=widgets.Layout(width='100%'),
                          ))
        ## END if self.figure_widget is not None...

        if (self._hover_posterior_preview_box is None) and (self.hover_posterior_preview_figure_widget is None) and self.is_figure_widget_mode:
            self._setup_hover_posterior_preview_widget()
        preview_widget = self._hover_posterior_preview_box if self._hover_posterior_preview_box is not None else self.hover_posterior_preview_figure_widget
        table_box = None
        if self.table_widget is not None:
            table_box = widgets.Box([self.table_widget], layout=widgets.Layout(overflow='auto', height='320px', min_width='200px', width='auto'))
        bottom_row_children = [v for v in (table_box, preview_widget) if v is not None]
        n_bottom: int = len(bottom_row_children)
        if n_bottom == 2:
            bottom_row = widgets.GridBox(
                children=bottom_row_children,
                layout=widgets.Layout(width='100%', height='340px', grid_template_columns='minmax(220px, 1fr) 640px', grid_template_rows='340px', grid_gap='8px', align_items='stretch'),
            )
        else:
            bottom_row = widgets.HBox(bottom_row_children, layout=widgets.Layout(height='340px', width='100%', align_items='stretch'))
        out_list.extend([
            # self.js_output,  # Include the Output widget to allow the buttons to perform their actions
            bottom_row,
            widgets.HBox([self.output_widget, ],
                          layout=widgets.Layout(height='300px', width='100%'),
                          ),
        ])
        out = widgets.VBox(out_list)

        display(out)
        
        ## I'd like `self.table_widget` to fill up its minimal required width, and `self.hover_posterior_preview_figure_widget` to fill as much as needed but it should be aligned to the right margin. The figure will change size dynmically. Both elements should be the same height
        

    # ==================================================================================================================== #
    # Plot Updating Functions                                                                                              #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['performance', 'sampling'], input_requires=[], output_provides=[], uses=['_density_based_sampling'], used_by=[], creation_date='2025-01-27 14:00', related_items=[])
    def get_sampled_plot_data(self, max_points: int = 5000, sampling_method: str = 'random') -> pd.DataFrame:
        """
        Returns a downsampled version of the active plot dataframe for scatter plot rendering.

        Args:
            max_points: Maximum number of points to display in scatter plot
            sampling_method: 'random', 'systematic', or 'density_based'

        Returns:
            Sampled dataframe for scatter plot display
        """
        df = self.active_plot_df

        if len(df) <= max_points:
            return df  # No need to downsample

        if sampling_method == 'random':
            # Random sampling - preserves distribution
            return df.sample(n=max_points, random_state=42)

        elif sampling_method == 'systematic':
            # Systematic sampling - every nth point
            step = len(df) // max_points
            return df.iloc[::step].head(max_points)

        elif sampling_method == 'density_based':
            # More sophisticated: sample more points in sparse areas
            return self._density_based_sampling(df, max_points)

        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    @function_attributes(short_name=None, tags=['private'], input_requires=[], output_provides=[], uses=[], used_by=['get_sampled_plot_data'], creation_date='2025-07-03 17:31', related_items=[])
    def _density_based_sampling(self, df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        """Density-based sampling: more points in sparse areas, fewer in dense areas."""
        from sklearn.neighbors import NearestNeighbors

        # Use x,y coordinates for density estimation
        coords = df[['delta_aligned_start_t', self.active_plot_variable_name]].values

        # Find k-nearest neighbors to estimate density
        k = min(50, len(df) // 10)  # Adaptive k
        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        distances, _ = nbrs.kneighbors(coords)

        # Inverse density as sampling weight (sparse areas get higher weight)
        density_weights = 1.0 / (distances.mean(axis=1) + 1e-10)

        # Sample based on inverse density
        sample_probs = density_weights / density_weights.sum()
        sampled_indices = np.random.choice(
            len(df), 
            size=max_points, 
            replace=False, 
            p=sample_probs
        )
        return df.iloc[sampled_indices]


    @function_attributes(short_name=None, tags=['interactivity'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-03 17:31', related_items=[])
    def on_click_scatter_points(self, trace, points, selector):
        allow_single_selection_only: bool = False
        replace_selected_points: bool = True
        enable_debug_print: bool = False
        

        if points.point_inds:
            ## has selection:
            if (allow_single_selection_only or (len(points.point_inds) < 2)):
                ind = points.point_inds[0]
                num_new_selections = 1
            else:
                ind = points.point_inds # allows multiple selections
                num_new_selections = len(ind)
                
            self.output_widget.clear_output()
            with self.output_widget:
                print(f"Clicked point index: {ind}")
                session_name = self.active_plot_df['session_name'].iloc[ind]
                custom_replay_name = self.active_plot_df['custom_replay_name'].iloc[ind]
                time_bin_size = self.active_plot_df['time_bin_size'].iloc[ind]
                start_t = self.active_plot_df['start'].iloc[ind]
                stop_t = self.active_plot_df['stop'].iloc[ind]
                print(f'start_t, stop_t: {start_t}, {stop_t}')
                print(f"session_name: {session_name}")
                print(f"custom_replay_name: {custom_replay_name}")
                print(f"time_bin_size: {time_bin_size}")                        
                print(f'num_new_selections: {num_new_selections}')
                

            # with df_filter.output_widget:
            # print(f'num_new_selections: {num_new_selections}')
            if (num_new_selections == 1):
                selected_event_session_context: IdentifyingContext = IdentifyingContext(session_name=session_name, custom_replay_name=custom_replay_name, time_bin_size=time_bin_size)
                print(f'selected_event_session_context: {selected_event_session_context}')
                # selected_event_context: IdentifyingContext = IdentifyingContext(start_t=start_t, stop_t=stop_t)
                curr_selected_points_dict = self.selected_points.get(selected_event_session_context, None)
                if curr_selected_points_dict is None:
                    self.selected_points[selected_event_session_context] = [] # [start_t, ] ## add start_t only
                    
                if replace_selected_points:
                    self.selected_points[selected_event_session_context]
                else:
                    if start_t not in self.selected_points[selected_event_session_context]:
                        self.selected_points[selected_event_session_context].append(start_t) # add to the selection points list
                                        
            else:
                # greater than one selection
                for i, (a_session_name, a_custom_replay_name, a_time_bin_size, a_start_t, a_stop_t) in enumerate(zip(session_name, custom_replay_name, time_bin_size, start_t, stop_t)):
                    
                    a_selected_event_session_context: IdentifyingContext = IdentifyingContext(session_name=a_session_name, custom_replay_name=a_custom_replay_name, time_bin_size=a_time_bin_size)
                    print(f'a_selected_event_session_context: {a_selected_event_session_context}')
                    # a_selected_event_context: IdentifyingContext = IdentifyingContext(start_t=a_start_t, stop_t=a_stop_t)

                    curr_selected_points_dict = self.selected_points.get(a_selected_event_session_context, None)
                    if curr_selected_points_dict is None:
                        self.selected_points[a_selected_event_session_context] = [] # [start_t, ] ## add start_t only                                
                    # if replace_selected_points:
                    #     df_filter.selected_points[a_selected_event_session_context]
                    # else:
                    if a_start_t not in self.selected_points[a_selected_event_session_context]:
                        self.selected_points[a_selected_event_session_context].append(a_start_t) # add to the selection points list
                        
            
            # df_filter.output_widget.clear_output()
            # with df_filter.output_widget:
            #     print(f"Clicked point index: {ind}")
            #     print(f'start_t, stop_t: {start_t}, {stop_t}')
            #     print(f"session_name: {session_name}")
            #     print(f"custom_replay_name: {custom_replay_name}")
            #     print(f"time_bin_size: {time_bin_size}")
                
            ## update selected points
            

            # ## try to update the selected heatmap posterior:
            # try:
            #     ## try to update by start_t, stop_t
            #     _heatmap_data = self.hover_posterior_data.get_posterior_data(session_name=session_name, custom_replay_name=custom_replay_name,
            #                                                         a_decoder_name='long_LR', last_selected_idx=ind)
                
            #     # _plot_hoverred_heatmap_preview_posterior(df_filter=df_filter, last_selected_idx=ind) #TODO 2024-11-26 07:32: - [ ] does this work?
            #     _plot_hoverred_heatmap_preview_posterior(df_filter=self, a_heatmap_img=_heatmap_data)

            # except Exception as e:
            #     print(f'encountered exception when trying to call `_plot_hoverred_heatmap_preview_posterior(..., last_selected_idx={ind}, start_t: {start_t}, stop_t: {stop_t}). Error {e}. Skipping.')
                                    
        else:
            ## no selection:
            # print(f'NOPE! points: {points}, trace: {trace}')
            # df_filter.output_widget.clear_output()
            if enable_debug_print:
                with self.output_widget:
                    print(f'NOPE! points: {points}, trace: {trace}')
            

    def on_update_selected_scatter_points(self, trace, points, selector):
        """ 
        updates self.selected_points
        
        """
        self.output_widget.clear_output()
        with self.output_widget:
            print(f'self.on_update_selected_scatter_points(...):')
            print("\tSelected points:", points.point_inds)
            ind = points.point_inds # allows multiple selections
            num_new_selections = len(ind)

            # for i, (a_session_name, a_custom_replay_name, a_time_bin_size, a_start_t, a_stop_t) in enumerate(zip(session_name, custom_replay_name, time_bin_size, start_t, stop_t)):
            for i, ind in enumerate(points.point_inds):
                a_session_name = self.active_plot_df['session_name'].iloc[ind]
                a_custom_replay_name = self.active_plot_df['custom_replay_name'].iloc[ind]
                a_time_bin_size = self.active_plot_df['time_bin_size'].iloc[ind]
                a_start_t = self.active_plot_df['start'].iloc[ind]
                a_stop_t = self.active_plot_df['stop'].iloc[ind]
                                

                a_selected_event_session_context: IdentifyingContext = IdentifyingContext(session_name=a_session_name, custom_replay_name=a_custom_replay_name, time_bin_size=a_time_bin_size)
                # print(f'a_selected_event_session_context: {a_selected_event_session_context}')
                # a_selected_event_context: IdentifyingContext = IdentifyingContext(start_t=a_start_t, stop_t=a_stop_t)

                curr_selected_points_dict = self.selected_points.get(a_selected_event_session_context, None)
                if curr_selected_points_dict is None:
                    self.selected_points[a_selected_event_session_context] = [] # [start_t, ] ## add start_t only                                
                # if replace_selected_points:
                #     df_filter.selected_points[a_selected_event_session_context]
                # else:
                if a_start_t not in self.selected_points[a_selected_event_session_context]:
                    self.selected_points[a_selected_event_session_context].append(a_start_t) # add to the selection points list
            # end for i, ind in ..
            print(f'self.selected_points: {self.selected_points}')


    @staticmethod
    def _coerce_hover_customdata_value(value):
        """Unwrap 0-d numpy/plotly customdata cells to a python scalar."""
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value).ravel()
            if arr.size == 0:
                return None
            value = arr[0]
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                pass
        return value


    def _hover_payload_from_customdata(self, trace, point_idx: int) -> Optional[Dict[str, Any]]:
        if trace.customdata is None:
            return None
        customdata = np.asarray(trace.customdata)
        if customdata.ndim == 1:
            row_vals = [customdata[point_idx]] if point_idx < customdata.shape[0] else []
        else:
            if point_idx >= customdata.shape[0]:
                return None
            row_vals = list(customdata[point_idx])
        col_names = self._hover_posterior_customdata_cols or list(self._HOVER_POSTERIOR_CUSTOMDATA_COLS)
        payload = {}
        for i, col_name in enumerate(col_names):
            payload[col_name] = self._coerce_hover_customdata_value(row_vals[i] if i < len(row_vals) else None)
        ## END for i, col_name in enumerate(col_names)....

        return payload


    def _flush_hover_posterior_preview(self):
        payload = self._hover_posterior_pending_payload
        if (payload is None) or (self.hover_posterior_data is None):
            return
        session_name = payload.get('session_name')
        custom_replay_name = payload.get('custom_replay_name')
        if session_name is not None:
            session_name = str(session_name)
        if custom_replay_name is not None:
            custom_replay_name = str(custom_replay_name)
        format_name = payload.get('format_name')
        animal = payload.get('animal')
        exper_name = payload.get('exper_name')
        if format_name is not None:
            format_name = str(format_name)
        if animal is not None:
            animal = str(animal)
        if exper_name is not None:
            exper_name = str(exper_name)
        start_t = payload.get('start')
        time_bin_size = payload.get('time_bin_size')
        try:
            if time_bin_size is not None:
                time_bin_size = float(time_bin_size)
        except Exception:
            time_bin_size = None
        try:
            if start_t is not None:
                start_t = float(start_t)
        except Exception:
            start_t = None
        hover_key = (session_name, custom_replay_name, time_bin_size, start_t)
        if hover_key == self._last_hover_posterior_key:
            return
        plot_name: str = str(getattr(self, 'active_plot_df_name', '') or '').lower()
        epochs_name: str = 'laps' if ('laps' in plot_name) else 'ripple'
        images_by_decoder = self.hover_posterior_data.get_posterior_data(
            session_name=session_name, custom_replay_name=custom_replay_name,
            start_t=start_t, time_bin_size=time_bin_size,
            format_name=format_name, animal=animal, exper_name=exper_name,
            return_all_decoders=True, last_selected_idx=None, epochs_name=epochs_name,
        )
        if images_by_decoder is None:
            self.update_preview_heatmap(title=f'No posterior for {session_name} start={start_t}')
            return
        self._last_hover_posterior_key = hover_key
        title_tbin = f', tbin={time_bin_size}' if time_bin_size is not None else ''
        self.update_preview_heatmap(images_by_decoder, title=f'{session_name}  start={start_t}{title_tbin}')


    def _on_hover_scatter_posterior(self, trace, points, selector):
        """Plotly scatter hover: look up HDF5 posterior by customdata and update the bottom preview.

        Must run on the Jupyter kernel thread — FigureWidget updates from `threading.Timer` do not paint.
        """
        if (self.hover_posterior_data is None) or (not getattr(points, 'point_inds', None)):
            return
        payload = self._hover_payload_from_customdata(trace, points.point_inds[0])
        if payload is None:
            return
        self._hover_posterior_pending_payload = payload
        self._flush_hover_posterior_preview()


    def _attach_hover_posterior_handlers(self, fig):
        """Attach on_hover to scatter traces in the center subplot. Safe to call after each plot rebuild."""
        if (self.hover_posterior_data is None) or (fig is None):
            return
        if self.hover_posterior_preview_figure_widget is not None:
            self.hover_posterior_preview_figure_widget.layout.title.text = 'Hover an epoch to preview posterior'
        traces = list(fig.select_traces(row=1, col=2)) or list(fig.data)
        for trace in traces:
            ttype = getattr(trace, 'type', None) or ''
            name = getattr(trace, 'name', None) or ''
            if (ttype not in ('scatter', 'scattergl', '')) and (not str(name).startswith('trace_scatter')):
                continue
            if ttype == 'histogram':
                continue
            trace.on_hover(self._on_hover_scatter_posterior)
        ## END for trace in traces....


    @function_attributes(short_name=None, tags=['plotting'], input_requires=[], output_provides=[], uses=['_perform_plot_pre_post_delta_scatter'], used_by=[], creation_date='2024-11-20 13:08', related_items=[])
    @classmethod
    def _build_plot_callback(cls, earliest_delta_aligned_t_start, latest_delta_aligned_t_end, save_plotly, should_save: bool = False, should_prepare_full_hover_click_interactivity: bool = False, resolution_multiplier: float=1, enable_debug_print=False, additional_fig_layout_kwargs=None, **extra_plot_kwargs):
        """ 
        
        should_prepare_full_hover_click_interactivity: bool:  ## slow when enabled
        
        legend_groups_to_hide=[0.05]
        
        """
        is_publication_ready_figure: bool = extra_plot_kwargs.get('is_publication_ready_figure', True)
        # fig_size_kwargs = {'width': 1650, 'height': 480}
        # fig_size_kwargs = {'width': resolution_multiplier*1650, 'height': resolution_multiplier*480}
        ## set up figure size
        if not is_publication_ready_figure:
            fig_size_kwargs = {'width': (resolution_multiplier * 1800), 'height': (resolution_multiplier*480)}
            # fig_size_kwargs = {'width': (resolution_multiplier * 792), 'height': (resolution_multiplier*(792/1800)*480)}
            
        else:        
            # fig_size_kwargs = {'width': 474, 'height': 126}
            # fig_size_kwargs = {'width': (resolution_multiplier * 1080), 'height': resolution_multiplier*480}
            fig_size_kwargs = {'width': (resolution_multiplier * 792), 'height': (resolution_multiplier * 320)}
            # fig_size_kwargs = {'width': 792}  # 8.25 inches * 96 DPI
            # fig_size_kwargs = {'width': 2475}  # 8.25 inches * 300 DPI


        is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=False, is_publication=True)
        pio.templates.default = template

        ## INPUTS: earliest_delta_aligned_t_start, latest_delta_aligned_t_end
        # should_save: bool = True        

        _new_perform_plot_pre_post_delta_scatter = partial(
            _helper_perform_plot_pre_post_delta_scatter,
            time_delta_tuple=(earliest_delta_aligned_t_start, 0.0, latest_delta_aligned_t_end),
            fig_size_kwargs=fig_size_kwargs, additional_fig_layout_kwargs=additional_fig_layout_kwargs,
            is_dark_mode=is_dark_mode,
            save_plotly=save_plotly,
            figure_sup_huge_title_text=extra_plot_kwargs.pop('figure_sup_huge_title_text', None),
        )

        _new_perform_plot_pre_post_delta_scatter_with_embedded_context = partial(
            _new_perform_plot_pre_post_delta_scatter,
            data_context=None,
        )
        
        extra_plot_kwargs = deepcopy(extra_plot_kwargs)


        def _build_filter_changed_plotly_plotting_callback_fn(df_filter: "DataFrameFilter", should_save:bool=False, **kwargs):
            """ `filtered_all_sessions_all_scores_ripple_df` versions -
            captures: _new_perform_plot_pre_post_delta_scatter_with_embedded_context, should_prepare_full_hover_click_interactivity, _perform_plot_pre_post_delta_scatter_with_embedded_context, should_save, extra_plot_kwargs
            
            """
            # df_filter.output_widget.clear_output(wait=True)
            from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import PlotlyFigureContainer
            if df_filter.hover_posterior_data is not None and (not isinstance(df_filter.hover_posterior_data, PosteriorPlottingDatasource)):
                df_filter.hover_posterior_data = PosteriorPlottingDatasource.from_loaded_containers(df_filter.hover_posterior_data)
            if df_filter.hover_posterior_data is not None and df_filter.hover_posterior_data.plot_heatmap_fn is None:
                df_filter.hover_posterior_data.plot_heatmap_fn = (lambda a_df_filter, images, *args, **kwargs: a_df_filter.update_preview_heatmap(images))

            active_plot_df_name: str = df_filter.active_plot_df_name
            active_plot_df: pd.DataFrame = df_filter.active_plot_df
            plot_variable_name: str = df_filter.active_plot_variable_name
            n_filtered: int = len(active_plot_df)

            ## Keep center subplot title in sync with the Plot df Name selector (do not keep construction-time main_title)
            active_main_title: str = active_plot_df_name.removeprefix('filtered_')
            try:
                data_context = active_plot_df.attrs.get('data_context', None)
                if data_context is not None:
                    title_prefix: str = data_context.get_description(subset_includelist=['title_prefix'])
                    if title_prefix:
                        active_main_title = title_prefix
            except Exception:
                pass

            ## Empty-filter display mode: show a centered label instead of attempting a broken scatter redraw
            if df_filter.is_figure_widget_mode and (n_filtered == 0):
                original_name: str = active_plot_df_name.removeprefix('filtered_')
                n_total: int = len(df_filter.original_df_dict.get(original_name, active_plot_df))
                fig = df_filter.figure_widget
                # Mirror plotly_pre_post_delta_scatter reuse clear (plotly_helpers.py ~635-641):
                fig.layout.annotations = fig.layout.annotations[:3]  # keep subplot titles
                fig.layout.shapes = []
                fig.data = []
                PlotlyFigureContainer.update_subplot_title(fig, row=1, col=2, new_title=active_main_title)
                fig.add_annotation(
                    text=f"<Filter excludes all {n_total}>",
                    xref="x2", yref="y2", x=0.5, y=0.5,
                    xanchor="center", yanchor="middle",
                    showarrow=False,
                    name="empty_filter_excludes_all_annotation",
                )
                df_filter.figure_widget = fig
            elif df_filter.is_figure_widget_mode:
                assert plot_variable_name in active_plot_df.columns, f"plot_variable_name: '{plot_variable_name}' is not present in active_plot_df.columns! Cannot plot!"

                if len(df_filter.time_bin_size) > 2:
                    non_selected_options = df_filter.time_bin_size[1:] # all but the first
                    legend_groups_to_hide = (deepcopy(non_selected_options))
                else:
                    legend_groups_to_hide = [] # Hide none. df_filter.time_bin_size # [0.05]
                    
                #TODO 2025-04-09 14:00: - [ ] Customization here -- which are enabled by default            

                # extra_plot_kwargs = deepcopy(extra_plot_kwargs)
                # active_plot_kwargs = extra_plot_kwargs | {'legend_groups_to_hide': legend_groups_to_hide}
                # active_plot_kwargs = active_plot_kwargs | kwargs
                active_plot_kwargs = (extra_plot_kwargs | {'legend_groups_to_hide': legend_groups_to_hide, 'main_title': active_main_title} | kwargs)
                plot_df: pd.DataFrame = deepcopy(active_plot_df)
                enable_hover_posterior: bool = df_filter.hover_posterior_data is not None
                if enable_hover_posterior:
                    if 'start' not in plot_df.columns:
                        start_src = next((c for c in df_filter._HOVER_POSTERIOR_START_COL_CANDIDATES if c in plot_df.columns), None)
                        if start_src is not None:
                            plot_df['start'] = plot_df[start_src]
                    present_custom_cols = [c for c in df_filter._HOVER_POSTERIOR_CUSTOMDATA_COLS if c in plot_df.columns]
                    df_filter._hover_posterior_customdata_cols = present_custom_cols
                    if len(present_custom_cols) > 0:
                        for col_name in present_custom_cols:
                            if str(plot_df[col_name].dtype) == 'category':
                                plot_df[col_name] = plot_df[col_name].astype(object)
                        ## END for col_name in present_custom_cols....

                        px_scatter_kwargs = dict(active_plot_kwargs.get('px_scatter_kwargs') or {})
                        px_scatter_kwargs['custom_data'] = present_custom_cols
                        active_plot_kwargs['px_scatter_kwargs'] = px_scatter_kwargs
                fig, new_fig_context, _extras_output_dict, figure_out_paths = _new_perform_plot_pre_post_delta_scatter_with_embedded_context(concatenated_ripple_df=plot_df, is_dark_mode=False, should_save=should_save, extant_figure=df_filter.figure_widget,
                                                                                                                                        variable_name=plot_variable_name, **active_plot_kwargs) # , enable_custom_widget_buttons=True

                if enable_hover_posterior:
                    fig = fig.update_layout(hovermode='closest')
                    df_filter._attach_hover_posterior_handlers(fig)

                # ==================================================================================================================== #
                # Hover/Click Interactivity (Slow)                                                                                     #
                # ==================================================================================================================== #
                if should_prepare_full_hover_click_interactivity:
                    fig = fig.update_layout(clickmode="event+select", hovermode='closest', dragmode='select')
                    if not enable_hover_posterior:
                        hover_cols = [c for c in ["session_name", "custom_replay_name", "time_bin_size", "start", "duration"] if c in active_plot_df.columns]
                        fig.update_traces(
                            hovertemplate="<b>sess:</b> %{customdata[0]} | <b>replay_name:</b> %{customdata[1]} | <b>time_bin_size:</b> %{customdata[2]}<br>"
                                        "<b>start:</b> %{customdata[3]}<br>",
                            customdata=active_plot_df[hover_cols].values if len(hover_cols) > 0 else None,
                            hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.4)", font=dict(color="black")),
                        )


                    def on_click(trace, points, selector):
                        if points.point_inds:
                            df_filter.on_click_scatter_points(trace=trace, points=points, selector=selector) ## pass through call to `df_filter.on_click_scatter_points(...)`
                            df_filter._on_hover_scatter_posterior(trace=trace, points=points, selector=selector)
                        else:
                            if enable_debug_print:
                                with df_filter.output_widget:
                                    print(f'NOPE! points: {points}, trace: {trace}')
                    ## END def on_click(t...     

                    fig.layout.hovermode = 'closest'
                    if len(fig.data) > 0:
                        ## prevent IndexError: tuple index out of range
                        fig.data[0].on_click(on_click)
                        fig.data[0].on_selection(df_filter.on_update_selected_scatter_points)


                    scatter_traces = list(fig.select_traces(selector=None, row=1, col=2))
                    for trace in scatter_traces:
                        trace.on_click(on_click)
                        trace.on_selection(df_filter.on_update_selected_scatter_points)
                    ## END for trace in scatter_traces....

                ## END if should_prepare_hover_tooltips...

                if fig is not None:
                    df_filter.figure_widget = fig

            ## END if df_filter.is_figure_widget_mode...
            df_filter.figure_widget.update_xaxes(col=2, range=[earliest_delta_aligned_t_start, latest_delta_aligned_t_end]) ## surprisingly this does fix it!


        ## end def _build_filter_changed_plotly_plotting_callback_fn(...)
        
        return _build_filter_changed_plotly_plotting_callback_fn


    # ==================================================================================================================== #
    # Data Update Functions                                                                                                #
    # ==================================================================================================================== #
    
    def _init_filtered_dataframes(self, enable_overwrite_is_filter_included_column: bool=True):
        """ builds the filtered dataframes from the original_df_dict. Initially they are unfiltered. """
        ## Update the 'is_filter_included' column on the original dataframes
        for name, df in self.original_df_dict.items():
            if enable_overwrite_is_filter_included_column or ('is_filter_included' not in df.columns):
                df['is_filter_included'] = True  # Initialize with default value
            filtered_name: str = f"filtered_{name}"
            filtered_df = deepcopy(df[df['is_filter_included']])
            # setattr(self, filtered_name, filtered_df) # instance-attributes method
            self._filtered_df_dict[filtered_name] = filtered_df # instance-dict method


    def _build_single_predicates_row_changed_df(self, a_dict) -> pd.DataFrame:
        """ 
        builds counts for debugging the effect of applying each predicate sequentially
        Called by: `self._build_merged_predicates_row_changed_df()`

        PURE: (does not modify self)

        Usage:

            predicate_operations_df_dict: Dict[str, pd.DataFrame] = {a_filtered_df_name:df_filter._build_predicates_row_changed_df(a_dict=a_dict) for a_filtered_df_name, a_dict in df_filter.step_by_step_predicate_filtered_row_counts_dict.items()}
            predicate_operations_df_dict

        """
        cummulative_only = {k:v for k, v in a_dict.items() if k.startswith('CUM_')}
        # META_only = {k:v for k, v in a_dict.items() if k.startswith('META_')}
        individual_predicates_only = {k:v for k, v in a_dict.items() if ((not k.startswith('META_')) and (not k.startswith('CUM_')))}

        individual_predicates_only_df: pd.DataFrame = pd.DataFrame({k:[v] for k, v in individual_predicates_only.items()}).T
        individual_predicates_only_df.columns = ['n_predicate_true_rows']
        # individual_predicates_only_df

        n_unfiltered: int = a_dict['META_n_unfiltered']
        n_post_filtered: int = a_dict['META_post_filters']

        (n_unfiltered, n_post_filtered)
        ## OUTPUTS: n_unfiltered, n_post_filtered


        individual_predicates_only_df['n_remaining_rows'] = cummulative_only.values()
        cum_num_rows = [n_unfiltered, *individual_predicates_only_df['n_remaining_rows'].to_numpy().tolist()] # .cumsum()
        cum_num_rows # [245837, 22846, 18632, 18632, 5228]
        assert cum_num_rows[-1] == n_post_filtered, f"cum_num_rows[-1]: {cum_num_rows[-1]} should equal n_post_filtered: {n_post_filtered}, but it does not!"
        cum_num_rows_diff = np.diff(cum_num_rows) ## number of rows each predicate was responsible for removing: array([-222991,   -4214,       0,  -13404])
        cum_num_rows_diff
        assert len(cum_num_rows_diff) == len(individual_predicates_only_df)
        individual_predicates_only_df['n_pred_filtered_rows'] = cum_num_rows_diff
        # cummulative_only_df
        individual_predicates_only_df['n_cum_filtered_rows'] = individual_predicates_only_df['n_pred_filtered_rows'].cumsum()

        return individual_predicates_only_df


    def _build_merged_predicates_row_changed_df(self) -> pd.DataFrame:
        """ 
        builds counts for debugging the effect of applying each predicate sequentially
        Calls `self._build_single_predicates_row_changed_df(...)` for each df type to filter before building a merged dataframe.
        Called By: `self.update_filtered_dataframes(...)`

        PURE: (does not modify self)

        Uses:
            `self.step_by_step_predicate_filtered_row_counts_dict`

        Usage:
            merged_predicate_operations_df: pd.DataFrame = df_filter._build_merged_predicates_row_changed_df()
            merged_predicate_operations_df

        """

        predicate_operations_df_dict: Dict[str, pd.DataFrame] = {a_filtered_df_name:self._build_single_predicates_row_changed_df(a_dict=a_dict) for a_filtered_df_name, a_dict in self.step_by_step_predicate_filtered_row_counts_dict.items()}

        self.active_plot_df_name = self.active_plot_df_name_selector_widget.value
        
        merged_predicate_operations_df: pd.DataFrame = predicate_operations_df_dict.get(self.active_plot_df_name, None)
        assert (merged_predicate_operations_df is not None), f'self.active_plot_df_name: "{self.active_plot_df_name}" is not in predicate_operations_df_dict, predicate_operations_df_dict.keys(): {list(predicate_operations_df_dict.keys())}. Have to build weird merged dict!'

        # if (merged_predicate_operations_df is None):
        #     print(f'ERROR: self.active_plot_df_name: "{self.active_plot_df_name}" is not in predicate_operations_df_dict, predicate_operations_df_dict.keys(): {list(predicate_operations_df_dict.keys())}. Have to build weird merged dict!')
        #     ## INPUTS: predicate_operations_df_dict
        #     # print(list(predicate_operations_df_dict.keys())) # ['filtered_all_sessions_ripple_time_bin_df', 'filtered_all_sessions_laps_time_bin_df']
        #     # [k.removeprefix('filtered_all_sessions_').removesuffix('_time_bin_df') for k in list(predicate_operations_df_dict.keys())] # ['ripple', 'laps']
        #     predicate_operations_df_keys_to_shortkeys_dict: Dict[str, str] = {k:k.removeprefix('filtered_all_sessions_').removesuffix('_time_bin_df') for k in list(predicate_operations_df_dict.keys())} # ['ripple', 'laps']
        #     predicate_operations_df_keys_to_shortkeys_dict

        #     merged_predicate_operations_df = []
        #     # predicate_operations_df = pd.concat([df for name, df in predicate_operations_df_dict.items()])
        #     # predicate_operations_df

        #     for a_df_name, df in predicate_operations_df_dict.items():
        #         a_short_key: str = predicate_operations_df_keys_to_shortkeys_dict[a_df_name]
        #         for a_row in df.itertuples():
        #             # replacement_idx: str = f"{name}_{a_row[0]}" # "filtered_all_sessions_ripple_time_bin_df_time_bin_size"
        #             replacement_idx: str = f"{a_short_key}|{a_row[0]}" # "ripple_time_bin_size"
        #             merged_predicate_operations_df.append((replacement_idx, *a_row[1:]))

        #     merged_predicate_operations_df: pd.DataFrame = pd.DataFrame.from_records(merged_predicate_operations_df, columns=['predicate_name', 'n_predicate_true_rows', 'n_remaining_rows', 'n_pred_filtered_rows', 'n_cum_filtered_rows'])
        ## OUTPUTS: merged_predicate_operations_df

        return merged_predicate_operations_df





    @function_attributes(short_name=None, tags=['update', 'MAIN', 'callback'], input_requires=['self.additional_filter_predicates'], output_provides=[], uses=[], used_by=['self._on_widget_change'], creation_date='2025-03-27 12:49', related_items=[])
    def update_filtered_dataframes(self, replay_name, time_bin_sizes=None, debug_print=True, enable_overwrite_is_filter_included_column: bool=True):
        """ Perform filtering on each DataFrame. Called after debounce delay scheduled by `self._on_widget_change` when a widget's value is changed
        
        
        Uses: `self.additional_filter_predicates`, .original_df_dict, 
            `self.on_filtered_dataframes_changed_callback_fns`

        Updates: `self.step_by_step_predicate_filtered_row_counts_dict`

        """
        # if time_bin_sizes is None:
        #     print("WARN: time_bin_sizes is None, falling back to widget's values.")
        #     time_bin_sizes = self.time_bin_size
            
        if not time_bin_sizes:
            print("Please select at least one Time Bin Size.")
            return

        with self._update_lock:
            self.output_widget.clear_output()
            with self.output_widget:
                # Convert time_bin_sizes to a list if it's not already
                if isinstance(time_bin_sizes, (float, int)):
                    time_bin_sizes = [time_bin_sizes]
                elif isinstance(time_bin_sizes, tuple):
                    time_bin_sizes = list(time_bin_sizes)


                enabled_filter_predicate_list = self.active_filter_predicate_selector_widget.value
                did_applying_predicate_fail_for_df_dict = {}
                new_step_by_step: Dict[str, Dict[str, int]] = {}  # local accumulator; publish once after the loop


                ## Update the 'is_filter_included' column on the original dataframes
                for name, df in self.original_df_dict.items():
                    filtered_name: str = f"filtered_{name}"
                    did_applying_predicate_fail_for_df_dict[filtered_name] = False ## start false

                    counts: Dict[str, int] = {'META_n_unfiltered': len(df)} ## number of rows prior to any filtering

                    if enable_overwrite_is_filter_included_column or ('is_filter_included' not in df.columns):
                        df['is_filter_included'] = True  # Initialize with default value

                    # Update based on conditions
                    # df['is_filter_included'] = (df['custom_replay_name'] == replay_name) & (df['time_bin_size'].isin(time_bin_sizes))

                    # df['is_filter_included'] = (df['custom_replay_name'] == replay_name) & (df['time_bin_size'].isin(time_bin_sizes))
                    # # self.step_by_step_predicate_filtered_row_counts_dict[filtered_name].update({'replay_name': len(df[(df['custom_replay_name'] == replay_name)])})
                    # df['is_filter_included'] = np.logical_and(df['is_filter_included'], is_predicate_true)
                    # self.step_by_step_predicate_filtered_row_counts_dict[filtered_name].update({'time_bin_size': len(df[(df['time_bin_size'].isin(time_bin_sizes))])})

                    a_predicate_name = 'replay_name'
                    is_predicate_true = (df['custom_replay_name'] == replay_name)
                    df['is_filter_included'] = np.logical_and(df['is_filter_included'], is_predicate_true)
                    counts[a_predicate_name] = int(np.sum(is_predicate_true)) ## just predicate alone
                    counts[f"CUM_{a_predicate_name}"] = int(df['is_filter_included'].sum()) ## CUM (cummulative)


                    a_predicate_name = 'time_bin_size'
                    is_predicate_true = (df['time_bin_size'].isin(time_bin_sizes))
                    df['is_filter_included'] = np.logical_and(df['is_filter_included'], is_predicate_true)
                    counts[a_predicate_name] = int(np.sum(is_predicate_true)) ## just predicate alone
                    counts[f"CUM_{a_predicate_name}"] = int(df['is_filter_included'].sum()) ## CUM (cummulative)


                    # self.step_by_step_predicate_filtered_row_counts_dict[filtered_name].update({'replay_name_AND_time_bin_size': len(df[df['is_filter_included']])})
                    # self.step_by_step_predicate_filtered_row_counts_dict[filtered_name].update({f"CUM_replay_name_AND_time_bin_size": len(df[df['is_filter_included']])}) ## CUM (cummulative)

                    ## Apply predicates
                    active_predicate_filter_name_modifier = []

                    for a_predicate_name, a_predicate_fn in self.additional_filter_predicates.items():
                        if a_predicate_name in enabled_filter_predicate_list:
                            did_predicate_fail: bool = False # whether an error was encountered when trying to evaluate this predicate for this df
                            try:
                                is_predicate_true = a_predicate_fn(df)
                                did_predicate_fail = False
                            except KeyError as e:
                                if debug_print:
                                    print(f'NOTE: failed to apply predicate "{a_predicate_name}" to df: {name}')                            
                                is_predicate_true = False
                                did_predicate_fail = True
                            except Exception as e:
                                did_predicate_fail = True
                                raise
                            if did_predicate_fail:
                                did_applying_predicate_fail_for_df_dict[filtered_name] = (did_applying_predicate_fail_for_df_dict[filtered_name] or did_predicate_fail)

                            df['is_filter_included'] = np.logical_and(df['is_filter_included'], is_predicate_true)
                            if not did_predicate_fail:
                                active_predicate_filter_name_modifier.append(a_predicate_name)

                                # step_by_step_predicate_filtered_row_counts_dict[filtered_name].update({a_predicate_name: len(df[df['is_filter_included']])})
                                counts[a_predicate_name] = int(np.sum(is_predicate_true)) ## just predicate alone
                                counts[f"CUM_{a_predicate_name}"] = int(df['is_filter_included'].sum()) ## CUM (cummulative)


                                # {a_filtered_df_name:{k:v for k, v in a_dict.items() if k.starts_with('CUM_')} for a_filtered_df_name, a_dict in self.step_by_step_predicate_filtered_row_counts_dict.items()}


                    ## END for a_predicate_name, a_predicate_fn in self.additional_filter_predicates.items()...

                    filtered_df = deepcopy(df[df['is_filter_included']])
                    counts['META_post_filters'] = len(filtered_df)
                    ## Individual predicates have no suffix, while cummulative (sequentially applied) predicate filters have a "CUM_" prefix, while overall/high-level total filters have a "META_" prefix.


                    ## update df metadata to indicate that it is filtered
                    df_context_dict = filtered_df.attrs['data_context'].to_dict() #  ## benedict dict
                    df_context_dict.pop('filter', None) ## remove old filter
                    if len(active_predicate_filter_name_modifier) > 0:
                        active_predicate_filter_name_modifier = '_'.join(active_predicate_filter_name_modifier) ## build string
                        df_context_dict['filter'] = active_predicate_filter_name_modifier
                    filtered_df.attrs['data_context'] = IdentifyingContext.init_from_dict(df_context_dict) ## update
                    filtered_df.attrs['did_filter_predicate_fail'] = did_applying_predicate_fail_for_df_dict[filtered_name]
                    # setattr(self, filtered_name, filtered_df) # instance-attributes method
                    self._filtered_df_dict[filtered_name] = filtered_df # instance-dict method
                    new_step_by_step[filtered_name] = counts

                ## END for name, df in self.original_df_dict.items()...

                # single publish — no mid-loop self.step_by_step...[...] lookups
                self.step_by_step_predicate_filtered_row_counts_dict = new_step_by_step

                ## Update sizes table:
                merged_predicate_operations_df = self._build_merged_predicates_row_changed_df()
                self.table_widget.data = merged_predicate_operations_df ## since self.filtered_size_info_df cannot be updated
                ## resize table widget's columns to auto-fit
                # self.table_widget.auto_fit_columns = False
                # self.table_widget.auto_fit_columns = True
                self._update_table_widget_column_widths()


                if did_applying_predicate_fail_for_df_dict[self.active_plot_df_name]:
                    print(f'!!! Warning!!! applying predicates failed for the current active plot df (self.active_plot_df_name: {self.active_plot_df_name})!\n\tthe plotted output has NOT been filtered!')
            ## END with self.output_widget
        ## END with self._update_lock
        
        for k, a_callback_fn in self.on_filtered_dataframes_changed_callback_fns.items():
            # print(f'k: {k}')
            try:
                a_callback_fn(self)
            except Exception as e:
                print(f'WARNING: callback_fn[{k}] failed with error: {e}, skipping.')
                raise
            
        ## END for k, a_callback_fn in self.on_filtered_dataframes_changed_callback_fns.items()...

        ## Update the preferred_filename from the dataframe metadata:
        self.on_widget_update_filename()

    

    def _cancel_pending_debounce(self):
        """Cancel any pending debounce timer so a sync update does not race a Timer callback."""
        if self.debounce_timer is not None:
            self.debounce_timer.cancel()
            self.debounce_timer = None


    def update_filters(self):
        self._cancel_pending_debounce()
        self.update_filtered_dataframes(replay_name=self.replay_name, time_bin_sizes=self.time_bin_size)


    def update_calling_namespace_locals(self):
        """ dangerous!! Updates the calling namespace (such as a jupyter notebook cell) """
        # Update the variables in the notebook's user namespace
        ipython = get_ipython()
        user_ns = ipython.user_ns

        for k, a_df in self.filtered_df_dict.items():
            user_ns[k] = a_df

        # # Update ripple DataFrames
        # user_ns['filtered_all_sessions_ripple_df'] = self.filtered_all_sessions_ripple_df
        # user_ns['filtered_all_sessions_ripple_time_bin_df'] = self.filtered_all_sessions_ripple_time_bin_df
        # user_ns['filtered_all_sessions_MultiMeasure_ripple_df'] = self.filtered_all_sessions_MultiMeasure_ripple_df
        # user_ns['filtered_all_sessions_all_scores_ripple_df'] = self.filtered_all_sessions_all_scores_ripple_df
        
        # # Update laps DataFrames
        # user_ns['filtered_all_sessions_laps_df'] = self.filtered_all_sessions_laps_df
        # user_ns['filtered_all_sessions_laps_time_bin_df'] = self.filtered_all_sessions_laps_time_bin_df
        # user_ns['filtered_all_sessions_MultiMeasure_laps_df'] = self.filtered_all_sessions_MultiMeasure_laps_df

    # Update instance values from a dictionary
    def update_instance_from_dict(self, update_dict):
        # Filter the dictionary to match only fields in the class
        field_names = {field.name for field in self.__attrs_attrs__}
        filtered_dict = {k: v for k, v in update_dict.items() if k in field_names}
        for k, v in filtered_dict.items():
            # setattr(self, k, v) # instance-attributes method
            self._filtered_df_dict[k] = v # instance-dict method
            
        # return evolve(self, **filtered_dict)


    # Accessor methods for the filtered DataFrames _______________________________________________________________________ #
    # # Accessor methods for ripple DataFrames
    # def get_filtered_all_sessions_ripple_df(self):
    #     return self.filtered_all_sessions_ripple_df
    
    # def get_filtered_all_sessions_ripple_time_bin_df(self):
    #     return self.filtered_all_sessions_ripple_time_bin_df
    
    # def get_filtered_all_sessions_MultiMeasure_ripple_df(self):
    #     return self.filtered_all_sessions_MultiMeasure_ripple_df
    
    # def get_filtered_all_sessions_all_scores_ripple_df(self):
    #     return self.filtered_all_sessions_all_scores_ripple_df
    
    # # Accessor methods for laps DataFrames
    # def get_filtered_all_sessions_laps_df(self):
    #     return self.filtered_all_sessions_laps_df
    
    # def get_filtered_all_sessions_laps_time_bin_df(self):
    #     return self.filtered_all_sessions_laps_time_bin_df
    
    # def get_filtered_all_sessions_MultiMeasure_laps_df(self):
    #     return self.filtered_all_sessions_MultiMeasure_laps_df


    @classmethod
    def safe_boolean_predicate_wrapper(cls, predicate_fn, df):
        """ returns False if predicate can't be evaluated"""
        try:
            return predicate_fn(df)
        except Exception as e:
            # raise e
            print(f'failed to apply predicate to df')
            return False


    @function_attributes(short_name=None, tags=['filtered_df', 'programmatic', 'df'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-09-21 12:54', related_items=[])
    def get_specific_filtered_df(self, a_name: str, debug_print: bool=False) -> Optional[pd.DataFrame]:
        """ returns the current filtered_df dataframe corresponding to the variable name `a_name`.

            concatenated_ripple_df =_subfn_get_filtered_df_variable(df_filter, a_name='filtered_all_sessions_laps_df')


            concatenated_ripple_df = df_filter.get_specific_filtered_df(a_name='filtered_all_sessions_laps_df')


        HISTORY: renamed from `_subfn_get_filtered_df_variable`

        """
        if getattr(self, a_name, None) is not None:
            return getattr(self, a_name)
        elif (self.filtered_df_dict.get(a_name, None) is not None):
            return self.filtered_df_dict[a_name]
        else:
            if debug_print:
                print(f'WARNING: could not find "{a_name}" in df_filter using either method.')
            return None ## return None, not found
    


    # ==================================================================================================================== #
    # Serialization                                                                                                        #
    # ==================================================================================================================== #
    #TODO 2024-11-20 08:26: - [ ] These were written by ChatGPT and I'm using them instead of my normal HDFSerialization stuff.    

    # Class variables for attribute names
    DATAFRAME_ATTR_NAMES = [
        'all_sessions_ripple_df',
        'all_sessions_ripple_time_bin_df',
        'all_sessions_MultiMeasure_ripple_df',
        'all_sessions_all_scores_ripple_df',
        'all_sessions_laps_df',
        'all_sessions_laps_time_bin_df',
        'all_sessions_MultiMeasure_laps_df',
        'filtered_all_sessions_ripple_df',
        'filtered_all_sessions_ripple_time_bin_df',
        'filtered_all_sessions_MultiMeasure_ripple_df',
        'filtered_all_sessions_all_scores_ripple_df',
        'filtered_all_sessions_laps_df',
        'filtered_all_sessions_laps_time_bin_df',
        'filtered_all_sessions_MultiMeasure_laps_df'
    ]

    SCALAR_ATTR_NAMES = [
        'active_plot_df_name',
        'active_plot_variable_name',
        'active_plot_fn_kwargs',
        'filename'
    ]


    def save_to_hdf(self, filename):
        """Saves the data (excluding widgets) to a single .hdf file."""
        import json
        with pd.HDFStore(filename, 'w') as store:
            # Save dataframes
            ## DATAFRAME_ATTR_NAMES
            DATAFRAME_ATTR_NAMES = self.original_df_names + self.filtered_df_names  # for 2025-05-02 - self._original_df_dict and self._filtered_df_dict mode

            # for attr_name in self.DATAFRAME_ATTR_NAMES:
            for attr_name in DATAFRAME_ATTR_NAMES:            
                df = getattr(self, attr_name, None)
                if df is not None:
                    store.put(attr_name, df)
            # Save scalar attributes
            scalar_attrs = {}
            for attr_name in self.SCALAR_ATTR_NAMES:
                attr_value = getattr(self, attr_name)
                try:
                    json.dumps(attr_value)
                    scalar_attrs[attr_name] = attr_value
                except (TypeError, ValueError):
                    # Not serializable, skip or set to None
                    scalar_attrs[attr_name] = None

            # Serialize scalar_attrs to JSON and store in root attributes
            store.get_storer('all_sessions_ripple_df').attrs.scalar_attrs = json.dumps(scalar_attrs)
            # Save additional_filter_predicates keys (since we can't save functions)
            store.get_storer('all_sessions_ripple_df').attrs.additional_filter_predicates_keys = list(self.additional_filter_predicates.keys())


    @classmethod
    def load_from_hdf(cls, filename) -> "DataFrameFilter":
        """Loads data from a .hdf file and returns a new instance.
        
        WARN: doesn't get all the exported DFs beyond the defaults (not addapted to the 2025-05-02 - self._original_df_dict and self._filtered_df_dict mode format)
        """
        import json

        with pd.HDFStore(filename, 'r') as store:
            # Load dataframes
            dataframes = {}
            for attr_name in cls.DATAFRAME_ATTR_NAMES:
                if attr_name in store:
                    dataframes[attr_name] = store[attr_name]
                else:
                    dataframes[attr_name] = None

            # Load scalar attributes from the attributes of the first dataframe
            attrs = store.get_storer('all_sessions_ripple_df').attrs
            scalar_attrs = json.loads(attrs.scalar_attrs)

            # Create a new instance with loaded data
            # instance = cls(
            #     all_sessions_ripple_df=dataframes.get('all_sessions_ripple_df', None),
            #     all_sessions_ripple_time_bin_df=dataframes.get('all_sessions_ripple_time_bin_df', None),
            #     all_sessions_MultiMeasure_ripple_df=dataframes.get('all_sessions_MultiMeasure_ripple_df', None),
            #     all_sessions_all_scores_ripple_df=dataframes.get('all_sessions_all_scores_ripple_df', None),
            #     all_sessions_laps_df=dataframes.get('all_sessions_laps_df', None),
            #     all_sessions_laps_time_bin_df=dataframes.get('all_sessions_laps_time_bin_df', None),
            #     all_sessions_MultiMeasure_laps_df=dataframes.get('all_sessions_MultiMeasure_laps_df', None),
            #     additional_filter_predicates={}  # Functions can't be serialized
            # )


            instance = cls(
                original_df_dict=dict(
                    all_sessions_ripple_df=dataframes.get('all_sessions_ripple_df', None),
                    all_sessions_ripple_time_bin_df=dataframes.get('all_sessions_ripple_time_bin_df', None),
                    all_sessions_MultiMeasure_ripple_df=dataframes.get('all_sessions_MultiMeasure_ripple_df', None),
                    all_sessions_all_scores_ripple_df=dataframes.get('all_sessions_all_scores_ripple_df', None),
                    all_sessions_laps_df=dataframes.get('all_sessions_laps_df', None),
                    all_sessions_laps_time_bin_df=dataframes.get('all_sessions_laps_time_bin_df', None),
                    all_sessions_MultiMeasure_laps_df=dataframes.get('all_sessions_MultiMeasure_laps_df', None),
                ),
                additional_filter_predicates={}  # Functions can't be serialized
            )




            # Set filtered dataframes
            for attr_name in cls.DATAFRAME_ATTR_NAMES:
                if attr_name.startswith('filtered_'):
                    # setattr(instance, attr_name, dataframes.get(attr_name, None))
                    instance._filtered_df_dict[attr_name] = dataframes.get(attr_name, None)

            # Set scalar attributes
            for attr_name in cls.SCALAR_ATTR_NAMES:
                if attr_name in scalar_attrs:
                    setattr(instance, attr_name, scalar_attrs[attr_name])
                else:
                    setattr(instance, attr_name, None)

            return instance
        

    @function_attributes(short_name=None, tags=['export', 'pdf', 'html', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-09 10:40', related_items=[])
    def export_for_publication(self, figures_parent_folder: Path, export_suffix: str = 'Laps', export_pdf: bool=True, export_html: bool = False):
        """ exports the figures for publication 
        
        Usage:
            figures_parent_folder: Path = Path(r'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/2025-06-06 - EXPORTS FOR PUBLICATION').resolve()
            _out_paths = df_filter.export_for_publication(figures_parent_folder=figures_parent_folder, export_html=False)
            _out_paths
        
        """
        fig = self.figure_widget

        # filename: str = Path(deepcopy(self.filename)).stem        
        # filename = f"ScatterOverTime_{filename}"
        basename: str = f'ScatterOverTime_{export_suffix}'
        _out_paths = {}
        if export_pdf:
            image_save_path = figures_parent_folder.joinpath(f'{basename}.pdf')
            fig.write_image(image_save_path)
            _out_paths['pdf'] = image_save_path

        if export_html:
            html_file_output = figures_parent_folder.joinpath(f'{basename}.html').resolve()
            pio.write_html(fig, file=html_file_output, auto_open=True)
            _out_paths['html'] = html_file_output
            
        return _out_paths





        

from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

@pd.api.extensions.register_dataframe_accessor("pho_LS_epoch")
class LongShortTrackDataframeAccessor(TimeColumnAliasesProtocol):
    """ Describes a dataframe with at least a neuron_id (aclu) column. Provides functionality regarding building globally (across-sessions) unique neuron identifiers.
    

    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor
    from neuropy.utils.indexing_helpers import NeuroPyDataframeAccessor
    from pyphocorehelpers.indexing_helpers import PhoDataframeAccessor
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import LongShortTrackDataframeAccessor
    
    
    """
    _time_column_name_synonyms = {"start":['begin','start_t','lap_start_t'],
        "stop":['end','stop_t'],
        "label":['name', 'id', 'flat_replay_idx','lap_id']
    }

    _required_column_names = ['start', 'stop']


    def __init__(self, pandas_obj: pd.DataFrame):
        pandas_obj = self.renaming_synonym_columns_if_needed(pandas_obj, required_columns_synonym_dict=self._time_column_name_synonyms)       #@IgnoreException 
        self._validate(pandas_obj)
        self._df = pandas_obj

    @staticmethod
    def _validate(obj):
        """ verify there is a column that identifies the spike's neuron, the type of cell of this neuron ('neuron_type'), and the timestamp at which each spike occured ('t'||'t_rel_seconds') """
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"object must be a pandas Dataframe but is of type: {type(obj)}!\nobj: {obj}")

    @classmethod
    def add_pre_post_delta_category_column_if_needed(cls, epochs_df: pd.DataFrame, t_delta:float, replace_existing:bool=True, start_time_col_name: str='start', end_time_col_name: Optional[str]=None) -> pd.DataFrame:
        """ 2025-05-01 - adds the 'pre_post_delta_category' column if it doesn't exist

        Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on
        
        WARNING: does NOT modify in place!

        Adds Columns: ['pre_post_delta_category']

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import LongShortTrackDataframeAccessor

            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()

            a_laps_decoded_marginal_posterior_df = LongShortTrackDataframeAccessor.add_pre_post_delta_category_column_if_needed(epochs_df=a_laps_decoded_marginal_posterior_df, t_delta=t_delta, start_time_col_name='lap_start_t')
            a_laps_decoded_marginal_posterior_df

        """
        # epochs_df = epochs_df.epochs.to_dataframe()
        # epochs_df[[labels_column_name]] = epochs_df[[labels_column_name]].astype('int')
        is_missing_column: bool = ('pre_post_delta_category' not in epochs_df.columns)
        if (is_missing_column or replace_existing):
            # Create the maze_id column:
            epochs_df['pre_post_delta_category'] = 'pre-delta' # all 'pre-delta' to start 
            if (end_time_col_name is not None):
                Assert.require_columns(dfs=epochs_df, required_columns=[start_time_col_name, end_time_col_name])
                epochs_df.loc[np.logical_and((epochs_df[start_time_col_name].to_numpy() >= t_delta), (epochs_df[end_time_col_name].to_numpy() >= t_delta)), 'pre_post_delta_category'] = 'post-delta' # second epoch, 'post-delta'
            else:
                ## no end-column
                Assert.require_columns(dfs=epochs_df, required_columns=[start_time_col_name])
                epochs_df.loc[(epochs_df[start_time_col_name].to_numpy() >= t_delta), 'pre_post_delta_category'] = 'post-delta' # second epoch, 'post-delta'
                
            epochs_df['pre_post_delta_category'] = epochs_df['pre_post_delta_category'].astype('str') # note the single vs. double brakets in the two cases. Not sure if it makes a difference or not
        else:
            # already exists and we shouldn't overwrite it:
            epochs_df[['pre_post_delta_category']] = epochs_df[['pre_post_delta_category']].astype('str') # note the single vs. double brakets in the two cases. Not sure if it makes a difference or not
        return epochs_df
            

    def adding_pre_post_delta_category_if_needed(self, t_delta:float, replace_existing:bool=True, start_time_col_name: str='start', end_time_col_name: Optional[str]=None) -> pd.DataFrame:
        """ 2025-05-01 - adds the 'pre_post_delta_category' column if it doesn't exist

        Add the pre_post_delta_category to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on
        
        WARNING: does NOT modify in place!

        Adds Columns: ['pre_post_delta_category']
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import LongShortTrackDataframeAccessor

            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
            filter_epochs = an_out_result.filter_epochs.pho_LS_epoch.adding_pre_post_delta_category_if_needed(t_delta=t_delta)
            filter_epochs

        """
        epochs_df: pd.DataFrame = self._df
        return self.add_pre_post_delta_category_column_if_needed(epochs_df=epochs_df, t_delta=t_delta, replace_existing=replace_existing, start_time_col_name=start_time_col_name, end_time_col_name=end_time_col_name)
    


class PhoPublicationFigureHelper:
    """ 2025-07-04 - Helps get matplotlib figures in the correct format to be published, allowing manipulation in Adobe Illustrator and vector outputs
    
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PhoPublicationFigureHelper
    
    
    """
    # delta_minus_str: str = '\\Delta -'        
    # delta_plus_str: str = '\\Delta +'
    # delta_minus_str: str = '⬖'        
    # delta_plus_str: str = '⬗'
    delta_minus_str: str = 't < \\Delta'        
    delta_plus_str: str = 't > \\Delta'

    # Extracted from Artboard 3.pdf (Fig3 laps|PBEs scatter panels)
    artboard3_fig3_style: Dict = {
        'colors': {
            'long': '#4D69B1',
            'short': '#DC1E3D',
            'title_body': '#231F20',
            'r_fit': '#7900B3',
            'marker': '#333333',
        },
        'sizes': {
            'track_title': 9.087,
            'title_body': 8.077,
            'axis_label': 7,
            'tick_label': 8.077,
            'r_value': 8,
        },
        'axes': {
            'linewidth': 0.75,
            'tick_width': 0.75,
            'tick_length': 3.5,
        },
        'scatter': {
            's': 5.0,  # diameter ≈ 2.236 pt → s = d²; notebook may still pass s explicitly
            'point_colors': '#33333333',
        },
    }

    
    @classmethod
    def rc_context_kwargs(cls, prepare_for_publication: bool=True, **kwargs) -> Dict:
        """
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PhoPublicationFigureHelper
            
            with mpl.rc_context(PhoPublicationFigureHelper.rc_context_kwargs(prepare_for_publication=prepare_for_publication) | {'figure.figsize': (10, 4), 'figure.dpi': '220'})
            
        """
        _out_rcparams = {'savefig.transparent': True, 'ps.fonttype': 42, 'pdf.fonttype': 42,  }
        _out_rcparams =( _out_rcparams | kwargs)
        if prepare_for_publication:
            ## OVERRIDE in the case of publications:
            _out_rcparams.update({'font.family': 'Arial', 'xtick.labelsize': 5, 'ytick.labelsize': 5, "axes.spines.right": False, "axes.spines.top": False, 'axes.linewidth': 0.8})
        return _out_rcparams # 'figure.dpi': '220', 










# ==================================================================================================================================================================================================================================================================================== #
# 2026-09-22 Figure Testing                                                                                                                                                                                                                                                            #
# ==================================================================================================================================================================================================================================================================================== #

@function_attributes(short_name=None, tags=['UNFINISHED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-09-24 16:44', related_items=[])
def _plot_recreated_figure():
    """ aimed to set the Plotly figure fonts/export sizes to match those of a reference .pdf

        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import _plot_recreated_figure

        new_fig = _plot_recreated_figure()
        new_fig

    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    # --- FORMATTING PARAMETERS (Adjust these based on exact PDF measurements) ---
    FONT_FAMILY = "Arial, sans-serif"
    TITLE_FONT_SIZE = 18
    AXIS_TITLE_FONT_SIZE = 14
    TICK_FONT_SIZE = 12
    ANNOTATION_FONT_SIZE = 14

    COLOR_LONG = "#1f77b4"  # Muted Blue
    COLOR_SHORT = "#ff7f0e" # Safety Orange

    # --- DUMMY DATA GENERATION (Replace with your actual data) ---
    np.random.seed(42)
    n_points = 2000

    # Generating "Delta-aligned Event Time" (-1000 to 1000)
    x_time = np.random.normal(loc=0, scale=400, size=n_points)
    x_time = np.clip(x_time, -1000, 1000)

    # Generating "Probability of Short Track" (0 to 1)
    y_prob = np.random.uniform(0, 1, size=n_points)

    # Simulating categories based on legend "Long Short"
    categories = np.where(np.random.rand(n_points) > 0.5, 'Long', 'Short')

    # Separate data for plotting
    x_long = x_time[categories == 'Long']
    y_long = y_prob[categories == 'Long']
    x_short = x_time[categories == 'Short']
    y_short = y_prob[categories == 'Short']

    # --- FIGURE INITIALIZATION ---
    # Creating a Joint Plot layout (Main scatter, Top Histogram, Right Histogram)
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.2, 0.8], # Top histogram takes 20% height, main plot 80%
        column_widths=[0.8, 0.2], # Main plot takes 80% width, right histogram 20%
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        shared_xaxes=True,
        shared_yaxes=True
    )

    # --- ADDING TRACES ---

    # 1. Main Scatter Plot (Row 2, Col 1)
    fig.add_trace(
        go.Scatter(
            x=x_long, y=y_long, mode='markers',
            name='Long', marker=dict(color=COLOR_LONG, size=4, opacity=0.6),
            showlegend=True
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_short, y=y_short, mode='markers',
            name='Short', marker=dict(color=COLOR_SHORT, size=4, opacity=0.6),
            showlegend=True
        ),
        row=2, col=1
    )

    # 2. Top Marginal Histogram (Row 1, Col 1)
    fig.add_trace(
        go.Histogram(
            x=x_time,
            marker_color='gray',
            showlegend=False,
            xbins=dict(start=-1000, end=1000, size=50)
        ),
        row=1, col=1
    )

    # 3. Right Marginal Histogram (Row 2, Col 2)
    fig.add_trace(
        go.Histogram(
            y=y_prob,
            marker_color='gray',
            showlegend=False,
            ybins=dict(start=0, end=1, size=0.05)
        ),
        row=2, col=2
    )

    # --- AXIS FORMATTING ---

    # Main X-Axis (Delta-aligned Event Time)
    fig.update_xaxes(
        title_text="Delta-aligned Event Time (seconds)",
        title_font=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
        tickvals=[-1000, -500, 0, 500, 1000],
        tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY),
        range=[-1050, 1050],
        showgrid=True, gridcolor='lightgray',
        row=2, col=1
    )

    # Main Y-Axis (Probability of Short Track)
    fig.update_yaxes(
        title_text="Probability of Short Track",
        title_font=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
        tickvals=[0, 0.5, 1],
        tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY),
        range=[-0.05, 1.05],
        showgrid=True, gridcolor='lightgray',
        row=2, col=1
    )

    # Top Histogram Y-Axis (# Events: 0, 500, 1000, 1500)
    fig.update_yaxes(
        title_text="# Events",
        title_font=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
        tickvals=[0, 500, 1000, 1500],
        tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY),
        range=[0, 1600],
        row=1, col=1
    )

    # Right Histogram X-Axis (# Events: 0, 200, 400, 600, 800)
    fig.update_xaxes(
        title_text="# Events",
        title_font=dict(size=AXIS_TITLE_FONT_SIZE, family=FONT_FAMILY),
        tickvals=[0, 200, 400, 600, 800],
        tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY),
        range=[0, 850],
        row=2, col=2
    )

    # --- ANNOTATIONS & LAYOUT ---

    # Add a vertical line at delta time = 0
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="black", row=1, col=1)

    # Annotations for "Pre-delta" and "Post-delta"
    fig.add_annotation(
        x=-500, y=1.02, xref="x", yref="y domain",
        text="Pre-delta", showarrow=False,
        font=dict(size=ANNOTATION_FONT_SIZE, family=FONT_FAMILY, color="black")
    )
    fig.add_annotation(
        x=500, y=1.02, xref="x", yref="y domain",
        text="Post-delta", showarrow=False,
        font=dict(size=ANNOTATION_FONT_SIZE, family=FONT_FAMILY, color="black")
    )

    # Final Layout Updates
    fig.update_layout(
        title=dict(
            text="Short-track Decoded Likelihood during PBEs",
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY),
            x=0.5, # Center the title
            y=0.98
        ),
        legend=dict(
            title=dict(text="Track Type", font=dict(family=FONT_FAMILY)),
            font=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY),
            x=0.85, y=0.85, # Position legend inside the top right of main plot
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=900,
        height=700,
        margin=dict(l=80, r=40, t=80, b=80)
    )

    return fig


# ==================================================================================================================================================================================================================================================================================== #
# 2026-10-14 - Analysis of LdC effect on Post-Delta Bars in Figure 2                                                                                                                                                                                                                   #
# ==================================================================================================================================================================================================================================================================================== #

## INPUTS: pbe, across_session_inst_fr_computation_dict
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import InstantaneousSpikeRateGroupsComputation, SpikeRateTrends ## for `Figure2Extended`

@metadata_attributes(short_name=None, tags=['LdC', 'Long/Short', 'SdC', 'PostDelta', 'figure2'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-10-14 15:34', related_items=[])
class Figure2Extended:
    """ Tests whether Figure-2 post-Δ (ReplayDeltaPlus) PBE activity of LdCs is elevated specifically on clear LONG-track PBEs.

    For each session: matches FAT `pbe` rows with `is_clear_LONG` onto `LxC_ReplayDeltaPlus` / `SxC_ReplayDeltaPlus` / `AnyC_ReplayDeltaPlus`
    `filter_epochs_df` by start/stop, then compares mean per-epoch FR of LdCs, SdCs, and non-(LdC∪SdC) on clear_LONG vs other PBEs.
    Aggregates those session means into `summary_df` (across-session mean/sem/median).

    Requires notebook-scope `pbe` (FAT per-epoch PBE df with `is_clear_LONG`, `session_name`, start/stop or parent_epoch_*).

    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import Figure2Extended

    ## INPUTS: pbe, across_session_inst_fr_computation_dict
    summary_df, session_activity_dict = Figure2Extended.main(across_session_inst_fr_computation_dict=across_session_inst_fr_computation_dict)
    summary_df
    # session_activity_dict[ctxt] == {'LdCs': {False: ..., True: ...}, 'SdCs': {...}, 'non_LdCs': {...}}  # keys: is_clear_LONG

    # single session:
    # _out_dict, _extras = Figure2Extended.compute_single_session(an_inst_fr_comp=across_session_inst_fr_computation_dict[ctxt], curr_session_context=ctxt)

    """
    @classmethod
    def compute_delta_plus_LdC_result(cls, a_specific_spikeratetrends_result: SpikeRateTrends, curr_sess_pbe: pd.DataFrame, curr_session_context: IdentifyingContext, debug_print=False):
        """ Match clear_LONG PBEs onto a SpikeRateTrends epoch list; return FR agg by is_clear_LONG.

        Returns: epoch_unit_fr_agg_df, epoch_unit_fr_df, filter_epochs_df, clear_long_activity, (fr_aclus_col_names, aclu_agg_column_names_list)
        """
        sess_name_underscore: str = curr_session_context.get_description(separator='_')  # 'kdiba_gor01_one_2006-6-08_14-26-15'

        inst_fr_df_list = a_specific_spikeratetrends_result.inst_fr_df_list
        epoch_unit_fr_df_list = a_specific_spikeratetrends_result.epoch_unit_fr_df_list
        filter_epochs_df = deepcopy(a_specific_spikeratetrends_result.filter_epochs_df).reset_index(drop=True)  # row i ↔ epoch_unit_fr_df_list[i]
        assert len(filter_epochs_df) == len(epoch_unit_fr_df_list), (len(filter_epochs_df), len(epoch_unit_fr_df_list))

        pbe_clear_long = curr_sess_pbe.loc[curr_sess_pbe['is_clear_LONG']].copy()

        # FAT often has parent_epoch_*; prefer those, else start/stop
        t0_col = 'parent_epoch_start_t' if 'parent_epoch_start_t' in pbe_clear_long.columns else 'start'
        t1_col = 'parent_epoch_end_t' if 'parent_epoch_end_t' in pbe_clear_long.columns else 'stop'
        if debug_print:
            print(f'pbe clear_LONG for {sess_name_underscore}: n={len(pbe_clear_long)} (using {t0_col}/{t1_col})')


        # --- match clear_LONG times onto filter_epochs_df ---
        atol = 1e-2  # seconds
        fe_starts = filter_epochs_df['start'].to_numpy(dtype=float)
        fe_stops = filter_epochs_df['stop'].to_numpy(dtype=float)

        matched_idxs = []
        match_rows = []
        for _, row in pbe_clear_long.iterrows():
            t0, t1 = float(row[t0_col]), float(row[t1_col])
            hit = np.where((np.abs(fe_starts - t0) <= atol) & (np.abs(fe_stops - t1) <= atol))[0]
            if len(hit) == 0:
                # looser: start-only
                hit = np.where(np.abs(fe_starts - t0) <= atol)[0]
            if len(hit):
                i = int(hit[0])
                matched_idxs.append(i)
                match_rows.append({
                    'filter_epochs_idx': i,
                    'start': fe_starts[i],
                    'stop': fe_stops[i],
                    'pbe_start': t0,
                    'pbe_stop': t1,
                    'P_Long': row.get('P_Long', np.nan),
                    'P_Short': row.get('P_Short', np.nan),
                })
            ## END if len(hit)...
        ## END for _, row in pbe_clear_long.iterrows()...

        matched_idxs = np.unique(matched_idxs)
        if debug_print:
            print(f'matched {len(matched_idxs)} / {len(pbe_clear_long)} clear_LONG → filter_epochs_df (atol={atol}s)')

        # boolean mask on filter_epochs_df
        filter_epochs_df['is_clear_LONG'] = False
        filter_epochs_df.loc[matched_idxs, 'is_clear_LONG'] = True
        # clear_long_filter_epochs_df = filter_epochs_df.loc[filter_epochs_df['is_clear_LONG']].copy()
        # display(clear_long_filter_epochs_df)

        # --- LxC activity for those matched epochs ---
        # epoch_unit_fr_df_list[i]: DataFrame with columns = LxC aclus, values = FR in that epoch
        n_LdC_active = []
        mean_LdC_fr = []
        for i in matched_idxs:
            fr_df = epoch_unit_fr_df_list[i]
            fr = fr_df.to_numpy(dtype=float).ravel() if hasattr(fr_df, 'to_numpy') else np.asarray(fr_df, dtype=float).ravel()
            # "active" = FR > 0 (same idea as rdf active_aclus)
            n_LdC_active.append(int(np.sum(np.isfinite(fr) & (fr > 0))))
            mean_LdC_fr.append(float(np.nanmean(fr)) if fr.size else np.nan)
        ## END for i in matched_idxs...

        clear_long_activity = pd.DataFrame({
            'filter_epochs_idx': matched_idxs,
            'start': filter_epochs_df.loc[matched_idxs, 'start'].to_numpy(),
            'stop': filter_epochs_df.loc[matched_idxs, 'stop'].to_numpy(),
            'n_LdC_active': n_LdC_active,
            'mean_LdC_fr': mean_LdC_fr,
        })
        if debug_print:
            display(clear_long_activity)
            print(f"clear_LONG: mean n_LdC_active={np.nanmean(n_LdC_active):.2f}  |  other epochs for contrast:")

        # optional contrast: non-clear epochs in the same filter_epochs_df
        other_idxs = filter_epochs_df.index[~filter_epochs_df['is_clear_LONG']].to_numpy()

        ## OUTPUTS: matched_idxs, other_idxs

        # matched_items = [epoch_unit_fr_df_list[i].to_dict() for i in matched_idxs]
        # other_items = [epoch_unit_fr_df_list[i].to_dict() for i in other_idxs]

        epoch_unit_fr_df = pd.DataFrame.from_records([df.to_dict() for i, df in enumerate(epoch_unit_fr_df_list)])
        fr_aclus_col_names = list(epoch_unit_fr_df.columns)
        epoch_unit_fr_df['epoch_IDX'] = epoch_unit_fr_df.index.astype(int)
        Assert.same_length(epoch_unit_fr_df, filter_epochs_df)
        # transfer_col_names
        if debug_print:
            print(f'fr_aclus_col_names: {fr_aclus_col_names}')
        epoch_unit_fr_df = pd.concat((filter_epochs_df, epoch_unit_fr_df), axis='columns')

        # Performed 4 aggregations grouped on column: 'is_clear_LONG'
        # agg_dict = {{f'{aclu}_mean':(int(aclu), 'mean'), f'{aclu}_max':(int(aclu), 'max')} for aclu in fr_aclus_col_names}
        agg_dict = {f'{aclu}_mean':(int(aclu), 'mean') for aclu in fr_aclus_col_names}
        aclu_agg_column_names_list = list(agg_dict.keys())
        epoch_unit_fr_agg_df = epoch_unit_fr_df.groupby(['is_clear_LONG']).agg(**agg_dict).reset_index()
        # epoch_unit_fr_agg_df = epoch_unit_fr_df.groupby(['is_clear_LONG']).agg(col8_mean=(8, 'mean'), col8_max=(8, 'max'), col97_mean=(97, 'mean'), col109_mean=(109, 'mean')).reset_index()
        epoch_unit_fr_agg_df.set_index('is_clear_LONG', drop=True, inplace=True)
        return epoch_unit_fr_agg_df, epoch_unit_fr_df, filter_epochs_df, clear_long_activity, (fr_aclus_col_names, aclu_agg_column_names_list)



    ## INPUTS: an_inst_fr_comp: InstantaneousSpikeRateGroupsComputation, curr_session_context: IdentifyingContext
    @classmethod
    def compute_single_session(cls, epochs_df: pd.DataFrame, an_inst_fr_comp: InstantaneousSpikeRateGroupsComputation, curr_session_context: IdentifyingContext):
        ## DeltaMinus
        # a_pbeDelta_result: SpikeRateTrends = an_inst_fr_comp.LxC_ReplayDeltaMinus  # long-maze replay pool; use Plus for short-maze
        # an_any_C_pbeDelta_result: SpikeRateTrends = an_inst_fr_comp.AnyC_ReplayDeltaMinus

        ## DeltaPlus (short-maze replay pool)
        a_pbeDelta_result: SpikeRateTrends = an_inst_fr_comp.LxC_ReplayDeltaPlus
        an_SxC_pbeDelta_result: SpikeRateTrends = an_inst_fr_comp.SxC_ReplayDeltaPlus
        an_any_C_pbeDelta_result: SpikeRateTrends = an_inst_fr_comp.AnyC_ReplayDeltaPlus

        if a_pbeDelta_result is None:
            print(f'\tWARN: a_pbeDelta_result (LxC_ReplayDeltaPlus) is None for curr_session_context: {curr_session_context}. Skipping.')

        sess_name_underscore: str = curr_session_context.get_description(separator='_')  # 'kdiba_gor01_one_2006-6-08_14-26-15'

        ## INPUTS: sess_name_underscore
        # --- this session's clear_LONG PBEs from FAT `pbe` ---
        curr_sess_pbe = epochs_df.loc[epochs_df['session_name'].astype(str).str.replace('-', '_', regex=False) == sess_name_underscore.replace('-', '_')].copy()
        # also try hyphenated FAT style if needed
        if curr_sess_pbe.empty:
            curr_sess_pbe = epochs_df.loc[epochs_df['session_name'].astype(str) == curr_session_context.get_description(separator='_')].copy()
        if curr_sess_pbe.empty:
            curr_sess_pbe = epochs_df.loc[epochs_df['session_name'].astype(str).str.contains(curr_session_context.session_name)].copy()

        LdC_epoch_unit_fr_agg_df, LdC_epoch_unit_fr_df, filter_epochs_df, clear_long_activity, (LdC_fr_aclus_col_names, LdC_aclu_agg_column_names_list) = cls.compute_delta_plus_LdC_result(a_specific_spikeratetrends_result=a_pbeDelta_result, curr_sess_pbe=curr_sess_pbe, curr_session_context=curr_session_context)
        an_LdC_all_aclus_mean_fr_dict = LdC_epoch_unit_fr_agg_df[LdC_aclu_agg_column_names_list].mean(axis='columns', skipna=True).to_dict()
        ## OUTPUTS: epoch_unit_fr_agg_df, an_LdC_all_aclus_mean_fr_dict

        an_SdC_all_aclus_mean_fr_dict = {}
        SdC_epoch_unit_fr_agg_df = None
        if an_SxC_pbeDelta_result is not None:
            SdC_epoch_unit_fr_agg_df, SdC_epoch_unit_fr_df, SdC_filter_epochs_df, SdC_clear_long_activity, (SdC_fr_aclus_col_names, SdC_aclu_agg_column_names_list) = cls.compute_delta_plus_LdC_result(a_specific_spikeratetrends_result=an_SxC_pbeDelta_result, curr_sess_pbe=curr_sess_pbe, curr_session_context=curr_session_context)
            an_SdC_all_aclus_mean_fr_dict = SdC_epoch_unit_fr_agg_df[SdC_aclu_agg_column_names_list].mean(axis='columns', skipna=True).to_dict()
        else:
            print(f'\tWARN: SxC_ReplayDeltaPlus is None for curr_session_context: {curr_session_context}. SdCs skipped.')

        an_any_C_epoch_unit_fr_agg_df, an_any_C_epoch_unit_fr_df, an_any_C_filter_epochs_df, an_any_C_clear_long_activity, (an_any_C_fr_aclus_col_names, an_any_C_aclu_agg_column_names_list) = cls.compute_delta_plus_LdC_result(a_specific_spikeratetrends_result=an_any_C_pbeDelta_result, curr_sess_pbe=curr_sess_pbe, curr_session_context=curr_session_context)
        ## any_C aclus include the LdCs/SdCs, so exclude those to look at the mean of the rest
        exclusive_aclu_col_names = set(list(LdC_epoch_unit_fr_agg_df.columns))
        if SdC_epoch_unit_fr_agg_df is not None:
            exclusive_aclu_col_names |= set(list(SdC_epoch_unit_fr_agg_df.columns))
        non_LdC_aclu_col_names = set(list(an_any_C_epoch_unit_fr_agg_df)) - exclusive_aclu_col_names
        an_non_LdC_aclu_all_aclus_mean_fr_dict = an_any_C_epoch_unit_fr_agg_df[non_LdC_aclu_col_names].mean(axis='columns', skipna=True).to_dict()

        ## OUTPUTS: an_LdC_all_aclus_mean_fr_dict, an_SdC_all_aclus_mean_fr_dict, an_non_LdC_aclu_all_aclus_mean_fr_dict
        _out_dict = {'LdCs': an_LdC_all_aclus_mean_fr_dict, 'SdCs': an_SdC_all_aclus_mean_fr_dict, 'non_LdCs': an_non_LdC_aclu_all_aclus_mean_fr_dict}
        return _out_dict, (an_any_C_epoch_unit_fr_agg_df, an_LdC_all_aclus_mean_fr_dict, an_SdC_all_aclus_mean_fr_dict, an_non_LdC_aclu_all_aclus_mean_fr_dict)


    @classmethod
    def build_across_session_cell_stats(cls, session_activity_dict, results_print=False):

        # Flatten: one row per (session, cell_group, is_clear_LONG)
        rows = []
        for ctxt, group_dict in session_activity_dict.items():
            base = ctxt.to_dict() if hasattr(ctxt, 'to_dict') else dict(
                format_name=ctxt.format_name, animal=ctxt.animal,
                exper_name=ctxt.exper_name, session_name=ctxt.session_name,
            )
            for cell_group, clear_dict in group_dict.items():  # 'LdCs' / 'SdCs' / 'non_LdCs'
                for is_clear_LONG, mean_n_active in clear_dict.items():  # False / True
                    rows.append({
                        **base,
                        'cell_group': cell_group,
                        'is_clear_LONG': bool(is_clear_LONG),
                        'mean_n_active': float(mean_n_active),
                    })
                ## END for is_clear_LONG, mean_n_active...
            ## END for cell_group, clear_dict...
        ## END for ctxt, group_dict...

        df = pd.DataFrame(rows)
        df['session_key'] = df['format_name'] + '_' + df['animal'] + '_' + df['exper_name'] + '_' + df['session_name']

        # Wide per session: columns = (cell_group, is_clear_LONG)
        wide = df.pivot_table(
            index=['session_key', 'animal', 'exper_name', 'session_name'],
            columns=['cell_group', 'is_clear_LONG'],
            values='mean_n_active',
            aggfunc='mean',
        ).sort_index(axis=1)
        wide.columns = [f'{g}_{"clearLONG" if c else "other"}' for g, c in wide.columns]
        wide = wide.reset_index()

        # Per-session deltas (clear_LONG − other)
        for g in ['LdCs', 'SdCs', 'non_LdCs']:
            clear_col, other_col = f'{g}_clearLONG', f'{g}_other'
            if (clear_col in wide.columns) and (other_col in wide.columns):
                wide[f'{g}_delta'] = wide[clear_col] - wide[other_col]
            ## END if (clear_col in wide.columns) and (other_col in wide.columns)...
        ## END for g in ['LdCs', 'SdCs', 'non_LdCs']...

        # Across-session averages
        summary = df.groupby(['cell_group', 'is_clear_LONG'], as_index=False).agg(
            n_sessions=('session_key', 'nunique'),
            mean=('mean_n_active', 'mean'),
            sem=('mean_n_active', 'sem'),
            median=('mean_n_active', 'median'),
        )
        if results_print:
            display(summary)

            # Compact: mean ± sem across sessions
            print(summary.pivot(index='cell_group', columns='is_clear_LONG', values='mean'))
            print('Across-session mean deltas (clearLONG − other):')
            delta_cols = [c for c in ['LdCs_delta', 'SdCs_delta', 'non_LdCs_delta'] if c in wide.columns]
            print(wide[delta_cols].agg(['mean', 'sem', 'median']))

        return summary



    @classmethod
    def main(cls, epochs_df: pd.DataFrame, across_session_inst_fr_computation_dict, **kwargs):
        session_activity_dict = {}
        for curr_session_context, an_inst_fr_comp in across_session_inst_fr_computation_dict.items():
            if an_inst_fr_comp is not None:
                try:
                    _out_dict, (an_any_C_epoch_unit_fr_agg_df, an_LdC_all_aclus_mean_fr_dict, an_SdC_all_aclus_mean_fr_dict, an_non_LdC_aclu_all_aclus_mean_fr_dict) = cls.compute_single_session(epochs_df=epochs_df, an_inst_fr_comp=an_inst_fr_comp, curr_session_context=curr_session_context)
                    session_activity_dict[curr_session_context] = _out_dict
                except (ValueError, AttributeError) as e:
                    print(f'WARN error e: {e} occured for curr_session_context: {curr_session_context}! Skipping.')
                    # raise e

            else:
                print(f'WARN an_inst_fr_comp is None for curr_session_context: {curr_session_context}! Skipping.')

        ## OUTPUTS: session_activity_dict

        summary_df = cls.build_across_session_cell_stats(session_activity_dict=session_activity_dict)
        return summary_df, session_activity_dict

        # ## False: not 'is_clear_LONG', True: 'is_clear_LONG'
        # {'LdCs': {False: ..., True: ...}, 'SdCs': {False: ..., True: ...}, 'non_LdCs': {False: ..., True: ...}}
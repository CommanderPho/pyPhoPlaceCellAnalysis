# CrossComputationComparisonHelpers
import numpy as np
from functools import reduce # _find_any_context_neurons

from neuropy.utils.dynamic_container import DynamicContainer, override_dict, overriding_dict_with, get_dict_subset
from neuropy.utils.misc import safe_item

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

from neuropy.utils.colors_util import get_neuron_colors # required for build_neurons_color_map 


from enum import Enum


class SplitPartitionMembership(Enum):
	"""Docstring for SplitPartitionMembership."""
	LEFT_ONLY = 0
	SHARED = 1
	RIGHT_ONLY = 2
	

class SetPartition(object):
	""" Converted from a one-off structure produced by `_compare_computation_results` as illustrated below:
		pf_neurons_diff = DynamicContainer(lhs_only=pf_neurons_lhs_unique, rhs_only=pf_neurons_rhs_unique, intersection=pf_neurons_both, either=pf_neurons_either,
			 shared=DynamicContainer(n_neurons=n_neurons, shared_fragile_neuron_IDXs=shared_fragile_neuron_IDXs, pairs=shared_fragile_neuron_IDXs_to_pairs, missing_neuron_IDXs=all_missing_IDXs, missing_neuron_ids=pf_neurons_missing_from_any, aclu_to_shared_fragile_IDX_map=aclu_to_shared_fragile_IDX_map)) 

		TODO: perhaps bring computations in here?
	"""
	def __init__(self, lhs_only, rhs_only, either, intersection, shared_structure):
		super(SetPartition, self).__init__()
		self.lhs_only = lhs_only
		self.rhs_only = rhs_only
		self.intersection = intersection
		self.either = either
		self._shared = shared_structure

	@property
	def n_neurons(self):
		"""The n_neurons property."""
		return len(self.either)
		
	@property
	def shared(self):
		""" DynamicContainer(n_neurons=self.n_neurons, shared_fragile_neuron_IDXs=shared_fragile_neuron_IDXs, pairs=shared_fragile_neuron_IDXs_to_pairs, missing_neuron_IDXs=all_missing_IDXs, missing_neuron_ids=pf_neurons_missing_from_any, aclu_to_shared_fragile_IDX_map=aclu_to_shared_fragile_IDX_map) """
		return self._shared
	@shared.setter
	def shared(self, value):
		self._shared = value


def _compare_computation_results(lhs_computation_results, rhs_computation_results) -> SetPartition:
	"""Computes the differences between two separate computation results, such as those computed for different epochs

	Args:
		lhs_computation_results (_type_): _description_
		rhs_computation_results (_type_): _description_

	Returns:
		_type_: _description_

	Usage:
		from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results
		pf_neurons_diff = _compare_computation_results(computation_results.maze1_PYR, computation_results.maze2_PYR)
		pf_neurons_diff

	"""
	def _subfn_extract_neuron_ids(computation_results):
		if isinstance(computation_results, ComputationResult):
			neuron_ids = computation_results.computed_data.pf2D.ratemap.neuron_ids
		elif isinstance(computation_results, (DynamicParameters, dict)):
			neuron_ids = computation_results['pf2D'].ratemap.neuron_ids # assume computed_data
		elif isinstance(computation_results, (list, np.ndarray)):
			# assume to be the neuron_ids directly
			neuron_ids = computation_results
		else:
			print(f'ERROR: type(computation_results): {type(computation_results)}, is unhandled')
			raise NotImplementedError
		return neuron_ids

	lhs_neuron_ids = _subfn_extract_neuron_ids(lhs_computation_results)
	rhs_neuron_ids = _subfn_extract_neuron_ids(rhs_computation_results)

	pf_neurons_lhs_unique = np.setdiff1d(lhs_neuron_ids, rhs_neuron_ids) # returns neurons present in lhs that are missing from rhs
	pf_neurons_rhs_unique = np.setdiff1d(rhs_neuron_ids, lhs_neuron_ids) # returns neurons present in rhs that are missing from lhs
	pf_neurons_both = np.intersect1d(rhs_neuron_ids, lhs_neuron_ids) # only those common in both (intersection/AND)

	pf_neurons_either = np.union1d(rhs_neuron_ids, lhs_neuron_ids) # those present in either (union/OR)
	n_neurons = len(pf_neurons_either)
	shared_fragile_neuron_IDXs = np.arange(n_neurons) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69]

	aclu_to_shared_fragile_IDX_map = {aclu:idx for idx, aclu in zip(shared_fragile_neuron_IDXs, pf_neurons_either)} # reverse lookup map from aclu to shared fragile index

	pf_neurons_missing_from_any = np.union1d(pf_neurons_lhs_unique, pf_neurons_rhs_unique) # a list of aclus that are missing from at least one of the inputs (meaning they're unique to one of them). e.g. [2, 4, 8, 13, 19, 105, 109]
	all_missing_IDXs = [aclu_to_shared_fragile_IDX_map.get(aclu, None) for aclu in pf_neurons_missing_from_any if aclu in pf_neurons_either] # the IDXs of all neurons missing from at least one of the inputs (meaning they're unique to one of them). e.g. [0, 2, 4, 7, 11, 67, 69]
	
	# lhs_missing_IDXs = [aclu_to_shared_fragile_IDX_map.get(aclu, np.nan) for aclu in pf_neurons_rhs_unique if aclu in aclu_to_shared_fragile_IDX_map]
	# rhs_missing_IDXs = [aclu_to_shared_fragile_IDX_map.get(aclu, np.nan) for aclu in pf_neurons_lhs_unique if aclu in aclu_to_shared_fragile_IDX_map]

	lhs_shared_IDXs_map = [safe_item(np.squeeze(np.argwhere(aclu == lhs_neuron_ids)), default=None) for aclu in pf_neurons_either] # [0, 1, None, 2, 3, 4, 5, None, 6, 7, 8, None, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
	rhs_shared_IDXs_map = [safe_item(np.squeeze(np.argwhere(aclu == rhs_neuron_ids)), default=None) for aclu in pf_neurons_either] # [None, 0, 1, 2, None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, None, 65, None]

	assert len(rhs_shared_IDXs_map) == n_neurons
	assert len(lhs_shared_IDXs_map) == n_neurons
	shared_fragile_neuron_IDXs_to_pairs = list(zip(lhs_shared_IDXs_map, rhs_shared_IDXs_map)) # [(0, None), (1, 0), (None, 1), (2, 2), (3, None), (4, 3), (5, 4), (None, 5), (6, 6), (7, 7), (8, 8), (None, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, None), (65, 65), (66, None)]
	# shared_fragile_neuron_IDXs_to_pairs 

	# pf_neurons_diff = DynamicContainer(lhs_only=pf_neurons_lhs_unique, rhs_only=pf_neurons_rhs_unique, intersection=pf_neurons_both, either=pf_neurons_either,
	#  	shared=DynamicContainer(n_neurons=n_neurons, shared_fragile_neuron_IDXs=shared_fragile_neuron_IDXs, pairs=shared_fragile_neuron_IDXs_to_pairs, missing_neuron_IDXs=all_missing_IDXs, missing_neuron_ids=pf_neurons_missing_from_any, aclu_to_shared_fragile_IDX_map=aclu_to_shared_fragile_IDX_map))
	pf_neurons_diff = SetPartition(lhs_only=pf_neurons_lhs_unique, rhs_only=pf_neurons_rhs_unique, intersection=pf_neurons_both, either=pf_neurons_either,
		shared_structure=DynamicContainer(n_neurons=n_neurons, shared_fragile_neuron_IDXs=shared_fragile_neuron_IDXs, pairs=shared_fragile_neuron_IDXs_to_pairs, missing_neuron_IDXs=all_missing_IDXs, missing_neuron_ids=pf_neurons_missing_from_any, aclu_to_shared_fragile_IDX_map=aclu_to_shared_fragile_IDX_map)
		)
	return pf_neurons_diff




def _find_any_context_neurons(*args):
	"""Given lists of ids/IDXs are arguments, it finds all unique ids/IDXs present in any of the lists.
	Returns:
		_type_: np.array

	Usage:
		from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _find_any_context_neurons
		all_results_neuron_ids_lists = [a_result.computed_data.pf2D.ratemap.neuron_ids for a_result in curr_active_pipeline.computation_results.values()]
		_find_any_context_neurons(*all_results_neuron_ids_lists) # array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27, 28, 29, 32, 33, 34, 36, 37, 38, 41, 43, 44, 45, 46, 47, 49, 51, 52, 53, 54, 55, 56, 59])
	"""
	# reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
	return reduce(np.union1d, tuple(args))


def build_neurons_color_map(n_neurons:int, sortby=None, cmap=None):
    """ neurons_colors_array = build_neurons_color_map(n_neurons, sortby=shared_fragile_neuron_IDXs, cmap=None) """
    if sortby is None:
        sort_ind = np.arange(n_neurons)
    elif isinstance(sortby, (list, np.ndarray)):
        # use the provided sort indicies
        sort_ind = sortby
    else:
        sort_ind = np.arange(n_neurons)

    # Use the get_neuron_colors function to generate colors for these neurons
    neurons_colors_array = get_neuron_colors(sort_ind, cmap=cmap)
    return neurons_colors_array



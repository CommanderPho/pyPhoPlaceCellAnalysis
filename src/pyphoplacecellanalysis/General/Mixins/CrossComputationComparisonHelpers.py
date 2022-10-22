# CrossComputationComparisonHelpers
import numpy as np

from neuropy.utils.dynamic_container import DynamicContainer, override_dict, overriding_dict_with, get_dict_subset
from neuropy.utils.misc import safe_item

# def _compare_filtered_sessions(lhs_sess, rhs_sess):

def _compare_computation_results(lhs_computation_results, rhs_computation_results):
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
	pf_neurons_lhs_unique = np.setdiff1d(lhs_computation_results.computed_data.pf2D.ratemap.neuron_ids, rhs_computation_results.computed_data.pf2D.ratemap.neuron_ids) # returns neurons present in lhs that are missing from rhs
	pf_neurons_rhs_unique = np.setdiff1d(rhs_computation_results.computed_data.pf2D.ratemap.neuron_ids, lhs_computation_results.computed_data.pf2D.ratemap.neuron_ids) # returns neurons present in rhs that are missing from lhs
	pf_neurons_either = np.union1d(rhs_computation_results.computed_data.pf2D.ratemap.neuron_ids, lhs_computation_results.computed_data.pf2D.ratemap.neuron_ids)

	shared_fragile_neuron_IDXs = np.arange(len(pf_neurons_either)) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69]
	lhs_shared_IDXs_map = [safe_item(np.squeeze(np.argwhere(aclu == lhs_computation_results.computed_data.pf2D.ratemap.neuron_ids)), default=None) for aclu in pf_neurons_either] # [0, 1, None, 2, 3, 4, 5, None, 6, 7, 8, None, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
	rhs_shared_IDXs_map = [safe_item(np.squeeze(np.argwhere(aclu == rhs_computation_results.computed_data.pf2D.ratemap.neuron_ids)), default=None) for aclu in pf_neurons_either] # [None, 0, 1, 2, None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, None, 65, None]

	shared_fragile_neuron_IDXs_to_pairs = list(zip(lhs_shared_IDXs_map, rhs_shared_IDXs_map)) # [(0, None), (1, 0), (None, 1), (2, 2), (3, None), (4, 3), (5, 4), (None, 5), (6, 6), (7, 7), (8, 8), (None, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, None), (65, 65), (66, None)]
	# shared_fragile_neuron_IDXs_to_pairs 

	pf_neurons_diff = DynamicContainer(lhs_only=pf_neurons_lhs_unique, rhs_only=pf_neurons_rhs_unique, either=pf_neurons_either, shared=DynamicContainer(shared_fragile_neuron_IDXs=shared_fragile_neuron_IDXs, pairs=shared_fragile_neuron_IDXs_to_pairs)) 
	return pf_neurons_diff


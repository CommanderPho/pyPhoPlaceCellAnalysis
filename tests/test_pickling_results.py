import unittest
from pathlib import Path
import pandas as pd
from attrs import define, field, Factory
import tables as tb
from typing import List, Dict

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle, SequenceBasedComputationsContainer
# from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrialByTrialActivityResult





#TODO 2023-08-23 10:41: - [ ] Skeleton of tests written by ChatGPT, write tests

""" #TODO 2024-05-29 09:50: - [ ] UNIFNISHED - needs to load the data to be tested

Write `unittest.TestCase`s for each of the following:

Test pickling/unpickling of:  SequenceBasedComputationsContainer,  WCorrShuffle, TrialByTrialActivityResult 


from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from typing import NewType

import neuropy.utils.type_aliases as types
from neuropy.utils.misc import build_shuffled_ids, shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration


from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData

DecodedEpochsResultsDict = NewType('DecodedEpochsResultsDict', Dict[types.DecoderName, DecodedFilterEpochsResult]) # A Dict containing the decoded filter epochs result for each of the four 1D decoder names
ShuffleIdx = NewType('ShuffleIdx', int)

wcorr_tool: WCorrShuffle = WCorrShuffle.init_from_templates(curr_active_pipeline=curr_active_pipeline)

wcorr_tool.compute_shuffles(num_shuffles=2)


## TEST: WCorrShuffle pickling
saveData('2024-05-29_test_wcorr_tool_no_complete_results.pkl', (wcorr_tool, ), safe_save=False) ## pickle it

## TEST: WCorrShuffle un-pickling
loaded_wcorr_tool: WCorrShuffle = loadData('2024-05-29_test_wcorr_tool_no_complete_results.pkl')[0]
loaded_wcorr_tool


## Test pickling container (`SequenceBasedComputationsContainer`):
SequenceBased_container: SequenceBasedComputationsContainer = SequenceBasedComputationsContainer(wcorr_ripple_shuffle=wcorr_tool, is_global=True)
SequenceBased_container

## TEST: SequenceBasedComputationsContainer pickling
saveData('2024-05-29_test_wcorr_tool_no_complete_results_container.pkl', (SequenceBased_container, ), safe_save=False)

## TEST: SequenceBasedComputationsContainer un-pickling
loaded_SequenceBased_container = loadData('2024-05-29_test_wcorr_tool_no_complete_results_container.pkl')[0]
loaded_SequenceBased_container




directional_trial_by_trial_activity_result: TrialByTrialActivityResult = curr_active_pipeline.global_computation_results.computed_data.get('TrialByTrialActivity', None)
any_decoder_neuron_IDs = directional_trial_by_trial_activity_result.any_decoder_neuron_IDs
active_pf_dt: PfND_TimeDependent = directional_trial_by_trial_activity_result.active_pf_dt
directional_lap_epochs_dict: Dict[str, Epoch] = directional_trial_by_trial_activity_result.directional_lap_epochs_dict
directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = directional_trial_by_trial_activity_result.directional_active_lap_pf_results_dicts
directional_active_lap_pf_results_dicts



"""

class TestPickleUnpickle(unittest.TestCase):

    def setUp(self):
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrialByTrialActivityResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle, SequenceBasedComputationsContainer
        # Setup for WCorrShuffle
        self.wcorr_tool = WCorrShuffle.init_from_templates(curr_active_pipeline=curr_active_pipeline)
        self.wcorr_tool.compute_shuffles(num_shuffles=2)
        
        # Setup for SequenceBasedComputationsContainer
        self.SequenceBased_container = SequenceBasedComputationsContainer(wcorr_ripple_shuffle=self.wcorr_tool, is_global=True)
        
        # Setup for TrialByTrialActivityResult
        self.trial_by_trial_result = TrialByTrialActivityResult()

    def test_WCorrShuffle_pickle_unpickle(self):
        # Test WCorrShuffle pickling
        saveData('test_wcorr_tool.pkl', (self.wcorr_tool, ), safe_save=False)
        
        # Test WCorrShuffle unpickling
        loaded_wcorr_tool = loadData('test_wcorr_tool.pkl')[0]
        self.assertEqual(self.wcorr_tool, loaded_wcorr_tool)

    def test_SequenceBasedComputationsContainer_pickle_unpickle(self):
        # Test SequenceBasedComputationsContainer pickling
        saveData('test_SequenceBased_container.pkl', (self.SequenceBased_container, ), safe_save=False)
        
        # Test SequenceBasedComputationsContainer unpickling
        loaded_SequenceBased_container = loadData('test_SequenceBased_container.pkl')[0]
        self.assertEqual(self.SequenceBased_container, loaded_SequenceBased_container)

    def test_TrialByTrialActivityResult_pickle_unpickle(self):
        # Test TrialByTrialActivityResult pickling
        saveData('test_trial_by_trial_result.pkl', (self.trial_by_trial_result, ), safe_save=False)
        
        # Test TrialByTrialActivityResult unpickling
        loaded_trial_by_trial_result = loadData('test_trial_by_trial_result.pkl')[0]
        self.assertEqual(self.trial_by_trial_result, loaded_trial_by_trial_result)




if __name__ == "__main__":
    unittest.main()

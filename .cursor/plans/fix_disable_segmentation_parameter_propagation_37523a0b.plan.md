---
name: Fix disable_segmentation parameter propagation
overview: Ensure `disable_segmentation=True` is the default and is passed throughout the entire call hierarchy to prevent crashes when running in parallel. Change all defaults from False to True, fix explicit False calls, and add missing parameters.
todos:
  - id: change_defaults_neuropy
    content: "Change default values in neuropy/core/position.py: perform_segment_trajectories and adding_segmented_trajectories_columns from False to True"
    status: completed
  - id: fix_explicit_false
    content: Fix explicit disable_segmentation=False call in _recompute_high_prob_mask_centroids (line 1441)
    status: completed
  - id: fix_missing_params_pending
    content: Add disable_segmentation=True to missing calls in PendingNotebookCode.py (lines 344, 511)
    status: completed
  - id: add_param_recompute_centroids
    content: Add disable_segmentation parameter to _recompute_high_prob_mask_centroids and propagate it through
    status: completed
  - id: verify_compute_future_past
    content: Verify compute_future_and_past_analysis properly handles disable_segmentation parameter
    status: completed
  - id: verify_all_calls
    content: Verify all call sites throughout codebase pass disable_segmentation correctly
    status: completed
    dependencies:
      - change_defaults_neuropy
      - fix_explicit_false
      - fix_missing_params_pending
      - add_param_recompute_centroids
---

# Fix disable_segmentation parameter propagation

## Problem

The `disable_segmentation` parameter defaults to `False` in several places, and there are calls that either explicitly set it to `False` or omit it entirely. When `disable_segmentation=False`, segmentation runs and crashes the server in parallel execution. We need to ensure `disable_segmentation=True` is the default everywhere and is explicitly passed through the entire call hierarchy.

## Changes Required

### 1. Change Default Values in Function Definitions

**File: `neuropy/core/position.py`**

- Line 657: `perform_segment_trajectories` - Change default from `disable_segmentation: bool = False` to `disable_segmentation: bool = True`
- Line 775: `adding_segmented_trajectories_columns` - Change default from `disable_segmentation: bool = False` to `disable_segmentation: bool = True`
- Update docstrings to reflect the new default

### 2. Fix Explicit `False` Calls

**File: `pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py`**

- Line 1441: Change `disable_segmentation=False` to `disable_segmentation=True` in `_recompute_high_prob_mask_centroids`

### 3. Fix Calls Missing the Parameter

**File: `pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py`**

- Line 344: Add `disable_segmentation=True` to `adding_segmented_trajectories_columns()` call
- Line 511: Add `disable_segmentation=True` to `adding_segmented_trajectories_columns(overwrite_existing=True)` call

### 4. Add Parameter to `_recompute_high_prob_mask_centroids`

**File: `pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py`**

- Line 1429: Add `disable_segmentation: bool = True` parameter to `_recompute_high_prob_mask_centroids` method
- Line 1441: Pass `disable_segmentation=disable_segmentation` to the `adding_segmented_trajectories_columns` call
- Line 1424: Update call site to pass `disable_segmentation=True` (or accept parameter from caller)
- Line 1551: Update call site in `recompute_relevant_position_active_mask_centroid_traj_angle` to pass `disable_segmentation=disable_segmentation`

### 5. Verify `compute_future_and_past_analysis` Parameter Handling

**File: `pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py`**

- Line 3376: Check if `compute_future_and_past_analysis` accepts `disable_segmentation` parameter
- Line 3838: Verify the call passes `disable_segmentation=True` correctly
- Ensure the parameter is passed through to all internal calls that need it

## Implementation Strategy

1. Start with changing defaults in `neuropy/core/position.py` (most fundamental)
2. Fix explicit `False` calls
3. Add missing parameters to calls
4. Update `_recompute_high_prob_mask_centroids` to accept and propagate the parameter
5. Verify all call sites pass the parameter correctly

## Testing Considerations

After changes, verify:

- All calls to segmentation functions explicitly pass `disable_segmentation=True` or use the new default
- No function defaults to `disable_segmentation=False`
- Parameter is propagated through the entire call hierarchy
- Parallel execution no longer crashes
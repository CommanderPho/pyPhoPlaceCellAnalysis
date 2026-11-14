# KDiba Compatibility Changes Report

This document details the exhaustive list of changes between the `pho-diba-2025-paper` branch and the `develop` branch that introduce compatibility guards and branching logic to distinguish KDiba linear track sessions from other session types (e.g., Bapun, Rachel).

## 1. Batch Job Completion and HDF5 Serialization
**Files affected:**
- `src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/BatchCompletionHandler.py`
- `src/pyphoplacecellanalysis/General/Batch/runBatch.py`

**Changes:**
- The base `PipelineCompletionResult` properties specific to KDiba sessions (`long_epoch_name`, `long_laps`, `long_replays`, `short_epoch_name`, `short_laps`, `short_replays`) were decoupled and moved into a new subclass `KDibaPipelineCompletionResult`.
- The corresponding PyTables representation was split into `PipelineCompletionResultTable` and `KDibaPipelineCompletionResultTable`.
- HDF5 output generation in `runBatch.py` now dynamically inspects results to create the correct table schema via `KDibaPipelineCompletionResultTable` when KDiba metrics are present.
- A `from_legacy_pipeline_completion_result` method was added to automatically upgrade legacy pipeline results loaded from memory if they contain KDiba-specific fields.

## 2. Extended Computations Include List
**Files affected:**
- `src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/BatchCompletionHandler.py`
- `src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py`

**Changes:**
- A static frozenset `_KDIBA_ONLY_EXTENDED_COMPUTATIONS` was added to define extended analyses that strictly require KDiba's track layout (e.g., `long_short_decoding_analyses`, `split_to_directional_laps`, `directional_decoders_evaluate_epochs`, `rank_order_shuffle_analysis`).
- If `curr_active_pipeline.is_kdiba_session()` returns `False`, the batch processor now aggressively strips these specific computations from `active_extended_computations_include_includelist` to prevent unhandled exceptions on non-KDiba data.
- `kdiba_session_post_fixup_completion_function` was removed from the default generic user completion tasks and is now injected dynamically for KDiba-only runs.

## 3. KDiba Checks in Pipeline Stages
**Files affected:**
- `src/pyphoplacecellanalysis/General/Pipeline/Stages/Computation.py`

**Changes:**
- Introduced a new top-level `is_kdiba_session()` helper function on both `ComputedPipelineStage` and `PipelineWithComputedPipelineStageMixin`. It assesses whether the session strictly adheres to the KDiba format (`active_sess_config.format_name.lower() == 'kdiba'`).
- Previous un-guarded calls to `self.find_LongShortGlobal_epoch_names()` (which strictly expects a long/short/global trio) were updated to `self.find_Global_epoch_name()` in scenarios where the computation does not depend on the track subdivisions.

## 4. Multi-Context / Epoch Computation Functions
**Files affected:**
- `src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py`
- `src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`
- `src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/RankOrderComputations.py`

**Changes:**
- Extraction of epoch contexts is now wrapped in `if is_kdiba_session:` conditions.
- Directional laps results and directional decoding instantiations in `DirectionalLapsResult.init_from_pipeline_natural_epochs` are guarded from running unconditionally.
- Hardcoded epoch unwrapping (e.g., `long_LR_name, short_LR_name, global_LR_name, ... = ['maze1_odd', 'maze2_odd', ...]`) is conditionally scoped to only KDiba environments.

## 5. GUI Rendering and Plotting Components
**Files affected:**
- `src/pyphoplacecellanalysis/PhoPositionalData/plotting/chunked_2d/SingleArtistMultiEpochBatchHelpers.py`
- `src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/MazeRenderingMixin.py`
- `src/pyphoplacecellanalysis/PhoPositionalData/plotting/chunked_2d/PhoOptimizedMultiEpochBatchRenderer.py`

**Changes:**
- Analytical track shape drawing logic which relied on extracting `long_xlim` and other KDiba-specific geometry boundaries was refactored with `try/except` guards handling `KeyError`.
- For datasets without KDiba properties, these plotting blocks fail gracefully with `WARN: non-kdiba track, cannot draw analytical track shape due to exception e: {e}` rather than crashing the visualization renderer.

## 6. Specific Result Generation and Notebook Code
**Files affected:**
- `src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py`
- `src/pyphoplacecellanalysis/SpecificResults/AcrossSessionResults.py`

**Changes:**
- Refactored `post_process_non_kdiba` and created `final_process_non_kdiba_all_comps` specifically tailored for non-KDiba datasets.
- Explicit checks added enforcing exclusivity between Bapun mode (`is_bapun_mode = curr_active_pipeline is not None`) and KDiba mode (`is_kdiba_mode = directional_laps_results is not None`).
- Functions meant exclusively for Bapun or Rachel datasets now check for session preconditions before executing directional decoding setups that lack `long`/`short` epoch components.

Let me start by exploring both versions' source structures to understand the scope of changes before doing a deep comparison.I'll explore both versions of the pyPhoPlaceCellAnalysis codebase to identify the directory structures and files in the key subdirectories you mentioned.Good structural overview. Now I need to do the deep code comparison — especially in the files that are shared between both versions to find actual breaking changes. Let me focus on the key areas that would break linear track sessions.Both branches share the same remote. Let me find the merge base and get the actual diff between these two versions.Now let me focus on the most critical files for linear track compatibility - the core pipeline files:I now have enough information to build a comprehensive analysis. Let me check a few more critical areas quickly:Now I have a thorough picture of the changes. Let me compile the comprehensive breaking-changes report:Here is the comprehensive report of breaking changes in the `develop` branch that affect KDiba-style linear track sessions:

---

## Breaking Changes in `develop` vs `release/pho-diba-2025-paper` (KDiba sessions)

Both versions share the same GitHub remote. The KDiba version is on `release/pho-diba-2025-paper`; the develop version is on `develop`. Their merge-base is `26f673d1`.

---

### 1. `TrackTemplates` / `BaseTrackTemplates` — structural field refactor (HIGH IMPACT)

**File:** `General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`

**What changed:**

The `TrackTemplates` class was split into `BaseTrackTemplates` (new abstract base) and `TrackTemplates` (concrete subclass). The four named decoder fields were replaced with a single `decoders_dict`:

```python
# KDiba version (old attrs fields):
long_LR_decoder: BasePositionDecoder = serialized_field(repr=False)
long_RL_decoder: BasePositionDecoder = serialized_field(repr=False)
short_LR_decoder: BasePositionDecoder = serialized_field(repr=False)
short_RL_decoder: BasePositionDecoder = serialized_field(repr=False)

# develop version (single dict field in BaseTrackTemplates):
decoders_dict: Dict[types.DecoderName, BasePositionDecoder] = serialized_field(default=Factory(dict), repr=False)
```

`TrackTemplates` re-adds named access as **property getters/setters** that proxy into `decoders_dict`, so attribute-level access (`track_templates.long_LR_decoder`) still works. However:

- **Pickle deserialization breaks** for any `.pkl` files produced with the old class, because attrs expects `long_LR_decoder` as a stored field on `BaseTrackTemplates` but it is now only on the subclass as a property. Older pickles will fail to construct.
- `get_decoder_names()`, `get_LR_decoder_names()`, `get_RL_decoder_names()` were **classmethods** → now **instance methods**. Any code calling `TrackTemplates.get_decoder_names()` (class call) will break.
- `get_decoders()` return type changed: old `DirectionalDecodersTuple` (a namedtuple) → new plain `tuple`. Code that unpacks by name (`result.long_LR`, `result.long_RL`, ...) breaks.
- `init_from_paired_decoders(LR_decoder_pair, RL_decoder_pair)` was **commented out** on `BaseTrackTemplates` and replaced with `init_from_paired_decoders_dicts(LR_decoders_dict, RL_decoders_dict)`. The concrete `TrackTemplates` subclass brings `init_from_paired_decoders` back, so direct construction should still work, but any code calling `BaseTrackTemplates.init_from_paired_decoders(...)` breaks.

---

### 2. `DirectionalLapsResult` — field names refactored (HIGH IMPACT)

**File:** same as above

**What changed:**

`DirectionalLapsResult` now inherits from a new `BaseDirectionalLapsResult`. The four named decoder fields were replaced with two dicts:

```python
# KDiba version (old named attrs fields):
long_LR_one_step_decoder_1D: BasePositionDecoder = ...
long_RL_one_step_decoder_1D: BasePositionDecoder = ...
short_LR_one_step_decoder_1D: BasePositionDecoder = ...
short_RL_one_step_decoder_1D: BasePositionDecoder = ...
long_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = ...
...etc

# develop version (two dicts in BaseDirectionalLapsResult):
one_step_decoder_1D_dict: Dict[str, BasePositionDecoder] = serialized_field(default=Factory(dict))
shared_aclus_only_one_step_decoder_1D_dict: Dict[str, BasePositionDecoder] = serialized_field(default=Factory(dict))
```

The concrete `DirectionalLapsResult` subclass re-exposes the old names as property getters/setters, so attribute access is preserved. But again:

- **Old pickles break**: attrs field layout changed; `__setstate__` on `BaseDirectionalLapsResult` has compatibility handling but it may not cover all edge cases.
- `get_decoder_names()`, `get_LR_decoder_names()`, `get_RL_decoder_names()` are now **instance methods** (same as `TrackTemplates` above).
- `get_templates()` now calls `BaseTrackTemplates.init_from_paired_decoders_dicts(...)` which expects dicts; the old signature `init_from_paired_decoders(LR_pair, RL_pair)` is gone from the base class path.

---

### 3. `PlacefieldComputations` — 3D-only short-circuit (HIGH IMPACT for linear track)

**File:** `General/Pipeline/Stages/ComputationFunctions/PlacefieldComputations.py`

**What changed:**

`_initial_placefield_computation` now inspects `active_session.config.format_name` via `DataSessionFormatRegistryHolder` and checks `format_cls.get_spatial_dimensionality(active_session)`. If the format reports `3`, it skips 1D/2D placefields entirely and computes only `pf3D`:

```python
# develop version:
uses_3d_only = format_cls is not None and format_cls.get_spatial_dimensionality(active_session) == 3
if uses_3d_only:
    prev_output_result.computed_data['pf3D'] = perform_compute_placefields_3d(...)
    prev_output_result.computed_data['pf1D'] = None   # ← BREAKS downstream code!
    prev_output_result.computed_data['pf2D'] = None   # ← BREAKS downstream code!
else:
    prev_output_result.computed_data['pf1D'], prev_output_result.computed_data['pf2D'] = perform_compute_placefields(...)
```

For KDiba linear track sessions (`format_name='kdiba'`), `get_spatial_dimensionality` should return 1 (or 2), so the branch is not taken. **But** if `format_cls` is `None` (registry lookup miss) or if `get_spatial_dimensionality` is undefined on the kdiba format class, the else-branch is taken but `pf1D`/`pf2D` may silently be set to `None`, crashing all downstream consumers.

The same guard is applied in `_initial_time_dependent_placefield_computation`, setting `pf1D_dt` and `pf2D_dt` to `None`.

---

### 4. `DefaultComputationFunctions` — 3D guard + clusterless decoder injection (MEDIUM IMPACT)

**File:** `General/Pipeline/Stages/ComputationFunctions/DefaultComputationFunctions.py`

**What changed:**

- `_perform_position_decoding_computation` now has a 3D-only guard that sets `pf1D_Decoder = None` and `pf2D_Decoder = None` if the session uses 3D placefields. Same caveat as above: a registry miss could silently null out the decoders for kdiba sessions.
- Additionally, a new computation function `_perform_clusterless_position_decoding_computation` was added that imports clusterless decoder modules at the top of the file (`ClusterlessRTCPositionDecoder`, `SpyglassClusterlessDecoder`, etc.). These are new files only present in the develop version. If those modules fail to import (missing dependencies: `replay_trajectory_classification`, etc.), the entire `DefaultComputationFunctions` **module import fails**, which would break the kdiba pipeline entirely.

---

### 5. `NeuropyPipeline` — `find_Global_epoch_name` behavior change (MEDIUM IMPACT)

**File:** `General/Pipeline/NeuropyPipeline.py`

**What changed:**

`find_Global_epoch_name` grew new logic for non-kdiba sessions (`dandi_nwb`, etc.), but the fallback for kdiba remains `find_LongShortGlobal_epoch_names()`. No direct kdiba breakage. However, a new `is_kdiba_session()` method was added (non-breaking).

The pipeline loading path now triggers `NWBDataSessionFormatRegisteredClass._ensure_standard_paradigm_epoch_labels` and similar fixups for NWB formats. These only activate for `format_name in {'dandi_nwb', 'dandi_nwb_001754', 'dandi_nwb_001695'}`, so kdiba is unaffected.

The `skip_save_on_initial_load` bypass logic changed slightly — NWB sessions now force-save even when the flag is set. Again, only affects NWB format sessions.

The HDF export path now checks for `pf3D` and skips `pf1D`/`pf2D` if it exists. If `pf3D` is erroneously populated for a kdiba session, HDF export will silently omit the 1D/2D placefields.

---

### 6. `DecodedFilterEpochsResult` — new field + renamed mask function (MEDIUM IMPACT)

**File:** `Analysis/Decoder/reconstruction.py`

**What changed:**

- New serialized field `decoding_slideby: Optional[float]` was added to `DecodedFilterEpochsResult`. Old pickles lacking this field are migrated in `__setstate__` via `self.decoding_slideby = self.__dict__.get('decoding_time_bin_hop', None)` and `decoding_time_bin_hop` is popped. This **forward-compat migration is present**, so old pickles load fine, but if code explicitly accesses `.decoding_time_bin_hop` on a new-format result it will be missing (KeyError or AttributeError).

- `mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin` signature is unchanged, but internally it now delegates to a new method `mask_computed_DecodedFilterEpochsResult_by_time_bin_inclusion_masks`. **The old method still exists by the same name** — no direct breakage, but callers that saved intermediate results (the return signature's second tuple) will find different tuple structure: previously `(is_time_bin_active_list, ...)`, now `(_out_is_time_bin_active_list, ...)` — same semantic content.

- `compute_marginals_df` now uses a generalized context layout via `DirectionalPseudo2DDecodersResult._resolve_pseudo2D_context_layout()`. For the 4-context (long_LR/long_RL/short_LR/short_RL) case expected by kdiba, the output `epochs_marginals_df` column names are now **dynamically derived** from `context_names` rather than hardcoded. Columns `['long_LR', 'long_RL', 'short_LR', 'short_RL', 'P_LR', 'P_RL', 'P_Long', 'P_Short']` should still appear for standard 4-decoder results, but only if `context_names` resolves correctly.

- `filter_epochs_decoder_result.filter_epochs` is now always coerced through `ensure_dataframe()` in `get_result_for_epoch_subset`, and `original_epoch_idx` is backed up before `reset_index`. Code that relied on the DataFrame index equalling epoch number will still work due to the backup column.

---

### 7. `ZhangReconstructionImplementation.neuropy_bayesian_prob` — new optional parameters (LOW IMPACT)

**File:** `Analysis/Decoder/reconstruction.py`

**What changed:**

New optional parameters added to the core Bayesian decoding function:
- `reliability_modifier_mode` (default `None`)
- `drop_negative_contributing_terms_mode` (default `False`)

These are additive/default-safe. No breakage unless code monkey-patches or inspects the function signature.

---

### 8. `BasePositionDecoder` — new optional fields (LOW IMPACT)

**File:** `Analysis/Decoder/reconstruction.py`

Four new non-serialized fields added (all with defaults):

```python
should_discount_silence: bool = non_serialized_field(default=False)
reliability_active: Optional[np.ndarray] = non_serialized_field(default=None, ...)
reliability_silent: Optional[np.ndarray] = non_serialized_field(default=None, ...)
drop_negative_contributing_terms_mode: bool = non_serialized_field(default=False)
reliability_modifier_mode: ReliabilityDecoderModifierMode = non_serialized_field(default=ReliabilityDecoderModifierMode.IGNORE)
```

All default-safe. Old pickles load fine. No breakage for kdiba sessions. The new `ReliabilityDecoderModifierMode` enum is added at module level — only a problem if it fails to import.

---

### 9. `DirectionalPseudo2DDecodersResult` — marginalization made generic (MEDIUM IMPACT)

**File:** same DirectionalPlacefieldGlobalComputationFunctions.py

**What changed:**

`determine_directional_likelihoods`, `determine_long_short_likelihoods`, `determine_non_marginalized_decoder_likelihoods`, `build_non_marginalized_raw_posteriors`, `build_custom_marginal_over_direction`, `build_custom_marginal_over_long_short` all gained a new optional `context_names: Optional[List[str]] = None` parameter.

More critically, the internal logic now calls `_resolve_pseudo2D_context_layout` and `_infer_direction_group_indices` / `_infer_track_identity_group_indices`. These infer direction/track-identity groupings from context names using string matching (names ending in `_LR`/`_RL` and containing `long`/`short`). For the standard 4-decoder kdiba result with names `['long_LR', 'long_RL', 'short_LR', 'short_RL']`, this should resolve correctly.

However, the `continuously_decoded_pseudo2D_decoder_dict` property type changed:

```python
# KDiba: Dict[float, DecodedFilterEpochsResult]
# develop: Dict[DecodingContinuousCacheKey, DecodedFilterEpochsResult]
#          where DecodingContinuousCacheKey = Tuple[float, float]
```

Cache lookups using a bare `float` key will now miss. Code doing `result.continuously_decoded_pseudo2D_decoder_dict[0.025]` will get a `KeyError`. Old cache dicts (serialized under float keys) are migrated via `coerce_continuously_decoded_cache_dict_keys()` in `__setstate__`, but code at the call site that uses float keys will break.

---

### 10. New imports in `DefaultComputationFunctions` from missing modules (HIGH IMPORT-TIME RISK)

**File:** `General/Pipeline/Stages/ComputationFunctions/DefaultComputationFunctions.py`

At module level, the develop version unconditionally imports:

```python
from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder import ClusterlessRTCPositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import ClusterlessDecodingParameters, ...
from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_decoder import SpyglassClusterlessDecoder
from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_adapters import ...
```

These modules are new in `develop` and depend on `replay_trajectory_classification` (third-party). If `replay_trajectory_classification` is not installed in the KDiba environment, importing `DefaultComputationFunctions` will raise `ModuleNotFoundError`, preventing **all** computations from running.

---

### Summary Table

| # | Area | Severity | Breaks kdiba pickles? | Breaks kdiba runtime? |
|---|------|----------|-----------------------|----------------------|
| 1 | `TrackTemplates` field refactor → `decoders_dict` | HIGH | Yes (old pickles) | `get_decoder_names()` class→instance |
| 2 | `DirectionalLapsResult` field refactor → dicts | HIGH | Yes (old pickles) | Same |
| 3 | Placefield computation 3D branch | HIGH | No | If registry miss: `pf1D=None` |
| 4 | `DefaultComputationFunctions` 3D guard + clusterless imports | HIGH | No | Import fails if `replay_trajectory_classification` absent |
| 5 | `NeuropyPipeline` format-specific fixups | LOW | No | No |
| 6 | `DecodedFilterEpochsResult` `decoding_slideby` + mask refactor | MEDIUM | Migrated in `__setstate__` | `decoding_time_bin_hop` access breaks |
| 7 | Bayesian prob new params | LOW | No | No |
| 8 | `BasePositionDecoder` new fields | LOW | No | No |
| 9 | `DirectionalPseudo2DDecodersResult` cache key type change | MEDIUM | Partially migrated | Float cache key lookups break |
| 10 | New unconditional clusterless module imports | HIGH | No | Import fails if dependency absent |
---
name: Preserve spike raster context title
overview: Keep the spike raster main window title anchored to immutable session context and append dock-specific titles instead of allowing context to be overwritten.
todos:
  - id: add-title-policy-state
    content: Add immutable base context title + optional dock suffix state in Spike3DRasterWindowWidget
    status: completed
  - id: centralize-title-composition
    content: Implement helper(s) to compose and apply main window title from base and suffix
    status: completed
  - id: route-title-writes-through-policy
    content: Update setWindowTitle and initialization paths to preserve base context semantics
    status: completed
  - id: remove-legacy-overwrite-calls
    content: Replace direct overwriting setWindowTitle calls in SpikeRasters.py with policy-aware usage
    status: completed
  - id: verify-core-title-flows
    content: Validate startup, dock-add, and legacy menu paths produce expected composed titles
    status: completed
isProject: false
---

# Preserve Spike Raster Window Context Title

## Goal
Ensure `spike_raster_window` always retains its original session-context title. Any later title changes (especially from dock additions/menus) should be composed as:
`{original_sess_context} - {dock_desired_title}`.

## Findings
- Initial context-aware title is set in [`C:/Users/pho/repos/EmotivEpoc/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/SpikeRasterWindows/Spike3DRasterWindowWidget.py`](C:/Users/pho/repos/EmotivEpoc/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/SpikeRasterWindows/Spike3DRasterWindowWidget.py), including `find_or_create_if_needed(...)`.
- A legacy/alternate display path in [`C:/Users/pho/repos/EmotivEpoc/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/SpikeRasters.py`](C:/Users/pho/repos/EmotivEpoc/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/SpikeRasters.py) sets `setWindowTitle(...)` directly and can overwrite context title.
- Dock creation/rename flows set dock titles, but there is no single policy layer that composes main-window title from immutable base context + dock title.

## Implementation Plan
1. In `Spike3DRasterWindowWidget`, introduce explicit title policy state:
   - Store immutable base context title once (e.g., `params.base_context_title`) when session context is first available.
   - Store optional dynamic suffix title (e.g., `params.dynamic_dock_title`).

2. Add a centralized recomposition helper in `Spike3DRasterWindowWidget`:
   - `compose_window_title(base_context_title, dynamic_dock_title)`
   - Returns base alone if no suffix, else `f"{base_context_title} - {dynamic_dock_title}"`.
   - Route all window-title writes through this helper.

3. Update `Spike3DRasterWindowWidget.setWindowTitle(...)` behavior to preserve context:
   - Treat incoming non-base titles as candidate dynamic suffixes.
   - Prevent direct replacement of base context.
   - Keep an internal explicit method for setting base context during initialization (`set_base_context_title(...)`) to avoid recursion/ambiguity.

4. In `SpikeRasters.py`, stop direct context-overwriting title writes:
   - Replace direct `setWindowTitle(f"Spike Raster Window - {active_config_name} - {a_file_prefix}")` calls with policy-aware call(s) on the window widget.
   - If needed, pass only dock/extra descriptor as suffix and let widget compose the final title.

5. Add/adjust dock-add integration (if needed for desired UX):
   - Where menu/command flows create new docks, pass the desired dock title into the policy (suffix setter) instead of assigning full window title.
   - Ensure close/reset behavior can fall back to base context-only title when no relevant dock suffix exists.

6. Verify behavior paths:
   - Initial launch still shows session-context title.
   - Adding a dock from menu yields `base_context - dock_title`.
   - Legacy display/menu paths no longer erase session context.
   - Re-adding/renaming docks maintains the composed title format.

## Notes
- Primary change points are confined to:
  - [`C:/Users/pho/repos/EmotivEpoc/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/SpikeRasterWindows/Spike3DRasterWindowWidget.py`](C:/Users/pho/repos/EmotivEpoc/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/SpikeRasterWindows/Spike3DRasterWindowWidget.py)
  - [`C:/Users/pho/repos/EmotivEpoc/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/SpikeRasters.py`](C:/Users/pho/repos/EmotivEpoc/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/SpikeRasters.py)
- This avoids broad refactors of dock internals while enforcing a stable single source of truth for main-window title composition.
# Refactoring Plan: PredictiveDecodingDisplayWidget

## Current Issues
1. **Lazy initialization**: Widgets created on-demand in `update_displayed_epoch()` causing:
   - Hard-to-diagnose errors when widgets don't exist
   - Inconsistent state between initialization and updates
   - Performance issues from repeated creation checks

2. **Poor separation of concerns**: 
   - All logic lumped into one massive `update_displayed_epoch()` method (~500 lines)
   - Mixed initialization and update logic
   - Helper functions defined inside methods

3. **Inconsistent widget management**:
   - Some widgets stored in `dock_canvas_widgets`, some in `trajectory_displaying_plotter`
   - Different update patterns for different widget types
   - No clear ownership model

## Proposed Structure

Following the pattern of `TimeSynchronizedPlotterBase` and similar widget classes:

```
PredictiveDecodingDisplayWidget
├── __attrs_post_init__()      # Basic validation and setup
├── setup()                     # Initialize data structures, calculate constants
├── buildUI()                   # Create dock area, initialize all widgets
│   ├── _build_dock_area()     # Create dock window and area
│   ├── _build_past_widget()   # Initialize past trajectory widget
│   ├── _build_posterior_widget() # Initialize decoded posterior widget
│   ├── _build_future_widget() # Initialize future trajectory widget
│   └── _build_epoch_control()  # Create slider and controls
├── update_displayed_epoch()    # Main update entry point
│   ├── _prepare_epoch_data()  # Extract and prepare data for current epoch
│   ├── _update_past_widget()  # Update past trajectory display
│   ├── _update_posterior_widget() # Update decoded posterior display
│   └── _update_future_widget() # Update future trajectory display
└── Helper methods
    ├── _calculate_max_subplots() # Calculate max subplots needed
    ├── _get_epoch_data()         # Extract data for specific epoch
    └── _update_posterior_plot()   # Update posterior plot (extracted helper)
```

## Key Improvements

### 1. Initialize All Widgets on `buildUI()`
- All three widgets (past, posterior, future) created immediately
- No lazy initialization checks needed
- Consistent state from start

### 2. Clear Separation of Concerns
- **Setup phase**: Data preparation, constants calculation
- **Build phase**: UI creation, widget initialization
- **Update phase**: Data refresh, plot updates

### 3. Consistent Widget Storage
```python
# Single source of truth for all display widgets
self.display_widgets: Dict[str, Any] = {
    'past': MatplotlibTimeSynchronizedWidget,
    'posterior': MatplotlibTimeSynchronizedWidget, 
    'future': MatplotlibTimeSynchronizedWidget
}

# Separate storage for trajectory plotters (if needed)
self.trajectory_plotters: Dict[str, DecodedTrajectoryMatplotlibPlotter] = {
    'past': DecodedTrajectoryMatplotlibPlotter,
    'future': DecodedTrajectoryMatplotlibPlotter
}
```

### 4. Simplified Update Logic
```python
def update_displayed_epoch(self, an_epoch_idx: int):
    """Update all displays for the given epoch index"""
    # Validate and prepare
    an_epoch_idx = self._validate_epoch_idx(an_epoch_idx)
    epoch_data = self._prepare_epoch_data(an_epoch_idx)
    
    # Update all widgets (they always exist)
    self._update_past_widget(epoch_data)
    self._update_posterior_widget(epoch_data)
    self._update_future_widget(epoch_data)
    
    # Update slider
    self._update_slider(an_epoch_idx)
```

### 5. Performance Optimizations
- Pre-calculate `max_subplots_per_category` once in `setup()`
- Cache extent calculation
- Avoid repeated widget lookups
- Use `draw_idle()` for non-critical updates

## Implementation Steps

1. **Extract helper methods** from `update_displayed_epoch()`:
   - `_prepare_epoch_data()`
   - `_calculate_max_subplots()`
   - `_get_posterior_data()`
   - `_update_posterior_plot()` (already exists as `_subfn_update_posterior_plot`)

2. **Refactor `buildUI()`**:
   - Create all three widgets immediately
   - Store references consistently
   - Remove placeholder widgets

3. **Simplify `update_displayed_epoch()`**:
   - Remove all `needed_init` checks
   - Call update methods for each widget
   - Remove widget creation logic

4. **Add update methods**:
   - `_update_past_widget(epoch_data)`
   - `_update_posterior_widget(epoch_data)`
   - `_update_future_widget(epoch_data)`

5. **Update `__attrs_post_init__`**:
   - Call `setup()` then `buildUI()`
   - Remove `init_UI()` call

## Benefits

✅ **Easier to debug**: Widgets always exist, no state checks needed
✅ **Better performance**: No repeated initialization checks
✅ **Clearer code**: Each method has single responsibility
✅ **Easier to extend**: Add new widgets by adding build/update methods
✅ **Consistent patterns**: Follows established widget class patterns
✅ **Type safety**: Clear widget types, easier to reason about

## Migration Notes

- Keep `dock_widgets` for dock references (needed for layout)
- Rename `dock_canvas_widgets` to `display_widgets` for clarity
- Keep `trajectory_displaying_plotter` but rename to `trajectory_plotters`
- All widgets use `MatplotlibTimeSynchronizedWidget` for consistency

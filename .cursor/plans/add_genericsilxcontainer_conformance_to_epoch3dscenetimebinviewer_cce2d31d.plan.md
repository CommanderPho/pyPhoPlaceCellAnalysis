---
name: Add GenericSilxContainer conformance to Epoch3DSceneTimeBinViewer
overview: Modify Epoch3DSceneTimeBinViewer to properly initialize both GenericSilxContainer (attrs-based) and qt.QWidget by explicitly initializing both in the correct order, extracting attrs field kwargs, and maintaining all existing functionality.
todos: []
---

# Add GenericSilxContainer Conformance to Epoch3DSceneTimeBinViewer

## Overview

The `Epoch3DSceneTimeBinViewer` class currently inherits from both `GenericSilxContainer` and `qt.QWidget` but needs proper initialization to conform to the attrs-based `GenericSilxContainer` without breaking existing functionality.

## Current State

- Class inherits from `GenericSilxContainer, qt.QWidget` (line 704)
- `__init__` calls `super().__init__()` with no arguments (line 743)
- All attrs fields in `GenericSilxContainer` have defaults, so initialization works but doesn't allow customization

## Required Changes

### 1. Modify `__init__` Method ([EpochTimeBinViewerWidget.py:742-809](src/pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py))

**Changes needed:**

- Extract attrs field kwargs (`name`, `plots`, `plot_data`, `ui`, `params`) from `**kwargs`
- Explicitly initialize `GenericSilxContainer` first with attrs kwargs
- Explicitly initialize `qt.QWidget` with remaining kwargs (e.g., `parent`)
- Keep all existing instance variable assignments and UI setup code unchanged

**Implementation approach:**

```python
def __init__(self, decoded_result, xbin_centers=None, ybin_centers=None, 
             locality_measures_df: Optional[pd.DataFrame] = None, 
             text_columns: Optional[List[str]] = None, 
             text_data_provider: Optional[TextDataProviderDatasource] = None,
             **kwargs):
    # Extract attrs field names from GenericSilxContainer
    attrs_field_names = {'name', 'plots', 'plot_data', 'ui', 'params'}
    attrs_kwargs = {k: v for k, v in kwargs.items() if k in attrs_field_names}
    qt_kwargs = {k: v for k, v in kwargs.items() if k not in attrs_field_names}
    
    # Initialize attrs class first
    GenericSilxContainer.__init__(self, **attrs_kwargs)
    
    # Initialize Qt widget
    qt.QWidget.__init__(self, **qt_kwargs)
    
    # Rest of existing initialization code unchanged...
```

### 2. Verify Field Access

- Ensure existing code that might access `self.plots`, `self.plot_data`, `self.ui`, `self.params` continues to work
- These are now provided by `GenericSilxContainer` with defaults, so existing code should work

## Testing Considerations

- Verify widget still displays correctly
- Test that attrs fields can be customized via kwargs
- Ensure Qt widget parent parameter works correctly
- Verify all existing functionality (sliders, scene window, labels, etc.) still works

## Files to Modify

- `src/pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py` - Modify `__init__` method (lines 742-809)
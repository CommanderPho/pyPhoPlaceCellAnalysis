---
name: Categorical colormap with saturation fade
overview: Create a helper function that generates a custom color function for multi_trajectory_color_plotter. Each trajectory gets a unique color from a categorical colormap, and saturation fades from 0.9 (high) at t_start to 0.3 (low) at t_end to show direction.
todos:
  - id: "1"
    content: Create create_categorical_saturation_fade_color_fn helper function with pre-computation of trajectory time ranges
    status: completed
  - id: "2"
    content: Implement categorical colormap loading and trajectory color assignment
    status: completed
  - id: "3"
    content: Implement color function closure with HSV saturation adjustment based on normalized time
    status: completed
  - id: "4"
    content: Add edge case handling (no time data, single points, etc.)
    status: completed
  - id: "5"
    content: Add usage example to multi_trajectory_color_plotter docstring
    status: completed
isProject: false
---

# Plan: Categorical Colormap with Saturation Fade

## Overview

Create a helper function `create_categorical_saturation_fade_color_fn` that generates a color function for `multi_trajectory_color_plotter`. This function will:

1. Assign each trajectory a unique color from a categorical colormap (e.g., matplotlib's 'tab10')
2. For each point, fade saturation from 0.9 (at trajectory start) to 0.3 (at trajectory end) based on normalized time
3. Preserve the hue and value of the base color, only adjusting saturation

## Implementation Details

### File to Modify

- `[PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)`

### Steps

1. **Add helper function after `multi_trajectory_color_plotter` function**
  - Function name: `create_categorical_saturation_fade_color_fn`
  - Parameters:
    - `position_dfs`: List of position dataframes (to compute t_start/t_end per trajectory)
    - `categorical_cmap`: Optional string name of categorical colormap (default: 'tab10')
    - `saturation_start`: Optional float (default: 0.9)
    - `saturation_end`: Optional float (default: 0.3)
  - Returns: A color function compatible with `color_fn` parameter
2. **Function logic:**
  - Pre-compute t_start[i] and t_end[i] for each trajectory from position_dfs
  - Load categorical colormap (matplotlib 'tab10' or similar)
  - Assign each trajectory a unique color from the colormap
  - Return a closure that:
    - Takes (x, y, t, trajectory_idx, point_idx, df) as parameters
    - Gets the base color for trajectory_idx
    - Normalizes t relative to t_start[trajectory_idx] and t_end[trajectory_idx]
    - Interpolates saturation from saturation_start to saturation_end based on normalized time
    - Converts HSV: (hue, saturation, value) -> QColor
    - Handles edge cases (no time data, single point trajectories, etc.)
3. **Color space conversion:**
  - Use QColor's HSV color space for saturation adjustment
  - Convert base color (from colormap) to HSV
  - Adjust saturation while preserving hue and value
  - Convert back to RGB/QColor
4. **Edge case handling:**
  - If trajectory has no time data: use base color with default saturation
  - If t_start == t_end: use saturation_start
  - If t is None or NaN: use saturation_start
  - If trajectory_idx is out of range: fallback to default color
5. **Add usage example to docstring:**
  - Show how to use the helper function with `multi_trajectory_color_plotter`

## Technical Notes

- Use matplotlib's categorical colormaps (tab10, Set3, etc.) which provide distinct colors
- QColor supports HSV color space via `getHsvF()` and `setHsvF()` methods
- Saturation in HSV ranges from 0.0 (grayscale) to 1.0 (fully saturated)
- The function will be a closure that captures the pre-computed trajectory time ranges and color assignments


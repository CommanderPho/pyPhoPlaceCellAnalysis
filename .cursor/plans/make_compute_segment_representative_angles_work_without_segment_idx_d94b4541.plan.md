---
name: Make compute_segment_representative_angles work without segment_idx
overview: Modify `compute_segment_representative_angles` to handle cases where 'segment_idx' column doesn't exist by automatically creating it with all values set to 0 (treating all rows as a single segment).
todos:
  - id: update_assertion
    content: Modify assertion to only require 'Vp' column, not 'segment_idx'
    status: completed
  - id: add_segment_idx_check
    content: Add logic to create 'segment_idx' column with all zeros if it doesn't exist
    status: completed
    dependencies:
      - update_assertion
  - id: update_docstring
    content: Update docstring to reflect that 'segment_idx' is optional and will be created if missing
    status: completed
---

# Make compute_segment_representative_angles work without segment_idx column

## Changes to `neuropy/core/position.py`

### Modify `compute_segment_representative_angles` method (lines 613-648)

1. **Update the assertion** (line 624):

- Change from requiring both 'Vp' and 'segment_idx' to only requiring 'Vp'
- Remove 'segment_idx' from the assertion check

2. **Add segment_idx creation logic** (after the assertion, before line 626):

- Check if 'segment_idx' column exists in `pos_df`
- If it doesn't exist, create it with all values set to 0: `pos_df['segment_idx'] = 0`
- This treats all rows as belonging to the same segment

3. **Update the docstring** (lines 615-622):

- Change the Args section to indicate that 'segment_idx' is optional
- Update the description to mention that if 'segment_idx' is missing, all rows are treated as segment 0

### Implementation details

The function will:

- Still require 'Vp' column (assertion remains for this)
- Check for 'segment_idx' column existence
- If missing, create it with `pos_df['segment_idx'] = 0` before processing
- Continue with existing logic using groupby on 'segment_idx'

This change makes the function more flexible and allows it to compute representative angles even when segmentation hasn't been performed.
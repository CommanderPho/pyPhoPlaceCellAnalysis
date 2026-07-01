1. Add missing property methods to `ClusterlessRTCPositionDecoder` inside `rtc_clusterless_decoder.py` so that it conforms to `BayesianPlacemapPositionDecoder`.
   These properties are:
   - `time_window_edges_binning_info`
   - `time_window_center_binning_info`
   - `active_time_windows`
   - `active_time_window_centers`
   - `P_x_given_n`

2. Run pre-commit instructions
3. Submit the changes.

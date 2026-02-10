import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Import the specialized packages with proper error handling
try:
    # Try different import patterns for bayesian-changepoint-detection
    try:
        # from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
        # from bayesian_changepoint_detection.priors import const_prior
        # from bayesian_changepoint_detection.distributions import StudentT
        import torch
        import time
        from functools import partial

        # Import the refactored modules
        from bayesian_changepoint_detection import (
            online_changepoint_detection,
            get_device,
            get_device_info,
            to_tensor,
        )
        from bayesian_changepoint_detection import offline_changepoint_detection
        from bayesian_changepoint_detection.priors import const_prior
        from bayesian_changepoint_detection.online_likelihoods import StudentT, MultivariateT
        from bayesian_changepoint_detection.hazard_functions import constant_hazard

        BOCD_AVAILABLE = True
        print("✓ bayesian-changepoint-detection loaded successfully")
    except ImportError as e:
        print(f"Warning: bayesian-changepoint-detection import failed: {e}")
        BOCD_AVAILABLE = False
except:
    BOCD_AVAILABLE = False

# try:
#     import clustpy
#     CLUSTPY_AVAILABLE = True
#     print("✓ clustpy loaded successfully")
# except ImportError as e:
#     print(f"Warning: clustpy import failed: {e}")
#     CLUSTPY_AVAILABLE = False
#     from sklearn.cluster import DBSCAN
    

CLUSTPY_AVAILABLE = False
from sklearn.cluster import DBSCAN

try:
    from bcubed import bcubed
    BCUBED_AVAILABLE = True
    print("✓ bcubed loaded successfully")
except ImportError:
    BCUBED_AVAILABLE = False

class OptimizedMovementBurstDetector:
    """
    Optimized movement burst detector using specialized packages.
    Fixed version with correct PyTorch usage and robust imports.
    """
    
    def __init__(self, min_burst_duration=0.5, min_rest_duration=1.0,
                 velocity_smoothing=0.1, bocd_hazard=100, 
                 clustering_method='dbscan', use_gpu=False):
        """
        Parameters:
        -----------
        min_burst_duration : float
            Minimum duration for a burst (seconds)
        min_rest_duration : float
            Minimum duration for rest periods (seconds)
        velocity_smoothing : float
            Gaussian smoothing sigma for velocity (seconds)
        bocd_hazard : float
            Hazard rate for BOCD (higher = more sensitive)
        clustering_method : str
            'dbscan', 'optics', or 'hdbscan'
        use_gpu : bool
            Use GPU acceleration if available
        """
        self.min_burst_duration = min_burst_duration
        self.min_rest_duration = min_rest_duration
        self.velocity_smoothing = velocity_smoothing
        self.bocd_hazard = bocd_hazard
        self.clustering_method = clustering_method
        self.use_gpu = use_gpu
        
        # Check GPU availability
        if self.use_gpu:
            try:
                import torch
                self.torch = torch
                self.has_gpu = torch.cuda.is_available()
                if self.has_gpu:
                    self.device = torch.device('cuda')
                    print(f"✓ GPU acceleration available: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = torch.device('cpu')
                    print("✗ GPU requested but not available. Using CPU.")
            except ImportError:
                self.has_gpu = False
                self.device = None
                print("✗ PyTorch not available. Using CPU only.")
        else:
            self.has_gpu = False
            self.device = None
    
    def preprocess_trajectory(self, pos_df):
        """Robust trajectory preprocessing with proper interpolation"""
        df = pos_df.copy().sort_values('t').reset_index(drop=True)
        
        # Handle duplicate timestamps
        if df['t'].duplicated().any():
            df = df.groupby('t').mean().reset_index()
        
        # Get original data
        t_original = df['t'].values
        x_original = df['x'].values
        y_original = df['y'].values
        
        # Calculate dt from median difference
        if len(t_original) > 1:
            dt = np.median(np.diff(t_original))
        else:
            dt = 0.033  # Default 30Hz if we can't calculate
        
        # Create regular time grid
        t_min, t_max = t_original.min(), t_original.max()
        t_reg = np.arange(t_min, t_max, dt)
        
        # Use scipy's interpolation (more robust than torch.interp)
        if len(t_original) >= 2:
            # Linear interpolation for x and y
            interp_x = interpolate.interp1d(t_original, x_original, 
                                           kind='linear', 
                                           fill_value='extrapolate',
                                           bounds_error=False)
            interp_y = interpolate.interp1d(t_original, y_original, 
                                           kind='linear', 
                                           fill_value='extrapolate',
                                           bounds_error=False)
            
            x_reg = interp_x(t_reg)
            y_reg = interp_y(t_reg)
        else:
            # Not enough points, return original
            x_reg = x_original
            y_reg = y_original
            t_reg = t_original
        
        result = pd.DataFrame({
            't': t_reg,
            'x': x_reg,
            'y': y_reg
        })
        
        return result
    

    def compute_movement_features(self, pos_df):
        """Extract comprehensive movement features"""
        t = pos_df['t'].values
        x = pos_df['x'].values
        y = pos_df['y'].values
        
        # Calculate dt
        if len(t) > 1:
            dt = np.mean(np.diff(t))
        else:
            dt = 0.033  # Default 30Hz
        
        # Compute velocity
        if len(x) > 1:
            vx = np.gradient(x, dt)
            vy = np.gradient(y, dt)
            speed = np.sqrt(vx**2 + vy**2)
        else:
            vx = np.zeros_like(x)
            vy = np.zeros_like(y)
            speed = np.zeros_like(x)
        
        # Apply smoothing
        if len(speed) > 1:
            sigma = max(1.0, self.velocity_smoothing / dt)  # Ensure sigma >= 1
            speed_smooth = gaussian_filter1d(speed, sigma=sigma)
        else:
            speed_smooth = speed.copy()
        
        # Compute acceleration and jerk
        if len(speed_smooth) > 1:
            acc = np.gradient(speed_smooth, dt)
            jerk = np.gradient(acc, dt)
        else:
            acc = np.zeros_like(speed_smooth)
            jerk = np.zeros_like(speed_smooth)
        
        # Compute angular velocity
        if len(vx) > 1 and len(vy) > 1:
            heading = np.arctan2(vy, vx)
            angular_vel = np.gradient(heading, dt)
        else:
            angular_vel = np.zeros_like(speed_smooth)
        
        # Compute local variance
        if len(speed_smooth) > 10:
            window = max(3, int(0.5/dt))  # 500ms window
            local_var = pd.Series(speed_smooth).rolling(window=window, 
                                                       center=True, 
                                                       min_periods=1).std().values
        else:
            local_var = np.zeros_like(speed_smooth)
        
        # Smooth local variance
        if len(local_var) > 1:
            local_var_smooth = gaussian_filter1d(local_var, sigma=2)
        else:
            local_var_smooth = local_var
        
        # Create feature matrix
        features = []
        
        # Speed feature
        if np.std(speed_smooth) > 0:
            features.append(speed_smooth / np.std(speed_smooth))
        else:
            features.append(speed_smooth)
        
        # Acceleration feature
        if len(acc) > 0 and np.std(acc) > 0:
            features.append(acc / np.std(acc))
        else:
            features.append(acc)
        
        # Jerk feature
        if len(jerk) > 0 and np.std(jerk) > 0:
            features.append(jerk / np.std(jerk))
        else:
            features.append(jerk)
        
        # Angular velocity feature
        if len(angular_vel) > 0 and np.std(np.abs(angular_vel)) > 0:
            features.append(np.abs(angular_vel) / np.std(np.abs(angular_vel)))
        else:
            features.append(np.abs(angular_vel))
        
        # Local variance feature
        if len(local_var_smooth) > 0 and np.std(local_var_smooth) > 0:
            features.append(local_var_smooth / np.std(local_var_smooth))
        else:
            features.append(local_var_smooth)
        
        # Stack all features
        if features:
            features_matrix = np.column_stack(features)
        else:
            features_matrix = np.zeros((len(speed_smooth), 1))
        
        return {
            't': t,
            'speed': speed,
            'speed_smooth': speed_smooth,
            'features': features_matrix,
            'dt': dt,
            'vx': vx,
            'vy': vy,
            'acc': acc,
            'jerk': jerk,
            'angular_vel': angular_vel
        }
    

    def bocd_detection_package(self, features_dict):
        """Use bayesian-changepoint-detection package for BOCD"""
        if not BOCD_AVAILABLE:
            print("Using fallback BOCD detection")
            return self.bocd_detection_fallback(features_dict)
        
        speed = features_dict['speed_smooth']
        
        try:
            # Convert to numpy array if needed
            if hasattr(speed, 'cpu'):
                speed = speed.cpu().numpy()
            
            # Use Student's t-distribution (robust to outliers)
            # Note: We're using the hazard constant directly
            hazard_const = self.bocd_hazard
            
            # Run offline changepoint detection
            Q, P, Pcp = offline_changepoint_detection(
                speed,
                lambda x: const_prior(x, hazard_const),
                StudentT(alpha=1.0, beta=1.0, kappa=1.0, mu=0.0),
                truncate=-40
            )
            
            # Extract changepoints from probability matrix
            # The changepoint probability is in Pcp
            changepoint_probs = np.exp(Pcp).sum(axis=0) / (np.exp(Pcp).sum() + 1e-10)
            
            # Find peaks in changepoint probability
            if len(changepoint_probs) > 10:
                min_distance = max(1, int(self.min_rest_duration / features_dict['dt']))
                peaks, properties = signal.find_peaks(
                    changepoint_probs,
                    height=np.percentile(changepoint_probs, 75),
                    distance=min_distance
                )
                changepoints = peaks.tolist()
            else:
                # Simple threshold if not enough points
                threshold = np.median(changepoint_probs) + np.std(changepoint_probs)
                changepoints = np.where(changepoint_probs > threshold)[0].tolist()
            
        except Exception as e:
            print(f"BOCD package failed with error: {e}")
            print("Using fallback detection")
            return self.bocd_detection_fallback(features_dict)
        
        # Create segments from changepoints
        segments = self.create_segments_from_changepoints(features_dict, changepoints)
        
        return segments, changepoints
    
    def bocd_detection_fallback(self, features_dict):
        """Fallback BOCD implementation using simpler method"""
        t = features_dict['t']
        speed = features_dict['speed_smooth']
        
        if len(speed) < 10:
            # Not enough data points
            segments = [{
                'start': t[0],
                'end': t[-1],
                'duration': t[-1] - t[0],
                'mean_speed': np.mean(speed) if len(speed) > 0 else 0,
                'std_speed': np.std(speed) if len(speed) > 1 else 0,
                'bocd_prob': 0.0
            }]
            return segments, []
        
        # Simple gradient-based changepoint detection
        gradient = np.abs(np.gradient(speed))
        
        # Smooth the gradient
        if len(gradient) > 10:
            smoothed_gradient = gaussian_filter1d(gradient, sigma=3)
        else:
            smoothed_gradient = gradient
        
        # Find peaks in gradient
        min_distance = max(1, int(self.min_rest_duration / features_dict['dt']))
        
        if len(smoothed_gradient) > min_distance * 2:
            peaks, properties = signal.find_peaks(
                smoothed_gradient,
                height=np.percentile(smoothed_gradient, 70),
                distance=min_distance
            )
            changepoints = peaks.tolist()
        else:
            # Use threshold if not enough points for peak finding
            threshold = np.median(smoothed_gradient) + 0.5 * np.std(smoothed_gradient)
            changepoints = np.where(smoothed_gradient > threshold)[0].tolist()
        
        # Create segments
        segments = self.create_segments_from_changepoints(features_dict, changepoints)
        
        return segments, changepoints
    
    def create_segments_from_changepoints(self, features_dict, changepoints):
        """Helper function to create segments from changepoint indices"""
        t = features_dict['t']
        speed = features_dict['speed_smooth']
        
        segments = []
        prev_idx = 0
        
        # Sort changepoints and remove duplicates
        changepoints = sorted(set(changepoints))
        changepoints = [cp for cp in changepoints if 0 < cp < len(t)]
        
        for cp_idx in changepoints:
            if cp_idx <= prev_idx:
                continue
                
            seg_t = t[prev_idx:cp_idx]
            seg_speed = speed[prev_idx:cp_idx]
            
            if len(seg_t) > 0:
                segments.append({
                    'start': seg_t[0],
                    'end': seg_t[-1],
                    'duration': seg_t[-1] - seg_t[0],
                    'mean_speed': np.mean(seg_speed) if len(seg_speed) > 0 else 0,
                    'std_speed': np.std(seg_speed) if len(seg_speed) > 1 else 0,
                    'bocd_prob': 0.0
                })
            prev_idx = cp_idx
        
        # Add last segment
        if prev_idx < len(t):
            seg_t = t[prev_idx:]
            seg_speed = speed[prev_idx:]
            segments.append({
                'start': seg_t[0],
                'end': seg_t[-1],
                'duration': seg_t[-1] - seg_t[0],
                'mean_speed': np.mean(seg_speed) if len(seg_speed) > 0 else 0,
                'std_speed': np.std(seg_speed) if len(seg_speed) > 1 else 0,
                'bocd_prob': 0.0
            })
        
        return segments
    

    def clustpy_clustering(self, features_dict, segments):
        """Use clustpy or sklearn for clustering of movement states"""
        if len(segments) < 2:
            # Not enough segments to cluster
            for seg in segments:
                seg['is_burst'] = False
                seg['cluster'] = -1
            return segments
        
        # Extract features from segments
        segment_features = []
        valid_segment_indices = []
        
        for i, seg in enumerate(segments):
            mask = (features_dict['t'] >= seg['start']) & (features_dict['t'] <= seg['end'])
            if np.sum(mask) > 0:
                # Get mean feature vector for this segment
                seg_feat = np.mean(features_dict['features'][mask], axis=0)
                segment_features.append(seg_feat)
                valid_segment_indices.append(i)
        
        if len(segment_features) < 2:
            # Not enough features for clustering
            for seg in segments:
                seg['is_burst'] = False
                seg['cluster'] = -1
            return segments
        
        X = np.array(segment_features)
        
        try:
            if CLUSTPY_AVAILABLE and hasattr(clustpy, 'partition'):
                # Try to use clustpy's DBSCAN
                if self.clustering_method == 'dbscan':
                    from clustpy.partition import DBSCAN as ClustpyDBSCAN
                    dbscan = ClustpyDBSCAN(eps=0.5, min_samples=2)
                    labels = dbscan.fit_predict(X)
                else:
                    # Fallback to sklearn
                    from sklearn.cluster import DBSCAN
                    dbscan = DBSCAN(eps=0.5, min_samples=2)
                    labels = dbscan.fit_predict(X)
            else:
                # Use sklearn DBSCAN
                from sklearn.cluster import DBSCAN
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                labels = dbscan.fit_predict(X)
            
            # Apply labels to segments
            for idx, seg_idx in enumerate(valid_segment_indices):
                segments[seg_idx]['cluster'] = int(labels[idx])
                # Mark as burst if not noise (-1)
                segments[seg_idx]['is_burst'] = (labels[idx] != -1)
            
            # Identify burst clusters (clusters with highest average speed)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)  # Remove noise label
            
            if len(unique_labels) > 0:
                cluster_speeds = []
                for lbl in unique_labels:
                    # Get segments with this label
                    cluster_seg_indices = [valid_segment_indices[i] for i, l in enumerate(labels) if l == lbl]
                    cluster_segments = [segments[i] for i in cluster_seg_indices]
                    avg_speed = np.mean([s['mean_speed'] for s in cluster_segments])
                    cluster_speeds.append((lbl, avg_speed))
                
                # Mark top clusters as bursts
                if cluster_speeds:
                    # Sort by speed (descending)
                    cluster_speeds.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top N clusters as bursts (at least 1, at most all)
                    n_burst_clusters = min(max(1, len(cluster_speeds) // 2), len(cluster_speeds))
                    burst_clusters = [cs[0] for cs in cluster_speeds[:n_burst_clusters]]
                    
                    # Update burst status
                    for seg in segments:
                        if 'cluster' in seg:
                            seg['is_burst'] = seg['cluster'] in burst_clusters
            
        except Exception as e:
            print(f"Clustering failed with error: {e}")
            print("Using simple threshold-based classification")
            
            # Fallback: classify based on speed threshold
            speeds = [s['mean_speed'] for s in segments]
            if len(speeds) > 0:
                threshold = np.percentile(speeds, 60)
                for seg in segments:
                    seg['is_burst'] = seg['mean_speed'] > threshold
                    seg['cluster'] = 0 if seg['is_burst'] else -1
        
        return segments
    

    def detect_bursts(self, pos_df, ground_truth=None):
        """Main detection pipeline using specialized packages"""
        print("Starting burst detection pipeline...")
        
        # 1. Preprocess trajectory
        print("  Step 1: Preprocessing trajectory...")
        df_clean = self.preprocess_trajectory(pos_df)
        print(f"    Original: {len(pos_df)} points -> Clean: {len(df_clean)} points")
        
        # 2. Extract movement features
        print("  Step 2: Computing movement features...")
        features_dict = self.compute_movement_features(df_clean)
        print(f"    Computed {features_dict['features'].shape[1]} features")
        
        # 3. Detect changepoints using BOCD
        print("  Step 3: Running Bayesian changepoint detection...")
        segments, changepoints = self.bocd_detection_package(features_dict)
        print(f"    Found {len(changepoints)} changepoints, creating {len(segments)} segments")
        
        # 4. Cluster segments
        print("  Step 4: Clustering segments...")
        segments = self.clustpy_clustering(features_dict, segments)
        
        # 5. Extract burst segments
        burst_segments = []
        for seg in segments:
            if seg.get('is_burst', False) and seg['duration'] >= self.min_burst_duration:
                # Add additional metrics
                mask = (df_clean['t'] >= seg['start']) & (df_clean['t'] <= seg['end'])
                if np.sum(mask) > 1:
                    seg_pos = df_clean[mask]
                    
                    # Calculate movement metrics
                    if len(seg_pos) > 1:
                        dx = np.diff(seg_pos['x'].values)
                        dy = np.diff(seg_pos['y'].values)
                        total_distance = np.sum(np.sqrt(dx**2 + dy**2))
                        
                        # Straight-line distance
                        start_end_dist = np.sqrt(
                            (seg_pos['x'].iloc[-1] - seg_pos['x'].iloc[0])**2 +
                            (seg_pos['y'].iloc[-1] - seg_pos['y'].iloc[0])**2
                        )
                        
                        seg['total_distance'] = total_distance
                        seg['straightness'] = start_end_dist / (total_distance + 1e-8)
                        seg['tortuosity'] = 1 - seg['straightness'] if seg['straightness'] <= 1 else 0
                        seg['n_points'] = len(seg_pos)
                
                burst_segments.append(seg)
        
        print(f"  Step 5: Found {len(burst_segments)} burst segments after filtering")
        
        # 6. Evaluate if ground truth is provided
        evaluation = None
        if ground_truth is not None and BCUBED_AVAILABLE:
            try:
                evaluation = self.evaluate_with_bcubed(burst_segments, ground_truth)
                print(f"  Evaluation: Precision={evaluation.get('precision', 0):.3f}, "
                      f"Recall={evaluation.get('recall', 0):.3f}")
            except:
                evaluation = None
        
        return {
            'bursts': burst_segments,
            'segments': segments,
            'changepoints': changepoints,
            'features': features_dict,
            'processed_data': df_clean,
            'evaluation': evaluation
        }


# Simplified analyzer for visualization
class BurstAnalyzer:
    """Post-processing and analysis of detected bursts"""
    
    def __init__(self):
        pass
    
    def summarize_bursts(self, detection_results):
        """Generate comprehensive summary statistics"""
        bursts = detection_results['bursts']
        
        if not bursts:
            return {"total_bursts": 0, "message": "No bursts detected"}
        
        summary = {
            'total_bursts': len(bursts),
            'total_burst_duration': sum(b['duration'] for b in bursts),
            'mean_burst_duration': np.mean([b['duration'] for b in bursts]),
            'std_burst_duration': np.std([b['duration'] for b in bursts]),
            'median_burst_duration': np.median([b['duration'] for b in bursts]),
            'mean_burst_speed': np.mean([b['mean_speed'] for b in bursts]),
            'total_distance': sum(b.get('total_distance', 0) for b in bursts),
        }
        
        # Add straightness metrics if available
        straightness_vals = [b.get('straightness', 0) for b in bursts if 'straightness' in b]
        if straightness_vals:
            summary['mean_straightness'] = np.mean(straightness_vals)
            summary['median_straightness'] = np.median(straightness_vals)
        
        # Add temporal statistics
        if len(bursts) > 1:
            intervals = [bursts[i+1]['start'] - bursts[i]['end'] 
                       for i in range(len(bursts)-1)]
            summary['mean_inter_burst_interval'] = np.mean(intervals)
            summary['median_inter_burst_interval'] = np.median(intervals)
            total_time = detection_results['processed_data']['t'].iloc[-1]
            summary['burst_frequency'] = len(bursts) / total_time
            summary['burst_duty_cycle'] = (summary['total_burst_duration'] / total_time) * 100
        
        return summary
    
    def visualize_results(self, detection_results, save_path=None):
        """Create a simple visualization of results"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 1. Trajectory plot
            ax = axes[0, 0]
            df = detection_results['processed_data']
            bursts = detection_results['bursts']
            
            # Plot full trajectory
            ax.plot(df['x'], df['y'], 'k-', alpha=0.3, linewidth=0.5, label='Full path')
            
            # Plot bursts
            colors = cm.rainbow(np.linspace(0, 1, len(bursts)))
            for i, burst in enumerate(bursts):
                mask = (df['t'] >= burst['start']) & (df['t'] <= burst['end'])
                if np.sum(mask) > 0:
                    ax.plot(df.loc[mask, 'x'], df.loc[mask, 'y'], 
                           '-', color=colors[i], linewidth=2, alpha=0.8,
                           label=f'Burst {i+1}')
            
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            ax.set_title('Animal Trajectory with Bursts')
            if len(bursts) <= 5:  # Only show legend if not too many bursts
                ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # 2. Speed profile
            ax = axes[0, 1]
            t = detection_results['features']['t']
            speed = detection_results['features']['speed_smooth']
            
            ax.plot(t, speed, 'b-', alpha=0.7, linewidth=1)
            
            # Shade burst regions
            ymin, ymax = ax.get_ylim()
            for burst in bursts:
                ax.axvspan(burst['start'], burst['end'], 
                          alpha=0.2, color='red', ymin=0, ymax=1)
            
            # Mark changepoints
            for cp in detection_results['changepoints']:
                if cp < len(t):
                    ax.axvline(t[cp], color='green', alpha=0.5, 
                              linestyle='--', linewidth=0.5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Speed')
            ax.set_title('Speed Profile with Bursts')
            ax.grid(True, alpha=0.3)
            
            # 3. Burst statistics
            ax = axes[1, 0]
            if bursts:
                metrics = ['duration', 'mean_speed']
                metric_names = ['Duration (s)', 'Mean Speed']
                
                x_pos = np.arange(len(bursts))
                width = 0.35
                
                durations = [b['duration'] for b in bursts]
                speeds = [b['mean_speed'] for b in bursts]
                
                bars1 = ax.bar(x_pos - width/2, durations, width, 
                              label='Duration', alpha=0.7, color='blue')
                bars2 = ax.bar(x_pos + width/2, speeds, width,
                              label='Speed', alpha=0.7, color='orange')
                
                ax.set_xlabel('Burst Index')
                ax.set_ylabel('Value')
                ax.set_title('Burst Statistics')
                ax.legend()
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'B{i+1}' for i in range(len(bursts))])
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No bursts detected', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Burst Statistics')
            
            # 4. Summary table
            ax = axes[1, 1]
            ax.axis('off')
            
            summary = self.summarize_bursts(detection_results)
            
            if summary['total_bursts'] > 0:
                summary_text = []
                summary_text.append(f"Total bursts: {summary['total_bursts']}")
                summary_text.append(f"Total duration: {summary['total_burst_duration']:.1f}s")
                summary_text.append(f"Mean duration: {summary['mean_burst_duration']:.2f}s")
                summary_text.append(f"Mean speed: {summary['mean_burst_speed']:.3f}")
                
                if 'burst_frequency' in summary:
                    summary_text.append(f"Frequency: {summary['burst_frequency']:.3f} Hz")
                
                if 'burst_duty_cycle' in summary:
                    summary_text.append(f"Duty cycle: {summary['burst_duty_cycle']:.1f}%")
                
                ax.text(0.1, 0.9, '\n'.join(summary_text), 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'No bursts detected', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
            return None
        except Exception as e:
            print(f"Visualization error: {e}")
            return None
        
        return fig, axes


# Test function to verify everything works
def test_detector():
    """Test the optimized detector with sample data"""
    print("=" * 60)
    print("Testing Optimized Movement Burst Detector")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_points = 1000
    t = np.linspace(0, 60, n_points)  # 60 seconds
    
    # Create movement with bursts
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    
    # Add some burst periods
    burst_periods = [(10, 15), (25, 30), (40, 45)]
    
    for i in range(1, n_points):
        # Base movement (small random walk)
        x[i] = x[i-1] + np.random.normal(0, 0.02)
        y[i] = y[i-1] + np.random.normal(0, 0.02)
        
        # Check if in burst period
        in_burst = False
        for start, end in burst_periods:
            if start <= t[i] <= end:
                in_burst = True
                break
        
        if in_burst:
            # Add burst movement
            x[i] += np.random.normal(0.1, 0.05)
            y[i] += np.random.normal(0.1, 0.05)
    
    pos_df = pd.DataFrame({'t': t, 'x': x, 'y': y})
    
    print(f"Created test data: {len(pos_df)} points")
    print(f"Time range: {t[0]:.1f}s to {t[-1]:.1f}s")
    print(f"Expected bursts at: {burst_periods}")
    
    # Initialize detector
    detector = OptimizedMovementBurstDetector(
        min_burst_duration=0.5,
        min_rest_duration=1.0,
        velocity_smoothing=0.1,
        bocd_hazard=50,
        clustering_method='dbscan',
        use_gpu=False
    )
    
    # Run detection
    print("\nRunning detection...")
    results = detector.detect_bursts(pos_df)
    
    # Analyze results
    analyzer = BurstAnalyzer()
    summary = analyzer.summarize_bursts(results)
    
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    print(f"Total bursts detected: {summary['total_bursts']}")
    
    if summary['total_bursts'] > 0:
        print(f"Total burst duration: {summary['total_burst_duration']:.2f}s")
        print(f"Mean burst duration: {summary['mean_burst_duration']:.2f}s")
        print(f"Mean burst speed: {summary['mean_burst_speed']:.3f}")
        
        if 'burst_frequency' in summary:
            print(f"Burst frequency: {summary['burst_frequency']:.3f} Hz")
        
        print("\nDetected bursts:")
        for i, burst in enumerate(results['bursts']):
            print(f"  Burst {i+1}: {burst['start']:.1f}s to {burst['end']:.1f}s "
                  f"(duration: {burst['duration']:.1f}s, "
                  f"speed: {burst['mean_speed']:.3f})")
    
    # Visualize
    print("\nGenerating visualization...")
    analyzer.visualize_results(results, save_path='test_burst_detection.png')
    
    return results


# ============================================
# EXAMPLE USAGE WITH NEW PACKAGES
# ============================================

def example_usage():
    """Demonstrate the optimized detector with new packages"""
    # Generate sample data
    pos_df = pd.DataFrame({
        't': np.linspace(0, 60, 1800),  # 60 seconds at 30Hz
        'x': np.cumsum(np.random.normal(0, 0.1, 1800)),
        'y': np.cumsum(np.random.normal(0, 0.1, 1800))
    })
    
    # Add simulated bursts
    burst_times = [(10, 15), (25, 30), (40, 45)]
    for start, end in burst_times:
        mask = (pos_df['t'] >= start) & (pos_df['t'] <= end)
        pos_df.loc[mask, 'x'] += np.cumsum(np.random.normal(0.5, 0.2, np.sum(mask))) * 0.1
        pos_df.loc[mask, 'y'] += np.cumsum(np.random.normal(0.5, 0.2, np.sum(mask))) * 0.1
    
    print("Data shape:", pos_df.shape)
    print(f"Duration: {pos_df['t'].iloc[-1]:.1f}s")
    
    # Initialize optimized detector
    detector = OptimizedMovementBurstDetector(
        min_burst_duration=0.8,
        min_rest_duration=1.2,
        velocity_smoothing=0.15,
        bocd_hazard=100,
        clustering_method='dip',  # Use DipExt from clustpy
        use_gpu=False  # Set to True if you have CUDA
    )
    
    # Detect bursts
    print("\nDetecting bursts with optimized pipeline...")
    results = detector.detect_bursts(pos_df)
    
    # Analyze results
    analyzer = BurstAnalyzer()
    summary = analyzer.summarize_bursts(results)
    
    print(f"\nDetection Summary:")
    print(f"  Total bursts: {summary['total_bursts']}")
    if summary['total_bursts'] > 0:
        print(f"  Total burst duration: {summary['total_burst_duration']:.1f}s")
        print(f"  Mean burst duration: {summary['mean_burst_duration']:.2f} ± {summary['std_burst_duration']:.2f}s")
        print(f"  Mean burst speed: {summary['mean_burst_speed']:.3f}")
        print(f"  Total distance during bursts: {summary['total_distance']:.2f}")
        if 'burst_frequency' in summary:
            print(f"  Burst frequency: {summary['burst_frequency']:.3f} Hz")
    
    # Display individual bursts
    print(f"\nDetected Bursts:")
    for i, burst in enumerate(results['bursts']):
        print(f"  Burst {i+1}: {burst['start']:.1f}-{burst['end']:.1f}s "
              f"(dur: {burst['duration']:.1f}s, speed: {burst['mean_speed']:.3f}, "
              f"dist: {burst.get('total_distance', 0):.2f})")
    
    # Visualize
    print("\nGenerating visualization...")
    analyzer.visualize_segmentation(results, save_path='optimized_burst_detection.png')
    
    return results


if __name__ == "__main__":
    # Check package availability
    print("Package Availability Check:")
    print(f"  bayesian-changepoint-detection: {'✓' if BOCD_AVAILABLE else '✗'}")
    print(f"  clustpy: {'✓' if CLUSTPY_AVAILABLE else '✗'}")
    print(f"  bcubed: {'✓' if BCUBED_AVAILABLE else '✗'}")
    print(f"  CUDA available: {'✓' if torch.cuda.is_available() else '✗'}")
    
    # Run test
    test_results = test_detector()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    

    # Run example
    results = example_usage()
    


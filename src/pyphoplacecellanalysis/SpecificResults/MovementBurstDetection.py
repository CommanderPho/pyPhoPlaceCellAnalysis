import numpy as np
import pandas as pd
from scipy import stats, signal, optimize, spatial
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class MovementBurstDetector:
    """
    Advanced movement burst detection using Bayesian changepoint detection
    with velocity/acceleration features and adaptive thresholding.
    
    Usage:
    
		from pyphoplacecellanalysis.SpecificResults.MovementBurstDetection import MovementBurstDetector
    """
    
    def __init__(self, min_burst_duration=0.5, min_rest_duration=1.0,
                 velocity_smoothing=0.1, method='bocd'):
        """
        Parameters:
        -----------
        min_burst_duration : float
            Minimum duration for a burst (seconds)
        min_rest_duration : float
            Minimum duration for rest periods (seconds)
        velocity_smoothing : float
            Gaussian smoothing sigma for velocity (seconds)
        method : str
            'bocd', 'hmm', or 'combo' (ensemble)
        """
        self.min_burst_duration = min_burst_duration
        self.min_rest_duration = min_rest_duration
        self.velocity_smoothing = velocity_smoothing
        self.method = method
        
    def preprocess_trajectory(self, pos_df):
        """Clean and prepare trajectory data"""
        # Ensure sorted by time
        df = pos_df.copy().sort_values('t')
        
        # Interpolate small gaps (less than 3 frames)
        dt = np.median(np.diff(df['t'].values))
        max_gap = 3 * dt
        
        # Create regular time grid
        t_min, t_max = df['t'].min(), df['t'].max()
        t_reg = np.arange(t_min, t_max, dt)
        
        # Interpolate
        df['x'] = np.interp(t_reg, df['t'], df['x'])
        df['y'] = np.interp(t_reg, df['t'], df['y'])
        df['t'] = t_reg
        
        return df
    
    def compute_movement_features(self, pos_df):
        """Extract comprehensive movement features"""
        t = pos_df['t'].values
        x = pos_df['x'].values
        y = pos_df['y'].values
        
        dt = np.mean(np.diff(t))
        
        # 1. Instantaneous velocity
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        speed = np.sqrt(vx**2 + vy**2)
        
        # 2. Smoothed velocity
        sigma = self.velocity_smoothing / dt
        speed_smooth = gaussian_filter1d(speed, sigma=sigma)
        
        # 3. Acceleration
        acc = np.gradient(speed_smooth, dt)
        
        # 4. Jerk (derivative of acceleration)
        jerk = np.gradient(acc, dt)
        
        # 5. Angular velocity (turning rate)
        heading = np.arctan2(vy, vx)
        angular_vel = np.gradient(heading, dt)
        
        # 6. Movement smoothness (spectral features)
        f, Pxx = signal.welch(speed_smooth, fs=1/dt, nperseg=min(256, len(speed_smooth)//4))
        spectral_entropy = stats.entropy(Pxx[Pxx > 0])
        
        # 7. Local variance (movement stability)
        window = max(3, int(0.5/dt))  # 500ms window
        local_var = pd.Series(speed_smooth).rolling(window=window, center=True).std().values
        
        # Create feature matrix
        features = np.column_stack([
            speed_smooth,
            acc,
            jerk,
            np.abs(angular_vel),
            gaussian_filter1d(local_var, sigma=2)
        ])
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        return {
            't': t,
            'speed': speed,
            'speed_smooth': speed_smooth,
            'features': features,
            'dt': dt
        }
    
    def bocd_detection(self, features_dict):
        """Bayesian Online Changepoint Detection (Adams & MacKay, 2007)"""
        t = features_dict['t']
        speed = features_dict['speed_smooth']
        dt = features_dict['dt']
        
        # Parameters for BOCD
        hazard = 1.0 / (self.min_rest_duration / dt)  # Hazard rate
        obs_model = 'studentt'  # Student's t-distribution (robust to outliers)
        
        # Implement BOCD
        R = np.zeros((len(speed) + 1, len(speed) + 1))
        R[0, 0] = 1
        
        # Sufficient statistics storage
        mean0 = np.mean(speed[:10])
        var0 = np.var(speed[:10]) + 1e-8
        kappa0 = 1.0
        nu0 = 1.0
        
        changepoints = []
        
        for n in range(1, len(speed) + 1):
            # Predict step
            R[1:n+1, n] = R[0:n, n-1] * (1 - hazard)
            R[0, n] = np.sum(R[0:n, n-1] * hazard)
            
            # Update step (Student's t-distribution)
            x = speed[n-1]
            
            # Update sufficient statistics for each run length
            for r in range(1, n+1):
                # Get data segment for this run length
                seg = speed[n-r:n-1]
                if len(seg) == 0:
                    continue
                
                # Calculate predictive probability
                mean_r = np.mean(seg) if len(seg) > 0 else mean0
                var_r = np.var(seg) if len(seg) > 1 else var0
                kappa_r = kappa0 + len(seg)
                nu_r = nu0 + len(seg)
                
                # Student's t predictive probability
                scale = var_r * (1 + 1/kappa_r)
                t_score = stats.t.pdf(x, df=nu_r, loc=mean_r, scale=np.sqrt(scale))
                R[r, n] *= t_score
            
            # Normalize
            R[:, n] = R[:, n] / (np.sum(R[:, n]) + 1e-8)
            
            # Detect changepoint
            if n > 1:
                max_run = np.argmax(R[:n+1, n])
                if max_run == 0:  # Run length reset to 0 indicates changepoint
                    changepoints.append(n-1)
        
        # Convert indices to times
        changepoint_times = t[changepoints] if changepoints else []
        
        # Create segments
        segments = []
        prev_idx = 0
        for cp_idx in changepoints:
            seg_t = t[prev_idx:cp_idx]
            seg_speed = speed[prev_idx:cp_idx]
            if len(seg_t) > 0:
                segments.append({
                    'start': seg_t[0],
                    'end': seg_t[-1],
                    'duration': seg_t[-1] - seg_t[0],
                    'mean_speed': np.mean(seg_speed),
                    'std_speed': np.std(seg_speed)
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
                'mean_speed': np.mean(seg_speed),
                'std_speed': np.std(seg_speed)
            })
        
        return segments, changepoint_times
    
    def adaptive_threshold_clustering(self, features_dict):
        """Adaptive thresholding with DBSCAN clustering"""
        t = features_dict['t']
        features = features_dict['features']
        speed = features_dict['speed_smooth']
        
        # Use DBSCAN for unsupervised state detection
        clustering = DBSCAN(eps=0.5, min_samples=int(self.min_burst_duration/features_dict['dt']))
        labels = clustering.fit_predict(features)
        
        # Classify clusters as burst/rest based on speed
        segments = []
        unique_labels = np.unique(labels[labels != -1])  # Exclude noise
        
        if len(unique_labels) == 0:
            # Fallback to speed threshold
            threshold = np.percentile(speed, 70)
            is_burst = speed > threshold
        else:
            cluster_speeds = [np.mean(speed[labels == lbl]) for lbl in unique_labels]
            burst_clusters = unique_labels[np.argsort(cluster_speeds)[-2:]]  # Top 2 speed clusters
            
            is_burst = np.zeros_like(speed, dtype=bool)
            for lbl in burst_clusters:
                is_burst[labels == lbl] = True
        
        # Find contiguous bursts
        state_changes = np.diff(np.concatenate(([0], is_burst.astype(int), [0])))
        start_indices = np.where(state_changes == 1)[0]
        end_indices = np.where(state_changes == -1)[0] - 1
        
        for start, end in zip(start_indices, end_indices):
            duration = t[end] - t[start] if end < len(t) else t[-1] - t[start]
            if duration >= self.min_burst_duration:
                segments.append({
                    'start': t[start],
                    'end': t[end] if end < len(t) else t[-1],
                    'duration': duration,
                    'mean_speed': np.mean(speed[start:end+1]),
                    'type': 'burst'
                })
        
        return segments
    
    def ensemble_detection(self, features_dict):
        """Combine multiple detection methods for robustness"""
        # Get BOCD segments
        bocd_segments, _ = self.bocd_detection(features_dict)
        
        # Get clustering segments
        cluster_segments = self.adaptive_threshold_clustering(features_dict)
        
        # Combine using intersection (more conservative)
        combined_segments = []
        
        for cseg in cluster_segments:
            for bseg in bocd_segments:
                # Check for overlap
                overlap_start = max(cseg['start'], bseg['start'])
                overlap_end = min(cseg['end'], bseg['end'])
                
                if overlap_start < overlap_end:
                    duration = overlap_end - overlap_start
                    if duration >= self.min_burst_duration:
                        mean_speed = np.mean(features_dict['speed_smooth'][
                            (features_dict['t'] >= overlap_start) & 
                            (features_dict['t'] <= overlap_end)
                        ])
                        
                        combined_segments.append({
                            'start': overlap_start,
                            'end': overlap_end,
                            'duration': duration,
                            'mean_speed': mean_speed,
                            'type': 'burst',
                            'confidence': min(cseg.get('mean_speed', 0)/np.max(features_dict['speed_smooth']),
                                            bseg.get('mean_speed', 0)/np.max(features_dict['speed_smooth']))
                        })
        
        # Merge overlapping segments
        if combined_segments:
            combined_segments.sort(key=lambda x: x['start'])
            merged = []
            current = combined_segments[0]
            
            for seg in combined_segments[1:]:
                if seg['start'] <= current['end'] + dt:  # Allow small gap
                    current['end'] = max(current['end'], seg['end'])
                    current['duration'] = current['end'] - current['start']
                    current['confidence'] = max(current['confidence'], seg['confidence'])
                else:
                    merged.append(current)
                    current = seg
            
            merged.append(current)
            return merged
        
        return combined_segments
    
    def detect_bursts(self, pos_df):
        """Main detection pipeline"""
        # Preprocess
        df_clean = self.preprocess_trajectory(pos_df)
        
        # Extract features
        features_dict = self.compute_movement_features(df_clean)
        
        # Apply detection method
        if self.method == 'bocd':
            segments, changepoints = self.bocd_detection(features_dict)
            # Classify segments as burst/rest
            speed_threshold = np.percentile(features_dict['speed_smooth'], 60)
            burst_segments = [s for s in segments if s['mean_speed'] > speed_threshold 
                            and s['duration'] >= self.min_burst_duration]
            
        elif self.method == 'hmm':
            burst_segments = self.adaptive_threshold_clustering(features_dict)
            
        else:  # 'combo'
            burst_segments = self.ensemble_detection(features_dict)
        
        # Post-process: filter by duration and add metadata
        filtered_segments = []
        for seg in burst_segments:
            if seg['duration'] >= self.min_burst_duration:
                # Extract position data for this segment
                mask = (df_clean['t'] >= seg['start']) & (df_clean['t'] <= seg['end'])
                seg_positions = df_clean[mask]
                
                # Calculate additional metrics
                if len(seg_positions) > 1:
                    total_distance = np.sum(np.sqrt(
                        np.diff(seg_positions['x'])**2 + 
                        np.diff(seg_positions['y'])**2
                    ))
                    seg['total_distance'] = total_distance
                    seg['straightness'] = (
                        np.sqrt((seg_positions['x'].iloc[-1] - seg_positions['x'].iloc[0])**2 +
                               (seg_positions['y'].iloc[-1] - seg_positions['y'].iloc[0])**2) /
                        (total_distance + 1e-8)
                    )
                    seg['tortuosity'] = 1 - seg['straightness']
                
                filtered_segments.append(seg)
        
        return {
            'bursts': filtered_segments,
            'features': features_dict,
            'processed_data': df_clean
        }


# ============================================
# VISUALIZATION AND VALIDATION TOOLS
# ============================================

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

def visualize_bursts(pos_df, detection_results, save_path=None):
    """Create comprehensive visualization of detected bursts"""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Trajectory with bursts highlighted
    ax1 = plt.subplot(2, 3, 1)
    t = detection_results['processed_data']['t'].values
    x = detection_results['processed_data']['x'].values
    y = detection_results['processed_data']['y'].values
    
    # Plot full trajectory
    ax1.plot(x, y, 'k-', alpha=0.3, linewidth=0.5, label='Full trajectory')
    
    # Color bursts
    cmap = cm.get_cmap('viridis')
    bursts = detection_results['bursts']
    
    for i, burst in enumerate(bursts):
        mask = (t >= burst['start']) & (t <= burst['end'])
        color = cmap(i / max(1, len(bursts)))
        ax1.plot(x[mask], y[mask], '-', color=color, linewidth=2,
                label=f'Burst {i+1}' if i < 5 else None)
        # Mark start and end
        ax1.plot(x[mask][0], y[mask][0], 'o', color=color, markersize=8)
        ax1.plot(x[mask][-1], y[mask][-1], 's', color=color, markersize=8)
    
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_title('Trajectory with Detected Bursts')
    ax1.legend(loc='best')
    ax1.axis('equal')
    
    # 2. Speed profile with bursts
    ax2 = plt.subplot(2, 3, 2)
    speed = detection_results['features']['speed_smooth']
    
    ax2.plot(t, speed, 'k-', linewidth=1, alpha=0.7, label='Speed')
    ax2.fill_between(t, 0, speed, alpha=0.3, color='gray')
    
    # Highlight bursts
    ymin, ymax = ax2.get_ylim()
    for burst in bursts:
        ax2.add_patch(Rectangle(
            (burst['start'], ymin),
            burst['duration'],
            ymax - ymin,
            alpha=0.2, color='red'
        ))
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed')
    ax2.set_title('Speed Profile with Bursts')
    ax2.legend(['Speed', 'Burst periods'])
    
    # 3. Velocity components
    ax3 = plt.subplot(2, 3, 3)
    vx = np.gradient(x, np.mean(np.diff(t)))
    vy = np.gradient(y, np.mean(np.diff(t)))
    
    ax3.plot(t, vx, 'b-', alpha=0.7, label='Vx')
    ax3.plot(t, vy, 'r-', alpha=0.7, label='Vy')
    
    for burst in bursts:
        ax3.axvspan(burst['start'], burst['end'], alpha=0.1, color='green')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Velocity Components')
    ax3.legend()
    
    # 4. Movement features
    ax4 = plt.subplot(2, 3, 4)
    features = detection_results['features']['features']
    feature_names = ['Speed', 'Accel', 'Jerk', 'AngVel', 'Var']
    
    for i in range(min(3, features.shape[1])):
        ax4.plot(t, features[:, i], label=feature_names[i])
    
    for burst in bursts:
        ax4.axvspan(burst['start'], burst['end'], alpha=0.1, color='gray')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Normalized Feature Value')
    ax4.set_title('Movement Features')
    ax4.legend()
    
    # 5. Burst statistics
    ax5 = plt.subplot(2, 3, 5)
    if bursts:
        burst_durations = [b['duration'] for b in bursts]
        burst_speeds = [b['mean_speed'] for b in bursts]
        burst_distances = [b.get('total_distance', 0) for b in bursts]
        
        x_pos = np.arange(len(bursts))
        width = 0.25
        
        ax5.bar(x_pos - width, burst_durations, width, label='Duration (s)', alpha=0.8)
        ax5.bar(x_pos, burst_speeds, width, label='Mean speed', alpha=0.8)
        ax5.bar(x_pos + width, burst_distances, width, label='Distance', alpha=0.8)
        
        ax5.set_xlabel('Burst Index')
        ax5.set_ylabel('Value')
        ax5.set_title('Burst Statistics')
        ax5.legend()
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f'B{i+1}' for i in range(len(bursts))])
    
    # 6. Phase portrait (speed vs acceleration)
    ax6 = plt.subplot(2, 3, 6)
    acc = np.gradient(speed, np.mean(np.diff(t)))
    
    scatter = ax6.scatter(speed[::10], acc[::10], c=t[::10], cmap='viridis', 
                         s=10, alpha=0.6, edgecolors='none')
    
    # Mark burst points
    for burst in bursts:
        mask = (t >= burst['start']) & (t <= burst['end'])
        ax6.scatter(speed[mask][::5], acc[mask][::5], 
                   color='red', s=20, alpha=0.7, edgecolors='black')
    
    ax6.set_xlabel('Speed')
    ax6.set_ylabel('Acceleration')
    ax6.set_title('Phase Portrait')
    plt.colorbar(scatter, ax=ax6, label='Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig


# ============================================
# EVALUATION METRICS
# ============================================

def evaluate_burst_detection(pos_df, ground_truth_bursts=None):
    """
    Evaluate detection quality with various metrics.
    If ground truth is provided, calculate precision/recall.
    """
    # Initialize detector with different methods
    detectors = {
        'BOCD': MovementBurstDetector(method='bocd'),
        'HMM/Clustering': MovementBurstDetector(method='hmm'),
        'Ensemble': MovementBurstDetector(method='combo')
    }
    
    results = {}
    
    for name, detector in detectors.items():
        print(f"\n{'='*50}")
        print(f"Testing {name} method")
        print('='*50)
        
        # Detect bursts
        detection_results = detector.detect_bursts(pos_df)
        bursts = detection_results['bursts']
        
        print(f"Detected {len(bursts)} bursts")
        
        if bursts:
            # Calculate statistics
            durations = [b['duration'] for b in bursts]
            speeds = [b['mean_speed'] for b in bursts]
            distances = [b.get('total_distance', 0) for b in bursts]
            
            print(f"Mean burst duration: {np.mean(durations):.2f} ± {np.std(durations):.2f} s")
            print(f"Total burst time: {np.sum(durations):.2f} s ({np.sum(durations)/pos_df['t'].iloc[-1]*100:.1f}% of total)")
            print(f"Mean burst speed: {np.mean(speeds):.2f} ± {np.std(speeds):.2f}")
            print(f"Total distance during bursts: {np.sum(distances):.2f}")
            
            # Inter-burst intervals
            if len(bursts) > 1:
                intervals = [bursts[i+1]['start'] - bursts[i]['end'] 
                           for i in range(len(bursts)-1)]
                print(f"Mean inter-burst interval: {np.mean(intervals):.2f} ± {np.std(intervals):.2f} s")
        
        results[name] = {
            'bursts': bursts,
            'detection_results': detection_results
        }
    
    return results


# ============================================
# EXAMPLE USAGE
# ============================================

def simulate_animal_trajectory(duration=60, sampling_rate=30):
    """Generate synthetic animal trajectory for testing"""
    np.random.seed(42)
    
    t = np.arange(0, duration, 1/sampling_rate)
    n = len(t)
    
    # Generate random walk with bursts
    x = np.zeros(n)
    y = np.zeros(n)
    
    # Create burst/non-burst pattern
    burst_probs = 0.3 * (np.sin(2*np.pi*t/15) + 1) + 0.1
    
    for i in range(1, n):
        if np.random.rand() < burst_probs[i]:
            # Burst mode: larger, directed steps
            dx = np.random.normal(0.5, 0.2)
            dy = np.random.normal(0.5, 0.2)
            angle = np.random.uniform(0, 2*np.pi)
            step_size = np.random.exponential(0.3)
            x[i] = x[i-1] + step_size * np.cos(angle)
            y[i] = y[i-1] + step_size * np.sin(angle)
        else:
            # Rest mode: small random movements
            x[i] = x[i-1] + np.random.normal(0, 0.05)
            y[i] = y[i-1] + np.random.normal(0, 0.05)
    
    # Add some noise
    x += np.random.normal(0, 0.02, n)
    y += np.random.normal(0, 0.02, n)
    
    return pd.DataFrame({'t': t, 'x': x, 'y': y})


# Main execution
if __name__ == "__main__":
    # Create or load your data
    # pos_df = pd.read_csv('your_data.csv')  # Your actual data
    pos_df = simulate_animal_trajectory(duration=120, sampling_rate=30)
    
    print(f"Data shape: {pos_df.shape}")
    print(f"Duration: {pos_df['t'].iloc[-1]:.1f} seconds")
    print(f"Sampling rate: {1/np.mean(np.diff(pos_df['t'])):.1f} Hz")
    
    # Initialize detector (using ensemble method for best results)
    detector = MovementBurstDetector(
        min_burst_duration=0.8,
        min_rest_duration=1.2,
        velocity_smoothing=0.15,
        method='combo'  # Best performing ensemble method
    )
    
    # Detect bursts
    results = detector.detect_bursts(pos_df)
    
    # Display results
    print(f"\nDetected {len(results['bursts'])} movement bursts:")
    print("-" * 60)
    for i, burst in enumerate(results['bursts']):
        print(f"Burst {i+1}:")
        print(f"  Time: {burst['start']:.1f} - {burst['end']:.1f} s "
              f"(duration: {burst['duration']:.1f} s)")
        print(f"  Mean speed: {burst['mean_speed']:.2f}")
        print(f"  Distance: {burst.get('total_distance', 0):.2f}")
        print(f"  Tortuosity: {burst.get('tortuosity', 0):.2f}")
        print()
    
    # Visualize
    fig = visualize_bursts(pos_df, results, save_path='burst_detection.png')
    
    # Evaluate different methods
    all_results = evaluate_burst_detection(pos_df)
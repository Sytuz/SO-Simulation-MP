"""
Visualization functions for bus simulation results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.signal import savgol_filter


def plot_results(stats, output_dir="results/bus", title="Bus Maintenance Simulation Results"):
    """Generate plots for the simulation results"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a clear figure with improved styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Improved color palette and styling
    colors = ['#1E88E5', '#FFA000']  # Blue and amber for better contrast
    
    # Common styling parameters
    linewidth = 2.5
    linewidth_raw = 1.0
    alpha_raw = 0.2
    alpha_fill = 0.15
    font_size_title = 14
    font_size_labels = 12
    font_size_legend = 10
    
    # Create time axis for time series plots (in hours)
    queue_length = len(stats['inspection_queue_lengths'])
    time_points = np.linspace(0, (queue_length-1) * 0.1, queue_length)
    
    # Apply moving average smoothing for time series data to improve readability
    window_size = min(20, queue_length // 5)  # 2-hour window (20 samples at 0.1h each), or smaller if data is limited
    if window_size < 2:
        window_size = 2  # Minimum window size
    
    def smooth_data(data, window=window_size):
        if len(data) < window * 2:
            return data  # Not enough data for meaningful smoothing
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Smooth the queue length data
    smooth_insp_queue = smooth_data(stats['inspection_queue_lengths'])
    smooth_repair_queue = smooth_data(stats['repair_queue_lengths'])
    
    # Adjust time axis for smoothed data - ensure it matches the smoothed data length
    smooth_time = time_points[window_size-1:window_size-1+len(smooth_insp_queue)]
    
    # Queue lengths over time - using smoothed data
    ax1.plot(smooth_time, smooth_insp_queue, color=colors[0], linewidth=linewidth, 
             label='Inspection Queue (smoothed)')
    ax1.plot(smooth_time, smooth_repair_queue, color=colors[1], linewidth=linewidth, 
             label='Repair Queue (smoothed)')
    
    # Add shaded area to highlight trends
    ax1.fill_between(smooth_time, smooth_insp_queue, alpha=alpha_fill, color=colors[0])
    ax1.fill_between(smooth_time, smooth_repair_queue, alpha=alpha_fill, color=colors[1])
    
    # Also show the original data with lower opacity for reference
    ax1.plot(time_points, stats['inspection_queue_lengths'], color=colors[0], 
             linewidth=linewidth_raw, alpha=alpha_raw)
    ax1.plot(time_points, stats['repair_queue_lengths'], color=colors[1], 
             linewidth=linewidth_raw, alpha=alpha_raw)
    
    ax1.set_title('Queue Lengths Over Time', fontsize=font_size_title, fontweight='bold')
    ax1.set_xlabel('Time (hours)', fontsize=font_size_labels)
    ax1.set_ylabel('Queue Length', fontsize=font_size_labels)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=font_size_legend)
    
    # Calculate cumulative utilization over time for better interpretation
    n_points = len(stats['inspection_utilization'])
    divisor = np.arange(1, n_points + 1)
    cum_insp_util = np.cumsum(stats['inspection_utilization']) / divisor
    cum_repair_util = np.cumsum(stats['repair_utilization']) / divisor
    
    # Utilization over time - plot cumulative average
    ax2.plot(time_points, cum_insp_util, color=colors[0], linewidth=linewidth, 
             label='Inspector (cumulative avg)')
    ax2.plot(time_points, cum_repair_util, color=colors[1], linewidth=linewidth, 
             label='Repair Stations (cumulative avg)')
    
    # Add shaded area to highlight trends
    ax2.fill_between(time_points, cum_insp_util, alpha=alpha_fill, color=colors[0])
    ax2.fill_between(time_points, cum_repair_util, alpha=alpha_fill, color=colors[1])
    
    # Show instantaneous utilization with lower opacity
    ax2.plot(time_points, stats['inspection_utilization'], color=colors[0], 
             linewidth=linewidth_raw, alpha=alpha_raw)
    ax2.plot(time_points, stats['repair_utilization'], color=colors[1], 
             linewidth=linewidth_raw, alpha=alpha_raw)
    
    ax2.set_title('Resource Utilization Over Time', fontsize=font_size_title, fontweight='bold')
    ax2.set_xlabel('Time (hours)', fontsize=font_size_labels)
    ax2.set_ylabel('Utilization', fontsize=font_size_labels)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=font_size_legend)
    
    # Delay histograms with enhanced styling
    if stats['inspection_delays']:
        bins = min(20, max(5, len(stats['inspection_delays']) // 10))
        ax3.hist(stats['inspection_delays'], bins=bins, alpha=0.7, color=colors[0])
        ax3.set_title('Inspection Delays', fontsize=font_size_title, fontweight='bold')
        ax3.set_xlabel('Delay (hours)', fontsize=font_size_labels)
        ax3.set_ylabel('Count', fontsize=font_size_labels)
        ax3.grid(True, alpha=0.3)
        
        # Add mean line
        mean_delay = np.mean(stats['inspection_delays'])
        ax3.axvline(x=mean_delay, color='red', linestyle='--', linewidth=1.5)
        ax3.text(mean_delay*1.05, ax3.get_ylim()[1]*0.9, 
                 f'Mean: {mean_delay:.2f} hours', 
                 fontsize=10, color='darkred')
    
    if stats['repair_delays']:
        bins = min(20, max(5, len(stats['repair_delays']) // 10))
        ax4.hist(stats['repair_delays'], bins=bins, alpha=0.7, color=colors[1])
        ax4.set_title('Repair Delays', fontsize=font_size_title, fontweight='bold')
        ax4.set_xlabel('Delay (hours)', fontsize=font_size_labels)
        ax4.set_ylabel('Count', fontsize=font_size_labels)
        ax4.grid(True, alpha=0.3)
        
        # Add mean line
        mean_delay = np.mean(stats['repair_delays'])
        ax4.axvline(x=mean_delay, color='red', linestyle='--', linewidth=1.5)
        ax4.text(mean_delay*1.05, ax4.get_ylim()[1]*0.9, 
                 f'Mean: {mean_delay:.2f} hours', 
                 fontsize=10, color='darkred')
    
    # Add overall title with information about the simulation
    plt.suptitle(title, fontsize=font_size_title+2, fontweight='bold', y=0.995)
    
    # Improve spacing
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # Save the high-resolution figure
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bus_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment_results(results, output_dir="results/bus"):
    """Generate plots to visualize how system performance changes with arrival rate"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    rates = [r['arrival_rate'] for r in results]
    insp_queues = [r['avg_inspection_queue_length'] for r in results]
    repair_queues = [r['avg_repair_queue_length'] for r in results]
    insp_delays = [r['avg_inspection_delay'] for r in results]
    repair_delays = [r['avg_repair_delay'] for r in results]
    insp_utils = [r['inspection_utilization'] for r in results]
    repair_utils = [r['repair_utilization'] for r in results]
    
    # Sort all data by arrival rate (ascending)
    sorted_indices = np.argsort(rates)
    rates = [rates[i] for i in sorted_indices]
    insp_queues = [insp_queues[i] for i in sorted_indices]
    repair_queues = [repair_queues[i] for i in sorted_indices]
    insp_delays = [insp_delays[i] for i in sorted_indices]
    repair_delays = [repair_delays[i] for i in sorted_indices]
    insp_utils = [insp_utils[i] for i in sorted_indices]
    repair_utils = [repair_utils[i] for i in sorted_indices]
    
    # Determine marker frequency based on number of data points
    n_points = len(rates)
    if n_points > 100:
        marker_freq = n_points // 20  # Show only ~20 markers across the range
    elif n_points > 50:
        marker_freq = n_points // 15
    else:
        marker_freq = max(1, n_points // 10)
    
    # Create a clear figure with improved styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Improved color palette
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange with better contrast
    
    # Common styling for all plots
    linewidth = 2.5
    marker_size = 8
    alpha_fill = 0.15
    font_size_title = 14
    font_size_labels = 12
    font_size_legend = 10
    
    # Define markers to use - use different ones for each series
    markers = ['o', 's']  # Circle for inspection, square for repair
    
    # 1. Queue lengths vs arrival rate with selective markers and shaded trend
    ax1.plot(rates, insp_queues, '-', color=colors[0], linewidth=linewidth, label='Inspection Queue')
    ax1.plot(rates, repair_queues, '-', color=colors[1], linewidth=linewidth, label='Repair Queue')
    
    # Add selective markers (not for every point)
    for i in range(0, len(rates), marker_freq):
        ax1.plot(rates[i], insp_queues[i], markers[0], color=colors[0], markersize=marker_size)
        ax1.plot(rates[i], repair_queues[i], markers[1], color=colors[1], markersize=marker_size)
    
    # Add shaded area to highlight trends
    ax1.fill_between(rates, insp_queues, alpha=alpha_fill, color=colors[0])
    ax1.fill_between(rates, repair_queues, alpha=alpha_fill, color=colors[1])
    
    ax1.set_title('Average Queue Length vs Arrival Rate', fontsize=font_size_title, fontweight='bold')
    ax1.set_xlabel('Arrival Rate (buses/hour)', fontsize=font_size_labels)
    ax1.set_ylabel('Average Queue Length', fontsize=font_size_labels)
    ax1.legend(fontsize=font_size_legend)
    ax1.grid(True, alpha=0.3)
    
    # 2. Delays vs arrival rate
    ax2.plot(rates, insp_delays, '-', color=colors[0], linewidth=linewidth, label='Inspection Delay')
    ax2.plot(rates, repair_delays, '-', color=colors[1], linewidth=linewidth, label='Repair Delay')
    
    # Add selective markers
    for i in range(0, len(rates), marker_freq):
        ax2.plot(rates[i], insp_delays[i], markers[0], color=colors[0], markersize=marker_size)
        ax2.plot(rates[i], repair_delays[i], markers[1], color=colors[1], markersize=marker_size)
    
    # Add shaded area
    ax2.fill_between(rates, insp_delays, alpha=alpha_fill, color=colors[0])
    ax2.fill_between(rates, repair_delays, alpha=alpha_fill, color=colors[1])
    
    ax2.set_title('Average Delay vs Arrival Rate', fontsize=font_size_title, fontweight='bold')
    ax2.set_xlabel('Arrival Rate (buses/hour)', fontsize=font_size_labels)
    ax2.set_ylabel('Average Delay (hours)', fontsize=font_size_labels)
    ax2.legend(fontsize=font_size_legend)
    ax2.grid(True, alpha=0.3)
    
    # 3. Utilization vs arrival rate
    ax3.plot(rates, insp_utils, '-', color=colors[0], linewidth=linewidth, label='Inspector')
    ax3.plot(rates, repair_utils, '-', color=colors[1], linewidth=linewidth, label='Repair Stations')
    
    # Add selective markers
    for i in range(0, len(rates), marker_freq):
        ax3.plot(rates[i], insp_utils[i], markers[0], color=colors[0], markersize=marker_size)
        ax3.plot(rates[i], repair_utils[i], markers[1], color=colors[1], markersize=marker_size)
    
    # Add shaded area
    ax3.fill_between(rates, insp_utils, alpha=alpha_fill, color=colors[0])
    ax3.fill_between(rates, repair_utils, alpha=alpha_fill, color=colors[1])
    
    # Add dashed line at 0.9 utilization threshold
    ax3.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                label='Critical Utilization (0.9)')
    
    ax3.set_title('Resource Utilization vs Arrival Rate', fontsize=font_size_title, fontweight='bold')
    ax3.set_xlabel('Arrival Rate (buses/hour)', fontsize=font_size_labels)
    ax3.set_ylabel('Utilization', fontsize=font_size_labels)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=font_size_legend)
    ax3.grid(True, alpha=0.3)
    
    # 4. System stability threshold visualization with clear annotation
    queue_threshold = 10
    ax4.axhline(y=queue_threshold, color='red', linestyle='--', linewidth=1.5, 
                label=f'Stability Threshold ({queue_threshold})', alpha=0.7)
    
    ax4.plot(rates, insp_queues, '-', color=colors[0], linewidth=linewidth, label='Inspection Queue')
    ax4.plot(rates, repair_queues, '-', color=colors[1], linewidth=linewidth, label='Repair Queue')
    
    # Add selective markers
    for i in range(0, len(rates), marker_freq):
        ax4.plot(rates[i], insp_queues[i], markers[0], color=colors[0], markersize=marker_size)
        ax4.plot(rates[i], repair_queues[i], markers[1], color=colors[1], markersize=marker_size)
    
    # Add shaded area
    ax4.fill_between(rates, insp_queues, alpha=alpha_fill, color=colors[0])
    ax4.fill_between(rates, repair_queues, alpha=alpha_fill, color=colors[1])
    
    # Try to find the stability crossover point
    critical_rate = None
    for i in range(len(rates)-1):
        if insp_queues[i] < queue_threshold and insp_queues[i+1] >= queue_threshold:
            critical_rate = rates[i]
            break
    
    # Add annotation for critical point if found
    if critical_rate:
        ax4.axvline(x=critical_rate, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        ax4.annotate(f'Critical Rate: ~{critical_rate:.2f}',
                    xy=(critical_rate, queue_threshold),
                    xytext=(critical_rate-0.2, queue_threshold+2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=font_size_legend,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    ax4.set_title('System Stability Analysis', fontsize=font_size_title, fontweight='bold')
    ax4.set_xlabel('Arrival Rate (buses/hour)', fontsize=font_size_labels)
    ax4.set_ylabel('Average Queue Length', fontsize=font_size_labels)
    ax4.legend(fontsize=font_size_legend)
    ax4.grid(True, alpha=0.3)
    
    # Add overall title with information about the experiment
    plt.suptitle('Bus Maintenance Facility Capacity Analysis', 
                fontsize=font_size_title+2, fontweight='bold', y=0.995)
    
    # Improve spacing
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Save the plot with high resolution
    plt.savefig(f'{output_dir}/arrival_rate_experiment.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment_results_smoothed(results, output_dir="results/bus"):
    """Generate plots to visualize how system performance changes with arrival rate
    with smoothing to handle dense data"""    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    rates = np.array([r['arrival_rate'] for r in results])
    insp_queues = np.array([r['avg_inspection_queue_length'] for r in results])
    repair_queues = np.array([r['avg_repair_queue_length'] for r in results])
    insp_delays = np.array([r['avg_inspection_delay'] for r in results])
    repair_delays = np.array([r['avg_repair_delay'] for r in results])
    insp_utils = np.array([r['inspection_utilization'] for r in results])
    repair_utils = np.array([r['repair_utilization'] for r in results])
    
    # Sort all data by arrival rate (ascending)
    sorted_indices = np.argsort(rates)
    rates = rates[sorted_indices]
    insp_queues = insp_queues[sorted_indices]
    repair_queues = repair_queues[sorted_indices]
    insp_delays = insp_delays[sorted_indices]
    repair_delays = repair_delays[sorted_indices]
    insp_utils = insp_utils[sorted_indices]
    repair_utils = repair_utils[sorted_indices]
    
    # Apply smoothing if there are enough data points
    n_points = len(rates)
    if n_points >= 11:  # Minimum points needed for savgol with window size 11
        window_size = min(11, n_points if n_points % 2 == 1 else n_points - 1)
        if window_size % 2 == 0:  # Must be odd
            window_size -= 1
        poly_order = min(3, window_size - 1)
        
        # Apply Savitzky-Golay filter for smoothing
        insp_queues_smooth = savgol_filter(insp_queues, window_size, poly_order)
        repair_queues_smooth = savgol_filter(repair_queues, window_size, poly_order)
        insp_delays_smooth = savgol_filter(insp_delays, window_size, poly_order)
        repair_delays_smooth = savgol_filter(repair_delays, window_size, poly_order)
        insp_utils_smooth = savgol_filter(insp_utils, window_size, poly_order)
        repair_utils_smooth = savgol_filter(repair_utils, window_size, poly_order)
    else:  # Not enough points for smoothing
        insp_queues_smooth = insp_queues
        repair_queues_smooth = repair_queues
        insp_delays_smooth = insp_delays
        repair_delays_smooth = repair_delays
        insp_utils_smooth = insp_utils
        repair_utils_smooth = repair_utils
    
    # Create a clear figure with improved styling
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Custom colors with high contrast
    colors = ['#1E88E5', '#FFA000']  # Blue and amber
    
    # Common parameters
    linewidth_smooth = 3
    linewidth_data = 1.2
    alpha_data = 0.3
    alpha_fill = 0.15
    
    # Function to plot both raw data and smoothed curve
    def plot_with_smoothing(ax, x, y_data, y_smooth, color, label):
        # Plot original data with lower opacity
        ax.plot(x, y_data, 'o', color=color, alpha=alpha_data, markersize=3, label=f'{label} (raw)')
        # Plot smoothed line with higher prominence
        ax.plot(x, y_smooth, '-', color=color, linewidth=linewidth_smooth, label=f'{label} (smoothed)')
        # Add light fill below the smoothed line
        ax.fill_between(x, y_smooth, alpha=alpha_fill, color=color)
    
    # 1. Queue lengths vs arrival rate
    ax1 = axes[0, 0]
    plot_with_smoothing(ax1, rates, insp_queues, insp_queues_smooth, colors[0], 'Inspection Queue')
    plot_with_smoothing(ax1, rates, repair_queues, repair_queues_smooth, colors[1], 'Repair Queue')
    
    ax1.set_title('Average Queue Length vs Arrival Rate', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Arrival Rate (buses/hour)', fontsize=12)
    ax1.set_ylabel('Average Queue Length', fontsize=12)
    ax1.legend(fontsize=10)
    
    # 2. Delays vs arrival rate
    ax2 = axes[0, 1]
    plot_with_smoothing(ax2, rates, insp_delays, insp_delays_smooth, colors[0], 'Inspection Delay')
    plot_with_smoothing(ax2, rates, repair_delays, repair_delays_smooth, colors[1], 'Repair Delay')
    
    ax2.set_title('Average Delay vs Arrival Rate', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Arrival Rate (buses/hour)', fontsize=12)
    ax2.set_ylabel('Average Delay (hours)', fontsize=12)
    ax2.legend(fontsize=10)
    
    # 3. Utilization vs arrival rate with threshold
    ax3 = axes[1, 0]
    plot_with_smoothing(ax3, rates, insp_utils, insp_utils_smooth, colors[0], 'Inspector')
    plot_with_smoothing(ax3, rates, repair_utils, repair_utils_smooth, colors[1], 'Repair Stations')
    
    # Add utilization threshold line
    util_threshold = 0.9
    ax3.axhline(y=util_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Critical Utilization ({util_threshold})')
    
    ax3.set_title('Resource Utilization vs Arrival Rate', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Arrival Rate (buses/hour)', fontsize=12)
    ax3.set_ylabel('Utilization', fontsize=12)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=10)
    
    # 4. System stability analysis with threshold
    ax4 = axes[1, 1]
    queue_threshold = 10
    ax4.axhline(y=queue_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Stability Threshold ({queue_threshold})')
    
    plot_with_smoothing(ax4, rates, insp_queues, insp_queues_smooth, colors[0], 'Inspection Queue')
    plot_with_smoothing(ax4, rates, repair_queues, repair_queues_smooth, colors[1], 'Repair Queue')
    
    # Find where inspection queue crosses the threshold (using smoothed data)
    critical_points = []
    for i in range(len(rates)-1):
        if insp_queues_smooth[i] < queue_threshold and insp_queues_smooth[i+1] >= queue_threshold:
            # Linear interpolation to find more precise crossing point
            x1, y1 = rates[i], insp_queues_smooth[i]
            x2, y2 = rates[i+1], insp_queues_smooth[i+1]
            critical_rate = x1 + (queue_threshold - y1) * (x2 - x1) / (y2 - y1)
            critical_points.append(critical_rate)
    
    # Add annotation for first critical point if found
    if critical_points:
        critical_rate = critical_points[0]
        ax4.axvline(x=critical_rate, color='red', linestyle=':', linewidth=2, alpha=0.5)
        ax4.annotate(f'Critical Rate: ~{critical_rate:.2f}',
                    xy=(critical_rate, queue_threshold),
                    xytext=(critical_rate-0.2, queue_threshold+2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    ax4.set_title('System Stability Analysis', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Arrival Rate (buses/hour)', fontsize=12)
    ax4.set_ylabel('Average Queue Length', fontsize=12)
    ax4.legend(loc='upper left', fontsize=10)
    
    # Add overall title
    plt.suptitle('Bus Maintenance Facility Performance Analysis\n(with Smoothed Trends)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Adjust spacing
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # Save the high-resolution figure
    plt.savefig(f'{output_dir}/arrival_rate_experiment_smoothed.png', dpi=300, bbox_inches='tight')
    
    # Generate small multiples version - just top plots in larger size
    plt.figure(figsize=(18, 7))
    
    # Queue length plot
    plt.subplot(1, 2, 1)
    rates_idx = np.argsort(rates)  # Get indices that would sort rates
    sorted_rates = rates[rates_idx]
    sorted_insp_queues = insp_queues[rates_idx]
    sorted_repair_queues = repair_queues[rates_idx]
    
    # Use scatter plot for raw data and line for trend
    plt.scatter(sorted_rates, sorted_insp_queues, s=20, alpha=0.5, color=colors[0], label='Inspection Queue (raw)')
    plt.scatter(sorted_rates, sorted_repair_queues, s=20, alpha=0.5, color=colors[1], label='Repair Queue (raw)')
    
    # Sort the smoothed data too
    sorted_insp_smooth = insp_queues_smooth[rates_idx]
    sorted_repair_smooth = repair_queues_smooth[rates_idx]
    
    plt.plot(sorted_rates, sorted_insp_smooth, color=colors[0], linewidth=3, label='Inspection Queue (trend)')
    plt.plot(sorted_rates, sorted_repair_smooth, color=colors[1], linewidth=3, label='Repair Queue (trend)')
    
    plt.axhline(y=queue_threshold, color='red', linestyle='--', linewidth=2, label=f'Stability Threshold ({queue_threshold})')
    
    plt.title('Queue Length vs Arrival Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Arrival Rate (buses/hour)', fontsize=12)
    plt.ylabel('Average Queue Length', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Utilization plot
    plt.subplot(1, 2, 2)
    sorted_insp_utils = insp_utils[rates_idx]
    sorted_repair_utils = repair_utils[rates_idx]
    sorted_insp_utils_smooth = insp_utils_smooth[rates_idx]
    sorted_repair_utils_smooth = repair_utils_smooth[rates_idx]
    
    plt.scatter(sorted_rates, sorted_insp_utils, s=20, alpha=0.5, color=colors[0], label='Inspector (raw)')
    plt.scatter(sorted_rates, sorted_repair_utils, s=20, alpha=0.5, color=colors[1], label='Repair Stations (raw)')
    
    plt.plot(sorted_rates, sorted_insp_utils_smooth, color=colors[0], linewidth=3, label='Inspector (trend)')
    plt.plot(sorted_rates, sorted_repair_utils_smooth, color=colors[1], linewidth=3, label='Repair Stations (trend)')
    
    plt.axhline(y=util_threshold, color='red', linestyle='--', linewidth=2, label=f'Critical Utilization ({util_threshold})')
    
    plt.title('Resource Utilization vs Arrival Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Arrival Rate (buses/hour)', fontsize=12)
    plt.ylabel('Utilization', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bus_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_n_simulation_results(stats, n, config):
    """Plot results from multiple simulations showing averages and min-max ranges
    
    Args:
        stats: Dictionary containing aggregate statistics
        n: Number of simulations run
        config: Simulation configuration
    """
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Results from {n} Simulations (Arrival Rate: {config.arrival_mean:.2f} buses/hour)', fontsize=16)
    
    # Plot 1: Queue Lengths
    ax = axs[0, 0]
    metrics = ['inspection_queue', 'repair_queue']
    labels = ['Inspection Queue', 'Repair Queue']
    values = [stats['avg_inspection_queue'], stats['avg_repair_queue']]
    min_vals = [stats['min_inspection_queue'], stats['min_repair_queue']]
    max_vals = [stats['max_inspection_queue'], stats['max_repair_queue']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Plot bars with error bars showing min-max range
    bars = ax.bar(x, values, width, yerr=np.array([
        [v - min_v for v, min_v in zip(values, min_vals)],
        [max_v - v for v, max_v in zip(values, max_vals)]
    ]), capsize=10, label='Average', color='skyblue')
    
    ax.set_ylabel('Queue Length')
    ax.set_title('Average Queue Lengths with Min-Max Range')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels moved to the right of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(10, 3),  # horizontal offset to the right, vertical offset
                    textcoords="offset points",
                    ha='left', va='bottom')  # left alignment pushes text to the right
    
    # Plot 2: Delays
    ax = axs[0, 1]
    metrics = ['inspection_delay', 'repair_delay']
    labels = ['Inspection Delay', 'Repair Delay']
    values = [stats['avg_inspection_delay'], stats['avg_repair_delay']]
    min_vals = [stats['min_inspection_delay'], stats['min_repair_delay']]
    max_vals = [stats['max_inspection_delay'], stats['max_repair_delay']]
    
    x = np.arange(len(metrics))
    
    bars = ax.bar(x, values, width, yerr=np.array([
        [v - min_v for v, min_v in zip(values, min_vals)],
        [max_v - v for v, max_v in zip(values, max_vals)]
    ]), capsize=10, label='Average', color='lightgreen')
    
    ax.set_ylabel('Time (hours)')
    ax.set_title('Average Delays with Min-Max Range')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(10, 3),  # horizontal offset to the right, vertical offset
                    textcoords="offset points",
                    ha='left', va='bottom')  # left alignment pushes text to the right
    
    # Plot 3: Utilization
    ax = axs[1, 0]
    metrics = ['inspection_utilization', 'repair_utilization']
    labels = ['Inspection Stations', 'Repair Stations']
    values = [stats['avg_inspection_utilization'], stats['avg_repair_utilization']]
    min_vals = [stats['min_inspection_utilization'], stats['min_repair_utilization']]
    max_vals = [stats['max_inspection_utilization'], stats['max_repair_utilization']]
    
    x = np.arange(len(metrics))
    
    bars = ax.bar(x, values, width, yerr=np.array([
        [v - min_v for v, min_v in zip(values, min_vals)],
        [max_v - v for v, max_v in zip(values, max_vals)]
    ]), capsize=10, label='Average', color='salmon')
    
    ax.set_ylabel('Utilization')
    ax.set_title('Average Station Utilization with Min-Max Range')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(10, 3),  # horizontal offset to the right, vertical offset
                    textcoords="offset points",
                    ha='left', va='bottom')  # left alignment pushes text to the right
    
    # Plot 4: Buses Processed
    ax = axs[1, 1]
    ax.set_title('Buses Processed Statistics')
    
    # Create text summary instead of a graph
    textstr = '\n'.join((
        f'Total Buses Processed: {stats["total_buses_processed"]}',
        f'Average Buses per Simulation: {stats["avg_buses_processed"]:.2f}',
        f'Minimum Buses in a Simulation: {stats["min_buses_processed"]}',
        f'Maximum Buses in a Simulation: {stats["max_buses_processed"]}',
        f'Range (Max-Min): {stats["max_buses_processed"] - stats["min_buses_processed"]}',
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.5, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center', bbox=props)
    ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'multiple_simulation_results.png'), dpi=300)
    plt.close()
    
    print(f"Multiple simulation results plot saved to {output_dir}/multiple_simulation_results.png")

def save_report(results, config, is_experiment=False, max_stable_rate=None, output_dir="results/bus"):
    """Save a text report of simulation results"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/simulation_report.txt', 'w') as f:
        # Write configuration information
        f.write("=== Bus Maintenance Facility Simulation Report ===\n\n")
        f.write("Configuration:\n")
        for key, value in vars(config).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        if is_experiment:
            # Write experiment results
            f.write("Experiment Results:\n")
            if max_stable_rate:
                f.write(f"Maximum stable arrival rate: {max_stable_rate['rate']:.3f} buses/hour\n")
                f.write(f"(equivalent to mean interarrival time: {1/max_stable_rate['rate']:.3f} hours)\n")
                f.write(f"At maximum stable rate - Inspection utilization: {max_stable_rate['inspection_util']:.2f}\n")
                f.write(f"At maximum stable rate - Repair utilization: {max_stable_rate['repair_util']:.2f}\n")
            else:
                f.write("No stable arrival rate found in the tested range.\n")
        else:
            # Write simulation results
            f.write("Simulation Results:\n")
            f.write(f"Buses processed: {results['buses_processed']}\n")
            f.write(f"Average inspection queue length: {results['avg_inspection_queue_length']:.2f}\n")
            f.write(f"Average repair queue length: {results['avg_repair_queue_length']:.2f}\n")
            f.write(f"Average inspection delay: {results['avg_inspection_delay']:.2f} hours\n")
            f.write(f"Average repair delay: {results['avg_repair_delay']:.2f} hours\n")
            f.write(f"Inspection station utilization: {results['inspection_utilization']:.2f}\n")
            f.write(f"Repair stations utilization: {results['repair_utilization']:.2f}\n")

def save_n_simulations_report(aggregate_stats, config, n, seed_info, output_dir="results/bus"):
    """Save a text report of multiple simulation results
    
    Args:
        aggregate_stats: Dictionary containing aggregate statistics
        config: Simulation configuration
        n: Number of simulations run
        seed_info: Information about seeds used (base seed and range)
        output_dir: Directory to save the report
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/multiple_simulations_report.txt', 'w') as f:
        # Write header
        f.write("=== Bus Maintenance Facility Multiple Simulations Report ===\n\n")
        
        # Write simulation metadata
        f.write(f"Number of Simulations Run: {n}\n")
        f.write(f"Base Random Seed: {seed_info['base']}\n")
        f.write(f"Seed Range: {seed_info['base']} to {seed_info['base'] + n - 1}\n\n")
        
        # Write configuration information
        f.write("Configuration:\n")
        for key, value in vars(config).items():
            if key != 'seed':  # Already reported above
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Write aggregate results
        f.write("Aggregate Simulation Results:\n")
        f.write(f"Total buses processed across all simulations: {aggregate_stats['total_buses_processed']}\n")
        f.write(f"Average buses processed per simulation: {aggregate_stats['avg_buses_processed']:.2f}\n")
        f.write(f"  Min: {aggregate_stats['min_buses_processed']}, Max: {aggregate_stats['max_buses_processed']}\n")
        f.write(f"  Range: {aggregate_stats['max_buses_processed'] - aggregate_stats['min_buses_processed']}\n\n")
        
        f.write(f"Average inspection queue length: {aggregate_stats['avg_inspection_queue']:.2f}\n")
        f.write(f"  Min: {aggregate_stats['min_inspection_queue']:.2f}, Max: {aggregate_stats['max_inspection_queue']:.2f}\n\n")
        
        f.write(f"Average repair queue length: {aggregate_stats['avg_repair_queue']:.2f}\n")
        f.write(f"  Min: {aggregate_stats['min_repair_queue']:.2f}, Max: {aggregate_stats['max_repair_queue']:.2f}\n\n")
        
        f.write(f"Average inspection delay: {aggregate_stats['avg_inspection_delay']:.2f} hours\n")
        f.write(f"  Min: {aggregate_stats['min_inspection_delay']:.2f}, Max: {aggregate_stats['max_inspection_delay']:.2f}\n\n")
        
        f.write(f"Average repair delay: {aggregate_stats['avg_repair_delay']:.2f} hours\n")
        f.write(f"  Min: {aggregate_stats['min_repair_delay']:.2f}, Max: {aggregate_stats['max_repair_delay']:.2f}\n\n")
        
        f.write(f"Average inspection station utilization: {aggregate_stats['avg_inspection_utilization']:.2f}\n")
        f.write(f"  Min: {aggregate_stats['min_inspection_utilization']:.2f}, Max: {aggregate_stats['max_inspection_utilization']:.2f}\n\n")
        
        f.write(f"Average repair stations utilization: {aggregate_stats['avg_repair_utilization']:.2f}\n")
        f.write(f"  Min: {aggregate_stats['min_repair_utilization']:.2f}, Max: {aggregate_stats['max_repair_utilization']:.2f}\n")
        
        # Add statistical summary
        f.write("\nStatistical Summary:\n")
        f.write(f"Standard deviation of buses processed: {statistics.stdev(aggregate_stats.get('buses_processed_list', [0])):.2f}\n")
        f.write(f"Standard deviation of inspection queue: {statistics.stdev(aggregate_stats.get('inspection_queue_list', [0])):.2f}\n")
        f.write(f"Standard deviation of repair queue: {statistics.stdev(aggregate_stats.get('repair_queue_list', [0])):.2f}\n")
        
        # Add timestamp
        from datetime import datetime
        f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Multiple simulations report saved to {output_dir}/multiple_simulations_report.txt")
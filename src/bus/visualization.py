"""
Visualization functions for bus simulation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_results(stats, output_dir="results/bus", title="Bus Maintenance Simulation Results"):
    """Generate plots for the simulation results"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Create time axis for time series plots (in hours)
    # Use linspace to ensure exact same length as the data arrays
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
    ax1.plot(smooth_time, smooth_insp_queue, label='Inspection Queue (smoothed)')
    ax1.plot(smooth_time, smooth_repair_queue, label='Repair Queue (smoothed)')
    
    # Also show the original data with lower opacity for reference
    ax1.plot(time_points, stats['inspection_queue_lengths'], 'b-', alpha=0.2)
    ax1.plot(time_points, stats['repair_queue_lengths'], 'orange', alpha=0.2)
    ax1.set_title('Queue Lengths Over Time')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Queue Length')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calculate cumulative utilization over time for better interpretation
    # Use array operations to ensure dimension consistency
    n_points = len(stats['inspection_utilization'])
    divisor = np.arange(1, n_points + 1)
    cum_insp_util = np.cumsum(stats['inspection_utilization']) / divisor
    cum_repair_util = np.cumsum(stats['repair_utilization']) / divisor
    
    # Utilization over time - plot cumulative average
    ax2.plot(time_points, cum_insp_util, label='Inspector (cumulative avg)')
    ax2.plot(time_points, cum_repair_util, label='Repair Stations (cumulative avg)')
    
    # Show instantaneous utilization with lower opacity
    ax2.plot(time_points, stats['inspection_utilization'], 'b-', alpha=0.1)
    ax2.plot(time_points, stats['repair_utilization'], 'orange', alpha=0.1)
    ax2.set_title('Resource Utilization Over Time')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Utilization')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    
    # Delay histograms
    if stats['inspection_delays']:
        ax3.hist(stats['inspection_delays'], bins=20, alpha=0.7)
        ax3.set_title('Inspection Delays')
        ax3.set_xlabel('Delay (hours)')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
    
    if stats['repair_delays']:
        ax4.hist(stats['repair_delays'], bins=20, alpha=0.7)
        ax4.set_title('Repair Delays')
        ax4.set_xlabel('Delay (hours)')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bus_simulation_results.png')


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
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Queue lengths vs arrival rate
    ax1.plot(rates, insp_queues, 'o-', label='Inspection Queue', linewidth=2)
    ax1.plot(rates, repair_queues, 's-', label='Repair Queue', linewidth=2)
    ax1.set_title('Average Queue Length vs Arrival Rate', fontsize=12)
    ax1.set_xlabel('Arrival Rate (buses/hour)', fontsize=11)
    ax1.set_ylabel('Average Queue Length', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Delays vs arrival rate
    ax2.plot(rates, insp_delays, 'o-', label='Inspection Delay', linewidth=2)
    ax2.plot(rates, repair_delays, 's-', label='Repair Delay', linewidth=2)
    ax2.set_title('Average Delay vs Arrival Rate', fontsize=12)
    ax2.set_xlabel('Arrival Rate (buses/hour)', fontsize=11)
    ax2.set_ylabel('Average Delay (hours)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Utilization vs arrival rate
    ax3.plot(rates, insp_utils, 'o-', label='Inspector', linewidth=2)
    ax3.plot(rates, repair_utils, 's-', label='Repair Stations', linewidth=2)
    ax3.set_title('Resource Utilization vs Arrival Rate', fontsize=12)
    ax3.set_xlabel('Arrival Rate (buses/hour)', fontsize=11)
    ax3.set_ylabel('Utilization', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)  # Utilization is between 0 and 1
    ax3.legend(fontsize=10)
    
    # System stability threshold visualization
    queue_threshold = 10
    ax4.axhline(y=queue_threshold, color='r', linestyle='--', label=f'Stability Threshold ({queue_threshold})', linewidth=2)
    ax4.plot(rates, insp_queues, 'o-', label='Inspection Queue', linewidth=2)
    ax4.plot(rates, repair_queues, 's-', label='Repair Queue', linewidth=2)
    ax4.set_title('System Stability Analysis', fontsize=12)
    ax4.set_xlabel('Arrival Rate (buses/hour)', fontsize=11)
    ax4.set_ylabel('Average Queue Length', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/arrival_rate_experiment.png')
    plt.close()


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
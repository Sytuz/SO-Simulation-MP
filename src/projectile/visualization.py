"""
Visualization functions for projectile motion simulation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from .config import SimConfig


def plot_results(euler_results, rk4_results, config, output_dir=None):
    """Create plots comparing the two methods"""
    if output_dir is None:
        output_dir = config.output_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Common styling parameters
    colors = ['#1976D2', '#D32F2F']  # Blue for Euler, Red for RK4
    linewidth = 2.0
    fontsize_title = 14
    fontsize_axis = 12
    fontsize_legend = 10
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Trajectory plot
    axs[0, 0].plot(euler_results['x'], euler_results['z'], color=colors[0], 
                   linewidth=linewidth, label='Euler')
    axs[0, 0].plot(rk4_results['x'], rk4_results['z'], color=colors[1], 
                   linewidth=linewidth, linestyle='--', label='RK4')
    axs[0, 0].set_title('Projectile Trajectory', fontsize=fontsize_title, fontweight='bold')
    axs[0, 0].set_xlabel('x position (m)', fontsize=fontsize_axis)
    axs[0, 0].set_ylabel('z position (m)', fontsize=fontsize_axis)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=fontsize_legend)
    
    # Position vs time
    axs[0, 1].plot(euler_results['time'], euler_results['x'], color=colors[0], 
                   linewidth=linewidth, label='x - Euler')
    axs[0, 1].plot(euler_results['time'], euler_results['z'], color=colors[0], 
                   linewidth=linewidth, linestyle=':', label='z - Euler')
    axs[0, 1].plot(rk4_results['time'], rk4_results['x'], color=colors[1], 
                   linewidth=linewidth, linestyle='--', label='x - RK4')
    axs[0, 1].plot(rk4_results['time'], rk4_results['z'], color=colors[1], 
                   linewidth=linewidth, linestyle='-.', label='z - RK4')
    axs[0, 1].set_title('Position vs Time', fontsize=fontsize_title, fontweight='bold')
    axs[0, 1].set_xlabel('Time (s)', fontsize=fontsize_axis)
    axs[0, 1].set_ylabel('Position (m)', fontsize=fontsize_axis)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=fontsize_legend)
    
    # Velocity vs time
    axs[1, 0].plot(euler_results['time'], euler_results['vx'], color=colors[0], 
                   linewidth=linewidth, label='vx - Euler')
    axs[1, 0].plot(euler_results['time'], euler_results['vz'], color=colors[0], 
                   linewidth=linewidth, linestyle=':', label='vz - Euler')
    axs[1, 0].plot(rk4_results['time'], rk4_results['vx'], color=colors[1], 
                   linewidth=linewidth, linestyle='--', label='vx - RK4')
    axs[1, 0].plot(rk4_results['time'], rk4_results['vz'], color=colors[1], 
                   linewidth=linewidth, linestyle='-.', label='vz - RK4')
    axs[1, 0].set_title('Velocity vs Time', fontsize=fontsize_title, fontweight='bold')
    axs[1, 0].set_xlabel('Time (s)', fontsize=fontsize_axis)
    axs[1, 0].set_ylabel('Velocity (m/s)', fontsize=fontsize_axis)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=fontsize_legend)
    
    # Error plot
    if config.method == "both":
        x_diff = np.abs(euler_results['x'] - rk4_results['x'])
        z_diff = np.abs(euler_results['z'] - rk4_results['z'])
        axs[1, 1].plot(euler_results['time'], x_diff, color='#1976D2', 
                      linewidth=linewidth, label='x position difference')
        axs[1, 1].plot(euler_results['time'], z_diff, color='#388E3C', 
                      linewidth=linewidth, label='z position difference')
        axs[1, 1].set_title('Difference Between Methods', fontsize=fontsize_title, fontweight='bold')
        axs[1, 1].set_xlabel('Time (s)', fontsize=fontsize_axis)
        axs[1, 1].set_ylabel('Absolute Difference (m)', fontsize=fontsize_axis)
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend(fontsize=fontsize_legend)
    
    # Add an overall title
    plt.suptitle('Projectile Motion Simulation Results', fontsize=fontsize_title+2, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/projectile_simulation.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence_study(results, euler_order, rk4_order, output_dir):
    """Plot the results of a convergence study"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(12, 8))
    
    dt_array = np.array([r['dt'] for r in results])
    euler_errors = np.array([r['euler_impact_error'] for r in results])
    rk4_errors = np.array([r['rk4_impact_error'] for r in results])
    
    # Plot the data points
    plt.loglog(dt_array, euler_errors, 'bo-', linewidth=2, markersize=8, label='Euler')
    plt.loglog(dt_array, rk4_errors, 'ro-', linewidth=2, markersize=8, label='RK4')
    
    # Add best-fit lines
    x_fit = np.logspace(np.log10(min(dt_array)), np.log10(max(dt_array)), 100)
    
    # Find coefficients for best fit lines (y = c * x^order)
    c_euler = np.mean(euler_errors / dt_array ** euler_order)
    c_rk4 = np.mean(rk4_errors / dt_array ** rk4_order)
    
    y_fit_euler = c_euler * x_fit ** euler_order
    y_fit_rk4 = c_rk4 * x_fit ** rk4_order
    
    plt.loglog(x_fit, y_fit_euler, 'b--', linewidth=1.5, alpha=0.7)
    plt.loglog(x_fit, y_fit_rk4, 'r--', linewidth=1.5, alpha=0.7)
    
    # Add theoretical reference lines
    ref_x = np.array([min(dt_array), max(dt_array)])
    plt.loglog(ref_x, ref_x ** 1, 'k:', linewidth=1, alpha=0.5, label='O(Δt¹)')
    plt.loglog(ref_x, ref_x ** 2, 'k-.', linewidth=1, alpha=0.5, label='O(Δt²)')
    plt.loglog(ref_x, ref_x ** 3, 'k--', linewidth=1, alpha=0.5, label='O(Δt³)')
    plt.loglog(ref_x, ref_x ** 4, 'k-', linewidth=1, alpha=0.5, label='O(Δt⁴)')
    
    plt.xlabel('Time step (s)', fontsize=12)
    plt.ylabel('Impact point error (m)', fontsize=12)
    plt.title('Error vs Time Step', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    
    # Create custom legend with convergence rates
    plt.legend(title=f'Convergence rates:\nEuler: O(Δt^{euler_order:.2f})\nRK4: O(Δt^{rk4_order:.2f})', 
               fontsize=10, loc='lower right')
    
    # Add annotations with exact values
    for i, dt in enumerate(dt_array):
        plt.annotate(f'{dt}s', 
                    xy=(dt, euler_errors[i]), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', 
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence_study.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_report(config, euler_results=None, rk4_results=None, precision=None, 
               landing_euler=None, landing_rk4=None, convergence_results=None, 
               output_dir=None):
    """Save a text report of simulation results"""
    if output_dir is None:
        output_dir = config.output_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/simulation_report.txt', 'w') as f:
        # Write configuration information
        f.write("=== Projectile Motion Simulation Report ===\n\n")
        f.write("Configuration:\n")
        for key, value in vars(config).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Standard simulation results
        if euler_results and config.method in ["euler", "both"]:
            f.write("Euler Method Results:\n")
            if landing_euler and landing_euler['found']:
                f.write(f"  Landing time: {landing_euler['time']:.2f} s\n")
                f.write(f"  Landing distance: {landing_euler['distance']:.2f} m\n")
            else:
                f.write("  No landing point detected\n")
            f.write("\n")
            
        if rk4_results and config.method in ["rk4", "both"]:
            f.write("RK4 Method Results:\n")
            if landing_rk4 and landing_rk4['found']:
                f.write(f"  Landing time: {landing_rk4['time']:.2f} s\n")
                f.write(f"  Landing distance: {landing_rk4['distance']:.2f} m\n")
            else:
                f.write("  No landing point detected\n")
            f.write("\n")
            
        # Precision comparison
        if precision and config.method == "both":
            f.write("Precision Comparison:\n")
            f.write(f"  Euler X error: {precision['euler_x_error']:.6f} m\n")
            f.write(f"  Euler Z error: {precision['euler_z_error']:.6f} m\n")
            f.write(f"  RK4 X error: {precision['rk4_x_error']:.6f} m\n")
            f.write(f"  RK4 Z error: {precision['rk4_z_error']:.6f} m\n")
            f.write(f"  Euler total error: {precision['euler_error_total']:.6f} m\n")
            f.write(f"  RK4 total error: {precision['rk4_error_total']:.6f} m\n")
            f.write(f"  RK4 is {precision['euler_error_total'] / precision['rk4_error_total']:.1f}x more accurate than Euler\n")
            f.write("\n")
            
        # Convergence study results
        if convergence_results:
            f.write("Convergence Study Results:\n")
            f.write(f"  Euler method convergence order: O(Δt^{convergence_results['euler_order']:.2f})\n")
            f.write(f"  RK4 method convergence order: O(Δt^{convergence_results['rk4_order']:.2f})\n")
            f.write("\n")
            f.write("  Detailed results by time step:\n")
            for result in convergence_results['results']:
                f.write(f"  dt = {result['dt']:.5f} s:\n")
                f.write(f"    Euler error: {result['euler_impact_error']:.6f} m\n")
                f.write(f"    RK4 error: {result['rk4_impact_error']:.6f} m\n")
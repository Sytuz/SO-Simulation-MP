"""
Main entry point for projectile motion simulation.
"""
import os
import sys
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projectile.config import load_config
from projectile.simulation import ProjectileSimulation
from projectile.visualization import plot_results, save_report
from projectile.analysis import run_precision_study


def run_simulation(config):
    """Run a single simulation with the specified configuration"""
    print("\nRunning simulation...")
    sim = ProjectileSimulation(config)
    euler_results, rk4_results = sim.run()
    
    # Calculate precision comparison if using both methods
    precision = None
    if config.method == "both":
        # Calculate maximum differences
        max_x_diff = np.max(np.abs(euler_results['x'] - rk4_results['x']))
        max_z_diff = np.max(np.abs(euler_results['z'] - rk4_results['z']))
        print(f"\nMaximum difference between methods:")
        print(f"X position: {max_x_diff:.6f} m")
        print(f"Z position: {max_z_diff:.6f} m")
        
        # Calculate precision comparison
        precision = sim.compare_precision()
        print(f"\nAverage error compared to reference solution:")
        print(f"Euler X: {precision['euler_x_error']:.6f} m")
        print(f"Euler Z: {precision['euler_z_error']:.6f} m")
        print(f"RK4 X: {precision['rk4_x_error']:.6f} m")
        print(f"RK4 Z: {precision['rk4_z_error']:.6f} m")
        print(f"Euler total error: {precision['euler_error_total']:.6f} m")
        print(f"RK4 total error: {precision['rk4_error_total']:.6f} m")
        print(f"RK4 is {precision['euler_error_total'] / precision['rk4_error_total']:.1f}x more accurate than Euler")
    
    # Find landing positions
    landing_euler = None
    landing_rk4 = None
    
    if config.method in ["euler", "both"]:
        landing_euler = sim.find_landing_position(euler_results)
        if landing_euler['found']:
            print(f"\nEuler method landing position:")
            print(f"Time: {landing_euler['time']:.2f} s")
            print(f"Distance: {landing_euler['distance']:.2f} m")
    
    if config.method in ["rk4", "both"]:
        landing_rk4 = sim.find_landing_position(rk4_results)
        if landing_rk4['found']:
            print(f"\nRK4 method landing position:")
            print(f"Time: {landing_rk4['time']:.2f} s")
            print(f"Distance: {landing_rk4['distance']:.2f} m")
    
    # Generate plots and save report
    plot_results(euler_results, rk4_results, config, config.output_dir)
    save_report(config, euler_results, rk4_results, precision, landing_euler, landing_rk4)
    
    return euler_results, rk4_results


def run_study(config):
    """Run a precision study with different time steps"""
    print("\nRunning precision study with different time steps...")
    results = run_precision_study(config, output_dir=config.output_dir)
    
    print(f"Euler method convergence order: O(Δt^{results['euler_order']:.2f})")
    print(f"RK4 method convergence order: O(Δt^{results['rk4_order']:.2f})")
    
    # Save the convergence study report
    save_report(config, convergence_results=results, output_dir=config.output_dir)
    
    return results


def main():
    """Main function to parse arguments and run the simulation"""
    parser = argparse.ArgumentParser(description='Projectile Motion Simulation')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--x0', type=float, help='Initial x position')
    parser.add_argument('--z0', type=float, help='Initial z position')
    parser.add_argument('--vx0', type=float, help='Initial x velocity')
    parser.add_argument('--vz0', type=float, help='Initial z velocity')
    parser.add_argument('--mass', type=float, help='Mass of the projectile')
    parser.add_argument('--air', type=float, help='Air resistance coefficient')
    parser.add_argument('--dt', type=float, help='Time step')
    parser.add_argument('--time', type=float, help='Final simulation time')
    parser.add_argument('--method', type=str, choices=['euler', 'rk4', 'both'], 
                        help='Simulation method: "euler", "rk4", or "both"')
    parser.add_argument('--study', action='store_true', 
                        help='Perform precision study with different time steps')
    parser.add_argument('--output', type=str, 
                        help='Output directory for saving results')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments if provided
    if args.x0 is not None:
        config.x0 = args.x0
    if args.z0 is not None:
        config.z0 = args.z0
    if args.vx0 is not None:
        config.vx0 = args.vx0
    if args.vz0 is not None:
        config.vz0 = args.vz0
    if args.mass is not None:
        config.mass = args.mass
    if args.air is not None:
        config.air_resistance = args.air
    if args.dt is not None:
        config.dt = args.dt
    if args.time is not None:
        config.t_final = args.time
    if args.method is not None:
        config.method = args.method
    if args.output is not None:
        config.output_dir = args.output
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Print simulation parameters
    print("\nProjectile Motion Simulation")
    print("===========================")
    print(f"Initial position: ({config.x0}, {config.z0}) m")
    print(f"Initial velocity: ({config.vx0}, {config.vz0}) m/s")
    print(f"Mass: {config.mass} kg")
    print(f"Air resistance coefficient: {config.air_resistance}")
    print(f"Gravity: {config.gravity} m/s²")
    print(f"Time step: {config.dt} s")
    print(f"Simulation time: {config.t_final} s")
    print(f"Method: {config.method}")
    print(f"Output directory: {config.output_dir}")
    
    # Run either precision study or standard simulation
    if args.study:
        results = run_study(config)
    else:
        results = run_simulation(config)
    
    return results


if __name__ == "__main__":
    main()
"""
Main entry point for projectile motion simulation.
"""
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projectile.config import load_config
from projectile.simulation import ProjectileSimulation
from projectile.visualization import plot_results, save_report
from projectile.analysis import run_precision_experiment


def run_simulation(config):
    """Run a single simulation with the specified configuration"""
    print("\nRunning simulation...")
    sim = ProjectileSimulation(config)
    euler_results, rk4_results = sim.run()
    
    # Print final state information in tabular format
    print("\n===== SIMULATION RESULTS =====")
    print("\nFinal State:")
    headers = ["Method", "Time (s)", "x (m)", "z (m)", "vx (m/s)", "vz (m/s)"]
    rows = []
    
    if config.method in ["euler", "both"]:
        rows.append([
            "Euler",
            f"{euler_results['time'][-1]:.2f}",
            f"{euler_results['x'][-1]:.2f}",
            f"{euler_results['z'][-1]:.2f}",
            f"{euler_results['vx'][-1]:.2f}",
            f"{euler_results['vz'][-1]:.2f}"
        ])
    
    if config.method in ["rk4", "both"]:
        rows.append([
            "RK4",
            f"{rk4_results['time'][-1]:.2f}",
            f"{rk4_results['x'][-1]:.2f}",
            f"{rk4_results['z'][-1]:.2f}",
            f"{rk4_results['vx'][-1]:.2f}",
            f"{rk4_results['vz'][-1]:.2f}"
        ])
    
    # Print table
    col_widths = [10, 10, 10, 10, 10, 10]
    print_table(headers, rows, col_widths)
    
    # Calculate precision comparison
    precision = sim.compare_precision()
    
    # Print precision comparison in tabular format
    print("\nComparison with high-precision reference solution:")
    headers = ["Method", "X Error (m)", "Z Error (m)", "Total Error (m)"]
    rows = [
        ["Euler", f"{precision['euler_x_error']:.6f}", f"{precision['euler_z_error']:.6f}", f"{precision['euler_error_total']:.6f}"],
        ["RK4", f"{precision['rk4_x_error']:.6f}", f"{precision['rk4_z_error']:.6f}", f"{precision['rk4_error_total']:.6f}"]
    ]
    print_table(headers, rows, [10, 15, 15, 15])
    
    # Calculate accuracy ratio with protection against very small values
    if precision['rk4_error_total'] > 1e-10:
        ratio = precision['euler_error_total'] / precision['rk4_error_total']
        print(f"\nRK4 is approximately {ratio:.1f}x more accurate than Euler")
    else:
        print("\nRK4 error is extremely small compared to Euler")
    
    # Find and display landing positions in tabular format
    landing_euler = None
    landing_rk4 = None
    
    if config.method in ["euler", "both"]:
        landing_euler = sim.find_landing_position(euler_results)
    
    if config.method in ["rk4", "both"]:
        landing_rk4 = sim.find_landing_position(rk4_results)
    
    print("\nLanding Position:")
    headers = ["Method", "Time (s)", "Distance (m)", "Found"]
    rows = []
    
    if landing_euler:
        rows.append([
            "Euler", 
            f"{landing_euler['time']:.2f}" if landing_euler['found'] else "N/A",
            f"{landing_euler['distance']:.2f}" if landing_euler['found'] else "N/A",
            "Yes" if landing_euler['found'] else "No"
        ])
    
    if landing_rk4:
        rows.append([
            "RK4", 
            f"{landing_rk4['time']:.2f}" if landing_rk4['found'] else "N/A",
            f"{landing_rk4['distance']:.2f}" if landing_rk4['found'] else "N/A",
            "Yes" if landing_rk4['found'] else "No"
        ])
    
    print_table(headers, rows, [10, 10, 15, 10])
    
    # Generate plots and save report
    plot_results(euler_results, rk4_results, config, config.output_dir)
    save_report(config, euler_results, rk4_results, 
                precision, landing_euler, landing_rk4)
    
    return euler_results, rk4_results

def print_table(headers, rows, col_widths):
    """Print a formatted table with headers and rows"""
    # Print header
    header_row = ""
    for i, header in enumerate(headers):
        header_row += f"{header:{col_widths[i]}} "
    print(header_row)
    
    # Print separator
    sep_row = ""
    for width in col_widths:
        sep_row += "-" * width + " "
    print(sep_row)
    
    # Print data rows
    for row in rows:
        row_str = ""
        for i, cell in enumerate(row):
            row_str += f"{cell:{col_widths[i]}} "
        print(row_str)


def run_experiment(config):
    """Run a precision experiment with different time steps"""
    print("\nRunning precision experiment with different time steps...")
    results = run_precision_experiment(config, output_dir=config.output_dir)
    
    # Print average convergence orders from multiple test cases
    print(f"Average Euler method convergence order: O(Δt^{results['avg_euler_order']:.2f})")
    print(f"Average RK4 method convergence order: O(Δt^{results['avg_rk4_order']:.2f})")
    
    # Print detailed results for each test case
    print("\nDetailed convergence orders by test case:")
    for case in results['test_cases']:
        print(f"  {case['name']}:")
        print(f"    Euler: O(Δt^{case['euler_order']:.2f})")
        print(f"    RK4: O(Δt^{case['rk4_order']:.2f})")
    
    # Save the convergence experiment report
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
    parser.add_argument('--experiment', action='store_true', 
                        help='Perform precision experiment with different time steps')
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
    
    # Run either precision experiment or standard simulation
    if args.experiment:
        results = run_experiment(config)
    else:
        results = run_simulation(config)
    
    return results


if __name__ == "__main__":
    main()
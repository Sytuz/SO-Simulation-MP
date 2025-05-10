"""
Main entry point for the bus maintenance facility simulation.
"""
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bus.config import load_config
from bus.simulation import BusMaintenanceSimulation
from bus.visualization import plot_results, plot_experiment_results, save_report
from bus.experiments import run_arrival_rate_experiment


def run_experiment(config, min_rate=0.1, max_rate=4.0, num_points=100):
    """Run the arrival rate experiment
    
    Args:
        config: Simulation configuration
        min_rate: Minimum arrival rate to test (buses/hour)
        max_rate: Maximum arrival rate to test (buses/hour)
        num_points: Number of test points for arrival rate experiment
        
    Returns:
        tuple: Results and max stable rate information
    """
    print("Running arrival rate experiment to find capacity limits...")
    results, max_stable_rate = run_arrival_rate_experiment(
        config, 
        min_rate=min_rate, 
        max_rate=max_rate,
        num_points=num_points
    )
    
    if max_stable_rate:
        print("\nExperiment Results Summary:")
        print(f"Maximum stable arrival rate: {max_stable_rate['rate']:.3f} buses/hour")
        print(f"(equivalent to mean interarrival time: {1/max_stable_rate['rate']:.3f} hours)")
    
    # Save report for experiment
    save_report(results, config, is_experiment=True, 
                max_stable_rate=max_stable_rate, output_dir=config.output_dir)
    
    # Plot experiment results with the specified output directory
    plot_experiment_results(results, output_dir=config.output_dir)
    
    return results, max_stable_rate


def run_single_simulation(config):
    """Run a single simulation with the given configuration
    
    Args:
        config: Simulation configuration
        
    Returns:
        dict: Simulation results
    """
    print(f"Running simulation for {config.simulation_time} hours...")
    sim = BusMaintenanceSimulation(config)
    results = sim.run()
    
    # Print results
    print("\nSimulation Results:")
    print(f"Buses processed: {results['buses_processed']}")
    print(f"Average inspection queue length: {results['avg_inspection_queue_length']:.2f}")
    print(f"Average repair queue length: {results['avg_repair_queue_length']:.2f}")
    print(f"Average inspection delay: {results['avg_inspection_delay']:.2f} hours")
    print(f"Average repair delay: {results['avg_repair_delay']:.2f} hours")
    print(f"Inspection station utilization: {results['inspection_utilization']:.2f}")
    print(f"Repair stations utilization: {results['repair_utilization']:.2f}")
    
    # Save report
    save_report(results, config, output_dir=config.output_dir)
    
    # Generate plots
    plot_results(sim.stats, output_dir=config.output_dir)
    
    return results


def main():
    """Main function to parse arguments and run the simulation"""
    parser = argparse.ArgumentParser(description='Bus Maintenance Facility Simulation')
    parser.add_argument('--config', type=str, 
                        help='Path to configuration file')
    parser.add_argument('--time', type=float, 
                        help='Simulation time in hours')
    parser.add_argument('--seed', type=int, 
                        help='Random seed')
    parser.add_argument('--output', type=str, 
                        help='Output directory for results')
    parser.add_argument('--experiment', action='store_true', 
                        help='Run arrival rate experiment')
    parser.add_argument('--min-rate', type=float, default=0.1, 
                        help='Minimum arrival rate to test (buses/hour)')
    parser.add_argument('--max-rate', type=float, default=4.0, 
                        help='Maximum arrival rate to test (buses/hour)')  
    parser.add_argument('--num-points', type=int, default=100,
                        help='Number of test points for arrival rate experiment')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.time:
        config.simulation_time = args.time
    if args.seed:
        config.seed = args.seed
    if args.output:
        config.output_dir = args.output
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Run either an experiment or a single simulation based on arguments
    if args.experiment:
        results = run_experiment(
            config, 
            min_rate=args.min_rate, 
            max_rate=args.max_rate,
            num_points=args.num_points
        )
    else:
        results = run_single_simulation(config)
    
    return results


if __name__ == "__main__":
    main()
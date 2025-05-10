"""
Main entry point for the bus maintenance facility simulation.
"""
import argparse
import os
import sys
import statistics

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bus.config import load_config
from bus.simulation import BusMaintenanceSimulation
from bus.visualization import plot_results, plot_experiment_results, plot_experiment_results_smoothed, plot_n_simulation_results, save_report, save_n_simulations_report
from bus.experiments import run_arrival_rate_experiment


def run_experiment(config, min_rate=0.1, max_rate=4.0, num_points=200):
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
    plot_experiment_results(results, output_dir="results/bus")
    plot_experiment_results_smoothed(results, output_dir="results/bus")
    
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

def run_n_simulations(config, n=100):
    """Run n simulations with the given configuration
    
    Args:
        config: Simulation configuration
        n: Number of simulations to run
        
    Returns:
        tuple: (List of individual simulation results, Dictionary of aggregate statistics)
    """
    results = []
    
    # Track metrics across all simulations
    buses_processed_list = []
    inspection_queue_list = []
    repair_queue_list = []
    inspection_delay_list = []
    repair_delay_list = []
    inspection_util_list = []
    repair_util_list = []
    
    base_seed = config.seed
    print(f"Starting {n} simulations with arrival rate: {config.arrival_mean:.2f} buses/hour (base seed: {base_seed})")
    
    for i in range(n):
        # Generate a unique seed for this simulation based on the base seed
        simulation_seed = base_seed + i
        
        print(f"Running simulation {i+1}/{n} with seed {simulation_seed}...")
        
        # Create a copy of the config and set the unique seed
        from copy import deepcopy
        sim_config = deepcopy(config)
        sim_config.seed = simulation_seed
        
        sim = BusMaintenanceSimulation(sim_config)
        result = sim.run()
        results.append(result)
        
        # Log detailed results for each run
        print(f"  Simulation {i+1} (seed {simulation_seed}) processed {result['buses_processed']} buses")
        print(f"  Avg inspection queue: {result['avg_inspection_queue_length']:.2f}, Avg repair queue: {result['avg_repair_queue_length']:.2f}")
        
        # Collect metrics for statistical analysis
        buses_processed_list.append(result['buses_processed'])
        inspection_queue_list.append(result['avg_inspection_queue_length'])
        repair_queue_list.append(result['avg_repair_queue_length'])
        inspection_delay_list.append(result['avg_inspection_delay'])
        repair_delay_list.append(result['avg_repair_delay'])
        inspection_util_list.append(result['inspection_utilization'])
        repair_util_list.append(result['repair_utilization'])
    
    # Calculate aggregate statistics
    aggregate_stats = {
        'total_buses_processed': sum(buses_processed_list),
        'avg_buses_processed': statistics.mean(buses_processed_list),
        'min_buses_processed': min(buses_processed_list),
        'max_buses_processed': max(buses_processed_list),
        
        'avg_inspection_queue': statistics.mean(inspection_queue_list),
        'min_inspection_queue': min(inspection_queue_list),
        'max_inspection_queue': max(inspection_queue_list),
        
        'avg_repair_queue': statistics.mean(repair_queue_list),
        'min_repair_queue': min(repair_queue_list),
        'max_repair_queue': max(repair_queue_list),
        
        'avg_inspection_delay': statistics.mean(inspection_delay_list),
        'min_inspection_delay': min(inspection_delay_list),
        'max_inspection_delay': max(inspection_delay_list),
        
        'avg_repair_delay': statistics.mean(repair_delay_list),
        'min_repair_delay': min(repair_delay_list),
        'max_repair_delay': max(repair_delay_list),
        
        'avg_inspection_utilization': statistics.mean(inspection_util_list),
        'min_inspection_utilization': min(inspection_util_list),
        'max_inspection_utilization': max(inspection_util_list),
        
        'avg_repair_utilization': statistics.mean(repair_util_list),
        'min_repair_utilization': min(repair_util_list),
        'max_repair_utilization': max(repair_util_list),
        
        # Store lists for additional statistical calculations
        'buses_processed_list': buses_processed_list,
        'inspection_queue_list': inspection_queue_list,
        'repair_queue_list': repair_queue_list
    }
    
    # Print summary statistics
    print("\nAggregate Results Summary:")
    print(f"Total buses processed across {n} simulations: {aggregate_stats['total_buses_processed']}")
    print(f"Average buses processed per simulation: {aggregate_stats['avg_buses_processed']:.2f} (min: {aggregate_stats['min_buses_processed']}, max: {aggregate_stats['max_buses_processed']})")
    print(f"Average inspection queue length: {aggregate_stats['avg_inspection_queue']:.2f} (min: {aggregate_stats['min_inspection_queue']:.2f}, max: {aggregate_stats['max_inspection_queue']:.2f})")
    print(f"Average repair queue length: {aggregate_stats['avg_repair_queue']:.2f} (min: {aggregate_stats['min_repair_queue']:.2f}, max: {aggregate_stats['max_repair_queue']:.2f})")
    print(f"Average inspection delay: {aggregate_stats['avg_inspection_delay']:.2f} hours (min: {aggregate_stats['min_inspection_delay']:.2f}, max: {aggregate_stats['max_inspection_delay']:.2f})")
    print(f"Average repair delay: {aggregate_stats['avg_repair_delay']:.2f} hours (min: {aggregate_stats['min_repair_delay']:.2f}, max: {aggregate_stats['max_repair_delay']:.2f})")
    
    # Plot the multiple simulation results
    plot_n_simulation_results(aggregate_stats, n, config)
    
    # Generate a comprehensive report file
    seed_info = {'base': base_seed}
    save_n_simulations_report(aggregate_stats, config, n, seed_info, output_dir=config.output_dir)
    
    return results, aggregate_stats

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
    parser.add_argument('--num-simulations', type=int, default=1,
                        help='Number of simulations to run')
    parser.add_argument('--min-rate', type=float, default=0.1, 
                        help='Minimum arrival rate to test (buses/hour)')
    parser.add_argument('--max-rate', type=float, default=4.0, 
                        help='Maximum arrival rate to test (buses/hour)')  
    parser.add_argument('--num-points', type=int, default=200,
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
    elif args.num_simulations > 1:
        results, aggregate_stats = run_n_simulations(config, n=args.num_simulations)
    else:
        results = run_single_simulation(config)
    
    return results


if __name__ == "__main__":
    main()
"""
Experiments for determining system stability and capacity.
"""
import numpy as np
from .config import SimConfig
from .simulation import BusMaintenanceSimulation


def is_system_stable(result, queue_threshold=10, delay_threshold=5.0, util_threshold=0.90):
    """Determine if the system is stable based on multiple criteria
    
    Args:
        result: Simulation result dictionary
        queue_threshold: Maximum acceptable average queue length
        delay_threshold: Maximum acceptable average delay in hours
        util_threshold: Maximum acceptable resource utilization
        
    Returns:
        bool: True if system is considered stable, False otherwise
    """
    # Check queue lengths
    queue_stable = (result['avg_inspection_queue_length'] < queue_threshold and
                    result['avg_repair_queue_length'] < queue_threshold)
    
    # Check waiting times
    delay_stable = (result['avg_inspection_delay'] < delay_threshold and
                    result['avg_repair_delay'] < delay_threshold)
    
    # Check resource utilization
    util_stable = (result['inspection_utilization'] < util_threshold and
                   result['repair_utilization'] < util_threshold)
    
    return queue_stable and delay_stable and util_stable


def run_arrival_rate_experiment(base_config, min_rate=0.1, max_rate=4.0, num_points=200):
    """Run simulations with different arrival rates to find capacity limit
    
    Args:
        base_config: Base simulation configuration
        min_rate: Minimum arrival rate to test (buses/hour)
        max_rate: Maximum arrival rate to test (buses/hour)
        num_points: Number of test points between min and max
    """
    #rates = np.logspace(np.log10(min_rate), np.log10(max_rate), num_points)
    # Generate rates with linear spacing
    rates = np.linspace(min_rate, max_rate, num_points)

    # Sort in ascending order to find stable rates first
    rates = sorted(rates)
    
    results = []
    unstable_count = 0
    
    for rate in rates:
        config = SimConfig(**vars(base_config))
        # Use a longer simulation time for rates approaching capacity limits
        if len(results) > 0 and results[-1]['inspection_utilization'] > 0.8:
            config.simulation_time = max(config.simulation_time, 320.0)  # Increase simulation time
            
        config.arrival_mean = 1/rate  # Convert rate to mean interarrival time
        
        print(f"Running simulation with arrival rate: {rate:.3f} buses/hour")
        sim = BusMaintenanceSimulation(config)
        result = sim.run()
        result['arrival_rate'] = rate
        results.append(result)
        
        # Use comprehensive stability test
        stable = is_system_stable(result)
        
        print(f"Rate: {result['arrival_rate']:.3f} buses/hour - "
              f"Stable: {stable} - "
              f"Insp. util: {result['inspection_utilization']:.2f} - "
              f"Repair util: {result['repair_utilization']:.2f} - "
              f"Insp. queue: {result['avg_inspection_queue_length']:.2f} - "
              f"Repair queue: {result['avg_repair_queue_length']:.2f}")
        
        # Early stopping if multiple consecutive unstable points are detected
        if not stable:
            unstable_count += 1
        else:
            unstable_count = 0
            
        # If 5 unstable points in a row were found and have at least 10 data points, stop the experiment
        # This is a heuristic to avoid running too many simulations after the system has clearly become unstable
        if unstable_count >= 5 and len(results) >= 10:
            print("Multiple consecutive unstable points detected, stopping experiment")
            break
            
        # Also stop if we see extreme instability at any point
        if result['avg_inspection_queue_length'] > 30 or result['avg_repair_queue_length'] > 30:
            print("Extreme queue buildup detected, stopping experiment")
            break
    
    # Determine the maximum stable arrival rate
    max_stable_rate = None
    for result in results:
        # Use the comprehensive stability test
        if is_system_stable(result) and (max_stable_rate is None or result['arrival_rate'] > max_stable_rate['rate']):
            max_stable_rate = {
                'rate': result['arrival_rate'],
                'inspection_util': result['inspection_utilization'],
                'repair_util': result['repair_utilization'],
                'inspection_queue': result['avg_inspection_queue_length'],
                'repair_queue': result['avg_repair_queue_length']
            }
    
    if max_stable_rate:
        print(f"\nMaximum stable arrival rate: {max_stable_rate['rate']:.3f} buses/hour")
        print(f"At maximum stable rate - Inspection utilization: {max_stable_rate['inspection_util']:.2f}, " 
              f"Repair utilization: {max_stable_rate['repair_util']:.2f}")
        print(f"Equivalent mean interarrival time: {1/max_stable_rate['rate']:.3f} hours")
    else:
        print("\nNo stable arrival rate found in the tested range.")
    
    return results, max_stable_rate
"""
Experiments for determining system stability and capacity.
"""
import numpy as np
from .config import SimConfig
from .simulation import BusMaintenanceSimulation
from .visualization import plot_experiment_results


def run_arrival_rate_experiment(base_config, rates=None):
    """Run simulations with different arrival rates to find capacity limit"""
    if rates is None:
        rates = [1/r for r in [4.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
    
    results = []
    
    for rate in rates:
        config = SimConfig(**vars(base_config))
        config.arrival_mean = 1/rate  # Convert rate to mean interarrival time
        
        print(f"Running simulation with arrival rate: {rate:.3f} buses/hour")
        sim = BusMaintenanceSimulation(config)
        result = sim.run()
        result['arrival_rate'] = rate
        results.append(result)
    
    # Determine the maximum stable arrival rate
    max_stable_rate = None
    for result in results:
        is_stable = (result['avg_inspection_queue_length'] < 10 and 
                    result['avg_repair_queue_length'] < 10)
        
        print(f"Rate: {result['arrival_rate']:.3f} buses/hour - "
              f"Stable: {is_stable} - "
              f"Insp. util: {result['inspection_utilization']:.2f} - "
              f"Repair util: {result['repair_utilization']:.2f} - "
              f"Insp. queue: {result['avg_inspection_queue_length']:.2f} - "
              f"Repair queue: {result['avg_repair_queue_length']:.2f}")
        
        if is_stable and (max_stable_rate is None or result['arrival_rate'] > max_stable_rate['rate']):
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
    else:
        print("\nNo stable arrival rate found in the tested range.")
    
    # Plot experiment results (now handled by main.py)
    
    return results, max_stable_rate
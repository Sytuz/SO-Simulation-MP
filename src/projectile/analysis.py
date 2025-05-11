"""
Analysis functions for projectile motion simulation.
"""
import numpy as np
from scipy.stats import linregress
from .config import SimConfig
from .simulation import ProjectileSimulation
from .visualization import plot_convergence_study


def run_precision_study(base_config, dt_values=None, output_dir=None):
    """Run simulations with different time steps to study numerical stability
    
    Args:
        base_config: Base simulation configuration
        dt_values: List of time steps to test
        output_dir: Directory to save results
        
    Returns:
        dict: Results of the convergence study
    """
    if dt_values is None:
        dt_values = [0.1, 0.05, 0.01, 0.005, 0.001]
        
    if output_dir is None:
        output_dir = base_config.output_dir
        
    results = []
    
    # Define very small time step for reference solution
    ref_config = SimConfig(**vars(base_config))
    ref_config.dt = 0.0001
    ref_config.method = "rk4"  # Use RK4 for reference
    
    print(f"Computing reference solution with dt={ref_config.dt}...")
    ref_sim = ProjectileSimulation(ref_config)
    _, ref_results = ref_sim.run()
    
    # Find reference landing position
    landing_info = ref_sim.find_landing_position(ref_results)
    if landing_info['found']:
        ref_impact_x = landing_info['distance']
        print(f"Reference landing distance: {ref_impact_x:.6f} m")
    else:
        # Use the final position as reference
        ref_impact_x = ref_results['x'][-1]
        print("Warning: Couldn't find landing point in reference solution")
        print(f"Using final position as reference: {ref_impact_x:.6f} m")
    
    # Test each time step
    for dt in dt_values:
        print(f"Testing time step dt={dt}...")
        config = SimConfig(**vars(base_config))
        config.dt = dt
        config.method = "both"
        
        sim = ProjectileSimulation(config)
        euler_results, rk4_results = sim.run()
        
        # Find landing positions
        euler_landing = sim.find_landing_position(euler_results)
        rk4_landing = sim.find_landing_position(rk4_results)
        
        # Measure impact points
        if euler_landing['found']:
            euler_impact_x = euler_landing['distance']
        else:
            euler_impact_x = euler_results['x'][-1]  # Use last point
            
        if rk4_landing['found']:
            rk4_impact_x = rk4_landing['distance']
        else:
            rk4_impact_x = rk4_results['x'][-1]  # Use last point
        
        results.append({
            'dt': dt,
            'euler_impact_error': abs(euler_impact_x - ref_impact_x),
            'rk4_impact_error': abs(rk4_impact_x - ref_impact_x)
        })
    
    # Calculate convergence rates using linear regression in log space
    dt_array = np.array([r['dt'] for r in results])
    euler_errors = np.array([r['euler_impact_error'] for r in results])
    rk4_errors = np.array([r['rk4_impact_error'] for r in results])
    
    log_dt = np.log10(dt_array)
    log_euler = np.log10(euler_errors)
    log_rk4 = np.log10(rk4_errors)
    
    # Linear regression in log space gives order of convergence
    euler_slope, euler_intercept, _, _, _ = linregress(log_dt, log_euler)
    rk4_slope, rk4_intercept, _, _, _ = linregress(log_dt, log_rk4)
    
    # Create the convergence plot
    plot_convergence_study(results, euler_slope, rk4_slope, output_dir)
    
    return {
        'results': results, 
        'euler_order': euler_slope, 
        'rk4_order': rk4_slope
    }
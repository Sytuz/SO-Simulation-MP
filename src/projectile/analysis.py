"""
Analysis functions for projectile motion simulation.
"""
import numpy as np
from scipy.stats import linregress
from .config import SimConfig
from .simulation import ProjectileSimulation
from .visualization import plot_convergence_study, plot_convergence_summary


def run_precision_experiment(base_config, dt_values=None, test_cases=None, output_dir=None):
    """Run simulations with different time steps across multiple test scenarios"""
    if dt_values is None:
        dt_values = [0.1, 0.05, 0.01, 0.005, 0.001]
        
    if output_dir is None:
        output_dir = base_config.output_dir
    
    if test_cases is None:
        # Default test cases with different initial conditions and physical parameters
        test_cases = [
            {"name": "No air resistance", "vx0": 50.0, "vz0": 50.0, "air_resistance": 0.0},
            {"name": "Default", "vx0": 50.0, "vz0": 50.0, "air_resistance": 0.01},
            {"name": "High velocity", "vx0": 100.0, "vz0": 80.0, "air_resistance": 0.01},
            {"name": "High air resistance", "vx0": 50.0, "vz0": 50.0, "air_resistance": 0.05}
        ]
    
    all_results = []
    
    # For each test case
    for test_case in test_cases:
        print(f"\nRunning test case: {test_case['name']}")
        
        # Apply test case parameters to base config
        test_config = SimConfig(**vars(base_config))
        for key, value in test_case.items():
            if key != "name" and hasattr(test_config, key):
                setattr(test_config, key, value)
        
        # Set up time range to get landing point for all methods
        test_config.t_final = 20.0  # Ensure enough time to land for all test cases
        
        # Generate reference solution with very small time step
        ref_config = SimConfig(**vars(test_config))
        ref_config.dt = 0.0001  # Very small time step for reference
        ref_config.method = "rk4"  # Use RK4 for reference
        
        print("Computing reference solution...")
        ref_sim = ProjectileSimulation(ref_config)
        _, ref_results = ref_sim.run()
        
        # Find reference landing position
        landing_info = ref_sim.find_landing_position(ref_results)
        if landing_info['found']:
            ref_impact_x = landing_info['distance']
            print(f"Reference landing distance: {ref_impact_x:.6f} m")
        else:
            # Use the final position as reference if no landing found
            ref_impact_x = ref_results['x'][-1]
            print("Warning: Couldn't find reference landing point")
            print(f"Using final position as reference: {ref_impact_x:.6f} m")
        
        case_results = []
        # Test each time step
        for dt in dt_values:
            print(f"Testing time step dt={dt}...")
            config = SimConfig(**vars(test_config))
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
            
            case_results.append({
                'dt': dt,
                'euler_impact_error': abs(euler_impact_x - ref_impact_x),
                'rk4_impact_error': abs(rk4_impact_x - ref_impact_x)
            })
        
        # Calculate convergence rates for this test case
        dt_array = np.array([r['dt'] for r in case_results])
        euler_errors = np.array([r['euler_impact_error'] for r in case_results])
        rk4_errors = np.array([r['rk4_impact_error'] for r in case_results])
        
        log_dt = np.log10(dt_array)
        log_euler = np.log10(euler_errors)
        log_rk4 = np.log10(rk4_errors)
        
        # Linear regression in log space gives order of convergence
        euler_slope, euler_intercept, _, _, _ = linregress(log_dt, log_euler)
        rk4_slope, rk4_intercept, _, _, _ = linregress(log_dt, log_rk4)
        
        all_results.append({
            'name': test_case['name'],
            'results': case_results,
            'euler_order': euler_slope,
            'rk4_order': rk4_slope,
            'parameters': {k: v for k, v in test_case.items() if k != "name"}
        })
        
        # Plot individual test case results
        plot_convergence_study(
            case_results, 
            euler_slope, 
            rk4_slope, 
            f"{output_dir}/case_{test_case['name'].lower().replace(' ', '_')}"
        )
    
    # Calculate average convergence orders
    avg_euler_order = np.mean([case['euler_order'] for case in all_results])
    avg_rk4_order = np.mean([case['rk4_order'] for case in all_results])
    
    print(f"\nAverage Euler convergence order: O(Δt^{avg_euler_order:.2f})")
    print(f"Average RK4 convergence order: O(Δt^{avg_rk4_order:.2f})")
    
    # Plot summary of all test cases
    plot_convergence_summary(all_results, output_dir)
    
    return {
        'test_cases': all_results,
        'avg_euler_order': avg_euler_order,
        'avg_rk4_order': avg_rk4_order
    }
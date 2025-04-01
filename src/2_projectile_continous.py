import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from dataclasses import dataclass
from typing import Tuple
from scipy.stats import linregress
from scipy.interpolate import interp1d


@dataclass
class SimConfig:
    """Configuration parameters for projectile simulation"""
    x0: float = 0.0           # Initial x position (m)
    z0: float = 0.0           # Initial z position (m)
    vx0: float = 50.0         # Initial x velocity (m/s)
    vz0: float = 50.0         # Initial z velocity (m/s)
    mass: float = 1.0         # Mass of projectile (kg)
    air_resistance: float = 0.01  # Air resistance coefficient
    gravity: float = 9.81     # Gravitational acceleration (m/s²)
    dt: float = 0.01          # Time step (s)
    t_final: float = 10.0     # Final simulation time (s)
    method: str = "both"      # Simulation method: "euler", "rk4", or "both"


class ProjectileSimulation:
    """Simulation of projectile motion with air resistance"""
    
    def __init__(self, config: SimConfig):
        self.config = config
        
        # State variables [x, z, vx, vz]
        self.state_euler = np.array([config.x0, config.z0, config.vx0, config.vz0])
        self.state_rk4 = np.array([config.x0, config.z0, config.vx0, config.vz0])
        
        # Results storage
        self.times = np.arange(0, config.t_final + config.dt, config.dt)
        self.results_euler = np.zeros((len(self.times), 4))
        self.results_rk4 = np.zeros((len(self.times), 4))
        
        # Store initial state
        self.results_euler[0] = self.state_euler
        self.results_rk4[0] = self.state_rk4

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate derivatives of state variables [x, z, vx, vz]
        
        This implements the corrected differential equations:
        - dx/dt = vx
        - dz/dt = vz
        - dvx/dt = -u/m * vx * |vx|  (air resistance proportional to v²)
        - dvz/dt = -g - u/m * vz * |vz|
        """
        x, z, vx, vz = state
        
        # Air resistance terms with correct sign handling
        resistance_x = -(self.config.air_resistance / self.config.mass) * vx * abs(vx)
        resistance_z = -(self.config.air_resistance / self.config.mass) * vz * abs(vz)
        
        # Derivatives
        dx_dt = vx
        dz_dt = vz
        dvx_dt = resistance_x
        dvz_dt = -self.config.gravity + resistance_z
        
        return np.array([dx_dt, dz_dt, dvx_dt, dvz_dt])

    def step_euler(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Forward Euler integration step"""
        derivatives = self.derivatives(state)
        return state + dt * derivatives

    def step_rk4(self, state: np.ndarray, dt: float) -> np.ndarray:
        """4th-order Runge-Kutta integration step"""
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + dt * k1 / 2)
        k3 = self.derivatives(state + dt * k2 / 2)
        k4 = self.derivatives(state + dt * k3)
        
        return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def run(self) -> Tuple[dict, dict]:
        """Run the simulation using both methods and return results"""
        # Iterate through time steps
        for i in range(1, len(self.times)):
            # Forward Euler
            if self.config.method in ["euler", "both"]:
                self.state_euler = self.step_euler(self.state_euler, self.config.dt)
                self.results_euler[i] = self.state_euler
            
            # Runge-Kutta 4
            if self.config.method in ["rk4", "both"]:
                self.state_rk4 = self.step_rk4(self.state_rk4, self.config.dt)
                self.results_rk4[i] = self.state_rk4
        
        # Prepare results
        euler_results = {
            'time': self.times,
            'x': self.results_euler[:, 0],
            'z': self.results_euler[:, 1],
            'vx': self.results_euler[:, 2],
            'vz': self.results_euler[:, 3]
        }
        
        rk4_results = {
            'time': self.times,
            'x': self.results_rk4[:, 0],
            'z': self.results_rk4[:, 1],
            'vx': self.results_rk4[:, 2],
            'vz': self.results_rk4[:, 3]
        }
        
        return euler_results, rk4_results

    def compare_precision(self, reference_dt: float = 0.0001) -> dict:
        """
        Compare precision of Euler vs RK4 by using a very small time step solution as reference
        
        Args:
            reference_dt: Very small time step for "exact" solution
            
        Returns:
            Dictionary with error metrics
        """
        # Create a reference simulation with very small time step
        ref_config = SimConfig(**vars(self.config))
        ref_config.dt = reference_dt
        ref_config.method = "rk4"  # Use RK4 for reference solution
        
        ref_sim = ProjectileSimulation(ref_config)
        _, ref_results = ref_sim.run()
        
        # Interpolate reference results to match the main simulation times
        ref_x = interp1d(ref_results['time'], ref_results['x'])(self.times)
        ref_z = interp1d(ref_results['time'], ref_results['z'])(self.times)
        
        # Calculate errors
        euler_x_error = np.mean(np.abs(self.results_euler[:, 0] - ref_x))
        euler_z_error = np.mean(np.abs(self.results_euler[:, 1] - ref_z))
        rk4_x_error = np.mean(np.abs(self.results_rk4[:, 0] - ref_x))
        rk4_z_error = np.mean(np.abs(self.results_rk4[:, 1] - ref_z))
        
        return {
            'euler_x_error': euler_x_error,
            'euler_z_error': euler_z_error,
            'rk4_x_error': rk4_x_error,
            'rk4_z_error': rk4_z_error,
            'euler_error_total': euler_x_error + euler_z_error,
            'rk4_error_total': rk4_x_error + rk4_z_error
        }


def plot_results(euler_results: dict, rk4_results: dict, config: SimConfig):
    """Create plots comparing the two methods"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory plot
    axs[0, 0].plot(euler_results['x'], euler_results['z'], 'b-', label='Euler')
    axs[0, 0].plot(rk4_results['x'], rk4_results['z'], 'r--', label='RK4')
    axs[0, 0].set_title('Projectile Trajectory')
    axs[0, 0].set_xlabel('x position (m)')
    axs[0, 0].set_ylabel('z position (m)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Position vs time
    axs[0, 1].plot(euler_results['time'], euler_results['x'], 'b-', label='x - Euler')
    axs[0, 1].plot(euler_results['time'], euler_results['z'], 'g-', label='z - Euler')
    axs[0, 1].plot(rk4_results['time'], rk4_results['x'], 'b--', label='x - RK4')
    axs[0, 1].plot(rk4_results['time'], rk4_results['z'], 'g--', label='z - RK4')
    axs[0, 1].set_title('Position vs Time')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Position (m)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Velocity vs time
    axs[1, 0].plot(euler_results['time'], euler_results['vx'], 'b-', label='vx - Euler')
    axs[1, 0].plot(euler_results['time'], euler_results['vz'], 'g-', label='vz - Euler')
    axs[1, 0].plot(rk4_results['time'], rk4_results['vx'], 'b--', label='vx - RK4')
    axs[1, 0].plot(rk4_results['time'], rk4_results['vz'], 'g--', label='vz - RK4')
    axs[1, 0].set_title('Velocity vs Time')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Velocity (m/s)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Error plot
    if config.method == "both":
        x_diff = np.abs(euler_results['x'] - rk4_results['x'])
        z_diff = np.abs(euler_results['z'] - rk4_results['z'])
        axs[1, 1].plot(euler_results['time'], x_diff, 'b-', label='x position difference')
        axs[1, 1].plot(euler_results['time'], z_diff, 'g-', label='z position difference')
        axs[1, 1].set_title('Difference Between Methods')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Absolute Difference (m)')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('../results/projectile_simulation.png')


def precision_study(base_config: SimConfig) -> dict:
    """Run simulations with different time steps to study numerical stability"""
    dt_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    results = []
    
    # Define very small time step for reference solution
    ref_config = SimConfig(**vars(base_config))
    ref_config.dt = 0.0001
    ref_config.method = "rk4"  # Use RK4 for reference
    
    ref_sim = ProjectileSimulation(ref_config)
    _, ref_results = ref_sim.run()
    
    # Test each time step
    for dt in dt_values:
        config = SimConfig(**vars(base_config))
        config.dt = dt
        config.method = "both"
        
        sim = ProjectileSimulation(config)
        euler_results, rk4_results = sim.run()
        
        # Measure impact point (when z returns to initial height)
        euler_impact_x = euler_results['x'][-1]  # Simplification - use last point
        rk4_impact_x = rk4_results['x'][-1]  # Simplification - use last point
        
        # Reference impact point
        ref_impact_x = ref_results['x'][-1]
        
        results.append({
            'dt': dt,
            'euler_impact_error': abs(euler_impact_x - ref_impact_x),
            'rk4_impact_error': abs(rk4_impact_x - ref_impact_x)
        })
    
    # Plot errors vs time step
    plt.figure(figsize=(10, 6))
    dt_array = np.array([r['dt'] for r in results])
    euler_errors = np.array([r['euler_impact_error'] for r in results])
    rk4_errors = np.array([r['rk4_impact_error'] for r in results])
    
    plt.loglog(dt_array, euler_errors, 'bo-', label='Euler')
    plt.loglog(dt_array, rk4_errors, 'ro-', label='RK4')
    
    # Add trend lines
    log_dt = np.log10(dt_array)
    log_euler = np.log10(euler_errors)
    log_rk4 = np.log10(rk4_errors)
    
    # Linear regression in log space gives order of convergence
    euler_slope, euler_intercept, _, _, _ = linregress(log_dt, log_euler)
    rk4_slope, rk4_intercept, _, _, _ = linregress(log_dt, log_rk4)
    
    plt.xlabel('Time step (s)')
    plt.ylabel('Impact point error (m)')
    plt.title('Error vs Time Step')
    plt.grid(True)
    plt.legend(title=f'Convergence rates:\nEuler: O(Δt^{euler_slope:.2f})\nRK4: O(Δt^{rk4_slope:.2f})')
    plt.savefig('convergence_study.png')
    plt.show()
    
    return {'results': results, 'euler_order': euler_slope, 'rk4_order': rk4_slope}


def load_config(config_file=None) -> SimConfig:
    """Load configuration from YAML file or use defaults"""
    config = SimConfig()
    
    if config_file:
        try:
            with open(config_file, 'r') as file:
                data = yaml.safe_load(file)
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration.")
    
    return config


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
    parser.add_argument('--output', type=str, default='', 
                        help='Base filename for saving output files')
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
    
    # Run precision study if requested
    if args.study:
        print("\nRunning precision study with different time steps...")
        results = precision_study(config)
        print(f"Euler method convergence order: O(Δt^{results['euler_order']:.2f})")
        print(f"RK4 method convergence order: O(Δt^{results['rk4_order']:.2f})")
        return
    
    # Run the simulation
    print("\nRunning simulation...")
    sim = ProjectileSimulation(config)
    euler_results, rk4_results = sim.run()
    
    # If comparing both methods, print the difference
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
    
    # Find landing position (where z crosses back to initial height)
    if config.method in ["euler", "both"]:
        landing_indices = np.where((euler_results['z'][1:] <= config.z0) & 
                                   (euler_results['z'][:-1] > config.z0))[0]
        if landing_indices.size > 0:
            landing_idx = landing_indices[0] + 1
            landing_time = euler_results['time'][landing_idx]
            landing_x = euler_results['x'][landing_idx]
            print(f"\nEuler method landing position:")
            print(f"Time: {landing_time:.2f} s")
            print(f"Distance: {landing_x:.2f} m")
    
    if config.method in ["rk4", "both"]:
        landing_indices = np.where((rk4_results['z'][1:] <= config.z0) & 
                                   (rk4_results['z'][:-1] > config.z0))[0]
        if landing_indices.size > 0:
            landing_idx = landing_indices[0] + 1
            landing_time = rk4_results['time'][landing_idx]
            landing_x = rk4_results['x'][landing_idx]
            print(f"\nRK4 method landing position:")
            print(f"Time: {landing_time:.2f} s")
            print(f"Distance: {landing_x:.2f} m")
    
    # Generate plots
    plot_results(euler_results, rk4_results, config)
    
    return euler_results, rk4_results


if __name__ == "__main__":
    main()
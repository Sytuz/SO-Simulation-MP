"""
Core simulation module for projectile motion.
"""
import numpy as np
from typing import Tuple
from scipy.interpolate import interp1d
from .config import SimConfig


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
        
        This implements the differential equations:
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
    
    def find_landing_position(self, results, initial_height=None) -> dict:
        """Find where the projectile lands (crosses back to initial height)"""
        if initial_height is None:
            initial_height = self.config.z0
            
        landing_indices = np.where((results['z'][1:] <= initial_height) & 
                                  (results['z'][:-1] > initial_height))[0]
        
        if landing_indices.size > 0:
            landing_idx = landing_indices[0] + 1
            landing_time = results['time'][landing_idx]
            landing_x = results['x'][landing_idx]
            
            return {
                'time': landing_time,
                'distance': landing_x,
                'found': True
            }
        
        return {'found': False}
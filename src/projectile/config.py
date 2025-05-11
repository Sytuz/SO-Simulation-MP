"""
Configuration module for projectile motion simulation.
"""
import yaml
from dataclasses import dataclass


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
    output_dir: str = "results/projectile"  # Output directory for results


def load_config(config_file=None):
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
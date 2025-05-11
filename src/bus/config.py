"""
Bus simulation configuration module.
"""
import yaml
from dataclasses import dataclass


@dataclass
class SimConfig:
    """Simulation configuration parameters"""
    simulation_time: float = 160.0
    arrival_mean: float = 2.0
    inspection_min: float = 0.25
    inspection_max: float = 1.05
    repair_probability: float = 0.3
    repair_min: float = 2.1
    repair_max: float = 4.5
    num_inspectors: int = 1
    num_repair_stations: int = 2
    seed: int = 42
    output_dir: str = "results/bus"  # Default output directory for results


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
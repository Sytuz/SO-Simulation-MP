import simpy
import random
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class SimConfig:
    """Simulation configuration parameters"""
    simulation_time: float = 160.0  # hours
    arrival_mean: float = 2.0  # hours (exponential distribution)
    inspection_min: float = 0.25  # hours (15 min)
    inspection_max: float = 1.05  # hours
    repair_probability: float = 0.3
    repair_min: float = 2.1  # hours
    repair_max: float = 4.5  # hours
    num_inspectors: int = 1
    num_repair_stations: int = 2
    seed: int = 42


class BusMaintenanceSimulation:
    """Bus maintenance facility discrete-event simulation"""
    
    def __init__(self, config):
        self.config = config
        self.env = simpy.Environment()
        
        # Resources
        self.inspector = simpy.Resource(self.env, capacity=config.num_inspectors)
        self.repair_stations = simpy.Resource(self.env, capacity=config.num_repair_stations)
        
        # Statistics tracking
        self.stats = {
            'inspection_queue_lengths': [],
            'repair_queue_lengths': [],
            'inspection_delays': [],
            'repair_delays': [],
            'inspection_utilization': [],
            'repair_utilization': []
        }
        
        # Set up bus counter and queue length tracking
        self.buses_processed = 0
        self.last_event_time = 0
        
        # Set random seed
        random.seed(config.seed)

    def run(self):
        """Start the simulation"""
        # Start the bus arrival process
        self.env.process(self.bus_arrival_process())
        
        # Set up statistics collection
        self.env.process(self.collect_statistics())
        
        # Run the simulation
        self.env.run(until=self.config.simulation_time)
        
        # Return the results
        return self.calculate_results()

    def bus_arrival_process(self):
        """Generate bus arrivals"""
        while True:
            # Create a new bus arrival
            self.buses_processed += 1
            bus_id = self.buses_processed
            
            # Start the bus processing
            self.env.process(self.bus_process(bus_id))
            
            # Wait for the next arrival (exponentially distributed)
            interarrival_time = random.expovariate(1.0 / self.config.arrival_mean)
            yield self.env.timeout(interarrival_time)
    
    def bus_process(self, bus_id):
        """Process a single bus through the maintenance facility"""
        # Bus arrives and waits for inspection
        arrival_time = self.env.now
        
        # Request an inspector
        with self.inspector.request() as request:
            # Wait until inspector is available (queue)
            yield request
            
            # Bus starts inspection
            inspection_start = self.env.now
            inspection_delay = inspection_start - arrival_time
            self.stats['inspection_delays'].append(inspection_delay)
            
            # Inspection takes time (uniformly distributed)
            inspection_time = random.uniform(
                self.config.inspection_min, 
                self.config.inspection_max
            )
            yield self.env.timeout(inspection_time)
            
            # End of inspection
            
        # Check if repair is needed (30% probability)
        if random.random() < self.config.repair_probability:
            # Bus needs repair
            repair_queue_entry = self.env.now
            
            # Request a repair station
            with self.repair_stations.request() as request:
                # Wait until repair station is available
                yield request
                
                # Bus starts repair
                repair_start = self.env.now
                repair_delay = repair_start - repair_queue_entry
                self.stats['repair_delays'].append(repair_delay)
                
                # Repair takes time (uniformly distributed)
                repair_time = random.uniform(
                    self.config.repair_min, 
                    self.config.repair_max
                )
                yield self.env.timeout(repair_time)
                
                # End of repair
    
    def collect_statistics(self):
        """Collect statistics about queues and resource utilization"""
        while True:
            # Record queue lengths
            self.stats['inspection_queue_lengths'].append(len(self.inspector.queue))
            self.stats['repair_queue_lengths'].append(len(self.repair_stations.queue))
            
            # Record resource utilization
            self.stats['inspection_utilization'].append(self.inspector.count / self.config.num_inspectors)
            self.stats['repair_utilization'].append(self.repair_stations.count / self.config.num_repair_stations)
            
            # Wait before next collection
            yield self.env.timeout(0.1)  # Sample every 0.1 simulation hours

    def calculate_results(self):
        """Calculate final simulation results"""
        results = {
            'avg_inspection_queue_length': np.mean(self.stats['inspection_queue_lengths']),
            'avg_repair_queue_length': np.mean(self.stats['repair_queue_lengths']),
            'avg_inspection_delay': np.mean(self.stats['inspection_delays']) if self.stats['inspection_delays'] else 0,
            'avg_repair_delay': np.mean(self.stats['repair_delays']) if self.stats['repair_delays'] else 0,
            'inspection_utilization': np.mean(self.stats['inspection_utilization']),
            'repair_utilization': np.mean(self.stats['repair_utilization']),
            'buses_processed': self.buses_processed
        }
        return results


def plot_results(stats, title="Bus Maintenance Simulation Results"):
    """Generate plots for the simulation results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Queue lengths over time
    ax1.plot(stats['inspection_queue_lengths'], label='Inspection Queue')
    ax1.plot(stats['repair_queue_lengths'], label='Repair Queue')
    ax1.set_title('Queue Lengths Over Time')
    ax1.set_xlabel('Time (samples every 0.1h)')
    ax1.set_ylabel('Queue Length')
    ax1.legend()
    
    # Utilization over time
    ax2.plot(stats['inspection_utilization'], label='Inspector')
    ax2.plot(stats['repair_utilization'], label='Repair Stations')
    ax2.set_title('Resource Utilization Over Time')
    ax2.set_xlabel('Time (samples every 0.1h)')
    ax2.set_ylabel('Utilization')
    ax2.legend()
    
    # Delay histograms
    if stats['inspection_delays']:
        ax3.hist(stats['inspection_delays'], bins=20, alpha=0.7)
        ax3.set_title('Inspection Delays')
        ax3.set_xlabel('Delay (hours)')
        ax3.set_ylabel('Count')
    
    if stats['repair_delays']:
        ax4.hist(stats['repair_delays'], bins=20, alpha=0.7)
        ax4.set_title('Repair Delays')
        ax4.set_xlabel('Delay (hours)')
        ax4.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('../results/bus_simulation_results.png')


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
    
    # Find the maximum stable arrival rate
    # (where queue lengths and delays don't grow too large)
    for result in results:
        is_stable = (result['avg_inspection_queue_length'] < 10 and 
                    result['avg_repair_queue_length'] < 10)
        print(f"Rate: {result['arrival_rate']:.3f} buses/hour - "
              f"Stable: {is_stable} - "
              f"Insp. util: {result['inspection_utilization']:.2f} - "
              f"Repair util: {result['repair_utilization']:.2f} - "
              f"Insp. queue: {result['avg_inspection_queue_length']:.2f} - "
              f"Repair queue: {result['avg_repair_queue_length']:.2f}")
    
    return results


def main():
    """Main function to parse arguments and run the simulation"""
    parser = argparse.ArgumentParser(description='Bus Maintenance Facility Simulation')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--time', type=float, help='Simulation time in hours')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--experiment', action='store_true', help='Run arrival rate experiment')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.time:
        config.simulation_time = args.time
    if args.seed:
        config.seed = args.seed
    
    if args.experiment:
        results = run_arrival_rate_experiment(config)
    else:
        # Run simulation with the specified configuration
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
        
        # Generate plots
        plot_results(sim.stats)
    
    return results


if __name__ == "__main__":
    main()
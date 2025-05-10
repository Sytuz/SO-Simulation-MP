"""
Core bus maintenance facility simulation model.
"""
import simpy
import random
import numpy as np


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
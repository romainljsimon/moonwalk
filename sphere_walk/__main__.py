import toml
import numpy as np
import os
import sphere_walk.brownian_motion as pw 
import sphere_walk.z_functions as zf
import pyarrow as pa
import pyarrow.parquet as pq
import struct
import sphere_walk.random_walk as rw

# Load configuration from the TOML file
config_file = 'config.toml'
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file '{config_file}' not found")

# Load configuration parameters from the TOML file
config = toml.load(config_file)

# Extract required variables, raising an error if any are missing
required_keys = ['n_walkers', 'walls', 'type']
for key in required_keys:
    if key not in config:
        raise KeyError(f"Required configuration key '{key}' is missing from the TOML file")

# Initialize main simulation parameters
n_walkers = config['n_walkers']
walls = config['walls']
simu_type = config['type']

# Define settings based on the type of simulation
if simu_type == 'brownian':
    # For Brownian motion: load simulation-specific values from configuration
    tmax = config['tmax']
    dt = config['dt']
    
    # Compute number of iterations based on total time and time step
    n_iter = int(tmax / dt)
    n_position = 2  # Position vector length for 2D space

    # If walls are enabled, require parameter 'a' from configuration
    if walls: 
        if 'a' not in config:
            raise KeyError(f"Required configuration key '{key}' is missing from the TOML file. Necessary because walls is True")
        else:
            a = config['a']
    else:
        a = 0

elif simu_type == 'walk':
    # For general random walk: use specific configuration values
    n_position = 9      # Position vector length in 3D space
    n_iter = config['n_iter']
    theta = config['theta']
    dt = 1              # Fixed time step

# Configure observables from the 'observables' list in config, defaulting to 'normal'
observables_str = config.get('observables', ['normal'])
column_names = ["walker"] + observables_str  # Column names for results table
observables_zf = [zf.ZFunctions(elt) for elt in observables_str]  # Create ZFunctions for each observable

# Initialize arrays for storing time and walker data
t_array = np.arange(n_iter)       # Array of time points
walker_array = []                 # Array to store walker objects

# Instantiate walker objects for the chosen simulation type
for walk in np.arange(n_walkers):
    if simu_type == 'brownian':
        # Append PointWalk object for each walker if simulation is Brownian
        walker_array.append(pw.PointWalk(dt=dt, walls=walls, a=a, observables_class=observables_zf))
    
    elif simu_type == 'walk':
        # Append XYZWalk object for each walker if simulation is general walk
        walker_array.append(rw.XYZWalk(theta=theta, walls=walls))

# Initialize arrays to store observables and walker positions
all_observables = np.zeros((n_walkers, len(observables_str)))
position_walkers = np.zeros((n_walkers, n_position))

# Populate initial values for walker positions and observables
for i, walker in enumerate(walker_array):
    position_walkers[i] = walker.position     # Initial positions
    all_observables[i] = walker.observables   # Initial observables

# Define log-based or linear time intervals for storing data snapshots
time_logarray = np.arange(0, n_iter)
time_logarray = np.array([0] + [int(elt) for elt in time_logarray] + [n_iter-1])

# Main simulation loop: iterates over each time step
for j, elt in enumerate(t_array):

    if j in time_logarray:  # Check if the current time step is in the log array
        # Convert positions and observables to Apache Arrow arrays
        position_arrays = [pa.array(position_walkers[:, i]) for i in range(position_walkers.shape[1])]
        observable_arrays = [pa.array(all_observables[:, i]) for i in range(all_observables.shape[1])]
        
        # Combine position and observable arrays into one array for table creation
        total_arrays = position_arrays + observable_arrays
        total_table = pa.table(total_arrays, names=walker_array[0].column_names)
        
        # Create and store custom metadata for each time snapshot, with the current time
        custom_metadata = {'time': struct.pack('f', j * dt)}
        existing_metadata = total_table.schema.metadata
        merged_metadata = {**custom_metadata, **existing_metadata} if existing_metadata else custom_metadata
        total_table = total_table.replace_schema_metadata(merged_metadata)
        
        # Write the data to a Parquet file for the current time step
        pq.write_table(total_table, f'out/out{j}.parquet')

    # Update positions and observables for each walker in the simulation
    for i, walker in enumerate(walker_array):
        walker.walk()                          # Execute walk step
        position_walkers[i] = walker.position   # Update position array
        all_observables[i] = walker.observables # Update observables array

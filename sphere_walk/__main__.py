import toml
import numpy as np
import os
from sphere_walk import *
import pyarrow as pa
import pyarrow.parquet as pq
import struct
import shutil


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
n_walkers = int(config['n_walkers'])
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
            raise KeyError(f"Required configuration key 'a' is missing from the TOML file. Necessary because walls is True")
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

else:
    raise ValueError(f"Configuration key 'simu_type' cannot accept value {simu_type}'. Good values are 'brownian' and 'walk'")

# Configure observables from the 'observables' list in config, defaulting to 'normal'
observables = config.get('observables', [])

cwd = os.getcwd()
folder_path = os.path.join(cwd, "out")
# Check if the out folder exists, and if not, create it in cwd
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)

os.makedirs(folder_path)

num = config.get('num', 50)
    

# Initialize arrays for storing time and walker data
t_array = np.arange(n_iter)       # Array of time points
walker_array = []                 # Array to store walker objects

# Instantiate walker objects for the chosen simulation type
for walk in np.arange(n_walkers):
    if simu_type == 'brownian':
        time_logarray = np.logspace(np.log10(dt), np.log10(tmax), num=num)
        time_logarray = np.round(time_logarray / dt) * dt
        time_logarray = np.concatenate(([0], np.unique(time_logarray)))
        # Append PointWalk object for each walker if simulation is Brownian
        walker_array.append(PointWalk(dt=dt, tmax=tmax, walls=walls, a=a, observables=observables, time_logarray=time_logarray))
    
    elif simu_type == 'walk':
        # Append XYZWalk object for each walker if simulation is general walk
        walker_array.append(XYZWalk(theta=theta, walls=walls))
        


# Define log-based or linear time intervals for storing data snapshots
#time_logarray = np.arange(0, n_iter)
log_spaced_values = np.logspace(0, np.log10(n_iter), num=20, base=10)
log_spaced_ints = np.unique(np.round(log_spaced_values).astype(int))
log_spaced_ints = log_spaced_ints[log_spaced_ints < n_iter]
#time_logarray = np.array([0] + [int(elt) for elt in log_spaced_ints])
# Main simulation loop: iterates over each time 

print(f'Starting simulation ...')
if simu_type == "walk":
    # Initialize arrays to store observables and walker positions
    all_observables = np.zeros((n_walkers, len(observables)))
    position_walkers = np.zeros((n_walkers, n_position))

    # Populate initial values for walker positions and observables
    for i, walker in enumerate(walker_array):
        position_walkers[i] = walker.position     # Initial positions
        all_observables[i] = walker.observables   # Initial observables
    for j, _ in enumerate(t_array):

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

elif simu_type == 'brownian':

    # Initialize arrays to store observables and walker positions
    all  = np.zeros((n_walkers, n_position + len(observables), len(time_logarray)))
    # Initial observables
    time_array = []
    for i, walker in enumerate(walker_array):
        print(f"walker {i}")
        walker.total_walk()
        try:
            all[i, :, :] = walker.total_x.T
            time_array = walker.time_array
        except:
            all[i, :, :len(walker.total_x)] = walker.total_x.T
            all[i, :, len(walker.total_x):] = np.nan
            if len(walker.time_array) > len(time_array):
                time_array = walker.time_array
    
    column_names = walker.column_names
    for j, time in enumerate(time_array):
        data = all[:, :, j]
        write_data(time, j, data, column_names)
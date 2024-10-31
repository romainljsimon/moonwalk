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

config = toml.load(config_file)

# Extract required variables, and raise an error if any are missing
required_keys = ['n_walkers', 'walls', 'type']
for key in required_keys:
    if key not in config:
        raise KeyError(f"Required configuration key '{key}' is missing from the TOML file")

n_walkers = config['n_walkers']
walls = config['walls']
simu_type = config['type']
if simu_type == 'brownian':
    # Get the values from the TOML configuration
    tmax = config['tmax']
    dt = config['dt']
    
    n_iter = int(tmax / dt)
    n_position = 2
    if walls: 
        if 'a' not in config:
            raise KeyError(f"Required configuration key '{key}' is missing from the TOML file. Necessary because walls is True")

        else:
            a = config['a']

    else:
        a = 0

elif simu_type == 'walk':
    n_position = 9
    n_iter = config['n_iter']
    theta = config['theta']
    dt = 1



# Handle observables
observables_str = config.get('observables', ['normal'])  # Default to ['normal'] if not in config
column_names = ["walker"] + observables_str
observables_zf = [zf.ZFunctions(elt) for elt in observables_str]

# Start calculations

t_array = np.arange(n_iter)


walker_array = []
for walk in np.arange(n_walkers):
    if simu_type == 'brownian':
        walker_array.append(pw.PointWalk(dt=dt, walls=walls, a=a, observables_class=observables_zf))
    
    elif simu_type== 'walk':
        walker_array.append(rw.XYZWalk(theta=theta, walls=walls))

all_observables = np.zeros((n_walkers, len(observables_str)))
position_walkers = np.zeros((n_walkers, n_position))

for i, walker in enumerate(walker_array):
    position_walkers[i] = walker.position
    all_observables[i] = walker.observables


#time_logarray = np.logspace(0, np.log10(n_iter), num=1000, endpoint=True)
time_logarray = np.arange(0, n_iter)
time_logarray = np.array([0] + [int(elt) for elt in time_logarray] +[n_iter-1])
for j, elt in enumerate(t_array):

    if j in time_logarray:
        
        position_arrays =[pa.array(position_walkers[:, i]) for i in range(position_walkers.shape[1])]
        observable_arrays = [pa.array(all_observables[:, i]) for i in range(all_observables.shape[1])]
        total_arrays = position_arrays + observable_arrays
        
        total_table = pa.table(total_arrays, names=walker_array[0].column_names)
        custom_metadata = {'time': struct.pack('f', j*dt)}
        existing_metadata = total_table.schema.metadata
        
        if existing_metadata is not None:
            merged_metadata = { **custom_metadata, **existing_metadata}
        else:
            merged_metadata = custom_metadata
        total_table = total_table.replace_schema_metadata(merged_metadata)
        pq.write_table(total_table, f'out/out{j}.parquet')

    for i, walker in enumerate(walker_array):
        walker.walk()
        position_walkers[i] = walker.position
        all_observables[i] = walker.observables
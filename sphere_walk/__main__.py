import toml
import numpy as np
import os
import sphere_walk.point_walk as pw 
import sphere_walk.z_functions as zf
import pyarrow as pa
import pyarrow.parquet as pq

# Load configuration from the TOML file
config_file = 'config.toml'
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file '{config_file}' not found")

config = toml.load(config_file)

# Extract required variables, and raise an error if any are missing
required_keys = ['tmax', 'dt', 'n_walkers', 'walls']
for key in required_keys:
    if key not in config:
        raise KeyError(f"Required configuration key '{key}' is missing from the TOML file")

# Get the values from the TOML configuration
tmax = config['tmax']
dt = config['dt']
n_walkers = config['n_walkers']
walls = config['walls']

if walls: 
    if 'a' not in config:
        raise KeyError(f"Required configuration key '{key}' is missing from the TOML file. Necessary because walls is True")

    else:
        a = config['a']

else:
    a = 0

# Handle observables
observables_str = config.get('observables', ['normal'])  # Default to ['normal'] if not in config
column_names = ["time"] + observables_str
observables_zf = [zf.ZFunctions(elt) for elt in observables_str]

# Start calculations
n_iter = int(tmax / dt)
t_array = np.arange(n_iter)
all_observables = np.zeros((n_walkers, len(observables_str) + 1))
all_observables[:, 0] = np.arange(n_walkers)

walker_array = []
for walk in np.arange(n_walkers):
    walker_array.append(pw.PointWalk(dt=dt, walls=walls, a=a, observables_class=observables_zf))

time_logarray = np.logspace(0, np.log10(n_iter), num=50, endpoint=True)
time_logarray = np.array([0] + [int(elt) for elt in time_logarray] +[n_iter-1])

for j, elt in enumerate(t_array):
    for i, walker in enumerate(walker_array):
        walker.random_walk()
        all_observables[i, 1:] = walker.observables
    if j in time_logarray:
        observable_arrays = [pa.array(all_observables[:, i]) for i in range(all_observables.shape[1])]
        observable_table = pa.Table.from_arrays(observable_arrays, names=column_names) 
        pq.write_table(observable_table, f'out/out{j}.parquet')



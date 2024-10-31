# Random Walk and Brownian Motion Simulation on the Sphere

This project provides a simulation framework for modeling various types of random walks, including Brownian motion of a point on the sphere and the random walk of a coordinate system on the sphre. Additional configurable parameters such as walls and different observables are implemented. The simulation supports dynamic configurations via a TOML file, allowing users to easily control the simulation parameters and output settings.
DO NOT USE BROWNIAN YET. THIS WILL BE CHANGED SOON.

## Download the project

Clone the git repo:
git clone https://github.com/romainljsimon/sphere_walk

Then install it with pip in the directory where you cloned the repo. Thus you will be able to launch the program in every directory of your computer.
 ```bash
pip install -e .
```

## Requirements

- **Python 3.x**
- **NumPy**: For numerical computations
- **Numba**: For JIT compilation to optimize performance
- **PyArrow**: For handling Parquet file storage
- **TOML**: To load configuration from a `config.toml` file

Install dependencies with:

```bash
pip install numpy numba pyarrow toml
```

## Project Structure

- **`z_functions.py`**: Defines mathematical functions used as observables in the simulation, including trigonometric, polynomial, and step functions. These observables track and analyze specific properties of the walker's position at each time step.
  
- **`brownian_motion.py`**: Implements a `PointWalk` class for simulating 2D Brownian motion with optional wall constraints. The walker’s movement and boundary interactions are determined based on user-specified parameters.

- **`random_walk.py`**: Contains the `XYZWalk` class, designed for 3D random walks with customizable rotation and boundary configurations. The random walk can be configured for unrestricted or bounded movement within a specified space.


## Configuration

The config.toml file specifies parameters for the simulation. Required fields include:

- **n_walkers**: Number of walkers to simulate.
- **walls**: Boolean flag to enable boundary constraints.
- **type**: Simulation type, either "brownian" or "walk".

*Additional fields depend on the chosen simulation type*:

### For brownian:
- **tmax**: Maximum time for simulation.
- **dt**: Time step increment.
- **a**: Boundary constraint parameter (if walls is enabled).

### For walk:
- **n_iter**: Number of iterations.
- **theta**: Angle for rotation in each step.

## Usage
To run the simulation, ensure you have a properly configured config.toml file in the working directory. Then, execute:

```bash
python -m sphere_walk
```

The simulation performs a specified number of iterations for each walker, storing data snapshots as Parquet files in the out/ directory.

## Output

The output files are stored in Parquet format in the out/ folder. Each file corresponds to a specific time step and includes the positions and observable values for all walkers.
Metadata, such as the current time, is stored in each Parquet file for easy post-processing and analysis.

## Example Configuration (config.toml)

```bash
n_walkers = 100
walls = true
type = "random"
n_iter = 10000
theta = 0.2
```

## License
This project is licensed under the MIT License.
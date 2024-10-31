import numpy as np
from numba import float64
from numba.experimental import jitclass

# Define the specification for JIT compilation with numba
# Used for performance optimization by defining types of attributes
spec = [
    ('position', float64[:]),           # 1D array for position coordinates
    ('position_array', float64[:, :, :]), # 3D array for a sequence of positions
    ('dt', float64)                     # Time increment for the walk
]

# Uncomment @jitclass(spec) to optimize with numba
#@jitclass(spec)
class PointWalk:
    """
    Class representing a point-like random walker on the 2-sphere. Its position is given
    by the longitude lambda and latitude phi.
    
    Attributes:
    -----------
    dt : float
        Time increment for each step in the walk.
    walls : bool
        If True, applies boundary conditions (limits the walker).
    a : float
        Factor used in boundary-conditioned position calculations.
    observables_class : list
        List of observables (functions) to apply to the walker at each step.
    observables : numpy.ndarray
        Array storing computed values of each observable.
    position : numpy.ndarray
        2D position vector for the walker.
    column_names : list
        Names for components of the position vector.
    """

    def __init__(self, dt: float = 0.1, walls: bool = False, a: float = 100, observables_class: list = []):
        """
        Initializes the PointWalk instance with parameters.
        
        Args:
        ----
        dt : float
            Time increment. Default is 0.1.
        walls : bool
            Enables boundary constraints if True. Default is False.
        a : float
            Scaling factor for boundary-conditioned calculations. Default is 100.
        observables_class : list
            Functions to apply for tracking observables during the walk.
        """
        self.dt = dt
        self.walls = walls
        self.a = a
        self.observables_class = observables_class
        self.observables = np.zeros(len(observables_class))  # Initialize observables array
        self.position = np.zeros(2)  # Initialize position in 2D space
        self.column_names = ['lambda', 'phi']  # Names for position components

    def walk(self):
        """
        Executes a single step of the random walk.
        
        Computes the new position based on whether walls are enabled.
        Updates observables by applying each observable function to the position.
        """
        # Choose the derivation method based on wall constraint
        deriv = self.deriv_position_walls() if self.walls else self.deriv_position()
        
        # Update each observable with the current position and derivative
        for j, zf in enumerate(self.observables_class):
            self.observables[j] += zf.foo(self.position, deriv)  # Increment observable values
            
        # Update position with derived displacement
        self.position[:2] += deriv

    def deriv_position(self):
        """
        Computes the position displacement without boundary constraints.
        
        Returns:
        -------
        numpy.ndarray : 2D displacement vector [lambda, phi].
        """
        lam = 1 / np.cos(self.position[1]) * np.random.normal(scale=np.sqrt(self.dt))
        phi = -0.5 * np.tan(self.position[1]) * self.dt + np.random.normal(scale=np.sqrt(self.dt))
        return np.array([lam, phi])

    def deriv_position_walls(self):
        """
        Computes the position displacement with boundary constraints.
        
        Returns:
        -------
        numpy.ndarray : 2D displacement vector [lambda, phi] considering wall constraints.
        """
        lam = (1 / np.cos(self.position[1]) * np.random.normal(scale=np.sqrt(self.dt)) 
               - self.a * self.position[0] * self.dt)
        phi = (-0.5 * np.tan(self.position[1]) * self.dt 
               + np.random.normal(scale=np.sqrt(self.dt)) 
               - (2 * self.a - 1) / 2 * self.position[1] * self.dt)
        return np.array([lam, phi])


# Redefine spec for the LineWalk class for JIT optimization
spec = [
    ('position', float64[:]),
    ('position_array', float64[:, :, :]),
    ('dt', float64)
]

# Uncomment @jitclass(spec) for numba optimization
# STILL UNDER CONSTRUCTION DO NOT USE
#@jitclass(spec)
class LineWalk:
    """
    Class representing a random walker in 3D space along a line.
    
    Attributes:
    -----------
    dt : float
        Time increment for each step in the walk.
    walls : bool
        If True, applies boundary conditions to constrain the walker.
    a : float
        Scaling factor for boundary-conditioned calculations.
    observables_class : list
        List of observable functions to apply during the walk.
    observables : numpy.ndarray
        Array to store computed values of each observable.
    position : numpy.ndarray
        3D position vector for the walker.
    """

    def __init__(self, dt: float = 0.1, walls: bool = False, a: float = 100, observables_class: list = []):
        """
        Initializes the LineWalk instance with given parameters.
        
        Args:
        ----
        dt : float
            Time increment. Default is 0.1.
        walls : bool
            Enables boundary constraints if True. Default is False.
        a : float
            Scaling factor for boundary-conditioned calculations. Default is 100.
        observables_class : list
            Functions to track observables during the walk.
        """
        self.dt = dt
        self.walls = walls
        self.a = a
        self.observables_class = observables_class
        self.observables = np.zeros(len(observables_class))  # Initialize observables array
        self.position = np.zeros(3)  # Initialize position in 3D space

    def random_walk(self):
        """
        Executes a single step of the random walk.
        
        Computes displacement based on wall constraints and updates observables.
        """
        # Select displacement calculation based on wall constraint
        deriv = self.deriv_position_walls() if self.walls else self.deriv_position()
        
        # Update each observable with the current position and derivative
        for j, zf in enumerate(self.observables_class):
            self.observables[j] += zf.foo(self.position, deriv)  # Increment observable values
        
        # Update position with derived displacement
        self.position[:2] += deriv

    def deriv_position(self):
        """
        Computes the position displacement in 3D without boundary constraints.
        
        Returns:
        -------
        numpy.ndarray : 3D displacement vector [dlam, dphi, theta].
        """
        dlam = 1 / np.cos(self.position[1]) * np.random.normal(scale=np.sqrt(self.dt))
        dphi = -0.5 * np.tan(self.position[1]) * self.dt + np.random.normal(scale=np.sqrt(self.dt))
        theta = np.random.normal(scale=np.sqrt(self.dt))
        return np.array([dlam, dphi, theta])

    def deriv_position_walls(self):
        """
        Computes the position displacement in 3D with boundary constraints.
        
        Returns:
        -------
        numpy.ndarray : 3D displacement vector [lambda, phi, theta] considering wall constraints.
        """
        lam = (1 / np.cos(self.position[1]) * np.random.normal(scale=np.sqrt(self.dt)) 
               - self.a * self.position[0] * self.dt)
        phi = (-0.5 * np.tan(self.position[1]) * self.dt 
               + np.random.normal(scale=np.sqrt(self.dt)) 
               - (2 * self.a - 1) / 2 * self.position[1] * self.dt)
        theta = np.random.normal(scale=np.sqrt(self.dt)) - self.a * self.position[2] * self.dt
        return np.array([lam, phi, theta])
import numpy as np
from numba import float64
from numba.experimental import jitclass
from .z_functions import ZFunctions
import sdeint
from julia import DifferentialEquations as DE
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

    def __init__(self, dt: float = 0.1, tmax : float= 10, walls: bool = False, a: float = 100, observables: list = [], time_logarray: list = []):
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
        self.tmax = tmax
        self.walls = walls
        self.a = a
        self.observables = np.zeros(len(observables))  # Initialize observables array
        self.position = np.zeros(2)  # Initialize position in 2D space
        self.column_names = ['lambda', 'phi'] + observables # Names for position components
        self.zf_observables = [ZFunctions(elt) for elt in observables]  # Create ZFunctions for each observable
        self.time_logarray = time_logarray


    def total_walk(self):
        """
        Executes a single step of the random walk.
        
        Computes the new position based on whether walls are enabled.
        Updates observables by applying each observable function to the position.
        """
        # Choose the derivation method based on wall constraint
        #deriv = self.deriv_position_walls() if self.walls else self.deriv_position()
        tspan = (0, self.tmax)
        total_len = len(self.position) + len(self.observables)
        x0 = np.zeros(len(self.position) + len(self.observables))
        x0[1] = np.pi / 2
        noise_array = np.ones((total_len, 2))
        noise_array[0, 0], noise_array[1, 1]= 0, 0
        if len(self.observables) !=0:
            noise_array[2:, 1] = 0
        
        if self.walls:
            jp = np.zeros((total_len, total_len))
            jp[0, 0], jp[1, 1] = 1, 1
            jp[2:, :2] = 1
            ff = DE.SDEFunction(self.calc_walls_ito_f, self.calc_ito_g)#, jac=self.calc_jac_walls_f, jac_prototype=np.diag(np.full(total_len, 1)))
            prob = DE.SDEProblem(ff, x0, tspan, noise_rate_prototype=noise_array)
            
        else:
            jp = np.zeros((total_len, total_len))
            jp[1, 1] = 1
            ff = DE.SDEFunction(self.calc_ito_f, self.calc_ito_g)#, jac=self.calc_jac_f, jac_prototype=np.diag(np.full(total_len, 1)))
            prob = DE.SDEProblem(ff, x0, tspan, noise_rate_prototype=noise_array)
        sol = DE.solve(prob, DE.EM(), dt=self.dt, saveat=self.time_logarray)
        self.total_x = np.array([np.array(elt) for elt in sol.u])
        self.time_array = np.array(sol.t)

    def calc_ito_f(self, dx, x, p, t):
        f_lambda = 0
        f_phi = 0.5 / np.tan(x[1])
        dx[0:2] = np.array([f_lambda, f_phi])
        for i, func in enumerate(self.zf_observables):
            zf = func.foo(x)
            dx[2+i] = zf[0]*f_lambda + zf[1]*f_phi
        return np.array(dx)
    
    def calc_walls_ito_f(self, dx, x, p, t):
        f_lambda = -self.a * x[0]
        f_phi = 0.5 / np.tan(x[1]) - (2 * self.a - 1) / 2 * (x[1] - np.pi/2)
        dx[0:2] = np.array([f_lambda, f_phi])
        for i, func in enumerate(self.zf_observables):
            zf = func.foo(x)
            dx[2+i] = zf[0]*f_lambda + zf[1]*f_phi
        return np.array(dx)

    def calc_ito_g(self, dx, x, p, t):
        dx[0, 0] = 1 / np.sin(x[1])
        dx[1, 1] = 1
        for i, func in enumerate(self.zf_observables):
            zf = func.foo(x)
            dx[2+i, 0:2] = zf[0]*dx[0,0], zf[1]*dx[1, 1]

        return np.array(dx)
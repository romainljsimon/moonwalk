import numpy as np
from numba import float64
from numba.experimental import jitclass

# Define the specification for JIT compilation with numba (optional optimization)
# This spec is currently unused within the class but could optimize memory for defined fields.
spec = [
    ('angle_array', float64[:]),  # Placeholder for a scalar array field
]

# Uncomment @jitclass(spec) if you want to compile the class with numba for optimization
#@jitclass(spec)
class ZFunctions:
    """
    ZFunctions is a class that defines different mathematical functions for use in physical simulations,
    such as random walk observables. It provides a selection of polynomial, trigonometric, and step functions.
    """

    # Class-level dictionaries to store function names and their legends
    # These dictionaries are shared by all instances of the class
    z_dict_func = {
        'cos': 'cos',
        'poly2': 'poly2',
        'poly4': 'poly4',
        'poly8': 'poly8',
        'poly16': 'poly16',
        'door': 'door'
    }

    z_dict_deriv_func = {
        'cos': 'deriv_cos',
        'poly2': 'poly2',
        'poly4': 'poly4',
        'poly8': 'poly8',
        'poly16': 'poly16',
        'door': 'door'
    }

    def __init__(self, name: str = 'cos'):
        """
        Initializes the ZFunctions class with a specific function name.

        Args:
            name (str): The name of the function to use (e.g., 'cos', 'poly2'). Defaults to 'cos'.
        """
        self.name = name

        # Validate that the function name exists in the dictionary
        if name not in ZFunctions.z_dict_func:
            raise ValueError(f"Invalid function name: {name}. Available functions: {list(ZFunctions.z_dict_func.keys())}")
        
        # Retrieve the function and legend for the given name
        self.foo = getattr(self, ZFunctions.z_dict_func[name])  # Function assigned based on name
        self.deriv_foo = getattr(self, ZFunctions.z_dict_deriv_func[name])  # Function assigned based on name

    # Function implementations for each type of transformation

    def cos(self, position):
        """
        Cosine function that scales the first derivative by cos(point[1]).
        Args:
            point (np.array): Point in space.
            deriv (np.array): Derivative at the point.
        Returns:
            float: Cosine-scaled first derivative.
        """
        return [np.sin(position[1]), 0]
    
    def deriv_cos(self, position, walls):
        if walls:
            return np.array([np.cos(position[1]), -position[0]*np.sin(position[1])])
        else:
            return np.array([0, 0])

    def poly(self, position, exp):
        """
        General polynomial function that scales the derivative by (1 - (2/pi) * point[1]^exp).
        Args:
            point (np.array): Point in space.
            deriv (np.array): Derivative at the point.
            exp (int): Exponent for the polynomial.
        Returns:
            float: Polynomial-scaled derivative.
        """
        return [1 - (2 / np.pi * position[1]) ** exp, 0]
    
    # Specific polynomial functions using the poly method with different exponents

    def poly2(self, position):
        """
        Second-degree polynomial function that calls poly() with exp=2.
        """
        return self.poly(position, 2)

    def poly4(self, position):
        """ 
        Second-degree polynomial function that calls poly() with exp=4.
        """
        return self.poly(position, 4)

    def poly8(self, position):
        """
        Second-degree polynomial function that calls poly() with exp=8.
        """
        return self.poly(position, 8)
    
    def poly16(self, position):
        """
        Sixteenth-degree polynomial function that calls poly() with exp=16.
        """
        return self.poly(position, 16)

    def door(self, position):
        """
        Door or rectangular function, which returns the normal function if position[1] is below a threshold
        (pi/4), and zero otherwise.
        Args:
            point (np.array): Point in space.
            deriv (np.array): Derivative at the point.
        Returns:
            float: Output of the door function.
        """
        if np.abs(position[1]-np.pi/2) <  np.pi / 8:
            return [1, 0]
        else:
            return self.cos(position)
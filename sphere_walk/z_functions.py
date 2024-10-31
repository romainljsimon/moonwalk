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
        'normal': 'normal',
        'cos': 'cos',
        'poly2': 'poly2',
        'poly4': 'poly4',
        'poly8': 'poly8',
        'poly16': 'poly16',
        'heaviside': 'heaviside'
    }

    z_dict_legend = {
        'normal': r'$\langle \lambda^2 \rangle$',  # Legend label for the 'normal' function
        'cos': r'$cos$',                           # Legend label for the 'cos' function
        'poly2': r'$poly2$',                       # Legend label for the second-degree polynomial
        'poly4': r'$poly4$',                       # Legend label for the fourth-degree polynomial
        'poly8': r'$poly8$',                       # Legend label for the eighth-degree polynomial
        'poly16': r'$poly16$',                     # Legend label for the sixteenth-degree polynomial
        'heaviside': r'$heaviside$'                # Legend label for the Heaviside step function
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
        self.legend = ZFunctions.z_dict_legend[name]            # Legend text for labeling plots

    # Function implementations for each type of transformation

    def normal(self, point, deriv):
        """
        Basic function that returns the first derivative component.
        Args:
            point (np.array): Point in space, containing spatial coordinates.
            deriv (np.array): Derivative at the point.
        Returns:
            float: First component of the derivative.
        """
        return deriv[0]

    def cos(self, point, deriv):
        """
        Cosine function that scales the first derivative by cos(point[1]).
        Args:
            point (np.array): Point in space.
            deriv (np.array): Derivative at the point.
        Returns:
            float: Cosine-scaled first derivative.
        """
        return np.cos(point[1]) * deriv[0]

    def poly(self, point, deriv, exp):
        """
        General polynomial function that scales the derivative by (1 - (2/pi) * point[1]^exp).
        Args:
            point (np.array): Point in space.
            deriv (np.array): Derivative at the point.
            exp (int): Exponent for the polynomial.
        Returns:
            float: Polynomial-scaled derivative.
        """
        return (1 - 2 / np.pi * point[1] ** exp) * deriv[0]
    
    # Specific polynomial functions using the poly method with different exponents

    def poly2(self, point, deriv):
        """
        Second-degree polynomial function that calls poly() with exp=2.
        """
        return self.poly(point, deriv, 2)

    def poly4(self, point, deriv):
        """
        Fourth-degree polynomial function that calls poly() with exp=4.
        """
        return self.poly(point, deriv, 4)

    def poly8(self, point, deriv):
        """
        Eighth-degree polynomial function that calls poly() with exp=8.
        """
        return self.poly(point, deriv, 8)

    def poly16(self, point, deriv):
        """
        Sixteenth-degree polynomial function that calls poly() with exp=16.
        """
        return self.poly(point, deriv, 16)

    def heaviside(self, point, deriv):
        """
        Heaviside step function, which returns the normal function if point[1] is below a threshold
        (3*pi/4), and zero otherwise.
        Args:
            point (np.array): Point in space.
            deriv (np.array): Derivative at the point.
        Returns:
            float: Output of the Heaviside step function.
        """
        if point[1] < 3 * np.pi / 4:
            return self.normal(point, deriv)
        else:
            return 0
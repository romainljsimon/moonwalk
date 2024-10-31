import numpy as np
from numba import float64
from numba.experimental import jitclass

# Define the specification for numba JIT compilation (optional for optimization)
spec = [
    ('angle_array', float64[:]),  # Example of a scalar array field (not used in the class currently)
]

#@jitclass(spec)
class ZFunctions:
    """
    ZFunctions is a class that defines different types of functions 
    for use in physical simulations (such as random walk observables).
    """

    # Class-level dictionaries (shared by all instances)
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
        'normal': r'$\langle \lambda^2 \rangle$',
        'cos': r'$cos$',
        'poly2': r'$poly2$',
        'poly4': r'$poly4$',
        'poly8': r'$poly8$',
        'poly16': r'$poly16$',
        'heaviside': r'$heaviside$'
    }

    def __init__(self, name: str = 'cos'):
        """
        Initializes the ZFunctions class with a specific function name.
        
        Args:
            name (str): The name of the function to use (e.g., 'cos', 'poly2'). Defaults to 'cos'.
        """
        self.name = name

        # Validate if the provided name exists in the class-level dictionary
        if name not in ZFunctions.z_dict_func:
            raise ValueError(f"Invalid function name: {name}. Available functions: {list(ZFunctions.z_dict_func.keys())}")
        
        # Set the function and legend based on the provided name
        self.foo = getattr(self, ZFunctions.z_dict_func[name])
        self.legend = ZFunctions.z_dict_legend[name]

    # Function definitions for various operations
    def normal(self, point, deriv):
        return deriv[0]

    def cos(self, point, deriv):
        return np.cos(point[1]) * deriv[0]

    def poly(self, point, deriv, exp):
        return (1 - 2 / np.pi * point[1] ** exp) * deriv[0]
    
    def poly2(self, point, deriv):
        return self.poly(point, deriv, 2)

    def poly4(self, point, deriv):
        return self.poly(point, deriv, 4)

    def poly8(self, point, deriv):
        return self.poly(point, deriv, 8)

    def poly16(self, point, deriv):
        return self.poly(point, deriv, 16)

    def heaviside(self, point, deriv):
        if point[1] < 3 * np.pi / 4:
            return self.normal(point, deriv)
        else:
            return 0

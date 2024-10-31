import numpy as np
from numba import float64
from numba.experimental import jitclass

spec = [
    ('position', float64[:]),
    ('position_array', float64[:, :, :]),
    ('dt', float64)         
]

#@jitclass(spec)
class PointWalk:

    def __init__(self, dt: float = 0.1, walls: bool =False, a: float = 100, observables_class: list = []):
        """_summary_

        Args:
            tmax (int, optional): _description_. Defaults to 100.
            dt (float, optional): _description_. Defaults to 0.1.
            walls (bool, optional): _description_. Defaults to False.
            a (float, optional): _description_. Defaults to 100.
            walk_type (str, optional): _description_. Defaults to "random".
            observables (list, optional): _description_. Defaults to [].
        """
        self.dt = dt
        self.walls = walls
        self.a = a
        self.observables_class = observables_class
        self.observables = np.zeros(len(observables_class))
        self.position = np.zeros(2)
        self.column_names = ['lambda', 'phi']

    def walk(self):
        """_summary_

        Args:
            n_iter (int): _description_
        """
        if self.walls:
            deriv = self.deriv_position_walls()
        else:
            deriv = self.deriv_position()
        
        for j, zf in enumerate(self.observables_class):
            foo = zf.foo
            self.observables[j] += foo(self.position, deriv)
        self.position[:2] += deriv

    def deriv_position(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        lam = 1 / np.cos(self.position[1])*np.random.normal(scale=np.sqrt(self.dt))
        phi = -1 / 2 * np.tan(self.position[1])*self.dt + np.random.normal(scale=np.sqrt(self.dt))
        return np.array([lam, phi])
    
    def deriv_position_walls(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        lam = 1 / np.cos(self.position[1])*np.random.normal(scale=np.sqrt(self.dt)) - self.a * self.position[0]*self.dt
        phi = -1 / 2 * np.tan(self.position[1])*self.dt + np.random.normal(scale=np.sqrt(self.dt))  - (2*self.a-1)/2 *self.position[1]* self.dt
        return np.array([lam, phi])
    
    
    """
    def directedwalk(self, theta: float, n_iter: int):
        for i in range(n_iter):
            vec = np.array([0, 1, 1])
            rand_u = vec / np.sqrt(2)
            cross_product = np.cross(rand_u, np.identity(rand_u.shape[0]) * -1)
            tot_rot = np.cos(theta) * np.identity(3) + (1-np.cos(theta))*np.outer(rand_u, rand_u) + np.sin(theta) *cross_product 
            test_rot = tot_rot @ np.ascontiguousarray(self.v1)
            self.v1 = test_rot
            self.v1_array[i+1] = self.v1
    """


spec = [
    ('position', float64[:]),
    ('position_array', float64[:, :, :]),
    ('dt', float64)         
]

#@jitclass(spec)
class LineWalk:

    def __init__(self, dt: float = 0.1, walls: bool =False, a: float = 100, observables_class: list = []):
        """_summary_

        Args:
            tmax (int, optional): _description_. Defaults to 100.
            dt (float, optional): _description_. Defaults to 0.1.
            walls (bool, optional): _description_. Defaults to False.
            a (float, optional): _description_. Defaults to 100.
            walk_type (str, optional): _description_. Defaults to "random".
            observables (list, optional): _description_. Defaults to [].
        """
        self.dt = dt
        self.walls = walls
        self.a = a
        self.observables_class = observables_class
        self.observables = np.zeros(len(observables_class))
        self.position = np.zeros(3)

    def random_walk(self):
        """_summary_

        Args:
            n_iter (int): _description_
        """
        if self.walls:
            deriv = self.deriv_position_walls()
        else:
            deriv = self.deriv_position()
        
        for j, zf in enumerate(self.observables_class):
            foo = zf.foo
            self.observables[j] += foo(self.position, deriv)
        self.position[:2] += deriv

    def deriv_position(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        dlam = 1 / np.cos(self.position[1])*np.random.normal(scale=np.sqrt(self.dt))
        dphi = -1 / 2 * np.tan(self.position[1])*self.dt + np.random.normal(scale=np.sqrt(self.dt))
        theta = np.random.normal(scale=np.sqrt(self.dt))
        return np.array([dlam, dphi, theta])
    
    def deriv_position_walls(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        lam = 1 / np.cos(self.position[1])*np.random.normal(scale=np.sqrt(self.dt)) - self.a * self.position[0]*self.dt
        phi = -1 / 2 * np.tan(self.position[1])*self.dt + np.random.normal(scale=np.sqrt(self.dt))  - (2*self.a-1)/2 *self.position[1]* self.dt
        theta = np.random.normal(scale=np.sqrt(self.dt)) - self.a * self.position[2]*self.dt
        return np.array([lam, phi, theta])
    
    
    """
    def directedwalk(self, theta: float, n_iter: int):
        for i in range(n_iter):
            vec = np.array([0, 1, 1])
            rand_u = vec / np.sqrt(2)
            cross_product = np.cross(rand_u, np.identity(rand_u.shape[0]) * -1)
            tot_rot = np.cos(theta) * np.identity(3) + (1-np.cos(theta))*np.outer(rand_u, rand_u) + np.sin(theta) *cross_product 
            test_rot = tot_rot @ np.ascontiguousarray(self.v1)
            self.v1 = test_rot
            self.v1_array[i+1] = self.v1
    """


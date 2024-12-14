from .brownian_motion import PointWalk
from .z_functions import ZFunctions
from .random_walk import XYZWalk
from .file_utils import write_data
from julia import Pkg

def ensure_dependency(package_name):
    """
    Check if a Julia package is added, and if not, add it.

    Args:
        package_name (str): The name of the Julia package.
    """
    # Get the list of currently added packages
    dependencies = Pkg.dependencies()  # Returns a list of package metadata
    package_names = [pkg_info.name for pkg_info in dependencies.values()]    # Check if the package is already installed
    if package_name not in package_names:
        print(f"Package '{package_name}' not found. Adding it...")
        Pkg.add(package_name)

ensure_dependency("DifferentialEquations")

__all__ = ['PointWalk', 'ZFunctions', 'XYZWalk', 'write_data']
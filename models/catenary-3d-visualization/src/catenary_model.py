import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation as R

# Constants
g = 9.81  # Gravity (m/s^2)

# === 1. Standard Catenary Model ===
def catenary_equation(x, C):
    """Compute standard catenary shape."""
    return (np.cosh(C * x) - 1) / C

def solve_catenary_params(L, l, H):
    """Find the catenary parameter C given length, horizontal distance, and height."""
    def objective(C):
        return (2 * np.sinh(C * l / 2) / C - L) ** 2  # Error function

    res = minimize_scalar(objective, bounds=(1e-5, 10), method='bounded')
    return res.x if res.success else None

def generate_catenary(x_range, L, l, H):
    """Generate a standard catenary curve."""
    C = solve_catenary_params(L, l, H)
    if C is None:
        raise ValueError("Failed to compute catenary parameter.")
    
    x = np.linspace(-l/2, l/2, 100)
    y = catenary_equation(x, C)
    
    return x, y

def apply_gamma_rotation(x, y, gamma_deg):
    """Apply γ-rotation around the catenary axis."""
    gamma_rad = np.radians(gamma_deg)
    R_gamma = R.from_euler('y', gamma_rad, degrees=False).as_matrix()[:2, :2]
    xy_rotated = R_gamma @ np.vstack((x, y))
    return xy_rotated[0, :], xy_rotated[1, :]

def apply_theta_rotation(x, y, theta_deg):
    """Apply θ-rotation to simulate hydrodynamic effects."""
    theta_rad = np.radians(theta_deg)
    R_theta = R.from_euler('z', theta_rad, degrees=False).as_matrix()[:2, :2]
    xy_rotated = R_theta @ np.vstack((x, y))
    return xy_rotated[0, :], xy_rotated[1, :]

# The file is ready for further development or integration with visualization code.
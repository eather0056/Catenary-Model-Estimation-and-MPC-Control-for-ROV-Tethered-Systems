import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation as R

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

# === 2. Augmented Catenary Model ===
def apply_gamma_rotation(x, y, gamma_deg):
    """Apply γ-rotation (Sway motion)."""
    gamma_rad = np.radians(gamma_deg)
    R_gamma = R.from_euler('y', gamma_rad, degrees=False).as_matrix()[:2, :2]
    xy_rotated = R_gamma @ np.vstack((x, y))
    return xy_rotated[0, :], xy_rotated[1, :]

def apply_theta_rotation(x, y, theta_deg):
    """Apply θ-rotation (Surge motion)."""
    theta_rad = np.radians(theta_deg)
    R_theta = R.from_euler('z', theta_rad, degrees=False).as_matrix()[:2, :2]
    xy_rotated = R_theta @ np.vstack((x, y))
    return xy_rotated[0, :], xy_rotated[1, :]

# === 3. Compute Augmented Catenary Model with Experimental Data ===
def process_experimental_data(csv_path):
    """Reads experimental data and applies the augmented catenary model."""
    df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    rod_end = df[['rod_end X', 'rod_end Y', 'rod_end Z']].to_numpy()
    robot_attach = df[['robot_cable_attach_point X', 'robot_cable_attach_point Y', 'robot_cable_attach_point Z']].to_numpy()
    
    # Compute catenary axis
    cable_vector = robot_attach - rod_end
    cable_length = np.linalg.norm(cable_vector, axis=1)
    
    # Assume initial parameters (can be tuned based on data)
    gamma_deg = 15  # Sway angle
    theta_deg = 10  # Surge angle
    L = np.mean(cable_length)  # Average cable length from data
    l = np.linalg.norm(robot_attach[:, :2] - rod_end[:, :2], axis=1).mean()  # Horizontal distance
    H = np.abs(robot_attach[:, 2] - rod_end[:, 2]).mean()  # Height difference
    
    # Compute standard catenary
    x, y = generate_catenary(x_range=(-l/2, l/2), L=L, l=l, H=H)
    
    # Apply augmentations
    x_gamma, y_gamma = apply_gamma_rotation(x, y, gamma_deg)
    x_theta, y_theta = apply_theta_rotation(x_gamma, y_gamma, theta_deg)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Standard Catenary", linestyle="dashed", color="blue")
    plt.plot(x_gamma, y_gamma, label=f"γ-Augmented (γ={gamma_deg}°)", linestyle="dotted", color="red")
    plt.plot(x_theta, y_theta, label=f"θγ-Augmented (θ={theta_deg}°)", color="black")
    
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Augmented Catenary Model with Experimental Data")
    plt.legend()
    plt.grid()
    plt.show()

# === 4. Run with Experimental Data ===
csv_path = "path_to_experimental_data.csv"  # Replace with actual path
process_experimental_data(csv_path)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from catenary_model import generate_catenary, apply_gamma_rotation, apply_theta_rotation

def plot_catenary_3d(L=3, l=2, H=1, gamma_deg=15, theta_deg=10):
    """Create a 3D plot of the catenary model with augmentations."""
    # Generate standard catenary
    x, y = generate_catenary(x_range=(-l/2, l/2), L=L, l=l, H=H)
    z = np.zeros_like(x)  # Assuming z=0 for the catenary in 2D

    # Apply γ-augmentation (Sway)
    x_gamma, y_gamma = apply_gamma_rotation(x, y, gamma_deg)
    z_gamma = np.zeros_like(x_gamma)

    # Apply θ-augmentation (Surge)
    x_theta, y_theta = apply_theta_rotation(x_gamma, y_gamma, theta_deg)
    z_theta = np.zeros_like(x_theta)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Standard Catenary', linestyle="dashed", color='blue')
    ax.plot(x_gamma, y_gamma, z_gamma, label=f'γ-Augmented (γ={gamma_deg}°)', linestyle="dotted", color='red')
    ax.plot(x_theta, y_theta, z_theta, label=f'θγ-Augmented (θ={theta_deg}°)', color='black')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D Visualization of Augmented Catenary Model')
    ax.legend()
    plt.show()

# Run the 3D visualization
plot_catenary_3d()
from pympc.models.catenary import Catenary
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a Catenary object
catenary = Catenary(length=3., reference_frame='ENU')

# Initial attachment points (P0 at origin, P1 along Y-axis)
P0 = np.array([0., 0., 0.])  
P1 = np.array([1., 1., 0.])  

def compute_catenary(start, end):
    """Compute catenary points between start and end."""
    out = catenary(start, end)
    return out[3] if out[3] is not None else np.array([start, end])

# Compute standard catenary (along Y-axis)
catenary_standard = compute_catenary(P0, P1)

# Rotation angles
theta = np.radians(-30)  # Initial tilt around X-axis
gamma = np.radians(10)   # Rotation around Y-axis (augments X-Z plane)

# Compute vector P0 → P1 and rotation axis for theta
v = P1 - P0
z = np.array([0, 0, 1])
x = v.copy()
x[2] = 0
x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9 else np.array([1., 0., 0.])
k_theta = np.cross(x, z)
k_theta = k_theta / np.linalg.norm(k_theta) if np.linalg.norm(k_theta) > 1e-9 else np.array([0., 1., 0.])

def rodrigues_rotation(v, k, theta):
    """Rotate vector v around axis k by theta using Rodrigues' formula."""
    k = k / np.linalg.norm(k)
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

# Compute rotated P1' for theta (tilted catenary)
P1_prime = P0 + rodrigues_rotation(v, k_theta, theta)
catenary_rotated_theta = compute_catenary(P0, P1_prime)

# Inverse theta rotation to realign endpoints to P0-P1
catenary_transformed = np.array([P0 + rodrigues_rotation(q - P0, k_theta, -theta) for q in catenary_rotated_theta])

# Apply gamma rotation around Y-axis (P0-P1 axis)
k_gamma = np.array([0, 1, 0])  # Rotate around Y-axis to keep P1 fixed
catenary_rotated_gamma = np.array([P0 + rodrigues_rotation(q - P0, k_gamma, gamma) for q in catenary_transformed])

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=45)
ax.set_proj_type('ortho')

# Original catenary (blue)
ax.plot(catenary_standard[:, 0], catenary_standard[:, 1], catenary_standard[:, 2], 
        'b', label="Original Catenary")

# Theta-rotated catenary (red dashed)
ax.plot(catenary_rotated_theta[:, 0], catenary_rotated_theta[:, 1], catenary_rotated_theta[:, 2], 
        'r--', label="Theta-Rotated Catenary")

# Transformed catenary after inverse theta (green dashed)
ax.plot(catenary_transformed[:, 0], catenary_transformed[:, 1], catenary_transformed[:, 2], 
        'g--', linewidth=1, label="Theta-Adjusted Catenary")

# Gamma-rotated catenary (magenta)
ax.plot(catenary_rotated_gamma[:, 0], catenary_rotated_gamma[:, 1], catenary_rotated_gamma[:, 2], 
        'm', linewidth=2, label="Full Augmented Catenary (Gamma)")

# Fixed points
ax.scatter(*P0, color='r', s=100, label="P0 (Fixed)")
ax.scatter(*P1, color='g', s=100, label="P1 (Fixed)")
ax.scatter(*P1_prime, color='r', s=100, label="P1_prime (Theta)")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Augmented Catenary with Gamma Rotation (γ={np.degrees(gamma):.1f}°)")
ax.legend()
plt.tight_layout()
plt.show()
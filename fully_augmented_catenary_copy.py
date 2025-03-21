from pympc.models.catenary import Catenary
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rodrigues_rotation(v, k, theta):
    """Apply Rodrigues' rotation formula to vector v around axis k by theta."""
    k = k / np.linalg.norm(k)  # Ensure unit vector
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

def transform_catenary(P0, P1, catenary, theta, gamma):
    """Apply Theta and Gamma rotation transformations to a catenary."""
    def compute_catenary(start, end):
        """Compute catenary points between start and end."""
        out = catenary(start, end)
        if out[3] is not None:
            return out[3]
        return np.array([start, end])
    
    # Compute standard catenary
    catenary_standard = compute_catenary(P0, P1)
    
    # Compute vector P0 → P1
    v = P1 - P0
    
    # Compute Theta rotation axis
    z = np.array([0, 0, 1])  # World Z-axis
    x = v.copy()
    x[2] = 0  # Project v onto XY-plane
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1., 0., 0.])
    else:
        x = x / np.linalg.norm(x)
    k_theta = np.cross(x, z)
    if np.linalg.norm(k_theta) < 1e-9:
        k_theta = np.array([0., 1., 0.])
    else:
        k_theta = k_theta / np.linalg.norm(k_theta)
    
    # Compute rotated P1' for Theta
    P1_prime = P0 + rodrigues_rotation(v, k_theta, theta)
    catenary_rotated_theta = compute_catenary(P0, P1_prime)
    
    # Apply inverse theta rotation to get transformed catenary (P0-P1)
    catenary_transformed = np.array([P0 + rodrigues_rotation(q - P0, k_theta, -theta) for q in catenary_rotated_theta])
    
    # Define the new rotation axis as the vector from P0 to P1 (normalized)
    k_gamma = P1 - P0
    k_gamma = k_gamma / np.linalg.norm(k_gamma)  # Normalize the pivot axis
    
    # Rotate each point in the transformed catenary around P0P1 vector
    catenary_rotated_gamma = np.array([
        P0 + rodrigues_rotation(q - P0, k_gamma, gamma) for q in catenary_transformed
    ])
    
    return catenary_standard, catenary_rotated_theta, catenary_transformed, catenary_rotated_gamma 

# Create a Catenary object
catenary = Catenary(length=3., reference_frame='ENU')

# Initial attachment points
P0 = np.array([0., 0., 0.])  # Fixed point A
P1 = np.array([1., 1., 1.])  # Fixed point B

theta = np.radians(-30)
gamma = np.radians(20)

# Get the final augmented catenary
catenary_standard, catenary_rotated_theta, catenary_transformed, catenary_rotated_gamma = transform_catenary(P0, P1, catenary, theta, gamma)

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=45)
ax.set_proj_type('ortho')

# Original catenary
ax.plot(catenary_standard[:, 0], catenary_standard[:, 1], catenary_standard[:, 2], 
        'b', label="Original Catenary")

# Rotated Catenary (P0 to P1')
ax.plot(catenary_rotated_theta[:, 0], catenary_rotated_theta[:, 1], catenary_rotated_theta[:, 2], 'r--', label="Rotated Catenary (Theta)")

# Transformed catenary (after theta adjustment)
ax.plot(catenary_transformed[:, 0], catenary_transformed[:, 1], catenary_transformed[:, 2], 
        'g--', linewidth=1, label="Theta Adjusted")

# Full augmented catenary (gamma rotated + scaled)
ax.plot(catenary_rotated_gamma[:, 0], catenary_rotated_gamma[:, 1], catenary_rotated_gamma[:, 2], 
        'm', linewidth=2, label="Full Augmented (Gamma)")

# Fixed points
ax.scatter(*P0, color='r', s=100, label="P0 (Fixed)")
ax.scatter(*P1, color='g', s=100, label="P1 (Fixed)")

# Formatting
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Full Augmented Catenary (Gamma={np.degrees(gamma):.1f}°)")
ax.legend()
plt.tight_layout()
plt.axis("equal")
plt.show()

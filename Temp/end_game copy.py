from pympc.models.catenary import Catenary
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a Catenary object
catenary = Catenary(length=3., reference_frame='ENU')

# Initial attachment points
P0 = np.array([0., 0., 0.])  # Fixed point A
P1 = np.array([1., 1., 0.])  # Fixed point B (modify to test)

def compute_catenary(start, end):
    """Compute catenary points between start and end."""
    out = catenary(start, end)
    if out[3] is not None:
        return out[3]
    return np.array([start, end])

# Compute standard catenary
catenary_standard = compute_catenary(P0, P1)

# Rotation angle (Theta in degrees → converted to radians)
theta = np.radians(30)

# Compute vector P0 → P1
v = P1 - P0  # Vector from P0 to P1

# Rotation Axis: Always the Y-axis (0,1,0) in ENU
z = np.array([0, 0, 1])# z world
x = v.copy()
x[2] = 0
x = x / np.linalg.norm(x)
k = np.cross(x,z)


def rodrigues_rotation(v, k, theta):
    """Apply Rodrigues' rotation formula to vector v around axis k by theta."""
    k = k / np.linalg.norm(k)  # Ensure unit vector
    v_rot = v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))
    return v_rot

# Compute rotated P1' (ensuring change in X-Z plane)
P1_prime = P0 + rodrigues_rotation(v, k, theta)
print(f"Rotated P1: {P1_prime}")

# Compute new catenary with P0 → P1'
catenary_rotated = compute_catenary(P0, P1_prime)

# Plot the catenaries
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=45)
ax.set_proj_type('ortho')

# Standard Catenary
ax.plot(catenary_standard[:, 0], catenary_standard[:, 1], catenary_standard[:, 2], 'b', label="Standard Catenary")

# Rotated Catenary (P0 to P1')
ax.plot(catenary_rotated[:, 0], catenary_rotated[:, 1], catenary_rotated[:, 2], 'r--', label="Rotated Catenary (Theta)")

# Fixed Points
ax.scatter(*P0, color='r', s=100, label="P0 (Fixed)")
ax.scatter(*P1, color='g', s=100, label="P1 (Original)")
ax.scatter(*P1_prime, color='m', s=100, label="P1' (Rotated)")

# Labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Catenary Rotation (Theta) around Y-axis")
ax.legend()

plt.show()

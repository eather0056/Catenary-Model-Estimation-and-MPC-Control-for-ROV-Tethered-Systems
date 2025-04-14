from pympc.models.catenary import Catenary
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main_fun import transform_catenary

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

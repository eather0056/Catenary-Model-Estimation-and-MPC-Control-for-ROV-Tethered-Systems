from pympc.models.catenary import Catenary
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main_fun import transform_catenary

# Create a Catenary object
catenary = Catenary(length=3., reference_frame='ENU')

# Initial attachment points
P0 = np.array([0., 0., 0.])  # Fixed point A
P1 = np.array([1., 1., -1.])  # Fixed point B

theta = np.radians(-10)
gamma = np.radians(20)

# Get the final augmented catenary
catenary_standard, catenary_rotated_theta, catenary_transformed, catenary_rotated_gamma = transform_catenary(P0, P1, catenary, theta, gamma)

# Find lowest point on catenary_standard (minimum z)
min_idx = np.argmin(catenary_standard[:, 2])
min_point = catenary_standard[min_idx]


# # Plotting
# fig = plt.figure(figsize=(12, 8))
# # fig.patch.set_facecolor('white')

# ax = fig.add_subplot(111, projection='3d')
# ax.set_facecolor('white')
# ax.view_init(elev=20, azim=45)
# ax.set_proj_type('ortho')
# Set white background


# # Original catenary
# ax.plot(catenary_standard[:, 0], catenary_standard[:, 1], catenary_standard[:, 2], 
#         'b', label="Original Catenary")

# # Rotated Catenary (P0 to P1')
# ax.plot(catenary_rotated_theta[:, 0], catenary_rotated_theta[:, 1], catenary_rotated_theta[:, 2], 'r--', label="Rotated Catenary (Theta)")

# # Transformed catenary (after theta adjustment)
# ax.plot(catenary_transformed[:, 0], catenary_transformed[:, 1], catenary_transformed[:, 2], 
#         'g--', linewidth=1, label="Theta Adjusted")

# # Full augmented catenary (gamma rotated + scaled)
# ax.plot(catenary_rotated_gamma[:, 0], catenary_rotated_gamma[:, 1], catenary_rotated_gamma[:, 2], 
#         'm', linewidth=2, label="Full Augmented (Gamma)")

# # Fixed points
# ax.scatter(*P0, color='r', s=100, label="P0 (Fixed)")
# ax.scatter(*P1, color='g', s=100, label="P1 (Fixed)")

# # Formatting
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title(f"Full Augmented Catenary (Gamma={np.degrees(gamma):.1f}°)")
# ax.legend()
# plt.tight_layout()
# plt.axis("equal")
# ax.grid(False)
# plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# White background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# View and projection settings
ax.view_init(elev=20, azim=45)
ax.set_proj_type('ortho')

# Plotting your data
ax.plot(catenary_standard[:, 0], catenary_standard[:, 1], catenary_standard[:, 2], 'b')
ax.plot(catenary_rotated_theta[:, 0], catenary_rotated_theta[:, 1], catenary_rotated_theta[:, 2], 'r--')
ax.plot(catenary_transformed[:, 0], catenary_transformed[:, 1], catenary_transformed[:, 2], 'g--', linewidth=1)
ax.plot(catenary_rotated_gamma[:, 0], catenary_rotated_gamma[:, 1], catenary_rotated_gamma[:, 2], 'm', linewidth=2)


# Points
ax.scatter(*P0, color='r', s=100)
ax.text(*P0, r'$(^vX_i,^vZ_i)$', fontsize=10, ha='right')
ax.scatter(*P1, color='g', s=100)
ax.text(*P1, r'$(^vX_f,^vZ_f)$', fontsize=10, ha='left')
ax.scatter(*min_point, color='black', s=60)
# ax.text(*min_point + 0.1, r'$s(x)$', fontsize=10, ha='center', va='top')

# Span line (horizontal dashed line)
ax.plot([P0[0], P1[0]], [P0[1], P1[1]], [min_point[2], min_point[2]], 'k--', label=r'$d$ (span)', linewidth=0.5)
ax.text(((P0[0] + P1[0]) / 2 ) - 0.1, (P0[1] + P1[1]) / 2, min_point[2], r'$d$', fontsize=10, ha='center', va='bottom')
# Sag lines from P0 and P1 to min_point level
ax.plot([P0[0], P0[0]], [P0[1], P0[1]], [min_point[2], P0[2]], 'k--', label=r'$h_i$ (sag)', linewidth=0.5)
ax.text(P0[0], P0[1], (min_point[2] + P0[2]) / 2, r'$h_i$', fontsize=10, ha='right', va='center')
ax.plot([P1[0], P1[0]], [P1[1], P1[1]], [min_point[2], P1[2]], 'k--', linewidth=0.5)
ax.text(P1[0], P1[1], (min_point[2] + P1[2]) / 2, r'$h_f$', fontsize=10, ha='left', va='center')

# Dashed line between P0 and P1 (representing theta)
ax.plot([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], 'k--', label=r'$\theta$', linewidth=0.5)
# ax.text((P0[0] + P1[0]) / 2, (P0[1] + P1[1]) / 2, (P0[2] + P1[2]) / 2, r'$\theta$', fontsize=10, ha='center', va='center')

# Dashed line between P0 and the end point of catenary_rotated_theta
end_point_theta = catenary_rotated_theta[1]
ax.plot([P0[0], end_point_theta[0]], [P0[1], end_point_theta[1]], [P0[2], end_point_theta[2]], 'k--', linewidth=0.5)
ax.text((P0[0] + end_point_theta[0]) / 2, (P0[1] + end_point_theta[1]) / 2, (P0[2] + end_point_theta[2]) / 2 +0.1, 
        r'$\theta_{rot}$', fontsize=10, ha='center', va='center')

# Dot at the end point of catenary_rotated_theta
ax.scatter(*end_point_theta, color='blue', s=80)
# ax.text(*end_point_theta, r'$(x_{end}, y_{end}, z_{end})$', fontsize=10, ha='left', va='bottom')



# Square box verticals from P0 and P1
ax.plot([P0[0], P0[0]], [P0[1], P0[1]], [P0[2], min_point[2]], 'k--')
ax.plot([P1[0], P1[0]], [P1[1], P1[1]], [P1[2], min_point[2]], 'k--')

# Local frame at min_point
length = 0.2
ax.quiver(min_point[0], min_point[1], min_point[2], length, 0, 0, color='olive')
ax.text(min_point[0] + length, min_point[1], min_point[2], r'$x_v$', color='olive', fontsize=10)
ax.quiver(min_point[0], min_point[1], min_point[2], 0, 0, length, color='olive')
ax.text(min_point[0], min_point[1], min_point[2] + length, r'$z_v$', color='olive', fontsize=10)
ax.text(min_point[0], min_point[1], min_point[2] - 0.15, r'$\mathcal{O}_v$', fontsize=10, color='purple', ha='center')

# Clean up axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
ax.set_title('')
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_visible(False)
    axis.line.set_color((1, 1, 1, 0))

plt.tight_layout()
plt.show()

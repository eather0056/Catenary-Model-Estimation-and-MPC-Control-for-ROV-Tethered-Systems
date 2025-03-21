# This script calculates the movement of a catenary (a curve formed by a hanging flexible chain or cable) 
# based on the angles theta and gamma. 
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pympc.models.catenary import Catenary
import matplotlib.animation as animation

# === CONSTANTS ===
NUM_FRAMES = 50  # Number of frames per experiment
STEP_DELAY = 0.05  # Pause time between frames
THETA = np.radians(10)  # Rotation angle in radians
GAMMA = np.radians(15)  # Rotation angle in radians

# === Catenary Object ===
catenary = Catenary(length=3., reference_frame='ENU')

# === Initial Positions ===
A_INIT = np.array([0., 0., 0.])  # ROV 1 (Fixed)
B_INIT = np.array([1., 1., 0.])  # ROV 2 (Movable)

a = A_INIT.copy()
b = B_INIT.copy()

# === PLOT SETUP ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=5)
ax.set_proj_type('ortho')

def rodrigues_rotation(v, k, angle):
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle))

def transform_catenary(P0, P1, catenary, theta, gamma):
    def compute_catenary(start, end):
        out = catenary(start, end)
        return out[3] if out[3] is not None else np.array([start, end])
    
    catenary_standard = compute_catenary(P0, P1)
    v = P1 - P0
    z = np.array([0, 0, 1])
    x = v.copy()
    x[2] = 0  # Project v onto XY-plane
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9 else np.array([1., 0., 0.])
    k_theta = np.cross(x, z)
    k_theta = k_theta / np.linalg.norm(k_theta) if np.linalg.norm(k_theta) > 1e-9 else np.array([0., 1., 0.])
    
    P1_prime = P0 + rodrigues_rotation(v, k_theta, theta)
    catenary_rotated_theta = compute_catenary(P0, P1_prime)
    catenary_transformed = np.array([P0 + rodrigues_rotation(q - P0, k_theta, -theta) for q in catenary_rotated_theta])
    
    k_gamma = P1 - P0
    k_gamma = k_gamma / np.linalg.norm(k_gamma)
    catenary_rotated_gamma = np.array([P0 + rodrigues_rotation(q - P0, k_gamma, gamma) for q in catenary_transformed])
    
    return catenary_rotated_gamma

def update_curve():
    global a, b
    out = catenary(a, b)
    if out[3] is not None:
        transformed_points = transform_catenary(a, b, catenary, THETA, GAMMA)
        catenary_line.set_data(transformed_points[:, 0], transformed_points[:, 1])
        catenary_line.set_3d_properties(transformed_points[:, 2])
    
    point_a._offsets3d = ([a[0]], [a[1]], [a[2]])
    point_b._offsets3d = ([b[0]], [b[1]], [b[2]])

def execute_experiment():
    def animate(frame):
        b[0] += 0.03  # Move ROV 2 in x-direction
        a[0] += 0.01
        update_curve()
        return catenary_line, point_a, point_b
    
    ani = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES, interval=STEP_DELAY * 1000, blit=False)
    ani.save('experiment_transformed_AB.gif', writer='imagemagick')
    print("Experiment completed and saved as experiment_transformed.gif!")

# === PLOT ELEMENTS ===
out = catenary(a, b)
if out[3] is not None:
    x_vals, y_vals, z_vals = out[3][:, 0], out[3][:, 1], out[3][:, 2]
else:
    x_vals, y_vals, z_vals = [a[0], b[0]], [a[1], b[1]], [a[2], b[2]]

catenary_line, = ax.plot(x_vals, y_vals, z_vals, 'b', label="Catenary Curve")
point_a = ax.scatter(*a, color='r', s=100, label="Point A (ROV 1 - Fixed)")
point_b = ax.scatter(*b, color='g', s=100, label="Point B (ROV 2 - Movable)")

def main():
    execute_experiment()
    plt.show()

if __name__ == "__main__":
    main()

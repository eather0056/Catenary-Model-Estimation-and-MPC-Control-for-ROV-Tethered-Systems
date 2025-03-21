import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pympc.models.catenary import Catenary
import time
import imageio
import os

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
    
    return catenary_rotated_gamma

# Load CSV file
data_path = "data.csv"
df = pd.read_csv(data_path)

# Create Catenary object
catenary = Catenary(length=3., reference_frame='ENU')

# Create folder to save frames
frames_folder = "catenary_frames"
os.makedirs(frames_folder, exist_ok=True)

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=45)
ax.set_proj_type('ortho')

# Fix axis limits
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Real-time plotting loop
frames = []
for index, row in df.iterrows():
    # Extract P0 and P1
    P0 = np.array([row['rod_end X'], row['rod_end Y'], row['rod_end Z']])/1000
    P1 = np.array([row['robot_cable_attach_point X'], row['robot_cable_attach_point Y'], row['robot_cable_attach_point Z']])/1000

    # Extract Theta and Gamma
    theta = row['Theta']
    gamma = row['Gamma']
    
    # Compute transformed catenary
    catenary_final = transform_catenary(P0, P1, catenary, theta, gamma)
    
    # Clear previous plot and update
    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Catenary Transformation at Time {row['Time']}s")
    
    ax.plot(catenary_final[:, 0], catenary_final[:, 1], catenary_final[:, 2], 'm', linewidth=2, label="Transformed Catenary")
    ax.scatter(*P0, color='r', s=100, label="P0 (Rod End)")
    ax.scatter(*P1, color='g', s=100, label="P1 (Cable Attach)")
    ax.legend()
    
    # Save frame
    frame_path = os.path.join(frames_folder, f"frame_{index:03d}.png")
    plt.savefig(frame_path)
    frames.append(frame_path)
    
    plt.pause(0.05)  # Real-time update speed

# Convert saved frames to GIF
gif_path = "catenary_simulation.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.05) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

print(f"GIF saved as {gif_path}")
plt.show()
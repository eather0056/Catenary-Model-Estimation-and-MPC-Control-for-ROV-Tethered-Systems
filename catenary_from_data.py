# This script 1 - directly use the data to animate the fully augmented catenary model import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pympc.models.catenary import Catenary
import time
import imageio
import os
from catenary_transform import transform_catenary

# Load CSV file
datasetname = "data.csv"
data_path = f"Data/{datasetname}"
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
    _, _, _, catenary_final = transform_catenary(P0, P1, catenary, theta, gamma)
    
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
gif_path = f"catenary_simulation_{datasetname}.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.05) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

print(f"GIF saved as {gif_path}")
plt.show()
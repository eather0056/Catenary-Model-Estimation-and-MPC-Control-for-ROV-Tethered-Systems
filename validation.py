import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.animation import FuncAnimation
# === Load data ===
df = pd.read_csv("Data/L_dynamique1x100dis2_0014_corrected_velocity.csv")
df.columns = df.columns.str.strip()

# === Extract relevant columns ===
t = 900

# === Convert cable coordinate colums to numeric if they exit ===
for i in range(0, 16):  # cable_0 to cable_15
    for axis in ['X', 'Y', 'Z']:
        orig_col = f"cable_{i} {axis}"
        cor_col = f"cable_cor_{i} {axis}"
        if orig_col in df.columns:
            df[orig_col] = pd.to_numeric(df[orig_col], errors='coerce')
        if cor_col in df.columns:
            df[cor_col] = pd.to_numeric(df[cor_col], errors='coerce')

# === Prepare cable point sets ===
original_points = []
corrected_points = []
rob_speed = df[["rob_speed X", "rob_speed Y", "rob_speed Z"]].values
rob_cor_speed = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values
used_indices = []  # Track which indices are valid

for i in range(0, 16):  # Adjust if fewer points are available
    cols_orig = [f"cable_{i} X", f"cable_{i} Y", f"cable_{i} Z"]
    cols_corr = [f"cable_cor_{i} X", f"cable_cor_{i} Y", f"cable_cor_{i} Z"]
    
    if all(col in df.columns for col in cols_orig + cols_corr):
        original_points.append(df[cols_orig].values)
        corrected_points.append(df[cols_corr].values)
        used_indices.append(i)
    else:
        print(f"Skipping cable point {i} due to missing columns")

if len(original_points) < 3:
    raise ValueError("Not enough valid cable points to compute rotation (need at least 3)")

# Stack into arrays: shape (n_samples, n_points, 3)
original_points = np.stack(original_points, axis=1)
corrected_points = np.stack(corrected_points, axis=1)

# === Check if data is valid at this time step ===
# if not (np.isfinite(df.corrected_points[t]).all() and
#         np.isfinite(df.rob_speed[t]) and
#         np.isfinite(df.rob_cor_speed[t])):
#     raise ValueError(f"Data at time step {t} is invalid.")

# === Get data for frame t ===
P = corrected_points[t]
v_orig = rob_speed[t]   # Original robot speed
v_cor = rob_cor_speed[t]  # Corrected robot speed
centroid_P = P.mean(axis=0)

# === Plot the cable and velocity vector ===
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection= '3d')

# Plot cable points
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='blue', label= 'Corrected cable pointe')

# Plot original velocity vector
ax.quiver(*centroid_P, *v_orig, color='red', label= 'Original robot speed')

# Plot corrected velocity vector
ax.quiver(*centroid_P, *v_cor, color='green', label= 'Corrected robot speed')

# Plot settings
ax.set_title(f"Velocity Vector at Time Step {t}")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.grid(True)
ax.set_box_aspect([1,1,1]) # Aspect ratio is 1:1:1

plt.tight_layout()
plt.show()


time_steps = [0, 25, 50, 75, 100, 125, 150, 175]  # Pick 8 valid steps
fig = plt.figure(figsize=(16, 10))

for idx, t in enumerate(time_steps):
    ax = fig.add_subplot(2, 4, idx+1, projection='3d')
    
    P = corrected_points[t]
    v_orig = rob_speed[t]
    v_cor = rob_cor_speed[t]
    
    if not (np.isfinite(P).all() and np.isfinite(v_orig).all() and np.isfinite(v_cor).all()):
        ax.set_title(f"t={t} (Invalid)")
        continue

    centroid = P.mean(axis=0)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='blue')
    ax.quiver(*centroid, *v_orig, length=0.1, color='red', normalize=True)
    ax.quiver(*centroid, *v_cor, length=0.1, color='green', normalize=True)

    ax.set_title(f"t={t}")
    ax.set_xlim(centroid[0]-0.2, centroid[0]+0.2)
    ax.set_ylim(centroid[1]-0.2, centroid[1]+0.2)
    ax.set_zlim(centroid[2]-0.2, centroid[2]+0.2)
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')

plt.suptitle("Robot Velocity Correction: Multiple Time Steps", fontsize=16)
plt.tight_layout()
plt.show()


# === Create figure and axes ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# === Initialization function ===
def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title("Robot Velocity Correction Over Time")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return []

# === Animation update function ===
def update(t):
    ax.cla()  # Clear previous frame

    P = corrected_points[t]
    v_orig = rob_speed[t]
    v_cor = rob_cor_speed[t]

    if not (np.isfinite(P).all() and np.isfinite(v_orig).all() and np.isfinite(v_cor).all()):
        return []

    centroid = P.mean(axis=0)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='blue', label='Corrected Cable Points')
    ax.quiver(*centroid, *v_orig, length=0.1, color='red', normalize=True, label='Original Velocity')
    ax.quiver(*centroid, *v_cor, length=0.1, color='green', normalize=True, label='Corrected Velocity')

    ax.set_xlim(centroid[0]-0.2, centroid[0]+0.2)
    ax.set_ylim(centroid[1]-0.2, centroid[1]+0.2)
    ax.set_zlim(centroid[2]-0.2, centroid[2]+0.2)
    ax.set_title(f"Time Step {t}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    return []

# === Run animation ===
num_frames = len(df)
ani = FuncAnimation(fig, update, frames=range(0, num_frames, 5), init_func=init, interval=150, blit=False)

plt.show()


alignment_scores = []

for t in range(len(df)):
    P = corrected_points[t]  # shape: (n_points, 3)
    v = rob_cor_speed[t]     # shape: (3,)
    
    if not (np.isfinite(P).all() and np.isfinite(v).all()):
        alignment_scores.append(np.nan)
        continue

    # Direction vector from cable_cor_0 to cable_cor_15
    cable_vec = P[-1] - P[0]
    norm_cable = np.linalg.norm(cable_vec)
    norm_v = np.linalg.norm(v)

    if norm_cable == 0 or norm_v == 0:
        alignment_scores.append(np.nan)
        continue

    # Cosine similarity (dot product of unit vectors)
    alignment = np.dot(cable_vec, v) / (norm_cable * norm_v)
    alignment_scores.append(alignment)

# Add to DataFrame
df["alignment_score"] = alignment_scores

# Plot over time
plt.figure(figsize=(12, 5))
plt.plot(df["alignment_score"], label="Velocity–Cable Alignment", color='green')
plt.axhline(1.0, linestyle='--', color='gray', linewidth=0.8)
plt.axhline(-1.0, linestyle='--', color='gray', linewidth=0.8)
plt.xlabel("Time Step")
plt.ylabel("Cosine Similarity")
plt.title("Alignment of Corrected Velocity with Cable Axis Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



alignment_orig = []
alignment_corr = []

for t in range(len(df)):
    P = corrected_points[t]
    v_orig = rob_speed[t]
    v_corr = rob_cor_speed[t]

    if not (np.isfinite(P).all() and np.isfinite(v_orig).all() and np.isfinite(v_corr).all()):
        alignment_orig.append(np.nan)
        alignment_corr.append(np.nan)
        continue

    # Cable axis vector (end - start)
    cable_vec = P[-1] - P[0]
    norm_cable = np.linalg.norm(cable_vec)
    norm_v_orig = np.linalg.norm(v_orig)
    norm_v_corr = np.linalg.norm(v_corr)

    if norm_cable == 0 or norm_v_orig == 0 or norm_v_corr == 0:
        alignment_orig.append(np.nan)
        alignment_corr.append(np.nan)
        continue

    # Cosine similarities
    align_o = np.dot(cable_vec, v_orig) / (norm_cable * norm_v_orig)
    align_c = np.dot(cable_vec, v_corr) / (norm_cable * norm_v_corr)

    alignment_orig.append(align_o)
    alignment_corr.append(align_c)

# Save to DataFrame
df["alignment_orig"] = alignment_orig
df["alignment_corr"] = alignment_corr

# === Plot ===
plt.figure(figsize=(14, 5))
plt.plot(df["alignment_orig"], label="Original Velocity Alignment", color='red', alpha=0.5)
plt.plot(df["alignment_corr"], label="Corrected Velocity Alignment", color='green', linewidth=2)
plt.axhline(1.0, linestyle='--', color='gray', linewidth=0.8)
plt.axhline(-1.0, linestyle='--', color='gray', linewidth=0.8)
plt.xlabel("Time Step")
plt.ylabel("Cosine Similarity")
plt.title("Original vs Corrected Velocity Alignment with Cable Axis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("Original Alignment Mean ± Std:", 
      np.nanmean(df["alignment_orig"]), 
      "±", 
      np.nanstd(df["alignment_orig"]))

print("Corrected Alignment Mean ± Std:", 
      np.nanmean(df["alignment_corr"]), 
      "±", 
      np.nanstd(df["alignment_corr"]))


plt.figure(figsize=(10, 4))
plt.hist(df["alignment_orig"], bins=50, alpha=0.5, label="Original", color='red')
plt.hist(df["alignment_corr"], bins=50, alpha=0.5, label="Corrected", color='green')
plt.xlabel("Alignment Score (cosine similarity)")
plt.ylabel("Frequency")
plt.title("Distribution of Velocity Alignment with Cable Axis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

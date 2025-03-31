import pandas as pd
import numpy as np

# === Load and clean data ===
df = pd.read_csv("Data/L_dynamique1x100dis2_0014.csv")
df.columns = df.columns.str.strip()

# === Convert cable coordinate columns to numeric if they exist ===
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

# === Define Kabsch function ===
def compute_rotation_kabsch(P, Q):
    """ P, Q: arrays of shape (N, 3) — original and corrected point sets """
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

# === Apply rotation to robot speed ===
rob_speed = df[["rob_speed X", "rob_speed Y", "rob_speed Z"]].values
rob_cor_speed = []

for t in range(len(df)):
    P = original_points[t]
    Q = corrected_points[t]

    if not (np.isfinite(P).all() and np.isfinite(Q).all()):
        print(f"Skipping time step {t} due to NaNs or Infs")
        rob_cor_speed.append([np.nan, np.nan, np.nan])
        continue

    try:
        R = compute_rotation_kabsch(P, Q)
        v_world = rob_speed[t]
        v_corrected = R @ v_world
        rob_cor_speed.append(v_corrected)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-3)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-3)

    except np.linalg.LinAlgError:
        print(f"Skipping time step {t} due to SVD failure")
        rob_cor_speed.append([np.nan, np.nan, np.nan])


rob_cor_speed = np.array(rob_cor_speed)

# === Save back to DataFrame ===
df["rob_cor_speed X"] = rob_cor_speed[:, 0]
df["rob_cor_speed Y"] = rob_cor_speed[:, 1]
df["rob_cor_speed Z"] = rob_cor_speed[:, 2]

# === Save to CSV ===
df.to_csv("Data/L_dynamique1x100dis2_0014_corrected_velocity.csv", index=False)
print("Corrected robot velocities saved successfully.")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Select a representative time frame ===
t = 100  # Change this to any frame you want to visualize

# === Check if data is valid at this time step ===
if not (np.isfinite(corrected_points[t]).all() and
        np.isfinite(rob_speed[t]).all() and
        np.isfinite(rob_cor_speed[t]).all()):
    raise ValueError(f"Invalid data (NaN or Inf) at time step {t}")

# === Get data for frame t ===
P = corrected_points[t]  # shape: (n_points, 3)
v_orig = rob_speed[t]    # original velocity (in tilted frame)
v_cor = rob_cor_speed[t] # corrected velocity (in world frame)
centroid = P.mean(axis=0)

# === Plot the cable and velocity vectors ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot cable points
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='blue', label='Corrected Cable Points')

# Plot original velocity vector (red)
ax.quiver(*centroid, *v_orig, length=0.1, color='red', normalize=True, label='Original Velocity (tilted frame)')

# Plot corrected velocity vector (green)
ax.quiver(*centroid, *v_cor, length=0.1, color='green', normalize=True, label='Corrected Velocity (world frame)')

# Plot settings
ax.set_title(f"Velocity Vectors at Time Step {t}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.grid(True)
ax.set_box_aspect([1,1,1])  # Equal aspect ratio

plt.tight_layout()
plt.show()

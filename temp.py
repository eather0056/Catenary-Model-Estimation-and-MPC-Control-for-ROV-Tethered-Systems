import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# === Parameters ===
DATA_DIR = "Data"
FILENAME = "L_dynamique1x100dis2_0014_corrected_velocity.csv"
FILEPATH = os.path.join(DATA_DIR, FILENAME)
SAMPLE_CABLES = 16
TIME_STEPS_PLOT = [0, 25, 50, 75, 100, 125, 150, 175]
TIME_ANIM_INTERVAL = 150
T_VIEW = 900  # single time step for snapshot


# === Helper Functions ===
def load_and_clean_data(filepath):
    """Load CSV and clean column names."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def convert_columns_to_numeric(df):
    """Convert relevant cable columns to numeric."""
    for i in range(SAMPLE_CABLES):
        for axis in ['X', 'Y', 'Z']:
            for prefix in [f"cable_{i}", f"cable_cor_{i}"]:
                col = f"{prefix} {axis}"
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def extract_cable_data(df):
    """Extract valid cable data and stack into arrays."""
    orig, corr, used = [], [], []

    for i in range(SAMPLE_CABLES):
        orig_cols = [f"cable_{i} {axis}" for axis in "XYZ"]
        corr_cols = [f"cable_cor_{i} {axis}" for axis in "XYZ"]

        if all(col in df.columns for col in orig_cols + corr_cols):
            orig.append(df[orig_cols].values)
            corr.append(df[corr_cols].values)
            used.append(i)
        else:
            print(f"Skipping cable point {i} due to missing columns")

    if len(orig) < 3:
        raise ValueError("Not enough cable points for rotation analysis.")

    return np.stack(orig, axis=1), np.stack(corr, axis=1), used


def plot_3d_snapshot(P, v_orig, v_cor, t):
    """Plot a 3D snapshot at time t."""
    centroid = P.mean(axis=0)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='blue', label='Cable Points')
    ax.quiver(*centroid, *v_orig, color='red', label='Speed on World Frame')
    ax.quiver(*centroid, *v_cor, color='green', label='Speed on Catenary Frame')
    ax.quiver(*centroid, *v_orig - v_cor, color='purple', label='Speed Difference')
    ax.set_xlim(P[:, 0].min(), P[:, 0].max())
    ax.set(title=f"Velocity at Time {t}", xlabel="X", ylabel="Y", zlabel="Z")
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def plot_multiple_snapshots(points, v_orig, v_cor, steps):
    """Plot multiple 3D snapshots over selected time steps."""
    fig = plt.figure(figsize=(16, 10))
    for idx, t in enumerate(steps):
        ax = fig.add_subplot(2, 4, idx+1, projection='3d')
        P, vo, vc = points[t], v_orig[t], v_cor[t]

        if not (np.isfinite(P).all() and np.isfinite(vo).all() and np.isfinite(vc).all()):
            ax.set_title(f"t={t} (Invalid)")
            continue

        centroid = P.mean(axis=0)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='blue')
        ax.quiver(*centroid, *vo, color='red', normalize=True)
        ax.quiver(*centroid, *vc, color='green', normalize=True)
        ax.set(title=f"t={t}")
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
    plt.suptitle("Velocity Snapshots", fontsize=16)
    plt.tight_layout()
    plt.show()


def animate_velocity(points, v_orig, v_cor, filename="velocity_animation.gif", fps=10):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Compute bounds safely
    all_points_flat = points.reshape(-1, 3)
    valid_points = all_points_flat[np.isfinite(all_points_flat).all(axis=1)]
    if valid_points.size == 0:
        raise ValueError("No valid 3D points available.")
    
    buffer = 0.2
    x_min, x_max = valid_points[:, 0].min() - buffer, valid_points[:, 0].max() + buffer
    y_min, y_max = valid_points[:, 1].min() - buffer, valid_points[:, 1].max() + buffer
    z_min, z_max = valid_points[:, 2].min() - buffer, valid_points[:, 2].max() + buffer

    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title("Velocity Animation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

    def update(t):
        ax.cla()
        P, vo, vc = points[t], v_orig[t], v_cor[t]
        if not (np.isfinite(P).all() and np.isfinite(vo).all() and np.isfinite(vc).all()):
            return

        centroid = P.mean(axis=0)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='blue', label='Cable Points')
        ax.quiver(*centroid, *vo, color='red', label='Velocity in World Frame')
        ax.quiver(*centroid, *vc, color='green', label='Velocity in catenary Frame')
        ax.quiver(*centroid, *vo - vc, color='purple', label='Velocity Difference')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(f"Time Step {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        ax.grid(True)

    anim = FuncAnimation(fig, update, frames=range(0, len(points), 5),
                         init_func=init, interval=150)

    # Save as GIF
    anim.save(filename, writer=PillowWriter(fps=fps), dpi=150)
    print(f"Animation saved as: {filename}")
    plt.close(fig)

def compute_alignment_scores(points, velocities):
    """Compute cosine similarity between cable vector and velocity."""
    scores = []
    for P, v in zip(points, velocities):
        if not (np.isfinite(P).all() and np.isfinite(v).all()):
            scores.append(np.nan)
            continue
        cable_vec = P[-1] - P[0]
        score = np.dot(cable_vec, v) / (np.linalg.norm(cable_vec) * np.linalg.norm(v)) if np.linalg.norm(cable_vec) and np.linalg.norm(v) else np.nan
        scores.append(score)
    return scores


def plot_alignment_over_time(df, col1, col2=None):
    """Plot alignment score(s) over time."""
    plt.figure(figsize=(12, 5))
    plt.plot(df[col1], label="Corrected", color='green', linewidth=2)
    if col2:
        plt.plot(df[col2], label="Original", color='red', alpha=0.5)
    plt.axhline(1.0, linestyle='--', color='gray')
    plt.axhline(-1.0, linestyle='--', color='gray')
    plt.title("Velocity Alignment with Cable Axis")
    plt.xlabel("Time Step")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histogram(df, col1, col2):
    """Histogram of alignment distributions."""
    plt.figure(figsize=(10, 4))
    plt.hist(df[col1], bins=50, alpha=0.5, label="Original", color='red')
    plt.hist(df[col2], bins=50, alpha=0.5, label="Corrected", color='green')
    plt.title("Alignment Score Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Main Execution ===
df = load_and_clean_data(FILEPATH)
df = convert_columns_to_numeric(df)
orig_pts, corr_pts, used_indices = extract_cable_data(df)

rob_speed = df[["rob_speed X", "rob_speed Y", "rob_speed Z"]].values
rob_cor_speed = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values

plot_3d_snapshot(corr_pts[T_VIEW], rob_speed[T_VIEW], rob_cor_speed[T_VIEW], T_VIEW)
plot_multiple_snapshots(corr_pts, rob_speed, rob_cor_speed, TIME_STEPS_PLOT)
animate_velocity(corr_pts, rob_speed, rob_cor_speed)


df["alignment_corr"] = compute_alignment_scores(corr_pts, rob_cor_speed)
df["alignment_orig"] = compute_alignment_scores(corr_pts, rob_speed)

plot_alignment_over_time(df, "alignment_corr", "alignment_orig")
plot_histogram(df, "alignment_orig", "alignment_corr")

print("Original Alignment Mean ± Std:", np.nanmean(df["alignment_orig"]), "±", np.nanstd(df["alignment_orig"]))
print("Corrected Alignment Mean ± Std:", np.nanmean(df["alignment_corr"]), "±", np.nanstd(df["alignment_corr"]))


# === Extract Data ===
time = df["Time"].values
vx = df["rob_speed X"].values
vy = df["rob_speed Y"].values
vz = df["rob_speed Z"].values

vx_cor = df["rob_cor_speed X"].values
vy_cor = df["rob_cor_speed Y"].values
vz_cor = df["rob_cor_speed Z"].values

# === Plotting ===
plt.figure(figsize=(12,6))
plt.subplot(2, 1, 1)
plt.plot(time, vx, label='X Velocity', color='red')
plt.plot(time, vy, label='Y Velocity', color='green')
plt.plot(time, vz, label='Z Velocity', color='blue')
plt.title(f"Velocity Components on World Frame Over Time for {FILENAME}")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, vx_cor, label='X Velocity', color='red')
plt.plot(time, vy_cor, label='Y Velocity', color='green')
plt.plot(time, vz_cor, label='Z Velocity', color='blue')
plt.title(f"Velocity Components on Catenary Frame Over Time for {FILENAME}")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.grid()
plt.tight_layout()
plt.show()

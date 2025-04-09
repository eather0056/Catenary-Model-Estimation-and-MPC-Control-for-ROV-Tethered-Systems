import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.lines import Line2D

# === Parameters ===
DATA_DIR = "Data"
FILENAME = "L_dynamique6y100dis1_0020.csv"
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

        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='blue', label='Cable Points')

        centroid = P.mean(axis=0)
        ax.quiver(*centroid, *vo, color='black', label='World Velocity')
        ax.quiver(*centroid, *vc, color='orange', label='Catenary Velocity')
        ax.quiver(*centroid, *(vo - vc), color='purple', label='Difference')

        selected_indices = np.linspace(0, len(P) - 1, 5, dtype=int)
        colors_W = ['red', 'green', 'blue']
        colors_C = ['cyan', 'olive', 'gray']

        for idx in selected_indices:
            point = P[idx]
            for i in range(3):
                vec_w = np.zeros(3)
                vec_w[i] = vo[i]
                ax.quiver(*point, *vec_w, color=colors_W[i], alpha=0.8, linewidth=1,
                        label=f'World V {["X", "Y", "Z"][i]}' if t == 0 else None)

            for i in range(3):
                vec_c = np.zeros(3)
                vec_c[i] = vc[i]
                ax.quiver(*point, *vec_c, color=colors_C[i], alpha=0.8, linewidth=1,
                        label=f'Catenary V {["X", "Y", "Z"][i]}' if t == 0 else None)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(f"Time Step {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True)
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='World Velocity'),
            Line2D([0], [0], color='orange', lw=2, label='Catenary Velocity'),
            Line2D([0], [0], color='purple', lw=2, label='Velocity Difference'),
            Line2D([0], [0], color='red', lw=2, label='World V X'),
            Line2D([0], [0], color='green', lw=2, label='World V Y'),
            Line2D([0], [0], color='blue', lw=2, label='World V Z'),
            Line2D([0], [0], color='cyan', lw=2, label='Catenary V X'),
            Line2D([0], [0], color='olive', lw=2, label='Catenary V Y'),
            Line2D([0], [0], color='gray', lw=2, label='Catenary V Z')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # === Save as PNG if this frame is selected ===
        if t in save_frames:
            save_path = os.path.join("Results", "velocity_frames", f"frame_{t:04d}.png")
            plt.savefig(save_path, dpi=300)
            print(f"Saved frame: {save_path}")


    anim = FuncAnimation(fig, update, frames=range(0, len(points), 5),
                         init_func=init, interval=100)

    # Save as GIF
    anim.save(filename, writer=PillowWriter(fps=fps), dpi=100)
    print(f"Animation saved as: {filename}")
    plt.close(fig)

# === Main Execution ===
df = load_and_clean_data(FILEPATH)
df = convert_columns_to_numeric(df)
orig_pts, corr_pts, used_indices = extract_cable_data(df)

rob_speed = df[["rob_speed X", "rob_speed Y", "rob_speed Z"]].values
rob_cor_speed = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values

plot_3d_snapshot(corr_pts[T_VIEW], rob_speed[T_VIEW], rob_cor_speed[T_VIEW], T_VIEW)
plot_multiple_snapshots(corr_pts, rob_speed, rob_cor_speed, TIME_STEPS_PLOT)

os.makedirs("velocity_frames", exist_ok=True) 
save_frames = [0, 250, 300, 350, 400]  
animate_velocity(corr_pts, rob_speed, rob_cor_speed)

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
plt.savefig(os.path.join("Results", "Experiment Transformed", f"velocity_components for file {FILENAME}.png"), dpi=300)


# 1. Magnitude difference
mag_raw = np.linalg.norm(rob_speed, axis=1)
mag_cor = np.linalg.norm(rob_cor_speed, axis=1)
mag_diff = np.abs(mag_raw - mag_cor)

plt.figure()
plt.plot(mag_raw, label="Original |v|")
plt.plot(mag_cor, label="Corrected |v|")
plt.title("Velocity Magnitude Before and After Rotation")
plt.xlabel("Time step")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.grid()
plt.savefig(os.path.join("Results", "Experiment Transformed", f"velocity_magnitude for file {FILENAME}.png"), dpi=300)

# 2. Angle between original and corrected (in degrees)
dot_prod = np.einsum('ij,ij->i', rob_speed, rob_cor_speed)
angle_rad = np.arccos(dot_prod / (mag_raw * mag_cor + 1e-8))  # small epsilon to avoid div-by-zero
angle_deg = np.degrees(angle_rad)

plt.figure()
plt.plot(angle_deg)
plt.title("Angle Between Raw and Corrected Velocity Vectors")
plt.xlabel("Time step")
plt.ylabel("Angle (degrees)")
plt.grid()
plt.savefig(os.path.join("Results", "Experiment Transformed", f"velocity_angle for file {FILENAME}.png"), dpi=300)

# 3. Optional: Compare components
plt.figure()
plt.plot(rob_speed[:, 0], label="Raw Vx")
plt.plot(rob_cor_speed[:, 0], label="Corrected Vx")
plt.title("X Component of Velocity")
plt.legend()
plt.grid()
plt.savefig(os.path.join("Results", "Experiment Transformed", f"velocity_x_component for file {FILENAME}.png"), dpi=300)
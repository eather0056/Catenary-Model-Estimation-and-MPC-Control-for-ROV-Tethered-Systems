import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pympc.models.catenary import Catenary
import imageio.v2 as imageio
import os
from PIL import Image
from main_fun import transform_catenary

# === Load CSV ===
datasetname = "L_dynamique6x100dis2_0033.csv"
data_path = f"Data/{datasetname}"
df = pd.read_csv(data_path)

# === Check for predicted columns ===
if 'Theta_Pred' not in df.columns or 'Gamma_Pred' not in df.columns:
    raise ValueError("Predicted Theta and Gamma not found in the CSV.")

# === Create catenary object ===
catenary = Catenary(length=3., reference_frame='ENU')

# === Prepare output ===
gif_path = f"catenary_simulation_{datasetname}.gif"
temp_img = "temp_frame.png"

# === Initialize error storage ===
error_accumulator = []

# === Create GIF writer ===
with imageio.get_writer(gif_path, mode='I', duration=0.05) as writer:
    for index, row in df.iterrows():
        # Create figure and subplots per frame to avoid memory leak
        fig = plt.figure(figsize=(12, 5))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax_hist = fig.add_subplot(1, 2, 2)

        P0 = np.array([row['rod_end X'], row['rod_end Y'], row['rod_end Z']]) / 1000
        P1 = np.array([row['robot_cable_attach_point X'], row['robot_cable_attach_point Y'], row['robot_cable_attach_point Z']]) / 1000

        theta = row['Theta']
        gamma = row['Gamma']
        theta_pred = row['Theta_Pred']
        gamma_pred = row['Gamma_Pred']

        _, _, _, catenary_true = transform_catenary(P0, P1, catenary, theta, gamma)
        _, _, _, catenary_pred = transform_catenary(P0, P1, catenary, theta_pred, gamma_pred)


        errors = np.linalg.norm(catenary_true - catenary_pred, axis=1)
        error_accumulator.append(errors)
        error_array = np.array(error_accumulator)
        mean_errors = np.mean(error_array, axis=0)

        # === 3D Plot ===
        ax3d.clear()
        ax3d.set_xlim([-0.5, 1.5])
        ax3d.set_ylim([-2, 1.5])
        ax3d.set_zlim([-1, 1])
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.set_title(f"Catenary at Time {row['Time']:.2f}s")
        ax3d.view_init(elev=30, azim=60)
        ax3d.set_proj_type('ortho')
        ax3d.plot(catenary_true[:, 0], catenary_true[:, 1], catenary_true[:, 2], 'm', linewidth=2, label="Reference")
        ax3d.plot(catenary_pred[:, 0], catenary_pred[:, 1], catenary_pred[:, 2], 'c--', linewidth=2, label="Predicted")
        ax3d.scatter(*P0, color='r', s=100, label="P0")
        ax3d.scatter(*P1, color='g', s=100, label="P1")
        ax3d.legend()

        # === Histogram ===
        ax_hist.clear()
        ax_hist.bar(range(len(mean_errors)), mean_errors, color='gray')
        ax_hist.set_title("Mean Error per Catenary Point")
        ax_hist.set_xlabel("Catenary Point Index")
        ax_hist.set_ylabel("Mean Error [m]")
        max_y = np.max(mean_errors) if np.max(mean_errors) > 0 else 1.0
        ax_hist.set_ylim(0, max_y * 1.2)

        # === Save and Resize Frame ===
        plt.tight_layout()
        plt.savefig(temp_img, dpi=80)  # LOW DPI to reduce size
        plt.close(fig)  # Prevent memory leak

        image = Image.open(temp_img).resize((800, 400))  # Resize to manageable dimensions
        writer.append_data(np.array(image))

# Clean up
if os.path.exists(temp_img):
    os.remove(temp_img)

print(f"\n✅ GIF saved to: {gif_path}")

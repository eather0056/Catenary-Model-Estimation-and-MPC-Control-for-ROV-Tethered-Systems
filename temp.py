import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Choose flie to visualize ===
filename = "L_dynamique6x100dis2_0033.csv"
data_dir = "Data"
filepath = os.path.join(data_dir, filename)

# === Load and clean data ===
df = pd.read_csv(filepath)
df.columns = df.columns.str.strip()

# === Check if necessary columns exist ===
required_cols = ["Time", 
                  "rob_speed X", "rob_speed Y", "rob_speed Z",
                  "rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z",
                  "rod_end X", "rod_end Y", "rod_end Z",
                  "robot_cable_attach_point X", "robot_cable_attach_point Y", "robot_cable_attach_point Z"]
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"Missing one or more of these columns in {filename}: {required_cols}")

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
plt.title(f"Original Velocity Components Over Time for {filename}")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, vx_cor, label='X Velocity', color='red')
plt.plot(time, vy_cor, label='Y Velocity', color='green')
plt.plot(time, vz_cor, label='Z Velocity', color='blue')
plt.title(f"Corrected Velocity Components Over Time for {filename}")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.grid()
plt.tight_layout()
plt.show()

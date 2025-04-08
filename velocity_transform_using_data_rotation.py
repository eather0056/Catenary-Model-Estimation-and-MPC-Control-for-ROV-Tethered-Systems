import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# === Load and clean data ===
df = pd.read_csv("Data/L_dynamique1x100dis2_0014.csv")
df.columns = df.columns.str.strip()

# === Get robot velocity in world (catenary) frame ===
rob_speed = df[['rob_speed X', 'rob_speed Y', 'rob_speed Z']].values

# === Prepare corrected velocity array ===
rob_cor_speed = []

# === Loop through each time step to apply rotation ===
for t in range(len(df)):
    try:
        # Extract rotation matrix R from catenary frame to vertical frame
        R = np.array([
            [df.at[t, 'exc1'], df.at[t, 'eyc1'], df.at[t, 'ezc1']],
            [df.at[t, 'exc2'], df.at[t, 'eyc2'], df.at[t, 'ezc2']],
            [df.at[t, 'exc3'], df.at[t, 'eyc3'], df.at[t, 'ezc3']],
        ])

        v_world = rob_speed[t]
        v_corrected = R @ v_world # Rotated velocity into vertical frame
        rob_cor_speed.append(v_corrected)
    except KeyError as e:
        print(f"Missing rotation matrix data at time {t}: {e}")
        rob_cor_speed.append([np.nan, np.nan, np.nan])

# === Convert to numpy array ===
rob_cor_speed = np.array(rob_cor_speed)

# === Add corrected velocity to DataFrame ===
df["rob_cor_speed X"] = rob_cor_speed[:, 0]
df["rob_cor_speed Y"] = rob_cor_speed[:, 1]
df["rob_cor_speed Z"] = rob_cor_speed[:, 2]

# === Save to CSV ===
df.to_csv("Data/L_dynamique1x100dis2_0014_corrected_velocity.csv", index=False)
print("Corrected robot velocities saved successfully.")


# === Choose flie to visualize ===
filename = "L_dynamique1x100dis2_0014_corrected_velocity.csv"
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
plt.grid()

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
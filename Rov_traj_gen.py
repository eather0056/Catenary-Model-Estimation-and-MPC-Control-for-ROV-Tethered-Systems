import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Define parameters
n_steps = 100
total_time = 10  # seconds
time = np.linspace(0, total_time, n_steps)
trajectory_name = "rov_trajectory"

# Initialize trajectories (12 states: position, orientation, velocity, angular velocity)
trajectory_0 = np.zeros((12, n_steps))  # ROV 1
trajectory_1 = np.zeros((12, n_steps))  # ROV 2
separation = 1  # Separation between ROVs

# Define experimental cases
exp_case = 14  # Choose from 1 to 14 

if exp_case == 1:  # Same Direction (Parallel)
    trajectory_0[0, :] = 0.03 * time  # x-position moves forward
    trajectory_1[0, :] = 0.03 * time
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[6, :] = 0.03  # Forward velocity
    trajectory_1[6, :] = 0.03
    
elif exp_case == 2:  # Same Direction (Different Speeds)
    trajectory_0[0, :] = 0.03 * time
    trajectory_1[0, :] = 0.06 * time
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[6, :] = 0.03
    trajectory_1[6, :] = 0.06
    
elif exp_case == 3:  # Opposite Directions
    trajectory_0[0, :] = 0.03 * time
    trajectory_1[0, :] = -0.03 * time
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[6, :] = 0.03
    trajectory_1[6, :] = -0.03
    
elif exp_case == 4:  # One Static, One Moving
    trajectory_0[0, :] = 0  # Static
    trajectory_1[0, :] = 0.05 * time  # Moving
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_1[6, :] = 0.5
    
elif exp_case == 5:  # Depth Variation (Same X-Y)
    trajectory_0[0, :] = 0.03 * time
    trajectory_1[0, :] = 0.03 * time
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[2, :] = 0.5  # Fixed depth
    trajectory_1[2, :] = np.linspace(0.5, 1.0, n_steps)  # Increasing depth
    trajectory_0[6, :] = 0.03
    trajectory_1[6, :] = 0.03
    
elif exp_case == 6:  # Depth Variation (Different Speeds)
    trajectory_0[0, :] = 0.03 * time
    trajectory_1[0, :] = 0.06 * time
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[2, :] = 0.5
    trajectory_1[2, :] = np.linspace(0.5, 1.0, n_steps)
    trajectory_0[6, :] = 0.03
    trajectory_1[6, :] = 0.06
    
elif exp_case == 7:  # Depth Variation (One Static)
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[2, :] = 0.5  # Static depth
    trajectory_1[2, :] = np.linspace(0.5, 1.0, n_steps)  # Increasing depth
    trajectory_0[6, :] = 0
    trajectory_1[6, :] = 0.05

elif exp_case == 8:  # Rapid ROV  Movement
    trajectory_0[0, :] = 0.05 * time
    trajectory_1[0, :] = 0.05 * time
    trajectory_0[1, :] = 0.05* np.sin(2 * np.pi * time)
    trajectory_1[1, :] = separation + 0.05* np.sin(2 * np.pi * time)
    trajectory_0[6, :] = 0.05 * np.cos(2 * np.pi * time / total_time)
    trajectory_1[6, :] = 0.05 * np.cos(2 * np.pi * time / total_time)
    
elif exp_case == 9:  # PRBS Movement (ROV 1)
    trajectory_0[0, :] = np.random.choice([-0.1, 0.1], n_steps) 
    trajectory_1[0, :] = 0.05 * time
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[6, :] = np.random.choice([-0.03, 0.03], n_steps)
    
elif exp_case == 10:  # PRBS Movement (Both ROVs)
    trajectory_0[0, :] = np.random.choice([-0.1, 0.1], n_steps) 
    trajectory_1[0, :] = np.random.choice([-0.1, 0.1], n_steps) 
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[6, :] = np.random.choice([-0.03, 0.03], n_steps)
    trajectory_1[6, :] = np.random.choice([-0.03, 0.03], n_steps)
    
elif exp_case == 11:  # Zig-Zag Movement
    trajectory_0[0, :] = 0.05 * time
    trajectory_1[0, :] = 0.05 * time
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[1, :] = 0.2* np.sin(2 * np.pi * time)
    trajectory_1[6, :] = 0.03
    
elif exp_case == 12:  # Circular Path
    trajectory_0[0, :] = 0.4 * np.cos(2 * np.pi * time / total_time)
    trajectory_0[1, :] = 0.4 * np.sin(2 * np.pi * time / total_time)
    trajectory_1[0, :] = 0.1 * np.cos(2 * np.pi * time / total_time)
    trajectory_1[1, :] = 0.1 * np.sin(2 * np.pi * time / total_time)


elif exp_case == 13:  # Large Excursions
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    trajectory_0[0, :] = 0.06 * time
    trajectory_1[0, :] = 0.06 * time
    trajectory_0[6, :] = 0.06
    trajectory_1[6, :] = 0.06
    
elif exp_case == 14:  # Static Cable Drift
    trajectory_1[1, :] = separation  # y-position separation (parallel)
    pass  # Both remain static

# Define directory path
directory = "Results/Trajectory Data"

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Define the filename
filename = f"{trajectory_name}_exp{exp_case}.csv"

# Full file path
file_path = os.path.join(directory, filename)

# === Save the data to a CSV file ===
with open(file_path, 'w') as f:
    # Write the CSV header
    f.write("br0_x, br0_y, br0_z, br0_phi, br0_theta, br0_psi, br0_u, br0_v, br0_w, br0_p, br0_q, br0_r, "
            "br1_x, br1_y, br1_z, br1_phi, br1_theta, br1_psi, br1_u, br1_v, br1_w, br1_p, br1_q, br1_r\n")

    # Write data rows
    for state_0, state_1 in zip(trajectory_0.T, trajectory_1.T):
        f.write(','.join(f'{v:.3f}' for v in state_0) + ',')
        f.write(','.join(f'{v:.3f}' for v in state_1) + '\n')

# Print confirmation message
print(f"Trajectory for experiment {exp_case} saved as {file_path}")

# Define plot filename
plot_filename = f"{trajectory_name}_exp{exp_case}_trajectory.png"
plot_file_path = os.path.join(directory, plot_filename)

# Function to plot and save the 3D trajectories
def plot_trajectories_3d(trajectory_0, trajectory_1, save_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(trajectory_0[0, :], trajectory_0[1, :], trajectory_0[2, :], label='ROV 1', color='blue')
    ax.plot(trajectory_1[0, :], trajectory_1[1, :], trajectory_1[2, :], label='ROV 2', color='red')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(f'3D Trajectories of ROVs for Experiment {exp_case}')
    ax.legend()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

    print(f"3D trajectory plot saved as {save_path}")

# Run the function to plot and save
plot_trajectories_3d(trajectory_0, trajectory_1, plot_file_path)
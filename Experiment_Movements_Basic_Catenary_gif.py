import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pympc.models.catenary import Catenary
import matplotlib.animation as animation

# === CONSTANTS ===
NUM_FRAMES = 50  # Number of frames per experiment
STEP_DELAY = 0.05  # Pause time between frames

# === Catenary Object ===
catenary = Catenary(length=3., reference_frame='ENU')

# === Initial Positions ===
A_INIT = np.array([0., 0., 0.])  # ROV 1 (Fixed)
B_INIT = np.array([1., 1., 0.])  # ROV 2 (Movable)

# === Create Working Copies ===
a = A_INIT.copy()
b = B_INIT.copy()

# === PLOT SETUP ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=45) # Change the view azim angle to 5
ax.set_proj_type('ortho')

# === COMPUTE INITIAL CATENARY ===
out = catenary(a, b)
if out[3] is not None:
    x_vals, y_vals, z_vals = out[3][:, 0], out[3][:, 1], out[3][:, 2]
else:
    x_vals, y_vals, z_vals = [a[0], b[0]], [a[1], b[1]], [a[2], b[2]]

# === PLOT ELEMENTS ===
catenary_line, = ax.plot(x_vals, y_vals, z_vals, 'b', label="Catenary Curve")
point_a = ax.scatter(*a, color='r', s=100, label="Point A (ROV 1 - Fixed)")
point_b = ax.scatter(*b, color='g', s=100, label="Point B (ROV 2 - Movable)")

# === AXIS CONFIGURATION ===
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
#ax.set_title("3D Catenary Simulation - Select Experiment")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-3, 1)
#ax.legend()

# === EXPERIMENT MOVEMENT PATTERNS ===
experiments = {
    "1": {"A": [0.03, 0, 0], "B": [0.03, 0, 0]},  # Same Direction (Parallel)
    "2": {"A": [0.03, 0, 0], "B": [0.06, 0, 0]},  # Same Direction (Different Speeds)
    "3": {"A": [0.03, 0, 0], "B": [-0.03, 0, 0]}, # Opposite Directions
    "4": {"A": [0, 0, 0], "B": [0.05, 0, 0]},     # One Static, One Moving
    "5": {"A": [0.03, 0, 0], "B": [0.03, 0, 0.05]},  # Depth Variation (Same X-Y)
    "6": {"A": [0.03, 0, 0], "B": [0.06, 0, 0.05]},  # Depth Variation (Different Speeds)
    "7": {"A": [0, 0, 0], "B": [0.05, 0, 0.02]},  # Depth Variation (One Static)
    "8": {"A": [0.05, 0, 0], "B": [0.05, 0, 0]},  # Rapid ROV Movement
    "9": {"A": ["PRBS", 0, 0], "B": [0.03, 0, 0]},  # PRBS Movement (ROV 1)
    "10": {"A": ["PRBS", 0, 0], "B": ["PRBS", 0, 0]},  # PRBS Movement (Both ROVs)
    "11": {"A": ["zigzag", 0, 0], "B": [0.05, 0, 0]},  # Zig-Zag Movement
    "12": {"A": ["circular", 0, 0], "B": ["circular", 0, 0]},  # Circular Path
    "13": {"A": [0.06, 0, 0], "B": [0.06, 0, 0]},  # Large Excursions
    "14": {"A": [0, 0, 0], "B": [0, 0, 0]},  # Static Cable Drift
}

# === FUNCTIONS ===
def update_curve():
    """Update the catenary curve and redraw the plot."""
    out = catenary(a, b)
    if out[3] is not None:
        new_points = out[3]
        catenary_line.set_data(new_points[:, 0], new_points[:, 1])
        catenary_line.set_3d_properties(new_points[:, 2])
    
    point_a._offsets3d = ([a[0]], [a[1]], [a[2]])
    point_b._offsets3d = ([b[0]], [b[1]], [b[2]])

def reset_positions():
    """Reset positions of A and B to initial states."""
    global a, b
    a[:] = A_INIT
    b[:] = B_INIT
    update_curve()

def execute_experiment(exp_id):
    """Execute the selected experiment with real-time animation."""
    movement = experiments.get(exp_id)
    if not movement:
        print("Invalid Experiment ID. Please select again.")
        return

    print(f"Running Experiment {exp_id}...")
    
    def animate(frame):
        # Move ROV 1 (Point A)
        if isinstance(movement["A"][0], str):
            if movement["A"][0] == "PRBS":
                a[0] = np.random.uniform(-0.05, 0.05)
            elif movement["A"][0] == "circular":
                angle = (frame / NUM_FRAMES) * 2 * np.pi
                a[0] = np.cos(angle) * 0.4
                a[1] = np.sin(angle) * 0.4
            elif movement["A"][0] == "zigzag":
                a[0] += 0.03
                a[1] = np.sin(a[0]) * 0.03
        else:
            a[:] = np.add(a, movement["A"])
        
        # Move ROV 2 (Point B)
        if isinstance(movement["B"][0], str):
            if movement["B"][0] == "PRBS":
                b[0] = np.random.uniform(-0.05, 0.05)
            elif movement["B"][0] == "circular":
                angle = (frame / NUM_FRAMES) * 2 * np.pi
                b[0] = np.cos(angle) * 0.05
                b[1] = np.sin(angle) * 0.05
            elif movement["B"][0] == "zigzag":
                b[0] += 0.05
                b[1] = np.sin(b[0]) * 0.03
        else:
            b[:] = np.add(b, movement["B"])
        
        update_curve()
        return catenary_line, point_a, point_b

    ani = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES, interval=STEP_DELAY*1000, blit=False)
    ani.save(f'experiment_{exp_id}.gif', writer='imagemagick')
    print(f"Experiment {exp_id} completed and saved as experiment_{exp_id}.gif!")
    reset_positions()

# === MAIN PROGRAM ===
while True:
    print("\nSelect an experiment (1-14) or type 'exit' to quit:")
    user_input = input("> ").strip()
    
    if user_input.lower() == "exit":
        break
    elif user_input in experiments:
        execute_experiment(user_input)
    else:
        print("Invalid input. Please enter a number between 1 and 14.")

plt.show()
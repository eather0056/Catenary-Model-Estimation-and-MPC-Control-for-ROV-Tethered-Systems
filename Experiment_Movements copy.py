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
def rodrigues_rotation(v, k, theta):
    """Rotate vector v around unit vector k by theta radians using Rodrigues' formula."""
    k = k / np.linalg.norm(k)
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

def transform_catenary(P0, P1, catenary, theta, gamma):
    """Apply Theta and Gamma rotation transformations to a catenary."""
    def compute_catenary(start, end):
        """Compute catenary points between start and end."""
        out = catenary(start, end)
        if out[3] is not None:
            return out[3]
        return np.array([start, end])
    
    # Compute standard catenary
    catenary_standard = compute_catenary(P0, P1)
    
    # Compute vector P0 → P1
    v = P1 - P0
    
    # Compute Theta rotation axis
    z = np.array([0, 0, 1])  # World Z-axis
    x = v.copy()
    x[2] = 0  # Project v onto XY-plane
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1., 0., 0.])
    else:
        x = x / np.linalg.norm(x)
    k_theta = np.cross(x, z)
    if np.linalg.norm(k_theta) < 1e-9:
        k_theta = np.array([0., 1., 0.])
    else:
        k_theta = k_theta / np.linalg.norm(k_theta)
    
    # Compute rotated P1' for Theta
    P1_prime = P0 + rodrigues_rotation(v, k_theta, theta)
    catenary_rotated_theta = compute_catenary(P0, P1_prime)
    
    # Apply inverse theta rotation to get transformed catenary (P0-P1)
    catenary_transformed = np.array([P0 + rodrigues_rotation(q - P0, k_theta, -theta) for q in catenary_rotated_theta])
    
    # Define the new rotation axis as the vector from P0 to P1 (normalized)
    k_gamma = P1 - P0
    if np.linalg.norm(k_gamma) < 1e-9:
        k_gamma = np.array([1., 0., 0.])  # Default axis if P0 and P1 are the same
    else:
        k_gamma = k_gamma / np.linalg.norm(k_gamma)  # Normalize the pivot axis
    
    # Rotate each point in the transformed catenary around P0P1 vector
    catenary_rotated_gamma = np.array([
        P0 + rodrigues_rotation(q - P0, k_gamma, gamma) for q in catenary_transformed
    ])
    
    return catenary_rotated_gamma 

# Initial transformed catenary with theta=0 and gamma=0 to match original setup
initial_catenary = transform_catenary(a, b, catenary, 0, 0)
x_vals, y_vals, z_vals = initial_catenary[:, 0], initial_catenary[:, 1], initial_catenary[:, 2]

# === PLOT ELEMENTS ===
catenary_line, = ax.plot(x_vals, y_vals, z_vals, 'b', label="Catenary Curve")
point_a = ax.scatter(*a, color='r', s=100, label="Point A (ROV 1 - Fixed)")
point_b = ax.scatter(*b, color='g', s=100, label="Point B (ROV 2 - Movable)")

# === AXIS CONFIGURATION ===
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-3, 1)

# === EXPERIMENT MOVEMENT PATTERNS ===
experiments = {
    "1": {"A": [0.03, 0, 0], "B": [0.03, 0, 0]},
    "2": {"A": [0.03, 0, 0], "B": [0.06, 0, 0]},
    "3": {"A": [0.03, 0, 0], "B": [-0.03, 0, 0]},
    "4": {"A": [0, 0, 0], "B": [0.05, 0, 0]},
    "5": {"A": [0.03, 0, 0], "B": [0.03, 0, 0.05]},
    "6": {"A": [0.03, 0, 0], "B": [0.06, 0, 0.05]},
    "7": {"A": [0, 0, 0], "B": [0.05, 0, 0.02]},
    "8": {"A": [0.05, 0, 0], "B": [0.05, 0, 0]},
    "9": {"A": ["PRBS", 0, 0], "B": [0.03, 0, 0]},
    "10": {"A": ["PRBS", 0, 0], "B": ["PRBS", 0, 0]},
    "11": {"A": ["zigzag", 0, 0], "B": [0.05, 0, 0]},
    "12": {"A": ["circular", 0, 0], "B": ["circular", 0, 0]},
    "13": {"A": [0.06, 0, 0], "B": [0.06, 0, 0]},
    "14": {"A": [0, 0, 0], "B": [0, 0, 0]},
}

# === FUNCTIONS ===
def update_curve():
    """Update the catenary curve and redraw the plot."""
    new_points = transform_catenary(a, b, catenary, 0, 0)
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
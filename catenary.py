# This code return a basic catenary with a interactive 3d plot that can be control by keyboard arrows

from pympc.models.catenary import Catenary
from numpy import array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import numpy as np

# Create a Catenary object
catenary = Catenary(length=3., reference_frame='ENU')

# Initial attachment points
a = array([0., 0., 0.])  # Fixed point A
b = array([1., 1., 0.])  # Movable point B

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set viewing angle and projection
ax.view_init(elev=20, azim=45)
ax.set_proj_type('ortho')

# Compute initial catenary curve
out = catenary(a, b)
if out[3] is not None:
    x_vals, y_vals, z_vals = out[3][:, 0], out[3][:, 1], out[3][:, 2]
else:
    x_vals, y_vals, z_vals = [a[0], b[0]], [a[1], b[1]], [a[2], b[2]]

# Plot elements
catenary_line, = ax.plot(x_vals, y_vals, z_vals, 'b', label="Catenary Curve")
point_a = ax.scatter(*a, color='r', s=100, label="Point A (Fixed)")
point_b = ax.scatter(*b, color='g', s=100, label="Point B (Movable)")

# Axis configuration
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Catenary Control (Use Arrow Keys + PgUp/PgDn)")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-3, 1)
ax.legend()

def update_curve():
    """Update the catenary curve and redraw."""
    global catenary_line, point_b
    
    # Calculate new catenary
    out = catenary(a, b)
    if out[3] is not None:
        new_points = out[3]
        catenary_line.set_data(new_points[:, 0], new_points[:, 1])
        catenary_line.set_3d_properties(new_points[:, 2])
    #print(out)
    # Update point B position
    point_b._offsets3d = ([b[0]], [b[1]], [b[2]])
    plt.draw()

def on_key_press(event):
    """Handle keyboard events for 3D movement."""
    step_size = 0.1  # Adjustment increment
    
    if event.key == 'left':
        b[0] -= step_size
    elif event.key == 'right':
        b[0] += step_size
    elif event.key == 'down':
        b[1] -= step_size
    elif event.key == 'up':
        b[1] += step_size
    elif event.key == 'pageup':
        b[2] += step_size
    elif event.key == 'pagedown':
        b[2] -= step_size
    else:
        return
    
    update_curve()

# Connect keyboard event handler
fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_catenary_3D(p0, p1, rope_length, num_points):
    """Compute 3D catenary curve given anchor points and rope length."""
    x0, y0, z0 = p0
    x1, y1, z1 = p1

    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    direct_dist = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if rope_length <= direct_dist:
        return np.linspace(p0, p1, num_points)

    half_span = direct_dist / 2
    a = half_span  
    for _ in range(100):  
        lhs = 2 * a * np.sinh(direct_dist / (2 * a))
        a_new = a * rope_length / lhs
        if abs(a_new - a) < 1e-6:
            a = a_new
            break
        a = a_new

    offset_height = a * np.cosh(half_span / a)

    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        xi = x0 + dx * t
        yi = y0 + dy * t
        zi = z0 + dz * t
        x_pos = (t * direct_dist) - half_span
        sag = offset_height - a * np.cosh(x_pos / a)
        zi -= sag  
        points.append([xi, yi, zi])

    return np.array(points)


class InteractiveCatenary3D:
    """Interactive 3D catenary simulation with draggable anchor."""

    def __init__(self):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.p0 = np.array([0, 0, 10])  # Fixed anchor
        self.p1 = np.array([10, 0, 10])  # Movable anchor
        self.rope_length = 12
        self.num_points = 100

        self.update_catenary()

        # Plot elements
        self.line, = self.ax.plot(self.curve[:, 0], self.curve[:, 1], self.curve[:, 2], 'b', lw=2)
        self.fixed_marker = self.ax.scatter(*self.p0, color='g', s=100, label="Fixed Point")
        self.movable_marker = self.ax.scatter(*self.p1, color='r', s=100, label="Movable Point")

        # Configure 3D plot
        self.ax.set_xlim(-5, 15)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(0, 15)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Interactive 3D Catenary")

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.dragging = False
        plt.show()

    def update_catenary(self):
        """Compute updated catenary curve based on current anchor positions."""
        self.curve = compute_catenary_3D(self.p0, self.p1, self.rope_length, self.num_points)

    def on_press(self, event):
        """Detects if we clicked near the movable anchor."""
        if event.inaxes is None:
            return

        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None:
            return

        # Project click onto 3D space
        clicked_3d = self.project_to_3d(x_click, y_click)
        if clicked_3d is not None:
            dist = np.linalg.norm(clicked_3d - self.p1)
            if dist < 1:  # If click is near movable anchor, start dragging
                self.dragging = True

    def on_drag(self, event):
        """Handles dragging of the movable anchor."""
        if not self.dragging or event.inaxes is None:
            return

        x_new, y_new = event.xdata, event.ydata
        if x_new is None or y_new is None:
            return

        # Project new position onto 3D space
        new_3d_pos = self.project_to_3d(x_new, y_new)
        if new_3d_pos is not None:
            self.p1[:2] = new_3d_pos[:2]  # Update only X, Y (keep Z same)
            self.update_catenary()

            # Update plot
            self.line.set_data(self.curve[:, 0], self.curve[:, 1])
            self.line.set_3d_properties(self.curve[:, 2])
            self.movable_marker._offsets3d = (self.p1[0:1], self.p1[1:2], self.p1[2:3])

            plt.draw()

    def on_release(self, event):
        """Stop dragging when the mouse is released."""
        self.dragging = False

    def project_to_3d(self, x_click, y_click):
        """Project a 2D mouse click to a 3D plane (keeping Z fixed)."""
        if x_click is None or y_click is None:
            return None

        # Use inverse transformation to estimate 3D position
        inv_transform = np.linalg.inv(self.ax.get_proj())
        x_norm = (x_click - self.ax.get_xlim()[0]) / (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 2 - 1
        y_norm = (y_click - self.ax.get_ylim()[0]) / (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 2 - 1
        z_fixed = self.p1[2]  # Keep Z constant while dragging

        # Convert to 3D coordinates
        projected_3d = np.dot(inv_transform, [x_norm, y_norm, z_fixed, 1])
        return projected_3d[:3] / projected_3d[3]  # Normalize

# Run interactive plot
InteractiveCatenary3D()

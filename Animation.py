# This script runs the animation of a catenary curve in 3D space using the Manim library.
# The catenary curve is transformed using Rodrigues' rotation formula to simulate the movement of two ROVs. 
# For running this, use the command: python -m manim -pql Animation.py CatenaryAnimation

from manim import *
import numpy as np
from pympc.models.catenary import Catenary

class CatenaryAnimation(ThreeDScene):
    def construct(self):
        catenary = Catenary(length=3., reference_frame='ENU')
        P0 = np.array([0., 0., 0.])
        P1 = np.array([1., 1., 0.])

        def compute_catenary(start, end):
            out = catenary(start, end)
            return out[3] if out[3] is not None else np.array([start, end])

        def to_manim_points(points):
            return [np.array([p[0], p[1], p[2]]) for p in points]

        def label_point(label, point, direction=UP):
            return Text(label).scale(0.4).next_to(Dot3D(point=point), direction)

        def show_curve(curve, label_text, plane='xz'):
            label = Text(label_text).scale(0.4).next_to(curve, UP, buff=0.3)
            if plane == 'xz':
                label.rotate(PI / 2, axis=UP)   # Move to XZ
                label.rotate(-PI, axis=RIGHT)          # Flip to face camera
            elif plane == 'yz':
                label.rotate(PI / 2, axis=OUT)     # Move to YZ
                label.rotate(PI, axis=RIGHT)       # Flip to face camera
            return label


        self.set_camera_orientation(phi=75 * DEGREES, theta=15 * DEGREES, zoom=1.5)

        axes = ThreeDAxes()
        self.play(Create(axes))

        # Step 1: Original Catenary
        catenary_standard = compute_catenary(P0, P1)
        standard_catenary = VMobject().set_points_smoothly(to_manim_points(catenary_standard)).set_color(BLUE)
        P0_dot = Dot3D(point=P0, color=RED, radius=0.1)
        P1_dot = Dot3D(point=P1, color=GREEN, radius=0.1)
        P0_label = label_point("P0", P0)
        P1_label = label_point("P1", P1)
        label_standard = show_curve(standard_catenary, "Original Catenary", plane='xz')
        P0_P1_line = Line3D(start=P0, end=P1, color=WHITE, stroke_width=1)
        self.play(Create(standard_catenary), FadeIn(P0_dot, P1_dot, P0_label, P1_label, label_standard))
        self.play(Create(P0_P1_line))
        self.wait(2)

        # Step 2: Rotate endpoint (Theta)
        theta = np.radians(-30)
        v = P1 - P0
        z = np.array([0, 0, 1])
        k_theta = np.cross(v, z)
        k_theta = k_theta / np.linalg.norm(k_theta) if np.linalg.norm(k_theta) > 1e-9 else np.array([0., 1., 0.])

        def rodrigues_rotation(v, k, theta):
            k = k / np.linalg.norm(k)
            return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

        P1_prime = P0 + rodrigues_rotation(v, k_theta, theta)
        catenary_rotated_theta = compute_catenary(P0, P1_prime)
        rotated_catenary = VMobject().set_points_smoothly(to_manim_points(catenary_rotated_theta)).set_color(RED)
        P1_prime_dot = Dot3D(point=P1_prime, color=YELLOW, radius=0.1)
        P1_prime_label = label_point("P1'", P1_prime)
        label_rotated = show_curve(rotated_catenary, "Rotated (Theta)", plane='xz')
        P0_P1_prime_line = Line3D(start=P0, end=P1_prime, color=YELLOW, stroke_width=1)
        self.play(Create(P0_P1_prime_line))
        # angle_label = Text("theta = -30°").scale(0.4).next_to(P0_dot, UP, buff=0.5)
        # self.play(Write(angle_label))
        self.play(FadeOut(label_standard, P0_P1_line, P0_P1_prime_line), Transform(standard_catenary, rotated_catenary), FadeIn(P1_prime_dot, P1_prime_label, label_rotated))
        self.wait(2)

        # Step 3: Undo Theta (Transform Back to P1)
        catenary_transformed = np.array([P0 + rodrigues_rotation(q - P0, k_theta, -theta) for q in catenary_rotated_theta])
        catenary_transformed_vm = VMobject().set_points_smoothly(to_manim_points(catenary_transformed)).set_color(GREEN)
        label_theta_fixed = show_curve(catenary_transformed_vm, "Theta Corrected", plane='xz')
        self.play(FadeOut(label_rotated), Transform(standard_catenary, catenary_transformed_vm), FadeOut(P1_prime_dot, P1_prime_label), FadeIn(label_theta_fixed))
        self.wait(2)

        # Step 4: Gamma rotation around P0P1
        gamma = np.radians(20)
        k_gamma = P1 - P0
        k_gamma = k_gamma / np.linalg.norm(k_gamma)
        catenary_rotated_gamma = np.array([P0 + rodrigues_rotation(q - P0, k_gamma, gamma) for q in catenary_transformed])
        gamma_catenary = VMobject().set_points_smoothly(to_manim_points(catenary_rotated_gamma)).set_color(ORANGE)
        label_gamma = show_curve(gamma_catenary, "Gamma Rotated", plane='xz')
        # gamma_label = Text("gamma = 20°").scale(0.5).next_to(P0_dot, UP, buff=0.5)
        # self.play(Write(gamma_label))
        self.play(FadeOut(label_theta_fixed), Transform(catenary_transformed_vm, gamma_catenary), FadeIn(label_gamma))
        self.wait(3)

        # # Step 5: Scaling
        # endpoint_rotated = catenary_rotated_gamma[-1]
        # original_vector = P1 - P0
        # rotated_vector = endpoint_rotated - P0
        # scaling_factor = np.linalg.norm(original_vector) / np.linalg.norm(rotated_vector) if np.linalg.norm(rotated_vector) > 1e-9 else 1.0
        # catenary_final = np.array([P0 + (q - P0) * scaling_factor for q in catenary_rotated_gamma])
        # final_catenary_vm = VMobject().set_points_smoothly(to_manim_points(catenary_final)).set_color(PURPLE)
        # label_final = show_curve(final_catenary_vm, "Final Scaled", plane='xz')
        # self.play(FadeOut(label_gamma), Transform(gamma_catenary, final_catenary_vm), FadeIn(label_final))
        # self.wait(3)

        # Cleanup
        self.play(FadeOut(standard_catenary, catenary_transformed_vm, gamma_catenary,
                          P0_dot, P1_dot, P0_label, P1_label, axes))
        self.wait(1)

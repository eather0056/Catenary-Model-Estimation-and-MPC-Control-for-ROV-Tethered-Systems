from manim import *
import numpy as np
from pympc.models.catenary import Catenary

class CatenaryAnimation(ThreeDScene):
    def construct(self):
        # Define parameters
        catenary = Catenary(length=3., reference_frame='ENU')
        P0 = np.array([0., 0., 0.])
        P1 = np.array([0., 1., 0.])
        
        def compute_catenary(start, end):
            out = catenary(start, end)
            if out[3] is not None:
                return out[3]
            return np.array([start, end])
        
        # Compute standard catenary
        catenary_standard = compute_catenary(P0, P1)
        
        # Convert numpy array to manim points
        def to_manim_points(points):
            return [np.array([p[0], p[1], p[2]]) for p in points]
        
        standard_catenary = VMobject()
        standard_catenary.set_points_smoothly(to_manim_points(catenary_standard))
        standard_catenary.set_color(BLUE)
        
        # Labels and Axes
        axes = ThreeDAxes()
        P0_dot = Dot3D(point=P0, color=RED, radius=0.1)
        P1_dot = Dot3D(point=P1, color=GREEN, radius=0.1)
        
        # Step 1: Show initial catenary
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes))
        self.play(Create(standard_catenary), FadeIn(P0_dot, P1_dot))
        self.wait(2)
        
        # Rotation transformations
        theta = np.radians(-30)
        gamma = np.radians(20)
        
        def rodrigues_rotation(v, k, theta):
            k = k / np.linalg.norm(k)
            return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))
        
        v = P1 - P0
        z = np.array([0, 0, 1])
        k_theta = np.cross(v, z)
        k_theta = k_theta / np.linalg.norm(k_theta) if np.linalg.norm(k_theta) > 1e-9 else np.array([0., 1., 0.])
        
        P1_prime = P0 + rodrigues_rotation(v, k_theta, theta)
        catenary_rotated_theta = compute_catenary(P0, P1_prime)

        rotated_catenary = VMobject()
        rotated_catenary.set_points_smoothly(to_manim_points(catenary_rotated_theta))
        rotated_catenary.set_color(RED)
        P1_prime_dot = Dot3D(point=P1_prime, color=YELLOW, radius=0.1)
        
        # Step 2: Rotate around Theta
        self.play(Transform(standard_catenary, rotated_catenary), FadeIn(P1_prime_dot))
        self.wait(2)

        # Step 3: Inverse rotation around Theta
        catenary_transformed = np.array([P0 + rodrigues_rotation(q - P0, k_theta, -theta) for q in catenary_rotated_theta])
        catenary_transformed_vm = VMobject().set_points_smoothly(to_manim_points(catenary_transformed))
        catenary_transformed_vm.set_color(GREEN)

        self.play(Transform(rotated_catenary, catenary_transformed_vm), FadeIn(P1_prime_dot))
        self.wait(2)

        # Step 4: Apply Gamma rotation around P0P1 vector
        k_gamma = P1 - P0
        k_gamma = k_gamma / np.linalg.norm(k_gamma)

        catenary_rotated_gamma = np.array([P0 + rodrigues_rotation(q - P0, k_gamma, gamma) for q in catenary_transformed])
        gamma_catenary = VMobject().set_points_smoothly(to_manim_points(catenary_rotated_gamma))
        gamma_catenary.set_color(GREEN)

        self.play(Transform(catenary_transformed_vm, gamma_catenary))
        self.wait(2)
        
        # Final scaling transformation
        endpoint_rotated = catenary_rotated_gamma[-1]
        original_vector = P1 - P0
        rotated_vector = endpoint_rotated - P0
        original_length = np.linalg.norm(original_vector)
        rotated_length = np.linalg.norm(rotated_vector)
        scaling_factor = original_length / rotated_length if rotated_length > 1e-9 else 1.0
        
        catenary_final = np.array([P0 + (q - P0) * scaling_factor for q in catenary_rotated_gamma])
        final_catenary_vm = VMobject().set_points_smoothly(to_manim_points(catenary_final))
        final_catenary_vm.set_color("#FF00FF")

        self.play(Transform(gamma_catenary, final_catenary_vm))
        self.wait(3)
        
        self.play(FadeOut(standard_catenary, rotated_catenary, gamma_catenary, final_catenary_vm, P0_dot, P1_dot, P1_prime_dot, axes))
        self.wait(1)

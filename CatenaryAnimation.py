from manim import *
import numpy as np
from pympc.models.catenary import Catenary

# Rodrigues' Rotation

def rodrigues_rotation(v, k, theta):
    k = k / np.linalg.norm(k)
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))


class AugmentedCatenaryScene(ThreeDScene):
    def construct(self):
        # Parameters
        catenary = Catenary(length=3., reference_frame='ENU')
        P0 = np.array([0., 0., 0.])
        P1 = np.array([1., 1., 0.])
        theta = np.radians(-30)
        gamma = np.radians(20)

        def compute_catenary(start, end):
            out = catenary(start, end)
            if out[3] is not None:
                return out[3]
            return np.array([start, end])

        def to_manim_points(points):
            return [np.array([p[0], p[1], p[2]]) for p in points]

        def label_point(label, point, direction=UP):
            return Text(label).scale(0.4).next_to(Dot3D(point=point), direction)

        def show_label(curve, text):
            label = Text(text).scale(0.4).next_to(curve, UP)
            label.rotate(PI/2, axis=UP)
            label.rotate(-PI, axis=RIGHT)
            return label

        self.set_camera_orientation(phi=70 * DEGREES, theta=15 * DEGREES, zoom=1.3)

        axes = ThreeDAxes()
        self.play(Create(axes))

        # Step 1: Original Catenary
        cat_original = compute_catenary(P0, P1)
        curve_orig = VMobject().set_points_smoothly(to_manim_points(cat_original)).set_color(BLUE)
        P0_dot = Dot3D(point=P0, color=RED, radius=0.07)
        P1_dot = Dot3D(point=P1, color=GREEN, radius=0.07)
        P0_label = label_point("P0", P0)
        P1_label = label_point("P1", P1)
        label_orig = show_label(curve_orig, "Original Catenary")

        self.play(Create(curve_orig), FadeIn(P0_dot, P1_dot, P0_label, P1_label, label_orig))
        self.wait(2)

        # Step 2: Theta Rotation
        vec = P1 - P0
        vec_proj = vec.copy(); vec_proj[2] = 0
        if np.linalg.norm(vec_proj) < 1e-9:
            vec_proj = np.array([1., 0., 0.])
        else:
            vec_proj /= np.linalg.norm(vec_proj)
        z = np.array([0, 0, 1])
        k_theta = np.cross(vec_proj, z)
        k_theta = k_theta / np.linalg.norm(k_theta)

        P1_prime = P0 + rodrigues_rotation(vec, k_theta, theta)
        cat_theta = compute_catenary(P0, P1_prime)
        curve_theta = VMobject().set_points_smoothly(to_manim_points(cat_theta)).set_color(RED)
        label_theta = show_label(curve_theta, "Rotated (Theta)")
        P1p_dot = Dot3D(P1_prime, color=YELLOW, radius=0.07)
        P1p_label = label_point("P1'", P1_prime)
        theta_tex = Text("Theta = -30 deg").scale(0.4).next_to(P0_dot, UP, buff=0.7)

        self.play(Write(theta_tex))
        self.play(FadeOut(label_orig, theta_tex), Transform(curve_orig, curve_theta), FadeIn(P1p_dot, P1p_label, label_theta))
        self.wait(2)

        # Step 3: Align back (Inverse Theta)
        cat_theta_back = np.array([P0 + rodrigues_rotation(p - P0, k_theta, -theta) for p in cat_theta])
        curve_back = VMobject().set_points_smoothly(to_manim_points(cat_theta_back)).set_color(GREEN)
        label_back = show_label(curve_back, "Theta Corrected")

        self.play(FadeOut(P1p_dot, P1p_label, label_theta), Transform(curve_orig, curve_back), FadeIn(label_back))
        self.wait(2)

        # Step 4: Gamma Rotation
        k_gamma = vec / np.linalg.norm(vec)
        cat_gamma = np.array([P0 + rodrigues_rotation(p - P0, k_gamma, gamma) for p in cat_theta_back])
        curve_gamma = VMobject().set_points_smoothly(to_manim_points(cat_gamma)).set_color(ORANGE)
        label_gamma = show_label(curve_gamma, "Gamma Rotated")
        gamma_tex = Text("Gamma = 20 deg").scale(0.4).next_to(P0_dot, RIGHT, buff=0.9)

        self.play(Write(gamma_tex))
        self.play(FadeOut(label_back), Transform(curve_orig, curve_gamma), FadeIn(label_gamma))
        self.wait(2)

        # Step 5: Scaling to match P1
        rot_end = cat_gamma[-1]
        scale = np.linalg.norm(P1 - P0) / np.linalg.norm(rot_end - P0)
        cat_final = np.array([P0 + (q - P0) * scale for q in cat_gamma])
        curve_final = VMobject().set_points_smoothly(to_manim_points(cat_final)).set_color(PURPLE)
        label_final = show_label(curve_final, "Full Augmented (Gamma)")

        self.play(Transform(curve_orig, curve_final), FadeIn(label_final))
        self.wait(3)

        # self.play(FadeOut(curve_orig, P0_dot, P1_dot, P0_label, P1_label, label_final, label_gamma, gamma_tex))
        self.play(*[FadeOut(mob) for mob in [curve_orig, curve_final, P0_dot, P1_dot, P0_label, P1_label, label_final] if mob is not None])
        self.wait(1)
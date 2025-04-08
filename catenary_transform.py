import numpy as np
from typing import Callable, Tuple

def rodrigues_rotation(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rotate a vector around a given axis using Rodrigues' rotation formula.

    Parameters:
    - vector (np.ndarray): The 3D vector to rotate.
    - axis (np.ndarray): The axis (unit vector) to rotate around.
    - angle_rad (float): Rotation angle in radians.

    Returns:
    - np.ndarray: Rotated vector.
    """
    axis = axis / np.linalg.norm(axis)  # Ensure axis is a unit vector
    return (
        vector * np.cos(angle_rad)
        + np.cross(axis, vector) * np.sin(angle_rad)
        + axis * np.dot(axis, vector) * (1 - np.cos(angle_rad))
    )


def transform_catenary(
    point_A: np.ndarray,
    point_B: np.ndarray,
    catenary_fn: Callable[[np.ndarray, np.ndarray], Tuple],
    theta_rad: float,
    gamma_rad: float
) -> dict:
    """
    Apply Theta and Gamma rotation transformations to a catenary between two 3D points.

    Parameters:
    - point_A (np.ndarray): The fixed point (start of the catenary), shape (3,).
    - point_B (np.ndarray): The movable point (end of the catenary), shape (3,).
    - catenary_fn (Callable): A function that takes (start, end) and returns a tuple,
                              where the 4th element (index 3) contains the catenary points.
    - theta_rad (float): Rotation angle (in radians) about a horizontal axis (Theta).
    - gamma_rad (float): Rotation angle (in radians) about the vector from point_A to point_B (Gamma).

    Returns:
    - dict: A dictionary with the following keys:
        - 'original': Catenary between point_A and point_B with no transformation.
        - 'theta_rotated': Catenary after Theta rotation.
        - 'theta_aligned': Theta-rotated catenary aligned back to original vector.
        - 'final': Final catenary after applying Gamma rotation.
    """

    def compute_catenary(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Extract the actual catenary points from the catenary function output."""
        output = catenary_fn(start, end)
        if output[3] is not None:
            return output[3]
        return np.array([start, end])

    # Step 1: Compute unmodified (original) catenary
    original_catenary = compute_catenary(point_A, point_B)

    # Step 2: Define rotation axis for Theta (horizontal axis perpendicular to XY projection)
    connection_vector = point_B - point_A
    xy_projection = connection_vector.copy()
    xy_projection[2] = 0

    if np.linalg.norm(xy_projection) < 1e-9:
        xy_projection = np.array([1., 0., 0.])
    else:
        xy_projection /= np.linalg.norm(xy_projection)

    z_axis = np.array([0, 0, 1])
    theta_axis = np.cross(xy_projection, z_axis)
    if np.linalg.norm(theta_axis) < 1e-9:
        theta_axis = np.array([0., 1., 0.])
    else:
        theta_axis /= np.linalg.norm(theta_axis)

    # Step 3: Apply Theta rotation to the connection vector and get new B'
    rotated_B = point_A + rodrigues_rotation(connection_vector, theta_axis, theta_rad)
    theta_rotated_catenary = compute_catenary(point_A, rotated_B)

    # Step 4: Inversely rotate points to align back to original vector
    theta_aligned_catenary = np.array([
        point_A + rodrigues_rotation(pt - point_A, theta_axis, -theta_rad)
        for pt in theta_rotated_catenary
    ])

    # Step 5: Define Gamma rotation axis as the normalized vector from A to B
    gamma_axis = point_B - point_A
    gamma_axis /= np.linalg.norm(gamma_axis)

    # Step 6: Apply Gamma rotation to each point
    final_catenary = np.array([
        point_A + rodrigues_rotation(pt - point_A, gamma_axis, gamma_rad)
        for pt in theta_aligned_catenary
    ])

    return original_catenary, theta_rotated_catenary, theta_aligned_catenary, final_catenary

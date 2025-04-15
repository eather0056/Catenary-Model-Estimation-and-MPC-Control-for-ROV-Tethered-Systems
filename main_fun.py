import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score
import numpy as np
from typing import Callable, Tuple
from scipy.interpolate import interp1d




# ================================================================================================================
# === Function For catenay Computation ===
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


# =======================================================================================================
# === Extract Features from dataset, functions for PySR ===
# Function to extract features from the dataset for training and testing -- main
# ==========================================================================================================
# === Load & Preprocess for Training===
# Set uniform time step
import wandb
import threading
import time
import os
import io
from PIL import Image

UNIFORM_DT = 0.05
def uniform_resample(df, dt=UNIFORM_DT):
    """Resample DataFrame to a uniform time step using vectorized interpolation."""
    if "Time" not in df.columns:
        raise ValueError("Missing 'Time' column in dataset.")
    
    time_orig = df["Time"].values
    time_uniform = np.arange(time_orig[0], time_orig[-1], dt)

    # Build new columns with interpolation
    data_dict = {"Time": time_uniform}
    for col in df.columns:
        if col != "Time":
            f = interp1d(time_orig, df[col].values, kind='linear', bounds_error=False, fill_value="extrapolate")
            data_dict[col] = f(time_uniform)

    # Create resampled DataFrame at once (avoids fragmentation)
    df_resampled = pd.DataFrame(data_dict)
    return df_resampled


def load_and_resample_all(file_list, dt=UNIFORM_DT):
    """Load multiple CSVs and resample them independently before merging."""
    dfs = []
    for f in file_list:
        df = pd.read_csv(f)
        if {"Theta", "Gamma", "Time"}.issubset(df.columns):
            df_clean = df.dropna(subset=["Theta", "Gamma", "Time"])
            df_resampled = uniform_resample(df_clean, dt)
            dfs.append(df_resampled)
        else:
            print(f"[WARNING] Missing columns in {f}, skipping.")
    return pd.concat(dfs, ignore_index=True)

# === Load All Datasets ===
def load_and_concat(files):
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df.dropna(subset=['Theta', 'Gamma'])

def extract_features(df):
    P0 = df[["rod_end X", "rod_end Y", "rod_end Z"]].values / 1000
    P1 = df[["robot_cable_attach_point X", "robot_cable_attach_point Y", "robot_cable_attach_point Z"]].values / 1000
    V1 = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values
    time = df["Time"].values

    acc_x = np.gradient(df["rob_cor_speed X"].values, time)
    acc_y = np.gradient(df["rob_cor_speed Y"].values, time)
    acc_z = np.gradient(df["rob_cor_speed Z"].values, time)
    A1 = np.stack([acc_x, acc_y, acc_z], axis=1)

    rel_vec = P1 - P0
    unit_rel = rel_vec / (np.linalg.norm(rel_vec, axis=1, keepdims=True) + 1e-8)
    tension = np.clip(np.linalg.norm(rel_vec, axis=1, keepdims=True), 1e-5, 10)

    dot_product = np.sum(V1 * unit_rel, axis=1, keepdims=True)
    norm_v1 = np.linalg.norm(V1, axis=1, keepdims=True) + 1e-8
    angle_proj = np.clip(dot_product / norm_v1, -1, 1)

    theta = df["Theta"].values.reshape(-1, 1)
    gamma = df["Gamma"].values.reshape(-1, 1)
    theta_prev = np.roll(theta, 1)
    gamma_prev = np.roll(gamma, 1)
    theta_prev[0] = theta[0]
    gamma_prev[0] = gamma[0]

    return np.hstack([P1, V1, A1, unit_rel, tension, angle_proj, theta, gamma, theta_prev, gamma_prev])

# === Derivatives ===
def compute_derivatives(df):
    time_array = df["Time"].values
    theta = df["Theta"].values
    gamma = df["Gamma"].values
    dtheta = np.gradient(theta, time_array)
    dgamma = np.gradient(gamma, time_array)
    return dtheta, dgamma

# === Background Logger for Training Progress ===
def log_pysr_progress(model, label, total_iters, interval=60):
    def _loop():
        while not getattr(model, "_finished", False):
            try:
                results = getattr(model, "equations_", None) or getattr(model, "equation_search_results", None)
                if results is not None and not results.empty:
                    best = results.loc[results["loss"].idxmin()]
                    progress = min(len(results) / total_iters * 100, 100)
                    wandb.log({
                        f"{label}/progress_percent": progress,
                        f"{label}/current_best_loss": best["loss"],
                        f"{label}/current_best_complexity": best["complexity"],
                        f"{label}/expressions_evaluated": len(results),
                        f"{label}/hall_of_fame_top": str(best["equation"]),
                    })
            except Exception as e:
                print(f"[Progress Log] {label} failed to log: {e}")
            time.sleep(interval)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()

# === Evaluation ===
def log_scatter_plot(actual, pred, label, output_dir):
    fig, ax = plt.subplots()
    ax.scatter(actual, pred, alpha=0.4)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    ax.set_title(f"{label}: Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    image = Image.open(buf)
    wandb.log({f"{label}_scatter": wandb.Image(image, caption=label)})

    plt.savefig(os.path.join(output_dir, f"{label}_scatter.png"))
    plt.close()

# === Convergence Plot ===
def log_convergence_plot(model, label, output_dir):
    # Get either equations_ or equation_search_results
    res = getattr(model, "equations_", None)
    if res is None or (hasattr(res, "empty") and res.empty):
        res = getattr(model, "equation_search_results", None)

    if res is None or (hasattr(res, "empty") and res.empty):
        print(f"[WARN] No convergence results found for {label}")
        return
    
    best = res.loc[res["loss"].idxmin()]
    plt.figure(figsize=(10, 6))
    plt.scatter(res["complexity"], res["loss"], alpha=0.4)
    plt.scatter(best["complexity"], best["loss"], color="red", label="Best")
    plt.xlabel("Complexity")
    plt.ylabel("Loss")
    plt.title(f"{label} Convergence")
    plt.grid()
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # FIX: Convert to PIL image
    image = Image.open(buf)
    wandb.log({f"{label}_convergence": wandb.Image(image, caption=f"{label} Convergence")})
    plt.savefig(os.path.join(output_dir, f"{label}_convergence.png"))
    plt.close()


# ==================================================================================================
# === Function for plot of model validation Simulated function to demonstrate integration-based evaluation ===
# Script to integrate theta and gamma using the trained models -- model_test
# ==================================================================================================
def integrate_theta_gamma(model_theta, model_gamma, X, time_array, theta_0, gamma_0):
    """
    Integrate symbolic dTheta/dt and dGamma/dt over time using Euler's method.
    
    Parameters:
    - model_theta: trained PySR model for dTheta/dt
    - model_gamma: trained PySR model for dGamma/dt
    - X: feature matrix (n_samples, n_features)
    - time_array: time values corresponding to X (n_samples,)
    - theta_0: initial value of Theta
    - gamma_0: initial value of Gamma

    Returns:
    - theta_pred: integrated Theta over time
    - gamma_pred: integrated Gamma over time
    """
    n = len(time_array)
    theta_pred = np.zeros(n)
    gamma_pred = np.zeros(n)
    theta_pred[0] = theta_0
    gamma_pred[0] = gamma_0

    for i in range(1, n):
        dt = time_array[i] - time_array[i - 1]
        dtheta_dt = model_theta.predict(X[i - 1:i])[0]
        dgamma_dt = model_gamma.predict(X[i - 1:i])[0]
        theta_pred[i] = theta_pred[i - 1] + dtheta_dt * dt
        gamma_pred[i] = gamma_pred[i - 1] + dgamma_dt * dt

    return theta_pred, gamma_pred


# === Helper function to smooth and uniformly sample time if needed ===
def preprocess_signals(df, sigma=2):
    """
    Smooth and validate time-series data.
    Returns time, smoothed Theta, smoothed Gamma.
    """
    time_array = df["Time"].values
    theta = gaussian_filter1d(df["Theta"].values, sigma=sigma)
    gamma = gaussian_filter1d(df["Gamma"].values, sigma=sigma)
    return time_array, theta, gamma

# === Visualization function ===
def plot_integration(time_array, theta_true, theta_pred, gamma_true, gamma_pred):
    theta_error = theta_pred - theta_true
    gamma_error = gamma_pred - gamma_true

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(time_array, theta_true, label="Theta True")
    axs[0].plot(time_array, theta_pred, '--', label="Theta Integrated")
    axs[0].set_ylabel("Theta (rad)")
    axs[0].legend()
    axs[0].set_title("Integrated Theta(t) vs. True")
    axs[0].grid()

    axs[1].plot(time_array, gamma_true, label="Gamma True")
    axs[1].plot(time_array, gamma_pred, '--', label="Gamma Integrated")
    axs[1].set_ylabel("Gamma (rad)")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()
    axs[1].set_title("Integrated Gamma(t) vs. True")
    axs[1].grid()

    axs[2].plot(time_array, theta_error, label="Theta Error", color="purple")
    # axs[2].plot(time_array, gamma_error, label="Gamma Error", color="orange")
    axs[2].set_ylabel("Error (rad)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()
    axs[2].set_title("Estimation Error")
    axs[2].grid()

    plt.tight_layout()
    plt.show()


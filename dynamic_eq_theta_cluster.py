import os
import joblib
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import glob
import io
import time
import threading
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
import wandb
from packaging import version
import pysr
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from collections import Counter
import re
from main_fun import *
from collections import defaultdict

# === Define The Run Name ===
# This is the name of the run that will be used in W&B and the output directory.
Run_Name = "TG_C6_al_50_F"

# === Set the timestamp for the run ===
# This will be used to create unique filenames for the output files.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["JULIA_DEBUG"] = "all"

# ===== Physics-Compliant Operators =====
unary_ops = [
    "square",    # Quadratic drag terms (x²)
    # "tanh",      # Angle saturation
    "neg",       # Sign inversion (implicit in subtraction)
    "cos",      # Angle projection
    "sin",      # Angle projection
    # "abs",       # Absolute value for tension
    # "cube",      # Cubic terms (x³)
]

# Cable parameters from experimental setup (Table I for cable 6)
L = 3.0  # [m]
cable_wet_weight = 1.521  # [N] (From Table I, cable 6: wet weight=1.521N)

wandb.init(
    project="Catenary_Dynamics_Differential",
    entity="eather0056",
    name=f"{Run_Name}_{timestamp}",
    tags=["symbolic", "dynamics", "nonlinear", "hydrodynamics"],
    notes="Augmented catenary dynamics discovery with physics-informed operator constraints. Features: theta (v_surge, v_sway², tension), gamma (v_sway, v_cross, delta_H). Cable 6 dataset.",
    config={
        "model": "PySR",
        "task": "Damped Catenary Dynamics",
        "niterations": 50,
        "binary_operators": ["+", "-", "*", "/"],  # Protected division
        "unary_operators": unary_ops,
        "loss": "loss(x, y) = (x - y)^2 + 0.01 * abs(x)",
        "constraints": {
            "/": (3, 2),  # Max depth 3, denominator terms ≤2
            "exp": 1,      # Max 1 exponential per expression
        },
        "nested_constraints": {
            "tanh": {"tanh": 0},  # No nested tanh
            "exp": {"exp": 0}      # No nested exponentials
        },
        "batching": True,
        "batch_size": 10000,
        "random_state": 42,
        "maxsize": 25,
        "procs": 0,
        "verbosity": 2,
        "deterministic": True,
        "model_selection": "accuracy",
    }
)

config = wandb.config

common_params = dict(
    niterations=config["niterations"],
    binary_operators=config["binary_operators"],
    unary_operators=config["unary_operators"],
    loss=config["loss"],
    # constraints=config["constraints"],
    # nested_constraints=config["nested_constraints"],
    batching=config["batching"],
    batch_size=config["batch_size"],
    random_state=config["random_state"],
    deterministic=config["deterministic"],
    procs=config["procs"],
    maxsize=config["maxsize"],
    )

# === Set up output directory ===
# This will be used to save the output files and models.
output_dir = f"outputs/{Run_Name}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)


# all_files = sorted(glob.glob("/home/mundus/mdeowan698/Catenary_Dynamic/Data/*.csv"))
# train_files, test_files = train_test_split(all_files, test_size=0.9, random_state=42)


# === Load and Combine Training Datasets ===
train_files = [
    "Data/L_dynamique6x100dis2_0033.csv",  
    "Data/L_dynamique6x100dis2_0034.csv",  
    "Data/L_dynamique6x100dis2_0035.csv",  
    "Data/L_dynamique6x200dis2_0030.csv",  
    "Data/L_dynamique6x200dis2_0031.csv",  
    "Data/L_dynamique6x200dis2_0032.csv",  
    "Data/L_dynamique6y100dis1_0018.csv",  
    "Data/L_dynamique6y100dis1_0019.csv",  
    "Data/L_dynamique6y100dis1_0020.csv",  
    "Data/L_dynamique6y100dis2_0021.csv",  
    "Data/L_dynamique6y100dis2_0022.csv",  
    "Data/L_dynamique6y100dis2_0023.csv",  
    "Data/L_dynamique6y200dis1_0025.csv",  
    "Data/L_dynamique6y200dis1_0026.csv",  
    "Data/L_dynamique6y200dis2_0027.csv",  
    "Data/L_dynamique6y200dis2_0028.csv",  
    "Data/L_dynamique6y200dis2_0029.csv"  
]

# === Test on New Dataset ===
test_files = ["Data/L_dynamique6y200dis1_0024.csv"]

# === Load and Resample All Datasets or load_and_concat===
# df_train = load_and_resample_all(train_files)
# df_test = load_and_resample_all(test_files)

df_train = load_and_concat(train_files)
df_test = load_and_concat(test_files)


# Step 1: Build theta and gamma features separately
X_train_theta = build_theta_features_valid(df_train, L, cable_wet_weight)    # for theta
X_train_gamma = build_gamma_features_valid(df_train, L, cable_wet_weight)    # for gamma

# Step 2: Compute targets
theta_dot, gamma_dot = compute_derivatives(df_train)

# === Step 1: Rescale gamma_dot and theta_dot ===
gamma_mean, gamma_std = gamma_dot.mean(), gamma_dot.std()
theta_mean, theta_std = theta_dot.mean(), theta_dot.std()

y_dgamma_dt_train = (gamma_dot - gamma_mean) / gamma_std
y_dtheta_dt_train = (theta_dot - theta_mean) / theta_std

# Step 3: Scale them separately
scaler_theta = StandardScaler()
scaler_gamma = StandardScaler()

X_train_theta_scaled = scaler_theta.fit_transform(X_train_theta)
X_train_gamma_scaled = scaler_gamma.fit_transform(X_train_gamma)


model_dtheta_dt = PySRRegressor(**common_params)
model_dgamma_dt = PySRRegressor(**common_params)

# Save current working directory
original_cwd = os.getcwd()

# Create PySR work folders
dtheta_out = os.path.join(output_dir, "dtheta_dt")
dgamma_out = os.path.join(output_dir, "dgamma_dt")
os.makedirs(dtheta_out, exist_ok=True)
os.makedirs(dgamma_out, exist_ok=True)

# === Train dTheta/dt ===
os.chdir(dtheta_out)
log_pysr_progress(model_dtheta_dt, "dTheta_dt", wandb.config["niterations"])
print("Training model for dTheta/dt...")
model_dtheta_dt.fit(
    X_train_theta,
    y_dtheta_dt_train,
    variable_names=[
        "theta_v_surge_l",      # [rad/s]
        "v_surge_l",            # [1/s]
        "theta_v_surge",        # [rad·m/s]
        "v_surge",              # [m/s]
        "v_surge_sq_l",         # [m/s²]
        "T_l",                  # [kg/s²]
        "sin_theta",            # [dimensionless]
        "delta_H_l",            # [dimensionless]
        "torque" ,            # [rad/s²]
    ]
)

model_dtheta_dt._finished = True
os.chdir(original_cwd)

# === Train dGamma/dt ===
os.chdir(dgamma_out)
log_pysr_progress(model_dgamma_dt, "dGamma_dt", wandb.config["niterations"])
print("Training model for dGamma/dt...")
model_dgamma_dt.fit(
    X_train_gamma,
    y_dgamma_dt_train,
    variable_names=[
        "gamma_v_sway_l",      # [rad/s]
        "v_sway_l",            # [1/s]
        "gamma_v_sway",        # [rad·m/s]
        "v_sway",              # [m/s]
        "v_sway_sq_l",         # [m/s²]
        "T_l",                  # [kg/s²]
        "sin_gamma",            # [dimensionless]
        "torque" ,            # [rad/s²]
    ]
)
model_dgamma_dt._finished = True
os.chdir(original_cwd)

# === Save Outputs ===
joblib.dump(model_dtheta_dt, os.path.join(output_dir, f"model_dtheta_dt.pkl"))
joblib.dump(model_dgamma_dt, os.path.join(output_dir, f"model_dgamma_dt.pkl"))

with open(os.path.join(output_dir, f"eq_dtheta_dt.txt"), "w") as f:
    f.write(str(model_dtheta_dt.get_best()))
with open(os.path.join(output_dir, f"eq_dgamma_dt.txt"), "w") as f:
    f.write(str(model_dgamma_dt.get_best()))

model_dtheta_dt.equations_.to_csv(os.path.join(output_dir, f"dtheta_results.csv"), index=False)
model_dgamma_dt.equations_.to_csv(os.path.join(output_dir, f"dgamma_results.csv"), index=False)

# === Save Scaler ===
joblib.dump(scaler_theta, os.path.join(output_dir, f"scaler_theta.pkl"))
joblib.dump(scaler_gamma, os.path.join(output_dir, f"scaler_gamma.pkl"))


# === Save Predictions for Inspection ===
# X_test = extract_features(df_test)
X_test_theta = build_theta_features_valid(df_test, L, cable_wet_weight)
X_test_gamma = build_gamma_features_valid(df_test, L, cable_wet_weight)
# X_scaled = scaler.transform(X_test)

X_test_theta_scaled = scaler_theta.fit_transform(X_test_theta)
X_test_gamma_scaled = scaler_gamma.fit_transform(X_test_gamma)

time = df_test["Time"].values

# === Predict Derivatives ===
dtheta_pred = model_dtheta_dt.predict(X_test_theta_scaled) * theta_std + theta_mean
dgamma_pred = model_dgamma_dt.predict(X_test_gamma_scaled) * gamma_std + gamma_mean

# === Integrate to Estimate Theta and Gamma ===
theta_est = np.zeros_like(dtheta_pred)
gamma_est = np.zeros_like(dgamma_pred)
theta_est[0] = df_test["Theta"].values[0]
gamma_est[0] = df_test["Gamma"].values[0]

for i in range(1, len(time)):
    dt = time[i] - time[i - 1]
    theta_est[i] = theta_est[i - 1] + dtheta_pred[i - 1] * dt
    gamma_est[i] = gamma_est[i - 1] + dgamma_pred[i - 1] * dt

theta_true = df_test["Theta"].values
gamma_true = df_test["Gamma"].values
theta_error = theta_true - theta_est
gamma_error = gamma_true - gamma_est

log_scatter_plot(theta_true, theta_est, "dTheta_dt", output_dir)
log_scatter_plot(gamma_true, gamma_est, "dGamma_dt", output_dir)

log_convergence_plot(model_dtheta_dt, "dTheta_dt", output_dir)
log_convergence_plot(model_dgamma_dt, "dGamma_dt", output_dir)

eq_str = str(model_dtheta_dt.get_best()["equation"])
used_features = Counter(re.findall(r"x\d+", eq_str))
wandb.log({f"feature_usage_dtheta_dt/{k}": v for k, v in used_features.items()})

wandb.log({
    "Number of Features For Theta": X_train_theta.shape[1],
    "Number of Training Samples for theta": len(X_train_theta),
    "Number of Features for Gamma": X_train_gamma.shape[1],
    "Number of Training Samples for Gamma": len(X_train_gamma),
    "eq_dtheta_dt_final": str(model_dtheta_dt.get_best()),
    "eq_dgamma_dt_final": str(model_dgamma_dt.get_best()),
    "r2_score_dtheta_dt": r2_score(theta_true, theta_est),
    "r2_score_dgamma_dt": r2_score(gamma_true, gamma_est),
    "dTheta_dt_best_complexity": model_dtheta_dt.get_best()["complexity"],
    "dGamma_dt_best_complexity": model_dgamma_dt.get_best()["complexity"],
    "dTheta_dt_error_hist": wandb.Histogram(theta_error),
    "dGamma_dt_error_hist": wandb.Histogram(gamma_error),
    "dTheta_dt_error_mean": np.mean(theta_error),
    "dTheta_dt_error_std": np.std(theta_error),
    "dGamma_dt_error_mean": np.mean(gamma_error),
    "dGamma_dt_error_std": np.std(gamma_error),
    "dTheta_dt_error_max": np.max(theta_error),
    "dTheta_dt_error_min": np.min(theta_error),
    "dGamma_dt_error_max": np.max(gamma_error),
    "dGamma_dt_error_min": np.min(gamma_error),
    "dTheta_dt_error_median": np.median(theta_error),
    "dGamma_dt_error_median": np.median(gamma_error),
})


# === Save to W&B as Artifact ===
artifact = wandb.Artifact(f"dynamics_models_{Run_Name}", type="model")
artifact.add_dir(output_dir)
artifact.add_dir(os.path.join(output_dir, "dtheta_dt"))
artifact.add_dir(os.path.join(output_dir, "dgamma_dt"))
wandb.log_artifact(artifact)

wandb.finish()

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
Run_Name = "C6_6_FF_1k_x_cons"

# === Set the timestamp for the run ===
# This will be used to create unique filenames for the output files.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["JULIA_DEBUG"] = "all"

unary_ops = ["sin", "cos", "abs", "square", "tanh"]

feature_names = [
    "P0_x", "P0_y", "P0_z",           # [m]
    "P1_x", "P1_y", "P1_z",           # [m]
    "V1_x", "V1_y", "V1_z",           # [m/s]
    "A1_x", "A1_y", "A1_z",           # [m/s²]
    "V1_unit_x", "V1_unit_y", "V1_unit_z",  # unitless
    "A1_unit_x", "A1_unit_y", "A1_unit_z",  # unitless
    "unit_rel_x", "unit_rel_y", "unit_rel_z",  # unitless
    "tension",                        # [m]
    "speed_norm",                    # [m/s]
    "acc_norm",                      # [m/s²]
    "angle_proj",                    # unitless
    "dot_VA",                        # [m²/s³]
    "cross_VA_x", "cross_VA_y", "cross_VA_z",  # [m²/s³]
    "theta",                         # [rad]
    "gamma"                          # [rad]
]

# === Define the units for each feature ===
# This is a list of units corresponding to the feature names.
feature_units = [
    "m", "m", "m",         # P1
    "m", "m", "m",         # P1
    "m/s", "m/s", "m/s",             # V1
    "m/s²", "m/s²", "m/s²",          # A1
    "unitless", "unitless", "unitless",  # V1_unit
    "unitless", "unitless", "unitless",  # A1_unit
    "unitless", "unitless", "unitless",  # unit_rel
    "m",                             # tension
    "m/s",                           # speed_norm
    "m/s²",                          # acc_norm
    "unitless",                      # angle_proj
    "m²/s³",                         # dot_VA
    "m²/s³", "m²/s³", "m²/s³",       # cross_VA
    "rad",                           # theta
    "rad"                            # gamma
]

constraints = {
    "add": {"arity": 2},
    "sub": {"arity": 2},
    "mul": {"arity": 2},
    "div": {"arity": 2},
    "sqrt": {"arity": 1},
    "tanh": {"arity": 1},
    "square": {"arity": 1},
    "sin": {"arity": 1},
    "cos": {"arity": 1},
    "log": {"arity": 1},
    "exp": {"arity": 1}
}

# Map unit type → list of feature indices
unit_to_indices = defaultdict(list)
for i, u in enumerate(feature_units):
    unit_to_indices[u].append(i)

wandb.init(
    project="Catenary_Dynamics_Differential",
    entity="eather0056",
    name=f"{Run_Name}_{timestamp}",
    tags=["symbolic", "dynamics", "nonlinear"],
    notes="Tanning for theta/gamma symbolic equations, 2 cable 6 dataset used load type concatenate, feature used P0, # [m] P1, # [m] V1, # [m/s] A1, # [m/s²] V1_unit, # unitless A1_unit, # unitless unit_rel, # unitless tension, # [m] speed_norm, # [m/s] acc_norm, # [m/s²] angle_proj, # unitless dot_VA, # [m²/s³] (proxy for work/power-type effects) cross_VA, # [m²/s³] theta, # [rad] gamma # [rad].",
    config={
        "model": "PySR",
        "task": "Differential Equation Discovery",
        "niterations": 1000,
        "binary_operators": ["+", "-", "*", "/"],
        # "complexity_of_operators": {"/": 5, "square": 2, "tanh": 3, "sin": 2, "cos": 2},
        "unary_operators": unary_ops,
        "batching": True,
        "batch_size": 5000,
        "random_state": 42,
        "maxsize": 30,
        "procs": 0,
        "verbosity": 1,
        "loss": "loss(x, y) = (x - y)^2 + 0.01 * abs(x)",
        "deterministic": True,
        "model_selection": "best",
        "should_simplify": False,
    }
)

config = wandb.config

common_params = dict(
    niterations=config["niterations"],
    binary_operators=config["binary_operators"],
    unary_operators=config["unary_operators"],
    # complexity_of_operators=config["complexity_of_operators"],
    model_selection=config["model_selection"],
    loss=config["loss"],
    verbosity=config["verbosity"],
    batching=config["batching"],
    batch_size=config["batch_size"],
    random_state=config["random_state"],
    deterministic=config["deterministic"],
    procs=config["procs"],
    maxsize=config["maxsize"],
    should_simplify=config["should_simplify"],
    # feature_names=feature_names,
    select_k_features=None,
    constraints=constraints,
    nested_constraints={
        "add": [unit_to_indices["m/s"], unit_to_indices["m/s"]],
        "sub": [unit_to_indices["rad"], unit_to_indices["rad"]],
        "sin": [unit_to_indices["rad"]],
        "cos": [unit_to_indices["rad"]],
        "tanh": [unit_to_indices["unitless"]],
        "sqrt": [unit_to_indices["unitless"]],
        "log": [unit_to_indices["unitless"]],
        "add": unit_to_indices["rad"],       # allow rad + rad
        "add": unit_to_indices["m"],         # allow m + m
    }

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
    # "Data/L_dynamique6y100dis1_0018.csv",  
    # "Data/L_dynamique6y100dis1_0019.csv",  
    # "Data/L_dynamique6y100dis1_0020.csv",  
    # "Data/L_dynamique6y100dis2_0021.csv",  
    # "Data/L_dynamique6y100dis2_0022.csv",  
    # "Data/L_dynamique6y100dis2_0023.csv",  
    # "Data/L_dynamique6y200dis1_0025.csv",  
    # "Data/L_dynamique6y200dis1_0026.csv",  
    # "Data/L_dynamique6y200dis2_0027.csv",  
    # "Data/L_dynamique6y200dis2_0028.csv",  
    # "Data/L_dynamique6y200dis2_0029.csv"  
]

# === Test on New Dataset ===
test_files = ["Data/L_dynamique6y200dis1_0024.csv"]

# === Load and Resample All Datasets or load_and_concat===
# df_train = load_and_resample_all(train_files)
# df_test = load_and_resample_all(test_files)

df_train = load_and_concat(train_files)
df_test = load_and_concat(test_files)


X_train = extract_features(df_train)
# Filter features for unit consistency
safe_indices = get_unit_safe_indices(feature_units)
X_train = X_train[:, safe_indices]

y_dtheta_dt_train, y_dgamma_dt_train = compute_derivatives(df_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

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
model_dtheta_dt.fit(X_train_scaled, y_dtheta_dt_train, feature_names=feature_names)
model_dtheta_dt._finished = True
os.chdir(original_cwd)

# === Train dGamma/dt ===
os.chdir(dgamma_out)
log_pysr_progress(model_dgamma_dt, "dGamma_dt", wandb.config["niterations"])
print("Training model for dGamma/dt...")
model_dgamma_dt.fit(X_train_scaled, y_dgamma_dt_train, feature_names=feature_names)
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
joblib.dump(scaler, os.path.join(output_dir, f"scaler.pkl"))

# === Save Predictions for Inspection ===
X_test = extract_features(df_test)
X_test = X_test[:, safe_indices]
X_scaled = scaler.transform(X_test)
time = df_test["Time"].values

# === Predict Derivatives ===
dtheta_pred = model_dtheta_dt.predict(X_scaled)
dgamma_pred = model_dgamma_dt.predict(X_scaled)

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

eq_str = str(model_dtheta_dt.get_best()["equation"])
used_features = Counter(re.findall(r"x\d+", eq_str))
wandb.log({f"feature_usage_dtheta_dt/{k}": v for k, v in used_features.items()})
safe_feature_names = [f"x{i} ({feature_units[i]})" for i in safe_indices]
wandb.log({"used_features_unit_safe": safe_feature_names})

wandb.log({
    "Number of Features": X_train.shape[1],
    "Number of Training Samples": len(X_train),
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

log_convergence_plot(model_dtheta_dt, "dTheta_dt", output_dir)
log_convergence_plot(model_dgamma_dt, "dGamma_dt", output_dir)

# === Save to W&B as Artifact ===
artifact = wandb.Artifact(f"dynamics_models_{Run_Name}", type="model")
artifact.add_dir(output_dir)
artifact.add_dir(os.path.join(output_dir, "dtheta_dt"))
artifact.add_dir(os.path.join(output_dir, "dgamma_dt"))
wandb.log_artifact(artifact)

wandb.finish()

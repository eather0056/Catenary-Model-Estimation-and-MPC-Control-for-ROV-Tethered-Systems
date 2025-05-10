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
from scipy.integrate import cumulative_trapezoid
from numpy import cumsum


# === Define The Run Name ===
# This is the name of the run that will be used in W&B and the output directory.
Run_Name = "C6_all_dd_1k_nf"

# === Set the timestamp for the run ===
# This will be used to create unique filenames for the output files.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["JULIA_DEBUG"] = "all"

unary_ops = ["sin", "cos", "abs", "square", "tanh"]

wandb.init(
    project="Catenary_Dynamics_Differential",
    entity="eather0056",
    name=f"{Run_Name}_{timestamp}",
    tags=["symbolic", "dynamics", "nonlinear"],
    notes="Tanning for theta/gamma symbolic equations, all cable 6 dataset used load type concatenate, feature used theta, gama, dtheta, dgamma, v_sway, v_surge, a_sway, a_surge, V_x, V_y, V_z, a_x, a_y, a_z, l.",
    config={
        "model": "PySR",
        "task": "Differential Equation Discovery",
        "niterations": 1000,
        "binary_operators": ["+", "-", "*", "/"],
        # "complexity_of_operators": {"/": 5, "square": 2, "tanh": 3, "sin": 2, "cos": 2},
        "unary_operators": unary_ops,
        "batching": True,
        "batch_size": 10000,
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
    constraints={
        "/": (-1, 1),  # Only allow division where the denominator is 1 term
    },

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


# X_train = extract_features(df_train)
X_train, targets = features_dd(df_train)

# Split into targets for thetä and gammä
target_theta_ddot = targets[:, 0]
target_gamma_ddot = targets[:, 1]

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
model_dtheta_dt.fit(
    X_train_scaled,
    target_theta_ddot,
    variable_names=[
        "theta", "gama", "dtheta", "dgamma", 
        "v_sway", "v_surge", "a_sway", "a_surge",
        "V_x",
        "V_y",
        "V_z",
        "a_x",
        "a_y",
        "a_z",
        "l",
    ],
)
model_dtheta_dt._finished = True
os.chdir(original_cwd)

# === Train dGamma/dt ===
os.chdir(dgamma_out)
log_pysr_progress(model_dgamma_dt, "dGamma_dt", wandb.config["niterations"])
print("Training model for dGamma/dt...")
model_dgamma_dt.fit(
    X_train_scaled,
    target_gamma_ddot,
    variable_names=[
        "theta", "gama", "dtheta", "dgamma", 
        "v_sway", "v_surge", "a_sway", "a_surge",
        "V_x",
        "V_y",
        "V_z",
        "a_x",
        "a_y",
        "a_z",
        "l",
    ],
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
joblib.dump(scaler, os.path.join(output_dir, f"scaler.pkl"))

# === Save Predictions for Inspection ===
# X_test = extract_features(df_test)
X_test, _ = features_dd(df_test)
X_scaled = scaler.transform(X_test)
time = df_test["Time"].values

# === Predict Derivatives ===
ddtheta_pred = model_dtheta_dt.predict(X_scaled)
ddgamma_pred = model_dgamma_dt.predict(X_scaled)

# 1st integration — velocity
dtheta_pred = cumulative_trapezoid(ddtheta_pred, time, initial=0)
dgamma_pred = cumulative_trapezoid(ddgamma_pred, time, initial=0)

# 2nd integration — position using cumulative sum
theta_est = df_test["Theta"].values[0] + cumsum(np.diff(time) * dtheta_pred[1:])
gamma_est = df_test["Gamma"].values[0] + cumsum(np.diff(time) * dgamma_pred[1:])

# Pad to match length
theta_est = np.insert(theta_est, 0, df_test["Theta"].values[0])
gamma_est = np.insert(gamma_est, 0, df_test["Gamma"].values[0])

theta_true = df_test["Theta"].values
gamma_true = df_test["Gamma"].values
theta_error = theta_true - theta_est
gamma_error = gamma_true - gamma_est

log_scatter_plot(theta_true, theta_est, "dTheta_dt", output_dir)
log_scatter_plot(gamma_true, gamma_est, "dGamma_dt", output_dir)

eq_str = str(model_dtheta_dt.get_best()["equation"])
used_features = Counter(re.findall(r"x\d+", eq_str))
wandb.log({f"feature_usage_dtheta_dt/{k}": v for k, v in used_features.items()})

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

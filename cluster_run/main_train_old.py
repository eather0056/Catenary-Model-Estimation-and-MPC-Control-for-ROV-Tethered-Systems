import os
import glob
import io
import time
import threading
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor
import wandb
from packaging import version
import pysr
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from collections import Counter
import re
from scipy.interpolate import interp1d


# === Run Name ===
Run_Name = "C6_2_1KIter_nf_es"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["JULIA_DEBUG"] = "all"

PYSR_VERSION = version.parse(pysr.__version__)
IS_NEW_PYSR = PYSR_VERSION >= version.parse("0.13.0")

# unary_ops = ["sin", "cos", "squre", "tanh"] # ["sin", "cos", "log", "sqrt"]
# if IS_NEW_PYSR:
#     unary_ops += ["safe_log", "safe_sqrt"]


# wandb.init(
#     project="Catenary_Dynamics_Differential",
#     entity="eather0056",
#     name=f"{Run_Name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#     tags=["symbolic", "dynamics", "test"],
#     notes="Tanning for theta/gamma symbolic equations, 0033 cable 6 dataset used with 1K iter 20 feature.",
#     config={
#         "model": "PySR",
#         "task": "Differential Equation Discovery",
#         "niterations": 1000,
#         "binary_operators": ["+", "-", "*", "/"],
#         "unary_operators": unary_ops,
#         "custom_unary_operators": {
#             "safe_log(x)": "log(abs(x) + 1e-5)",
#             "safe_sqrt(x)": "sqrt(abs(x))"
#         } if IS_NEW_PYSR else None,
#         "batching": True,
#         "batch_size": 5000,
#         "random_state": 42,
#         "maxsize": 40,
#         "procs": 0,
#         "verbosity": 1,
#         "loss": "loss(x, y) = (x - y)^2 + 0.01 * abs(x)",
#         "deterministic": True,
#         "multithreading": False,
#         # "parallelism": "serial", # For older versions of PySR
#         "model_selection": "best",
        
#     }
# )

# config = wandb.config

# common_params = dict(
#     niterations=config["niterations"],
#     binary_operators=config["binary_operators"],
#     unary_operators=config["unary_operators"],
#     model_selection=config["model_selection"],
#     complexity_of_operators={"/": 5, "sqrt": 3, "log": 3, "sin": 2, "cos": 2},
#     verbosity=config["verbosity"],
#     batching=config["batching"],
#     batch_size=config["batch_size"],
#     random_state=config["random_state"],
#     maxsize=config["maxsize"],
# )

# # Version-specific logic
# if IS_NEW_PYSR:
#     common_params["elementwise_loss"] = config["loss"]
#     common_params["parallelism"] = "serial"
#     common_params["custom_unary_operators"] = {
#         "safe_log(x)": "log(abs(x) + 1e-5)",
#         "safe_sqrt(x)": "sqrt(abs(x))"
#     }
# else:
#     common_params["loss"] = config["loss"]
#     common_params["deterministic"] = config["deterministic"]
#     common_params["procs"] = config["procs"]
#     common_params["multithreading"] = config["multithreading"]

# === Unary Operators (nonlinear included) ===
unary_ops = ["sin", "cos", "abs", "square", "tanh"]

# === Common Config for Both Local and Cluster (with older PySR) ===
wandb.init(
    project="Catenary_Dynamics_Differential",
    entity="eather0056",
    name=f"{Run_Name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    tags=["symbolic", "dynamics", "nonlinear"],
    notes="Tanning for theta/gamma symbolic equations, 2 cable 6 dataset used with 1K iter with these feature P1, V1, A1, unit_rel, tension, angle_proj, theta, gamma, theta_prev, gamma_prev",
    config={
        "model": "PySR",
        "task": "Differential Equation Discovery",
        "niterations": 1000,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": unary_ops,
        "custom_unary_operators": None,
        "batching": True,
        "batch_size": 5000,
        "random_state": 42,
        "maxsize": 40,
        "procs": 0,
        "verbosity": 1,
        "loss": "loss(x, y) = (x - y)^2 + 0.01 * abs(x)",
        "deterministic": True,
        "multithreading": False,
        "model_selection": "best",
        
    }
)

config = wandb.config

# === PySR Regressor Parameters ===
common_params = dict(
    niterations=config["niterations"],
    binary_operators=config["binary_operators"],
    unary_operators=config["unary_operators"],
    model_selection=config["model_selection"],
    complexity_of_operators={"/": 5, "square": 2, "tanh": 3, "sin": 2, "cos": 2},
    verbosity=config["verbosity"],
    batching=config["batching"],
    batch_size=config["batch_size"],
    random_state=config["random_state"],
    maxsize=config["maxsize"],
    should_simplify=False,
)

if IS_NEW_PYSR:
    common_params["elementwise_loss"] = config["loss"]
    common_params["parallelism"] = "serial"
    common_params["custom_unary_operators"] = config["custom_unary_operators"]
else:
    common_params["loss"] = config["loss"]
    common_params["deterministic"] = config["deterministic"]
    common_params["procs"] = config["procs"]
    common_params["multithreading"] = config["multithreading"]

output_dir = f"outputs/{Run_Name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# === Load & Preprocess ===
# Set uniform time step
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

# all_files = sorted(glob.glob("/home/mundus/mdeowan698/Catenary_Dynamic/Data/*.csv"))
# train_files, test_files = train_test_split(all_files, test_size=0.9, random_state=42)


# === Load and Combine Training Datasets ===
train_files = [
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x100dis2_0033.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x100dis2_0034.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x100dis2_0035.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x200dis2_0030.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x200dis2_0031.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x200dis2_0032.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis1_0018.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis1_0019.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis1_0020.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis2_0021.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis2_0022.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis2_0023.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis1_0025.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis1_0026.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis2_0027.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis2_0028.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis2_0029.csv"  
]

# === Test on New Dataset ===
test_files = ["/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis1_0024.csv"]


df_train = load_and_resample_all(train_files)
df_test = load_and_resample_all(test_files)

# === Features & Derivatives ===
def extract_features(df):
    # Position and velocity
    P0 = df[["rod_end X", "rod_end Y", "rod_end Z"]].values / 1000  # anchor point
    P1 = df[["robot_cable_attach_point X", "robot_cable_attach_point Y", "robot_cable_attach_point Z"]].values / 1000
    V1 = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values

    # Time and acceleration
    time_array = df["Time"].values
    # acc_x = np.gradient(df["rob_cor_speed X"].values, time)
    # acc_y = np.gradient(df["rob_cor_speed Y"].values, time)
    # acc_z = np.gradient(df["rob_cor_speed Z"].values, time)

    acc_x = np.gradient(df["rob_cor_speed X"].values, time_array)
    acc_y = np.gradient(df["rob_cor_speed Y"].values, time_array)
    acc_z = np.gradient(df["rob_cor_speed Z"].values, time_array)
    A1 = np.stack([acc_x, acc_y, acc_z], axis=1)

    # Cable direction unit vector
    rel_vec = P1 - P0
    unit_rel = rel_vec / (np.linalg.norm(rel_vec, axis=1, keepdims=True) + 1e-8)

    # Cable tension proxy: ||P1 - P0||
    tension = np.clip(np.linalg.norm(rel_vec, axis=1, keepdims=True), 1e-5, 10)

    # Angular states
    theta = df["Theta"].values.reshape(-1, 1)
    gamma = df["Gamma"].values.reshape(-1, 1)
    cos_theta = np.cos(theta)
    sin_gamma = np.sin(gamma)

    # Previous timestep angle (autoregressive feature)
    theta_prev = np.roll(theta, 1)
    gamma_prev = np.roll(gamma, 1)
    theta_prev[0] = theta[0]
    gamma_prev[0] = gamma[0]

    # Angle between V1 and cable direction (cosine similarity)
    dot_product = np.sum(V1 * unit_rel, axis=1, keepdims=True)
    norm_v1 = np.linalg.norm(V1, axis=1, keepdims=True) + 1e-8
    angle_proj = dot_product / norm_v1
    angle_proj = np.clip(angle_proj, -1, 1)

    # Include surge/sway classification if known
    # Placeholder: you can replace this with actual metadata or tagging system
    # For now, assume all samples are unknown (0, 0)
    surge_flag = np.zeros((len(df), 1))
    sway_flag = np.zeros((len(df), 1))

    return np.hstack([
        P1, V1, A1, unit_rel,
        tension,
        angle_proj,
        theta, gamma,
        theta_prev, gamma_prev
    ])

# === Derivative Targets ===
def compute_derivatives(df):
    time_array = df["Time"].values
    theta = gaussian_filter1d(df["Theta"].values, sigma=2)
    gamma = gaussian_filter1d(df["Gamma"].values, sigma=2)
    dtheta = np.gradient(theta, time_array)
    dgamma = np.gradient(gamma, time_array)
    return dtheta, dgamma

X_train = extract_features(df_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(extract_features(df_test))
y_dtheta_dt_train, y_dgamma_dt_train = compute_derivatives(df_train)

X_test = extract_features(df_test)
y_dtheta_dt_test, y_dgamma_dt_test = compute_derivatives(df_test)

wandb.config.update({
    "gaussian_filter_sigma": 2,
    "input_features": X_train.shape[1],
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
})


model_dtheta_dt = PySRRegressor(**common_params)
model_dgamma_dt = PySRRegressor(**common_params)

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
model_dtheta_dt.fit(X_train, y_dtheta_dt_train)
model_dtheta_dt._finished = True
os.chdir(original_cwd)

# === Train dGamma/dt ===
os.chdir(dgamma_out)
log_pysr_progress(model_dgamma_dt, "dGamma_dt", wandb.config["niterations"])
print("Training model for dGamma/dt...")
model_dgamma_dt.fit(X_train, y_dgamma_dt_train)
model_dgamma_dt._finished = True
os.chdir(original_cwd)


# === Save Outputs ===
joblib.dump(model_dtheta_dt, os.path.join(output_dir, f"model_dtheta_dt_{timestamp}.pkl"))
joblib.dump(model_dgamma_dt, os.path.join(output_dir, f"model_dgamma_dt_{timestamp}.pkl"))

with open(os.path.join(output_dir, f"eq_dtheta_dt_{timestamp}.txt"), "w") as f:
    f.write(str(model_dtheta_dt.get_best()))
with open(os.path.join(output_dir, f"eq_dgamma_dt_{timestamp}.txt"), "w") as f:
    f.write(str(model_dgamma_dt.get_best()))

model_dtheta_dt.equations_.to_csv(os.path.join(output_dir, f"dtheta_results_{timestamp}.csv"), index=False)
model_dgamma_dt.equations_.to_csv(os.path.join(output_dir, f"dgamma_results_{timestamp}.csv"), index=False)


# === Evaluation ===
def log_scatter_plot(actual, pred, label):
    fig, ax = plt.subplots()
    ax.scatter(actual, pred, alpha=0.4)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    ax.set_title(f"{label}: Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # ✅ Convert buffer to PIL image
    image = Image.open(buf)
    wandb.log({f"{label}_scatter": wandb.Image(image, caption=label)})

    plt.savefig(os.path.join(output_dir, f"{label}_scatter.png"))
    plt.close()


log_scatter_plot(y_dtheta_dt_test, model_dtheta_dt.predict(X_test), "dTheta_dt")
log_scatter_plot(y_dgamma_dt_test, model_dgamma_dt.predict(X_test), "dGamma_dt")


eq_str = str(model_dtheta_dt.get_best()["equation"])
used_features = Counter(re.findall(r"x\d+", eq_str))
wandb.log({f"feature_usage_dtheta_dt/{k}": v for k, v in used_features.items()})

errors_theta = model_dtheta_dt.predict(X_test) - y_dtheta_dt_test
errors_gamma = model_dgamma_dt.predict(X_test) - y_dgamma_dt_test

wandb.log({
    "Number of Features": X_train.shape[1],
    "Number of Training Samples": len(X_train),
    "eq_dtheta_dt_final": str(model_dtheta_dt.get_best()),
    "eq_dgamma_dt_final": str(model_dgamma_dt.get_best()),
    "r2_score_dtheta_dt": r2_score(y_dtheta_dt_test, model_dtheta_dt.predict(X_test)),
    "r2_score_dgamma_dt": r2_score(y_dgamma_dt_test, model_dgamma_dt.predict(X_test)),
    "dTheta_dt_best_complexity": model_dtheta_dt.get_best()["complexity"],
    "dGamma_dt_best_complexity": model_dgamma_dt.get_best()["complexity"],
    "dTheta_dt_error_hist": wandb.Histogram(errors_theta),
    "dGamma_dt_error_hist": wandb.Histogram(errors_gamma),
    "dTheta_dt_error_mean": np.mean(errors_theta),
    "dTheta_dt_error_std": np.std(errors_theta),
    "dGamma_dt_error_mean": np.mean(errors_gamma),
    "dGamma_dt_error_std": np.std(errors_gamma),
    "dTheta_dt_error_max": np.max(errors_theta),
    "dTheta_dt_error_min": np.min(errors_theta),
    "dGamma_dt_error_max": np.max(errors_gamma),
    "dGamma_dt_error_min": np.min(errors_gamma),
    "dTheta_dt_error_median": np.median(errors_theta),
    "dGamma_dt_error_median": np.median(errors_gamma),
})

# === Convergence Plot ===
def log_convergence_plot(model, label):
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


log_convergence_plot(model_dtheta_dt, "dTheta_dt")
log_convergence_plot(model_dgamma_dt, "dGamma_dt")

# === Save to W&B as Artifact ===
artifact = wandb.Artifact(f"dynamics_models_{Run_Name}", type="model")
artifact.add_dir(output_dir)
artifact.add_dir(os.path.join(output_dir, "dtheta_dt"))
artifact.add_dir(os.path.join(output_dir, "dgamma_dt"))
wandb.log_artifact(artifact)

wandb.finish()

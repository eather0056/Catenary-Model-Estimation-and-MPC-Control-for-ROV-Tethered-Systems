# === Improved Training Pipeline for Symbolic Dynamics (Theta, Gamma) ===

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from pysr import PySRRegressor
import wandb

# === Logging ===
wandb.init(project="Catenary_Dynamics_Improved", config={
    "niterations": 20000,
    "batch_size": 5000,
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos", "log", "sqrt"],
})

config = wandb.config
run_id = wandb.run.id
output_dir = f"outputs/improved_run_{run_id}"
os.makedirs(output_dir, exist_ok=True)

# === Feature Extraction ===
def extract_features(df):
    P0 = df[["rod_end X", "rod_end Y", "rod_end Z"]].values / 1000
    P1 = df[["robot_cable_attach_point X", "robot_cable_attach_point Y", "robot_cable_attach_point Z"]].values / 1000
    V1 = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values
    A1 = np.gradient(V1, axis=0)
    rel_vec = P1 - P0
    unit_rel = rel_vec / (np.linalg.norm(rel_vec, axis=1, keepdims=True) + 1e-8)
    theta = df["Theta"].values.reshape(-1, 1)
    gamma = df["Gamma"].values.reshape(-1, 1)
    sin_theta = np.sin(theta)
    cos_gamma = np.cos(gamma)
    dot_v1_rel = np.sum(V1 * unit_rel, axis=1, keepdims=True)
    features = np.hstack([P1, V1, A1, unit_rel, theta, gamma, sin_theta, cos_gamma, dot_v1_rel])
    return features

# === Target Derivatives ===
def compute_targets(df):
    time = df['Time'].values
    theta = gaussian_filter1d(df['Theta'].values, sigma=2)
    gamma = gaussian_filter1d(df['Gamma'].values, sigma=2)
    dtheta = np.gradient(theta, time)
    dgamma = np.gradient(gamma, time)
    return dtheta, dgamma

# === Load and Process Data ===
def load_data(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["Theta", "Gamma", "Time"])
    X = extract_features(df)
    y_dtheta, y_dgamma = compute_targets(df)
    return X, y_dtheta, y_dgamma

train_files = glob.glob("Data/L_dynamique6x*.csv") + glob.glob("Data/L_dynamique6y*.csv")
X_train, y_dtheta_train, y_dgamma_train = load_data(train_files)

# === Normalize Features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# === Train Symbolic Models ===
def train_model(X, y, label):
    model = PySRRegressor(
        niterations=config.niterations,
        binary_operators=config.binary_operators,
        unary_operators=config.unary_operators,
        complexity_of_operators={"/": 5, "sqrt": 3, "log": 3},
        maxsize=30,
        model_selection="best",
        elementwise_loss="loss(x, y) = (x - y)^2 + 0.01 * abs(x)",
        deterministic=True,
        parallelism="serial",
        procs=0,
        verbosity=1,
        batching=True,
        batch_size=config.batch_size,
        random_state=42,
    )
    model.fit(X, y)
    print(f"Best Equation for {label}:", model.get_best())
    wandb.log({f"{label}_eq": str(model.get_best())})
    return model

model_theta = train_model(X_train, y_dtheta_train, "dTheta_dt")
model_gamma = train_model(X_train, y_dgamma_train, "dGamma_dt")

# === Save Models ===
model_theta.equations_.to_csv(os.path.join(output_dir, "dtheta_results.csv"), index=False)
model_gamma.equations_.to_csv(os.path.join(output_dir, "dgamma_results.csv"), index=False)

# === Evaluation ===
def evaluate_model(model, X, y_true, label):
    from sklearn.metrics import r2_score
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    print(f"R² Score for {label}: {r2:.4f}")
    wandb.log({f"{label}_r2": r2})
    plt.figure()
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(f"{label}: Prediction vs True")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_fit.png"))
    wandb.log({f"{label}_fit_plot": wandb.Image(os.path.join(output_dir, f"{label}_fit.png"))})

# Re-evaluate on training data for now
evaluate_model(model_theta, X_train, y_dtheta_train, "dTheta_dt")
evaluate_model(model_gamma, X_train, y_dgamma_train, "dGamma_dt")

wandb.finish()

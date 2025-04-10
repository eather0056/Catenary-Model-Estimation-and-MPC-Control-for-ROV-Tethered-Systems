import os
import glob
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pysr import PySRRegressor
import wandb


os.environ["JULIA_DEBUG"] = "all"

# === Wandb Initialization ===
# Initialize wandb
wandb.init(
    project="Catenary_Dynamics",
    entity='eather0056',
    config={
        "model": "PySR",
        "niterations": 1000,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "exp", "log"],
        "batching": True,
        "batch_size": 5000,
    }
)

# === Load and Split Data ===
all_files = sorted(glob.glob("/home/mundus/mdeowan698/Catenary_Dynamic/Data/*.csv"))
train_files, test_files = train_test_split(all_files, test_size=0.9, random_state=42)

def load_and_concat(files):
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df.dropna(subset=['Theta', 'Gamma'])

df_train = load_and_concat(train_files)
df_test = load_and_concat(test_files)

# === Feature Engineering ===
def extract_features(df):
    P0 = df[['rod_end X', 'rod_end Y', 'rod_end Z']].values / 1000
    P1 = df[['robot_cable_attach_point X', 'robot_cable_attach_point Y', 'robot_cable_attach_point Z']].values / 1000
    V1 = df[['rob_cor_speed X', 'rob_cor_speed Y', 'rob_cor_speed Z']].values
    rel_vec = P1 - P0
    cable_len = np.linalg.norm(rel_vec, axis=1).reshape(-1, 1)
    speed_mag = np.linalg.norm(V1, axis=1).reshape(-1, 1)
    features = np.hstack([P0, P1, V1, rel_vec, cable_len, speed_mag])
    return features

X_train = extract_features(df_train)
y_theta_train = df_train['Theta'].values
y_gamma_train = df_train['Gamma'].values

X_test = extract_features(df_test)
y_theta_test = df_test['Theta'].values
y_gamma_test = df_test['Gamma'].values

# === Train PySR Models ===
common_params = dict(
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log"],
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",  # Replace old 'elementwise_loss'
    verbosity=2,
    random_state=42,
    deterministic=True,
    procs=0,
    batching=True,
    batch_size=5000,
)


model_sym_theta = PySRRegressor(**common_params)
model_sym_gamma = PySRRegressor(**common_params)

print("Training symbolic model for Theta...")
model_sym_theta.fit(X_train, y_theta_train)
theta_best_eq = model_sym_theta.get_best()
wandb.log({"best_equation_theta": str(theta_best_eq)})
wandb.log({"r2_train_theta": r2_score(y_theta_train, model_sym_theta.predict(X_train))})

print("Training symbolic model for Gamma...")
model_sym_gamma.fit(X_train, y_gamma_train)
gamma_best_eq = model_sym_gamma.get_best()
wandb.log({"best_equation_gamma": str(gamma_best_eq)})
wandb.log({"r2_train_gamma": r2_score(y_gamma_train, model_sym_gamma.predict(X_train))})

# === Test and Log Plots ===
theta_test_pred = model_sym_theta.predict(X_test)
gamma_test_pred = model_sym_gamma.predict(X_test)

wandb.log({
    "r2_test_theta": r2_score(y_theta_test, theta_test_pred),
    "r2_test_gamma": r2_score(y_gamma_test, gamma_test_pred),
})

def log_plot(actual, predicted, name):
    fig, ax = plt.subplots()
    ax.scatter(actual, predicted, alpha=0.5)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name} - Actual vs Predicted")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    wandb.log({f"{name}_scatter": wandb.Image(buf)})
    plt.close(fig)

log_plot(y_theta_test, theta_test_pred, "Theta")
log_plot(y_gamma_test, gamma_test_pred, "Gamma")

wandb.finish()

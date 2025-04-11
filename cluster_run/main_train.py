import os
import glob
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor
import wandb

# === Set JULIA debug mode for cluster troubleshooting ===
os.environ["JULIA_DEBUG"] = "all"

# === Initialize Weights & Biases ===
wandb.init(
    project="Catenary_Dynamics",
    entity="eather0056",
    config={
        "model": "PySR",
        "niterations": 1000,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "exp", "log"],
        "batching": True,
        "batch_size": 5000,
    }
)

run_id = wandb.run.id
output_dir = f"outputs/training_{run_id}"
os.makedirs(output_dir, exist_ok=True)

# # === Load all Data ===
# all_files = sorted(glob.glob("/home/mundus/mdeowan698/Catenary_Dynamic/Data/*.csv"))
# train_files, test_files = train_test_split(all_files, test_size=0.9, random_state=42)

# === Load and Combine Training Datasets ===
train_files = [
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x100dis2_0033.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x100dis2_0034.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x100dis2_0035.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x200dis2_0030.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x200dis2_0031.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x200dis2_0032.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis1_0018.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis1_0019.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis1_0020.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis2_0021.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis2_0022.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis2_0023.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis1_0025.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis1_0026.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis2_0027.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis2_0028.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis2_0029.csv"  
]

# === Test on New Dataset ===
test_files = ["/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y200dis1_0024.csv"]

def load_and_concat(files):
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df.dropna(subset=["Theta", "Gamma"])

df_train = load_and_concat(train_files)
df_test = load_and_concat(test_files)

# === Feature Engineering ===
def extract_features(df):
    P0 = df[["rod_end X", "rod_end Y", "rod_end Z"]].values / 1000
    P1 = df[["robot_cable_attach_point X", "robot_cable_attach_point Y", "robot_cable_attach_point Z"]].values / 1000
    V1 = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values
    rel_vec = P1 - P0
    cable_len = np.linalg.norm(rel_vec, axis=1).reshape(-1, 1)
    speed_mag = np.linalg.norm(V1, axis=1).reshape(-1, 1)
    return np.hstack([P0, P1, V1, rel_vec, cable_len, speed_mag])

X_train = extract_features(df_train)
y_theta_train = df_train["Theta"].values
y_gamma_train = df_train["Gamma"].values
X_test = extract_features(df_test)
y_theta_test = df_test["Theta"].values
y_gamma_test = df_test["Gamma"].values

# === Configure PySR ===
common_params = dict(
    niterations=wandb.config["niterations"],
    binary_operators=wandb.config["binary_operators"],
    unary_operators=wandb.config["unary_operators"],
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    verbosity=2,
    random_state=42,
    deterministic=True,
    procs=0,
    batching=wandb.config["batching"],
    batch_size=wandb.config["batch_size"],
)

# === Train Models ===
model_sym_theta = PySRRegressor(**common_params)
model_sym_gamma = PySRRegressor(**common_params)

print("Training symbolic model for Theta...")
model_sym_theta.fit(X_train, y_theta_train)
theta_best_eq = model_sym_theta.get_best()

print("Training symbolic model for Gamma...")
model_sym_gamma.fit(X_train, y_gamma_train)
gamma_best_eq = model_sym_gamma.get_best()

# === Save Models & Equations ===
joblib.dump(model_sym_theta, os.path.join(output_dir, "model_theta.pkl"))
joblib.dump(model_sym_gamma, os.path.join(output_dir, "model_gamma.pkl"))

with open(os.path.join(output_dir, "equation_theta.txt"), "w") as f:
    f.write(str(theta_best_eq))
with open(os.path.join(output_dir, "equation_gamma.txt"), "w") as f:
    f.write(str(gamma_best_eq))

model_sym_theta.equation_search_results.to_csv(os.path.join(output_dir, "theta_results.csv"), index=False)
model_sym_gamma.equation_search_results.to_csv(os.path.join(output_dir, "gamma_results.csv"), index=False)

# === Log Key Metrics to WandB ===
wandb.log({
    "best_equation_theta": str(theta_best_eq),
    "r2_train_theta": r2_score(y_theta_train, model_sym_theta.predict(X_train)),
    "r2_test_theta": r2_score(y_theta_test, model_sym_theta.predict(X_test)),
    "best_equation_gamma": str(gamma_best_eq),
    "r2_train_gamma": r2_score(y_gamma_train, model_sym_gamma.predict(X_train)),
    "r2_test_gamma": r2_score(y_gamma_test, model_sym_gamma.predict(X_test)),
})

# === Plotting Helper ===
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
    plt.close()

log_plot(y_theta_test, model_sym_theta.predict(X_test), "Theta")
log_plot(y_gamma_test, model_sym_gamma.predict(X_test), "Gamma")

# === Convergence Plot Logging ===
def log_convergence_plot(model, label):
    results = model.equation_search_results
    if results.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(results['complexity'], results['loss'], c='blue', alpha=0.6)
    best_idx = results['loss'].idxmin()
    best_point = results.loc[best_idx]
    plt.scatter([best_point['complexity']], [best_point['loss']], color='red', label="Best")
    plt.xlabel("Complexity")
    plt.ylabel("Loss")
    plt.title(f"{label} Convergence")
    plt.legend()
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    wandb.log({f"{label}_convergence_plot": wandb.Image(buf)})
    plt.close()
    wandb.log({
        f"{label}_min_loss": best_point['loss'],
        f"{label}_complexity": best_point['complexity'],
        f"{label}_score": best_point.get("score", np.nan),
    })

log_convergence_plot(model_sym_theta, "Theta")
log_convergence_plot(model_sym_gamma, "Gamma")

# === Save Artifacts to WandB ===
artifact = wandb.Artifact(f"pysr-models-{run_id}", type="model")
artifact.add_dir(output_dir)
wandb.log_artifact(artifact)

wandb.finish()

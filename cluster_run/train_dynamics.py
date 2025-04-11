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


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["JULIA_DEBUG"] = "all"

# === Initialize W&B ===
wandb.init(
    project="Catenary_Dynamics_Differential",
    entity="eather0056",
    config={
        "model": "PySR",
        "task": "Differential Equation Discovery",
        "niterations": 15000,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "exp", "log", "sqrt"],
        "batching": True,
        "batch_size": 5000,
    }
)

run_id = wandb.run.id
output_dir = f"outputs/differential_training_{run_id}"
os.makedirs(output_dir, exist_ok=True)

# === Load & Preprocess ===
def load_and_concat(files):
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df.dropna(subset=["Theta", "Gamma", "Time"])

# all_files = sorted(glob.glob("/home/mundus/mdeowan698/Catenary_Dynamic/Data/*.csv"))
# train_files, test_files = train_test_split(all_files, test_size=0.9, random_state=42)


# === Load and Combine Training Datasets ===
train_files = [
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x100dis2_0033.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x100dis2_0034.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x100dis2_0035.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x200dis2_0030.csv",  
    "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x200dis2_0031.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6x200dis2_0032.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis1_0018.csv",  
    # "/home/mundus/mdeowan698/Catenary_Dynamic/Data/L_dynamique6y100dis1_0019.csv",  
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


df_train = load_and_concat(train_files)
df_test = load_and_concat(test_files)

# === Features & Derivatives ===
def extract_features(df):
    P0 = df[["rod_end X", "rod_end Y", "rod_end Z"]].values / 1000
    P1 = df[["robot_cable_attach_point X", "robot_cable_attach_point Y", "robot_cable_attach_point Z"]].values / 1000
    V1 = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values
    rel_vec = P1 - P0
    cable_len = np.linalg.norm(rel_vec, axis=1).reshape(-1, 1)
    speed_mag = np.linalg.norm(V1, axis=1).reshape(-1, 1)
    return np.hstack([P0, P1, V1, rel_vec, cable_len, speed_mag])

def compute_derivatives(df):
    time = df["Time"].values
    dtheta = np.gradient(df["Theta"].values, time)
    dgamma = np.gradient(df["Gamma"].values, time)
    return dtheta, dgamma

X_train = extract_features(df_train)
y_dtheta_dt_train, y_dgamma_dt_train = compute_derivatives(df_train)

X_test = extract_features(df_test)
y_dtheta_dt_test, y_dgamma_dt_test = compute_derivatives(df_test)

# === PySR Parameters ===
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

model_dtheta_dt = PySRRegressor(**common_params)
model_dgamma_dt = PySRRegressor(**common_params)

# === Background Logger for Training Progress ===
def log_pysr_progress(model, label, total_iters, interval=60):
    def _loop():
        while not getattr(model, "_finished", False):
            try:
                results = model.equation_search_results
                if not results.empty:
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

# === Train with Live Logging ===
log_pysr_progress(model_dtheta_dt, "dTheta_dt", wandb.config["niterations"])
print("Training model for dTheta/dt...")
model_dtheta_dt.fit(X_train, y_dtheta_dt_train)
model_dtheta_dt._finished = True

log_pysr_progress(model_dgamma_dt, "dGamma_dt", wandb.config["niterations"])
print("Training model for dGamma/dt...")
model_dgamma_dt.fit(X_train, y_dgamma_dt_train)
model_dgamma_dt._finished = True

# === Save Outputs ===
joblib.dump(model_dtheta_dt, os.path.join(output_dir, f"model_dtheta_dt_{timestamp}.pkl"))
joblib.dump(model_dgamma_dt, os.path.join(output_dir, "model_dgamma_dt_{timestamp}.pkl"))

with open(os.path.join(output_dir, f"eq_dtheta_dt_{timestamp}.txt"), "w") as f:
    f.write(str(model_dtheta_dt.get_best()))
with open(os.path.join(output_dir, f"eq_dgamma_dt_{timestamp}.txt"), "w") as f:
    f.write(str(model_dgamma_dt.get_best()))

model_dtheta_dt.equation_search_results.to_csv(os.path.join(output_dir, f"dtheta_results_{timestamp}.csv"), index=False)
model_dgamma_dt.equation_search_results.to_csv(os.path.join(output_dir, f"dgamma_results_{timestamp}.csv"), index=False)

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


# === W&B Logs ===
wandb.log({
    "eq_dtheta_dt_final": str(model_dtheta_dt.get_best()),
    "eq_dgamma_dt_final": str(model_dgamma_dt.get_best()),
})

# === Evaluation ===
def log_scatter_plot(actual, pred, label):
    fig, ax = plt.subplots()
    ax.scatter(actual, pred, alpha=0.4)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    ax.set_title(f"{label}: dActual vs dPredicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    wandb.log({f"{label}_scatter": wandb.Image(buf)})
    plt.close()

log_scatter_plot(y_dtheta_dt_test, model_dtheta_dt.predict(X_test), "dTheta_dt")
log_scatter_plot(y_dgamma_dt_test, model_dgamma_dt.predict(X_test), "dGamma_dt")

# === Convergence Plot ===
def log_convergence_plot(model, label):
    res = model.equation_search_results
    if res.empty: return
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
    wandb.log({f"{label}_convergence": wandb.Image(buf)})
    plt.close()

log_convergence_plot(model_dtheta_dt, "dTheta_dt")
log_convergence_plot(model_dgamma_dt, "dGamma_dt")

# === Save to W&B as Artifact ===
artifact = wandb.Artifact(f"dynamics_models_{run_id}", type="model")
artifact.add_dir(output_dir)
artifact.add_dir(os.path.join(output_dir, "dtheta_dt"))
artifact.add_dir(os.path.join(output_dir, "dgamma_dt"))
wandb.log_artifact(artifact)


wandb.finish()

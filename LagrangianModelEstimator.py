# Enhanced version of the LagrangianPipeline setup and execution,
# now with better expression constraints, operator complexity, and seeding.

import os
import joblib
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from datetime import datetime
import wandb
from main_fun import *
from lagrangian_pipeline import LagrangianPipeline  # assumes class is defined here
import matplotlib.pyplot as plt
import sympy


# === Setup ===
Run_Name = "Lg_C6_split_1K_20it"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs/{Run_Name}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.environ["JULIA_DEBUG"] = "all"

wandb.init(
    project="Catenary_Dynamics_Differential",
    entity="eather0056",
    name=f"{Run_Name}_{timestamp}",
    tags=["symbolic", "dynamics", "lagrangian", "enhanced"],
    notes="Improved symbolic Lagrangian training: operator penalties, expression seeds, scaled input split featue used P1, V1, A1, unit_rel, tension, angle_proj, theta, gamma.",
    config={
        "model": "PySR",
        "task": "Lagrangian Discovery",
        "niterations": 20,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "tanh", "exp", "log"],
        "complexity_of_operators": {"+": 1, "-": 1, "*": 2, "/": 5},
        "loss": "loss(x, y) = (x - y)^2",
        "random_state": 42,
        "maxsize": 30,
        "verbosity": 1,
        "procs": 0,
        "LAGRANGIAN_MODE": "split",  # Options: "split" or "full"
    }
)

config = wandb.config

# === Load Data ===
train_files = [
    "Data/L_dynamique6x200dis2_0031.csv",
    "Data/L_dynamique6y200dis2_0029.csv"
]
test_files = ["Data/L_dynamique6y200dis1_0024.csv"]

df_train = load_and_concat(train_files)
df_test = load_and_concat(test_files)

# === Run Lagrangian Pipeline ===
pipeline = LagrangianPipeline(
    model_params=dict(
        niterations=config.niterations,
        binary_operators=config.binary_operators,
        unary_operators=config.unary_operators,
        complexity_of_operators=config.complexity_of_operators,
        model_selection="best",
        loss=config.loss,
        verbosity=config.verbosity,
        procs=config.procs,
        maxsize=config.maxsize,
        extra_sympy_mappings={
            "log": sympy.log,
            "exp": sympy.exp,
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tanh": sympy.tanh,
            "sqrt": sympy.sqrt
        },

    ),
    mode=config.LAGRANGIAN_MODE  # <--- NEW
)

LAGRANGIAN_MODE = "split"
# Ensure features are diverse and scaled (if necessary inside pipeline)
mse_theta, mse_gamma = pipeline.run(df_train, output_dir)


# Optional: print input stats to check for variation
print("θ range:", np.min(pipeline.theta), np.max(pipeline.theta))
print("γ range:", np.min(pipeline.gamma), np.max(pipeline.gamma))
print("dθ mean/std:", np.mean(pipeline.dtheta), np.std(pipeline.dtheta))
print("dγ mean/std:", np.mean(pipeline.dgamma), np.std(pipeline.dgamma))


if LAGRANGIAN_MODE == "split":
    with open(os.path.join(output_dir, "T_expression.txt"), "w") as f:
        f.write(str(pipeline.T_expr_str))
    with open(os.path.join(output_dir, "V_expression.txt"), "w") as f:
        f.write(str(pipeline.V_expr_str))


# === Log Results to W&B ===
log_dict = {
    "EL_residual_mse_theta": mse_theta,
    "EL_residual_mse_gamma": mse_gamma,
}

if LAGRANGIAN_MODE == "full":
    log_dict["best_lagrangian"] = str(pipeline.best_eq["equation"])
else:  # split mode
    log_dict["T_expression"] = pipeline.T_expr_str
    log_dict["V_expression"] = pipeline.V_expr_str
    log_dict["lagrangian_expression"] = f"({pipeline.T_expr_str}) - ({pipeline.V_expr_str})"

wandb.log(log_dict)



# === Save model ===
if LAGRANGIAN_MODE == "full":
    joblib.dump(pipeline.model, os.path.join(output_dir, "model_lagrangian.pkl"))
    with open(os.path.join(output_dir, "best_lagrangian.txt"), "w") as f:
        f.write(str(pipeline.best_eq["equation"]))
    pipeline.model.equations_.to_csv(os.path.join(output_dir, "lagrangian_equation_history.csv"), index=False)
elif LAGRANGIAN_MODE == "split":
    joblib.dump(pipeline.model_T, os.path.join(output_dir, "model_T.pkl"))
    joblib.dump(pipeline.model_V, os.path.join(output_dir, "model_V.pkl"))
    with open(os.path.join(output_dir, "T_expression.txt"), "w") as f:
        f.write(str(pipeline.T_expr_str))
    with open(os.path.join(output_dir, "V_expression.txt"), "w") as f:
        f.write(str(pipeline.V_expr_str))
    with open(os.path.join(output_dir, "lagrangian_expression.txt"), "w") as f:
        f.write(f"({pipeline.T_expr_str}) - ({pipeline.V_expr_str})")

# === Save symbolic equation history ===
if config.LAGRANGIAN_MODE == "full":
    pipeline.model.equations_.to_csv(os.path.join(output_dir, "lagrangian_equation_history.csv"), index=False)
elif config.LAGRANGIAN_MODE == "split":
    pipeline.model_T.equations_.to_csv(os.path.join(output_dir, "T_equation_history.csv"), index=False)
    pipeline.model_V.equations_.to_csv(os.path.join(output_dir, "V_equation_history.csv"), index=False)


# === Save time-series and derivatives ===
np.savez(
    os.path.join(output_dir, "trajectory_data.npz"),
    theta=pipeline.theta,
    gamma=pipeline.gamma,
    dtheta=pipeline.dtheta,
    dgamma=pipeline.dgamma,
    ddtheta=pipeline.ddtheta,
    ddgamma=pipeline.ddgamma,
    time=df_train["Time"].values,
    X_lagr=pipeline.X_lagr,
)

# === Save E-L residuals for plotting or inspection ===
res_θ = np.array([
    pipeline.EOM_θ_func(θ, γ, dθ, dγ, ddθ, ddγ)
    for θ, γ, dθ, dγ, ddθ, ddγ in zip(
        pipeline.theta,
        pipeline.gamma,
        pipeline.dtheta,
        pipeline.dgamma,
        pipeline.ddtheta,
        pipeline.ddgamma
    )
])

res_γ = np.array([
    pipeline.EOM_γ_func(θ, γ, dθ, dγ, ddθ, ddγ)
    for θ, γ, dθ, dγ, ddθ, ddγ in zip(
        pipeline.theta,
        pipeline.gamma,
        pipeline.dtheta,
        pipeline.dgamma,
        pipeline.ddtheta,
        pipeline.ddgamma
    )
])

time = np.asarray(df_train["Time"].values).flatten()


# Ensure consistent 1D shapes
res_θ = np.atleast_1d(res_θ).flatten()
res_γ = np.atleast_1d(res_γ).flatten()
time = np.atleast_1d(df_train["Time"].values).flatten()

np.savez(
    os.path.join(output_dir, "euler_lagrange_residuals.npz"),
    residual_theta=res_θ,
    residual_gamma=res_γ,
)

print("time shape:", time.shape)
print("res_θ shape:", res_θ.shape)
print("res_γ shape:", res_γ.shape)


# === Plot E-L residuals (two subplots) ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(time, res_θ, label="E-L Residual θ")
plt.title("Euler–Lagrange Residual (θ)")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time, res_γ, label="E-L Residual γ", color="orange")
plt.title("Euler–Lagrange Residual (γ)")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "euler_lagrange_residuals.png"))
plt.close()

# === Plot Combined Residuals (Overlay) ===
plt.figure(figsize=(10, 4))
plt.plot(time, res_θ, label="θ residual")
plt.plot(time, res_γ, label="γ residual")
plt.legend()
plt.title("Euler–Lagrange Residuals")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals_plot.png"))


# === Prepare test set for evaluation ===
pipeline.prepare_data(df_test)

res_θ_test = np.array([
    pipeline.EOM_θ_func(θ, γ, dθ, dγ, ddθ, ddγ)
    for θ, γ, dθ, dγ, ddθ, ddγ in zip(
        pipeline.theta, pipeline.gamma,
        pipeline.dtheta, pipeline.dgamma,
        pipeline.ddtheta, pipeline.ddgamma
    )
])

res_γ_test = np.array([
    pipeline.EOM_γ_func(θ, γ, dθ, dγ, ddθ, ddγ)
    for θ, γ, dθ, dγ, ddθ, ddγ in zip(
        pipeline.theta, pipeline.gamma,
        pipeline.dtheta, pipeline.dgamma,
        pipeline.ddtheta, pipeline.ddgamma
    )
])

# === Metrics ===
mse_theta_test = np.mean(res_θ_test**2)
mse_gamma_test = np.mean(res_γ_test**2)
print(f"[TEST] E-L residuals MSE: θ={mse_theta_test:.6e}, γ={mse_gamma_test:.6e}")

# === Log to W&B ===
wandb.log({
    "test_EL_residual_mse_theta": mse_theta_test,
    "test_EL_residual_mse_gamma": mse_gamma_test,
})

# === Save or plot residuals if needed ===
np.savez(
    os.path.join(output_dir, "test_euler_lagrange_residuals.npz"),
    residual_theta=res_θ_test,
    residual_gamma=res_γ_test,
)


artifact = wandb.Artifact(f"Lagrangian_model_{Run_Name}", type="model")
artifact.add_dir(output_dir)
wandb.log_artifact(artifact)

wandb.finish()

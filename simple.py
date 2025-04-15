import os
import joblib
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# === Config ===
csv_path = "Data/L_dynamique6x100dis2_0033.csv"
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# === Feature Extraction ===
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
    time = df["Time"].values
    theta = df["Theta"].values
    gamma = df["Gamma"].values
    dtheta = np.gradient(theta, time)
    dgamma = np.gradient(gamma, time)
    return dtheta, dgamma

# === Load and Prepare Data ===
df = pd.read_csv(csv_path)
X = extract_features(df)
y_dtheta_dt, y_dgamma_dt = compute_derivatives(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PySR Config ===
def make_model():
    return PySRRegressor(
        model_selection="best",
        niterations=1000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "abs", "square", "tanh"],
        loss="loss(x, y) = (x - y)^2 + 0.01 * abs(x)",
        complexity_of_operators={"/": 5, "square": 2, "tanh": 3, "sin": 2, "cos": 2},
        deterministic=True,
        should_simplify=False,
        verbosity=1,
        multithreading=False,
        procs=0,
        batching=True,
        batch_size=1000,
        maxsize=30,
        random_state=42,
        temp_equation_file=True,
    )

# === Train and Save for dTheta/dt ===
model_theta = make_model()
model_theta.fit(X_scaled, y_dtheta_dt)
pred_theta = model_theta.predict(X_scaled)
print("dTheta/dt R² Score:", r2_score(y_dtheta_dt, pred_theta))
print("Best Equation (Theta):", model_theta.get_best())

# Save
joblib.dump(model_theta, os.path.join(save_dir, "model_dtheta_dt.pkl"))
with open(os.path.join(save_dir, "eq_dtheta_dt.txt"), "w") as f:
    f.write(str(model_theta.get_best()))
model_theta.equations_.to_csv(os.path.join(save_dir, "equations_dtheta_dt.csv"), index=False)

# === Train and Save for dGamma/dt ===
model_gamma = make_model()
model_gamma.fit(X_scaled, y_dgamma_dt)
pred_gamma = model_gamma.predict(X_scaled)
print("dGamma/dt R² Score:", r2_score(y_dgamma_dt, pred_gamma))
print("Best Equation (Gamma):", model_gamma.get_best())

# Save
joblib.dump(model_gamma, os.path.join(save_dir, "model_dgamma_dt.pkl"))
with open(os.path.join(save_dir, "eq_dgamma_dt.txt"), "w") as f:
    f.write(str(model_gamma.get_best()))
model_gamma.equations_.to_csv(os.path.join(save_dir, "equations_dgamma_dt.csv"), index=False)

# === Save predictions for inspection ===
results_df = df.copy()
results_df["dTheta_pred"] = pred_theta
results_df["dGamma_pred"] = pred_gamma
results_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

# === Save Scaler ===
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

print("✅ All models and data saved to:", save_dir)

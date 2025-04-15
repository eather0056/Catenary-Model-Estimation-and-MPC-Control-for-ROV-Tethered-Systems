import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score
import pathlib
from sympy import latex

# === Load Test Dataset ===
df = pd.read_csv("Data\\L_dynamique6x100dis2_0033.csv")  # Use double backslash for Windows
time = df["Time"].values

# === Extract Features (same function as training) ===
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

X_test = extract_features(df)

pathlib.PosixPath = pathlib.WindowsPath
# === Load Trained Models ===
model_dgamma_dt = joblib.load("outputs/saved_models/model_dgamma_dt.pkl")
model_dtheta_dt = joblib.load("outputs/saved_models/model_dtheta_dt.pkl")

# # === Print Best Equations and LaTeX ===
# print("Best Equation for dTheta/dt:")
# print(model_dtheta_dt.get_best())
# print("\nLaTeX for dTheta/dt:")
# print(model_dtheta_dt.latex())

# print("\nBest Equation for dGamma/dt:")
# print(model_dgamma_dt.get_best())
# print("\nLaTeX for dGamma/dt:")
# print(model_dgamma_dt.latex())

# === Get LaTeX from sympy_format ===
best_eq_dtheta = model_dtheta_dt.get_best()
sympy_expr = best_eq_dtheta["sympy_format"]
latex_expr = latex(sympy_expr)

print(f"\nManual LaTeX for dTheta/dt:\n{latex_expr}")

best_eq_dgamma = model_dgamma_dt.get_best()
sympy_expr = best_eq_dgamma["sympy_format"]
latex_expr_gammma = latex(sympy_expr)

print(f"\nManual LaTeX for dGamma/dt:\n{latex_expr_gammma}")

# Save to file
with open("latex_dtheta_dt.tex", "w") as f:
    f.write(f"${latex_expr}$")

best_eq_dgamma = model_dgamma_dt.get_best()
sympy_expr_gamma = best_eq_dgamma["sympy_format"]
latex_expr_gamma = latex(sympy_expr_gamma)

with open("latex_dgamma_dt.tex", "w") as f:
    f.write(f"${latex_expr_gamma}$")

best_eq_dtheta = model_dtheta_dt.get_best()
best_eq_dgamma = model_dgamma_dt.get_best()

theta_func = best_eq_dtheta["lambda_format"]
gamma_func = best_eq_dgamma["lambda_format"]

dtheta_dt_pred = theta_func(X_test)
dgamma_dt_pred = gamma_func(X_test)


# === Integrate to Estimate Theta(t) and Gamma(t) ===
theta0 = df["Theta"].values[0]
gamma0 = df["Gamma"].values[0]

theta_est = [theta0]
gamma_est = [gamma0]

for i in range(1, len(time)):
    dt = time[i] - time[i - 1]
    theta_est.append(theta_est[-1] + dtheta_dt_pred[i - 1] * dt)
    gamma_est.append(gamma_est[-1] + dgamma_dt_pred[i - 1] * dt)

theta_est = np.array(theta_est)
gamma_est = np.array(gamma_est)

# === Ground Truth and Errors ===
theta_true = df["Theta"].values
gamma_true = df["Gamma"].values

theta_error = theta_est - theta_true
gamma_error = gamma_est - gamma_true

# === R² Scores ===
print(f"\nR² Score for Theta(t): {r2_score(theta_true, theta_est):.4f}")
print(f"R² Score for Gamma(t): {r2_score(gamma_true, gamma_est):.4f}")

# === Time-Series Plot ===
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(time, theta_true, label="True Theta", color="blue")
plt.plot(time, theta_est, '--', label="Predicted Theta", color="red")
plt.ylabel("Theta (rad)")
plt.title("Theta(t) Prediction")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, gamma_true, label="True Gamma", color="blue")
plt.plot(time, gamma_est, '--', label="Predicted Gamma", color="red")
plt.ylabel("Gamma (rad)")
plt.title("Gamma(t) Prediction")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, theta_error, label="Theta Error", color="purple")
plt.plot(time, gamma_error, label="Gamma Error", color="orange")
plt.title("Estimation Error")
plt.ylabel("Error (rad)")
plt.xlabel("Time (s)")
plt.legend()

plt.tight_layout()
plt.show()

# === Scatter Plot: Pred vs True ===
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(theta_true, theta_est, alpha=0.4)
plt.plot([min(theta_true), max(theta_true)], [min(theta_true), max(theta_true)], 'r--')
plt.xlabel("True Theta")
plt.ylabel("Predicted Theta")
plt.title("Theta(t): Pred vs True")

plt.subplot(1, 2, 2)
plt.scatter(gamma_true, gamma_est, alpha=0.4)
plt.plot([min(gamma_true), max(gamma_true)], [min(gamma_true), max(gamma_true)], 'r--')
plt.xlabel("True Gamma")
plt.ylabel("Predicted Gamma")
plt.title("Gamma(t): Pred vs True")

plt.tight_layout()
plt.show()

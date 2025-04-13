import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify, lambdify
from pysr import best
from pysr import get_sympy_equation


# === 1. Feature Extraction Function ===
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

    theta = df["Theta"].values.reshape(-1, 1)
    gamma = df["Gamma"].values.reshape(-1, 1)
    cos_theta = np.cos(theta)
    sin_gamma = np.sin(gamma)

    dot_product = np.sum(V1 * unit_rel, axis=1, keepdims=True)
    norm_v1 = np.linalg.norm(V1, axis=1, keepdims=True) + 1e-8
    angle_proj = dot_product / norm_v1

    return np.hstack([P1, V1, A1, unit_rel, theta, gamma, cos_theta, sin_gamma, angle_proj])


# === 2. Load dataset ===
df = pd.read_csv("Data/L_dynamique6x100dis2_0035.csv")
X = extract_features(df)

# === 3. Function to load PySR equation and evaluate it ===
def load_best_model(folder_path, inputs):
    hof = pd.read_csv(f"{folder_path}/hall_of_fame.csv")
    equation = best(hof)["equation"]
    print(f"\nBest equation from {folder_path}:")
    print(equation)

    expr = get_sympy_equation(best(hof))
    func = lambdify(expr.free_symbols, expr, modules=["numpy"])
    y_pred = func(*inputs.T)
    return y_pred, str(expr)


# === 4. Predict dtheta_dt ===
y_theta_pred, eq_theta = load_best_model("20250412_214744_lpGawP", X)
y_theta_true = df["dtheta_dt"].values

# === 5. Predict dgamma_dt ===
y_gamma_pred, eq_gamma = load_best_model("20250412_225159_GufPvS", X)
y_gamma_true = df["dgamma_dt"].values

# === 6. Plot results ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(y_theta_true, label="True dtheta_dt")
plt.plot(y_theta_pred, label="Predicted dtheta_dt", linestyle='--')
plt.title("dtheta_dt Prediction")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_gamma_true, label="True dgamma_dt")
plt.plot(y_gamma_pred, label="Predicted dgamma_dt", linestyle='--')
plt.title("dgamma_dt Prediction")
plt.legend()

plt.tight_layout()
plt.show()

# === 7. Print LaTeX equations for thesis/report ===
from sympy import latex
print("\nLaTeX Equation for dtheta_dt:")
print(latex(sympify(eq_theta)))

print("\nLaTeX Equation for dgamma_dt:")
print(latex(sympify(eq_gamma)))

import numpy as np
import pandas as pd
import sympy as sp
import joblib
import matplotlib.pyplot as plt
from main_fun import extract_features  # make sure this is present

# === Load Test Dataset ===
df_test = pd.read_csv("Data/L_dynamique6y200dis1_0024.csv")
features = extract_features(df_test)
time = df_test["Time"].values
theta_true = df_test["Theta"].values
gamma_true = df_test["Gamma"].values

# === Load Trained Lagrangian Models ===
model_T = joblib.load("outputs/Lg_C6_split_Hy_1K_20it_20250424_165101/model_T.pkl")
model_V = joblib.load("outputs/Lg_C6_split_Hy_1K_20it_20250424_165101/model_V.pkl")

# === Build Symbolic Lagrangian ===
θ, γ, dθ, dγ, ddθ, ddγ = sp.symbols("θ γ dθ dγ ddθ ddγ")
T_expr = sp.sympify(str(model_T.get_best()["equation"]))
V_expr = sp.sympify(str(model_V.get_best()["equation"]))
L_expr = T_expr - V_expr

# === Derive and Solve for θ̈ ===
dL_ddθ = sp.diff(L_expr, dθ)
dL_dθ = sp.diff(L_expr, θ)
d_dL_ddθ = (
    sp.diff(dL_ddθ, θ)*dθ + sp.diff(dL_ddθ, γ)*dγ +
    sp.diff(dL_ddθ, dθ)*ddθ + sp.diff(dL_ddθ, dγ)*ddγ
)
EOM_θ = d_dL_ddθ - dL_dθ
sol_ddθ = sp.solve(EOM_θ, ddθ)[0]
ddθ_func = sp.lambdify([θ, γ, dθ, dγ], sol_ddθ, modules=["numpy"])

# === Derive and Solve for γ̈ ===
dL_ddγ = sp.diff(L_expr, dγ)
dL_dγ = sp.diff(L_expr, γ)
d_dL_ddγ = (
    sp.diff(dL_ddγ, θ)*dθ + sp.diff(dL_ddγ, γ)*dγ +
    sp.diff(dL_ddγ, dθ)*ddθ + sp.diff(dL_ddγ, dγ)*ddγ
)
EOM_γ = d_dL_ddγ - dL_dγ
sol_ddγ = sp.solve(EOM_γ, ddγ)[0]
ddγ_func = sp.lambdify([θ, γ, dθ, dγ], sol_ddγ, modules=["numpy"])

# === Initialize Simulation ===
theta_est = np.zeros_like(theta_true)
gamma_est = np.zeros_like(gamma_true)
vtheta = np.zeros_like(theta_true)
vgamma = np.zeros_like(gamma_true)

theta_est[0] = theta_true[0]
gamma_est[0] = gamma_true[0]
vtheta[0] = np.gradient(theta_true, time)[0]
vgamma[0] = np.gradient(gamma_true, time)[0]

# === Run Forward Integration ===
for i in range(1, len(time)):
    dt = time[i] - time[i - 1]
    aθ = ddθ_func(theta_est[i - 1], gamma_est[i - 1], vtheta[i - 1], vgamma[i - 1])
    aγ = ddγ_func(theta_est[i - 1], gamma_est[i - 1], vtheta[i - 1], vgamma[i - 1])

    vtheta[i] = vtheta[i - 1] + aθ * dt
    theta_est[i] = theta_est[i - 1] + vtheta[i - 1] * dt

    vgamma[i] = vgamma[i - 1] + aγ * dt
    gamma_est[i] = gamma_est[i - 1] + vgamma[i - 1] * dt

# === Plot Results ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(time, theta_true, label="True θ", linewidth=2)
plt.plot(time, theta_est, label="Estimated θ", linestyle="--")
plt.title("Theta: Prediction vs Ground Truth")
plt.xlabel("Time (s)")
plt.ylabel("Theta (rad)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time, gamma_true, label="True γ", linewidth=2)
plt.plot(time, gamma_est, label="Estimated γ", linestyle="--")
plt.title("Gamma: Prediction vs Ground Truth")
plt.xlabel("Time (s)")
plt.ylabel("Gamma (rad)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("hybrid_lagrangian_prediction.png")
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pathlib
import os
from main_fun import *
from scipy.integrate import cumulative_trapezoid
import joblib


# === Load Dataset ===
file_name = "L_dynamique6y200dis1_0025.csv"
file_path = pathlib.Path("Data") / file_name
df = pd.read_csv(file_path)
X_theta, _ = features_dd(df)
X_gamma, _ = features_dd(df)
scaler_theta = joblib.load("outputs/dd_C6_all_50_ld_20250617_152853/scaler.pkl")  
scaler_gamma = joblib.load("outputs/dd_C6_Y_1k_ld_20250617_193707/scaler.pkl")
X_scaled_theta = scaler_theta.transform(X_theta)
X_scaled_gamma = scaler_gamma.transform(X_gamma)

time = df["Time"].values
theta_true = df["Theta"].values
gamma_true = df["Gamma"].values

# === Custom Equations: Insert Your Lambda Functions Here ===
custom_eq_dtheta_dt = lambda X: 0.0753 * (
    (X[:, 6] * (X[:, 0] + 0.4637) - X[:, 4]) -
    np.sin(np.tanh(X[:, 2]) ** 4) * (X[:, 1] + X[:, 5] - 1.1639)
)

custom_eq_dgamma_dt = lambda X: -0.0547 * (
    (X[:, 6] * 1.8169 - np.abs(X[:, 6]) * ((np.sin(-1.7310 * X[:, 7]) / 0.5578 - X[:, 8]))) +
    np.tanh(X[:, 2]) ** 2 + X[:, 3]
)

# === Evaluate Model ===
lags = 2
time_valid = time[lags:]
theta_0 = theta_true[lags]
gamma_0 = gamma_true[lags]

dtheta_pred = custom_eq_dtheta_dt(X_scaled_theta[lags:])
dgamma_pred = custom_eq_dgamma_dt(X_scaled_gamma[lags:])

dtheta = cumulative_trapezoid(dtheta_pred, time_valid, initial=0)
theta_est = theta_0 + cumulative_trapezoid(dtheta, time_valid, initial=0)

dgamma = cumulative_trapezoid(dgamma_pred, time_valid, initial=0)
gamma_est = gamma_0 + cumulative_trapezoid(dgamma, time_valid, initial=0)

# === Evaluate R² ===
theta_true_valid = theta_true[lags:]
gamma_true_valid = gamma_true[lags:]
theta_r2 = r2_score(theta_true_valid, theta_est)
gamma_r2 = r2_score(gamma_true_valid, gamma_est)

print(f"Custom Equation R² (Theta): {theta_r2:.4f}")
print(f"Custom Equation R² (Gamma): {gamma_r2:.4f}")

# === Plotting ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(time_valid, theta_true_valid, label="True Theta")
plt.plot(time_valid, theta_est, '--', label="Estimated Theta")
plt.title("Theta(t) Comparison")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time_valid, gamma_true_valid, label="True Gamma")
plt.plot(time_valid, gamma_est, '--', label="Estimated Gamma")
plt.title("Gamma(t) Comparison")
plt.legend()

plt.tight_layout()
plt.show()

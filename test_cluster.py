import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import pathlib
import os
from main_fun import *

# pathlib.PosixPath = pathlib.WindowsPath

# === Define The Run Name ===
ranname = "C6_2_FF_1k_20250425_121524"
# === Load Models and Scaler ===
model_theta = joblib.load("outputs/C6_2_FF_1k_20250425_121524/model_dtheta_dt.pkl")  
model_gamma = joblib.load("outputs/C6_2_FF_1k_20250425_121524/model_dgamma_dt.pkl")  
scaler = joblib.load("outputs/C6_2_FF_1k_20250425_121524/scaler.pkl")  

# === Load and Preprocess Dataset ===
file_name = "L_dynamique6y200dis1_0024.csv"

output_dir = "Results/mode_test"
output_path = pathlib.Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Load the dataset
file_path = pathlib.Path("Data") / file_name
df = pd.read_csv(file_path)


# === Extract Features ===
X = extract_features(df)
X_scaled = scaler.transform(X)
time = df["Time"].values

# === Predict Derivatives ===
dtheta_pred = model_theta.predict(X_scaled)
dgamma_pred = model_gamma.predict(X_scaled)

# === Integrate to Estimate Theta and Gamma ===
theta_est = np.zeros_like(dtheta_pred)
gamma_est = np.zeros_like(dgamma_pred)
theta_est[0] = df["Theta"].values[0]
gamma_est[0] = df["Gamma"].values[0]

for i in range(1, len(time)):
    dt = time[i] - time[i - 1]
    theta_est[i] = theta_est[i - 1] + dtheta_pred[i - 1] * dt
    gamma_est[i] = gamma_est[i - 1] + dgamma_pred[i - 1] * dt

theta_true = df["Theta"].values
gamma_true = df["Gamma"].values
theta_error = theta_true - theta_est
gamma_error = gamma_true - gamma_est

# === print these equeation ===
print(f"\neq_dtheta_dt: {model_theta.get_best()}")
print(f"\neq_dgamma_dt: {model_gamma.get_best()}")

# print(model_theta.equations_[["complexity", "loss", "equation"]])
# print(model_gamma.equations_[["complexity", "loss", "equation"]])

# === R² Scores ===
print(f"\nR² Score for Theta(t): {r2_score(theta_true, theta_est):.4f}")
print(f"R² Score for Gamma(t): {r2_score(gamma_true, gamma_est):.4f}")

# === Time-Series Plot ===
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(time, theta_true, label="True Theta", color="blue")
plt.plot(time, theta_est, '--', label="Predicted Theta", color="red")
plt.ylabel("Theta (rad)")
plt.title("Theta(t) Prediction")
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(time, gamma_true, label="True Gamma", color="blue")
plt.plot(time, gamma_est, '--', label="Predicted Gamma", color="red")
plt.ylabel("Gamma (rad)")
plt.title("Gamma(t) Prediction")
plt.legend()
plt.grid()

plt.subplot(4, 2, 5)
plt.plot(time, theta_error, label="Theta Error", color="purple")
plt.title("Theta(t)  Estimation Error")
plt.legend()
plt.ylabel("Error (rad)")
plt.xlabel("Time (s)")

plt.subplot(4, 2, 6)
plt.plot(time, gamma_error, label="Gamma Error", color="orange")
plt.title("Gamma(t)  Estimation Error")
plt.ylabel("Error (rad)")
plt.xlabel("Time (s)")
plt.legend()

theta_percentage_error = (theta_error / theta_true) * 100
gamma_percentage_error = (gamma_error / gamma_true) * 100

plt.subplot(4, 2, 7)
plt.plot(time, theta_percentage_error, label="Theta % Error", color="purple")
plt.title("Theta(t) Estimation Percentage Error")
plt.legend()
plt.ylabel("Percentage Error (%)")
plt.xlabel("Time (s)")

plt.subplot(4, 2, 8)
plt.plot(time, gamma_percentage_error, label="Gamma % Error", color="orange")
plt.title("Gamma(t)  Estimation Percentage Error")
plt.ylabel("Percentage Error (%)")
plt.xlabel("Time (s)")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"predictions_{file_name}.png"))


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
plt.savefig(os.path.join( output_dir,f"{ranname}_pred_vs_true_{file_name}.png"))

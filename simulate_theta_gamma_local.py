import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score

# === Load Test Dataset ===
df = pd.read_csv("Data\L_dynamique6y200dis2_0027.csv")
time = df["Time"].values

# === Extract Features (same function as training) ===
def extract_features(df):
    # Position and velocity
    P0 = df[["rod_end X", "rod_end Y", "rod_end Z"]].values / 1000  # anchor point
    P1 = df[["robot_cable_attach_point X", "robot_cable_attach_point Y", "robot_cable_attach_point Z"]].values / 1000
    V1 = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values

    # Time and acceleration
    time_array = df["Time"].values
    # acc_x = np.gradient(df["rob_cor_speed X"].values, time)
    # acc_y = np.gradient(df["rob_cor_speed Y"].values, time)
    # acc_z = np.gradient(df["rob_cor_speed Z"].values, time)

    acc_x = np.gradient(df["rob_cor_speed X"].values, time_array)
    acc_y = np.gradient(df["rob_cor_speed Y"].values, time_array)
    acc_z = np.gradient(df["rob_cor_speed Z"].values, time_array)
    A1 = np.stack([acc_x, acc_y, acc_z], axis=1)

    # Cable direction unit vector
    rel_vec = P1 - P0
    unit_rel = rel_vec / (np.linalg.norm(rel_vec, axis=1, keepdims=True) + 1e-8)

    # Angular states
    theta = df["Theta"].values.reshape(-1, 1)
    gamma = df["Gamma"].values.reshape(-1, 1)
    cos_theta = np.cos(theta)
    sin_gamma = np.sin(gamma)

    # Angle between V1 and cable direction (cosine similarity)
    dot_product = np.sum(V1 * unit_rel, axis=1, keepdims=True)
    norm_v1 = np.linalg.norm(V1, axis=1, keepdims=True) + 1e-8
    angle_proj = dot_product / norm_v1  # projection-based similarity

    # # Cable norm (used in log/sqrt)
    # x14 = np.linalg.norm(P1 - P0, axis=1, keepdims=True)

    # # Apply safe functions manually
    # x14_log = np.log(np.clip(x14, 1e-5, None))
    # x14_sqrt = np.sqrt(np.abs(x14))

    return np.hstack([P1, V1, A1, unit_rel, theta, gamma, cos_theta, sin_gamma, angle_proj])

X_test = extract_features(df)

# === Load Trained Models ===
# model_dtheta_dt = joblib.load("outputs/theta/checkpoint.pkl")
# model_dgamma_dt = joblib.load("outputs/gamma/checkpoint.pkl")

# === Load Trained Models ===
model_dtheta_dt = joblib.load("outputs/C6_2_1KIter_LP_20250414_210401/dgamma_dt/outputs/20250414_211516_fByKxH/checkpoint.pkl")
model_dgamma_dt = joblib.load("outputs/C6_2_1KIter_LP_20250414_210401/dtheta_dt/outputs/20250414_210403_GkBcVG/checkpoint.pkl")

# === Print Equations ===
print("Best Equation for dTheta/dt:")
print(model_dtheta_dt.get_best())

print("\nBest Equation for dGamma/dt:")
print(model_dgamma_dt.get_best())

# === Print LaTeX Equations ===
print("\nLaTeX Equation for dTheta/dt:")
print(model_dtheta_dt.latex())

print("\nLaTeX Equation for dGamma/dt:")
print(model_dgamma_dt.latex())

with open("latex_dtheta_dt.tex", "w") as f:
    f.write(model_dtheta_dt.latex())

with open("latex_dgamma_dt.tex", "w") as f:
    f.write(model_dgamma_dt.latex())

print(model_dtheta_dt.feature_names_in_)
print(X_test.shape[0])
print(model_dgamma_dt.feature_names_in_)
print(X_test.shape)

# === Predict Derivatives ===
dtheta_dt_pred = model_dtheta_dt.predict(X_test)
dgamma_dt_pred = model_dgamma_dt.predict(X_test)

# === Integrate to Get Theta(t) and Gamma(t) ===
theta0 = df["Theta"].values[0]
gamma0 = df["Gamma"].values[0]

theta_est = [theta0]
gamma_est = [gamma0]

for i in range(1, len(time)):
    dt = time[i] - time[i - 1]
    theta_next = theta_est[-1] + dtheta_dt_pred[i - 1] * dt
    gamma_next = gamma_est[-1] + dgamma_dt_pred[i - 1] * dt
    theta_est.append(theta_next)
    gamma_est.append(gamma_next)

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

# === Subplot: True vs Estimated + Error ===
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(time, theta_true, label="True Theta", color="blue")
plt.plot(time, theta_est, '--', label="Predicted Theta", color="red")
plt.ylabel("Theta (rad)")
plt.title("Theta(t) Prediction")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, gamma_true, label="True Gamma", color="blue")
plt.plot(time, gamma_est, '--', label="Predicted Gamma", color="red")
plt.ylabel("Gamma (rad)")
plt.title("Gamma(t) Prediction")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, theta_error, label="Theta Error", color="purple")
plt.plot(time, gamma_error, label="Gamma Error", color="orange")
plt.title("Estimation Error")
plt.ylabel("Error (rad)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# === Scatter Plot ===
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

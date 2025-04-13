import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score

# === Load Dataset ===
df = pd.read_csv("Data\L_dynamique6x100dis2_0035.csv")
time = df["Time"].values

# === Feature Extractor (must match training) ===
def extract_features(df):
    # Position and velocity
    P0 = df[["rod_end X", "rod_end Y", "rod_end Z"]].values / 1000  # anchor point
    P1 = df[["robot_cable_attach_point X", "robot_cable_attach_point Y", "robot_cable_attach_point Z"]].values / 1000
    V1 = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].values

    # Time and acceleration
    time = df["Time"].values
    acc_x = np.gradient(df["rob_cor_speed X"].values, time)
    acc_y = np.gradient(df["rob_cor_speed Y"].values, time)
    acc_z = np.gradient(df["rob_cor_speed Z"].values, time)
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

    return np.hstack([P1, V1, A1, unit_rel, theta, gamma, cos_theta, sin_gamma, angle_proj])

X = extract_features(df)

# === Load Models ===
model_dtheta_dt = joblib.load("outputs\differential_training_new_feature\model_dtheta_dt_20250412_163500.pkl")
model_dgamma_dt = joblib.load("outputs\differential_training_new_feature\model_dgamma_dt_20250412_163500.pkl")

print("LaTeX Equation for dTheta/dt:\n", model_dtheta_dt.latex())
print("\nLaTeX Equation for dGamma/dt:\n", model_dgamma_dt.latex())

# === RK4 Integration ===
def rk4_integration(model, x_input, time, y0):
    y = [y0]
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        x0 = x_input[i - 1]
        x1 = x_input[i]

        def f(x): return model.predict(x.reshape(1, -1))[0]

        k1 = f(x0)
        k2 = f((x0 + x1) / 2)
        k3 = f((x0 + x1) / 2)
        k4 = f(x1)

        y_next = y[-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        y.append(y_next)
    return np.array(y)

# === Integrate dTheta/dt and dGamma/dt
theta0 = df["Theta"].values[0]
gamma0 = df["Gamma"].values[0]

theta_est = rk4_integration(model_dtheta_dt, X, time, theta0)
gamma_est = rk4_integration(model_dgamma_dt, X, time, gamma0)

# === Ground Truth
theta_true = df["Theta"].values
gamma_true = df["Gamma"].values
theta_error = theta_est - theta_true
gamma_error = gamma_est - gamma_true

# === R² Scores
print(f"\nR² Score for Theta(t): {r2_score(theta_true, theta_est):.4f}")
print(f"R² Score for Gamma(t): {r2_score(gamma_true, gamma_est):.4f}")

# === Time Series + Error Plots
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(time, theta_true, label="True Theta", color="blue")
plt.plot(time, theta_est, '--', label="Predicted Theta (RK4)", color="red")
plt.ylabel("Theta (rad)")
plt.title("Theta(t) Prediction")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, gamma_true, label="True Gamma", color="blue")
plt.plot(time, gamma_est, '--', label="Predicted Gamma (RK4)", color="red")
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

# === Scatter Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(theta_true, theta_est, alpha=0.5)
plt.plot([min(theta_true), max(theta_true)], [min(theta_true), max(theta_true)], 'r--')
plt.xlabel("True Theta")
plt.ylabel("Predicted Theta")
plt.title("Theta(t): Pred vs True")

plt.subplot(1, 2, 2)
plt.scatter(gamma_true, gamma_est, alpha=0.5)
plt.plot([min(gamma_true), max(gamma_true)], [min(gamma_true), max(gamma_true)], 'r--')
plt.xlabel("True Gamma")
plt.ylabel("Predicted Gamma")
plt.title("Gamma(t): Pred vs True")

plt.tight_layout()
plt.show()

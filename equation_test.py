import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from sklearn.metrics import r2_score

# === Load Dataset ===
file_name = "L_dynamique6y200dis1_0024.csv"
df = pd.read_csv(f"Data/{file_name}")

# === Extract Time and Angles ===
time = df["Time"].values
theta = df["Theta"].values
gamma = df["Gamma"].values

# === Compute Angular Velocities ===
dtheta = np.gradient(theta, time)
dgamma = np.gradient(gamma, time)

# === Compute Accelerations ===
V_x = df["rob_cor_speed X"].values / 1000
V_y = df["rob_cor_speed Y"].values / 1000
V_z = df["rob_cor_speed Z"].values / 1000
a_x = np.gradient(V_x, time)
a_y = np.gradient(V_y, time)
a_z = np.gradient(V_z, time)

# === Add Lagged Terms ===
theta_t = theta
gamma_t = gamma
theta_t_2 = np.roll(theta, 2)
theta_t_1 = np.roll(theta, 1)
gamma_t_1 = np.roll(gamma, 1)
dtheta_t_2 = np.roll(dtheta, 2)
dgamma_t = dgamma
dgamma_t_2 = np.roll(dgamma, 2)

# === Remove First 2 Rows for Valid Lag ===
lags = 2
valid_idx = slice(lags, None)
time_valid = time[valid_idx]
theta_true = theta[valid_idx]
gamma_true = gamma[valid_idx]


# ddtheta_pred = 0.0752 * (
#     (a_y * (theta_t + 0.4636) - a_x) -
#     (np.sin(np.square(np.square(np.tanh(dgamma_t_2)))) *
#      ((dtheta_t_2 + a_z) - 1.163))
# )

# ddgamma_pred = -0.0547 * (
#     ((a_y * 1.816) -
#      (np.abs(a_y) * ((np.sin(gamma_t * -1.7310137) / 0.55776376) - gamma_t_1))) +
#     np.square(np.tanh(dgamma_t)) +
#     dgamma_t_2
# )


# === Manually Implement Custom Equations ===
ddtheta_pred = 0.0752 * (
    (a_y * (theta_t + 0.4636) - a_x) -
    (np.sin(np.square((np.tanh(dgamma_t_2)))) *
     ((dtheta_t_2 + a_z) - 1.1638))
)

ddgamma_pred = -0.0547 * (
    ((a_y * 1.8169) -
     (a_y * ((np.sin(gamma_t * -1.731) / 0.5578)))) +
    np.sin(np.tanh(gamma_t_1)) +
    dgamma_t_2
)

# === Trim Predictions ===
ddtheta_pred = ddtheta_pred[valid_idx].flatten()
ddgamma_pred = ddgamma_pred[valid_idx].flatten()

# === Double Integration ===
theta_dot = cumulative_trapezoid(ddtheta_pred, time_valid, initial=0)
gamma_dot = cumulative_trapezoid(ddgamma_pred, time_valid, initial=0)
theta_est = theta_true[0] + cumulative_trapezoid(theta_dot, time_valid, initial=0)
gamma_est = gamma_true[0] + cumulative_trapezoid(gamma_dot, time_valid, initial=0)

# === R² Scores ===
theta_r2 = r2_score(theta_true, theta_est)
gamma_r2 = r2_score(gamma_true, gamma_est)

# === Plot Comparison ===
plt.figure(figsize=(14, 6))

plt.subplot(2,1, 1)
plt.plot(time_valid, theta_true, label="True Theta", color="blue")
plt.plot(time_valid, theta_est, '--', label="Predicted Theta", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Theta (rad)")
plt.title(f"Theta(t) Prediction — R² = {theta_r2:.4f}")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_valid, gamma_true, label="True Gamma", color="blue")
plt.plot(time_valid, gamma_est, '--', label="Predicted Gamma", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Gamma (rad)")
plt.title(f"Gamma(t) Prediction — R² = {gamma_r2:.4f}")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(f"manual_prediction_results_{file_name}.png")
plt.show()

print(f"\nR² Score for Theta(t): {theta_r2:.4f}")
print(f"R² Score for Gamma(t): {gamma_r2:.4f}")

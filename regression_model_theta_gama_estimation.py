import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Load your CSV dataset
dataset_path = "Data/L_dynamique1x100dis2_0013.csv"  # Change this to your file path
df = pd.read_csv(dataset_path)

# Prepare data
P0 = df[['rod_end X', 'rod_end Y', 'rod_end Z']].values / 1000  # Convert to meters
P1 = df[['robot_cable_attach_point X', 'robot_cable_attach_point Y', 'robot_cable_attach_point Z']].values / 1000
V1 = df[['rob_speed X', 'rob_speed Y', 'rob_speed Z']].values  # Already in m/s
Time = df['Time'].values
Theta = df['Theta'].values
Gamma = df['Gamma'].values

# Finite difference to compute dTheta/dt and dGamma/dt
dt = np.diff(Time)
dTheta_dt = np.diff(Theta) / dt
dGamma_dt = np.diff(Gamma) / dt

# Input features: P0, P1, V1
features = np.hstack([P0[:-1], P1[:-1], V1[:-1]])

# Train polynomial regression (degree 2)
poly_model_theta = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model_gamma = make_pipeline(PolynomialFeatures(2), LinearRegression())

poly_model_theta.fit(features, dTheta_dt)
poly_model_gamma.fit(features, dGamma_dt)

# Simulate theta/gamma using predicted dtheta/dt and dgamma/dt
theta_sim = [Theta[0]]
gamma_sim = [Gamma[0]]
time_sim = [Time[0]]

for i in range(len(Time) - 1):
    dt_i = Time[i + 1] - Time[i]
    input_features = np.concatenate([P0[i], P1[i], V1[i]]).reshape(1, -1)

    dtheta_dt_pred = poly_model_theta.predict(input_features)[0]
    dgamma_dt_pred = poly_model_gamma.predict(input_features)[0]

    theta_next = theta_sim[-1] + dtheta_dt_pred * dt_i
    gamma_next = gamma_sim[-1] + dgamma_dt_pred * dt_i

    theta_sim.append(theta_next)
    gamma_sim.append(gamma_next)
    time_sim.append(Time[i + 1])

# Save output to CSV
output_df = pd.DataFrame({
    'Time': time_sim,
    'Theta_Actual': Theta,
    'Theta_Simulated': theta_sim,
    'Gamma_Actual': Gamma,
    'Gamma_Simulated': gamma_sim
})

output_csv_path = "theta_gamma_simulation.csv"
output_df.to_csv(output_csv_path, index=False)

print(f"Simulation data saved to: {output_csv_path}")

import matplotlib.pyplot as plt

# Plot Theta comparison
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(Time, Theta, label='Theta Actual', color='blue')
plt.plot(time_sim, theta_sim, label='Theta Simulated', linestyle='--', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('Theta Actual vs Simulated')
plt.legend()

# Plot Gamma comparison
plt.subplot(2, 1, 2)
plt.plot(Time, Gamma, label='Gamma Actual', color='blue')
plt.plot(time_sim, gamma_sim, label='Gamma Simulated', linestyle='--', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Gamma (rad)')
plt.title('Gamma Actual vs Simulated')
plt.legend()

plt.tight_layout()
plt.show()
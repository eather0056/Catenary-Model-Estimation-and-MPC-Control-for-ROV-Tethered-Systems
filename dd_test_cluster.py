import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import pathlib
import os
from main_fun import *
from scipy.integrate import cumulative_trapezoid
from numpy import cumsum

# pathlib.PosixPath = pathlib.WindowsPath
# # === Define The Run Name ===
ranname = "final_test_dd_C6"
# === Load Models and Scaler ===
model_theta = joblib.load("outputs/dd_C6_all_50_ld_20250617_152853/model_dtheta_dt.pkl")  
model_gamma = joblib.load("outputs/dd_C6_Y_1k_ld_20250617_193707/model_dgamma_dt.pkl")  
scaler_theta = joblib.load("outputs/dd_C6_all_50_ld_20250617_152853/scaler.pkl")  
scaler_gamma = joblib.load("outputs/dd_C6_Y_1k_ld_20250617_193707/scaler.pkl")

# # # # # === Define The Run Name ===
# ranname = "C6_all_10k_20250416_103809"
# # === Load Models and Scaler ===
# model_theta = joblib.load("outputs/C6_all_10k_20250416_103809/model_dtheta_dt.pkl")  
# model_gamma = joblib.load("outputs/C6_all_10k_20250416_103809/model_dgamma_dt.pkl")  
# scaler_theta = joblib.load("outputs/C6_all_10k_20250416_103809/scaler.pkl")  
# scaler_gamma = joblib.load("outputs/C6_all_10k_20250416_103809/scaler.pkl")

# Cable parameters from experimental setup (Table I for cable 6)
L = 3.0  # [m]
cable_wet_weight = 1.521  # [N] (From Table I, cable 6: wet weight=1.521N)


# === Load and Preprocess Dataset ===
file_name = "L_dynamique6y200dis1_0024.csv"

output_dir = "Results/mode_test"
output_path = pathlib.Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Load the dataset
file_path = pathlib.Path("Data") / file_name
df = pd.read_csv(file_path)

# # Extract features for DD theta and gamma
X_theta, _ = features_dd(df)
X_gamma, _ = features_dd(df)
y_dtheta_dt, y_dgamma_dt = compute_dderivatives(df)

# Extract features for D theta and gamma
# X_theta = extract_features(df)
# X_gamma = extract_features(df)
# y_dtheta_dt, y_dgamma_dt = compute_derivatives(df)

X_scaled_theta = scaler_theta.transform(X_theta)
X_scaled_gamma = scaler_gamma.transform(X_gamma)

theta_true = df["Theta"].values
gamma_true = df["Gamma"].values
time = df["Time"].values

#21, 23
# === Choose Specific Equation by Index or Filter Criteria ===
selected_eq_theta = model_theta.equations_[model_theta.equations_["complexity"] == 21].iloc[0]
selected_eq_gamma = model_gamma.equations_[model_gamma.equations_["complexity"] == 23].iloc[0]

# Print selected equations
print(f"\n[Custom] eq_dtheta_dt: {selected_eq_theta['equation']}")
print(f"[Custom] eq_dgamma_dt: {selected_eq_gamma['equation']}")


# === Get Lambda Function (if not already callable) ===
def get_lambda_function(eq_row):
    f = eq_row["lambda_format"]
    return f if callable(f) else eval(f)

# === Get Lambda Functions ===
predict_theta_custom = get_lambda_function(selected_eq_theta)
predict_gamma_custom = get_lambda_function(selected_eq_gamma)

# === Predict Derivatives Using Custom Equations ===
dtheta_pred = predict_theta_custom(X_scaled_theta)
dgamma_pred = predict_gamma_custom(X_scaled_gamma)

# # === Predict Derivatives ===
# dtheta_pred = model_theta.predict(X_scaled_theta)
# dgamma_pred = model_gamma.predict(X_scaled_gamma)


# # === First Derivative Initialization (Velocity) ===
# theta_dot = np.zeros_like(dtheta_pred)
# gamma_dot = np.zeros_like(dgamma_pred)

# # === First Integration: Angular Velocities ===
# for i in range(1, len(time)):
#     dt = time[i] - time[i - 1]
#     theta_dot[i] = theta_dot[i - 1] + dtheta_pred[i - 22] * dt
#     gamma_dot[i] = gamma_dot[i - 1] + dgamma_pred[i - 21] * dt

# # === Second Integration: Angular Positions ===
# theta_est = np.zeros_like(dtheta_pred)
# gamma_est = np.zeros_like(dgamma_pred)
# theta_est[0] = df["Theta"].values[0]
# gamma_est[0] = df["Gamma"].values[0]

# for i in range(1, len(time)):
#     dt = time[i] - time[i - 1]
#     theta_est[i] = theta_est[i - 1] + theta_dot[i - 1] * dt
#     gamma_est[i] = gamma_est[i - 1] + gamma_dot[i - 1] * dt

# # === Compare with Ground Truth ===
# theta_error = theta_true - theta_est
# gamma_error = gamma_true - gamma_est

# Align time array with model output shape
lags=2
time_valid = df["Time"].values[lags:]
theta_0 = df["Theta"].values[lags]
gamma_0 = df["Gamma"].values[lags]

# 1st integration
dtheta_pred = cumulative_trapezoid(dtheta_pred, time_valid, initial=0)
dgamma_pred = cumulative_trapezoid(dgamma_pred, time_valid, initial=0)

# 2nd integration
theta_est = theta_0 + cumulative_trapezoid(dtheta_pred, time_valid, initial=0)
gamma_est = gamma_0 + cumulative_trapezoid(dgamma_pred, time_valid, initial=0)

# # # 1st integration
# theta_est = theta_0 + cumulative_trapezoid(dtheta_pred, time_valid, initial=0)
# gamma_est = gamma_0 + cumulative_trapezoid(dgamma_pred, time_valid, initial=0)

# # === Save predicted theta and gamma to original DataFrame ===
df_pred = df.copy()
df_pred = df_pred.iloc[lags:].copy()  # align to valid time range
df_pred["Theta_Pred"] = theta_est
df_pred["Gamma_Pred"] = gamma_est

# Save predicted Theta and Gamma to original CSV in Data folder
output_csv_path = pathlib.Path("Data") / file_name
df_pred.to_csv(output_csv_path, index=False)
print(f"Saved predicted values to: {output_csv_path}")


# Ground truth
theta_true = df["Theta"].values[lags:]
gamma_true = df["Gamma"].values[lags:]

theta_error = theta_true - theta_est
gamma_error = gamma_true - gamma_est

# === Initialize Storage ===
theta_r2_scores = []
gamma_r2_scores = []

nrmse_theta = np.sqrt(np.mean((theta_true - theta_est)**2)) / (theta_true.max() - theta_true.min())
nrmse_gamma = np.sqrt(np.mean((gamma_true - gamma_est)**2)) / (gamma_true.max() - gamma_true.min())


# === print these equeation ===
print(f"\neq_dtheta_dt: {model_theta.get_best()}")
print(f"\neq_dgamma_dt: {model_gamma.get_best()}")

# print(model_theta.equations_[["complexity", "loss", "equation"]])
# print(model_gamma.equations_[["complexity", "loss", "equation"]])

# === R² Scores ===
print(f"\nR² Score for Theta(t): {r2_score(theta_true, theta_est):.4f}")
print(f"R² Score for Gamma(t): {r2_score(gamma_true, gamma_est):.4f}")

# === nRMSE Scores ===
print(f"\nnRMSE Score for Theta(t): {nrmse_theta:.4f}")
print(f"nRMSE Score for Gamma(t): {nrmse_gamma:.4f}")

# === Time-Series Plot ===
plt.figure(figsize=(14, 8))

plt.subplot(4, 2, 1)
plt.plot(time_valid, theta_true, label="True Theta", color="blue")
plt.plot(time_valid, theta_est, '--', label="Predicted Theta", color="red")
plt.ylabel("Theta (rad)")
plt.title("Theta(t) Prediction")
plt.legend()
plt.grid()

plt.subplot(4, 2, 2)
plt.plot(time_valid, gamma_true, label="True Gamma", color="blue")
plt.plot(time_valid, gamma_est, '--', label="Predicted Gamma", color="red")
plt.ylabel("Gamma (rad)")
plt.title("Gamma(t) Prediction")
plt.legend()
plt.grid()

plt.subplot(4, 2, 3)
plt.plot(time_valid, theta_error, label="Theta Error", color="purple")
plt.title("Theta(t)  Estimation Error")
plt.legend()
plt.ylabel("Error (rad)")
plt.xlabel("Time (s)")

plt.subplot(4, 2, 4)
plt.plot(time_valid, gamma_error, label="Gamma Error", color="orange")
plt.title("Gamma(t)  Estimation Error")
plt.ylabel("Error (rad)")
plt.xlabel("Time (s)")
plt.legend()

theta_percentage_error = (theta_error / theta_true) * 100
gamma_percentage_error = (gamma_error / gamma_true) * 100

plt.subplot(4, 2, 5)
plt.plot(time_valid, theta_percentage_error, label="Theta % Error", color="purple")
plt.title("Theta(t) Estimation Percentage Error")
plt.legend()
plt.ylabel("Percentage Error (%)")
plt.xlabel("Time (s)")

plt.subplot(4, 2, 6)
plt.plot(time_valid, gamma_percentage_error, label="Gamma % Error", color="orange")
plt.title("Gamma(t)  Estimation Percentage Error")
plt.ylabel("Percentage Error (%)")
plt.xlabel("Time (s)")
plt.legend()

plt.subplot(4, 2, 7)
plt.plot(time, y_dtheta_dt, label="True ddTheta", color="blue")
plt.plot(time_valid, dtheta_pred, '--', label="Predicted ddTheta", color="red")
plt.ylabel("ddTheta (rad/s)")
plt.title("ddt Theta(t) Prediction")
plt.legend()
plt.grid()

plt.subplot(4, 2, 8)
plt.plot(time, y_dgamma_dt, label="True ddGamma", color="blue")
plt.plot(time_valid, dgamma_pred, '--', label="Predicted ddGamma", color="red")
plt.ylabel("ddGamma (rad/s)")
plt.title("ddt Gamma(t) Prediction")
plt.legend()
plt.grid()


plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{ranname}_predictions_{file_name}.png"))


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

# === Loop Over All Theta Equations ===
for idx, row in model_theta.equations_.iterrows():
    eq_func = row["lambda_format"]  # use directly
    dtheta_pred = eq_func(X_scaled_theta)

    theta_dot = cumulative_trapezoid(dtheta_pred, time_valid, initial=0)
    theta_est = cumulative_trapezoid(theta_dot, time_valid)
    theta_est = np.insert(theta_est, 0, theta_true[0])


    r2 = r2_score(theta_true, theta_est)
    theta_r2_scores.append((row["complexity"], row["equation"], r2))


# === Loop Over All Gamma Equations ===
for idx, row in model_gamma.equations_.iterrows():
    eq_func = row["lambda_format"]  # use directly
    dgamma_pred = eq_func(X_scaled_gamma)

    gamma_dot = cumulative_trapezoid(dgamma_pred, time_valid, initial=0)
    gamma_est = cumulative_trapezoid(gamma_dot, time_valid)
    gamma_est = np.insert(gamma_est, 0, gamma_true[0])


    r2 = r2_score(gamma_true, gamma_est)
    gamma_r2_scores.append((row["complexity"], row["equation"], r2))


# === Display Results ===
print("\nTheta R² Scores:")
for c, eq, r2 in theta_r2_scores:
    print(f"Complexity {c:2d} | R² = {r2:.4f} | {eq}")

print("\nGamma R² Scores:")
for c, eq, r2 in gamma_r2_scores:
    print(f"Complexity {c:2d} | R² = {r2:.4f} | {eq}")


theta_r2_scores.append((row["complexity"], row["equation"], r2))
gamma_r2_scores.append((row["complexity"], row["equation"], r2))

# === Convert to DataFrames ===
df_theta = pd.DataFrame(theta_r2_scores, columns=["complexity", "equation", "r2_score"])
df_gamma = pd.DataFrame(gamma_r2_scores, columns=["complexity", "equation", "r2_score"])

# === Optional: Keep best R² per complexity ===
df_theta = df_theta.groupby("complexity", as_index=False).max()
df_gamma = df_gamma.groupby("complexity", as_index=False).max()

# === Pareto Frontier Plot ===
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(df_theta["complexity"], df_theta["r2_score"], label="Theta (ddθ)", marker="o", color="blue")
plt.title("Complexity vs R² Score for Theta (ddθ)")
plt.xlabel("Model Complexity")
plt.ylabel("R² Score")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df_gamma["complexity"], df_gamma["r2_score"], label="Gamma (ddγ)", marker="o", color="orange")
plt.title("Complexity vs R² Score for Gamma (ddγ)")
plt.xlabel("Model Complexity")
plt.ylabel("R² Score")
plt.grid(True)
plt.tight_layout()

# Save the figure
pareto_path = os.path.join(output_dir, f"{ranname}_pareto_frontier.png")
plt.savefig(pareto_path)
print(f"Pareto frontier plot saved to: {pareto_path}")

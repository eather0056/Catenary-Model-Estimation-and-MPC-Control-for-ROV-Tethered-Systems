# This script is used to test the integration of the trained models on a new dataset.
import joblib
from main_fun import *
import pathlib


# === Load Test Data ===
df_test = pd.read_csv("Data/L_dynamique6x200dis2_0031.csv")
X_test = extract_features(df_test)
print(f"Theta feature shape: {X_test.shape}")

time_array, theta_true, gamma_true = preprocess_signals(df_test, sigma=2)

pathlib.PosixPath = pathlib.WindowsPath
# === Load Trained Models ===
model_dgamma_dt = joblib.load("outputs/C6_2_1KIter_11f_20250415_003430/dtheta_dt/outputs/20250415_003434_2Gl798/checkpoint.pkl")
model_dtheta_dt = joblib.load("outputs/C6_2_1KIter_11f_20250415_003430/dtheta_dt/outputs/20250415_003434_2Gl798/checkpoint.pkl")

# === Initial Values from Data ===
theta_0 = theta_true[0]
gamma_0 = gamma_true[0]

# === Integrate the Equations ===
theta_pred, gamma_pred = integrate_theta_gamma(
    model_theta=model_dtheta_dt,
    model_gamma=model_dgamma_dt,
    X=X_test,
    time_array=time_array,
    theta_0=theta_0,
    gamma_0=gamma_0
)

# === Evaluation Metrics ===
print(f"R² Score for Theta(t): {r2_score(theta_true, theta_pred):.4f}")
print(f"R² Score for Gamma(t): {r2_score(gamma_true, gamma_pred):.4f}")
print(f"Theta feature name: {model_dtheta_dt.feature_names_in_}")


# === Plot Results ===
plot_integration(time_array, theta_true, theta_pred, gamma_true, gamma_pred)

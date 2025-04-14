# This script is used to test the integration of the trained models on a new dataset.
import joblib
from main_fun import *

# === Load Test Data ===
df_test = pd.read_csv("Data/L_dynamique6x100dis2_0035.csv")
X_test = extract_features(df_test)
time_array, theta_true, gamma_true = preprocess_signals(df_test, sigma=2)

# === Load Trained Models ===
model_dtheta_dt = joblib.load("outputs/20250412_214744_lpGawP/checkpoint.pkl")
model_dgamma_dt = joblib.load("outputs/20250412_225159_GufPvS/checkpoint.pkl")

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

# === Plot Results ===
plot_integration(time_array, theta_true, theta_pred, gamma_true, gamma_pred)

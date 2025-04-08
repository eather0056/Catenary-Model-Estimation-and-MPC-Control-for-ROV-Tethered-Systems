import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# === Step 1: Load and Combine Training Datasets ===
train_files = [
    "Data/L_dynamique1x100dis2_0013.csv",
    "Data/L_dynamique1x100dis2_0014.csv",
    "Data/L_dynamique1y200dis1_0004.csv",
    "Data/L_dynamique1x200dis2_0011.csv",
    "Data/L_dynamique1x200dis2_0012.csv",
    "Data/L_dynamique1y200dis1_0005.csv",
    "Data/L_dynamique1y200dis1_0006.csv"
]

train_dfs = [pd.read_csv(f) for f in train_files]
df_train = pd.concat(train_dfs, ignore_index=True)

# Remove any rows where Theta or Gamma is NaN
df_train = df_train.dropna(subset=['Theta', 'Gamma'])

# Prepare training data
P0_train = df_train[['rod_end X', 'rod_end Y', 'rod_end Z']].values / 1000
P1_train = df_train[['robot_cable_attach_point X', 'robot_cable_attach_point Y', 'robot_cable_attach_point Z']].values / 1000
V1_train = df_train[['rob_speed X', 'rob_speed Y', 'rob_speed Z']].values
Theta_train = df_train['Theta'].values
Gamma_train = df_train['Gamma'].values

features_train = np.hstack([P0_train, P1_train, V1_train])

# Train direct prediction models
model_theta_direct = RandomForestRegressor(n_estimators=100, random_state=42)
model_gamma_direct = RandomForestRegressor(n_estimators=100, random_state=42)

model_theta_direct.fit(features_train, Theta_train)
model_gamma_direct.fit(features_train, Gamma_train)

# === Step 2: Define Testing + Visualization Function ===
def test_direct_model(test_file, model_theta, model_gamma):
    df_test = pd.read_csv(test_file)
    P0 = df_test[['rod_end X', 'rod_end Y', 'rod_end Z']].values / 1000
    P1 = df_test[['robot_cable_attach_point X', 'robot_cable_attach_point Y', 'robot_cable_attach_point Z']].values / 1000
    V1 = df_test[['rob_speed X', 'rob_speed Y', 'rob_speed Z']].values
    Time = df_test['Time'].values
    Theta = df_test['Theta'].values
    Gamma = df_test['Gamma'].values

    features = np.hstack([P0, P1, V1])
    theta_pred = model_theta.predict(features)
    gamma_pred = model_gamma.predict(features)

    # Save CSV
    base_name = os.path.basename(test_file).replace(".csv", "")
    output_csv_path = f"direct_theta_gamma_{base_name}.csv"
    output_df = pd.DataFrame({
        'Time': Time,
        'Theta_Actual': Theta,
        'Theta_Predicted': theta_pred,
        'Gamma_Actual': Gamma,
        'Gamma_Predicted': gamma_pred
    })
    output_df.to_csv(output_csv_path, index=False)

    # Plot
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(Time, Theta, label='Theta Actual', color='blue')
    plt.plot(Time, theta_pred, label='Theta Predicted', linestyle='--', color='red')
    plt.title(f'Theta Comparison - {base_name}')
    plt.ylabel("Theta (rad)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(Time, Gamma, label='Gamma Actual', color='blue')
    plt.plot(Time, gamma_pred, label='Gamma Predicted', linestyle='--', color='red')
    plt.title(f'Gamma Comparison - {base_name}')
    plt.ylabel("Gamma (rad)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(Time, theta_pred - Theta, label='Theta Error', color='purple')
    plt.plot(Time, gamma_pred - Gamma, label='Gamma Error', color='orange')
    plt.title(f'Prediction Error - {base_name}')
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.legend()

    plt.tight_layout()
    plt.show()
    print(f"✅ Saved: {output_csv_path}")

# === Step 3: Run Test ===
test_direct_model("Data/L_dynamique1x100dis2_0015.csv", model_theta_direct, model_gamma_direct)

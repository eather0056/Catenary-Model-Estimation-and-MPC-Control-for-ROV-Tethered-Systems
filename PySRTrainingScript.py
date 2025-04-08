# This script trains symbolic regression models using PySR for predicting Theta and Gamma angles based on the provided dataset.
# It includes data preprocessing, model training, and evaluation steps. The script also generates plots for model interpretation and convergence analysis. 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pysr import PySRRegressor

# === Step 1: Load and Combine Training Datasets ===
train_files = [
    "Data/L_dynamique6x100dis2_0033.csv",  
    "Data/L_dynamique6x100dis2_0034.csv",  
    "Data/L_dynamique6x100dis2_0035.csv",  
    "Data/L_dynamique6x200dis2_0030.csv",  
    "Data/L_dynamique6x200dis2_0031.csv",  
    "Data/L_dynamique6x200dis2_0032.csv",  
    "Data/L_dynamique6y100dis1_0018.csv",  
    "Data/L_dynamique6y100dis1_0019.csv",  
    "Data/L_dynamique6y100dis1_0020.csv",  
    "Data/L_dynamique6y100dis2_0021.csv",  
    "Data/L_dynamique6y100dis2_0022.csv",  
    "Data/L_dynamique6y100dis2_0023.csv",  
    "Data/L_dynamique6y200dis1_0025.csv",  
    "Data/L_dynamique6y200dis1_0026.csv",  
    "Data/L_dynamique6y200dis2_0027.csv",  
    "Data/L_dynamique6y200dis2_0028.csv",  
    "Data/L_dynamique6y200dis2_0029.csv"  
]
train_dfs = [pd.read_csv(f) for f in train_files]
df_train = pd.concat(train_dfs, ignore_index=True)
df_train = df_train.dropna(subset=['Theta', 'Gamma'])

# === Step 2: Feature Engineering ===
def extract_features(df):
    P0 = df[['rod_end X', 'rod_end Y', 'rod_end Z']].values / 1000
    P1 = df[['robot_cable_attach_point X', 'robot_cable_attach_point Y', 'robot_cable_attach_point Z']].values / 1000
    # V1 = df[['rob_speed X', 'rob_speed Y', 'rob_speed Z']].values
    V1 = df[['rob_cor_speed X', 'rob_cor_speed Y', 'rob_cor_speed Z']].values
    rel_vec = P1 - P0
    cable_len = np.linalg.norm(rel_vec, axis=1).reshape(-1, 1)
    speed_mag = np.linalg.norm(V1, axis=1).reshape(-1, 1)
    features = np.hstack([P0, P1, V1, rel_vec, cable_len, speed_mag])
    return features

X_train = extract_features(df_train)
y_theta_train = df_train['Theta'].values
y_gamma_train = df_train['Gamma'].values

# === Step 3: Train Symbolic Regression Models ===
model_sym_theta = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log"],
    model_selection="best",
    elementwise_loss="loss(x, y) = (x - y)^2",
    verbosity=1,
    random_state=42,
    deterministic=True,
    parallelism='serial',
    batching=True,
    batch_size=5000,
)

model_sym_gamma = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log"],
    model_selection="best",
    elementwise_loss="loss(x, y) = (x - y)^2",
    verbosity=1,
    random_state=42,
    deterministic=True,
    parallelism='serial',
    batching=True,
    batch_size=5000,
)

print("Training symbolic model for Theta...")
model_sym_theta.fit(X_train, y_theta_train)
print("\nBest equation for Theta:")
print(model_sym_theta.get_best())

print("\nTraining symbolic model for Gamma...")
model_sym_gamma.fit(X_train, y_gamma_train)
print("\nBest equation for Gamma:")
print(model_sym_gamma.get_best())

# === Step 3.5: Training Interpretation Plots ===
def interpret_training(model, X, y, label):
    y_pred = model.predict(X)
    error = y_pred - y

    plt.figure(figsize=(16, 8))

    # 1. Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y, y_pred, alpha=0.4, label=label)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label="Ideal")
    plt.xlabel(f"{label} Actual")
    plt.ylabel(f"{label} Predicted")
    plt.title(f"{label} - Predicted vs Actual (Training)")
    plt.legend()

    # 2. Residuals
    plt.subplot(1, 2, 2)
    plt.hist(error, bins=50, color='purple', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title(f"{label} - Residual Distribution")

    plt.tight_layout()
    plt.show()

    # Optional: Show top 5 equations from Pareto front
    print(f"\nTop 5 equations for {label}:")
    print(model)

# Plot training interpretation for Theta
interpret_training(model_sym_theta, X_train, y_theta_train, label="Theta")

# Plot training interpretation for Gamma
interpret_training(model_sym_gamma, X_train, y_gamma_train, label="Gamma")

def plot_convergence(model, label):
    results = model.equation_search_results
    if results.emplty:
        print(f"No convergence data available for {label}.")
        return
    
    complexities = results['complexity']
    losses = results['loss']
    scores = results.get('score', None)

    plt.figure(figsize=(10,6))
    plt.scatter(complexities, losses, c='blue', alpha=0.6, label="Equations")

    # Highlight the best equation
    best = results.loc[results['loss'].indxmin()]
    plt.scatter([best['complexity']], [best['loss']], color='red', s=80, label="Best Equation")

    plt.title(f"{label} - Convergence Plot\n(complexity vs loss)")
    plt.xlabel("Equation Complexity")
    plt.ylabel("loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    

# Plot convergence for Theta
plot_convergence(model_sym_theta, label="Theta")

# Plot convergence for Gamma
plot_convergence(model_sym_gamma, label="Gamma")

# === Step 4: Define Test + Plot Function ===
def test_and_plot_symbolic(test_file, model_sym_theta, model_sym_gamma):
    df_test = pd.read_csv(test_file).dropna(subset=['Theta', 'Gamma'])
    X_test = extract_features(df_test)
    Time = df_test['Time'].values
    Theta = df_test['Theta'].values
    Gamma = df_test['Gamma'].values

    theta_pred = model_sym_theta.predict(X_test)
    gamma_pred = model_sym_gamma.predict(X_test)

    base_name = os.path.basename(test_file).replace(".csv", "")
    output_df = pd.DataFrame({
        'Time': Time,
        'Theta_Actual': Theta,
        'Theta_Predicted': theta_pred,
        'Gamma_Actual': Gamma,
        'Gamma_Predicted': gamma_pred
    })
    output_df.to_csv(f"symbolic_theta_gamma_{base_name}.csv", index=False)

    # Plotting
    error_theta = theta_pred - Theta
    error_gamma = gamma_pred - Gamma

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
    plt.plot(Time, error_theta, label='Theta Error', color='purple')
    plt.plot(Time, error_gamma, label='Gamma Error', color='orange')
    plt.title(f'Prediction Error - {base_name}')
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Saved: symbolic_theta_gamma_{base_name}.csv")
    return r2_score(Theta, theta_pred), r2_score(Gamma, gamma_pred)

# === Step 5: Test on New Dataset ===
test_file = "Data/L_dynamique6y200dis1_0024.csv"
r2_theta, r2_gamma = test_and_plot_symbolic(test_file, model_sym_theta, model_sym_gamma)
print(f"\nR² Score - Theta: {r2_theta:.4f}, Gamma: {r2_gamma:.4f}")

import pandas as pd
import numpy as np
import os

# === Directory with data files ===
data_dir = "Data"

# === Loop through all CSV files in the directory ===
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_dir, filename)
        print(f"Processing file: {filename}")

        # === Load and clean data ===
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        # === Drop unnecessary columns if they exist ===
        cols_to_drop = ["vx_c", "vy_c", "vz_c"]
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        # === Check if necessary rotation matrix and velocity columns exist ===
        required_cols = ['rob_speed X', 'rob_speed Y', 'rob_speed Z',
                         'exc1', 'eyc1', 'ezc1',
                         'exc2', 'eyc2', 'ezc2',
                         'exc3', 'eyc3', 'ezc3']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns in {filename}. Skipping...")
            continue

        # === Extract robot velocity ===
        rob_speed = df[['rob_speed X', 'rob_speed Y', 'rob_speed Z']].values
        rob_cor_speed = []

        # === Apply rotation matrix at each time step ===
        for t in range(len(df)):
            try:
                R = np.array([
                    [df.at[t, 'exc1'], df.at[t, 'eyc1'], df.at[t, 'ezc1']],
                    [df.at[t, 'exc2'], df.at[t, 'eyc2'], df.at[t, 'ezc2']],
                    [df.at[t, 'exc3'], df.at[t, 'eyc3'], df.at[t, 'ezc3']],
                ])

                v_world = rob_speed[t]
                v_corrected = R @ v_world  # Rotate velocity
                rob_cor_speed.append(v_corrected)
            except Exception as e:
                print(f"Error at row {t} in {filename}: {e}")
                rob_cor_speed.append([np.nan, np.nan, np.nan])

        # === Convert to numpy and add corrected velocity to DataFrame ===
        rob_cor_speed = np.array(rob_cor_speed)
        df["rob_cor_speed X"] = rob_cor_speed[:, 0]
        df["rob_cor_speed Y"] = rob_cor_speed[:, 1]
        df["rob_cor_speed Z"] = rob_cor_speed[:, 2]

        # === Save corrected data, overwriting original file ===
        df.to_csv(filepath, index=False)
        print(f"Corrected data saved to {filename}\n")

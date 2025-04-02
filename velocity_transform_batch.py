# This script processes multiple CSV files containing robot speed and cable point data.
# #         num_valid_per_frame.append(valid_count)
import pandas as pd
import numpy as np
import os

# === Define cable point correction function ===
def compute_rotation_kabsch(P, Q):
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

# === List of files to process ===
csv_files = [
    "L_dynamique6x100dis2_0033.csv", "L_dynamique6x100dis2_0034.csv", "L_dynamique6x100dis2_0035.csv",
    "L_dynamique6x200dis2_0030.csv", "L_dynamique6x200dis2_0031.csv", "L_dynamique6x200dis2_0032.csv",
    "L_dynamique6y100dis1_0018.csv", "L_dynamique6y100dis1_0019.csv", "L_dynamique6y100dis1_0020.csv",
    "L_dynamique6y100dis2_0021.csv", "L_dynamique6y100dis2_0022.csv", "L_dynamique6y100dis2_0023.csv",
    "L_dynamique6y200dis1_0024.csv", "L_dynamique6y200dis1_0025.csv", "L_dynamique6y200dis1_0026.csv",
    "L_dynamique6y200dis2_0027.csv", "L_dynamique6y200dis2_0028.csv", "L_dynamique6y200dis2_0029.csv"
]

data_dir = "Data"

# === Process each file ===
for filename in csv_files:
    filepath = os.path.join(data_dir, filename)
    print(f"\n🔧 Processing {filename}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Convert cable point columns to numeric
    for i in range(16):
        for axis in ['X', 'Y', 'Z']:
            for prefix in ["cable_", "cable_cor_"]:
                col = f"{prefix}{i} {axis}"
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Extract cable points
    original_points = []
    corrected_points = []
    for i in range(16):
        orig_cols = [f"cable_{i} {axis}" for axis in ['X', 'Y', 'Z']]
        cor_cols = [f"cable_cor_{i} {axis}" for axis in ['X', 'Y', 'Z']]
        if all(col in df.columns for col in orig_cols + cor_cols):
            original_points.append(df[orig_cols].values)
            corrected_points.append(df[cor_cols].values)

    if len(original_points) < 3:
        print("⚠️  Not enough cable points, skipping file.")
        continue

    # Stack as arrays: (n_samples, n_points, 3)
    original_points = np.stack(original_points, axis=1)
    corrected_points = np.stack(corrected_points, axis=1)

    rob_speed = df[["rob_speed X", "rob_speed Y", "rob_speed Z"]].values
    rob_cor_speed = []

    bad_frames = 0

    for t in range(len(df)):
        P = original_points[t]
        Q = corrected_points[t]

        if not (np.isfinite(P).all() and np.isfinite(Q).all()):
            rob_cor_speed.append([np.nan, np.nan, np.nan])
            bad_frames += 1
            continue

        if P.shape[0] < 3:
            rob_cor_speed.append([np.nan, np.nan, np.nan])
            bad_frames += 1
            continue

        if np.linalg.norm(P - Q) < 1e-6:
            # Too little movement, skip
            rob_cor_speed.append([np.nan, np.nan, np.nan])
            bad_frames += 1
            continue

        try:
            R = compute_rotation_kabsch(P, Q)

            if not np.allclose(R.T @ R, np.eye(3), atol=1e-2) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-2):
                print(f"⚠️  Frame {t}: Bad rotation matrix (det={np.linalg.det(R):.3f})")
                rob_cor_speed.append([np.nan, np.nan, np.nan])
                bad_frames += 1
                continue

            v_world = rob_speed[t]
            v_corrected = R @ v_world
            rob_cor_speed.append(v_corrected)

        except np.linalg.LinAlgError:
            print(f"❌ Frame {t}: SVD failure")
            rob_cor_speed.append([np.nan, np.nan, np.nan])
            bad_frames += 1

    rob_cor_speed = np.array(rob_cor_speed)
    df["rob_cor_speed X"] = rob_cor_speed[:, 0]
    df["rob_cor_speed Y"] = rob_cor_speed[:, 1]
    df["rob_cor_speed Z"] = rob_cor_speed[:, 2]

    df.to_csv(filepath, index=False)
    print(f"✅ Finished {filename} — bad frames skipped: {bad_frames}")

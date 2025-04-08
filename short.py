# This script checks for NaN values in the corrected velocity columns of multiple CSV files.
#             # Calculate the corrected speed using the provided formula
import pandas as pd
import os

csv_files = [
    "L_dynamique6x100dis2_0033.csv", "L_dynamique6x100dis2_0034.csv", "L_dynamique6x100dis2_0035.csv",
    "L_dynamique6x200dis2_0030.csv", "L_dynamique6x200dis2_0031.csv", "L_dynamique6x200dis2_0032.csv",
    "L_dynamique6y100dis1_0018.csv", "L_dynamique6y100dis1_0019.csv", "L_dynamique6y100dis1_0020.csv",
    "L_dynamique6y100dis2_0021.csv", "L_dynamique6y100dis2_0022.csv", "L_dynamique6y100dis2_0023.csv",
    "L_dynamique6y200dis1_0024.csv", "L_dynamique6y200dis1_0025.csv", "L_dynamique6y200dis1_0026.csv",
    "L_dynamique6y200dis2_0027.csv", "L_dynamique6y200dis2_0028.csv", "L_dynamique6y200dis2_0029.csv"
]

data_dir = "Data"
total_nan_summary = {}

print("Checking for NaN in corrected velocity columns:\n")

for filename in csv_files:
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)

    if not all(col in df.columns for col in ["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]):
        print(f"❌ {filename}: Missing one or more corrected speed columns.")
        continue

    n_nans = df[["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]].isna().sum().sum()
    total_nan_summary[filename] = n_nans

    if n_nans == 0:
        print(f"✅ {filename}: No NaN values found.")
    else:
        print(f"⚠️  {filename}: {n_nans} NaN values found in corrected speed columns.")

# === Summary ===
print("\nSummary of files with NaNs:")
for file, count in total_nan_summary.items():
    if count > 0:
        print(f"- {file}: {count} NaNs")



###################################################################################
# ## === Step 1: Check for NaN values in corrected velocity columns ===
# import pandas as pd
# import numpy as np
# import os   

# def check_valid_points_per_frame(df, filename):
#     print(f"\nChecking valid cable points per time step for: {filename}")
#     num_valid_per_frame = []

#     for t in range(len(df)):
#         valid_count = 0
#         for i in range(16):
#             cols = [f"cable_{i} {axis}" for axis in ['X', 'Y', 'Z']] + [f"cable_cor_{i} {axis}" for axis in ['X', 'Y', 'Z']]
#             if all(col in df.columns for col in cols):
#                 if df.loc[t, cols].notna().all():
#                     valid_count += 1
#         num_valid_per_frame.append(valid_count)

#     num_valid_per_frame = np.array(num_valid_per_frame)
#     print(f"Time steps with < 3 valid cable points: {(num_valid_per_frame < 3).sum()} out of {len(df)}")
#     print(f"Minimum points available in any frame: {num_valid_per_frame.min()}")
#     print(f"Median valid points per frame: {np.median(num_valid_per_frame)}")


# df_problem = pd.read_csv("Data/L_dynamique6y200dis1_0025.csv")
# df_problem.columns = df_problem.columns.str.strip()
# check_valid_points_per_frame(df_problem, "L_dynamique6y200dis1_0025.csv")

####################################################################################################

# # === Step 1: Interpolate NaN values in corrected speed columns ===
# import pandas as pd
# import os

# csv_files = [
#     "L_dynamique6x100dis2_0033.csv", "L_dynamique6x100dis2_0034.csv", "L_dynamique6x100dis2_0035.csv",
#     "L_dynamique6x200dis2_0030.csv", "L_dynamique6x200dis2_0031.csv", "L_dynamique6x200dis2_0032.csv",
#     "L_dynamique6y100dis1_0018.csv", "L_dynamique6y100dis1_0019.csv", "L_dynamique6y100dis1_0020.csv",
#     "L_dynamique6y100dis2_0021.csv", "L_dynamique6y100dis2_0022.csv", "L_dynamique6y100dis2_0023.csv",
#     "L_dynamique6y200dis1_0024.csv", "L_dynamique6y200dis1_0025.csv", "L_dynamique6y200dis1_0026.csv",
#     "L_dynamique6y200dis2_0027.csv", "L_dynamique6y200dis2_0028.csv", "L_dynamique6y200dis2_0029.csv"
# ]

# data_dir = "Data"

# for filename in csv_files:
#     filepath = os.path.join(data_dir, filename)
#     print(f"\n📈 Interpolating: {filename}")

#     df = pd.read_csv(filepath)
#     df.columns = df.columns.str.strip()

#     target_cols = ["rob_cor_speed X", "rob_cor_speed Y", "rob_cor_speed Z"]

#     if not all(col in df.columns for col in target_cols):
#         print(f"⚠️  Skipping {filename}: Missing corrected speed columns.")
#         continue

#     # Interpolate linearly (in-place)
#     df[target_cols] = df[target_cols].interpolate(method='linear', limit_direction='both')

#     # Optional: check remaining NaNs
#     nan_remaining = df[target_cols].isna().sum().sum()
#     if nan_remaining > 0:
#         print(f"⚠️  {nan_remaining} NaNs could not be interpolated (likely at edges).")

#     df.to_csv(filepath, index=False)
#     print(f"✅ Interpolation complete and saved: {filename}")

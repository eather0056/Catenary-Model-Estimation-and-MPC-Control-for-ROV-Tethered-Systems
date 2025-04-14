
import numpy as np
import pandas as pd

# === Load & Preprocess ===
def load_and_concat(files):
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df.dropna(subset=["Theta", "Gamma", "Time"])

# all_files = sorted(glob.glob("/home/mundus/mdeowan698/Catenary_Dynamic/Data/*.csv"))
# train_files, test_files = train_test_split(all_files, test_size=0.9, random_state=42)


# === Load and Combine Training Datasets ===
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

df_train = load_and_concat(train_files)

dt = np.diff(df_train["Time"].values)
print("Time step variance:", np.var(dt))

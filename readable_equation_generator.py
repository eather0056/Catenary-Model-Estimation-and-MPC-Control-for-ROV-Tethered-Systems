# This script reads a CSV file containing symbolic equations and converts them into a more human-readable format.
# # It also prints the top N equations based on their complexity and loss values.
# # The equations are formatted to replace symbolic variables with descriptive names.
import pandas as pd
import re

# === Step 1: Define readable feature names ===
feature_names = [
    "x_0", "y_0", "z_0",        # rod_end (P0)
    "x_1", "y_1", "z_1",        # robot_attach_point (P1)
    "v_x", "v_y", "v_z",        # robot speed (V1)
    "r_x", "r_y", "r_z",        # relative vector (P1 - P0)
    "L",                        # cable length
    "V"                         # speed magnitude
]

# === Step 2: Helper function to prettify symbolic equations ===
def pretty_print_equation(equation, feature_names):
    for i, name in enumerate(feature_names):
        # Replace both x₀ (unicode) and x0 (ascii) variants
        equation = re.sub(fr'x₍?{i}₎?', name, equation)
        equation = equation.replace(f'x{i}', name)
    return equation

# === Step 3: Load hall_of_fame.csv ===
hall_of_fame_path = "outputs/20250401_112648_v9XJrt/hall_of_fame.csv"  # Adjust path if needed
df_eqs = pd.read_csv(hall_of_fame_path)

# Print column names to debug (optional)
print("Available columns:", df_eqs.columns.tolist())

# === Step 4: Print top N readable equations ===
N = 10  # You can change this
for i in range(min(N, len(df_eqs))):
    raw_eq = df_eqs.loc[i, 'Equation']  # Capital 'E' — adjust if needed
    pretty = pretty_print_equation(raw_eq, feature_names)
    loss = df_eqs.loc[i, 'Loss']
    complexity = df_eqs.loc[i, 'Complexity']
    print(f"\n[Complexity: {complexity}, Loss: {loss:.2e}]\n  y = {pretty}")

import pandas as pd
import re

# Load the CSV file
csv_path = "outputs/C6_all_10k_20250416_103809/dgamma_results.csv"
df = pd.read_csv(csv_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Correct variable mapping
variable_mapping = {
    "x0": "P_{1x}", "x1": "P_{1y}", "x2": "P_{1z}",
    "x3": "V_{1x}", "x4": "V_{1y}", "x5": "V_{1z}",
    "x6": "A_{1x}", "x7": "A_{1y}", "x8": "A_{1z}",
    "x9": "\\hat{r}_x", "x10": "\\hat{r}_y", "x11": "\\hat{r}_z",
    "x12": "T", "x13": "\\cos(\\phi)",
    "x14": "\\theta", "x15": "\\gamma",
    "x16": "\\theta_{t-1}", "x17": "\\gamma_{t-1}"
}

# Escape backslashes in LaTeX variable names to avoid regex issues
escaped_variable_mapping = {
    k: latex_var.replace("\\", r"\\") for k, latex_var in variable_mapping.items()
}

# Replace the variables in the equations with LaTeX-safe names
def convert_equation_to_latex_fixed(eq):
    for var, latex_var in escaped_variable_mapping.items():
        # replace whole words (e.g., x1, not inside x10)
        eq = re.sub(rf"\b{var}\b", "{" + latex_var + "}", eq)
    return f"${eq}$"

# Convert equations to LaTeX
df["latex_equation"] = df["equation"].apply(convert_equation_to_latex_fixed)

# Save the result
latex_output_path = "outputs/C6_all_10k_20250416_103809/latex_equations_dgamma.csv"
df.to_csv(latex_output_path, index=False)

print("Saved to:", latex_output_path)

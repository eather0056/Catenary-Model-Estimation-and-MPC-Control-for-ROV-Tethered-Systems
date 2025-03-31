from pysr import PySRRegressor

# Load the already-trained model (same config as before)
from pysr import PySRRegressor

model_sym_theta = PySRRegressor.from_file(
    run_directory="outputs/20250331_152142_L2XeZk",
    model_selection="best"  # ⬅️ valid option
)

# Show the best equation based on lowest loss
print(model_sym_theta.get_best())
print(model_sym_theta.sympy())
print(model_sym_theta.latex())

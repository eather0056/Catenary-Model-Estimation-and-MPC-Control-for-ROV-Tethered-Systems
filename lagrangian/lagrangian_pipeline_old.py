import numpy as np
from pysr import PySRRegressor
import sympy as sp
from main_fun import *

from pysr import PySRRegressor

class LagrangianPipeline:
    def __init__(self, model_params=None):
        default_params = dict(
            niterations=1000,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos"],
            complexity_of_operators={"+": 1, "-": 1, "*": 2, "/": 5},
            loss="loss(x, y) = (x - y)^2",
            model_selection="best",
            verbosity=1,
            procs=0,
            maxsize=30,
            extra_sympy_mappings={"log": "log"},
        )
        final_params = default_params if model_params is None else {**default_params, **model_params}
        self.model = PySRRegressor(**final_params)  # <--- this must exist


    def prepare_data(self, df):
        # Extract features
        features = extract_features(df)
        self.X = features[:, :-2]
        self.theta = df["Theta"].values
        self.gamma = df["Gamma"].values
        self.time = df["Time"].values

        self.dtheta = np.gradient(self.theta, self.time)
        self.dgamma = np.gradient(self.gamma, self.time)
        self.ddtheta = np.gradient(self.dtheta, self.time)
        self.ddgamma = np.gradient(self.dgamma, self.time)

        # Lagrangian input vector: [θ, γ, dθ, dγ]
        self.X_lagr = np.column_stack([self.theta, self.gamma, self.dtheta, self.dgamma])

    def train_lagrangian(self):
        print("Training symbolic Lagrangian with PySR...")

        # Manually seed the model via feature design
        # Generate synthetic features inspired by known expressions
        seed_1 = self.X_lagr[:, 2]**2 + self.X_lagr[:, 3]**2     # dθ² + dγ²
        seed_2 = self.X_lagr[:, 0]**2 + self.X_lagr[:, 1]**2     # θ² + γ²
        seed_3 = self.X_lagr[:, 2]*self.X_lagr[:, 0] + self.X_lagr[:, 3]*self.X_lagr[:, 1]  # θ·dθ + γ·dγ

        # Replace dummy target with something dynamic for boosting expression search
        seeds = np.stack([seed_1, seed_2, seed_3], axis=1)
        seed_target = np.mean(seeds, axis=1)

        self.model.fit(self.X_lagr, seed_target)
        self.best_eq = self.model.get_best()
        print(f"Best Lagrangian found: {self.best_eq['equation']}")



    def compute_EL_residuals(self):
        print("Computing Euler–Lagrange residuals...")
        θ, γ, dθ, dγ, ddθ, ddγ = sp.symbols('θ γ dθ dγ ddθ ddγ')
        replacements = {"x0": "θ", "x1": "γ", "x2": "dθ", "x3": "dγ"}
        expr_str = self.best_eq["equation"]
        for key, val in replacements.items():
            expr_str = expr_str.replace(key, val)
        L_expr = sp.sympify(expr_str)

        # Derive EOM for θ
        dL_ddθ = sp.diff(L_expr, dθ)
        dL_dθ = sp.diff(L_expr, θ)
        d_dL_ddθ = (
            sp.diff(dL_ddθ, θ)*dθ + sp.diff(dL_ddθ, γ)*dγ +
            sp.diff(dL_ddθ, dθ)*ddθ + sp.diff(dL_ddθ, dγ)*ddγ
        )
        EOM_θ = d_dL_ddθ - dL_dθ
        self.EOM_θ_func = sp.lambdify([θ, γ, dθ, dγ, ddθ, ddγ], EOM_θ, modules="numpy")

        # Derive EOM for γ
        dL_ddγ = sp.diff(L_expr, dγ)
        dL_dγ = sp.diff(L_expr, γ)
        d_dL_ddγ = (
            sp.diff(dL_ddγ, θ)*dθ + sp.diff(dL_ddγ, γ)*dγ +
            sp.diff(dL_ddγ, dθ)*ddθ + sp.diff(dL_ddγ, dγ)*ddγ
        )
        EOM_γ = d_dL_ddγ - dL_dγ
        self.EOM_γ_func = sp.lambdify([θ, γ, dθ, dγ, ddθ, ddγ], EOM_γ, modules="numpy")

    def evaluate(self):
        res_θ = self.EOM_θ_func(self.theta, self.gamma, self.dtheta, self.dgamma, self.ddtheta, self.ddgamma)
        res_γ = self.EOM_γ_func(self.theta, self.gamma, self.dtheta, self.dgamma, self.ddtheta, self.ddgamma)
        mse_θ = np.mean(res_θ**2)
        mse_γ = np.mean(res_γ**2)
        print(f"Lagrangian E-L residuals MSE: θ={mse_θ:.6e}, γ={mse_γ:.6e}")
        return mse_θ, mse_γ

    def run(self, df):
        self.prepare_data(df)
        self.train_lagrangian()
        self.compute_EL_residuals()
        return self.evaluate()

import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from pysr import PySRRegressor
from main_fun import *

class LagrangianPipeline:
    def __init__(self, model_params=None, mode="full"):
        self.mode = mode
        default_params = dict(
            niterations=1000,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "tanh", "exp", "log"],
            complexity_of_operators={"+": 1, "-": 1, "*": 2, "/": 5},
            loss="loss(x, y) = (x - y)^2",
            model_selection="best",
            verbosity=1,
            procs=0,
            maxsize=30,
            extra_sympy_mappings={"log": "log"},
        )
        self.model_params = default_params if model_params is None else {**default_params, **model_params}

    def prepare_data(self, df):
        features = extract_features(df)
        self.X = features[:, :-2]
        self.theta = df["Theta"].values
        self.gamma = df["Gamma"].values
        self.time = df["Time"].values

        self.dtheta = np.gradient(self.theta, self.time)
        self.dgamma = np.gradient(self.gamma, self.time)
        self.ddtheta = np.gradient(self.dtheta, self.time)
        self.ddgamma = np.gradient(self.dgamma, self.time)

        self.X_lagr = np.column_stack([self.theta, self.gamma, self.dtheta, self.dgamma])

    def train_lagrangian(self):
        print(f"Training symbolic Lagrangian in '{self.mode}' mode...")

        if self.mode == "full":
            self.model = PySRRegressor(**self.model_params)
            if hasattr(self.model, "equation_search_options"):
                self.model.equation_search_options["population"] = [
                    "x2**2 + x3**2",  # dθ² + dγ²
                    "x0**2 + x1**2",  # θ² + γ²
                    "x0*x2 + x1*x3"   # θ·dθ + γ·dγ
                ]

            seed_target = self.X_lagr[:, 2]**2 + self.X_lagr[:, 3]**2
            self.model.fit(self.X_lagr, seed_target)

            if hasattr(self.model, "run_history_"):
                self.run_history = self.model.run_history_

            self.best_eq = self.model.get_best()
            self.L_expr_str = self.best_eq["equation"]
            print(f"Best Lagrangian found: {self.L_expr_str}")

        elif self.mode == "split":
            # === Train Kinetic Energy T(dθ, dγ)
            X_T = self.X_lagr[:, [2, 3]]
            T_target = X_T[:, 0]**2 + X_T[:, 1]**2
            self.model_T = PySRRegressor(**self.model_params)
            if hasattr(self.model_T, "equation_search_options"):
                self.model_T.equation_search_options["population"] = [
                    "x0**2 + x1**2",  # dθ² + dγ²
                    "x0*x1",          # dθ·dγ
                    "x0 + x1**2"
                ]
            self.model_T.fit(X_T, T_target)
            self.T_expr_str = self.model_T.get_best()["equation"]
            if hasattr(self.model_T, "run_history_"):
                self.run_history_T = self.model_T.run_history_

            # === Train Potential Energy V(θ, γ)
            X_V = self.X_lagr[:, [0, 1]]
            V_target = X_V[:, 0]**2 + X_V[:, 1]**2
            self.model_V = PySRRegressor(**self.model_params)
            if hasattr(self.model_V, "equation_search_options"):
                self.model_V.equation_search_options["population"] = [
                    "x0**2 + x1**2",  # θ² + γ²
                    "x0*x1",          # θ·γ
                    "x0 + x1**2"
                ]
            self.model_V.fit(X_V, V_target)
            self.V_expr_str = self.model_V.get_best()["equation"]
            if hasattr(self.model_V, "run_history_"):
                self.run_history_V = self.model_V.run_history_

            # Final Lagrangian
            self.L_expr_str = f"({self.T_expr_str}) - ({self.V_expr_str})"
            self.run_history = getattr(self, "run_history_T", None)

            print(f"Best Kinetic Energy T: {self.T_expr_str}")
            print(f"Best Potential Energy V: {self.V_expr_str}")


    def compute_EL_residuals(self):
        print("Computing Euler–Lagrange residuals...")
        θ, γ, dθ, dγ, ddθ, ddγ = sp.symbols('θ γ dθ dγ ddθ ddγ')
        replacements = {"x0": "θ", "x1": "γ", "x2": "dθ", "x3": "dγ"}
        expr_str = self.L_expr_str
        for key, val in replacements.items():
            expr_str = expr_str.replace(key, val)
        L_expr = sp.sympify(expr_str)

        dL_ddθ = sp.diff(L_expr, dθ)
        dL_dθ = sp.diff(L_expr, θ)
        d_dL_ddθ = (
            sp.diff(dL_ddθ, θ)*dθ + sp.diff(dL_ddθ, γ)*dγ +
            sp.diff(dL_ddθ, dθ)*ddθ + sp.diff(dL_ddθ, dγ)*ddγ
        )
        EOM_θ = d_dL_ddθ - dL_dθ
        self.EOM_θ_func = sp.lambdify([θ, γ, dθ, dγ, ddθ, ddγ], EOM_θ, modules="numpy")

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

    def save_training_metrics(self, output_dir):
        def save_history(df, name):
            if df is None:
                print(f"[WARN] No history for {name}")
                return
            df.to_csv(os.path.join(output_dir, f"{name}_training_history.csv"), index=False)
            wandb.save(os.path.join(output_dir, f"{name}_training_history.csv"))
            plt.figure(figsize=(10, 4))
            plt.plot(df["iteration"], df["loss"], label="Loss")
            if "score" in df.columns:
                plt.plot(df["iteration"], df["score"], label="Score")
            plt.title(f"{name} PySR Metrics")
            plt.legend()
            plt.tight_layout()
            path = os.path.join(output_dir, f"{name}_metrics_plot.png")
            plt.savefig(path)
            plt.close()
            wandb.save(path)

        if self.mode == "full" and hasattr(self, "run_history"):
            save_history(self.run_history, "lagrangian")
        elif self.mode == "split":
            if hasattr(self, "run_history_T"):
                save_history(self.run_history_T, "kinetic_energy_T")
            if hasattr(self, "run_history_V"):
                save_history(self.run_history_V, "potential_energy_V")

    def run(self, df, output_dir=None):
        self.prepare_data(df)
        self.train_lagrangian()
        self.compute_EL_residuals()
        if output_dir:
            self.save_training_metrics(output_dir)
        return self.evaluate()

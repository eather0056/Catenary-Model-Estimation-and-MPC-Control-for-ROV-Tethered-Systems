import os
os.environ["JULIA_PYTHONCALL_EXE"] = "/home/mundus/mdeowan698/bin/julia-1.9.4/bin/julia"

from pysr import PySRRegressor
import numpy as np

X = np.random.randn(100, 3)
y = X[:, 0] + X[:, 1]**2 - np.sin(X[:, 2])

model = PySRRegressor(
    niterations=10,
    binary_operators=["+", "-", "*"],
    unary_operators=["cos", "sin"]
)

model.fit(X, y)

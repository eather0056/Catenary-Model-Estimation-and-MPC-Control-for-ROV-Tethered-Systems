from pysr import PySRRegressor
import numpy as np

X = np.random.rand(100, 2)
y = X[:, 0]**2 + X[:, 1]

model = PySRRegressor(niterations=5)
model.fit(X, y)

print(model.get_best())

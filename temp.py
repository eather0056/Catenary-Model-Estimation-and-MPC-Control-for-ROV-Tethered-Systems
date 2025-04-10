from pysr import PySRRegressor

X = [[1.0], [2.0], [3.0]]
y = [1.0, 4.0, 9.0]

model = PySRRegressor(
    niterations=5,
    binary_operators=["+", "-", "*", "/"],
)

model.fit(X, y)
print(model)

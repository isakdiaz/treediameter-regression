import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd

np.set_printoptions(suppress=True)
df = pd.read_csv("values.csv")

print(df)
x = df[['distance', 'pixels']]
y = df[['diameter']]


linear_regression = LinearRegression(normalize=True)
# linear_regression = Ridge(normalize=True)



poly = PolynomialFeatures(2, interaction_only = False)
xPoly = poly.fit_transform(x)


# linear_regression.fit(x,y)

linear_regression.fit(xPoly,y)


y_pred = linear_regression.predict(xPoly)

print(type(y_pred))
print(type(y.to_numpy()))

results = np.concatenate((y_pred, y.to_numpy(), y_pred - y.to_numpy()), axis = 1)
print(results)
print(linear_regression.coef_)
print(linear_regression.intercept_)
print(poly.get_feature_names())
print("MSE is: ", mean_squared_error(y_pred, y.to_numpy()))


distance = 170
pixelWidth = 146
x0 = -0.179362
x1 = -0.2876224
x0sq = 0.000187
x0x1 = 0.0026005
x1sq = 0.00046542
b = 36.186

diameter = distance * x0 + pixelWidth * x1 + distance**2 * x0sq + distance * pixelWidth * x0x1 + pixelWidth**2 * x1sq + b

print(diameter)


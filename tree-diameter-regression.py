import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd

MEASUREMENTS_CSV = "firebase-measurements.csv"


np.set_printoptions(suppress=True)
df = pd.read_csv(MEASUREMENTS_CSV)

print("Original Dataframe")
print(df.head())
x = df[['distance', 'pixel_width']]
y = df[['diameter']]


linear_regression = LinearRegression(normalize=True)
# linear_regression = Ridge(normalize=True)

poly = PolynomialFeatures(2, interaction_only = False)
xPoly = poly.fit_transform(x)


# linear_regression.fit(x,y)

linear_regression.fit(xPoly,y)
y_pred = linear_regression.predict(xPoly)


results = np.concatenate((y_pred, y.to_numpy(), y_pred - y.to_numpy()), axis = 1)
print("Results Dataframe")
print(results[:5])
print("Coefficients = ", linear_regression.coef_[0])
print("Intercept = ", linear_regression.intercept_[0])
print("Feature Names: ", poly.get_feature_names())
print("MSE for model is: ", mean_squared_error(y_pred, y.to_numpy()))


distance = 1.4443359375
pixel_width = 72.0
ground_truth = 22.28169059753418
# x0 = -0.179362
# x1 = -0.2876224
# x0sq = 0.000187
# x0x1 = 0.0026005
# x1sq = 0.00046542
# b = 36.186

x0 = linear_regression.coef_[0][1]
x1 = linear_regression.coef_[0][2]
x0sq = linear_regression.coef_[0][3]
x0x1 = linear_regression.coef_[0][4]
x1sq = linear_regression.coef_[0][5]
b = linear_regression.intercept_[0]

print("C# Variables")
print("var x0 = %.6f\nvar x1 = %.6f,\nvar x0sq = %.6f,\nvar x0x1 = %.6f,\nvar x1sq = %.6f,\nvar b = %.6f" % (x0, x1, x0sq, x0x1, x1sq, b))
diameter = distance * x0 + pixel_width * x1 + distance**2 * x0sq + distance * pixel_width * x0x1 + pixel_width**2 * x1sq + b

print(f"With a distance of {distance}cm and pixel width of {pixel_width}px, the diameter is calcualted to be {diameter}cm")
print(f"Ground truth was {ground_truth}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('./data.csv')

x = df[['TV','Radio','Newspaper']]
y = df['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
mlr = LinearRegression()
mlr.fit(x_train, y_train)

intercept = mlr.intercept_
coef = list(zip(mlr.coef_))
print("Intercept: ", mlr.intercept_)
print("Coefficients:", list(zip(x, mlr.coef_)))

# Predicition of test set
y_pred_mlr = mlr.predict(x_test)
print("Prediction for test set: {}".format(y_pred_mlr))

# Predicted values
mlr_diff = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_mlr})
print(mlr_diff.head())

# Model Evaluation
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("C:\\diabetes_012_health_indicators_BRFSS2015.csv")
df
df.shape
df.Diabetes_012.value_counts(normalize=1)
df.columns

X = df.iloc[:, 1:]
X.columns
y = df.iloc[:, 0]
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
X_test.shape
df.Diabetes_012.value_counts(normalize=1).plot(kind="bar")
pd.Series(y_pred).value_counts(normalize=1)
accuracy_score(y_test, y_pred)

df.corr("pearson")["Diabetes_012"]
plt.bar(df.corr("pearson")["Diabetes_012"].index, df.corr("pearson")["Diabetes_012"])
plt.xticks(rotation=90)

plt.show()
df.columns

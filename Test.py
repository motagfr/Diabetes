import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

df = pd.read_csv("C:\\diabetes_012_health_indicators_BRFSS2015.csv")
df
df.shape
df.Diabetes_012.value_counts(normalize=1)
df.columns


X = df.iloc[:, 1:]
X.columns
X.shape
y = df.iloc[:, 0]
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
pd.Series(y).value_counts().plot(kind="bar")

sm = SMOTE(random_state=1)
X_res, y_res = sm.fit_resample(X_train, y_train)
pd.Series(y_res).value_counts().plot(kind="bar")

X_train, X_valid, y_train, y_valid = train_test_split(
    X_res, y_res, test_size=0.2, random_state=1
)


knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
X_test.shape
df.Diabetes_012.value_counts(normalize=1).plot(kind="bar")
pd.Series(y_pred).value_counts(normalize=1)
accuracy_score(y_test, y_pred)

df.corr("pearson")["Diabetes_012"]
plt.bar(df.corr("pearson")["Diabetes_012"].index, df.corr("pearson")["Diabetes_012"])
plt.xticks(rotation=90)

plt.figure(figsize=(10, 8))
df.columns
sns.heatmap(
    df[["Diabetes_012", "HighBP", "HighChol", "CholCheck", "BMI"]].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation Heatmap")
plt.show()

plt.show()
df.columns

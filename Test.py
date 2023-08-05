from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

df = pd.read_csv("C:\\diabetes_012_health_indicators_BRFSS2015.csv")
df
df.shape
df.Diabetes_012.value_counts(normalize=1)
df.columns

# Preprocessing

X = df.iloc[:, 1:]
X.columns
X.shape
y = df.iloc[:, 0]
y.value_counts()
pd.Series(y.value_counts()).plot(kind="bar")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
fig1 = pd.Series(y_train).value_counts().plot(kind="bar")
# pd.Series(y_test).value_counts().plot(kind="bar")


sm = SMOTE(random_state=1)
X_res, y_res = sm.fit_resample(X_train, y_train)
fig2 = pd.Series(y_res).value_counts().plot(kind="bar")

X_train, X_val, y_train, y_val = train_test_split(
    X_res, y_res, test_size=0.2, random_state=1
)

# Correlations

df.corr("pearson")["Diabetes_012"]
plt.figure(figsize=(10, 8))
plt.bar(df.corr("pearson")["Diabetes_012"].index, df.corr("pearson")["Diabetes_012"])
plt.xticks(rotation=90)

plt.figure(figsize=(20, 20))
df.columns
sns.heatmap(
    df[df.columns].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation Heatmap")
plt.show()


# Modeling


# KNN

acc = []
for i in range(2, 31, 2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_val = knn.predict(X_val)
    y_pred_val
    y_pred_test = knn.predict(X_test)
    y_pred_test
    print(
        f"Validation prediction accuracy score for K={i}:",
        accuracy_score(y_val, y_pred_val),
    )
    print(
        f"Test prediction accuracy score for K={i}:",
        accuracy_score(y_test, y_pred_test),
    )
    acc.append(accuracy_score(y_val, y_pred_val))
acc

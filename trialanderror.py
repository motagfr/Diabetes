import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

relative_path = "src\diabetes.csv"
absolute_path = os.path.join(os.getcwd(), relative_path)
with open(absolute_path, "r") as file:
    df = pd.read_csv(absolute_path)
# print(absolute_path)
print("Dataset shape:")
print(df.shape)
df.isnull().sum()
print("\nFrequency of each label value:")
print(df.Diabetes_012.value_counts())
print("\nDistribution of the target:")
print(df.Diabetes_012.value_counts(normalize=1))
print("\nFeatures:")
print(df.columns)
df.Diabetes_012.replace(2.0, 1.0, inplace=True)

print(df.columns)
df.columns = df.columns.str.replace(" ", "")
df.drop(
    ["Fruits", "Veggies", "NoDocbcCost", "Education", "Income"], axis=1, inplace=True
)
print(df.shape)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

print(y.value_counts())

bmi_scaler = StandardScaler()
cols_to_scale = ["BMI"]
bmi_scaler.fit(X[cols_to_scale])
X.loc[:, cols_to_scale] = bmi_scaler.transform(X[cols_to_scale])
fig = X.hist(column="BMI", figsize=(5, 5))


X.hist(column="GenHlth", figsize=(5, 5))
gh_scaler = StandardScaler()
cols_to_scale = ["GenHlth"]
gh_scaler.fit(X[cols_to_scale])
X[cols_to_scale] = gh_scaler.transform(X[cols_to_scale])
X.hist(column="GenHlth", figsize=(5, 5))

import warnings

warnings.filterwarnings("ignore")

X.hist(column="MentHlth", figsize=(5, 5))
ment_scaler = MinMaxScaler()
cols_to_scale = ["MentHlth"]
ment_scaler.fit(X[cols_to_scale])
X[cols_to_scale] = ment_scaler.transform(X[cols_to_scale])
X.hist(column="MentHlth", figsize=(5, 5))

X.hist(column="PhysHlth", figsize=(5, 5))
phys_scaler = MinMaxScaler()
cols_to_scale = ["PhysHlth"]
phys_scaler.fit(X[cols_to_scale])
X[cols_to_scale] = phys_scaler.transform(X[cols_to_scale])
X.hist(column="PhysHlth", figsize=(5, 5))

X.hist(column="Age", figsize=(5, 5))
age_scaler = MinMaxScaler()
cols_to_scale = ["Age"]
age_scaler.fit(X[cols_to_scale])
X[cols_to_scale] = age_scaler.transform(X[cols_to_scale])
X.hist(column="Age", figsize=(5, 5))

# X.hist(column="Income", figsize=(5, 5))
# income_cat_to_avg_map = {
#     1: 5,
#     2: 12.5,
#     3: 17.5,
#     4: 22.5,
#     5: 30.0,
#     6: 42.5,
#     7: 62.5,
#     8: 75,
# }
# X = X.assign(Income=X.Income.map(income_cat_to_avg_map))
# X.hist(column="Income", figsize=(5, 5))
# inc_scaler = MinMaxScaler()
# cols_to_scale = ["Income"]
# inc_scaler.fit(X[cols_to_scale])
# X[cols_to_scale] = inc_scaler.transform(X[cols_to_scale])
# X.hist(column="Income", figsize=(5, 5))

# X.hist(column="Education", figsize=(5, 5))
# edu_scaler = MinMaxScaler()
# cols_to_scale = ["Education"]
# edu_scaler.fit(X[cols_to_scale])
# X[cols_to_scale] = edu_scaler.transform(X[cols_to_scale])
# X.hist(column="Education", figsize=(5, 5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1
)
# pd.Series(y_test).value_counts().plot(kind="bar")
smt = SMOTE(random_state=1)
X_train, y_train = smt.fit_resample(X_train, y_train)

rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred_val = rf.predict(X_val)
print(classification_report(y_val, y_pred_val))
y_pred_test = rf.predict(X_test)
print(classification_report(y_test, y_pred_test))


lr = LogisticRegression(solver="newton-cholesky", random_state=0, penalty="l2")
lr.fit(X_train, y_train)
y_pred_val = lr.predict(X_val)
print(classification_report(y_val, y_pred_val))
y_pred_test = lr.predict(X_test)
print(classification_report(y_test, y_pred_test))

sv = SGDClassifier(random_state=0)
sv.fit(X_train, y_train)
y_pred_val = sv.predict(X_val)
print(classification_report(y_val, y_pred_val))
y_pred_test = sv.predict(X_test)
print(classification_report(y_test, y_pred_test))

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
y_pred_val = dt.predict(X_val)
print(classification_report(y_val, y_pred_val))
y_pred_test = dt.predict(X_test)
print(classification_report(y_test, y_pred_test))


xg = XGBClassifier(random_state=0)
xg.fit(X_train, y_train)
y_pred_val = xg.predict(X_val)
print(classification_report(y_val, y_pred_val))
y_pred_test = xg.predict(X_test)
print(classification_report(y_test, y_pred_test))

ml = MLPClassifier(random_state=0)
ml.fit(X_train, y_train)
y_pred_val = ml.predict(X_val)
print(classification_report(y_val, y_pred_val))
y_pred_test = ml.predict(X_test)
print(classification_report(y_test, y_pred_test))

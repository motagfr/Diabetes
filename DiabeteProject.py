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
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
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
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

df = pd.read_csv("https://raw.githubusercontent.com/imsalione/Diabete/main/src/diabetes.csv")
df
df.shape
df.isnull().sum()
df.Diabetes_012.value_counts()
df.columns
df.Diabetes_012.value_counts(normalize=1)
df.Diabetes_012.replace(2.0, 1.0, inplace=True)


# Correlation

plt.figure(figsize=(10, 8))
df.corr("pearson")["Diabetes_012"]
fig2 = plt.bar(
    df.corr("pearson")["Diabetes_012"].index, df.corr("pearson")["Diabetes_012"]
)
plt.xticks(rotation=90)

plt.figure(figsize=(20, 20))
sns.heatmap(
    df[df.columns].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation Heatmap")
plt.show()

# Drop features with weak correlation

df.drop(["Fruits", "Veggies", "NoDocbcCost"], axis=1, inplace=True)
df.shape
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
df.columns = df.columns.str.replace(" ", "")
print(y.value_counts())
y.value_counts()

for c in X.columns:
    hist = X.hist(column=c, figsize=(4, 4))


bmi_scaler = StandardScaler()
cols_to_scale = ["BMI"]
bmi_scaler.fit(X[cols_to_scale])
X.loc[:, cols_to_scale] = bmi_scaler.transform(X[cols_to_scale])
fig = X.hist(column="BMI", figsize=(5, 5))
import warnings

warnings.filterwarnings("ignore")


X.hist(column="GenHlth", figsize=(5, 5))
gh_scaler = StandardScaler()
cols_to_scale = ["GenHlth"]
gh_scaler.fit(X[cols_to_scale])
X[cols_to_scale] = gh_scaler.transform(X[cols_to_scale])
X.hist(column="GenHlth", figsize=(5, 5))


from sklearn.preprocessing import MinMaxScaler

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


X.hist(column="Education", figsize=(5, 5))
edu_scaler = MinMaxScaler()
cols_to_scale = ["Education"]
edu_scaler.fit(X[cols_to_scale])
X[cols_to_scale] = edu_scaler.transform(X[cols_to_scale])
X.hist(column="Education", figsize=(5, 5))


X.hist(column="Income", figsize=(5, 5))
income_cat_to_avg_map = {
    1: 5,
    2: 12.5,
    3: 17.5,
    4: 22.5,
    5: 30.0,
    6: 42.5,
    7: 62.5,
    8: 75,
}
X = X.assign(Income=X.Income.map(income_cat_to_avg_map))
X.hist(column="Income", figsize=(5, 5))
inc_scaler = MinMaxScaler()
cols_to_scale = ["Income"]
inc_scaler.fit(X[cols_to_scale])
X[cols_to_scale] = inc_scaler.transform(X[cols_to_scale])
X.hist(column="Income", figsize=(5, 5))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1
)
# pd.Series(y_test).value_counts().plot(kind="bar")
smt = SMOTETomek(random_state=1)
X_train, y_train = smt.fit_resample(X_train, y_train)

import warnings

warnings.filterwarnings("ignore")

pd.Series(y_train).value_counts()  # tomlink=134888 Smote=136796
X_train.shape[0] + X_test.shape[0] + X_val.shape[0]  # tomlink=361101 smote=364917

#####################################################
# Random Forest

ne = [100]
mf = range(2, 15)
md = range(6, 15)
best = (0, None, (0, 0, 0))

for a in ne:
    for b in mf:
        for c in md:
            print("a={},b={},c={}".format(a, b, c))
            rf_orig = RandomForestClassifier(
                n_estimators=a, max_features=b, max_depth=c
            )
            rf_orig.fit(X_train, y_train)
            train_score = rf_orig.score(X_train, y_train)
            val_score = rf_orig.score(X_val, y_val)
            if best[0] < val_score:
                best = (val_score, rf_orig, (a, b, c))
            print(train_score)
            print(val_score)

print(best)

# Test Data Prediction

rf = RandomForestClassifier(n_estimators=120, max_features=15, max_depth=15)
rf.fit(X_train, y_train)
train_score = rf.score(X_train, y_train)
test_score = rf.score(X_test, y_test)
print(train_score)
print(test_score)

#############
#How to determine n_estimators

from sklearn.model_selection import cross_val_score

n_estimators_range = np.arange(1, 111, 10)

train_scores = []

for n_estimators in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    train_score = np.mean(cross_val_score(rf, X, y, cv=5, scoring='accuracy'))
    train_scores.append(train_score)


plt.plot(n_estimators_range, train_scores, label='Training score')
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
plt.title('Training performance as a function of n_estimators')
plt.legend()
plt.show()
######################################
# Assuming you have trained a tree-based model called 'model'

# Get feature importances
importances = model.feature_importances_

# Get the feature names
feature_names = ["feature1", "feature2", ...]  # Replace with your actual feature names

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names accordingly
sorted_feature_names = [feature_names[i] for i in indices]

# Plot the feature importances
plt.barh(sorted_feature_names, importances[indices])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Tree-based Model)")
plt.show()


##############################################
# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define the logistic regression model
model = LogisticRegression()

# Define the hyperparameters to tune
parameters = {"penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100]}

# Perform grid search using cross-validation
grid_search = GridSearchCV(model, parameters, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding mean cross-validated score
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# Assuming you have trained a logistic regression model called 'model'

# Get the absolute coefficients
coefs = np.abs(model.coef_[0])

# Get the feature names
feature_names = ["feature1", "feature2", ...]  # Replace with your actual feature names

# Plot the feature importances
plt.barh(feature_names, coefs)
plt.xlabel("Coefficient Magnitude")
plt.ylabel("Feature")
plt.title("Feature Importance (Logistic Regression)")
plt.show()

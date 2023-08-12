import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
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
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
    minmax_scale,
)
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
print(df.Diabetes_012.value_counts(normalize=1))
df.columns = df.columns.str.replace(" ", "")


df.drop(
    ["Fruits", "Veggies", "NoDocbcCost", "AnyHealthcare", "Smoker", "Stroke"],
    axis=1,
    inplace=True,
)
print(df.shape)

# df.hist(column="BMI", figsize=(5, 5))
bmi_scaler = StandardScaler()
cols_to_scale = ["BMI"]
bmi_scaler.fit(df[cols_to_scale])
df.loc[:, cols_to_scale] = bmi_scaler.transform(df[cols_to_scale])
# fig = df.hist(column="BMI", figsize=(5, 5))

# df.hist(column="GenHlth", figsize=(5, 5))
gh_scaler = StandardScaler()
cols_to_scale = ["GenHlth"]
gh_scaler.fit(df[cols_to_scale])
df.loc[:, cols_to_scale] = gh_scaler.transform(df[cols_to_scale])
# df.hist(column="GenHlth", figsize=(5, 5))

# df.hist(column="MentHlth", figsize=(5, 5))
ment_scaler = MinMaxScaler()
cols_to_scale = ["MentHlth"]
ment_scaler.fit(df[cols_to_scale])
df.loc[:, cols_to_scale] = ment_scaler.transform(df[cols_to_scale])
# df.hist(column="MentHlth", figsize=(5, 5))

# df.hist(column="PhysHlth", figsize=(5, 5))
phys_scaler = MinMaxScaler()
cols_to_scale = ["PhysHlth"]
phys_scaler.fit(df[cols_to_scale])
df.loc[:, cols_to_scale] = phys_scaler.transform(df[cols_to_scale])
# df.hist(column="PhysHlth", figsize=(5, 5))

# df.hist(column="Age", figsize=(5, 5))
age_scaler = MinMaxScaler()
cols_to_scale = ["Age"]
age_scaler.fit(df[cols_to_scale])
df.loc[:, cols_to_scale] = age_scaler.transform(df[cols_to_scale])
# df.hist(column="Age", figsize=(5, 5))
import warnings

warnings.filterwarnings("ignore")

# df.hist(column="Income", figsize=(5, 5))
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
# df = df.assign(Income=X.Income.map(income_cat_to_avg_map))
df.hist(column="Income", figsize=(5, 5))
inc_scaler = MinMaxScaler()
cols_to_scale = ["Income"]
inc_scaler.fit(df[cols_to_scale])
df[cols_to_scale] = inc_scaler.transform(df[cols_to_scale])
# df.hist(column="Income", figsize=(5, 5))

# df.hist(column="Education", figsize=(5, 5))
edu_scaler = MinMaxScaler()
cols_to_scale = ["Education"]
edu_scaler.fit(df[cols_to_scale])
df[cols_to_scale] = edu_scaler.transform(df[cols_to_scale])
# df.hist(column="Education", figsize=(5, 5))

# Sex!

df["Sex"].value_counts(normalize=True)

gender = df["Sex"]
gender = gender.values.reshape(-1, 1)
encoder = OneHotEncoder(categories="auto", sparse_output=False)
gender_encoded = encoder.fit_transform(gender)
print(gender_encoded)
gender_encoded_df = pd.DataFrame(
    gender_encoded, columns=encoder.get_feature_names_out(["gender"])
)
df.reset_index(drop=True, inplace=True)
gender_encoded_df.reset_index(drop=True, inplace=True)
df_encoded = pd.concat([df.drop("Sex", axis=1), gender_encoded_df], axis=1)
df_encoded

df_encoded.iloc[:, 0].value_counts(normalize=1)

X = df_encoded.iloc[:, 1:]
y = df_encoded.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=df_encoded[["gender_0.0", "gender_1.0"]],
    random_state=1,
)
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.2, random_state=1
# )
# pd.Series(y_test).value_counts().plot(kind="bar")
smt = SMOTE(random_state=1)
X_train, y_train = smt.fit_resample(X_train, y_train)

# Optimal LogisticRegression model

logistic_reg = LogisticRegression(random_state=0)

param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.1, 1.0, 3.0, 5.0, 10.0],
    "solver": ["saga"],
    "max_iter": [100, 200, 300],
    "fit_intercept": [True, False],
}

grid_search = GridSearchCV(
    estimator=logistic_reg,
    param_grid=param_grid,
    scoring="f1_weighted",  # Use F1-score as the scoring metric
    cv=5,
    refit=True,
)

grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average="weighted")
print("F1-score on test set: ", f1)

# The most significant features

coefficients = best_model.coef_[0]

feature_names = list(X.columns)
feature_coefficients = list(zip(feature_names, coefficients))
sorted_feature_coefficients = sorted(
    feature_coefficients, key=lambda x: abs(x[1]), reverse=True
)

for feature, coefficient in sorted_feature_coefficients:
    print(f"Feature: {feature}, Coefficient: {coefficient}")


######################################
# Randomized search logistic

logistic_reg = LogisticRegression()

param_dist = {
    "penalty": ["l1", "l2"],
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "solver": ["liblinear", "saga"],
    "max_iter": [100, 200, 300, 500],
    "class_weight": [None, "balanced"],
    "fit_intercept": [True, False],
}

random_search = RandomizedSearchCV(
    logistic_reg, param_distributions=param_dist, n_iter=10, cv=5
)

random_search.fit(X_train, y_train)
print("Best parameters: ", random_search.best_params_)

best_model2 = random_search.best_estimator_
y_pred = best_model2.predict(X_test)
f1 = f1_score(y_test, y_pred, average="weighted")
print("F1-score on test set: ", f1)

# The most significant features

coefficients = best_model2.coef_[0]

feature_names = list(X.columns)
feature_coefficients = list(zip(feature_names, coefficients))
sorted_feature_coefficients = sorted(
    feature_coefficients, key=lambda x: x[1], reverse=True
)

for feature, coefficient in sorted_feature_coefficients:
    print(f"Feature: {feature}, Coefficient: {coefficient}")

#############################
# Instantiate the SGDClassifier
sgd = SGDClassifier()

param_dist = {
    'loss':['log', 'modified_huber'],
    'penalty': ['l1', 'l2'],
    'alpha': np.random.uniform(0, 0.01, 5),
    'learning_rate': ['constant', 'optimal'],
    'eta0': np.random.uniform(0, 0.1, 5),
    'max_iter': [100,200,300]
}
random_search = RandomizedSearchCV(
    sgd,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    random_state=42
)

# Fit the random search to your data
random_search.fit(X_train, y_train)

# Retrieve the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

##################################
rf = RandomForestClassifier(random_state=0, class_weight="balanced")
rf.fit(X_train, y_train)
# y_pred_val = rf.predict(X_val)
# print(classification_report(y_val, y_pred_val))
y_pred_test = rf.predict(X_test)
print(
    classification_report(
        y_test,
        y_pred_test,
    )
)

# feature selection again

################################

lr = LogisticRegression(solver="liblinear", random_state=0, penalty="l2", C=5)
lr.fit(X_train, y_train)
# y_pred_val = lr.predict(X_val)
# print(classification_report(y_val, y_pred_val))
y_pred_test = lr.predict(X_test)
print(classification_report(y_test, y_pred_test))

sv = SGDClassifier(random_state=0)
sv.fit(X_train, y_train)
# y_pred_val = sv.predict(X_val)
# print(classification_report(y_val, y_pred_val))
y_pred_test = sv.predict(X_test)
print(classification_report(y_test, y_pred_test))

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
# y_pred_val = dt.predict(X_val)
# print(classification_report(y_val, y_pred_val))
y_pred_test = dt.predict(X_test)
print(classification_report(y_test, y_pred_test))


xg = XGBClassifier(random_state=0)
xg.fit(X_train, y_train)
# y_pred_val = xg.predict(X_val)
# print(classification_report(y_val, y_pred_val))
y_pred_test = xg.predict(X_test)
print(classification_report(y_test, y_pred_test))

ml = MLPClassifier(random_state=0)
ml.fit(X_train, y_train)
# y_pred_val = ml.predict(X_val)
# print(classification_report(y_val, y_pred_val))
y_pred_test = ml.predict(X_test)
print(classification_report(y_test, y_pred_test))


# Create an SGDClassifier object
sgd = SGDClassifier()

# Define the parameter grid to search over
param_grid = {
    "loss": "log",
    "penalty": ["l1", "l2"],
    "alpha": np.random.uniform(0, 0.1, size=3),
    "learning_rate": ["constant", "optimal"],
    "max_iter": [100, 200],
    "tol": [1e-3, 1e-4],
}

# Create a GridSearchCV object with nested cross-validation
inner_cv = 3  # Inner cross-validation folds
outer_cv = 2  # Outer cross-validation folds

grid_search = GridSearchCV(
    estimator=sgd, param_grid=param_grid, scoring="f1_weighted", cv=inner_cv, refit=True
)

# Perform nested cross-validation
nested_scores = cross_val_score(grid_search, X=X_train, y=y_train, cv=outer_cv)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters: ", grid_search.best_params_)

# Evaluate the best model on the testing data
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Accuracy on test set: ", accuracy)

# Print the nested cross-validation scores
print("Nested CV scores: ", nested_scores)
print("Average nested CV score: ", nested_scores.mean())

# Evaluate the best model on the testing data and calculate the F1-score
y_pred = best_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average="weighted")

print("F1-score on test set: ", f1)

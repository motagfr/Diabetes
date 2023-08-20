import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

relative_path = "src\diabetes.csv"
absolute_path = os.path.join(os.getcwd(), relative_path)
with open(absolute_path, "r") as file:
    df = pd.read_csv(absolute_path)
# print(absolute_path)


df.Diabetes_012.replace(2.0, 1.0, inplace=True)
# df.columns = df.columns.str.replace(" ", "")
df.drop(
    ["Fruits", "Veggies", "NoDocbcCost", "AnyHealthcare", "Smoker", "Stroke"],
    axis=1,
    inplace=True,
)

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
# # df.hist(column="Income", figsize=(5, 5))

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

gender = df["Sex"]
gender = gender.values.reshape(-1, 1)
encoder = OneHotEncoder(categories="auto", sparse_output=False)
gender_encoded = encoder.fit_transform(gender)
gender_encoded_df = pd.DataFrame(
    gender_encoded, columns=encoder.get_feature_names_out(["gender"])
)

df.reset_index(drop=True, inplace=True)
gender_encoded_df.reset_index(drop=True, inplace=True)
df_encoded = pd.concat([df.drop("Sex", axis=1), gender_encoded_df], axis=1)
X = df_encoded.iloc[:, 1:]
y = df_encoded.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=df_encoded[["gender_0.0", "gender_1.0"]],
    random_state=1,
)
smt = SMOTE(random_state=1)
X_train, y_train = smt.fit_resample(X_train, y_train)

sgd = SGDClassifier(random_state=0)

param_dist = {
    "loss": ["log", "modified_huber"],
    "penalty": ["l1", "l2"],
    "alpha": np.random.uniform(0, 0.01, 5),
    "learning_rate": ["constant", "optimal"],
    "eta0": np.random.uniform(0, 0.1, 5),
    "max_iter": [100, 200, 300],
}
random_search = RandomizedSearchCV(
    sgd, param_distributions=param_dist, n_iter=10, cv=5, random_state=42
)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_model = random_search.best_estimator_

print("\nBest parameters: ", random_search.best_params_)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average="weighted")
print("\nF1-score on test set: ", f1)

print(classification_report(y_test, y_pred))
print("\n\nThe confusion matrix:\n")
conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
print(conf_mat)
ax = sns.heatmap(conf_mat, annot=True)

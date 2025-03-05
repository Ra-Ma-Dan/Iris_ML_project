from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


iris_data = load_iris()
X_data = iris_data["data"]
y_data = iris_data["target"]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=42)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

k_n_neibours = KNeighborsClassifier(n_neighbors=1)
k_n_neibours.fit(X_train, y_train)

#path_for_model = "C:\\Users\\NCC\\Documents\\Ra Ma Dan\\ML Pro\\My_Models"
path_for_model = "C:\\Users\\NCC\\Documents\\Ra Ma Dan\\ML Pro\\My_Models\\iris_predic_model.pkl"
#joblib.dump("iris_predic_model.pkl", "..//..//My_Models")
#joblib.dump(k_n_neibours, path_for_model)

iris = pd.DataFrame(X_train, columns=iris_data.feature_names)
#grr = plt.scatter(iris, c=y_train, figsize=(15, 15), marker="o", hist_kwds={"bins": 20}, s=60, alph=0.8)
iris.head()
prediction = k_n_neibours.predict(X_test)
prediction
r2 = r2_score(prediction, y_test)
np_mean = np.mean(prediction == y_test)
knn_score = k_n_neibours.score(X_test, y_test)
print(f"r2_score: {r2 * 100:.2f}")
print(f"np_mean: {np_mean * 100:.2f}")
print(f"knn_score: {knn_score * 100:.2f}")
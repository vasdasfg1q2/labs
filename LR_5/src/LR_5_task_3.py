import io
import os
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

data = np.loadtxt("data_random_forests.txt", delimiter=",")
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

parameter_grid = [
    {"n_estimators": [100], "max_depth": [2, 4, 7, 12, 16]},
    {"max_depth": [4], "n_estimators": [25, 50, 100, 250]},
]

metrics = ["precision_weighted", "recall_weighted"]

for metric in metrics:
    print("\n##### Searching optimal parameters for", metric)

    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0),
        parameter_grid, cv=5, scoring=metric,
    )
    classifier.fit(X_train, y_train)

    print("\nGrid scores for the parameter grid:")
    # у новому sklearn — cv_results_, а не grid_scores_
    means = classifier.cv_results_["mean_test_score"]
    params_list = classifier.cv_results_["params"]
    for p, m in zip(params_list, means):
        print(f"  {p} --> {round(m, 3)}")

    print("\nBest parameters:", classifier.best_params_)

    y_pred = classifier.predict(X_test)
    print("\nPerformance report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

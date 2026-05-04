import io
import sys
import warnings

import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

input_file = "income_data.txt"

X = []
count_class1 = 0
count_class2 = 0
# той самий датасет, що й у task_1 (методичка с.5: «Використовуючи
# набір даних та код з попереднього завдання...»)
max_datapoints = 25000

with open(input_file, "r") as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if "?" in line:
            continue
        data = line[:-1].split(", ")
        if data[-1] == "<=50K" and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == ">50K" and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# SVM з поліноміальним ядром, degree=8 (методичка с.5)
# max_iter=20000 — щоб poly d=8 на ~50k точок гарантовано завершився
classifier = SVC(kernel="poly", degree=8, random_state=0, max_iter=20000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

print("SVM з поліноміальним ядром (degree=8)")
print("Accuracy:  " + str(round(100 * accuracy_score(y_test, y_test_pred), 2)) + "%")
print("Precision: " + str(round(100 * precision_score(y_test, y_test_pred, average="weighted"), 2)) + "%")
print("Recall:    " + str(round(100 * recall_score(y_test, y_test_pred, average="weighted"), 2)) + "%")
print("F1:        " + str(round(100 * f1_score(y_test, y_test_pred, average="weighted"), 2)) + "%")

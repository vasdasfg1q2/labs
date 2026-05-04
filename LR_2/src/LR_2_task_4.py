import io
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

input_file = "income_data.txt"

# той самий датасет, що й у task_1 (методичка: «по аналогії із task 2.3»
# для income_data — тобто повний набір з 25000 точок на клас)
X = []
count_class1 = 0
count_class2 = 0
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

models = []
# методичка с.15: LogisticRegression(solver='liblinear', multi_class='ovr').
# multi_class у sklearn ≥1.7 видалено — використовуємо OneVsRestClassifier
# як еквівалент OvR (тут income-датасет бінарний, тому ефективно це
# просто звичайна логістична регресія, але обгортка лишається задля
# дослівної відповідності методичці).
models.append(("LR", OneVsRestClassifier(LogisticRegression(solver="liblinear"))))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
# на ~50k точок без масштабування SVC дуже повільний — обмежуємо max_iter
models.append(("SVM", SVC(gamma="auto", max_iter=20000)))

results = []
names_list = []
print(f"Датасет: {X.shape[0]} точок, {X.shape[1]} ознак")
print("Порівняння алгоритмів (stratified 10-fold CV, accuracy):")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train,
                                 cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names_list.append(name)
    print(f"{name:>5}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

plt.boxplot(results, tick_labels=names_list)
plt.title("Algorithm Comparison (income_data)")
plt.tight_layout()
plt.savefig("outputs/task_4_algo_comparison.png", dpi=120)
plt.close()

# повна оцінка кращої моделі
# визначаємо кращу за середнім accuracy на CV
best_idx = int(np.argmax([r.mean() for r in results]))
best_name, best_model = models[best_idx]
print()
print(f"Найкраща модель за CV: {best_name}")
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)
print(f"Accuracy на test: {accuracy_score(y_test, preds):.4f}")
print()
print("Confusion matrix:")
print(confusion_matrix(y_test, preds))
print()
print("Classification report:")
print(classification_report(y_test, preds))

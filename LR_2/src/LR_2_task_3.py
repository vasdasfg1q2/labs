import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# --- крок 1. завантаження та вивчення даних ---
iris_dataset = load_iris()

print("Ключі iris_dataset:")
print(list(iris_dataset.keys()))
print()
print("Назви відповідей:", iris_dataset["target_names"])
print("Назви ознак:", iris_dataset["feature_names"])
print("Тип масиву data:", type(iris_dataset["data"]))
print("Форма масиву data:", iris_dataset["data"].shape)
print()
print("Перші 5 прикладів:")
print(iris_dataset["data"][:5])
print()
print("Тип target:", type(iris_dataset["target"]))
print("Відповіді:", iris_dataset["target"])
print()

# --- через pandas ---
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
try:
    dataset = read_csv(url, names=names)
except Exception:
    # fallback: у разі відсутності інтернету будуємо DataFrame з load_iris
    from pandas import DataFrame

    dataset = DataFrame(iris_dataset["data"], columns=names[:4])
    dataset["class"] = [iris_dataset["target_names"][t]
                        for t in iris_dataset["target"]]
    dataset["class"] = "Iris-" + dataset["class"]

print("Форма dataset:", dataset.shape)
print()
print("Перші 20 рядків:")
print(dataset.head(20))
print()
print("Статистика:")
print(dataset.describe())
print()
print("Розподіл за класами:")
print(dataset.groupby("class").size())

# --- крок 2. візуалізація ---
dataset.plot(kind="box", subplots=True, layout=(2, 2),
             sharex=False, sharey=False)
plt.tight_layout()
plt.savefig("outputs/task_3_boxplot.png", dpi=120)
plt.close()

dataset.hist()
plt.tight_layout()
plt.savefig("outputs/task_3_histogram.png", dpi=120)
plt.close()

scatter_matrix(dataset)
plt.tight_layout()
plt.savefig("outputs/task_3_scatter_matrix.png", dpi=120)
plt.close()

# --- крок 3. тренувальний/тестовий набір ---
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# --- крок 4. 6 моделей + stratified 10-fold CV ---
models = []
models.append(("LR", LogisticRegression(max_iter=200)))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))

results = []
names_list = []
print()
print("Порівняння алгоритмів (stratified 10-fold CV, accuracy):")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train,
                                 cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names_list.append(name)
    print(f"{name:>5}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# графік порівняння
plt.boxplot(results, tick_labels=names_list)
plt.title("Algorithm Comparison")
plt.tight_layout()
plt.savefig("outputs/task_3_algo_comparison.png", dpi=120)
plt.close()

# --- крок 6-7. прогноз на контрольній вибірці SVC ---
model = SVC(gamma="auto")
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print()
print("Оцінка SVC на контрольній вибірці:")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print()
print("Confusion matrix:")
print(confusion_matrix(Y_validation, predictions))
print()
print("Classification report:")
print(classification_report(Y_validation, predictions))

# --- крок 8. прогноз для нової точки ---
X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма масиву X_new:", X_new.shape)
prediction = model.predict(X_new)
print("Прогноз:", prediction)
print("Передбачена мітка:", prediction[0])

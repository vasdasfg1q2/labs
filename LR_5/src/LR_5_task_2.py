import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utilities import visualize_classifier

os.makedirs("outputs", exist_ok=True)


def run(balance):
    data = np.loadtxt("data_imbalance.txt", delimiter=",")
    X, y = data[:, :-1], data[:, -1]

    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])

    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75,
                facecolors="black", edgecolors="black",
                linewidth=1, marker="x")
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75,
                facecolors="white", edgecolors="black",
                linewidth=1, marker="o")
    plt.title(f"Input data (balance={balance})")
    plt.tight_layout()
    plt.savefig(f"outputs/task_2_{'balanced' if balance else 'imbalanced'}_input.png", dpi=120)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    params = {"n_estimators": 100, "max_depth": 4, "random_state": 0}
    if balance:
        params["class_weight"] = "balanced"

    classifier = ExtraTreesClassifier(**params)
    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train,
                          f"Training (balance={balance})",
                          f"outputs/task_2_{'balanced' if balance else 'imbalanced'}_train.png")

    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test,
                          f"Test (balance={balance})",
                          f"outputs/task_2_{'balanced' if balance else 'imbalanced'}_test.png")

    class_names = ["Class-0", "Class-1"]
    print(f"\nClassifier performance on training dataset (balance={balance})\n")
    print(classification_report(y_train, classifier.predict(X_train),
                                 target_names=class_names, zero_division=0))
    print(f"\nClassifier performance on test dataset (balance={balance})\n")
    print(classification_report(y_test, y_test_pred,
                                 target_names=class_names, zero_division=0))


if __name__ == "__main__":
    # спочатку без балансу — демонструємо проблему
    print("Без балансу:")
    run(balance=False)
    print("\nЗ балансом class_weight='balanced':")
    run(balance=True)

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")

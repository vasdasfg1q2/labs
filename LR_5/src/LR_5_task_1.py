import argparse
import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utilities import visualize_classifier

os.makedirs("outputs", exist_ok=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Classify data using Ensemble Learning techniques")
    parser.add_argument("--classifier-type", dest="classifier_type",
                        required=False, default="rf",
                        choices=["rf", "erf"],
                        help="Type of classifier: 'rf' або 'erf'")
    return parser


def run(classifier_type):
    data = np.loadtxt("data_random_forests.txt", delimiter=",")
    X, y = data[:, :-1], data[:, -1]

    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])

    # візуалізація вхідних даних
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75,
                facecolors="white", edgecolors="black", linewidth=1, marker="s")
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75,
                facecolors="white", edgecolors="black", linewidth=1, marker="o")
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75,
                facecolors="white", edgecolors="black", linewidth=1, marker="^")
    plt.title("Input data")
    plt.tight_layout()
    plt.savefig(f"outputs/task_1_{classifier_type}_input.png", dpi=120)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    params = {"n_estimators": 100, "max_depth": 4, "random_state": 0}
    if classifier_type == "rf":
        classifier = RandomForestClassifier(**params)
        label = "RandomForest"
    else:
        classifier = ExtraTreesClassifier(**params)
        label = "ExtraTrees"

    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train,
                          f"{label} — training dataset",
                          f"outputs/task_1_{classifier_type}_train.png")

    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test,
                          f"{label} — test dataset",
                          f"outputs/task_1_{classifier_type}_test.png")

    class_names = ["Class-0", "Class-1", "Class-2"]
    print("#" * 40)
    print(f"\n{label} performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train),
                                 target_names=class_names, zero_division=0))
    print("#" * 40 + "\n")

    print("#" * 40)
    print(f"\n{label} performance on test dataset\n")
    print(classification_report(y_test, y_test_pred,
                                 target_names=class_names, zero_division=0))
    print("#" * 40 + "\n")

    # оцінка довірливості
    test_datapoints = np.array([[5, 5], [3, 6], [6, 4],
                                 [7, 2], [4, 4], [5, 2]])
    print("\nConfidence measure:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = "Class-" + str(np.argmax(probabilities))
        print(f"Datapoint: {datapoint}  probabilities: {np.round(probabilities, 3)} "
              f"-> {predicted_class}")

    visualize_classifier(classifier, test_datapoints,
                          [0] * len(test_datapoints),
                          f"{label} — test points",
                          f"outputs/task_1_{classifier_type}_points.png")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args.classifier_type)

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")

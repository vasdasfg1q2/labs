import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

iris = load_iris()
X, y = iris.data, iris.target

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=0)

clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)

print("Ridge classifier на Iris")
print("Accuracy:         ", np.round(metrics.accuracy_score(ytest, ypred), 4))
print("Precision:        ", np.round(metrics.precision_score(ytest, ypred, average="weighted"), 4))
print("Recall:           ", np.round(metrics.recall_score(ytest, ypred, average="weighted"), 4))
print("F1 Score:         ", np.round(metrics.f1_score(ytest, ypred, average="weighted"), 4))
print("Cohen Kappa:      ", np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
print("Matthews Corrcoef:", np.round(metrics.matthews_corrcoef(ytest, ypred), 4))
print()
print("Classification Report:")
print(metrics.classification_report(ytest, ypred))

# heatmap confusion matrix
mat = confusion_matrix(ytest, ypred)
sns.set()
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False)
plt.xlabel("true label")
plt.ylabel("predicted label")
plt.tight_layout()
plt.savefig("outputs/task_5_confusion.jpg", dpi=120)
plt.close()

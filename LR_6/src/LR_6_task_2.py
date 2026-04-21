import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# Датасет про ціни на квитки іспанських швидкісних поїздів (RENFE)
URL = ("https://raw.githubusercontent.com/susanli2016/"
       "Machine-Learning-with-Python/master/data/renfe_small.csv")
try:
    df = pd.read_csv(URL)
except Exception as exc:
    print(f"Не вдалось завантажити CSV з URL: {exc}")
    sys.exit(1)

print(f"Форма датасету: {df.shape}")
print(f"Колонки: {list(df.columns)}")
print()
print("Перші 5 рядків:")
print(df.head().to_string())
print()
print("Пропущені значення по колонках:")
print(df.isna().sum())

# робимо цільовою змінною ціновий діапазон (категорія): low/mid/high
df = df.dropna(subset=["price"])
df["price_category"] = pd.qcut(df["price"], q=3, labels=["low", "mid", "high"])

# ознаки: train_type, train_class, fare, origin, destination
feature_cols = ["origin", "destination", "train_type", "train_class", "fare"]
df = df.dropna(subset=feature_cols + ["price_category"])

# кодуємо категоріальні ознаки
encoders = {}
X = np.empty((len(df), len(feature_cols)))
for i, col in enumerate(feature_cols):
    le = LabelEncoder()
    X[:, i] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
y = LabelEncoder().fit_transform(df["price_category"].astype(str))

print(f"\nX shape: {X.shape}, y shape: {y.shape}")
print(f"Розподіл класів (low/mid/high): {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5, stratify=y)

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== GaussianNB на RENFE ===")
print(classification_report(y_test, y_pred, target_names=["low", "mid", "high"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# heatmap без seaborn
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(["low", "mid", "high"])
ax.set_yticklabels(["low", "mid", "high"])
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Confusion matrix RENFE (GaussianNB)")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("outputs/task_2_renfe_confusion.png", dpi=120)
plt.close()

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")

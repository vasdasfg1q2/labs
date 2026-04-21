import io
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# Методичка використовує load_boston, але його вилучено з sklearn >= 1.2
# (етичні міркування). Замість нього використовуємо California Housing —
# це та сама задача регресії цін на житло, 8 ознак + target.
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = shuffle(housing.data, housing.target, random_state=7)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# AdaBoost з базовим регресором DecisionTreeRegressor глибини 4
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400, random_state=7,
)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

feature_importances = regressor.feature_importances_
feature_names = housing.feature_names
feature_importances = 100.0 * (feature_importances / max(feature_importances))
index_sorted = np.flipud(np.argsort(feature_importances))
pos = np.arange(index_sorted.shape[0]) + 0.5

plt.figure(figsize=(8, 5))
plt.bar(pos, feature_importances[index_sorted], align="center")
plt.xticks(pos, np.array(feature_names)[index_sorted], rotation=45)
plt.ylabel("Relative Importance")
plt.title("Feature importance (AdaBoost regressor)")
plt.tight_layout()
plt.savefig("outputs/task_4_feature_importance.png", dpi=120)
plt.close()

print("\nRelative feature importance:")
for i in index_sorted:
    print(f"  {feature_names[i]:<10} {feature_importances[i]:>6.2f}")

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")

import io
import sys

import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# PlayTennis dataset — таблиця з методички (14 днів, 3 ознаки, мітка Play)
data = [
    ["Sunny",    "High",   "Weak",   "No"],
    ["Sunny",    "High",   "Strong", "No"],
    ["Overcast", "High",   "Weak",   "Yes"],
    ["Rain",     "High",   "Weak",   "Yes"],
    ["Rain",     "Normal", "Weak",   "Yes"],
    ["Rain",     "Normal", "Strong", "No"],
    ["Overcast", "Normal", "Strong", "Yes"],
    ["Sunny",    "High",   "Weak",   "No"],
    ["Sunny",    "Normal", "Weak",   "Yes"],
    ["Rain",     "Normal", "Weak",   "Yes"],
    ["Sunny",    "Normal", "Strong", "Yes"],
    ["Overcast", "High",   "Strong", "Yes"],
    ["Overcast", "Normal", "Weak",   "Yes"],
    ["Rain",     "High",   "Strong", "No"],
]
data = np.array(data)

# окремі LabelEncoder на кожну колонку
encoders = [LabelEncoder() for _ in range(4)]
X_enc = np.empty(data.shape, dtype=int)
for i in range(4):
    X_enc[:, i] = encoders[i].fit_transform(data[:, i])

X = X_enc[:, :-1]
y = X_enc[:, -1]

# CategoricalNB — для категоріальних ознак з Laplace smoothing
clf = CategoricalNB(alpha=1e-10)  # майже нульова згладжка = чистий NB
clf.fit(X, y)

# Варіант 7 студента №22: Outlook=Overcast, Humidity=High, Wind=Strong
query_raw = ["Overcast", "High", "Strong"]
query_enc = np.array([
    encoders[i].transform([query_raw[i]])[0] for i in range(3)
]).reshape(1, -1)

proba = clf.predict_proba(query_enc)[0]
pred = clf.predict(query_enc)[0]
yes_label = encoders[3].transform(["Yes"])[0]
no_label = encoders[3].transform(["No"])[0]

print("Датасет Play Tennis (14 днів, 3 ознаки):")
print("  Outlook ∈ {Sunny, Overcast, Rain}")
print("  Humidity ∈ {High, Normal}")
print("  Wind ∈ {Weak, Strong}")
print()
print(f"Варіант студента 22 → №7: {query_raw}")
print()
print(f"P(Yes | query) = {proba[yes_label]:.4f}")
print(f"P(No  | query) = {proba[no_label]:.4f}")
print(f"Прогноз: Play = {encoders[3].inverse_transform([pred])[0]}")
print()
# Ручна перевірка (як у методичці §2)
yes_rows = data[data[:, -1] == "Yes"]
no_rows = data[data[:, -1] == "No"]
p_yes = len(yes_rows) / len(data)
p_no = len(no_rows) / len(data)

def cond(col_idx, value, rows):
    return (rows[:, col_idx] == value).sum() / len(rows)

p_overcast_yes = cond(0, "Overcast", yes_rows)
p_overcast_no = cond(0, "Overcast", no_rows)
p_high_yes = cond(1, "High", yes_rows)
p_high_no = cond(1, "High", no_rows)
p_strong_yes = cond(2, "Strong", yes_rows)
p_strong_no = cond(2, "Strong", no_rows)

yes_num = p_overcast_yes * p_high_yes * p_strong_yes * p_yes
no_num = p_overcast_no * p_high_no * p_strong_no * p_no
print("Ручний розрахунок Naive Bayes:")
print(f"  P(Yes)·Π P(xᵢ|Yes) = {yes_num:.4f}")
print(f"  P(No)·Π P(xᵢ|No)   = {no_num:.4f}")
if (yes_num + no_num) > 0:
    print(f"  Нормалізована P(Yes) = {yes_num/(yes_num+no_num):.4f}")
    print(f"  Нормалізована P(No)  = {no_num/(yes_num+no_num):.4f}")
else:
    print("  Сума ймовірностей = 0 → модель невпевнена (Overcast+High+Strong → лише день D12 з Yes, недостатньо для No)")

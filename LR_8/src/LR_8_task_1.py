import io
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # приглушуємо info-повідомлення TF

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("outputs", exist_ok=True)

# Методичка написана на TF1 (placeholder, Session). У TF2 використовуємо
# eager execution + tf.GradientTape — функціональний еквівалент.
tf.random.set_seed(0)
np.random.seed(0)

n_samples, batch_size, num_steps = 1000, 100, 20000
X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)

# Навчані змінні: k ~ N(0, 1), b = 0. Еквівалент tf.Variable з методички.
k = tf.Variable(tf.random.normal((1, 1)), name="slope")
b = tf.Variable(tf.zeros((1,)), name="bias")

# Оптимізатор — SGD, як у методичці (tf.train.GradientDescentOptimizer).
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

display_step = 2000
print(f"TensorFlow {tf.__version__} — навчання y = k·x + b")
print(f"n_samples={n_samples}, batch_size={batch_size}, num_steps={num_steps}")
print()

for i in range(num_steps):
    indices = np.random.choice(n_samples, batch_size, replace=False)
    X_batch = X_data[indices]
    y_batch = y_data[indices]

    # GradientTape автоматично відстежує операції для автодиференціювання.
    with tf.GradientTape() as tape:
        y_pred = tf.matmul(X_batch, k) + b
        loss = tf.reduce_sum((y_batch - y_pred) ** 2)

    # Обчислюємо градієнти по k і b, застосовуємо SGD step.
    grads = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(grads, [k, b]))

    if (i + 1) % display_step == 0:
        print(f"Епоха {i+1}: {loss.numpy():.8f}, "
              f"k={k.numpy().item():.4f}, b={b.numpy().item():.4f}")

print()
print(f"Еталон: k=2.0, b=1.0 (дані y = 2x + 1 + N(0, 2))")
print(f"Отримано:  k={k.numpy().item():.4f}, b={b.numpy().item():.4f}")

# Візуалізація
plt.figure(figsize=(7, 5))
idx_show = np.random.choice(n_samples, 200, replace=False)
plt.scatter(X_data[idx_show], y_data[idx_show], alpha=0.4, s=15,
            color="steelblue", label="data")

xs = np.linspace(1, 10, 100)
k_val, b_val = k.numpy().item(), b.numpy().item()
plt.plot(xs, k_val * xs + b_val, "r-", linewidth=2,
         label=f"навчена: y = {k_val:.2f}x + {b_val:.2f}")
plt.plot(xs, 2 * xs + 1, "g--", linewidth=2, label="еталон: y = 2x + 1")
plt.xlabel("X"); plt.ylabel("y")
plt.title(f"Лінійна регресія (SGD, TensorFlow {tf.__version__})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/task_1_fit.png", dpi=120)
plt.close()

if matplotlib.get_backend().lower() != "agg":
    plt.show()
plt.close("all")

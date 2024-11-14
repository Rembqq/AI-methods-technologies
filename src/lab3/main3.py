import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Генеруємо випадкові дані з двома ознаками (наприклад, зріст і вага)
np.random.seed(42)
data = np.vstack((
    np.random.normal([50, 150], [5, 10], (100, 2)),
    np.random.normal([70, 170], [5, 10], (100, 2)),
    np.random.normal([90, 190], [5, 10], (100, 2))
))

# Візуалізація початкових даних
plt.scatter(data[:, 0], data[:, 1])
plt.title("Початкові дані")
plt.xlabel("Зріст")
plt.ylabel("Вага")
plt.show()

# Параметри FCM
n_clusters = 3  # Кількість кластерів
m = 1.5  # Ступінь нечіткості
epsilon = 1e-6  # Точність для зупинки алгоритму

# Застосування FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T, n_clusters, m, error=epsilon, maxiter=1000, init=None)

# Виведення центрів кластерів
print("Центри кластерів:\n", cntr)

# Графік зміни значень цільової функції
plt.figure()
plt.plot(jm, marker='o')
plt.title("Зміна значень цільової функції")
plt.xlabel("Номер ітерації")
plt.ylabel("Значення цільової функції")
plt.grid(True)
plt.show()

# Візуалізація кластеризації
plt.figure()
colors = ['r', 'g', 'b']
for i in range(n_clusters):
    plt.scatter(data[:, 0], data[:, 1], c=u[i, :], cmap='viridis', label=f"Кластер {i+1}")
    plt.scatter(cntr[i][0], cntr[i][1], color=colors[i], marker='X', s=200, label=f"Центр {i+1}")
plt.legend()
plt.title("Кластеризація даних методом FCM")
plt.xlabel("Зріст")
plt.ylabel("Вага")
plt.grid(True)
plt.show()

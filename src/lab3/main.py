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
plt.xlabel("Вага")
plt.ylabel("Зріст")
plt.show()

# Параметри FCM
m = 1.5  # Ступінь нечіткості
epsilon = 1e-6  # Точність для зупинки алгоритму
max_clusters = 10  # Максимальна кількість кластерів для аналізу

# Список для збереження коефіцієнтів FPC
fpc_values = []

# Обчислення FCM для різних кількостей кластерів
for n_clusters in range(2, max_clusters + 1):
    # Застосування FCM
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data.T, n_clusters, m, error=epsilon, maxiter=1000, init=None)

    # Збереження коефіцієнта FPC
    fpc_values.append((n_clusters, fpc))

    # Виведення у консоль кількості кластерів і FPC
    print(f"Кількість кластерів: {n_clusters}, FPC: {fpc:.4f}")

# Побудова графіка коефіцієнта FPC залежно від кількості кластерів
clusters, fpc_scores = zip(*fpc_values)
plt.figure()
plt.plot(clusters, fpc_scores, marker='o', linestyle='-', color='b')
plt.title("Коефіцієнт FPC(розбиття) залежно від кількості кластерів")
plt.xlabel("Кількість кластерів")
plt.ylabel("Коефіцієнт FPC")
plt.grid(True)
plt.show()

# Кластеризація для фіксованої кількості кластерів
n_clusters = 3  # Фіксована кількість кластерів для візуалізації
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T, n_clusters, m, error=epsilon, maxiter=1000, init=None)

# Виведення центрів кластерів
print("Центри кластерів:\n", cntr)

# Графік зміни значень цільової функції
# plt.figure()
# plt.plot(jm, marker='o')
# plt.title("Зміна значень цільової функції")
# plt.xlabel("Номер ітерації")
# plt.ylabel("Значення цільової функції")
# #plt.ylim(1, 20000)  # Зміна діапазону осі Y від 1 до 10
# plt.grid(True)
#plt.show()

# Візуалізація кластеризації
plt.figure()
colors = ['r', 'g', 'b']
for i in range(n_clusters):
    plt.scatter(data[:, 0], data[:, 1], c=u[i, :], cmap='viridis')  # Прибираємо label
    plt.scatter(cntr[i][0], cntr[i][1], color=colors[i], marker='X', s=200)  # Прибираємо label
plt.title("Кластеризація даних методом FCM")
plt.xlabel("Вага")
plt.ylabel("Зріст")
plt.grid(True)
plt.show()

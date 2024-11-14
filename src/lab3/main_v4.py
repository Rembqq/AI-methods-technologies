import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# Завантажуємо дані Iris
iris = datasets.load_iris()
data = iris.data.T  # Транспонуємо для сумісності з FCM
labels_true = iris.target  # Істинні мітки для перевірки кластеризації

# Знижуємо розмірність до 2D для візуалізації
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data.T).T  # Знову транспонуємо для FCM

# Параметри FCM
n_clusters = 3  # Очікуємо 3 кластери, оскільки в наборі даних Iris є три види квітів
m = 2.0  # Ступінь нечіткості
epsilon = 1e-6  # Точність для зупинки алгоритму

# Застосування FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_2d, n_clusters, m, error=epsilon, maxiter=1000, init=None)

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

# Візуалізація кластеризації у 2D-просторі
plt.figure()
colors = ['r', 'g', 'b']
for i in range(n_clusters):
    plt.scatter(data_2d[0, :], data_2d[1, :], c=u[i, :], cmap='viridis', label=f"Кластер {i+1}")
    plt.scatter(cntr[i][0], cntr[i][1], color=colors[i], marker='X', s=200, label=f"Центр {i+1}")
plt.legend()
plt.title("Кластеризація даних Iris методом FCM")
plt.xlabel("Перша компонента PCA")
plt.ylabel("Друга компонента PCA")
plt.grid(True)
plt.show()

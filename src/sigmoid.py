import numpy as np
import matplotlib.pyplot as plt

# Основна сигмоїда (відкрита справа або зліва)
def sigmoid(x, k, c):
    return 1 / (1 + np.exp(-k * (x - c)))

# Двостороння сигмоїда
def double_sigmoid(x, k1, k2, c1, c2):
    return sigmoid(x, k1, c1) * (1 - sigmoid(x, k2, c2))

# Немсиметрична двостороння сигмоїда
def asym_double_sigmoid(x, k1, k2, c1, c2):
    left = 1 / (1 + np.exp(-k1 * (x - c1)))
    right = 1 / (1 + np.exp(k2 * (x - c2)))
    return np.minimum(left, right)

# Параметри для побудови графіків
x = np.linspace(0, 10, 100)

# Параметри для основної сигмоїди
k_main = 2
c_main = 5

# Параметри для двосторонньої сигмоїди
k1_double, k2_double = 2, 2
c1_double, c2_double = 3, 7

# Параметри для несиметричної двосторонньої сигмоїди
k1_asym, k2_asym = 3, 1
c1_asym, c2_asym = 3, 7

# Обчислення функцій
main_sigmoid_y = sigmoid(x, k_main, c_main)
double_sigmoid_y = double_sigmoid(x, k1_double, k2_double, c1_double, c2_double)
asym_double_sigmoid_y = asym_double_sigmoid(x, k1_asym, k2_asym, c1_asym, c2_asym)

# Візуалізація
plt.figure(figsize=(12, 8))

# Основна сигмоїда
plt.subplot(3, 1, 1)
plt.plot(x, main_sigmoid_y, 'b', label="Основна сигмоїда")
plt.title('Основна одностороння сигмоїда')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

# Двостороння сигмоїда
plt.subplot(3, 1, 2)
plt.plot(x, double_sigmoid_y, 'r', label="Двостороння сигмоїда")
plt.title('Двостороння сигмоїда')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

# Немсиметрична двостороння сигмоїда
plt.subplot(3, 1, 3)
plt.plot(x, asym_double_sigmoid_y, 'g', label="Немсиметрична двостороння сигмоїда")
plt.title('Несиметрична двостороння сигмоїда')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

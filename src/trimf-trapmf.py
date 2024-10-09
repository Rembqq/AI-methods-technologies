import numpy as np
import matplotlib.pyplot as plt

# Трикутна функція приналежності
def triangular(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

# Трапецієподібна функція приналежності
def trapezoidal(x, a, b, c, d):
    return np.maximum(np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)), 0)

# Параметри для функцій
x = np.linspace(0, 10, 100)
a_tri, b_tri, c_tri = 2, 5, 8  # Параметри трикутної функції
a_trap, b_trap, c_trap, d_trap = 2, 4, 6, 8  # Параметри трапецієподібної функції

# Обчислення функцій
tri_y = triangular(x, a_tri, b_tri, c_tri)
trap_y = trapezoidal(x, a_trap, b_trap, c_trap, d_trap)

# Візуалізація
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(x, tri_y, 'b', label="Triangular")
plt.title('Трикутна функція приналежності')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, trap_y, 'r', label="Trapezoidal")
plt.title('Трапецієподібна функція приналежності')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

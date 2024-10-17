import numpy as np
import matplotlib.pyplot as plt

# Функції приналежності A і B
def membership_A(x):
    return np.piecewise(x, [x <= 4, x > 4], [lambda x: x / 4, lambda x: (10 - x) / 6])

def membership_B(x):
    return np.piecewise(x, [x <= 3, x > 3], [lambda x: x / 3, lambda x: (9 - x) / 6])

# Мінімум (AND)
def fuzzy_and(A, B):
    return np.minimum(A, B)

# Максимум (OR)
def fuzzy_or(A, B):
    return np.maximum(A, B)

# Заперечення (NOT)
def fuzzy_not(A):
    return 1 - A

# Вхідний діапазон
x = np.linspace(0, 10, 100)

# Обчислення значень функцій приналежності A і B
A = membership_A(x)
B = membership_B(x)

# Застосування логічних операторів
A_and_B = fuzzy_and(A, B)
A_or_B = fuzzy_or(A, B)
not_A = fuzzy_not(A)

# Візуалізація результатів
plt.figure(figsize=(10, 8))

# Функція приналежності A
plt.subplot(2, 1, 1)
plt.plot(x, A, 'b', label='A(x)')
plt.plot(x, B, 'r', label='B(x)')
plt.fill_between(x, A_and_B, color='g', alpha=0.3, label='A AND B')
plt.title('Кон\'юнкція (AND) - Мінімум')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

# Диз'юнкція (OR)
plt.subplot(2, 1, 2)
plt.plot(x, A, 'b', label='A(x)')
plt.plot(x, B, 'r', label='B(x)')
plt.fill_between(x, A_or_B, color='orange', alpha=0.3, label='A OR B')
plt.title('Диз\'юнкція (OR) - Максимум')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

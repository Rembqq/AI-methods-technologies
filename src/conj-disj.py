import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Вихідні ФП (як приклади)
x = np.linspace(0, 10, 100)
v1 = np.exp(-0.1 * (x - 5)**2)  # Перша функція приналежності (ФП)
v2 = np.exp(-0.05 * (x - 5)**2)  # Друга функція приналежності (ФП)

# Кон'юнктивний оператор: min(v1, v2)
def conjunctive_operator(v1, v2):
    return np.minimum(v1, v2)

# Диз'юнктивний оператор: max(v1, v2)
def disjunctive_operator(v1, v2):
    return np.maximum(v1, v2)

# Обчислюємо результати для обох операторів
conjunctive_result = conjunctive_operator(v1, v2)
disjunctive_result = disjunctive_operator(v1, v2)

# Візуалізація результатів
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Графік для кон'юнктивного оператора (min)
ax[0].plot(x, v1, 'k--', label='v1 (ФП1)')
ax[0].plot(x, v2, 'k:', label='v2 (ФП2)')
ax[0].plot(x, conjunctive_result, 'k-', label='min(v1, v2)')
ax[0].set_title('Кон\'юнкція (min)')
ax[0].set_ylim([0.4, 1.0])
ax[0].legend()

# Графік для диз'юнктивного оператора (max)
ax[1].plot(x, v1, 'k--', label='v1 (ФП1)')
ax[1].plot(x, v2, 'k:', label='v2 (ФП2)')
ax[1].plot(x, disjunctive_result, 'k-', label='max(v1, v2)')
ax[1].set_title('Диз\'юнкція (max)')
ax[1].set_ylim([0.4, 1.0])
ax[1].legend()

plt.show()

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

# Основна одностороння сігмоїдна функція (відкрита зліва)
sigmoid_left = fuzz.sigmf(x, 0, 1)  # (x, c, b)

# Основна одностороння сігмоїдна функція (відкрита справа)
sigmoid_right = fuzz.sigmf(x, 0, -1)  # (x, c, b)

# Двостороння сігмоїдна функція
double_sided = fuzz.psigmf(x, 0, 1, 0, 1)  # (x, c1, b1, c2, b2)

# Несиметрична сігмоїдна функція
asymmetric = fuzz.dsigmf(x, 0, 1, 1, -1)  # (x, c, b1, b2)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid_left, label='Одностороння (ліва)', linestyle='--')
plt.plot(x, sigmoid_right, label='Одностороння (права)', linestyle='--')
plt.plot(x, double_sided, label='Двостороння', linestyle='-')
plt.plot(x, asymmetric, label='Несиметрична', linestyle='-')
plt.title('Сігмоїдні функції')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.legend()
plt.grid()
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.show()
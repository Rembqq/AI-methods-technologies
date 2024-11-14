import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Вихідна нечітка множина
x = np.linspace(0, 10, 100)
v1 = np.exp(-0.1 * (x - 5)**2)

# Доповнення нечіткої множини
complement = 1 - v1

# Візуалізація
plt.plot(x, v1, label='Нечітка множина A')
plt.plot(x, complement, label='Доповнення A')
plt.title('Доповнення нечіткої множини')
plt.legend()
plt.show()


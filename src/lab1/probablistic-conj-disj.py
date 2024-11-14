import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

v1 = np.exp(-0.1 * (x - 5)**2)
v2 = np.exp(-0.05 * (x - 5)**2)

# Вірогідна кон'юнкція: вірогідний мінімум
probabilistic_conjunction = v1 * v2

# Вірогідна диз'юнкція: вірогідний максимум
probabilistic_disjunction = v1 + v2 - v1 * v2


plt.plot(x, probabilistic_conjunction, label='Вірогідна Кон\'юнкція')
plt.plot(x, probabilistic_disjunction, label='Вірогідна Диз\'юнкція')
plt.title('Вірогідна інтерпретація')
plt.legend()
plt.show()

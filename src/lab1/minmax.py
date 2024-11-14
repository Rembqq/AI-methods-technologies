import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

v1 = np.exp(-0.1 * (x - 5)**2)
v2 = np.exp(-0.05 * (x - 5)**2)

# Кон'юнкція (min)
conjunction = np.minimum(v1, v2)

# Диз'юнкція (max)
disjunction = np.maximum(v1, v2)

# Візуалізація
plt.plot(x, conjunction, label='Кон\'юнкція (min)')
plt.plot(x, disjunction, label='Диз\'юнкція (max)')
plt.title('Мінімаксна інтерпретація')
plt.legend()
plt.show()


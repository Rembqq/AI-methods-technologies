import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

# Проста функція Гаусса
gaussian = fuzz.gaussmf(x, mean=5, sigma=1)

# Двостороння функція Гаусса
gaussian_bell = fuzz.gauss2mf(x, mean1=3, sigma1=1, mean2=7, sigma2=1)

# Візуалізація
plt.plot(x, gaussian, label='Проста Гауссова')
plt.plot(x, gaussian_bell, label='Двостороння Гауссова')
plt.title('Функції Гаусса')
plt.legend()
plt.show()


import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

# Z-функція
z_function = fuzz.zmf(x, 3, 7)

# S-функція
s_function = fuzz.smf(x, 3, 7)

# Пі-функція
pi_function = fuzz.pimf(x, 3, 5, 7, 9)


plt.plot(x, z_function, label='Z-функція')
plt.plot(x, s_function, label='S-функція')
plt.plot(x, pi_function, label='Пі-функція')
plt.title('Поліноміальні функції приналежності')
plt.legend()
plt.show()

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 100)


triangular = fuzz.trimf(x, [2, 5, 8])


trapezoidal = fuzz.trapmf(x, [2, 4, 6, 8])


plt.plot(x, triangular, label='Трикутна')
plt.plot(x, trapezoidal, label='Трапецієподібна')
plt.title('Трикутна і Трапецієподібна функції приналежності')
plt.legend()
plt.show()

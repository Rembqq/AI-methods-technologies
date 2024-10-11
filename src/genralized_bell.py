import numpy as np
import matplotlib.pyplot as plt

# Узагальнена дзвінова функція приналежності
def generalized_bell(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a)**(2 * b))

# Параметри функції
x = np.linspace(0, 10, 100)
a = 2  # ширина
b = 4  # крутизна
c = 5  # центр

# Обчислення функції
bell_y = generalized_bell(x, a, b, c)

# Візуалізація
plt.plot(x, bell_y, label="Узагальнений дзвін", color='green')
plt.title('Функція приналежності "Узагальнений дзвін"')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()
plt.show()

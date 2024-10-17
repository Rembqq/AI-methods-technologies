import numpy as np
import matplotlib.pyplot as plt

# Вихідна нечітка множина
x = np.linspace(0, 10, 100)
y_ = np.exp(-0.1 * (x - 5)**2)  # Функція приналежності нечіткої множини A

# Доповнення нечіткої множини
def complement(mu_A):
    return 1 - mu_A

# Обчислюємо доповнення
y_complement = complement(y_)

# Візуалізація
plt.figure(figsize=(8, 5))

# Вихідна нечітка множина
plt.plot(x, y_, 'k-', label='Нечітка множина Y')

# Доповнення нечіткої множини
plt.plot(x, y_complement, 'r-', label='Доповнення Y (1 - y*)')

plt.title('Доповнення нечіткої множини')
plt.xlabel('x')
plt.ylabel('Ступінь приналежності')
plt.ylim([0, 1])
plt.legend()

plt.show()

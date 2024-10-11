import numpy as np
import matplotlib.pyplot as plt

# Проста Гауссова функція приналежності
def gaussian(x, c, sigma):
    return np.exp(-((x - c)**2) / (2 * sigma**2))

# Двостороння Гауссова функція приналежності
def bilateral_gaussian(x, c, sigma1, sigma2):
    return np.where(x <= c,
                    np.exp(-((x - c)**2) / (2 * sigma1**2)),
                    np.exp(-((x - c)**2) / (2 * sigma2**2)))

# Параметри
x = np.linspace(0, 10, 100)  # визначаємо x
c = 5  # центр для обох функцій

# Параметри для двосторонньої Гауссової функції
sigma1 = 1  # ліве стандартне відхилення
sigma2 = 2  # праве стандартне відхилення

# Обчислення двосторонньої Гауссової функції
bilateral_gaussian_y = bilateral_gaussian(x, c, sigma1, sigma2)

# Візуалізація двосторонньої Гауссової функції
plt.plot(x, bilateral_gaussian_y, label="Двостороння Гауссова", color='orange')
plt.title('Двостороння функція приналежності Гаусса')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

# Параметри для простої Гауссової функції
sigma = 1  # стандартне відхилення

# Обчислення простої Гауссової функції
gaussian_y = gaussian(x, c, sigma)

# Візуалізація простої Гауссової функції
plt.plot(x, gaussian_y, label="Проста Гауссова", color='blue')
plt.title('Функції приналежності Гаусса')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

# Відображення обох графіків
plt.show()

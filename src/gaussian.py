import numpy as np
import matplotlib.pyplot as plt

# Проста Гауссова функція приналежності
def gaussian(x, c, sigma):
    return np.exp(-((x - c)**2) / (2 * sigma**2))

# Двостороння Гауссова функція приналежності (з плавним переходом)
def bilateral_gaussian(x, c, sigma1, sigma2):
    return np.where(x <= c,
                    np.exp(-((x - c)**2) / (2 * sigma1**2)),
                    np.exp(-((x - c)**2) / (2 * sigma2**2)))

# Параметри для двосторонньої Гауссової функції
x = np.linspace(0, 10, 100)
c = 5  # центр функції
sigma1 = 1.5  # стандартне відхилення лівої частини
sigma2 = 1.5  # стандартне відхилення правої частини

# Обчислення двосторонньої Гауссової функції
bilateral_gaussian_y = bilateral_gaussian(x, c, sigma1, sigma2)

# Обчислення простої Гауссової функції
sigma = 1  # стандартне відхилення для простої функції
gaussian_y = gaussian(x, c, sigma)

# Візуалізація
plt.plot(x, bilateral_gaussian_y, label="Двостороння Гауссова", color='orange')
plt.plot(x, gaussian_y, label="Проста Гауссова", color='blue')

# Налаштування графіка
plt.title('Функції приналежності Гаусса')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()
plt.show()

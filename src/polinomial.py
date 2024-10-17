import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# S-функція
def s_function(x, a, b):
    result = np.piecewise(x,
                          [x <= a, (x > a) & (x <= (a + b) / 2), (x > (a + b) / 2) & (x <= b), x > b],
                          [0, lambda x: 2 * ((x - a) / (b - a)) ** 2,
                           lambda x: 1 - 2 * ((b - x) / (b - a)) ** 2, 1])
    return result

# Z-функція
def z_function(x, a, b):
    result = np.piecewise(x,
                          [x <= a, (x > a) & (x <= (a + b) / 2), (x > (a + b) / 2) & (x <= b), x > b],
                          [1, lambda x: 1 - 2 * ((x - a) / (b - a)) ** 2,
                           lambda x: 2 * ((b - x) / (b - a)) ** 2, 0])
    return result

# PI-функція (комбінація S і Z-функцій)
def pi_function(x, a, b, c, d):
    s_part = s_function(x, a, b)
    z_part = z_function(x, c, d)
    return np.maximum(s_part, z_part)

# Параметри для побудови графіків
x = np.linspace(0, 10, 200)

# Параметри для S- і Z-функцій
a_s, b_s = 2, 5
a_z, b_z = 5, 8

# Параметри для PI-функції
a_pi, b_pi, c_pi, d_pi = 2, 4, 6, 8

# Обчислення функцій
s_y = s_function(x, a_s, b_s)
z_y = z_function(x, a_z, b_z)
pi_y = pi_function(x, a_pi, b_pi, c_pi, d_pi)

# Візуалізація
plt.figure(figsize=(12, 8))

# S-функція
plt.subplot(3, 1, 1)
plt.plot(x, s_y, 'b', label="S-функція")
plt.title('S-функція')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

# Z-функція
plt.subplot(3, 1, 2)
plt.plot(x, z_y, 'r', label="Z-функція")
plt.title('Z-функція')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

# PI-функція
plt.subplot(3, 1, 3)
plt.plot(x, pi_y, 'g', label="PI-функція")
plt.title('PI-функція')
plt.xlabel('x')
plt.ylabel('Приналежність')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
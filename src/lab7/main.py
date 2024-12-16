import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Функція однієї змінної
def func_single_variable(x):
    #return (np.sin(x) / x,) if x != 0 else (0,)
    return np.where(x != 0, np.sin(x) / x, 0)
# Функція двох змінних
def func_two_variables(x):
    x1, x2 = x
    return -((x1 - x2) * np.sin(x1 + x2))

# Межі для оптимізації
bounds_single = [(-4, 1)]  # Для функції однієї змінної
bounds_two = [(-1, 3), (-1, 3)]  # Для функції двох змінних

# Мінімізація функції однієї змінної
result_single = differential_evolution(func_single_variable, bounds_single, maxiter=100, popsize=10)
x_min_single = result_single.x[0]
y_min_single = result_single.fun

# Максимізація функції двох змінних
result_two = differential_evolution(func_two_variables, bounds_two, maxiter=100, popsize=10)
x1_max, x2_max = result_two.x
z_max = -result_two.fun

# Графік функції однієї змінної
x = np.linspace(-4, 1, 500)
y = func_single_variable(x)
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="y = sin(x) / x", color="black")
plt.scatter([x_min_single], [y_min_single], color="red", label=f"Min: x={x_min_single:.3f}, y={y_min_single:.3f}")
plt.title("single var graph - minimization ")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
#plt.savefig("minimization.png")
plt.show()

# Тривимірний графік функції двох змінних
x1 = np.linspace(-1, 3, 100)
x2 = np.linspace(-1, 3, 100)
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
z = (x1_mesh - x2_mesh) * np.sin(x1_mesh + x2_mesh)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x1_mesh, x2_mesh, z, cmap="viridis", alpha=0.8)
ax.scatter(x1_max, x2_max, z_max, color="red", s=50, label=f"Max: x1={x1_max:.3f}, x2={x2_max:.3f}, z={z_max:.3f}")
ax.set_title("Two vars graph - maximization")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("z")
ax.legend()
#plt.savefig("maximization.png")
plt.show()

# Результати
# Результати
print("=" * 40)
print("РЕЗУЛЬТАТИ ОПТИМІЗАЦІЇ".center(40))
print("=" * 40)
print("Функція однієї змінної:")
print(f"  Мінімальне значення:")
print(f"    x = {x_min_single:.3f}")
print(f"    y = {y_min_single:.3f}")
print("-" * 40)
print("Функція двох змінних:")
print(f"  Максимальне значення:")
print(f"    x1 = {x1_max:.3f}")
print(f"    x2 = {x2_max:.3f}")
print(f"    z = {z_max:.3f}")
print("=" * 40)


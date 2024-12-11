import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from skfuzzy import control as ctrl
import skfuzzy as fuzz

# 1. Налаштування випадкових значень для відтворюваності
np.random.seed(42)
tf.random.set_seed(42)

# 2. Генерація випадкових даних для вхідних змінних
n_samples = 1000  # кількість зразків
temperature = np.random.uniform(10, 40, size=n_samples)  # температура (C)
vibration = np.random.uniform(0, 10, size=n_samples)  # вібрація
pressure = np.random.uniform(1, 5, size=n_samples)  # тиск

# 3. Перетворення категорійних змінних в числовий формат для моделі
product_defect = np.random.choice([0, 1], size=n_samples)  # дефекти продукції

# 4. Формування вибірки в DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'product_defect': product_defect
})

# 5. Генерація залежної змінної "споживання енергії" з додаванням випадкового шуму
energy_consumption = 0.5 * temperature + 0.3 * vibration + 0.2 * pressure + np.random.normal(0, 1, n_samples)

df['energy_consumption'] = energy_consumption

# 6. Нечіткі змінні
temperature_fuzz = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
vibration_fuzz = ctrl.Antecedent(np.arange(0, 11, 1), 'vibration')
pressure_fuzz = ctrl.Antecedent(np.arange(0, 6, 1), 'pressure')
defect_fuzz = ctrl.Consequent(np.arange(0, 2, 1), 'defect')

# 7. Функції належності для нечітких змінних
temperature_fuzz['low'] = fuzz.trapmf(temperature_fuzz.universe, [0, 0, 10, 20])
temperature_fuzz['medium'] = fuzz.trimf(temperature_fuzz.universe, [15, 25, 35])
temperature_fuzz['high'] = fuzz.trapmf(temperature_fuzz.universe, [30, 40, 40, 40])

vibration_fuzz['low'] = fuzz.trapmf(vibration_fuzz.universe, [0, 0, 2, 4])
vibration_fuzz['medium'] = fuzz.trimf(vibration_fuzz.universe, [3, 5, 7])
vibration_fuzz['high'] = fuzz.trapmf(vibration_fuzz.universe, [6, 8, 10, 10])

pressure_fuzz['low'] = fuzz.trapmf(pressure_fuzz.universe, [0, 0, 1, 2])
pressure_fuzz['medium'] = fuzz.trimf(pressure_fuzz.universe, [1, 2, 3])
pressure_fuzz['high'] = fuzz.trapmf(pressure_fuzz.universe, [2, 4, 5, 5])

defect_fuzz['no_defect'] = fuzz.trimf(defect_fuzz.universe, [0, 0, 1])
defect_fuzz['defect'] = fuzz.trimf(defect_fuzz.universe, [0, 1, 1])

# 8. Правила нечіткої логіки
rule1 = ctrl.Rule(temperature_fuzz['high'] & vibration_fuzz['high'] & pressure_fuzz['high'], defect_fuzz['defect'])
rule2 = ctrl.Rule(temperature_fuzz['low'] & vibration_fuzz['low'] & pressure_fuzz['low'], defect_fuzz['no_defect'])

# 9. Створення системи нечіткої логіки
defect_ctrl = ctrl.ControlSystem([rule1, rule2])
defect_sim = ctrl.ControlSystemSimulation(defect_ctrl)

# 10. Обчислення нечітких виходів і додавання їх як нових ознак
defect_output = []
for i in range(n_samples):
    defect_sim.input['temperature'] = df['temperature'].iloc[i]
    defect_sim.input['vibration'] = df['vibration'].iloc[i]
    defect_sim.input['pressure'] = df['pressure'].iloc[i]

    # Запуск обчислення
    defect_sim.compute()

    # Перевірка, чи є значення в дефекті
    if 'defect' in defect_sim.output:
        defect_output.append(defect_sim.output['defect'])
    else:
        defect_output.append(0)  # Якщо значення немає, присвоюємо 0

df['defect_output'] = defect_output

# 11. Формування вибірки
X = df[['temperature', 'vibration', 'pressure', 'defect_output']].values
y = df['energy_consumption'].values

# 12. Поділ на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 13. Нормалізація даних
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 14. Нейронна мережа
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 15. Компіляція моделі
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 16. Навчання моделі (кількість епох збільшена до 200)
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

# 17. Передбачення та обчислення помилки
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))

# Виведення результатів
print(f'Середня квадратична помилка (MSE): {mse}')
print(f'Середня абсолютна помилка (MAE): {mae}')

# 18. Графік втрат під час навчання
plt.plot(history.history['loss'], label='Втрати на тренувальному наборі', color='blue')
plt.plot(history.history['val_loss'], label='Втрати на тестовому наборі', color='red')
plt.title('Зміна значень втрат під час тренування', fontsize=14)  # перефразували заголовок
plt.xlabel('Етапи навчання', fontsize=12)  # змінили назву осі X
plt.ylabel('Втрати', fontsize=12)  # змінили назву осі Y
plt.legend()
plt.show()

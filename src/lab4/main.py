import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM
from sklearn.metrics import mean_absolute_error

# Генерація даних відповідно до нових формул
np.random.seed(42)
a = np.random.uniform(0.1, 10, 1000)  # Уникнення поділу на 0
b = np.random.uniform(0.1, 10, 1000)
c = np.random.uniform(0.1, 10, 1000)

# Формула для моделювання даних
x = -25 / a + c - b * a
y = (1 + c * b / 2)
z = x / y  # Вихід

# Перетворення у форму даних для моделі
X = np.stack([a, b, c], axis=1)  # Вхід: a, b, c
y = z  # Вихід: z

# Розподіл даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Додавання виміру для часових кроків (необхідно для RNN)
X_train_rnn = np.expand_dims(X_train, axis=1)
X_test_rnn = np.expand_dims(X_test, axis=1)

# Функція для побудови, навчання та візуалізації моделі
def build_and_evaluate_model_with_plots(model_type, layers, neurons, X_train, y_train, X_test, y_test, config_name):
    model = Sequential()

    if model_type == 'elman':
        input_shape = (X_train.shape[1], X_train.shape[2])
        for i in range(layers):
            model.add(LSTM(neurons, input_shape=input_shape if i == 0 else None,
                           return_sequences=(i < layers - 1)))

    elif model_type == 'feed_forward':
        X_train, X_test = X_train[:, 0, :], X_test[:, 0, :]
        for _ in range(layers):
            model.add(Dense(neurons, activation='relu', input_dim=X_train.shape[1]))

    elif model_type == 'cascade_forward':
        X_train, X_test = X_train[:, 0, :], X_test[:, 0, :]
        for _ in range(layers):
            model.add(Dense(neurons, activation='relu', input_dim=X_train.shape[1]))
            model.add(Dense(neurons, activation='relu'))

    # Вихідний шар
    model.add(Dense(1))

    # Компіляція моделі
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Навчання з збереженням історії
    history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)

    # Оцінка
    predictions = model.predict(X_test)
    relative_mae = mean_absolute_error(y_test, predictions) / np.mean(np.abs(y_test))

    # Побудова графіків
    plt.figure(figsize=(12, 5))

    # Графік 1: Залежність передбачень від реальних значень
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions, color='blue', alpha=0.7, label="Передбачення")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', label="Ідеальна лінія")
    plt.title(f"Тип мережі: {model_type.capitalize()}, {config_name}")
    plt.xlabel("Реальні значення (R)")
    plt.ylabel("Передбачені значення (P)")
    plt.legend()

    # Графік 2: Залежність помилки від епох
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], color='red')
    plt.ylim(0.0, 0.4)
    plt.title(f"Крива втрат для: {model_type.capitalize()}, {config_name}")
    plt.xlabel("Епохи")
    plt.ylabel("Помилка")

    # Показати та зберегти графік
    plt.tight_layout()
    plt.show()

    return relative_mae

results_with_plots = []

# Feed Forward (пряме розповсюдження)
results_with_plots.append(("Feed Forward, 1 прихований шар, 10 нейронів",
                           build_and_evaluate_model_with_plots('feed_forward', 1, 10, X_train_rnn, y_train, X_test_rnn, y_test, "1 layer (10) FN")))
results_with_plots.append(("Feed Forward, 1 прихований шар, 20 нейронів",
                           build_and_evaluate_model_with_plots('feed_forward', 1, 20, X_train_rnn, y_train, X_test_rnn, y_test, "1 layer (20) FN")))

# Cascade Forward (каскадне розповсюдження)
results_with_plots.append(("Cascade Forward, 1 прихований шар, 20 нейронів",
                           build_and_evaluate_model_with_plots('cascade_forward', 1, 20, X_train_rnn, y_train, X_test_rnn, y_test, "1 layer (20) CF")))
results_with_plots.append(("Cascade Forward, 2 приховані шари, по 10 нейронів",
                           build_and_evaluate_model_with_plots('cascade_forward', 2, 10, X_train_rnn, y_train, X_test_rnn, y_test, "2 layers (10) CF")))

# Elman (рекурентна мережа)
results_with_plots.append(("Elman, 1 прихований шар, 15 нейронів",
                           build_and_evaluate_model_with_plots('elman', 1, 15, X_train_rnn, y_train, X_test_rnn, y_test, "1 layer (15) Elman")))
results_with_plots.append(("Elman, 3 приховані шари, по 5 нейронів",
                           build_and_evaluate_model_with_plots('elman', 3, 5, X_train_rnn, y_train, X_test_rnn, y_test, "3 layers (5) Elman")))

# Виведення результатів
for config, error in results_with_plots:
    print(f"{config}: Середня відносна помилка = {error:.4f}")

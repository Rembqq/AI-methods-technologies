import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
X = np.random.rand(1000, 10)  # 1000 образцов с 10 признаками
y = X.sum(axis=1) + np.random.normal(scale=0.1, size=(1000,))  # Целевая переменная с небольшим шумом

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Добавление измерения для временных шагов (необходимо для RNN)
X_train_rnn = np.expand_dims(X_train, axis=1)
X_test_rnn = np.expand_dims(X_test, axis=1)

# Модифицированная функция для построения, обучения и сохранения истории обучения
def build_and_evaluate_model_with_plots(model_type, layers, neurons, X_train, y_train, X_test, y_test, config_name):
    model = Sequential()

    if model_type == 'elman':
        input_shape = (X_train.shape[1], X_train.shape[2])
        for i in range(layers):
            model.add(SimpleRNN(neurons, input_shape=input_shape if i == 0 else None,
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

    # Выходной слой
    model.add(Dense(1))

    # Компиляция модели
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Обучение с сохранением истории
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

    # Оценка
    predictions = model.predict(X_test)
    relative_mae = mean_absolute_error(y_test, predictions) / np.mean(np.abs(y_test))

    # Построение графиков
    plt.figure(figsize=(12, 5))

    # График 1: Зависимость предсказаний от реальных значений
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions, color='red', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
    plt.title(f"Network Type: {model_type.capitalize()}, {config_name}")
    plt.xlabel("Actual Values (R)")
    plt.ylabel("Predicted Values (P)")

    # График 2: Зависимость ошибки от эпох
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], color='green')
    plt.title(f"Loss Curve for: {model_type.capitalize()}, {config_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Показать и сохранить график
    plt.tight_layout()
    plt.show()

    return relative_mae

# Исследование и построение графиков
results_with_plots = []

# Feed Forward (прямое распространение)
results_with_plots.append(("Feed Forward, 1 hidden layer, 10 neurons",
                           build_and_evaluate_model_with_plots('feed_forward', 1, 10, X_train_rnn, y_train, X_test_rnn, y_test, "1 layer (10) FN")))
results_with_plots.append(("Feed Forward, 1 hidden layer, 20 neurons",
                           build_and_evaluate_model_with_plots('feed_forward', 1, 20, X_train_rnn, y_train, X_test_rnn, y_test, "1 layer (20) FN")))

# Cascade Forward (каскадное распространение)
results_with_plots.append(("Cascade Forward, 1 hidden layer, 20 neurons",
                           build_and_evaluate_model_with_plots('cascade_forward', 1, 20, X_train_rnn, y_train, X_test_rnn, y_test, "1 layer (20) CF")))
results_with_plots.append(("Cascade Forward, 2 hidden layers, 10 neurons each",
                           build_and_evaluate_model_with_plots('cascade_forward', 2, 10, X_train_rnn, y_train, X_test_rnn, y_test, "2 layers (10) CF")))

# Elman (рекуррентная сеть)
results_with_plots.append(("Elman, 1 hidden layer, 15 neurons",
                           build_and_evaluate_model_with_plots('elman', 1, 15, X_train_rnn, y_train, X_test_rnn, y_test, "1 layer (15) Elman")))
results_with_plots.append(("Elman, 3 hidden layers, 5 neurons each",
                           build_and_evaluate_model_with_plots('elman', 3, 5, X_train_rnn, y_train, X_test_rnn, y_test, "3 layers (5) Elman")))

# Вывод результатов
for config, error in results_with_plots:
    print(f"{config}: Average Relative Error = {error:.4f}")

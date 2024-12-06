import numpy as np

# Функція Хебба для навчання
class HebbianNetwork:
    def __init__(self, input_size, num_labels):
        self.weights = {label: np.zeros(input_size) for label in num_labels}  # Ваги для кожної букви

    def train(self, training_data):
        for label, vector in training_data.items():
            self.weights[label] += vector  # Оновлення ваг для кожної букви

    def normalize_weights(self):
        # Нормалізація ваг для кожної букви
        for label in self.weights:
            norm = np.linalg.norm(self.weights[label])
            if norm != 0:
                self.weights[label] = self.weights[label] / norm

    def predict_label(self, input_vector):
        # Обчислення косинусної подібності між вхідними даними та вагою кожної букви
        activations = {
            label: np.dot(vector, input_vector) / (np.linalg.norm(vector) * np.linalg.norm(input_vector))
            for label, vector in self.weights.items()
        }
        return max(activations, key=activations.get)  # Повернути букву з максимальною подібністю


# Вхідні дані у вигляді матриць (5x5)
letters = {
    "M": [
        [1, -1, 1, -1, 1],
        [1,  1, 1,  1, 1],
        [1, -1, 1, -1, 1],
        [1, -1, 1, -1, 1],
        [1, -1, 1, -1, 1]
    ],
    "Y": [
        [1, -1, 1, -1,  1],
        [-1, 1,  1,  1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1]
    ],
    "K": [
        [1, -1, 1, -1, -1],
        [1,  1, 1, -1, -1],
        [1,  1, -1, 1, -1],
        [1,  1, 1, -1, -1],
        [1, -1, 1, -1, -1]
    ],
    "T": [
        [1,  1,  1,  1,  1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1]
    ],
    "Noisy M": [
        [1, -1, 1, -1, 1],
        [1,  1, 1,  1, 1],
        [1, -1, 1, -1, -1],  # Один змінений піксель
        [1, -1, 1, -1, 1],
        [1, -1, 1, -1, 1]
    ],
    "Noisy Y": [
        [1, -1,  1, -1, 1],
        [-1, 1,  1,  1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, -1, -1, -1]  # Змінений останній ряд
    ],
    "Noisy T": [
        [1,  1,  1,  1,  1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, -1, -1, -1],  # Змінений піксель
        [-1, -1, 1, -1, -1]
    ],
    "Noisy K": [
        [1, -1, 1, -1, -1],
        [1,  1, 1, -1, -1],
        [1,  1, -1, -1, -1],  # Один змінений піксель
        [1,  1, 1, -1, -1],
        [1, -1, 1, -1, -1]
    ]
}

# Перетворення матриць у вектори
inputs = {key: np.array(value).flatten() for key, value in letters.items()}

# Навчання
labels = ["M", "Y", "K", "T"]
network = HebbianNetwork(input_size=25, num_labels=labels)

# Навчання мережі на навчальних даних
training_data = {label: inputs[label] for label in labels}
network.train(training_data)
network.normalize_weights()

# Тестування
results = {key: network.predict_label(value) for key, value in inputs.items()}

# Виведення результатів
for letter, result in results.items():
    print(f"{letter}: {result}")

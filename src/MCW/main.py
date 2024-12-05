import numpy as np
import matplotlib.pyplot as plt

# --- Нейронна мережа ---
class NeuralNetwork:
    def __init__(self, use_nguyen_widrow=False):
        # Ініціалізація шарів 2-3-4-6-8-8-1
        self.layers = [2, 3, 4, 6, 8, 8, 1]
        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)]
        self.biases = [np.random.randn(size) for size in self.layers[1:]]

        if use_nguyen_widrow:
            self.apply_nguyen_widrow()

    def apply_nguyen_widrow(self):
        # Реалізація модифікованого методу Нгуєна-Відроу
        for i in range(len(self.weights)):
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            beta = 0.7 * fan_out ** (1 / fan_in)

            # Нормалізація ваг
            norm = np.linalg.norm(self.weights[i])
            self.weights[i] *= beta / norm

            # Додаткова ініціалізація зміщень
            self.biases[i] = np.random.uniform(-beta, beta, size=self.biases[i].shape)

    def forward(self, inputs):
        # Прямий прохід через шари
        activation = inputs
        for weight, bias in zip(self.weights, self.biases):
            activation = np.dot(activation, weight) + bias
            activation = np.tanh(activation)  # Використовуємо тангенс як активацію
        return activation

    def mutate(self, mutation_rate=0.1):
        # Мутація ваг і зсувів
        for i in range(len(self.weights)):
            if np.random.rand() < mutation_rate:
                self.weights[i] += np.random.randn(*self.weights[i].shape) * 0.1
                self.biases[i] += np.random.randn(*self.biases[i].shape) * 0.1

# --- Генерація даних ---
def generate_data():
    # Генеруємо цільову функцію: z = sin(|x|) * sin(|y|)
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    x, y = np.meshgrid(x, y)
    z = np.sin(np.abs(x)) * np.sin(np.abs(y))
    return x, y, z

# --- Фітнес-функція ---
def evaluate_fitness(nn, x, y, z):
    # Оцінка придатності: середньоквадратична помилка
    z_pred = np.array([nn.forward(np.array([xi, yi])) for xi, yi in zip(x.flatten(), y.flatten())])
    return -np.mean((z.flatten() - z_pred.flatten())**2)

# --- Ініціалізація популяції ---
def initialize_population(size, use_nguyen_widrow):
    return [NeuralNetwork(use_nguyen_widrow) for _ in range(size)]

# --- Селекція ---
def select(population, fitness):
    # Відбір найкращих індивідів
    sorted_indices = np.argsort(fitness)[::-1]
    return [population[i] for i in sorted_indices[:len(population) // 2]]

# --- Кросовер ---
def crossover(selected_population):
    offspring = []
    for _ in range(len(selected_population) * 2):
        parent1, parent2 = np.random.choice(selected_population, 2)
        child = NeuralNetwork()
        # Поєднання ваг батьків
        child.weights = [(p1 + p2) / 2 for p1, p2 in zip(parent1.weights, parent2.weights)]
        child.biases = [(p1 + p2) / 2 for p1, p2 in zip(parent1.biases, parent2.biases)]
        offspring.append(child)
    return offspring

# --- Мутація ---
def mutate(population, mutation_rate=0.1):
    for ind in population:
        ind.mutate(mutation_rate)
    return population

# --- Основний цикл ---
def main():
    generations = 200  # Кількість поколінь
    population_size = 50  # Розмір популяції
    use_nguyen_widrow = True  # Використовувати метод Нгуєна-Відроу
    population = initialize_population(population_size, use_nguyen_widrow)  # Початкова популяція

    # Дані для тренування
    x, y, z = generate_data()

    # Список для зберігання похибки на кожному поколінні
    errors = []

    # Основний цикл навчання
    for generation in range(generations):
        # Оцінка придатності
        fitness = [evaluate_fitness(ind, x, y, z) for ind in population]

        # Вивід результатів
        best_nn = population[np.argmax(fitness)]  # Найкраща нейронна мережа
        z_pred = np.array([best_nn.forward(np.array([xi, yi])) for xi, yi in zip(x.flatten(), y.flatten())])
        z_pred_mean = np.mean(z_pred)

        print(f"Покоління {generation + 1}:")
        print(f"  Результат нейронної мережі (середній): {z_pred_mean:.4f}")
        print(f"  Справжнє значення: {np.mean(z):.4f}")

        # Зберігаємо похибку для побудови графіку
        error = np.mean((z.flatten() - z_pred.flatten())**2)
        errors.append(error)

        # Селекція, кросовер і мутація
        selected_population = select(population, fitness)
        offspring = crossover(selected_population)
        population = mutate(offspring)

    print("Навчання завершено.")

    # Побудова графіка похибки
    plt.plot(range(generations), errors)
    plt.xlabel('Покоління')
    plt.ylabel('Похибка (MSE)')
    plt.title('Зміна похибки протягом поколінь')
    plt.show()

if __name__ == "__main__":
    main()

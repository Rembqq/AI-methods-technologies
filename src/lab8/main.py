import random
import numpy as np
from deap import base, creator, tools

NUM_UNITS = 10
NUM_HUBS = 3
POPULATION = 300
GENERATIONS = 125
ELITES = 15
COMPETITORS = 2
MAX_UNITS_PER_HUB = 5
MAX_CHANNEL_SIZE = 300

# Генерація випадкової матриці трафіку
traffic_matrix = np.random.randint(1, 100, size=(NUM_UNITS, NUM_UNITS))
np.fill_diagonal(traffic_matrix, 0)

# Створення класів для фітнесу та індивідів
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Визначення "інструментарію" (toolbox)
toolbox = base.Toolbox()
toolbox.register("attribute", lambda: random.randint(1, NUM_HUBS))
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: generate_valid_individual())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    total_traffic = 0
    hub_loads = [0] * NUM_HUBS
    channel_traffic = [0] * NUM_HUBS
    penalties = 0

    if not check_ring_topology(individual):
        penalties += 1000

    for i in range(NUM_UNITS):
        hub_loads[individual[i] - 1] += 1
        for j in range(i + 1, NUM_UNITS):
            if individual[i] != individual[j]:
                traffic = traffic_matrix[i][j]
                total_traffic += traffic
                channel_traffic[individual[i] - 1] += traffic
                channel_traffic[individual[j] - 1] += traffic

    penalties += sum(max(0, load - MAX_UNITS_PER_HUB) for load in hub_loads)

    if total_traffic > MAX_CHANNEL_SIZE:
        penalties += total_traffic - MAX_CHANNEL_SIZE

    return total_traffic + penalties,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=COMPETITORS)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=NUM_HUBS, indpb=0.2)

def check_ring_topology(individual):
    hubs = set(individual)
    if len(hubs) != NUM_HUBS:
        return False

    for i in range(NUM_HUBS):
        current_hub = i + 1
        next_hub = (i + 1) % NUM_HUBS + 1
        if current_hub not in individual or next_hub not in individual:
            return False
    return True

def generate_valid_individual():
    while True:
        individual = [random.randint(1, NUM_HUBS) for _ in range(NUM_UNITS)]
        if check_ring_topology(individual):
            return individual

def main():
    random.seed(42)

    # Ініціалізація популяції
    population = toolbox.population(n=POPULATION)

    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)

    # Hall of Fame для збереження найкращих рішень
    hall_of_fame = tools.HallOfFame(1)

    best_generation = 0
    min_traffic = float("inf")

    for generation in range(GENERATIONS):
        # Сортування популяції та відбір еліти
        population.sort(key=lambda ind: ind.fitness.values[0])
        next_generation = population[:ELITES]

        while len(next_generation) < POPULATION:
            parents = toolbox.select(population, k=2)
            offspring1, offspring2 = toolbox.mate(parents[0], parents[1])
            del offspring1.fitness.values, offspring2.fitness.values
            next_generation.extend([offspring1, offspring2])

        # Мутація нащадків
        for individual in next_generation[ELITES:]:
            if random.random() < 0.5:
                toolbox.mutate(individual)
                del individual.fitness.values

        # Оновлення популяції
        population[:] = next_generation
        for individual in population:
            if not individual.fitness.valid:
                individual.fitness.values = toolbox.evaluate(individual)

        # Оновлення Hall of Fame
        hall_of_fame.update(population)

        # Найкращий індивід поточного покоління
        best_individual = tools.selBest(population, k=1)[0]
        print(f"Epoch {generation + 1}: Best Fitness = {best_individual.fitness.values[0]}")

        # Оновлення мінімального трафіку
        if best_individual.fitness.values[0] < min_traffic:
            min_traffic = best_individual.fitness.values[0]
            best_generation = generation + 1

    # Фінальний висновок
    best_solution = hall_of_fame[0]
    print("\n============================================")
    print("          Найкраще рішення з Hall of Fame")
    print("============================================")
    print(f"Рішення: {best_solution}")
    print(f"Мінімальний трафік: {evaluate(best_solution)[0]}")
    print(f"Досягнуто в поколінні: {best_generation}")
    print("============================================")

if __name__ == "__main__":
    main()

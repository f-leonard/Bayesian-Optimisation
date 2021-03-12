#GENETHIC ALGORITHM THAT ACCEPTS INPUT DATA ,OUTPUT DATA, THE NUMBER OF NEURONS AND THE NUMBER OF MODELS TO INITIALISE THE WEIGHTS AND RETURN THE OPTIMISED WEIGHTS



import statistics
import numpy as np
import random
import pandas

def GA(data_inputs, data_outputs,neurons,models):
    data_outputs = np.squeeze(data_outputs)

    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-2 * x))

    def mse(y_measured, y_predicted):
        mean_squared_error = statistics.mean((y_measured - y_predicted) ** 2)
        return mean_squared_error

    def matrix_to_vector(matrix_population_weights):
        population_vector = []
        for sol_index in range(matrix_population_weights.shape[0]):
            current_vector = []
            for layer_index in range(matrix_population_weights.shape[1]):
                vector_weights = np.reshape(matrix_population_weights[sol_index, layer_index],
                                            newshape=(matrix_population_weights[sol_index, layer_index].size))
                current_vector.extend(vector_weights)
            population_vector.append(current_vector)
        return np.array(population_vector)

    def vector_to_matrix(vector_population_weights, matrix_population_weights):
        matrix_weights = []
        for sol_index in range(matrix_population_weights.shape[0]):
            start = 0
            end = 0
            for layer_index in range(matrix_population_weights.shape[1]):
                end = end + matrix_population_weights[sol_index, layer_index].size
                current_vector = vector_population_weights[sol_index, start:end]
                matrix_layer_weights = np.reshape(current_vector,
                                                  newshape=(matrix_population_weights[sol_index, layer_index].shape))
                matrix_weights.append(matrix_layer_weights)
                start = end
        return np.reshape(matrix_weights, newshape=matrix_population_weights.shape)

    def mating(population, fitness, num_parents):
        parents = np.empty((num_parents, population.shape[1]))
        for parent_num in range(num_parents):
            min_fitness_index = np.where(fitness == np.min(fitness))
            min_fitness_index = min_fitness_index[0][0]
            parents[parent_num, :] = population[min_fitness_index, :]
            fitness[min_fitness_index] = +99999999
        return parents

    def crossoverc(parents, offspring_size):
        offspring = np.empty(offspring_size)
        crossover_point = np.uint32(offspring_size[1] / 2)
        for k in range(offspring_size[0]):
            parent1_index = k % parents.shape[0]
            parent2_index = (k + 1) % parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_index, crossover_point:]
        return offspring

    def mutationc(offspring_crossover, mutation_percent):
        num_mutations = np.uint32(mutation_percent * offspring_crossover.shape[1] / 100)
        mutation_indices = np.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))
        for index in range(offspring_crossover.shape[0]):
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[index, mutation_indices] = offspring_crossover[index, mutation_indices] + random_value
        return offspring_crossover

    def fitnessc(weights_matrix, data_inputs, data_outputs, activation="relu"):
        error = np.empty(shape=(weights_matrix.shape[0]))

        for sol_index in range(weights_matrix.shape[0]):
            current_solution_matrix = weights_matrix[sol_index, :]
            data_inputs = np.array(data_inputs)
            error[sol_index] = predict_outputs(current_solution_matrix, data_inputs, data_outputs
                                               )
            print("error is ", error[sol_index])
        return error

    def predict_outputs(weights_matrix, data_inputs, data_outputs):
        predictions = np.zeros(shape=(data_inputs.shape[0]))
        for sample_index in range(data_inputs.shape[0]):
            result = data_inputs[sample_index, :]
            a = 0
            b = 0
            for current_weights in weights_matrix:
                if a % 2 == 0:
                    result = np.matmul(result, current_weights)
                if a % 2 == 1:
                    result = np.add(result, current_weights)
                    value = result[0]
                    for i in range(len(value)):
                        r = value[i]
                        value[i] = relu(r)
                    result[0] = value
                a = a + 1
            predictions[sample_index] = result
        error = mse(predictions, data_outputs)
        return error


    mutation = 10
    number_parents_mating = models - 4
    number_generations =20
    mutation_percentage = mutation
    initial_population_weights = []

    for current_solution in np.arange(0,models):
        HL1_neurons = neurons
        input_HL1_weights = np.random.uniform(low=-1, high=1, size=(data_inputs.shape[1], HL1_neurons))

        bias1_weights = np.random.uniform(low=-1, high=1, size=(1, HL1_neurons))

        HL2_neurons =neurons
        HL1_HL2_weights = np.random.uniform(low=-1, high=1, size=(HL1_neurons, HL2_neurons))

        bias2_weights = np.random.uniform(low=-1, high=1, size=(1, HL2_neurons))

        output_neurons = 1
        HL2_output_weights = np.random.uniform(low=-1, high=1, size=(HL2_neurons, output_neurons))

        initial_population_weights.append(
            np.array([input_HL1_weights, bias1_weights, HL1_HL2_weights, bias2_weights, HL2_output_weights]))
    population_weights_matrix = np.array(initial_population_weights)
    population_weights_vector = matrix_to_vector(population_weights_matrix)
    best_outputs = []
    accuracies = np.empty(shape=(number_generations))

    for generation in range(number_generations):
        population_weights_matrix = vector_to_matrix(population_weights_vector, population_weights_matrix)
        fitness = fitnessc(population_weights_matrix, data_inputs, data_outputs, activation="relu")
        accuracies[generation] = fitness[0]
        parents = mating(population_weights_vector, fitness.copy(), number_parents_mating)
        crossover = crossoverc(parents, offspring_size=(
        population_weights_vector.shape[0] - parents.shape[0], population_weights_vector.shape[1]))
        mutation = mutationc(crossover, mutation_percent=mutation_percentage)
        population_weights_vector[0:parents.shape[0], :] = parents
        population_weights_vector[parents.shape[0]:, :] = mutation
    population_weights_matrix = vector_to_matrix(population_weights_vector, population_weights_matrix)
    best_weights = population_weights_matrix[0, :]
    predictions = predict_outputs(best_weights, data_inputs, data_outputs)
    fit = fitness[0]
    fit = int(fit)
    Training = pandas.DataFrame()

    Training.insert(0, 'GA Training', accuracies)
    Training.to_csv("GA Training Loss.csv", index=False)



    return best_weights

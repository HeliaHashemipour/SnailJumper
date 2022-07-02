from datetime import datetime
import copy
from player import Player
import numpy as np
import pandas as pd


class Evolution:
    def __init__(self):
        self.accuracy = []  # List of accuracy values (min, max, avg) for each generation
        self.game_mode = "Neuroevolution"
        self.mutate_num = 0  # Number of mutations is the number of players that will be mutated (μ)

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)
        # TODO (Additional: Learning curve)
        s_players = sorted(players, key=lambda player: player.fitness, reverse=True)  # Sort by fitness (highest to
        # lowest)
        max = s_players[0].fitness  # max(players)
        min = s_players[len(s_players) - 1].fitness  # min(players)
        average = sum([player.fitness for player in players]) / len(
            [player.fitness for player in players])  # average fitness of the population
        print([min, max, average])  # Print the min, max, and average fitness of the population
        self.accuracy.append((min, max, average))  # Add the min, max, and average fitness to the accuracy list (for
        # plotting)
        self.mutate_num = 0  # Reset the number of mutations
        # data = pd.DataFrame(np.array(evolution.accuracy), columns=["min", "max", "avg"])
        # csv_name = "generation_analysis_" + datetime.now().strftime("%d%H%M%S") + ".csv"
        # data.to_csv(csv_name)

        return self.roulette_wheel(players, num_players)  # Return the next population based on the selection method

    def cal_cumulative_probabilities(self, players):  # Calculate cumulative probabilities
        total_fitness = 0  # Total fitness of all players in the population
        for player in players:  # For each player in the population
            total_fitness += player.fitness  # Add the fitness to the total fitness (Sum of all fitnesses)
        probabilities = []  # List of probabilities (cumulative probabilities)
        for player in players:  # For each player in the population
            probabilities.append(player.fitness / total_fitness)  # Add the probability to the list of probabilities
        for i in range(1, len(players)):  # For each player after the first one (i.e. the first player has no
            # probability)
            probabilities[i] += probabilities[i - 1]  # Add the probability to the list (cumulative probabilities)
        return probabilities  # Return the list of probabilities for the next generation

    def roulette_wheel(self, players, parent_numbers):  # Roulette wheel selection
        probabilities = self.cal_cumulative_probabilities(players)  # Calculate cumulative probabilities
        new_population = []  # List of players that will be returned
        for temp in np.random.uniform(low=0, high=1, size=parent_numbers):  # parent_numbers is the number of parents
            # that we want to return
            for i, probability in enumerate(probabilities):  # Find the player that corresponds to the probability  (
                # i.e. the parent)
                if temp <= probability:  # If the probability is greater than the probability of the player
                    res = self.clone_player(players[i])  # Clone the player and add it to the new population
                    new_population.append(res)  # Add the player to the new population and break the loop
                    break  # Break the loop (We only need to find the first player that corresponds to the probability)
        return new_population  # Return the list of players

    def sus(self, players, num_players):  # SUS selection
        probabilities = self.cal_cumulative_probabilities(players)  # Calculate cumulative probabilities
        new_population = []  # List of players that will be returned
        move = (probabilities[len(probabilities) - 1] - np.random.uniform(0, 1 / num_players,
                                                                          1)) / num_players  # Step size for SUS
        # selection (1/num_players) is the step size for the roulette wheel selection (0.1)
        for i in range(num_players):  # num_players = num_players // q_size
            temp = (i + 1) * move  # Calculate the probability of the next player to be selected
            for i, probability in enumerate(probabilities):  # Find the player that corresponds to the probability
                if temp <= probability:  # If the probability is greater than the probability of the player
                    res = self.clone_player(players[i])  # Clone the player and add it to the new population
                    new_population.append(res)  # Add the player to the new population and break the loop
                    break  # Break the loop (We only need to find the first player that corresponds to the probability)
        return new_population  # Return the new population of players

    def q_tournament(self, players, num_players, q_size=2):  # Q-tournament selection
        next_population = []  # List of players that will be returned (μ + λ)
        for i in range(num_players):  # num_players = num_players // q_size
            temp_population = []  # list of players that will be used to calculate fitness values for the next
            # generation
            for j in range(q_size):  # q_size is the size of the tournament (2 by default)
                temp_population.append(players[np.random.randint(0, len(players))])  # random player from the population
            temp_population.sort(key=lambda x: x.fitness, reverse=True)  # Sort by fitness (highest to lowest)
            next_population.append(temp_population[0])  # Take the best player and add it to the next population
        return next_population  # Return the next population based on the previous population

    def mutation(self, child, threshold):  # Mutation function (Mutation rate = threshold)
        chance = np.random.uniform(0, 1, 1)  # Random chance to mutate (0-1)
        if chance < threshold:  # If the chance is less than the threshold (mutation probability)
            self.mutate_num += 1  # Increment the number of mutations by 1 (for the next generation)
            child.nn.weight_1 += np.random.randn(child.nn.weight_1.shape[0] *
                                                 child.nn.weight_1.shape[1]).reshape(child.nn.weight_1.shape[0],
                                                                                     child.nn.weight_1.shape[1])  #
            # Add random noise to the weights (gaussian distribution)
        chance = np.random.uniform(0, 1, 1)
        if chance < threshold:
            self.mutate_num += 1
            child.nn.weight_2 += np.random.randn(child.nn.weight_2.shape[0] *
                                                 child.nn.weight_2.shape[1]).reshape(child.nn.weight_2.shape[0],
                                                                                     child.nn.weight_2.shape[1])
        chance = np.random.uniform(0, 1, 1)
        if chance < threshold:
            self.mutate_num += 1
            child.nn.bias_1 += np.random.randn(child.nn.bias_1.shape[0] *
                                               child.nn.bias_1.shape[1]).reshape(child.nn.bias_1.shape[0],
                                                                                 child.nn.bias_1.shape[1])
        chance = np.random.uniform(0, 1, 1)
        if chance < threshold:
            self.mutate_num += 1
            child.nn.bias_2 += np.random.randn(child.nn.bias_2.shape[0] *
                                               child.nn.bias_2.shape[1]).reshape(child.nn.bias_2.shape[0],
                                                                                 child.nn.bias_2.shape[1])

    def crossover(self, child1, child2, parent1, parent2):  # Crossover (uniform crossover)
        section_1 = int(child1.shape[0] / 3)  # Section 1“ is the first third of the child
        section_2 = int(2 * child1.shape[0] / 3)  # Section 2“ is the second third of the child
        rnd = np.random.uniform(0, 1, 1)  # Random number between 0 and 1

        if rnd < 0.5:  # If the random number is less than 0.5
            child1[:section_1, :] = parent2[:section_1:,
                                    :]  # Copy the first third of the parent2 to the first third of the child1
            child1[section_1:section_2, :] = parent1[section_1:section_2,
                                             :]  # Copy the second third of the parent1 to the second third of the child1
            child1[section_2:, :] = parent2[section_2:,
                                    :]  # Copy the third third of the parent2 to the third third of the  child1
            child2[:section_1, :] = parent1[:section_1:,
                                    :]  # Copy the first third of the parent1 to the first third of the child2
            child2[section_1:section_2, :] = parent2[section_1:section_2,
                                             :]  # Copy the second third of the parent2 to the second third of the child2
            child2[section_2:, :] = parent1[section_2:,
                                    :]  # Copy the third third of the parent1 to the third third of the child2
        else:
            child1[:section_1, :] = parent1[:section_1:,
                                    :]  # Copy the first third of the parent1 to the first third of the child1
            child1[section_1:section_2, :] = parent2[section_1:section_2,
                                             :]  # Copy the second third of the parent2 to the second third of the child1
            child1[section_2:, :] = parent1[section_2:,
                                    :]  # Copy the third third of the parent1 to the third third of the child1
            child2[:section_1, :] = parent2[:section_1:,
                                    :]  # Copy the first third of the parent2 to the first third of the child2
            child2[section_1:section_2, :] = parent1[section_1:section_2,
                                             :]  # Copy the second third of the parent1 to the second third of the child2
            child2[section_2:, :] = parent2[section_2:,
                                    :]  # Copy the third third of the parent2 to the third third of the child2

    def operations(self, parent1, parent2):  # Operations
        threshold = 0.3  # Threshold for the mutation probability
        child1 = self.clone_player(parent1)  # Clone the first parent
        child2 = self.clone_player(parent2)  # Clone the second parent
        # weights
        self.crossover(child1.nn.weight_1, child2.nn.weight_1, parent1.nn.weight_1, parent2.nn.weight_1)  # Crossover
        self.crossover(child1.nn.weight_2, child2.nn.weight_2, parent1.nn.weight_2, parent2.nn.weight_2)  # Crossover
        # biases
        self.crossover(child1.nn.bias_1, child2.nn.bias_1, parent1.nn.bias_1, parent2.nn.bias_1)  # Crossover
        self.crossover(child1.nn.bias_2, child2.nn.bias_2, parent1.nn.bias_2, parent2.nn.bias_2) # Crossover
        # mutation
        self.mutation(child1, threshold)
        self.mutation(child2, threshold)
        return [child1, child2]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None  # If the previous generation is None
        if first_generation:  # If this is the first generation
            return [Player(self.game_mode) for _ in
                    range(num_players)]  # Return a list of num_players number of children
        else:  # If this is not the first generation
            prev_parents = prev_players.copy()  # Copy the previous generation
            prev_parents = self.sus(prev_parents, len(prev_parents))  # Get the survivors
            children = []  # Initialize the children list
            for i in range(0, len(prev_parents), 2):  # For each pair of parents
                children += self.operations(prev_parents[i], prev_parents[i + 1])  # Get the children
            return children  # Return the children

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

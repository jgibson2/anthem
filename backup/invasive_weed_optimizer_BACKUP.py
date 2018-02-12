"""
We will use our linear estimate as a basis for a genetic algorithm
Essentially, what we'll do is randomly permute the parameters and assign fitness scores based on how well the data fit
the model. Then, we'll select the best fitting organisms and "breed" them into a new generation with a random selection
of other organisms. Eventually the population should coalesce to the optimum values
"""
from thermodynamic_states import *
import numpy as np
import random
import math

'''
Gaussian chaos map
'''
class GaussianChaos:
    def __init__(self):
        self.k = random.random()

    def get(self):
        if self.k == 0:
            self.k = random.random()
            return 0.0
        self.k = (1.0 / self.k) - math.floor(1.0 / self.k)
        return self.k

'''
Contains values and states of a particular Organism
'''
class Weed:

    def __init__(self, weights):
        self.weights = weights
        self.avg_error = None
        self.avg_fitness = None
        self.total_error = None
        self.total_fitness = None

    '''
        Calculate fitness score for this organism based on a target value
            @param 
                concentrations_list: list of list of concentrations of TFs, RNAP in order of columns of mat
                mat: matix representing all possible binding sites
                target: list of target values
                active_states: vector corresponding to active states in the binding sites matrix
            @return 
                average fitness score
        '''

    def fitness(self, concentrations_list, mat, target_list, active_states=None, recalculate=False):
        if len(concentrations_list) == 0 or len(target_list) == 0:
            raise ValueError('No values to calculate fitness for!')
        if self.total_error is not None and not recalculate:
            return 1.0 / (self.total_error + 0.00001)
        e_tot = 0.0
        for concentrations, target in zip(concentrations_list, target_list):
            # calculate sum of active states
            a = active(concentrations, mat, self.weights, active_states=active_states)
            # calculate partition function value
            p = partition(concentrations, mat, self.weights)
            # calculate error
            e = abs(((a / p) / target) - 1)
            e_tot += e
            # return fitness
        self.total_error = e_tot
        self.total_fitness = 1.0 / (self.total_error + 0.00001)
        self.avg_error = e_tot / len(target_list)
        self.avg_fitness = 1.0 / (self.avg_error + 0.00001)
        return self.total_fitness

    '''
        Calculate average fitness score for this organism based on a target value
            @param 
                concentrations_list: list of list of concentrations of TFs, RNAP in order of columns of mat
                mat: matix representing all possible binding sites
                target: list of target values
                active_states: vector corresponding to active states in the binding sites matrix
                recalculate: Boolean to recalculate the fitness value
            @return 
                average fitness score
        '''

    def average_fitness(self, concentrations_list, mat, target_list, active_states=None, recalculate=False):
        if len(concentrations_list) == 0 or len(target_list) == 0:
            raise ValueError('No values to calculate fitness for!')
        if self.avg_error is not None and not recalculate:
            return 1.0 / (self.avg_error + 0.00001)
        e_tot = 0.0
        for concentrations, target in zip(concentrations_list, target_list):
            # calculate sum of active states
            a = active(concentrations, mat, self.weights, active_states=active_states)
            # calculate partition function value
            p = partition(concentrations, mat, self.weights)
            # calculate error
            e = abs(((a / p) / target) - 1)
            e_tot += e
            # return fitness
        self.total_error = e_tot
        self.total_fitness = 1.0 / (self.total_error + 0.00001)
        self.avg_error = e_tot / len(target_list)
        self.avg_fitness = 1.0 / (self.avg_error + 0.00001)
        return self.avg_fitness



'''
Represents a Generation of organisms that can interbreed
'''
class InvasiveWeedOptimizer:
    def __init__(self, mat, concentrations_list=[], target_list=[], active_states=None, pop_size=100, init_size=10,
                 initial_std_dev=10.0, final_std_dev=0.01, min_seeds = 1, max_seeds=10, nmi=3):
        self.weeds = []
        self.concentrations_list = concentrations_list
        self.target_list = target_list
        self.mat = mat
        self.active_states = active_states
        self.pop_size = pop_size
        self.init_size = init_size
        self.initial_std_dev = initial_std_dev
        self.final_std_dev = final_std_dev
        self.min_seeds = min_seeds
        self.max_seeds = max_seeds
        self.min_fitness = None
        self.max_fitness = None
        self.std_dev = initial_std_dev
        self.nmi = nmi  # nonlinear modulation index
        for i in range(init_size):
            # seed the initial population using a uniform distribution encompassing 99% of the variability between upper
            # and lower bounds
            weights = np.array([random.random() * 3.0 * initial_std_dev for i in range(mat.shape[0])])
            org = Weed(weights)
            self.weeds.append(org)
    '''
    Breed the population, removing the least fit organisms as needed to keep the population size at the maximum
    '''
    def refine(self, iterations=10):
        for i in range(iterations):
            # let the population reproduce
            new_weeds = []
            for w in self.weeds:
                new_weeds.extend(self.reproduce(w, self.std_dev))
            self.weeds.extend(new_weeds)

            # sort the array and remove the least fit organisms, if necessary
            self.weeds.sort(key= lambda w: w.fitness(self.concentrations_list, self.mat, self.target_list,
                            active_states=self.active_states), reverse=True)
            print('Fitness before cull: ', self.weeds[0].avg_fitness, '>', self.weeds[-1].avg_fitness)
            if len(self.weeds) > self.pop_size:
                self.weeds = self.weeds[:self.pop_size]
            print('Fitness after cull: ', self.weeds[0].avg_fitness, '>', self.weeds[-1].avg_fitness)

            # reset the min and max fitnesses
            self.min_fitness = None
            self.max_fitness = None
            # determine the new std dev
            self.std_dev = math.pow(iterations - i, self.nmi) / math.pow(iterations, self.nmi) * \
                           (self.initial_std_dev - self.final_std_dev) + self.final_std_dev

    def estimate(self):
        return self.weeds[0].weights

    def add(self, concentrations, prob):
        self.concentrations_list.append(concentrations)
        self.target_list.append(prob)

    '''
    Let the population of plants reproduce
        @param 
            weed: Weed to reproduce
            std_dev: current std dev
    '''
    def reproduce(self, weed, std_dev):
        offspring = []
        # if either value is none, recalculate the maximum and minimum fitnesses
        if self.min_fitness is None or self.max_fitness is None:
            fitness_list = [w.fitness(self.concentrations_list, self.mat, self.target_list,
                                      active_states=self.active_states) for w in self.weeds]
            self.min_fitness = min(fitness_list)
            self.max_fitness = max(fitness_list)
        # calculate the fitness of the current weed
        w_fit = weed.fitness(self.concentrations_list, self.mat, self.target_list,
                             active_states=self.active_states)
        # calculate the number of seeds this plant will produce
        # to do this, we map where the fitness value of the current plant is in the range of fitnesses to its
        # corresponding position in the range in the number of seeds
        num_seeds = round((w_fit - self.min_fitness) / (self.max_fitness - self.min_fitness) *
                          (self.max_seeds - self.min_seeds) + self.min_seeds)
        # now, we seed the area around the plant according to the given standard deviation
        for i in range(int(num_seeds)):
            weights = [np.random.normal(loc=w, scale=std_dev) for w in weed.weights]
            offspring.append(Weed(weights))
        return offspring

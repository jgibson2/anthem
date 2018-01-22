"""
We will use our linear estimate as a basis for a genetic algorithm
Essentially, what we'll do is randomly permute the parameters and assign fitness scores based on how well the data fit
the model. Then, we'll select the best fitting organisms and "breed" them into a new generation with a random selection
of other organisms. Eventually the population should coalesce to the optimum values
"""

from linear_estimator import active, partition
import numpy as np


'''
Contains values and states of a particular Organism
'''
class Organism:

    def __init__(self, weights):
        print(weights)
        self.weights = weights
        self.avg_fitness = None

    '''
    Calculate fitness score for this organism based on a target value
        @param 
            concentrations: list of concentrations of TFs, RNAP in order of columns of mat
            mat: matix representing all possible binding sites
            target: target value
            active_states: vector corresponding to active states in the binding sites matrix
        @return 
            fitness score
    '''
    def fitness(self, concentrations, mat, target, active_states=None):
        # calculate sum of active states
        a = active(concentrations, mat, self.weights, active_states=active_states)
        # calculate partition function value
        p = partition(concentrations, mat, self.weights)
        # calculate error
        e = abs(((a / p) / target) - 1)
        # return fitness
        return e

    '''
        Calculate average fitness score for this organism based on a target value
            @param 
                concentrations_list: list of list of concentrations of TFs, RNAP in order of columns of mat
                mat: matix representing all possible binding sites
                target: list of target values
                active_states: vector corresponding to active states in the binding sites matrix
            @return 
                average fitness score
        '''

    def average_fitness(self, concentrations_list, mat, target_list, active_states=None):
        if len(concentrations_list) == 0 or len(target_list) == 0:
            raise ValueError('No values to calculate fitness for!')
        if self.avg_fitness is not None:
            return self.avg_fitness
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
        self.avg_fitness = e_tot / len(target_list)
        return self.avg_fitness

    '''
    Mate one organism with another, averaging their weights and changes with random permutation
        @param 
            organism: Organism to mate with
    '''
    def mate(self, organism):
        # average weights and changes
        w = [float(self.weights[i] + x) / 2.0 for i,x in enumerate(organism.weights)]
        # introduce random variation (crossing over)
        w = [w[i] + np.random.normal(loc=w[i], scale=abs(self.weights[i]-organism.weights[i])/4.0) for i in range(len(w))]
        # return a new Organism
        return Organism(w)


'''
Represents a Generation of organisms that can interbreed
'''
class GeneticEstimator:
    def __init__(self, mat, concentrations_list=[], target_list=[], active_states=None, pop_size=100, init_size=10, lower_bound=-100, upper_bound=100):
        self.organisms = []
        self.concentrations_list = concentrations_list
        self.target_list = target_list
        self.mat = mat
        self.active_states = active_states
        self.pop_size = pop_size
        self.init_size = init_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        for i in range(init_size):
            # seed the initial population using a normal distribution encompassing 99% of the variability between upper
            # and lower bounds
            weights = np.random.normal(loc=lower_bound+((upper_bound-lower_bound)/2.0),
                                       scale=(upper_bound-lower_bound)/3.0, size=mat.shape[0])
            org = Organism(weights)
            self.organisms.append(org)

    '''
    Breed the population, removing the least fit organisms as needed to keep the population size at the maximum
    '''
    def refine(self, iterations=10):
        for i in range(iterations):
            # get the weights (average fitnesses) of the individuals in the population
            p = [org.average_fitness(self.concentrations_list, self.mat, self.target_list,
                                     active_states=self.active_states) for org in self.organisms]
            p_max = max(p)
            p = list(map(lambda x: (p_max - x) / p_max, p))
            p_sum = sum(p)
            p = list(map(lambda x: x / p_sum, p))
            # now have the population interbreed, choosing mates from the population according to their fitness scores
            # such that the most fit individuals have a better chance of breeding
            #print(sorted(p))
            new_orgs = []
            for org in self.organisms:
                new_orgs.append(org.mate(np.random.choice(self.organisms, p=p)))
            # add them to the original list and re-sort
            self.organisms.extend(new_orgs)
            self.organisms.sort(key=lambda org: org.average_fitness(self.concentrations_list, self.mat, self.target_list,
                                                                    active_states=self.active_states), reverse=True)
            if len(self.organisms) > self.pop_size:
                # cull the herd if there are too many individuals
                self.organisms = self.organisms[0:self.pop_size]

    def estimate(self):
        return self.organisms[0].weights

    def add(self, concentrations, prob):
        self.concentrations_list.append(concentrations)
        self.target_list.append(prob)

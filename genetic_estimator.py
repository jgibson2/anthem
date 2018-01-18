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

    def __init__(self, weights, changes):
        if not len(weights) == len(changes):
            raise ValueError('Length of weights array does not correspond to changes array!')
        self.weights = weights
        self.changes = changes

    '''
    Calculate fitness score between zero and one for this organism based on a target value
        @param 
            concentrations: list of concentrations of TFs, RNAP in order of columns of mat
            mat: matix representing all possible binding sites
            target: target value
            active_states: vector corresponding to active states in the binding sites matrix
        @return 
            fitness score from 0 to 1 (1 being highest fitness)
    '''
    def fitness(self, concentrations, mat, target, active_states=None):
        # calculate sum of active states
        a = active(concentrations, mat, self.weights, active_states=active_states)
        # calculate partition function value
        p = partition(concentrations, mat, self.weights)
        # calculate error
        e = abs(target - (float(a)/float(p))) / (float(a)/float(p))
        # return fitness
        return 1 - e

    '''
    Mate one organism with another, averaging their weights and changes with random permutation
        @param 
            organism: Organism to mate with
    '''
    def mate(self, organism):
        # average weights and changes
        w = list(map(lambda i,x: float(self.weights[i] + x) / 2.0) for y in enumerate(organism.weights))
        c = list(map(lambda i,x: float(self.changes[i] + x) / 2.0) for y in enumerate(organism.changes))
        # introduce random variation (crossing over)
        c = [c[i] + np.random.normal(loc=c[i], scale=abs(self.changes[i]-organism.changes[i])) for i in range(len(c))]
        w = [w[i] + c[i] for i in range(len(w))]
        # return a new Organism
        return Organism(w, c)


'''
Represents a Generation of organisms that can interbreed
'''

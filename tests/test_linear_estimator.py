import unittest
from linear_estimator import *
import numpy as np
import math
import random


class MyTestCase(unittest.TestCase):
    def test_estimator(self):
        # TFs
        tfs = ['A','B']
        # get matrix
        mat = build_binding_combination_matrix(tfs)
        # get estimator
        est = BoltzmannWeightsLinearEstimator(mat)
        # random weights
        weights = [random.randint(1,10) for i in range(mat.shape[0])]
        weights[0] = 1
        # get random equations
        for i in range(1000):
            # random concentrations
            concentrations = np.random.rand(len(tfs)+1)
            est.add(concentrations, active(concentrations, mat, weights)/partition(concentrations, mat, weights))
        #est.estimate()
        est.weights = [x + np.random.rand() for x in weights]
        for a,b in zip(weights, est.weights):
            print('Actual: ' + str(a), 'Estimate: ' + str(b), sep='\t')
        est.refine()
        print('----------------------------------')
        for a,b in zip(weights, est.weights):
            print('Actual: ' + str(a), 'Estimate: ' + str(b), sep='\t')


if __name__ == '__main__':
    unittest.main()

import unittest
from linear_estimator import *
from genetic_estimator import *
import random


class MyTestCase(unittest.TestCase):
    def test_estimator(self):
        # TFs
        tfs = ['A', 'B']
        # get matrix
        mat = build_binding_combination_matrix(tfs)
        # get estimator
        est = GeneticEstimator(mat, lower_bound=0.0, upper_bound=10.0)
        # random weights
        weights = [random.randint(1, 10) for i in range(mat.shape[0])]
        weights[0] = 1

        # get random equations
        for i in range(100):
            # random concentrations
            concentrations = np.random.rand(len(tfs) + 1)
            est.add(concentrations, active(concentrations, mat, weights) / partition(concentrations, mat, weights))
        print('Done adding equations!', flush=True)
        est.refine()
        print('----------------------------------')
        for a, b in zip(weights, est.estimate()):
            print('Actual: ' + str(a), 'Estimate: ' + str(b), sep='\t')


if __name__ == '__main__':
    unittest.main()

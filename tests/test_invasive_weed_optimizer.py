import unittest

from backup.invasive_weed_optimizer_BACKUP import *
from thermodynamic_states import *


class MyTestCase(unittest.TestCase):
    def test_estimator(self):
        # TFs
        tfs = ['A', 'B']
        # get matrix
        mat = build_binding_combination_matrix(tfs)
        # get estimator
        est = InvasiveWeedOptimizer(mat, initial_std_dev=3.0, final_std_dev=0.001)
        # random weights
        weights = [random.randint(1, 10) for i in range(mat.shape[0])]
        weights[0] = 1

        weed = Weed(weights)

        # get random equations
        for i in range(500):
            # random concentrations
            concentrations = np.random.rand(len(tfs) + 1)
            est.add(concentrations, active(concentrations, mat, weights) / partition(concentrations, mat, weights))

        weed.fitness(est.concentrations_list, est.mat, est.target_list)
        print('\nBaseline: ', weed.avg_fitness, 'Error: ', weed.avg_error)

        print('Done adding equations!', flush=True)
        est.refine(iterations=50)
        print('----------------------------------')
        for a, b in zip(weights, est.estimate()):
            # scale first weight to 1.0
            print('Actual: ' + str(a), 'Estimate: ' + str(b / est.estimate()[0]), sep='\t')
        print('----------------------------------')
        print('Average error:', est.weeds[0].avg_error)


if __name__ == '__main__':
    unittest.main()

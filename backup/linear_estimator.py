import scipy as sp
import math
from sklearn import linear_model
from thermodynamic_states import *


'''
Class to handle estimation of parameters
Note that this does NOT take into account the denominator and is an EXTREMELY POOR fit
We will only be using this in order to provide a base for a more sophisticated learning model since sklearn is fast and
easy
'''


class BoltzmannWeightsLinearEstimator:
    def __init__(self, mat, active_states=None):
        self.mat = mat
        self.active_states = active_states
        self.model = linear_model.LinearRegression()
        self.conc_list = list()
        self.prob_list = list()
        self.state_values_list = list()
        self.weights = []  # this is what we're after

    '''
    Compute the values and add them to a list
        @param 
            concentrations: list of concentrations of TFs, RNAP in order of columns of mat
            prob: probability transcript is being produced at any given moment
            active_states: vector corresponding to active states in the binding sites matrix
    '''

    def add(self, concentrations, prob):
        if self.active_states is None:
            self.active_states = build_active_states_vector(self.mat)
        # multiply the values in each row of the state matrix by the concentrations (to give the state matrix,
        # but with concentrations -- e.g. take the Hanamard product of each row and the concentrations and use those
        # as the rows of a new matrix
        s = np.apply_along_axis(lambda x: np.multiply(x, concentrations), axis=1, arr=self.mat)
        # now, we have to correct for the zeros in the original matrix, so we add the original matrix with every 0 and 1
        # flipped to the other value (investigate: faster this way or via np.vectorize?)
        s += np.ones(self.mat.shape) - self.mat
        # now take the row-wise product of the resulting product
        s = np.prod(s, axis=1)
        # we will only use the active states, so only consider those (TODO: decide on this)
        s = np.multiply(s, self.active_states)
        # now, we can fit it to the model with n features (investigate: take out zeros/inactive combinations?)
        self.state_values_list.append(s.reshape(1, -1))
        # in addition, we will keep track of the concentrations and probabilities used in the estimate for later use
        self.conc_list.append(concentrations)
        self.prob_list.append(prob)

    '''
    Fit the linear model
    '''
    def estimate(self):
        self.model.fit(np.array(self.state_values_list).reshape(len(self.prob_list), -1), self.prob_list)
        self.weights = self.model.coef_

    '''
    Refine the model guess using the Newton-Raphson method. See http://fourier.eng.hmc.edu/e161/lectures/ica/node13.html
    '''

    def refine(self, iterations=10):
        if not len(self.weights) > 0:
            raise ValueError('Model has not been trained!')
        for i in range(iterations):
            # first, we need to compute the Jacobian of the transcription probability function
            # to do this, we need to find the derivatives of each function (with varying concentrations)
            # these will all have the same form, since we are only interested in finding the weights
            # however, we will have to subtract the probability of transcription from the active/partition function
            # to ensure that the root is zero -- though this does not change the derivative, so we are okay
            jacobian = np.array([self.jacobian_row(c, self.mat, p, self.weights, active_states=self.active_states) for
                                 c, p in zip(self.conc_list, self.prob_list)])
            print(jacobian)
            # now, we need to generate the function value matrix. In this case, we divide the value of the active states
            # by the partition function, and then we subtract the probability value. Do this for each set of
            # concentrations and probabilities
            f = np.array([(active(c, self.mat, self.weights, active_states=self.active_states) /
                           partition(c, self.mat, self.weights)) - p for
                          c, p in zip(self.conc_list, self.prob_list)])
            # much of the time, the matrix will be over-specified, so we compute the pseudo-inverse using SVD methods
            jacobian_pinv = np.linalg.pinv(jacobian)
            # now, we simply perform the method:
            self.weights = self.weights - np.dot(jacobian_pinv, f)

    '''
    Returns an array representing the Jacobian row corresponding to that equation
        @param 
            concentrations: list of concentrations of TFs, RNAP in order of columns of mat
            mat: matix representing all possible binding sites
            prob: probability transcript is being produced at any given moment
            active_states: vector corresponding to active states in the binding sites matrix
    '''

    def jacobian_row(self, concentrations, mat, prob, weights_estimate, active_states=None):
        # each entry in the Jacobian will follow the same format:
        # (P[c] - A[c])/(P^2) where P is partition, A is active, and c are concentration expressions
        # of the respective states

        # multiply the values in each row of the state matrix by the concentrations (to give the state matrix,
        # but with concentrations -- e.g. take the Hanamard product of each row and the concentrations and use those
        # as the rows of a new matrix
        s = np.apply_along_axis(lambda x: np.multiply(x, concentrations), axis=1, arr=mat)
        # now, we have to correct for the zeros in the original matrix, so we add the original matrix with every 0 and 1
        # flipped to the other value (investigate: faster this way or via np.vectorize?)
        s += np.ones(mat.shape) - mat
        # now take the row-wise product of the resulting product
        s = np.prod(s, axis=1)
        # now, we have the concentration expressions for each state. We now iterate over the values of the resultant
        # matrix and compute the Jacobian row
        # note: unsure about first value. We could set it to one directly, or we could use it to check our estimation.
        # worth looking into!!!
        row = list()
        for value in s:
            P = partition(concentrations, mat, weights_estimate)
            A = active(concentrations, mat, weights_estimate, active_states=active_states)
            d = ((P * value) - (A * value)) / math.pow(P, 2)
            row.append(d)
        return row

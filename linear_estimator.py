"""
The purpose of this code is to fit the parameters of a thermodynamic model to data consisting
   of RNA-seq experiments on WT and TF-deletion strains and Calling Cards (CC) experiments on a
   set of TFs in the WT background. In the future, data could also include calling cards in TF deletion
   backgrounds and expression in multi-TF deletion strains.

   The inputs include the RNA-seq and calling cards data as well as a binary vector indicating which genes
   encode TF (regulatorsMask) and a binary vector indicating which genes are present in the strain and
   being modeled as TFs or targets in the model (genePresenceMask).

   The model includes the following parameters, all of which have to be estimated

   o alpha[gene], relating the probability of gene's promoter being occupied by RNAP to the mRNA concentration
     of the gene. One per target gene. Includes transcription initiation rate and RNA degradation rate.
   o rnapProm[prom], representing the affinity of RNAP for the core promoter of gene prom, related to basal
     transcription rate. One per target gene. Includes concentration of polymerase which is constant.
   o tfProm[tf,prom], representing the association constant of TF tf for the promoter of gene prom.
     Must be multiplied by tfActiveConc[tf, expt] to obtain the bound promoter to unbound promoter ratio.
     One per target-TF pair. Hopefully, most of these can be inferred from the CC hops.
   o tfConc[tf, expt], the effective concentration of active TF tf.
   o a[tf], relating the probability of a promoter being occupied by tf to the number of hops observed
     for TF tf, average across all experiments.
   o tfRNAP[tf], the effect of bound TF tf on the affinity of the RNAP for the promoter. For now, this
     is considered to be constant for all promoters but that should be reexamined. One per TF.
   o tfTF[tf,interactingTF] the effect of bound TF tf on the affinity of TF interactingTF for the promoter.
     For now, this is considered to be the same for all promoters but that should be reexamined.

    If we want to model multiple sites per promoter without modeling cooperation or competition
    then we simply add up the hop counts for each site -- nothing different. However, if we want
    to model multiple sites for TF j with interaction between the sites, we have to have another
    parameter for each pair of interacting sites:

  o w[j,k,i,x,y] the effect of TF j bound at site x on promoter i on the affinity of TF k for site y
    on promoter i. Initially, we will assume these are all zero. Otherwise, the number of parameters
    blows up pretty fast.

We define the partition function as the sum of the Boltzmann weights multiplied by the concentrations in the permutation
of available states
"""

import scipy as sp
import math
from sklearn import linear_model
import numpy as np
import itertools

'''
Build a matrix representing all possible binding combinations of TFs and RNAP
    @param:
        tfs: list of transcription factors
    @return 
        Numpy array corresponding to all possible binding states
'''


def build_binding_combination_matrix(tfs):
    # add RNAP to the list of things that can bind
    tfs_cpy = tfs + ['RNAP']
    # for each possible number of combinations, add it to a list. We'll use this in a minute to create the matrix
    all_combinations = list()
    for i in range(0, len(tfs_cpy) + 1):
        all_combinations += list(itertools.combinations(tfs_cpy, i))
    return np.array([[1 if c in comb else 0 for c in tfs_cpy] for comb in all_combinations])


'''
Build vector indicating which states are transcriptionally active
    @param 
        mat: matrix representing all possible binding states
    @return 
        vector corresponding to all transcriptionally active states
'''


def build_active_states_vector(mat):
    # we know that RNAP is the last column. Any state with RNAP is transcriptionally active Thus we can just take
    # the last column
    return mat[:, -1]


'''
Calculates the value of the partition function from the vector of Boltzmann weights
    @param 
        concentrations: list of concentrations of TFs, RNAP in order of columns of mat
        mat: matix representing all possible binding sites
        weights: Boltzmann weights of all states (first value is 1)
    @return 
        Returns integer value of partition function
'''


def partition(concentrations, mat, weights):
    # multiply the values in each row of the state matrix by the concentrations (to give the state matrix,
    # but with concentrations -- e.g. take the Hanamard product of each row and the concentrations and use those
    # as the rows of a new matrix
    s = np.apply_along_axis(lambda x: np.multiply(x, concentrations), axis=1, arr=mat)
    # now, we have to correct for the zeros in the original matrix, so we add the original matrix with every 0 and 1
    # flipped to the other value (investigate: faster this way or via np.vectorize?)
    s += np.ones(mat.shape) - mat
    # now take the row-wise product of the resulting product
    s = np.prod(s, axis=1)
    # calculate the dot product of the row products and the Boltzmann weights to get the
    return np.dot(s, weights)


'''
Calculates the value of the sum of all transcriptionally active states
    @param 
        concentrations: list of concentrations of TFs, RNAP in order of columns of mat
        mat: matix representing all possible binding sites
        weights: Boltzmann weights of all states (first value is 1)
        active_states: vector corresponding to active states in the binding sites matrix
    @return 
        Returns integer value of partition function
'''


def active(concentrations, mat, weights, active_states=None):
    if active_states is None:
        active_states = build_active_states_vector(mat)
    return partition(concentrations, mat, np.multiply(weights, active_states))


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

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
Build vector corresponding to which states have a bound transcription factor
    @param 
        mat: matrix
        idx: column index (0-based) of transcription factor
    @return 
        vector corresponding to states in which TF is bound
'''

def build_tf_states_vector(mat, idx):
    return mat[:, idx]


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
        mat: matrix representing all possible binding sites
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
Builds Boltzmann weights from individual components. Assume N transcription factors
    @param 
        mat: binding state matrix
        rnapProm: scalar value indicating the affinity of the RNAP to the promoter
        tfProm: 1 x N matrix representing the affinities of each TF to the promoter
        tfRNAP: 1 x N matrix representing the effect each TF has on the RNAP binding to the promoter
        tfTF: N x N matrix representing how each transcription factor interacts with each other TF
        alpha: proportionality constant
    @return 
        weights: vector of weights
'''

def build_weights_vector(mat, rnapProm, tfProm, tfRNAP, tfTF, alpha):
    weights = []
    # append the value for rnaProm to the vector of tfProms
    v = np.append(tfProm, rnapProm)
    # add the tfRNAP vector to the interacting factors matrix
    n = len(tfProm)
    m = np.append(tfTF, tfRNAP.reshape(1, n), axis=0)
    m = np.append(m, np.append(tfRNAP, [1.0]).reshape(n+1, 1), axis=1)
    for state_row in mat:
        # pull out the relevant promoter affinity values
        r = np.multiply(state_row, v)
        # do the ones trick and multiply along the rows
        r += np.ones(state_row.shape) - state_row
        r = np.prod(r, axis=1)
        # get the combinations of interacting TFs and RNAP, then multiply the interaction coefficients into r
        tf_indicies = [a for a,b in filter(lambda x: x[1] == 1, enumerate(state_row))]
        tf_combs = itertools.combinations(tf_indicies, 2)
        for i,j in tf_combs:
            r *= m[i,j]
        weights.append(r)
    # multiply by alpha
    weights *= alpha
    return np.array(weights)
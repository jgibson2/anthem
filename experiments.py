"""
Organization:
    - Experiments
        o Each Experiment has a set of concentrations, a set of empirical parameters, and a set of Genes
            - tfConc, assume rnapConc = 1
            - Parameters can either be determined from CC data (h) or differential expression data (m)
                o The active states change based on the type of experiment; in the CC experiments the active states are
                  those in which the tagged TF is bound; in the DE experiments the active states are the states in which
                  the RNAP is bound
        o Concentrations are shared across Genes
    - Genes
        o Each Gene has a set of weights indicating affinities of RNAP and each TF for the promoter
    - Transcription Factors
        o Each Transcription Factor has a coefficient indicating its effect on the binding of RNAP
"""

from thermodynamic_states import *

class Experiment:
    def __init__(self, type, gene_list, tfs, constants=None, concentrations=None, data=None):
        self.concentrations = [] if concentrations is None else concentrations
        self.gene_list = gene_list
        self.h_list = [] if data is None else data if type == 'h' else None
        self.m_list = [] if data is None else data if type == 'm' else None
        if (self.h_list is not None and not len(gene_list) == len(self.h_list)) \
                or (self.m_list is not None and not len(gene_list) == len(self.m_list)):
            raise ValueError('Length of gene list not equal to length of data!')


class TranscriptionFactors:
    def __init__(self, tfs, tfRNAP, tfTF):
        self.tfs = tfs
        self.tfRNAP = tfRNAP
        self.tfTF = tfTF


class Gene:
    def __init__(self, rnapProm, tfProm, tfs, alpha):
        self.rnapProm = rnapProm
        self.tfProm = tfProm
        self.tfs = tfs
        self.alpha = alpha
        self.mat = build_binding_combination_matrix(tfs)
        if not len(tfProm) == len(tfs):
            raise ValueError('Length of tfProm not equal to length of tfs!')

    def calculate_model_result(self, concentrations, active_states):
        w = build_weights_vector(self.mat, self.rnapProm, self.tfProm, self.tfs.tfRNAP, self.tfs.tfTF, self.alpha)
        a = active(concentrations, self.mat, w, active_states=active_states)
        p = partition(concentrations, self.mat, w)
        return float(a)/p

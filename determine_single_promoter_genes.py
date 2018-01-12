"""
In order to determine genes to run our models on, we need to determine genes that are acted on by one promoter.
Since we have the differential expression data from each knockout experiment, we can use that data to determine these
genes

To get started, we'll make a comparison of each gene's expression level between each experiment. Since the values
are FPKM values, they should be relatively normalized. Therefore we can determine if a gene has a significant decrease
to near-zero expression, and if so, we can link that gene to the deleted promoter

To determine significance, we can use the following metric: Without the lowest, near zero sample (find this
constant threshold), determine the sample standard deviation of the remaining samples. Use this to calculate the
probability that the near-zero event occurred by chance using a Student's T distribution.
"""

from scipy.stats import t
import numpy as np
from gene_entry import GeneEntry
import sys
import math

'''
Generate statistically significant single-promoter genes. This should corroborate the data from the callingCards dataset
Specifically, we use the callingCards data to determine genes that have significant interaction with the TF. Then, we 
use the differential expression data to find the expression levels of these genes. To figure out if this gene is 
regulated by only one promoter, we can enforce a lower bound on the FPKM value for the minimum experiment and we can 
verify that the TF is deleted in that experiment. We then put a confidence bound on this variation occurring by chance 
in comparison to the non-deleted values
    @param:
        entry_dict: dictionary of GeneEntry objects with experimental data
        tf_deletions_dict: dictionary of experiment identifiers to lists of deleted TFs
        pval: Max p value to be considered significant
        lower_bound: lower bound of expression level to be considered near zero
    @yield
        name of significant gene
'''
def generate_significant_genes(entry_dict, tf_deletions_dict, pval=0.05, lower_bound=100000, enforce_lower_bound=True):
    for gene_name, entry_obj in entry_dict.items():
        try:
            # check if we have data for this gene
            if len(entry_obj.DIFEXPRLVLS) == 0:
                raise ValueError('No differential expression data for gene {}!'.format(gene_name))
            # get the minimum expression value and check if it's lower than lower_bound
            # in addition, make sure that the deleted TF for this particular calling card dataset is also deleted
            # in the experiment
            min_experiment, min_value = min(entry_obj.DIFEXPRLVLS.items(), key=lambda x: x[1])
            if (min_value < lower_bound or not enforce_lower_bound) and \
                    entry_obj.DELTF in tf_deletions_dict[min_experiment]:
                # now, for all experiments that don't contain the deleted TFs, add them to a list
                non_deleted_values = []
                for experiment, del_tfs in tf_deletions_dict.items():
                    add = True
                    for tf in tf_deletions_dict[min_experiment]:
                        if tf in del_tfs:
                            add = False
                            break
                    if add:
                        non_deleted_values.append(entry_obj.DIFEXPRLVLS[experiment])
                # throw an exception if we don't have enough values (will be caught)
                if len(non_deleted_values) < 2:
                    raise ValueError('Not enough values to perform statistical tests!')
                # calculate the sample standard deviation
                non_deleted_sample_std_dev = np.std(non_deleted_values, dtype=np.float64) \
                    / math.sqrt(float(len(non_deleted_values)))
                # now, we will use the Student's T distribution to determine whether this is a significant result
                # we know that this will be left-tailed
                # see here for reference to call:
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
                # determine the probability that a result is less than or equal to the current value
                #p = t.cdf(min_value, len(non_deleted_values)-1, np.mean(non_deleted_values), non_deleted_sample_std_dev)
                p = t.cdf(min_value, 1, np.mean(non_deleted_values),
                          non_deleted_sample_std_dev)
                if p < pval:
                    # yield the gene name
                    #print(gene_name, entry_obj.DIFEXPRLVLS, p, non_deleted_values, min_value)
                    yield gene_name
        # handle thrown exceptions
        except ValueError as ve:
            #print(ve, sys.stderr)
            pass
        except KeyError as ke:
            raise KeyError('Experiment {} does not have corresponding TF deletion data!'.format(ke))
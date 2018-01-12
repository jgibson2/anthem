"""
This file contains utilities for reading, interpreting, and normalizing .gnashy files. These files have the following
tab-separated format:

"geneID", "geneName", "chromosome", "TSS", "TxStop", "PromExpect", "PromCount", "PromPVal", "TxExpect", "TxCount", "TxPVal"

The most relevant ones are PromExpect, the expected number of hops in the promoter of the gene based on a null model,
and PromCount, the actual number. We'll probably want to use "background subtracted normalized"
hops -- normalized to the number of hops in that experiment. Only hops above the expectation (i.e. background
subtracted) should be considered a signal of binding.

In addition, we will use normalized gene expression data to estimate the thermodynamic model parameters. Under directory
"expressionData", the file "gidsWithCommonNames" has the same number of lines as the FPKM files so presumably that's the
key. The first 6 lines are blank -- I'm not sure why. We'll have to figure out how to filter this gene set down later.
For now you'll probably focus on the genes with significant binding hits in the calling cards data, as you'll see in the
notebook referenced above there are only around 80 of them.
"""

import sys
from gene_entry import GeneEntry


'''
Extract relevant parameters from an input stream of a .gnashy file
    @param:
        file : File object corresponding to .gnashy file
        pval (optional kwarg)
    @return:
        dictionary of GeneEntry objects corresponding to each gene in the .gnashy file 
'''
def read_gnashy(file, pval=0.05, **kwargs):
    gene_dict = dict()
    for line in file:
        line = line.strip()
        try:
            entry = GeneEntry(**kwargs)
            is_sig = entry.parse_gnashy_entry(line.split('\t'), pval=pval)
            if entry.NAME in gene_dict:
                raise ValueError('Gene name already encountered in .gnashy file!')
            if is_sig:
                gene_dict[entry.NAME] = entry
        except ValueError as ve:
            print(ve, sys.stderr)
    return gene_dict


'''
Read the contents of a differential expression file and pair it with the appropriate GeneEntry
    @param:
        expr_file: File object corresponding to differential expression values for genes (one per line)
        name_file: File of gene names (one per line)
        entry_dict: dict of GeneEntries corresponding to gene names
        expr_name: name of experiment
'''
def read_diff_expr(expr_file, name_file, entry_dict, expr_name):
    for gene_name, expr_line in zip(name_file, expr_file):
        gene_name_list = gene_name.strip().split('\t')
        gene_name = gene_name_list[1] if len(gene_name_list) == 2 else None
        expr_line = expr_line.strip()
        expr_val = float(expr_line)
        if gene_name and gene_name in entry_dict:
            entry_dict[gene_name].DIFEXPRLVLS[expr_name] = expr_val
        else:
            #print("Gene name {0} not found in GeneEntry dictionary!".format(gene_name))
            pass



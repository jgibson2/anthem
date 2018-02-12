"""
Class defines a gene entry containing specific information for each gene. This includes
    - Gene ID
    - Gene Name
    - Chromosome
    - TSS
    - TxStop
    - PromCount
    - PromExpect
    - Differential expression information
        o List of experiment identifiers
        o List of FPKM values for each experiment
    - Deleted TF (if any)
    - Background Subtracted Hop Counts
"""


class GeneEntry:
    '''
    Constructor, takes keyword arguments
    '''
    def __init__(self, **kwargs):
        self.ID = kwargs.get('ID', None)
        self.NAME = kwargs.get('NAME', None)
        self.CHROM = kwargs.get('CHROM', None)
        self.START = kwargs.get('START', None)
        self.END = kwargs.get('END', None)
        self.PROMCNT = kwargs.get('PROMCNT', None)
        self.PROMEXP = kwargs.get('PROMEXP', None)
        self.DIFEXPRLVLS = kwargs.get('DIFEXPRLVLS', dict())
        self.DELTF = kwargs.get('DELTF', None)
        self.BSHC = kwargs.get('BSHC', None)

    '''
    Parses GNASHY file entry in the form of list of string values:
    "geneID", "geneName", "chromosome", "TSS", "TxStop", "PromExpect", "PromCount", "PromPVal", "TxExpect", "TxCount", "TxPVal"
    Returns true if pval parameter is specified and the PromPVal value is less than or equal to pval OR pval is not specified
    '''
    def parse_gnashy_entry(self, gnashy_value_list, **kwargs):
        if not len(gnashy_value_list) == 11:
            raise ValueError('Invalid GNASHY format!')
        self.ID = gnashy_value_list[0]
        self.NAME = gnashy_value_list[1]
        self.CHROM = gnashy_value_list[2]
        self.START = int(gnashy_value_list[3])
        self.END = int(gnashy_value_list[4])
        self.PROMEXP = float(gnashy_value_list[5])
        self.PROMCNT = float(gnashy_value_list[6])
        self.BSHC = self.PROMCNT - self.PROMEXP
        if 'pval' in kwargs and float(gnashy_value_list[7]) > kwargs['pval']:
            return False
        return True

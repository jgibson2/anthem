from determine_single_promoter_genes import generate_significant_genes
from data_handling import read_gnashy, read_diff_expr
from gene_entry import GeneEntry


'''
run the main program
'''
def run():
    # work with these diff expr files
    filepath_prefix = "C:\\Users\\john\\Data\\michaelProject\\modelingCallingCards\\expressionData\\exprFPKMFiles"
    files = ['BY4741-minusLys-1-03.04.16-04.14.16-35.expr',
             'MB0479-minusLys-1-03.04.16-04.14.16-32.expr',
             'YBR033W-minusLys-1-03.04.16-04.14.16-20.expr',
             'YDR034C-minusLys-1-03.04.16-04.14.16-26.expr',
             'YKL038W-minusLys-1-03.04.16-04.14.16-23.expr',
             'YM7728-minusLys-1-03.04.16-04.14.16-29.expr']
    gene_name_file = "C:\\Users\\john\\Data\\michaelProject\\modelingCallingCards\\expressionData\\gids.with_common_name"
    # work with these callingCard files
    cc_filepath_prefix = "C:\\Users\\john\\Data\\michaelProject\\modelingCallingCards\\callingCardsData"
    cc_files = ['NULL_model_results.Eds1-Tagin-Lys_filtered.gnashy',
                'NULL_model_results.Lys14-tagin-Lys_filtered.gnashy',
                'NULL_model_results.RGT1-Tagin-Lys_filtered.gnashy']
    cc_dict = {
        'NULL_model_results.Eds1-Tagin-Lys_filtered.gnashy': 'Eds1',
        'NULL_model_results.Lys14-tagin-Lys_filtered.gnashy': 'Lys14',
        'NULL_model_results.RGT1-Tagin-Lys_filtered.gnashy': 'Rgt1'
    }
    # build the experiment deletion dict
    expr_dict = {
        'BY4741': [],
        'MB0479': ['Eds1', 'Rgt1'],
        'YBR033W': ['Eds1'],
        'YDR034C': ['Lys14'],
        'YKL038W': ['Rgt1'],
        'YM7728': ['Eds1', 'Lys14']
    }

    # filepath_prefix = "C:\\Users\\john\\Data\\michaelProject\\modelingCallingCards\\expressionData\\exprFPKMFiles"
    # files = ['BY4741-plusLys-1-03.04.16-04.14.16-15.expr',
    #          'MB0479-plusLys-1-03.04.16-04.14.16-12.expr',
    #          'YBR033W-plusLys-1-03.04.16-04.14.16-1.expr',
    #          'YDR034C-plusLys-1-03.04.16-04.14.16-6.expr',
    #          'YKL038W-plusLys-1-03.04.16-04.14.16-3.expr',
    #          'YM7728-plusLys-1-03.04.16-04.14.16-9.expr']
    # gene_name_file = "C:\\Users\\john\\Data\\michaelProject\\modelingCallingCards\\expressionData\\gids.with_common_name"
    # # work with these callingCard files
    # cc_filepath_prefix = "C:\\Users\\john\\Data\\michaelProject\\modelingCallingCards\\callingCardsData"
    # cc_files = ['NULL_model_results.Eds1-Tagin+Lys_filtered.gnashy',
    #             'NULL_model_results.Lys14-tagin+Lys_filtered.gnashy',
    #             'NULL_model_results.RGT1-Tagin+Lys_filtered.gnashy']
    # cc_dict = {
    #     'NULL_model_results.Eds1-Tagin+Lys_filtered.gnashy': 'Eds1',
    #     'NULL_model_results.Lys14-tagin+Lys_filtered.gnashy': 'Lys14',
    #     'NULL_model_results.RGT1-Tagin+Lys_filtered.gnashy': 'Rgt1'
    # }
    # # build the experiment deletion dict
    # expr_dict = {
    #     'BY4741': [],
    #     'MB0479': ['Eds1', 'Rgt1'],
    #     'YBR033W': ['Eds1'],
    #     'YDR034C': ['Lys14'],
    #     'YKL038W': ['Rgt1'],
    #     'YM7728': ['Eds1', 'Lys14']
    # }

    for cc_file in cc_files:
        with open(cc_filepath_prefix + '\\' + cc_file) as ccf:
            entry_dict = read_gnashy(ccf, DELTF=cc_dict[cc_file])
            for file in files:
                expr_id = file.split('-')[0]
                with open(filepath_prefix + '\\' + file) as f, open(gene_name_file) as gnf:
                    read_diff_expr(f, gnf, entry_dict, expr_id)
            for gene in generate_significant_genes(entry_dict, expr_dict, enforce_lower_bound=True):
                print(cc_file, gene)


if __name__ == '__main__':
    run()
    print('Done')

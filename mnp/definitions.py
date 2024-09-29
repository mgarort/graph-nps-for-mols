# %%

from pathlib import Path
import pandas as pd
import pickle

# NOTE Using uppercase with all variables to indicate they are definitions


# Repo root directory (useful for building paths to datasets)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Other directories
EXPERIMENTS_DIR = ROOT_DIR / 'experiments'
DATA_DIR = ROOT_DIR / 'data'
PLOTS_DIR = ROOT_DIR / 'plots'
RESULTS_DIR = ROOT_DIR / 'results'
TABLES_DIR = ROOT_DIR / 'tables'


# Molecules in train and test sets (for datapoint split), and
# proteins in train and test sets (for function split)
df_dockstring_datapoint_split = pd.read_csv(DATA_DIR / 'dockstring' / 'datapoint_split.tsv', sep='\t', index_col=False)
DOCKSTRING_DATAPOINT_SPLIT_DICT = {}
DOCKSTRING_DATAPOINT_SPLIT_DICT['train'] = list(df_dockstring_datapoint_split.loc[df_dockstring_datapoint_split['set'] =='train', 'object'].values)
DOCKSTRING_DATAPOINT_SPLIT_DICT['test'] = list(df_dockstring_datapoint_split.loc[df_dockstring_datapoint_split['set'] =='test', 'object'].values)

# Functions for ftrain and ftest (53 targets for train, 5 targets for test)
# the convention for the keys of this dictionary is:
# - first, the type of score: e.g. "plain", "bo", "aug" 
# - second, an underscore "_"
# - third, the function split: e.g. "train", "adaptrain", "test", "adaptest"
DOCKSTRING_FUNCTION_SPLIT_DICT = {}
DOCKSTRING_FUNCTION_SPLIT_DICT['plain_train'] = ['PPARD', 'ABL1', 'ADAM17', 'ADRB1', 'ADRB2',
                                'AKT2', 'MAOB', 'CASP3', 'DHFR', 'PTK2', 'FGFR1',
                                'HMGCR', 'HSP90AA1', 'MAPKAPK2', 'MAP2K1', 'NOS1',
                                'PDE5A', 'PTPN1', 'ROCK1', 'AKT1', 'AR', 'CDK2',
                                'CSF1R', 'ESR1', 'NR3C1', 'IGF1R', 'JAK2', 'LCK',
                                'MET', 'MMP13', 'PTGS2', 'PPARA', 'PPARG', 'REN',
                                'ADORA2A', 'ACHE', 'BACE1', 'CA2', 'CYP2C9',
                                'CYP3A4', 'HSD11B1', 'DPP4', 'DRD2', 'DRD3',
                                'EGFR', 'F10', 'GBA', 'MAPK1', 'MAPK14', 'PLK1',
                                'SRC', 'THRB', 'KDR']
DOCKSTRING_FUNCTION_SPLIT_DICT['plain_test'] = ['ESR2', 'F2', 'KIT', 'PARP1', 'PGR']


num_aug_functions = 1000
all_aug_functions = [f'aug_{i}' for i in range(num_aug_functions)]

# Add BO functions to ftrain and ftest too:
# - ftrain: add all augmented functions that are not based on F2, JAK2 or LCK.
# - ftest: objective functions 'selective_JAK2' and 'promiscuous_PPAR'

targets_to_exclude = ['F2', 'JAK2', 'LCK']
aug_functions_to_exclude = []
operations_path = DATA_DIR / 'dockstring' / 'dockstring_aug_operations.pkl'
with open(operations_path, 'rb') as f:
    operations = pickle.load(f)
for function_k, function_v in operations.items():
    for term_k, term_v in function_v.items():
        if term_k in targets_to_exclude:
            aug_functions_to_exclude.append(function_k)
aug_functions_to_include = [aug_function for aug_function in all_aug_functions
                                         if aug_function not in aug_functions_to_exclude]
DOCKSTRING_FUNCTION_SPLIT_DICT['aug_train'] = aug_functions_to_include
DOCKSTRING_FUNCTION_SPLIT_DICT['aug_test'] = aug_functions_to_exclude + ['selective_JAK2',
                                                                         'promiscuous_PPAR',
                                                                         'druglike_F2']
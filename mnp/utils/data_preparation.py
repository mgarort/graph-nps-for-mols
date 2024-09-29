import numpy as np
import pandas as pd
from mnp.utils.chemistry import smiles_to_qed
from mnp.definitions import DATA_DIR, DOCKSTRING_DATAPOINT_SPLIT_DICT
from dockstring.benchmarks.original import (F2_score, promiscuous_PPAR_score,
                                            selective_JAK2_score)
import pickle
import rdkit.Chem as Chem
from rdkit.Chem import QED
from tqdm import tqdm
tqdm.pandas()



def modify_single_target(target_scores):
    modifications = {}
    # with prob 0.5, multiply by coefficient between -2 and 2
    if np.random.uniform() < 0.5:
        coeff = np.random.uniform(low=-2, high=2)
        target_scores = coeff * target_scores
        modifications['coeff'] = coeff
    else:
        modifications['coeff'] = None
    # with prob 0.25, either choose min or choose max between value and median
    if np.random.uniform() < 0.25:
        median = np.median(target_scores)
        array = np.zeros((target_scores.shape[0], 2))
        array[:,0] = target_scores
        array[:,1] = median
        if np.random.uniform() < 0.5:
            target_scores = np.min(array, axis=1)
            modifications['minmax'] = 'min'
        else:
            target_scores = np.max(array, axis=1)
            modifications['minmax'] = 'max'
    else:
        modifications['minmax'] = None
    return target_scores, modifications


def create_augmented_dockstring_dataset(dockstring: pd.DataFrame,
                                        qed_scores: pd.Series,
                                        num_augmented_functions: int) -> pd.DataFrame:
    """
    Augment the data in the following fashion
    - Pick between 1 and 4 scoring functions at random.
    - With a probability of 0.5, multiply by a coefficient between -2 and 2.
    - With a probability of 0.25:
        - With a prob of 0.5, choose min between the value and its median.
        - With a prob of 0.5, choose max between the value and its median.
    - With a probability of 0.5, add QED term.
    """

    dockstring_targets = dockstring.columns[2:60]
    num_dockstring_targets = len(dockstring_targets)
    num_molecules = dockstring.shape[0]
    qed_penalties = 10 * (1 - qed_scores.values)

    dockstring_augmented = np.zeros((num_molecules, num_augmented_functions))

    operations = {}
    all_augmented_names = []

    for i_augmented in range(num_augmented_functions):
        augmented_name = f'aug_{i_augmented}'
        all_augmented_names.append(augmented_name)
        operations[augmented_name] = {}
        # pick between 1 and 4 targets to be included
        np.random.seed(i_augmented)
        num_targets_chosen =  np.random.randint(low=1, high=5)
        idx_targets_chosen = np.random.permutation(num_dockstring_targets)[:num_targets_chosen]
        targets_chosen = dockstring_targets[idx_targets_chosen]
        # modify each of the targets
        modified_scores = np.zeros((num_molecules, num_targets_chosen))
        for i_target, each_target in enumerate(targets_chosen):
            these_unmodified_scores = dockstring[each_target].values
            these_modified_scores, modifications = modify_single_target(these_unmodified_scores)
            modified_scores[:,i_target] = these_modified_scores
            operations[augmented_name][each_target] = modifications
        # add together
        augmented_scores = modified_scores.sum(axis=1)
        # with prob 0.5, add qed term
        if np.random.uniform() < 0.5:
            augmented_scores += qed_penalties
            operations[augmented_name]['qed'] = True
        else:
            operations[augmented_name]['qed'] = False
        # save
        dockstring_augmented[:, i_augmented] = augmented_scores

    dockstring_augmented = pd.DataFrame(dockstring_augmented,
                                        index=dockstring.index,
                                        columns=all_augmented_names)
    dockstring_augmented = pd.concat((dockstring[['inchikey', 'smiles']],
                                      dockstring_augmented), axis=1)
    return dockstring_augmented, operations


def compute_selective_jak2(dockstring: pd.DataFrame, qed_scores: pd.Series) -> pd.Series:
    """
    Compute DOCKSTRING's selective JAK2 function for BO.
    """
    assert dockstring.shape[0] == len(qed_scores)
    selective_jak2 = []
    for jak2, lck, qed in zip(dockstring['JAK2'],
                              dockstring['LCK'],
                              qed_scores):
        selective_jak2.append(selective_JAK2_score(JAK2=jak2, LCK=lck, QED=qed))
    return pd.Series(selective_jak2, name='selective_JAK2')


def compute_promiscuous_ppar(dockstring: pd.DataFrame, qed_scores: pd.Series) -> pd.Series:
    """
    Compute DOCKSTRING's promiscuous PPAR function for BO.
    """
    assert dockstring.shape[0] == len(qed_scores)
    promiscuous_ppar = []
    for ppara, ppard, pparg, qed in zip(dockstring['PPARA'],
                                        dockstring['PPARD'],
                                        dockstring['PPARG'],
                                        qed_scores):
        promiscuous_ppar.append(promiscuous_PPAR_score(PPARA=ppara, PPARD=ppard,
                                                       PPARG=pparg, QED=qed))
    return pd.Series(promiscuous_ppar, name='promiscuous_PPAR')


def compute_druglike_f2(dockstring: pd.DataFrame, qed_scores: pd.Series) -> pd.Series:
    """
    Compute DOCKSTRING's F2 function for BO.
    """
    assert dockstring.shape[0] == len(qed_scores)
    druglike_f2 = []
    for f2, qed in zip(dockstring['F2'], qed_scores):
        druglike_f2.append(F2_score(F2=f2, QED=qed))
    return pd.Series(druglike_f2, name='druglike_f2')


def create_bo_objectives_dataset(dockstring: pd.DataFrame,
                                 qed_scores: pd.Series) -> pd.DataFrame:
    """
    Compute DOCKSTRING's BO objectives from the dockstring dataframe.
    """
    # compute objective functions
    selective_jak2 = compute_selective_jak2(dockstring, qed_scores)
    promiscuous_ppar = compute_promiscuous_ppar(dockstring, qed_scores)
    druglike_f2 = compute_druglike_f2(dockstring, qed_scores)
    # create dataframe
    bo_df = pd.DataFrame(index=dockstring.index,
                         columns=['selective_JAK2','promiscuous_PPAR','druglike_F2'])
    bo_df['selective_JAK2'] = selective_jak2
    bo_df['promiscuous_PPAR'] = promiscuous_ppar
    bo_df['druglike_F2'] = druglike_f2
    bo_df = pd.concat([dockstring[['inchikey','smiles']],
                       bo_df], axis=1)
    return bo_df


def append_qed_to_columns(col: str) -> str:
    if col not in ['inchikey', 'smiles', 'split']:
        col = col + '_qed'
    return col


def prepare_dockstring_scores(score_type: str, size: str) -> None:
    """
    Prepare dataframe with DOCKSTRING scores.
    - score_type: whether to use plain scores or QED-modified scores.
    - size: either "tiny" (for a set of 500 train + 500 test) "small" (for a set
            of 2500 train + 2500 test), "medium" (for a set of 30000+30000) or
            "large" (for the full DOCKSTRING dataset).
    """
    path = DATA_DIR / 'dockstring' / 'dockstring-dataset.tsv'
    dockstring = pd.read_csv(path, sep='\t')

    # Select train and test sets
    mask_train = dockstring['inchikey'].isin(DOCKSTRING_DATAPOINT_SPLIT_DICT['train'])
    mask_test = dockstring['inchikey'].isin(DOCKSTRING_DATAPOINT_SPLIT_DICT['test'])
    train = dockstring.loc[mask_train]
    test = dockstring.loc[mask_test]

    # Select the appropriate size
    if size == 'large':
        pass
    else:
        if isinstance(size, int):
            num_samples = size
        elif size == 'tiny':
            num_samples = 500
        elif size == 'small':
            num_samples = 2500
        elif size == 'medium':
            num_samples = 30000
        else:
            raise ValueError("'size' should be either 'tiny', 'small', 'smallmedium', 'medium' or 'large'")
        train = train.sample(num_samples, random_state=0)
        test = test.sample(num_samples, random_state=0)

    # Create column with the appropriate set label    
    train['split'] = 'train'
    test['split'] = 'test'
    dockstring = pd.concat([train,test]).reset_index(drop=True)

    # Plain, QED-modified or augmented scores
    if score_type == 'plain':
        df = dockstring
    elif score_type == 'qed':
        df = dockstring
        mols = df['smiles'].map(Chem.MolFromSmiles)
        qeds = mols.progress_map(QED.qed).values.reshape(-1,1)
        df.iloc[:,2:-1] = df.iloc[:,2:-1] + 10 * (1 - qeds)
        df.columns = df.columns.map(append_qed_to_columns)
    elif score_type == 'aug':
        qed_scores = dockstring['smiles'].progress_map(smiles_to_qed)
        aug, operations = create_augmented_dockstring_dataset(dockstring=dockstring,
                                                  qed_scores=qed_scores,
                                                  num_augmented_functions=1000)
        objectives = create_bo_objectives_dataset(dockstring=dockstring,
                                                  qed_scores=qed_scores)
        df = pd.concat([aug, objectives.iloc[:,2:]], axis=1)
        path = DATA_DIR / 'dockstring' / f'dockstring_aug_operations_{size}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(operations, f)   
    else:
        raise ValueError("'score_type' should be either 'plain', 'qed' or 'aug'.")
    path = DATA_DIR / 'dockstring' / f'dockstring_{size}_{score_type}.tsv'
    df.to_csv(path, sep='\t',index=False, float_format='%.2f')


def scale_column(column: pd.Series, scale: list) -> pd.Series:
    """
    column: column of dataframe to scale.
    scale: tuple of two floats representing the desired minimum and maximum of
        the scaled column.
    """
    scale_min = float(scale[0])
    scale_max = float(scale[1])
    scale_spread =  scale_max - scale_min
    column_min =column.min()
    column_max = column.max()
    column_spread = column_max - column_min
    assert scale_spread > 0
    if column_spread <= 0:
        print('hello')
        raise RuntimeError
    scaled_column = (column - column_min) / column_spread * scale_spread + scale_min
    return scaled_column


def clip_column(column: pd.Series,
                min_value: float, max_value: float) -> pd.Series:
    """
    column: column of dataframe to scale.
    min_value: elements under this value will be set to this value.
    max_value: elements over this value will be set to this value.
    """
    name = column.name
    df = pd.DataFrame(column)
    df['min'] = min_value
    df['max'] = max_value
    df[name] = df[[name, 'min']].max(axis=1)
    df[name] = df[[name, 'max']].min(axis=1)
    clipped_column = df[name]
    return clipped_column
from typing import List, Union, Tuple, Optional
import pandas as pd
from mnp.definitions import (DATA_DIR,
                             DOCKSTRING_DATAPOINT_SPLIT_DICT,
                             DOCKSTRING_FUNCTION_SPLIT_DICT)
from mnp.datasets.graphs import (MolecularGraphMetaDataset, MolecularGraphSimpleDataset)
from mnp.datasets.vectors import (FingerprintMetaDataset,
                                  FingerprintSimpleDataset,
                                #   MNISTDataset,
                                  SinusoidMetaDataset)
from mnp.utils.scripting import (choose_function_strings, choose_datapoint_strings,
                                 print_num_labeled_datapoints_message)
from mnp.utils.data_preparation import scale_column
from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np
from sklearn.decomposition import PCA
import random
import pickle
from tqdm import tqdm



# DOCKSTRING

def get_dockstring_dataframe(size: str, score_type: Union[str,List[str]],
                             function_split: Union[None,str,List[str]],
                             datapoint_split: Union[None,str,List[str]]) -> pd.DataFrame:
    """
    Load DOCKSTRING pandas dataframe. 
    
    - score_type : "plain" for plain scores or "qed" for scores
    modified with QED

        s_target(l) = s_target(l) + 10 * (1 - QED(l))

    - function_split : name(s) of protein targets to load, or None for all.
    - datapoint_split : (list with) "train" or "test".
    - size : either "small" for the split with 2500 train, 2500 test or "large" 
    for full DOCKSTRING dataset.
    """
    # Load all types of scores
    dockstring = []
    if isinstance(score_type, str):
        score_type = [score_type]
    for each_score_type in score_type:
        path = DATA_DIR / 'dockstring' / f'dockstring_{size}_{each_score_type}.tsv'
        df = pd.read_csv(path, sep='\t')
        dockstring.append(df)
    # Concatenate all types of scores into same dataframe
    dockstring = pd.concat(dockstring, axis=1)
    # Remove duplicate columns (inchikey, smiles and split)
    dockstring = dockstring.loc[:,~dockstring.columns.duplicated()].copy()
    # Select function and datapoint splits
    dockstring = select_dockstring_split(dockstring=dockstring,
                                         score_type=score_type,
                                         function_split=function_split,
                                         datapoint_split=datapoint_split)
    return dockstring


def select_dockstring_split(dockstring: pd.DataFrame,
                            score_type: str,
                            function_split: Union[None,str,List[str]],
                            datapoint_split: Union[None,str,List[str]]) -> pd.DataFrame:
    """
    Each of the arguments "function_split" and "datapoint_split" can be None
    (for no selection), a string "train" or "test" (for selection of a single set) 
    or a list of strings "train" or "set" (for selection of several sets).
    """
    # Select datapoints across the row dimension
    if datapoint_split is not None:
        datapoints_selected = []
        if isinstance(datapoint_split, str):
            datapoint_split = [datapoint_split]
        for set in datapoint_split:
            datapoints_selected += DOCKSTRING_DATAPOINT_SPLIT_DICT[set]
        mask_datapoint = dockstring['inchikey'].isin(datapoints_selected)
        dockstring = dockstring.loc[mask_datapoint]
    
    # Select functions across the column dimension
    # (make sure to retain "inchikey" and "smiles")
    if function_split is not None:
        functions_selected = ['inchikey', 'smiles']
        if isinstance(function_split, str):
            function_split = [function_split]
        for each_score_type in score_type:
            for set in function_split:
                functions_selected += DOCKSTRING_FUNCTION_SPLIT_DICT[f'{each_score_type}_{set}']
        mask_function = dockstring.columns.isin([] + functions_selected)
        dockstring = dockstring.loc[:,mask_function]

    return dockstring


def sample_dtrain(df_dtrain: pd.DataFrame, num_dtrain_to_sample: int,
                  random_seed: int) -> pd.DataFrame:
    np.random.seed(random_seed)
    random.seed(random_seed)
    df_dtrain = df_dtrain.sample(n=num_dtrain_to_sample,
                                 random_state=random_seed)
    return df_dtrain


def get_dockstring_metasplits(ftrain_dtrain_size: str, ftrain_dtest_size: str,
                              ftest_dtrain_size: str, ftest_dtest_size: str,
                              dtrain_score_type: str, dtest_score_type: str,
                              input_type: str,
                              num_contexts_range: Tuple[int],
                              num_targets_range: Tuple[int],
                              num_dtrain_to_sample: Optional[int]=None,
                              random_seed: int=0,
                              train_dset: Union[str,List[str]]='train',
                              test_dset: Union[str, List[str]]='test',
                              function_splits_type: str='default',
                              split_mode: str='disjoint_rigid') -> Tuple[Dataset]:
    """
    Function to get DOCKSTRING splits for meta-learning:
    - Function train, datapoint train (ftrain, dtrain)
    - Function train, datapoint test (ftrain, dtest)
    - Function test, datapoint train (ftest, dtrain)
    - Function test, datapoint test (ftest, dtest)
    """

    # select dataset class based on input type
    if input_type == 'graphs':
        DatasetClass = MolecularGraphMetaDataset
        counts = None
    elif input_type == 'fingerprints':
        DatasetClass = FingerprintMetaDataset
        counts = False
    elif input_type == 'count_fingerprints':
        DatasetClass = FingerprintMetaDataset
        counts = True
    else:
        raise ValueError('"input_type" should be either "graphs" or "fingerprints".')

    # select strings to choose the adaptation or default function splits
    # (e.g. 'train', 'test', 'adaptation_train', 'adaptation_test', etc)
    function_train_str, function_test_str = choose_function_strings(function_splits_type=function_splits_type)

    # Load dataframes
    df_ftrain_dtrain = get_dockstring_dataframe(size=ftrain_dtrain_size, 
                                                score_type=dtrain_score_type,
                                                function_split=function_train_str,
                                                datapoint_split=train_dset)
    df_ftrain_dtest = get_dockstring_dataframe(size=ftrain_dtest_size, 
                                            score_type=dtest_score_type,
                                            function_split=function_train_str,
                                            datapoint_split=test_dset)
    df_ftest_dtrain = get_dockstring_dataframe(size=ftest_dtrain_size,
                                            score_type=dtrain_score_type,
                                            function_split=function_test_str,
                                            datapoint_split=train_dset)
    df_ftest_dtest = get_dockstring_dataframe(size=ftest_dtest_size,
                                            score_type=dtest_score_type,
                                            function_split=function_test_str,
                                            datapoint_split=test_dset)

    # select smaller sample at random (for few-shot learning experiments)
    if num_dtrain_to_sample is not None:
        df_ftrain_dtrain = sample_dtrain(df_dtrain=df_ftrain_dtrain,
                                         num_dtrain_to_sample=num_dtrain_to_sample,
                                         random_seed=random_seed)
        df_ftest_dtrain = sample_dtrain(df_dtrain=df_ftest_dtrain,
                                        num_dtrain_to_sample=num_dtrain_to_sample,
                                        random_seed=random_seed)

    # create Pytorch datasets
    ds_ftrain_dtrain = DatasetClass(smiles=df_ftrain_dtrain['smiles'].values,
                                    functions=df_ftrain_dtrain.iloc[:,2:], 
                                    num_contexts_range=num_contexts_range, 
                                    num_targets_range=num_targets_range,
                                    split_mode=split_mode,
                                    counts=counts)
    ds_ftrain_dtest = DatasetClass(smiles=df_ftrain_dtest['smiles'].values,
                                    functions=df_ftrain_dtest.iloc[:,2:], 
                                    num_contexts_range=num_contexts_range, 
                                    num_targets_range=num_targets_range,
                                    split_mode=split_mode,
                                    counts=counts)
    ds_ftest_dtrain = DatasetClass(smiles=df_ftest_dtrain['smiles'].values,
                                    functions=df_ftest_dtrain.iloc[:,2:],
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range,
                                    split_mode=split_mode,
                                    counts=counts)
    ds_ftest_dtest = DatasetClass(smiles=df_ftest_dtest['smiles'].values,
                                    functions=df_ftest_dtest.iloc[:,2:],
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range,
                                    split_mode=split_mode,
                                    counts=counts)

    return ds_ftrain_dtrain, ds_ftrain_dtest, ds_ftest_dtrain, ds_ftest_dtest



def get_dockstring_simplesplits(function_names: Union[str, List[str]],
                              dtrain_size: str, dtest_size: str,
                              dtrain_score_type: str, dtest_score_type: str,
                              input_type: str,
                              num_dtrain_to_sample: Optional[int]=None,
                              random_seed: int=0) -> Tuple[Dataset]:
    """
    Function to get DOCKSTRING splits for for a single function:

    dtrain, dtest
    """
    # select dataset class based on input type
    if input_type == 'graphs':
        DatasetClass = MolecularGraphSimpleDataset
    elif input_type == 'fingerprints':
        DatasetClass = FingerprintSimpleDataset
    else:
        raise ValueError('"input_type" should be either "graphs" or "fingerprints".')

    # load dataframes
    df_dtrain = get_dockstring_dataframe(size=dtrain_size, 
                                         score_type=dtrain_score_type,
                                         function_split=['train','test'],
                                         datapoint_split='train')
    df_dtest = get_dockstring_dataframe(size=dtest_size, 
                                        score_type=dtest_score_type,
                                        function_split=['train','test'],
                                        datapoint_split='test')

    # select smaller sample at random (for few-shot learning experiments)
    if num_dtrain_to_sample is not None:
        df_dtrain = sample_dtrain(df_dtrain=df_dtrain,
                                  num_dtrain_to_sample=num_dtrain_to_sample,
                                  random_seed=random_seed)

    # create Pytorch datasets
    if isinstance(function_names, str):
        function_names = [function_names]
    ds_dtrain = DatasetClass(smiles=df_dtrain['smiles'],
                                functions=df_dtrain[function_names])
    ds_dtest = DatasetClass(smiles=df_dtest['smiles'],
                                functions=df_dtest[function_names])

    return ds_dtrain, ds_dtest



# BO

def get_bo_dataframe(molecular_function: str) -> pd.DataFrame:
    """
    Load pandas dataframe with BO objectives
    """
    # load
    path = DATA_DIR / 'molopt' / 'bo_objectives.tsv'
    df = pd.read_csv(path, sep='\t')[['smiles', molecular_function]]
    # mask null values
    mask = ~df[molecular_function].isnull()
    df = df.loc[mask]
    return df


def get_bo_dataset(input_type: str,
                   molecular_function: str) -> Tuple[Dataset]:
    # select dataset class based on input type
    if input_type == 'graphs':
        DatasetClass = MolecularGraphMetaDataset
    elif input_type == 'fingerprints':
        DatasetClass = FingerprintMetaDataset
    else:
        raise ValueError('"input_type" should be either "graphs" or "fingerprints".')

    # load dataframe
    df = get_bo_dataframe(molecular_function=molecular_function)

    # create Pytorch dataset
    # TODO we use dummy number of contexts and targets ranges for now
    ds = DatasetClass(smiles=df['smiles'].values,
                      functions=df[[molecular_function]], 
                      num_contexts_range=(25, 150), 
                      num_targets_range=(25, 150))
    return ds




# MNIST

# def get_mnist_metasplits(num_contexts_range: Tuple[int],
#                          num_targets_range: Tuple[int],
#                          overlap_contexts_and_targets: bool=False,
#                          dim_positional_encoding: int=400,
#                          adaptation_splits: bool=False) -> Tuple[Dataset]:
#     """
#     When regression, we can use:
#     - all digits from MNIST train as ftrain.
#     - all digits from MNIST test as ftest.
    
#     When adaptation, we can use:
#     - 0,1,2,3,4,5,6,7,9 from MNIST train as ftrain.
#     - 8 from MNIST test as ftest.

#     Unlike with molecules, here we don't differentiate between dtrain and dtest
#     because it makes sense to train on all the pixels, not just on some of the
#     pixels.
#     """
#     if adaptation_splits:
#         function_train_str = 'adaptation_train'
#         function_test_str = 'adaptation_test'
#     else:
#         function_train_str = 'train'
#         function_test_str = 'test'
#     ds_ftrain = MNISTDataset(function_split=function_train_str,
#                  num_contexts_range=num_contexts_range, 
#                  num_targets_range=num_targets_range,
#                  overlap_contexts_and_targets=overlap_contexts_and_targets,
#                  dim_positional_encoding=dim_positional_encoding)
#     ds_ftest = MNISTDataset(function_split=function_test_str,
#                  num_contexts_range=num_contexts_range, 
#                  num_targets_range=num_targets_range,
#                  overlap_contexts_and_targets=overlap_contexts_and_targets,
#                  dim_positional_encoding=dim_positional_encoding)
#     return ds_ftrain, ds_ftest




######################
# Sinusoid functions #
######################


def compute_sine_curve(x: np.ndarray, amplitude: float, frequency: float,
                       horizontal_shift: float) -> np.ndarray:
    return amplitude * np.sin(frequency * (x - horizontal_shift))


def create_sinusoids_dataset(num_functions: int, num_datapoints: int,
                             random_seed_offset: int,
                             amplitude_range: Tuple[float,float]=(0.1, 5.0),
                             frequency_catalog: List[float]=[1.],
                             horizontal_shift_range: Tuple[float,float]=(0., np.pi),
                             x_range: Tuple[float,float]=(-5.0, 5.0)) -> dict:

        x = np.zeros((num_functions, num_datapoints))
        y = np.zeros((num_functions, num_datapoints))
        amplitudes = np.zeros(num_functions)
        frequencies = np.zeros(num_functions)
        horizontal_shifts = np.zeros(num_functions)

        for i in range(num_functions):
            rng = np.random.default_rng(random_seed_offset + i)
            # choose amplitude
            amplitude = rng.uniform(amplitude_range[0], amplitude_range[1])
            amplitudes[i] = amplitude
            # choose frequency
            num_frequencies = len(frequency_catalog)
            chosen_index = rng.permutation(num_frequencies)[0]
            frequency = frequency_catalog[chosen_index]
            frequencies[i] = frequency
            # choose horizontal shift
            horizontal_shift = rng.uniform(horizontal_shift_range[0],
                                      horizontal_shift_range[1])
            horizontal_shifts[i] = horizontal_shift
            # compute sinusoid
            x[i] = rng.uniform(x_range[0], x_range[1], num_datapoints)
            y[i] = compute_sine_curve(x[i], amplitude=amplitude,
                                      frequency=frequency,
                                      horizontal_shift=horizontal_shift)

        return {'x': x, 'y': y,
                'amplitudes': amplitudes,
                'frequencies': frequencies,
                'horizontal_shifts': horizontal_shifts}


def create_sumsinusoids_dataset(num_functions: int, num_datapoints: int,
                    random_seed_offset: int,
                    amplitude_range: Tuple[float,float]=(0.5, 10.0),
                    frequency_range: Tuple[float, float]=(0.5, 2.5),
                    horizontal_shift_range: Tuple[float,float]=(0., np.pi),
                    x_range: Tuple[float,float]=(-5.0, 5.0)) -> dict:
        """
        Similar to sinusoids dataset, but the output y is the sum of two
        sinusoid functions, rather than a single sinusoid function.
        """

        x = np.zeros((num_functions, num_datapoints))
        y = np.zeros((num_functions, num_datapoints))
        y_1 = np.zeros((num_functions, num_datapoints))
        y_2 = np.zeros((num_functions, num_datapoints))
        amplitudes = np.zeros((num_functions, 2))
        frequencies = np.zeros((num_functions, 2))
        horizontal_shifts = np.zeros((num_functions, 2))

        for i in range(num_functions):
            rng = np.random.default_rng(random_seed_offset + i)
            amplitude = rng.uniform(amplitude_range[0], amplitude_range[1], 2)
            amplitudes[i] = amplitude
            frequency = rng.uniform(frequency_range[0], frequency_range[1], 2)
            frequencies[i] = frequency
            horizontal_shift = rng.uniform(horizontal_shift_range[0],
                                      horizontal_shift_range[1], 2)
            horizontal_shifts[i] = horizontal_shift
            x[i] = rng.uniform(x_range[0], x_range[1], num_datapoints)
            y_1[i] = compute_sine_curve(x=x[i], amplitude=amplitude[0],
                                     frequency=frequency[0],
                                     horizontal_shift=horizontal_shift[0])
            y_2[i] = compute_sine_curve(x=x[i], amplitude=amplitude[1],
                                     frequency=frequency[1],
                                     horizontal_shift=horizontal_shift[1])
            y[i] = y_1[i] + y_2[i]

        return {'x': x, 'y': y, 'y_1': y_1, 'y_2': y_2,
                'amplitudes': amplitudes,
                'frequencies': frequencies,
                'horizontal_shifts': horizontal_shifts}



def get_sinusoid_metasplits(num_contexts_range: Tuple[int],
                            num_targets_range: Tuple[int],
                            num_train_functions: int=int(1e4),
                            num_test_functions: int=int(1e4),
                            ftrain_dtrain_size: int=1000,
                            ftrain_dtest_size: int=1000,
                            ftest_dtrain_size: int=1000,
                            ftest_dtest_size: int=1000,
                            amplitude_range: Tuple[float,float]=(0.1, 5.0),
                            frequency_catalog: List[float]=[1.],
                            horizontal_shift_range: Tuple[float,float]=(0., np.pi),
                            sum: bool=False) -> Tuple[Dataset, Dataset]:
    # create ftrain datasets
    num_train_datapoints = ftrain_dtrain_size + ftrain_dtest_size
    if not sum:
        ftrain_seed_offset = 0
        ftrain = create_sinusoids_dataset(num_functions=num_train_functions,
                                num_datapoints=num_train_datapoints,
                                random_seed_offset=ftrain_seed_offset,
                                amplitude_range=amplitude_range,
                                frequency_catalog=frequency_catalog,
                                horizontal_shift_range=horizontal_shift_range)
    else:
        ftrain_seed_offset = 2 * int(1e6)
        ftrain = create_sumsinusoids_dataset(num_functions=num_train_functions,
                                num_datapoints=num_train_datapoints,
                                random_seed_offset=ftrain_seed_offset,
                                amplitude_range=amplitude_range,
                                frequency_catalog=frequency_catalog,
                                horizontal_shift_range=horizontal_shift_range)     

    ftrain_dtrain = {'x': ftrain['x'][:,:ftrain_dtrain_size],
                     'y': ftrain['y'][:,:ftrain_dtrain_size],
                     'amplitudes': ftrain['amplitudes'],
                     'frequencies': ftrain['frequencies'],
                     'horizontal_shifts': ftrain['horizontal_shifts']}

    ftrain_dtest = {'x': ftrain['x'][:,ftrain_dtrain_size:],
                     'y': ftrain['y'][:,ftrain_dtrain_size:],
                     'amplitudes': ftrain['amplitudes'],
                     'frequencies': ftrain['frequencies'],
                     'horizontal_shifts': ftrain['horizontal_shifts']}

    ds_ftrain_dtrain = SinusoidMetaDataset(x=ftrain_dtrain['x'],
                                    y=ftrain_dtrain['y'],
                                    amplitudes=ftrain_dtrain['amplitudes'],
                                    frequencies=ftrain_dtrain['frequencies'],
                                    horizontal_shifts=ftrain_dtrain['horizontal_shifts'],
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range)

    ds_ftrain_dtest = SinusoidMetaDataset(x=ftrain_dtest['x'],
                                    y=ftrain_dtest['y'],
                                    amplitudes=ftrain_dtest['amplitudes'],
                                    frequencies=ftrain_dtest['frequencies'],
                                    horizontal_shifts=ftrain_dtest['horizontal_shifts'],
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range)
    # create ftest datasets
    num_test_datapoints = ftest_dtrain_size + ftest_dtest_size
    if not sum:
        ftest_seed_offset = int(1e6)
        ftest = create_sinusoids_dataset(num_functions=num_test_functions,
                                num_datapoints=num_test_datapoints,
                                random_seed_offset=ftest_seed_offset,
                                amplitude_range=amplitude_range,
                                frequency_catalog=frequency_catalog,
                                horizontal_shift_range=horizontal_shift_range)
    else:
        ftest_seed_offset = 3 * int(1e6)
        ftest = create_sumsinusoids_dataset(num_functions=num_test_functions,
                                num_datapoints=num_test_datapoints,
                                random_seed_offset=ftest_seed_offset,
                                amplitude_range=amplitude_range,
                                frequency_catalog=frequency_catalog,
                                horizontal_shift_range=horizontal_shift_range)  

    ftest_dtrain = {'x': ftest['x'][:,:ftest_dtrain_size],
                     'y': ftest['y'][:,:ftest_dtrain_size],
                     'amplitudes': ftest['amplitudes'],
                     'frequencies': ftest['frequencies'],
                     'horizontal_shifts': ftest['horizontal_shifts']}

    ftest_dtest = {'x': ftest['x'][:,ftest_dtrain_size:],
                     'y': ftest['y'][:,ftest_dtrain_size:],
                     'amplitudes': ftest['amplitudes'],
                     'frequencies': ftest['frequencies'],
                     'horizontal_shifts': ftest['horizontal_shifts']}

    ds_ftest_dtrain = SinusoidMetaDataset(x=ftest_dtrain['x'],
                                    y=ftest_dtrain['y'],
                                    amplitudes=ftest_dtrain['amplitudes'],
                                    frequencies=ftest_dtrain['frequencies'],
                                    horizontal_shifts=ftest_dtrain['horizontal_shifts'],
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range)

    ds_ftest_dtest = SinusoidMetaDataset(x=ftest_dtest['x'],
                                    y=ftest_dtest['y'],
                                    amplitudes=ftest_dtest['amplitudes'],
                                    frequencies=ftest_dtest['frequencies'],
                                    horizontal_shifts=ftest_dtest['horizontal_shifts'],
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range)

    return ds_ftrain_dtrain, ds_ftrain_dtest, ds_ftest_dtrain, ds_ftest_dtest



# LD50

# TODO Add functionality to select datapoint split
#      For this you first need to define the datapoint split somehow

def get_ld50_dataframe(function_split: Union[None,str,List[str]],
                       datapoint_split: Union[None,str,List[str]]) -> pd.DataFrame:
    """
    Load LD50 dataset

    - function_split: either 'train' or 'test'.
    - datapoint_split: either 'train', 'test' (for cluster split) or
        'random_train', 'random_test' (for random split).
    """
    df = pd.read_csv(DATA_DIR / 'ld50' / 'LD50_dataset.csv').drop(columns=['RTECS_ID', 'CAS_RN'])
    df = df.rename(columns={'SMILES': 'smiles'})

    if datapoint_split == 'train' or datapoint_split == 'test':
        raise NotImplementedError('So far only a random datapoint split is implemented.')

    # Select datapoints across the row dimension
    if datapoint_split is not None:
        datapoints_selected = []
        if isinstance(datapoint_split, str):
            datapoint_split = [datapoint_split]
        for set in datapoint_split:
            datapoints_selected += LD50_DATAPOINT_SPLIT_DICT[set]
        mask_datapoint = df['smiles'].isin(datapoints_selected)
        df = df.loc[mask_datapoint]
    
    # Select functions across the column dimension
    # (make sure to retain "inchikey" and "smiles")
    if function_split is not None:
        functions_selected = ['smiles']
        if isinstance(function_split, str):
            function_split = [function_split]
        for set in function_split:
            functions_selected += LD50_FUNCTION_SPLIT_DICT[set]
        mask_function = df.columns.isin(functions_selected)
        df = df.loc[:,mask_function]

    # reset index
    df = df.reset_index(drop=True)

    # compute statistics about min and max number of labeled datapoints across
    # all functions
    num_labeled = (~df.drop(columns=['smiles']).isnull()).sum(axis=0).sort_values(ascending=False)
    max_num_datapoints = num_labeled[0]
    min_num_datapoints = num_labeled[-1]
    
    return df, (min_num_datapoints, max_num_datapoints)
# %%

def get_ld50_metasplits(input_type: str,
                        num_contexts_range: Tuple[int],
                        num_targets_range: Tuple[int],
                        datapoint_split_type: str,
                        num_dtrain_to_sample: Optional[int]=None) -> Tuple[Dataset]:
    """
    Function to get DOCKSTRING splits for meta-learning:
    - Function train, datapoint train (ftrain, dtrain)
    - Function train, datapoint test (ftrain, dtest)
    - Function test, datapoint train (ftest, dtrain)
    - Function test, datapoint test (ftest, dtest)
    """

    # select dataset class based on input type
    if input_type == 'graphs':
        DatasetClass = MolecularGraphMetaDataset
    elif input_type == 'fingerprints':
        raise NotImplementedError
    else:
        raise ValueError('"input_type" should be either "graphs" or "fingerprints".')

    # select strings to choose the random or other datapoint splits
    # (e.g. 'train', 'test', 'random_train', 'random_test', etc)
    datapoint_strings = choose_datapoint_strings(splits_type=datapoint_split_type)
    datapoint_train_str, datapoint_test_str = datapoint_strings

    # Load dataframes
    df_ftrain_dtrain, stat_ftrain_dtrain = get_ld50_dataframe(function_split='train',
                                          datapoint_split=datapoint_train_str)
    df_ftrain_dtest, stat_ftrain_dtest = get_ld50_dataframe(function_split='train',
                                         datapoint_split=datapoint_test_str)
    df_ftest_dtrain, stat_ftest_dtrain = get_ld50_dataframe(function_split='test',
                                         datapoint_split=datapoint_train_str)
    df_ftest_dtest, stat_ftest_dtest = get_ld50_dataframe(function_split='test',
                                        datapoint_split=datapoint_test_str)


    print_num_labeled_datapoints_message(stat_ftrain_dtrain,
                                         stat_ftrain_dtest,
                                         stat_ftest_dtrain,
                                         stat_ftest_dtest)


    # select smaller sample at random (for few-shot learning experiments)
    if num_dtrain_to_sample is not None:
        raise NotImplementedError

    # create Pytorch datasets
    ds_ftrain_dtrain = DatasetClass(smiles=df_ftrain_dtrain['smiles'].values,
                                    functions=df_ftrain_dtrain.iloc[:,1:], 
                                    num_contexts_range=num_contexts_range, 
                                    num_targets_range=num_targets_range)
    ds_ftrain_dtest = DatasetClass(smiles=df_ftrain_dtest['smiles'].values,
                                    functions=df_ftrain_dtest.iloc[:,1:], 
                                    num_contexts_range=num_contexts_range, 
                                    num_targets_range=num_targets_range)
    ds_ftest_dtrain = DatasetClass(smiles=df_ftest_dtrain['smiles'].values,
                                    functions=df_ftest_dtrain.iloc[:,1:],
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range)
    ds_ftest_dtest = DatasetClass(smiles=df_ftest_dtest['smiles'].values,
                                    functions=df_ftest_dtest.iloc[:,1:],
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range)

    return ds_ftrain_dtrain, ds_ftrain_dtest, ds_ftest_dtrain, ds_ftest_dtest


# uniprot ids

def get_uniprot_ids() -> pd.DataFrame:
    path = DATA_DIR / 'uniprot_ids' / 'uniprot_ids.tsv'
    df = pd.read_csv(path, sep='\t').set_index('UniProt accession')
    return df


# bioembeddings

def dict_to_df(dictionary: dict) -> pd.DataFrame:
    # collect keys and values into lists
    keys = []
    values = []
    for k,v in dictionary.items():
        keys.append(k)
        values.append(v)
    # stack values into a matrix
    values = np.vstack(values)
    # convert to dataframe
    return pd.DataFrame(values, index=keys)

def get_bioembeddings(embedder_name: str) -> pd.DataFrame:
    """
    embedder_name: one of
        - 'bepler'
        - 'cpcprot'
        - 'plus_rnn'
        - 'prottrans_albert_bfd'
        - 'prottrans_bert_bfd'
        - 'prottrans_t5_xl_u50'
        - 'prottrans_xlnet_uniref100'
        - 'seqvec'
        - 'unirep'
    """
    path = DATA_DIR / 'bioembeddings' / f'embeddings_{embedder_name}.pkl'
    with open(path, 'rb') as f:
        bioembeddings = pickle.load(f)
    bioembeddings = dict_to_df(bioembeddings)
    return bioembeddings


# ifeatures

def get_ifeatures(normalize: bool=False) -> pd.DataFrame:
    path = DATA_DIR / 'ifeatures' / 'descriptors.tsv'
    df = pd.read_csv(path, sep='\t').set_index('#')
    mask_constant = df.std() == 0
    df = df.loc[:,~mask_constant]
    if normalize:
        df = (df - df.mean()) / df.std()
    return df



# LINCS

def unbury_from_list(elem: Union[None, list]) -> Union[None, np.ndarray]:

    # for measurement values
    if isinstance(elem, list):
        return elem[0][0]
    # for 'canonical_smiles'
    elif isinstance(elem, str):
        return elem
    # for 'cluster' values
    elif isinstance(elem, int):
        return elem
    # for missing measurement values
    elif elem is None:
        return None
    elif np.isnan(elem):
        return np.nan
    else:
        print(type(elem))
        print(elem)
        raise ValueError

def get_lincs_dataframe(function_split: Union[str, List[str]],
                        datapoint_split: Union[str, List[str]],
                        select_single_gene: Optional[str]=None) -> pd.DataFrame:
    # create empty dataframe
    path = DATA_DIR / 'lincs' / 'y_below75_train.pkl'
    with open(path, 'rb') as f:
        ref_df = pickle.load(f)
    df = pd.DataFrame(index=ref_df.index, columns=ref_df.columns)
    df['canonical_smiles'] = ref_df['canonical_smiles']
    df['cluster'] = ref_df['cluster']
    del ref_df
    # identify desired cell lines
    if isinstance(function_split, str):
        function_split = [function_split]
    path = DATA_DIR / 'lincs' / 'function_split.tsv'
    cell_lines = pd.read_csv(path, sep='\t')
    mask_cell_lines = cell_lines['fset'].isin(function_split)
    cell_lines = cell_lines.loc[mask_cell_lines, 'cell_line'].tolist()
    # restrict df to desired cell lines only
    df = df.loc[:, cell_lines + ['canonical_smiles', 'cluster']]
    # fill with every measurement in the loaded dataframes
    if select_single_gene is not None:
        path = DATA_DIR / 'lincs' / 'gene_column_positions.tsv'
        gene_column_positions = pd.read_csv(path, sep='\t').set_index('gene_symbol')
        single_gene_idx = gene_column_positions.loc[select_single_gene,
                                                    'column_position']
    if isinstance(datapoint_split, str):
        datapoint_split = [datapoint_split]
    for each_dset in datapoint_split:
        # load this dset
        path = DATA_DIR / 'lincs' / f'y_below75_{each_dset}.pkl'
        with open(path, 'rb') as f:
            df_dset = pickle.load(f)
        # restrict to desired cell lines only
        df_dset = df_dset.loc[:, cell_lines]
        # identify measured cell_line,compound pairs
        notnull_pairs = list(df_dset[df_dset.notnull()].stack().index)
        # transfer the measured pairs from the dset dataframe to the
        # empty dataframe
        for each_notnull_pair in tqdm(notnull_pairs):
            each_compound, each_line = each_notnull_pair
            if select_single_gene is None:
                df.loc[each_compound, each_line] = [[df_dset.loc[each_compound,
                                                                 each_line]]] # hack
            else:
                df.loc[each_compound, each_line] = df_dset.loc[each_compound,
                                                               each_line][single_gene_idx].astype(float)
    # unbury arrays in lists
    if select_single_gene is None:
        df = df.applymap(unbury_from_list) # hack
    # return filled dataframe
    return df


def get_values_info(df: pd.DataFrame) -> tuple:
    # stack all the values and row, col info in a sparse dataset
    values = np.stack(df.stack().values)
    info = np.stack(df.stack().index)
    return values, info

def get_principal_components(df: pd.DataFrame, n_components=1) -> PCA:
    # get the principal components of the vectors in a sparse dataset
    values, _ = get_values_info(df)
    pca = PCA(n_components=n_components).fit(values)
    return pca
    
def project(df, pca):
    # project the vectors in a sparse dataset along principal components
    values, info = get_values_info(df)
    scores = pca.transform(values)
    return scores, info

def recreate_dataframe(df: pd.DataFrame, scores: np.ndarray, info: np.ndarray) -> pd.DataFrame:
    # given an example sparse dataframe of vectors, a set of scores for the 
    # vectors and the info about their location, recreate the same dataframe
    # but with the scores instead of the vectors
    df_scores = pd.DataFrame(index=df.index, columns=df.columns)
    for each_info, each_scores in zip(info, scores):
        row, col = each_info
        if each_scores.ndim == 1:
            each_scores = each_scores[0]
        df_scores.loc[row, col] = each_scores
    return df_scores

def get_scores_dataframe(df, pca):
    # given a sparse dataset with vectos in the non-null values and the
    # principal components for that vector type, project along the components
    # and make a new dataframe with the scores
    scores, info = project(df, pca)
    df_scores = recreate_dataframe(df, scores, info)
    return df_scores

def lincs_pca_decomposition(df_ftrain_dtrain, df_ftrain_dtest, df_ftest_dtrain, df_ftest_dtest):
    non_numerical_columns = ['canonical_smiles', 'cluster']
    pca = get_principal_components(df_ftrain_dtrain.drop(columns=non_numerical_columns))
    scores_ftrain_dtrain = get_scores_dataframe(df_ftrain_dtrain.drop(columns=non_numerical_columns), pca)
    scores_ftrain_dtest = get_scores_dataframe(df_ftrain_dtest.drop(columns=non_numerical_columns), pca)
    scores_ftest_dtrain = get_scores_dataframe(df_ftest_dtrain.drop(columns=non_numerical_columns), pca)
    scores_ftest_dtest = get_scores_dataframe(df_ftest_dtest.drop(columns=non_numerical_columns), pca)

    scores_ftrain_dtrain[non_numerical_columns] = df_ftrain_dtrain[non_numerical_columns]
    scores_ftrain_dtest[non_numerical_columns] = df_ftrain_dtest[non_numerical_columns]
    scores_ftest_dtrain[non_numerical_columns] = df_ftest_dtrain[non_numerical_columns]
    scores_ftest_dtest[non_numerical_columns] = df_ftest_dtest[non_numerical_columns]

    return scores_ftrain_dtrain, scores_ftrain_dtest, scores_ftest_dtrain, scores_ftest_dtest


def get_lincs_metasplits(input_type: str,
                    num_contexts_range: Tuple[int],
                    num_targets_range: Tuple[int],
                    num_dtrain_to_sample: Optional[int]=None,
                    random_seed: int=0,
                    pca_decomposition: bool=False,
                    select_single_gene: Optional[str]=None,
                    # train_dset: Union[str,List[str]]='train',
                    # test_dset: Union[str, List[str]]='test',
                    ) -> Tuple[Dataset]:
    """
    Function to get DOCKSTRING splits for meta-learning:
    - Function train, datapoint train (ftrain, dtrain)
    - Function train, datapoint test (ftrain, dtest)
    - Function test, datapoint train (ftest, dtrain)
    - Function test, datapoint test (ftest, dtest)
    """

    # select dataset class based on input type
    if input_type == 'graphs':
        DatasetClass = MolecularGraphMetaDataset
    elif input_type == 'fingerprints':
        DatasetClass = FingerprintMetaDataset
    else:
        raise ValueError('"input_type" should be either "graphs" or "fingerprints".')

    # Load dataframes
    df_ftrain_dtrain = get_lincs_dataframe(function_split='train',
                                           datapoint_split='train',
                                           select_single_gene=select_single_gene)
    df_ftrain_dtest = get_lincs_dataframe(function_split='train',
                                          datapoint_split='test',
                                           select_single_gene=select_single_gene)
    df_ftest_dtrain = get_lincs_dataframe(function_split='test',
                                          datapoint_split='train',
                                           select_single_gene=select_single_gene)
    df_ftest_dtest = get_lincs_dataframe(function_split='test',
                                         datapoint_split='test',
                                           select_single_gene=select_single_gene)

    if pca_decomposition:
        df_pca = lincs_pca_decomposition(df_ftrain_dtrain, df_ftrain_dtest,
                                         df_ftest_dtrain, df_ftest_dtest)
        df_ftrain_dtrain, df_ftrain_dtest, df_ftest_dtrain, df_ftest_dtest = df_pca

    # select smaller sample at random (for few-shot learning experiments)
    if num_dtrain_to_sample is not None:
        df_ftrain_dtrain = sample_dtrain(df_dtrain=df_ftrain_dtrain,
                                         num_dtrain_to_sample=num_dtrain_to_sample,
                                         random_seed=random_seed)
        df_ftest_dtrain = sample_dtrain(df_dtrain=df_ftest_dtrain,
                                        num_dtrain_to_sample=num_dtrain_to_sample,
                                        random_seed=random_seed)

    # create Pytorch datasets
    if pca_decomposition or select_single_gene is not None:
        y_len = 1
    else:
        y_len = 978
    ds_ftrain_dtrain = DatasetClass(smiles=df_ftrain_dtrain['canonical_smiles'].values,
                                    functions=df_ftrain_dtrain.iloc[:,:-2].astype(np.float32),
                                    num_contexts_range=num_contexts_range, 
                                    num_targets_range=num_targets_range,
                                    y_len=y_len)
    ds_ftrain_dtest = DatasetClass(smiles=df_ftrain_dtest['canonical_smiles'].values,
                                    functions=df_ftrain_dtest.iloc[:,:-2].astype(np.float32),
                                    num_contexts_range=num_contexts_range, 
                                    num_targets_range=num_targets_range,
                                    y_len=y_len)
    ds_ftest_dtrain = DatasetClass(smiles=df_ftest_dtrain['canonical_smiles'].values,
                                    functions=df_ftest_dtrain.iloc[:,:-2].astype(np.float32),
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range,
                                    y_len=y_len)
    ds_ftest_dtest = DatasetClass(smiles=df_ftest_dtest['canonical_smiles'].values,
                                    functions=df_ftest_dtest.iloc[:,:-2].astype(np.float32),
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range,
                                    y_len=y_len)

    return ds_ftrain_dtrain, ds_ftrain_dtest, ds_ftest_dtrain, ds_ftest_dtest
# %%


# sider

def get_sider_dataframe(function_split: Union[str, List[str]],
                        datapoint_split: Union[str, List[str]],
                        make_sparse: bool=False,
                        random_state: int=0) -> pd.DataFrame:

    path = DATA_DIR / 'sider' / 'sider.tsv'
    df = pd.read_csv(path, sep='\t').set_index('stereo_id')

    # discard molecules with more than 75 atoms
    mask_small_mols = df['num_atoms'] <=  75
    df = df.loc[mask_small_mols]

    # datapoint split
    if isinstance(datapoint_split, str):
        datapoint_split = [datapoint_split]
    mask_datapoint = df['split'].isin(datapoint_split)
    df = df.loc[mask_datapoint]

    # function split
    path = DATA_DIR / 'sider' / 'function_split.tsv'
    df_function_split = pd.read_csv(path, sep='\t')
    if isinstance(function_split, str):
        function_split = [function_split]
    mask_function = df_function_split['split'].isin(function_split)
    functions_selected = df_function_split.loc[mask_function, 'function'].tolist()
    df = df.loc[:, functions_selected + ['smiles']]

    # recreate sparsity
    if make_sparse:
        num_datapoints = df.shape[0]
        # we will create 50% sparsity
        sample_size = int(num_datapoints * 0.5)
        for i, each_function in enumerate(functions_selected):
            np.random.seed(seed=i+random_state)
            idx_to_make_sparse = np.random.choice(num_datapoints,
                                                  size=sample_size,
                                                  replace=False)
            idx_to_make_sparse = df.index[idx_to_make_sparse]
            df.loc[idx_to_make_sparse,each_function] = None

    return df


def get_sider_metasplits(input_type: str,
                         num_contexts_range: Tuple[int],
                         num_targets_range: Tuple[int],
                         train_dset: Union[str,List[str]]='train',
                         test_dset: Union[str, List[str]]='test',
                         make_sparse: bool=False,
                        ) -> Tuple[Dataset]:

    # select dataset class based on input type
    if input_type == 'graphs':
        DatasetClass = MolecularGraphMetaDataset
        counts = None
    elif input_type == 'fingerprints':
        DatasetClass = FingerprintMetaDataset
        counts = False
    elif input_type == 'count_fingerprints':
        DatasetClass = FingerprintMetaDataset
        counts = True
    else:
        raise ValueError('"input_type" should be either "graphs" or "fingerprints".')

    # Load dataframes
    df_ftrain_dtrain = get_sider_dataframe(function_split='train',
                                           datapoint_split=train_dset,
                                           make_sparse=make_sparse)
    df_ftrain_dtest = get_sider_dataframe(function_split='train',
                                           datapoint_split=test_dset,
                                           make_sparse=make_sparse)
    df_ftest_dtrain = get_sider_dataframe(function_split='test',
                                           datapoint_split=train_dset,
                                           make_sparse=make_sparse)
    # the actual test datapoints on which to compute metrics (ftest, dtest)
    # are not made sparse
    df_ftest_dtest = get_sider_dataframe(function_split='test',
                                           datapoint_split=test_dset,
                                           make_sparse=False)

    # create Pytorch datasets
    ds_ftrain_dtrain = DatasetClass(smiles=df_ftrain_dtrain['smiles'].values,
                                    functions=df_ftrain_dtrain.drop(columns=['smiles']), 
                                    num_contexts_range=num_contexts_range, 
                                    num_targets_range=num_targets_range,
                                    split_mode='disjoint_rigid',
                                    counts=counts,
                                    prediction_mode='binary_classification')
    ds_ftrain_dtest = DatasetClass(smiles=df_ftrain_dtest['smiles'].values,
                                    functions=df_ftrain_dtest.drop(columns=['smiles']), 
                                    num_contexts_range=num_contexts_range, 
                                    num_targets_range=num_targets_range,
                                    split_mode='disjoint_rigid',
                                    counts=counts,
                                    prediction_mode='binary_classification')
    ds_ftest_dtrain = DatasetClass(smiles=df_ftest_dtrain['smiles'].values,
                                    functions=df_ftest_dtrain.drop(columns=['smiles']),
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range,
                                    split_mode='disjoint_rigid',
                                    counts=counts,
                                    prediction_mode='binary_classification')
    ds_ftest_dtest = DatasetClass(smiles=df_ftest_dtest['smiles'].values,
                                    functions=df_ftest_dtest.drop(columns=['smiles']),
                                    num_contexts_range=num_contexts_range,
                                    num_targets_range=num_targets_range,
                                    split_mode='disjoint_rigid',
                                    counts=counts,
                                    prediction_mode='binary_classification')

    return ds_ftrain_dtrain, ds_ftrain_dtest, ds_ftest_dtrain, ds_ftest_dtest



def get_sider_simplesplits(function_names: Union[str, List[str]],
                              input_type: str) -> Tuple[Dataset]:
    """
    Function to get SIDER splits for for a single function:

    dtrain, dtest
    """
    # select dataset class based on input type
    if input_type == 'graphs':
        DatasetClass = MolecularGraphSimpleDataset
    elif input_type == 'fingerprints':
        DatasetClass = FingerprintSimpleDataset
    else:
        raise ValueError('"input_type" should be either "graphs" or "fingerprints".')

    # load dataframes
    df_dtrain = get_sider_dataframe(function_split=['train','test'],
                                    datapoint_split='train')
    df_dtest = get_sider_dataframe(function_split=['train','test'],
                                    datapoint_split='test')

    # create Pytorch datasets
    if isinstance(function_names, str):
        function_names = [function_names]
    ds_dtrain = DatasetClass(smiles=df_dtrain['smiles'],
                             functions=df_dtrain[function_names],
                             prediction_mode='binary_classification')
    ds_dtest = DatasetClass(smiles=df_dtest['smiles'],
                             functions=df_dtest[function_names],
                             prediction_mode='binary_classification')

    return ds_dtrain, ds_dtest
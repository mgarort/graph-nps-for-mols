from mnp.definitions import DATA_DIR
from mnp.utils.chemistry import many_smiles_to_morganfp
import torch
import numpy as np
from torch.utils.data import Dataset
import gzip
import pickle
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
from typing import Union, List, Tuple, Optional
import math



class FingerprintMetaDataset(Dataset):

    def __init__(self, smiles: pd.Series, functions: pd.DataFrame,
                 num_contexts_range: Tuple[int,int], 
                 num_targets_range: Tuple[int,int],
                 split_mode: str='disjoint_rigid',
                 featurization_progress_bar: bool=True,
                 counts: bool=False,
                 prediction_mode: str='regression',
                 y_len: int=1) -> None:
        """
        Args:
            - smiles: series of molecules in SMILES format.
            - functions: dataframe with the function values to use as prediction
                targets. Its columns must contain the function names, and the
                index must be the same as that of the smiles series.
            - num_contexts_range: range to randomly select the number of context
                datapoints C (number of x_c and y_c). To select a fixed size,
                set to (fixed_size, fixed_size+1)
            - num_targets_range: range to randomly select the number of target
                datapoints T (number of x_t and y_t).
            - split_mode: either "overlap", "disjoint_rigid" or "disjoint_flexible".
            -featurization_progress_bar: whether the show a progress bar during
                featurization.
            - prediction_mode: either "regression" or "binary_classification".
        """
        super().__init__()

        # General attributes
        self.smiles = smiles
        self.functions = functions
        self.function_names = functions.columns.to_list()
        self.counts = counts
        self.prediction_mode = prediction_mode
        self.y_len = y_len


        # Save the desired size ranges for context and size to use later by __getitem__
        self.num_contexts_range = num_contexts_range
        self.num_targets_range = num_targets_range
        self.split_mode = split_mode

        # Create fingerprint data (this will be the inputs x)
        # The inputs x are the same per molecule, so they will be reused for different
        # functions with the same molecule.
        fps = many_smiles_to_morganfp(self.smiles,
                                      progress_bar=featurization_progress_bar,
                                      counts=counts)
        self.x = torch.tensor(fps).float()

        # Model outputs (y)
        # NOTE The shape of y is (num_functions, num_mols)
        self.y = self.functions.T.values

        # Sparseness
        # the following are needed for avoiding null values in sparse datasets
        self.mol_numeric_index = np.array(range(len(functions)))
        self.mask_labeled = ~functions.T.isnull()
        

    def get_ct_indices_for_one_function(self, function_idx: Optional[int]=None):
        '''
        Get indices of context (c) and target (t) datapoints for a single function
        with index function_idx:
        
        - function_idx is None if we are dealing with a dense dataset, since in
          that case all labels are available so we don't need to check for null
          values within each function, and we can get indices in the same way
          regardless of the specific function.
        - function_idx is an integer if we are dealing with a sparse dataset,
          since in that case different labels are available for different
          functions and we need to check for null values within each function.

        This function works together with shuffled_batch_collate_fn 
        in order to select batches of datapoints that:

        - Datapoints selected are not the same across functions.
        - Different batches can have different numbers of datapoints.

        The way this works is the following:

        - First, get_ct_indices_for_one_function selects context and target 
          datapoints at random for each function. It selects the maximum number 
          of context and target datapoints.
        - Second, shuffled_batch_collate_fn slices the full batch, selecting a
          certain random number of context and target datapoints. This way,
          different batches will have different numbers of datapoints.
        '''
        max_num_contexts = self.num_contexts_range[1]
        max_num_targets = self.num_targets_range[1]
        # function_idx is None if not checking for null values, and an integer
        # if checking for null values
        function_num_datapoints = self.num_datapoints(function_idx)

        if self.split_mode == 'overlap':
            # "overlap" allows for overlapping of context and target points
            assert max_num_contexts <= function_num_datapoints
            assert max_num_targets <= function_num_datapoints
            shuffled_indices_1 = np.random.permutation(function_num_datapoints)
            shuffled_indices_2 = np.random.permutation(function_num_datapoints)
            context_indices = shuffled_indices_1[:max_num_contexts]
            target_indices = shuffled_indices_2[:max_num_targets]
        elif self.split_mode == 'disjoint_rigid':
            # "disjoint_rigid" gets disjoint points as contexts and targets, and
            # the number of contexts and targets must remain within the range
            # specified by the user
            assert max_num_contexts + max_num_targets <= function_num_datapoints
            shuffled_indices = np.random.permutation(function_num_datapoints)
            context_indices = shuffled_indices[:max_num_contexts]
            target_indices = shuffled_indices[max_num_contexts:max_num_contexts+max_num_targets]
        elif self.split_mode == 'disjoint_flexible':
            # "disjoint_flexible" gets disjoint points as contexts and targets:
            # - if there are enough labeled datapoints for the current function,
            #   the number of contexts and targets remains within the range
            #   specified by the user.
            # - if there are not enough labeled datapoints for the current function,
            #   we use half the datapoints as contexts and half the datapoints as contexts
            shuffled_indices = np.random.permutation(function_num_datapoints)
            if max_num_contexts + max_num_targets <= function_num_datapoints:
                context_indices = shuffled_indices[:max_num_contexts]
                target_indices = shuffled_indices[max_num_contexts:max_num_contexts+max_num_targets]
            else:
                half = int((function_num_datapoints) / 2)
                context_indices = shuffled_indices[:half]
                target_indices = shuffled_indices[half:function_num_datapoints]                
        else:
            raise ValueError('split_mode should be "overlap", "disjoint_rigid" or "disjoint_flexible".')

        labeled_indices = self.labeled_indices(function_idx)
        context_indices = labeled_indices[context_indices]
        target_indices = labeled_indices[target_indices]

        return context_indices, target_indices


    def __len__(self):
        return self.num_functions()

    def num_functions(self):
        return self.y.shape[0]
    
    def num_datapoints(self, function_idx: Optional[int]=None):
        return self.mask_labeled.iloc[function_idx].sum()

    def num_features(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        # get shuffled indices for context and target so that each function presents the
        # datapoints in different order and the batch contains different datapoints across
        # different functions
        context_indices, target_indices = self.get_ct_indices_for_one_function(idx)
        # self.x has dimensions (num_mols x fingerprint_size)
        x_c = self.x[context_indices]
        x_t = self.x[target_indices]
        # self.y has dimensions (num_functions x num_mols)
        y_c = self.y[idx, context_indices]
        y_t = self.y[idx, target_indices]
        if self.y_len == 1:
            y_c = torch.tensor(y_c)[:, None]
            y_t = torch.tensor(y_t)[:, None]
        else:
            y_c = torch.tensor(np.stack(y_c))
            y_t = torch.tensor(np.stack(y_t))            
        # self.smiles has dimensions (num_mols)
        smiles_c = self.smiles[context_indices]
        smiles_t = self.smiles[target_indices]
        # Function names
        function_name = self.function_names[idx]
        return {'x_c': x_c, 'x_t': x_t, 'y_c': y_c, 'y_t': y_t, 
                'smiles_c': smiles_c, 'smiles_t': smiles_t,
                'function_name': function_name}

    def getfullitem(self, idx):
        """
        Get values of the `idx`-th function for all context and target points.
        NOTE Dimensions be used with normal batching in a dataloader, i.e. it
        still requires the function dimension at the beginning.
        """
        # select only those molecules which have labels
        labeled_indices = self.labeled_indices(idx)
        fullitem =  {
                # with extra dimension at the beginning
                'x': self.x[None, labeled_indices],
                'y': torch.tensor(self.y)[[[idx]], labeled_indices],
                'smiles': self.smiles[None, labeled_indices],
                # without extra dimension at the beginning
                'function_name': self.function_names[idx],
                'functions': self.functions.iloc[labeled_indices, [idx]],
                'mask_labeled': self.mask_labeled.iloc[[idx], labeled_indices]}
        fullitem['mol_numeric_index'] = np.array(range(len(labeled_indices)))
        return fullitem

    def getfullitembyname(self, fname: str) -> dict:
        """
        Similar to getfullitem, but with function string name instead of integer
        index.
        NOTE Dimensions to be used manually, i.e. it already produces the 
        function dimension at the beginning
        """
        idx = self.function_names.index(fname)
        return self.getfullitem(idx)

    def labeled_indices(self, function_idx: int):
        mask_labeled = self.mask_labeled.iloc[function_idx]
        labeled_indices = self.mol_numeric_index[mask_labeled]
        return labeled_indices




class FingerprintSimpleDataset(Dataset):

    """
    Pytorch dataset to hold a single molecular dataset represented as fingerprints.
    """

    def __init__(self, smiles: pd.Series, functions: pd.DataFrame,
                 progress_bar: bool=True,
                 prediction_mode: str='regression') -> None:
        """
        - smiles: series of molecules in SMILES format.
        - function: dataframe with a single function values to use as prediction targets. Its
            columns must contain the function names, and the index must be the same as that
            of the smiles series.
        """
        super().__init__()

        # General attributes
        self.smiles = smiles
        assert functions.shape[1] == 1 # make sure that we are creating a dataset with a single function
        self.functions = functions
        self.function_names = functions.columns.to_list()
        self.prediction_mode = prediction_mode

        # Create fingerprint data (this will be the inputs x)
        # The inputs x are the same per molecule, so they will be reused for different
        # functions with the same molecule.
        fps = many_smiles_to_morganfp(smiles=self.smiles, progress_bar=progress_bar)
        self.x = torch.tensor(fps).float()

        # Model outputs (y)
        # NOTE The shape of y is (num_mols, 1)
        self.y = torch.tensor(self.functions.values)


    def num_datapoints(self):
        return self.x.shape[0]


    def num_functions(self):
        num_functions = self.y.shape[1]
        assert num_functions == 1
        return num_functions

    def __len__(self):
        """
        Get length of the dataset, which for a dataset with a single function is
        the number of datapoints
        """
        return self.num_datapoints()


    def __getitem__(self, idx):
        """
        Get values of the `idx`-th datapoint/molecule.
        """
        # x has dimensions (num_mols, fp_len)
        x = self.x[idx]
        # y has dimensions (num_mols, 1)
        y = self.y[idx]
        # Function names
        return {'x': x, 'y': y, 
                'function_name': self.function_names}

    def getfullitem(self, _):
        """
        Get values of the whole dataset.
        """
        fullitem =  {'x': self.x, 'y': self.y}
        # Add the function dimension at the beginning so that the model
        # can process the item as if it came from a batch
        fullitem = {k: v.unsqueeze(0) for k,v in fullitem.items()}
        fullitem['function_name'] = self.function_names[0]
        return fullitem





######################
# Sinusoid functions #
######################

class SinusoidMetaDataset(Dataset):

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 amplitudes: np.ndarray,
                 frequencies: np.ndarray,
                 horizontal_shifts: np.ndarray,
                 num_contexts_range: Tuple[int,int],
                 num_targets_range: Tuple[int,int]) -> None:
        """
        Inspired by the sinusoid functions in the MAML paper.

        Args:
            - fset: either 'train' or 'test'. They differ in the random seed offset,
                so that the functions are different between 'train' and 'test',
                but are reproducible within 'train' and within 'test' across
                different runs.
            - amplitude range, horizontal_shifts_range and input_range: parameters of
                the sinusoid functions. Default values are same as
                github.com/cbfinn/maml for reproducibility.
            - num_contexts_range: range to randomly select the number of context
                datapoints C (number of x_c and y_c). To select a fixed size, set
                to (fixed_size, fixed_size+1)
            - num_targets_range: range to randomly select the number of target
                datapoints T (number of x_t and y_t).
        """

        # general attributes
        self.x = torch.tensor(x).float()[:,:,None]
        self.y = torch.tensor(y).float()
        self.prediction_mode = 'regression'
        self.amplitudes = torch.tensor(amplitudes).float()
        self.frequencies = torch.tensor(frequencies).float()
        self.horizontal_shifts = torch.tensor(horizontal_shifts).float()

        # give names to functions for compatibility with DOCKSTRING metadatasets
        self.function_names = [f'{i}' for i in range(x.shape[0])]

        # Save the desired size ranges for context and size to use later by __getitem__
        self.num_contexts_range = num_contexts_range
        self.num_targets_range = num_targets_range

        # Make sure that the max contexts and targets does not exceed max datapoints
        if num_contexts_range[1] + num_targets_range[1] > self.num_datapoints():
            raise ValueError('Maximum number of contexts plus targets cannot exceed available datapoints.')


    def get_ct_indices_for_one_function(self):
        '''
        This function will shuffle all the datapoints within each function. This way,
        we will later on be able to choose datapoints independently across different
        functions with a single shuffling in the collate function.

        The datapoints across functions could overlap but they will most likely not
        be the exact same set of indices.
        '''

        '''
        Get indices of context (c) and target (t) datapoints for a single function.

        This function works together with shuffled_batch_collate_fn 
        in order to select batches of datapoints that:

        - Datapoints selected are not the same across functions.
        - Different batches can have different numbers of datapoints.

        The way this works is the following:

        - First, get_ct_indices_for_one_function selects context and target 
          datapoints at random for each function. It selects the maximum number 
          of context and target datapoints.
        - Second, shuffled_batch_collate_fn slices the full batch, selecting a
          certain random number of context and target datapoints. This way,
          different batches will have different numbers of datapoints.
        '''
        max_num_contexts = self.num_contexts_range[1]
        max_num_targets = self.num_targets_range[1]

        assert max_num_contexts + max_num_targets <= self.num_datapoints()
        shuffled_indices = np.random.permutation(self.num_datapoints())
        context_indices = shuffled_indices[:max_num_contexts]
        target_indices = shuffled_indices[max_num_contexts:max_num_contexts+max_num_targets]
        return context_indices, target_indices


    def num_functions(self):
        return self.x.shape[0]

    def num_datapoints(self):
        return self.x.shape[1]

    def __len__(self):
        return self.num_functions()

    def num_features(self):
        """sinusoid functions are \R -> \R"""
        return 1

    def __getitem__(self, idx: int) -> dict:
        context_indices, target_indices = self.get_ct_indices_for_one_function()
        amplitude = self.amplitudes[idx]
        frequency = self.frequencies[idx]
        horizontal_shifts = self.horizontal_shifts[idx]
        x_c = self.x[idx, context_indices]
        y_c = self.y[idx, context_indices]
        x_t = self.x[idx, target_indices]
        y_t = self.y[idx, target_indices]
        function_name = self.function_names[idx]
        return {'amplitude': amplitude,
                'frequency': frequency,
                'horizontal_shifts': horizontal_shifts,
                'function_name': function_name,
                'x_c': x_c, 'y_c': y_c,
                'x_t': x_t, 'y_t': y_t}        

    def getfullitem(self, idx: int) -> dict:
        """
        Get values of the `idx`-th function for all context and target points.
        NOTE Dimensions to be used manually, i.e. it already produces the 
        function dimension at the beginning
        """
        x = self.x[idx].clone().detach().unsqueeze(0)
        y = self.y[idx].clone().detach().unsqueeze(0)
        amplitude = self.amplitudes[idx]
        frequency = self.frequencies[idx]
        horizontal_shift = self.horizontal_shifts[idx]
        function_name = self.function_names[idx]    
        return {'amplitude': amplitude,
                'frequency': frequency,
                'horizontal_shift': horizontal_shift,
                'function_name': function_name, 
                'x': x, 'y': y}

    def getfullitembyname(self, fname: str) -> dict:
        """
        Similar to getfullitem, but with function string name instead of integer
        index.
        """
        idx = int(fname)
        return self.getfullitem(idx)

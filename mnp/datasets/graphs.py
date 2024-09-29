from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union

from mnp.molecular_featurizer import MolecularFeaturizer

from tqdm import tqdm
tqdm.pandas()


class MolecularGraphMetaDataset(Dataset):

    """
    Pytorch dataset to represent molecular datasets using the MolecularFeaturizer.
    """

    def __init__(self, smiles: np.ndarray, functions: pd.DataFrame,
                 num_contexts_range: Tuple[int,int],
                 num_targets_range: Tuple[int,int],
                 split_mode: str='disjoint_rigid',
                 max_num_atoms: int=75,
                 use_chirality: bool=True, use_stereochemistry: bool=True,
                 featurization_progress_bar: bool=True,
                 prediction_mode: str='regression',
                 y_len: int=1,
                 **kwargs) -> None:
        """
        - smiles: molecules to featurize in SMILES format.
        - functions: dataframe with the function values to use as prediction
            targets. Its columns must contain the function names, and the index
            must be the same as that of the smiles series.
        - use_chirality: whether to use chirality information.
        - use_stereochemistry: whether to use stereochemistry information.
        - split_mode: either "overlap", "disjoint_rigid" or "disjoint_flexible".
        -featurization_progress_bar: whether the show a progress bar during
            featurization.
        - prediction_mode: either "regression" or "binary_classification".
        """
        super().__init__()

        # Save the desired size ranges for context and size to use later by __getitem__
        self.num_contexts_range = num_contexts_range
        self.num_targets_range = num_targets_range
        self.split_mode = split_mode

        # Featurizer
        # NOTE Now the featurizer contains the attributes hydrogens_implicit,\
        # use_chirality, use_steoreochemistry, atoms_allowed and max_num_atoms
        self.featurizer = MolecularFeaturizer(max_num_atoms=max_num_atoms,
                                    use_chirality=use_chirality, 
                                    use_stereochemistry=use_stereochemistry,
                                    progress_bar=featurization_progress_bar)

        # General attributes
        self.smiles = smiles
        self.functions = functions
        self.function_names = functions.columns.to_list()
        self.prediction_mode = prediction_mode
        self.y_len = y_len

        # Model inputs (atoms, atoms' mask and bonds)
        # NOTE The size of each input is
        # - atoms (num_mols, max_num_atoms, num_atom_features)
        # - atoms_mask (num_mols, max_num_atoms, 1)
        # - bonds (num_mols, max_num_atoms, max_num_atoms, num_bond_features)
        # - adjacencies (num_mols, max_num_atoms, max_num_atoms)
        self.atoms, self.atoms_mask, self.bonds, self.adjacencies = self.featurizer(self.smiles)

        # Model outputs (y)
        # NOTE The shape of self.y is (num_functions, num_mols), and the shape
        #      of the final y in the batch is (num_functions, num_mols, y_len)
        self.y = self.functions.T.values

        # Sparseness
        # the following are needed for avoiding null values in sparse datasets
        self.mol_numeric_index = np.array(range(len(functions)))
        self.mask_labeled = functions.T.notnull()

    def reset_mol_numeric_index(self) -> None:
        self.mol_numeric_index = np.array(range(len(self.functions)))

    def reset_mask_labeled(self) -> None:
        self.mask_labeled = self.functions.T.notnull()

    def num_datapoints(self, function_idx: Optional[int]=None):
        return self.mask_labeled.iloc[function_idx].sum()

    # Pytorch dataset methods

    def num_functions(self):
        return self.y.shape[0]

    def __len__(self):
        """
        Get length of the dataset, which for NPs is the number of functions.
        """
        return self.num_functions()
        

    def __getitem__(self, idx):
        """
        Get values of the `idx`-th function for a random selection of context and
        target points.
        """
        # passing the function index to the function that gets the context and target
        # datapoint indices because some datasets may be sparse and have null values
        context_indices, target_indices = self.get_ct_indices_for_one_function(idx)
        # Model inputs (atoms, atoms_mask, bonds and adjacencies)
        atoms_c = self.atoms[context_indices]
        atoms_mask_c = self.atoms_mask[context_indices]
        bonds_c = self.bonds[context_indices]
        adjacencies_c = self.adjacencies[context_indices]
        smiles_c = self.smiles[context_indices]
        atoms_t = self.atoms[target_indices]
        atoms_mask_t = self.atoms_mask[target_indices]
        bonds_t = self.bonds[target_indices]
        adjacencies_t = self.adjacencies[target_indices]
        smiles_t = self.smiles[target_indices]
        # Model outputs
        # y has dimensions (num_functions, num_mols, y_len)
        y_c = self.y[idx, context_indices]
        y_t = self.y[idx, target_indices]
        if self.y_len == 1:
            y_c = torch.tensor(y_c)[:,None]
            y_t = torch.tensor(y_t)[:,None]
        else:
            y_c = torch.tensor(np.stack(y_c))
            y_t = torch.tensor(np.stack(y_t))
        # Function names
        function_name = self.function_names[idx]
        return {'atoms_c': atoms_c, 'atoms_mask_c': atoms_mask_c, 
                'bonds_c': bonds_c, 'adjacencies_c': adjacencies_c,
                'smiles_c': smiles_c, 'y_c': y_c,
                'atoms_t': atoms_t, 'atoms_mask_t': atoms_mask_t,
                'bonds_t': bonds_t, 'adjacencies_t': adjacencies_t,
                'smiles_t': smiles_t, 'y_t': y_t,
                'function_name': function_name}


    def getfullitem(self, idx: int) -> dict:
        """
        Get values of the `idx`-th function for all context and target points.
        NOTE Dimensions be used with normal batching in a dataloader, i.e. it
        still requires the function dimension at the beginning.
        """
        # select only those molecules which have labels
        labeled_indices = self.labeled_indices(idx)
        fullitem = {
            # with extra dimension at the beginning
            'atoms': self.atoms[None, labeled_indices],
            'atoms_mask': self.atoms_mask[None, labeled_indices],
            'bonds': self.bonds[None, labeled_indices],
            'adjacencies': self.adjacencies[None, labeled_indices],
            'smiles': self.smiles[None, labeled_indices],
            # without extra dimension at the beginning
            'function_name': self.function_names[idx],
            'functions': self.functions.iloc[labeled_indices, [idx]],
            'mask_labeled': self.mask_labeled.iloc[[idx], labeled_indices]}
        fullitem['mol_numeric_index'] = np.array(range(len(labeled_indices)))
        if self.y_len == 1:
            fullitem['y'] = torch.tensor(self.y)[[[idx]], labeled_indices]
        else:
            # self.y.shape
            # torch.Size([12324, 978])
            fullitem['y'] = torch.tensor(np.stack(self.y[idx, labeled_indices]))[None, :, :]
        return fullitem

    def get_idx_from_name(self, fname: str) -> int:
        idx = self.function_names.index(fname)
        return idx


    def getfullitembyname(self, fname: str) -> dict:
        """
        Similar to getfullitem, but with function string name instead of integer
        index.
        NOTE Dimensions to be used manually, i.e. it already produces the 
        function dimension at the beginning
        """
        idx = self.get_idx_from_name(fname)
        return self.getfullitem(idx)
        

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
            try:
                assert max_num_contexts + max_num_targets <= function_num_datapoints
            except AssertionError as e:
                print(e)
            shuffled_indices = np.random.permutation(function_num_datapoints)
            context_indices = shuffled_indices[:max_num_contexts]
            target_indices = shuffled_indices[max_num_contexts:max_num_contexts+max_num_targets]
        elif self.split_mode == 'forced_overlap':
            # the targets include both the targets and the contexts
            try:
                assert max_num_contexts + max_num_targets <= function_num_datapoints
            except AssertionError as e:
                print(e)
            shuffled_indices = np.random.permutation(function_num_datapoints)
            context_indices = shuffled_indices[:max_num_contexts]
            target_indices = shuffled_indices[max_num_contexts:max_num_contexts+max_num_targets]
            target_indices = np.concatenate([context_indices, target_indices])
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


    def labeled_indices(self, function_idx: int):
        mask_labeled = self.mask_labeled.iloc[function_idx]
        labeled_indices = self.mol_numeric_index[mask_labeled]
        return labeled_indices


class MolecularGraphSimpleDataset(Dataset):

    """
    Pytorch dataset to hold a simple molecular dataset using the
    MolecularFeaturizer.

    - The unit of simple datasets is the datapoint, and that is what simple
      batches are made of.
    - The unit of meta datasets are functions, and that is what meta datasets
      are made of.

    NOTE Simple datasets may contain several y's for each datapoint if doing
    multi-task learning.
    """

    def __init__(self, smiles: pd.Series, functions: pd.DataFrame,
                 use_chirality: bool=True, use_stereochemistry: bool=True,
                 progress_bar: bool=True,
                 prediction_mode: str='regression') -> None:
        """
        - smiles: series of molecules to featurize in SMILES format.
        - function: dataframe with a single function values to use as prediction targets. Its
            columns must contain the function names, and the index must be the same as that
            of the smiles series.
        - use_chirality: whether to use chirality information.
        - use_stereochemistry: whether to use stereochemistry information.
        """
        super().__init__()

        # Featurizer
        # NOTE Now the featurizer contains the attributes hydrogens_implicit,
        # use_chirality, use_steoreochemistry, atoms_allowed and max_num_atoms
        self.featurizer = MolecularFeaturizer(use_chirality=use_chirality, 
                                              use_stereochemistry=use_stereochemistry,
                                              progress_bar=progress_bar)

        # General attributes
        self.smiles = smiles
        self.functions = functions
        self.function_names = functions.columns.to_list()
        self.prediction_mode = prediction_mode

        # Model inputs (atoms, atoms' mask and bonds)
        # NOTE The size of each input is
        # - atoms (num_mols, max_num_atoms, num_atom_features)
        # - atoms_mask (num_mols, max_num_atoms, 1)
        # - bonds (num_mols, max_num_atoms, max_num_atoms, num_bond_features)
        # - adjacencies (num_mols, max_num_atoms, max_num_atoms)
        self.atoms, self.atoms_mask, self.bonds, self.adjacencies = self.featurizer(self.smiles)

        # Model outputs (y)
        # NOTE The shape of y is (num_mols, 1)
        self.y = torch.tensor(self.functions.values)


    def num_datapoints(self):
        return self.atoms.shape[0]


    def num_functions(self):
        """
        A simple dataset with multiple functions will result in multi-task learning.
        """
        num_functions = self.y.shape[1]
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
        atoms = self.atoms[idx]
        atoms_mask = self.atoms_mask[idx]
        bonds = self.bonds[idx]
        adjacencies = self.adjacencies[idx]
        smiles = self.smiles[idx]
        # Model outputs
        # y has dimensions (num_mols, 1)
        y = self.y[idx]
        # Function names
        return {'atoms': atoms, 'atoms_mask': atoms_mask, 
                'bonds': bonds, 'adjacencies': adjacencies,
                'smiles': smiles,
                'y': y, 'function_name': self.function_names}


    def getfullitem(self, _):
        """
        Get values of the whole dataset.
        """
        fullitem = {'atoms': self.atoms, 'atoms_mask': self.atoms_mask,
                'bonds': self.bonds, 'adjacencies': self.adjacencies,
                'y': self.y}
        # Add the function dimension at the beginning so that the model
        # can process the item as if it came from a batch
        fullitem = {k: v.unsqueeze(0) for k,v in fullitem.items()}
        fullitem['smiles'] = self.smiles[None, :]
        fullitem['function_name'] = self.function_names
        return fullitem
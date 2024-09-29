import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
tqdm.pandas()
from itertools import product
import argparse
import random
import pandas as pd

import pickle
from typing import Union, Optional, List, Tuple
from collections import OrderedDict
from copy import deepcopy
import gc
import wandb

from mnp.definitions import RESULTS_DIR
from mnp.datasets.graphs import (MolecularGraphMetaDataset,
                                 MolecularGraphSimpleDataset)
from mnp.datasets.vectors import (FingerprintMetaDataset,
                                  FingerprintSimpleDataset,
                                #   MNISTDataset,
                                  SinusoidMetaDataset)
from mnp.datasets.collate import (vectors_shuffled_batch_collate_fn, 
                                  graphs_shuffled_batch_collate_fn)
from mnp.losses import (ReconstructionLoss, ReconstructionRegularizationLoss,
                        ApproximateReconstructionRegularizationLoss)
from mnp.params import fixed_params
from mnp.models.graphs import (MolecularGraphAttentionNN,
                               MolecularGraphAttentionCNP,
                               MolecularGraphAttentionLNP) 
from mnp.models.vectors import (FullyConnectedCNP, AttentionCNP,
                                FullyConnectedLNP,
                                FullyConnectedNN)
from mnp.models.classical import FSS, KNN, RF, XGB, Dummy
from mnp.models.gp import (ApproximateGaussianProcessOnFingerprints,
                           ExactGaussianProcessOnFingerprints,
                           TanimotoKernel)
from mnp.utils.metrics import compute_phi



def map_experiment_id_to_params(params_catalogue: dict,
                                experiment_id: int) -> Union[None,dict]:
    """
    Helper function to map the experiment array index (single integer) to a 
    combination of parameters (dictionary of values). Useful to run many
    repetitions of the same experiment with different parameters.

    Args:
        params_catalogue: dictionary of parameters and values to be used. Keys
            are the names of the parameters, values are lists of possible param values.
        experiment_id: experiment index.

    Returns:
        If experiment index is smaller or equal than the number of parameter
        combinations, return a dictionary with the parameter values selected for this
        expeiment index.
        If experiment index is larger than the number of parameter combinations,
        return None to indicate that current job should be cancelled (this happens if
        we start an excess of jobs).
    """
    param_name_list = [key for key,value in params_catalogue.items()]
    param_size_list = [len(value) for key,value in params_catalogue.items()]
    product_indices = list(product(*[range(size) for size in param_size_list]))
    max_index = len(product_indices) - 1
    # If slurm_id is larger than needed, return None to indicate that
    # current job should be cancelled because there is an excess of jobs
    if experiment_id > max_index:
        return None
    # Otherwise, map the slurm array index to indices for each parameter
    else:
        # Choose param indices based on slurm_id
        chosen_indices = product_indices[experiment_id]
        # Choose params
        chosen_params = {}
        for param_name, corresponding_index in zip(param_name_list,chosen_indices):
            key = param_name
            value = params_catalogue[param_name][corresponding_index]
            chosen_params[key] = value
        return chosen_params


def map_params_to_experiment_id(chosen_params: dict,
                                params_catalogue: dict) -> Union[None,dict]:
    """
    """
    # Assert that the keys are the same
    if len(chosen_params.keys()) != len(params_catalogue.keys()):
        raise ValueError('Chosen params and params catalogue must have the ' \
                         'same number of entries.')
    for key1, key2 in  zip(sorted(chosen_params.keys()),
                           sorted(params_catalogue.keys())):
        if key1 != key2:
            raise ValueError('Chosen params and params catalogue must have ' \
                             'the same keys.')
    # Find the appropriate id
    catalogue_size = 1
    for k,v in params_catalogue.items():
        assert isinstance(v, list)
        catalogue_size *= len(v)
    for experiment_id in range(catalogue_size):
        these_chosen_params = map_experiment_id_to_params(params_catalogue,
                                                          experiment_id)
        if these_chosen_params == chosen_params:
            return experiment_id
    return None



def set_random_seed(seed: int) -> None:
    # Get high-quality seed in [-1e6, +1e6] from manual seed
    np.random.seed(seed)
    seed = np.random.randint(low=0, high=1e6)
    # Set seed for the 3 seeds in your Python neighbourhood
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(model: torch.nn.Module, experiment_name: str, experiment_id: int) -> None:
    path = RESULTS_DIR / experiment_name / f'model_{experiment_id}.pt'
    torch.save(model.state_dict(), path)


def save_params(params: dict, experiment_name: str,
                       experiment_id: int) -> None:
    path = RESULTS_DIR / experiment_name / f'params_{experiment_id}.pkl'
    with open(path, 'bw') as f:
        pickle.dump(params, f)


def save_results(results: dict, experiment_name: str, experiment_id: int) -> None:
    path = RESULTS_DIR / experiment_name / f'results_{experiment_id}.pkl'
    with open(path, 'bw') as f:
        pickle.dump(results, f)


def get_tensor_memory_size(tensor: torch.Tensor) -> float:
    """
    Calculate memory size of a tensor in gigabytes.
    """
    gb_size = tensor.nelement() * tensor.element_size() / 1e9
    return gb_size


def add_suffix_to_keys(dictionary: dict, suffix: str) -> dict:
    """
    Add a suffix to every key in a dictionary.
    """
    return { f'{k}{suffix}': v for k,v in dictionary.items()}


def cast_values_to_float(dictionary: dict) -> dict:
    """
    In a dictionary of Pytorch tensors, cast every value to float type.
    """
    new_dictionary = {}
    for k,v in dictionary.items():
        if isinstance(v, torch.Tensor):
            new_dictionary[k] = v.float()
        else:
            new_dictionary[k] = v
    return new_dictionary



def get_submodels_params(model: nn.Module,
                         submodels_names: List[str]) -> dict:
    """
    Given a list of strings containing submodels names, return a dictionary
    with all the named parameters in those submodels, such that:
    
    - the keys are the parameter names.
    - the values are the parameters.
    """
    # get submodels
    # (user-specified submodels may not be in the right order, so
    #  we re-create our own)
    user_submodels_names = submodels_names
    if isinstance(user_submodels_names, str):
        user_submodels_names = [user_submodels_names]
    submodels = []
    submodels_names = []
    named_modules = {name: module for name, module in model.named_modules()}
    for each_submodel_name in user_submodels_names:
        if each_submodel_name in named_modules.keys():
            submodels_names.append(each_submodel_name)
            submodels.append(named_modules[each_submodel_name])
        else:
            raise ValueError(f'Submodel {each_submodel_name} not found.')
    # make lists with the parameters and with the parameter names of all
    # the submodels of interest
    submodels_params = []
    submodels_params_names = []
    for each_submodel_name, each_submodel in zip(submodels_names, submodels):
        # parameters
        each_params = list(each_submodel.parameters())
        submodels_params += each_params
        # parameter names
        each_params_names =  [each_submodel_name + '.' + elem[0] for elem in
                             list(each_submodel.named_parameters())]
        submodels_params_names += each_params_names
    # make dictionary
    submodels_params_dict = {k:v for k,v in zip(submodels_params_names,
                                                submodels_params)}
    return submodels_params_dict

def should_create_graph(order, epoch):
    if order == 'first':
        create_graph = False
    elif order == 'second':
        create_graph = True
    elif order == 'anneal':
        if epoch <= 50:
            create_graph = False
        else:
            create_graph = True
    else:
        raise ValueError('Parameter "order" should be either "first", "second" or "anneal".')
    return create_graph


def get_grad_dict(submodels_params_names, submodels_params,
                  inner_loss, create_graph):
    grad = torch.autograd.grad(inner_loss, submodels_params,
                               create_graph=create_graph)
    grad_dict = {k:v for (k,v) in zip(submodels_params_names, grad)}
    return grad_dict


def fine_tuning_step(model: nn.Module, submodels_names: Union[str, List[str]],
                     batch: dict, criterion: nn.modules.loss._Loss, lr: float,
                     device: str, debug: bool, order: str='first',
                     epoch: Optional[int]=None) -> OrderedDict:
    """
    Function that takes one gradient step to improve prediction of the context
    points.
    - model: prediction model.
    - submodel_name: name of submodel (module in model) whose parameters we
        want to fine tune.
    - criterion: loss function to use (e.g. for CNP, ReconstructionLoss).
    - lr: learning rate.
    - device: either 'cpu' or 'cuda'.
    - order: either 'first' (for first-order approximation), 'second' (for full
        MAML) or 'anneal' (for the annealing in MAML++, where first 50 epochs are
        first order and after that it's second order).
    - epoch: used only if doing order annealing.

    Returns an ordered dictionary with the fine-tuned parameters after one learning step.
    """

    # compute fine-tuning loss
    inner_pred = model(batch) # prediction before fine-tuning
    if isinstance(criterion, ReconstructionLoss):
        inner_loss = criterion(
                        batch['y_c'].to(torch.device(device)), 
                        inner_pred['y_c_hat_mean'],
                        inner_pred['y_c_hat_var'])
    elif isinstance(criterion, nn.MSELoss):
        inner_loss = criterion(
                        inner_pred['y_c_hat_mean'],
                        batch['y_c'].to(torch.device(device)))
    elif isinstance(criterion, ReconstructionRegularizationLoss):
        inner_loss, rec, reg = criterion(y_t=batch['y_c'].to(torch.device(device)),
                                    y_t_hat_mean=inner_pred['y_c_hat_mean'],
                                    y_t_hat_var=inner_pred['y_c_hat_var'],
                                    full_z_dist=inner_pred['full_z_dist'],
                                    context_z_dist=inner_pred['context_z_dist'])
    elif isinstance(criterion, ApproximateReconstructionRegularizationLoss):
        inner_loss, rec, reg = criterion(y_t=batch['y_c'].to(torch.device(device)),
                                    y_t_hat_mean=inner_pred['y_c_hat_mean'],
                                    y_t_hat_var=inner_pred['y_c_hat_var'],
                                    full_z_dist=inner_pred['full_z_dist'],
                                    context_z_dist=inner_pred['context_z_dist'],
                                    full_z_sample=inner_pred['full_z_sample'])
    else:
        raise ValueError('criterion class is not recognized.')

    if debug:
        if inner_loss.isnan().any().item():
            print('Here there is a inner loss nan.')

    # get submodels params
    submodels_params_dict = get_submodels_params(model=model,
                                                 submodels_names=submodels_names)
    submodels_params_names = [k for k,v in submodels_params_dict.items()]
    submodels_params = [v for k,v in submodels_params_dict.items()]
    
    # compute gradient w.r.t. submodel parameters
    create_graph = should_create_graph(order, epoch)
    grad_dict = get_grad_dict(submodels_params_names, submodels_params,
                  inner_loss, create_graph)

    # inspect sizes of gradients
    if debug:
        max = -1000
        min = +1000
        for k, v in grad_dict.items():
            v_max = v.max().item()
            v_min = v.min().item()
            if v_max > max:
                max = v_max
            if v_min < min:
                min = v_min
        if max > 30 or min < -30:
            print(f'Max of grad {max}, min of grad {min}')

    if debug:
        for k,v in grad_dict.items():
            if v.isnan().any().item():
                print('Here there is a grad nan')

    # clip gradients to make algorithm robust
    # -10 and +10 is the clipping of the gradients in the outer loop of MAML++
    for k, v in grad_dict.items():
        grad_dict[k] = torch.clamp(v, min=-10, max=+10)

    # take gradient step only in submodel parameters
    params = OrderedDict()
    for (name, param) in model.named_parameters():
        if name in submodels_params_names:
            params[name] = param - lr * grad_dict[name]
            if debug:
                if params[name].isnan().any().item():
                    print('Here is an update nan.')
        else:
            params[name] = param

    return params



def apply_finetuning_steps(model, submodels_to_tune,
                          num_finetuning_steps, criterion,
                          batch, lr, finetuning_order, device,
                          verbose,
                          in_place: bool=True):
    """
    Apply fine-tuning steps (in place or not).
    """
    if not in_place:
        model = deepcopy(model)
    # finetune with short-tuning
    # (this uses a meta-batch with _c contexts and _t targets)
    for _ in range(num_finetuning_steps):
        shorttuned_params = fine_tuning_step(model=model,
                                submodels_names=submodels_to_tune,
                                batch=batch,
                                criterion=criterion,
                                lr=lr,
                                device=device,
                                debug=verbose,
                                order=finetuning_order)
        model.load_state_dict(shorttuned_params)

    return model


def get_annealing_coefficients(i_epoch, anneal_start, anneal_end):
    anneal_range = anneal_end - anneal_start
    coefficient_2 = np.min((1, np.max((i_epoch - anneal_start, 0)) / anneal_range))
    coefficient_1 = 1 - coefficient_2
    return coefficient_1, coefficient_2



def get_collate_fn_for_dataset(ds: Dataset,
                               num_contexts_range: Tuple[int],
                               num_targets_range: Tuple[int]):
    """
    Given a dataset of MNIST images, fingerprints or molecular graphs,
    get the appropriate collate function for neural processes that is
    appropriate for them.
    Also initialize it with the number of contexts and targets desired.
    """
    if isinstance(ds, MolecularGraphMetaDataset):
        collate_fn = lambda batch : graphs_shuffled_batch_collate_fn(batch, 
                                        num_contexts_range=num_contexts_range,
                                        num_targets_range=num_targets_range)
                                        
    elif (isinstance(ds, FingerprintMetaDataset)
        #   or isinstance(ds, MNISTDataset)
          or isinstance(ds, SinusoidMetaDataset)):
        collate_fn = lambda batch : vectors_shuffled_batch_collate_fn(batch, 
                                        num_contexts_range=num_contexts_range,
                                        num_targets_range=num_targets_range)
    return collate_fn


def get_dataloader_for_dataset(ds: Dataset,
                               batch_size: int,
                               shuffle: bool=True,
                               num_contexts_range: Optional[Tuple[int]]=None,
                               num_targets_range: Optional[Tuple[int]]=None):

    if ((num_contexts_range == (None, None) and num_targets_range == (None, None)) or
        (num_contexts_range is None and num_targets_range is None)):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    elif num_contexts_range != (None, None) and num_targets_range != (None, None):
        collate_fn = get_collate_fn_for_dataset(ds=ds,
                                        num_contexts_range=num_contexts_range,
                                        num_targets_range=num_targets_range)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, 
                        collate_fn=collate_fn)
    else:
        raise ValueError('num_contexts_range and num_targets_range should ' \
                         'either both be None or both not be None')
    return dl


def train_model(model: nn.Module,
                ds_train: Dataset,
                batch_size: int,
                criterion: nn.modules.loss._Loss,
                num_epochs: int,
                optimizer: optim.Optimizer=optim.Adam,
                lr: float=0.001,
                use_scheduler: bool=True,
                submodels_to_train: Optional[Union[str,List[str]]]=None,
                num_contexts_range: Optional[Tuple[int]]=None,
                num_targets_range: Optional[Tuple[int]]=None,
                maml_regime: Optional[str]=None,   # None, first, second, anneal,
                submodels_to_maml: Optional[Union[str, List[str]]]=None,
                maml_lr: Optional[float]=0.001,
                wandb_freq: Optional[int]=None,
                release_mem_freq: Optional[int]=None,
                use_tqdm: bool=True,
                verbose: bool=True,
                in_place: bool=True,
                save_model_at_epochs: Optional[List[int]]=None,
                experiment_name: Optional[str]=None,
                experiment_id: Optional[int]=None): 
    """
    Train model (in-place by default, since this allows to stop training halfway
    and get the half-trained model).
    """
    if not in_place:
        model = deepcopy(model)

    # get device
    device = model.device

    # initialize train dataloader
    dl_train = get_dataloader_for_dataset(ds=ds_train, batch_size=batch_size,
                            shuffle=True, num_contexts_range=num_contexts_range,
                            num_targets_range=num_targets_range)

    # initialize optimizer and scheduler
    if submodels_to_train is not None:
        submodels_params = get_submodels_params(model=model,
                                                submodels_names=submodels_to_train)
        params = [{'params': v} for k,v in submodels_params.items()]
    else:
        params=model.parameters()
    optimizer = optimizer(params=params, lr=lr, weight_decay=1e-5)
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                        T_max=num_epochs,
                                                        eta_min=1e-5)

    # training loop
    num_batches_in_epoch = len(dl_train)
    for i_epoch in tqdm(range(num_epochs), disable=(not use_tqdm)):
        for i_batch, batch in enumerate(dl_train):

            # cast to float
            batch = cast_values_to_float(batch)

            # zero the parameter gradients
            optimizer.zero_grad()


            # if using MAML
            if maml_regime is not None:

                # get fine-tuned parameters
                finetuned_params = fine_tuning_step(model=model,
                                                    submodels_names=submodels_to_maml,
                                                    batch=batch,
                                                    criterion=criterion,
                                                    lr=maml_lr,
                                                    device=device,
                                                    debug=verbose,
                                                    order=maml_regime,
                                                    epoch=i_epoch)
                # compute predictions and loss with finetuned parameters
                original_params = deepcopy(model.state_dict())
                model.load_state_dict(finetuned_params)
                output = model(batch)
                # if using MAML, there will always be context and targets in each
                # batch. Hence we always use 'y_t_hat_mean', etc instead of 
                # 'y_hat_mean', etc.
                if isinstance(criterion, ReconstructionLoss):
                    loss = criterion(batch['y_t'].to(torch.device(device)), 
                                    output['y_t_hat_mean'], output['y_t_hat_var'])
                elif isinstance(criterion, nn.MSELoss):
                    loss = criterion(output['y_t_hat_mean'],
                                     batch['y_t'].to(torch.device(device)))
                elif isinstance(criterion, ReconstructionRegularizationLoss):
                    loss, rec, reg = criterion(batch['y_t'].to(torch.device(device)),
                                        output['y_t_hat_mean'],
                                        output['y_t_hat_var'],
                                        output['full_z_dist'],
                                        output['context_z_dist'])
                elif isinstance(criterion, ApproximateReconstructionRegularizationLoss):
                    loss, rec, reg = criterion(y_t=batch['y_t'].to(torch.device(device)),
                                        y_t_hat_mean=output['y_t_hat_mean'],
                                        y_t_hat_var=output['y_t_hat_var'],
                                        full_z_dist=output['full_z_dist'],
                                        context_z_dist=output['context_z_dist'],
                                        full_z_sample=output['full_z_sample'])
                else:
                    raise ValueError('criterion class is not recognized.')                   
                # compute gradients
                loss.backward()
                # clamp gradients, following MAML++
                # https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py#L335
                for name, param in model.named_parameters():
                    param.grad.data.clamp_(-10, 10)
                # restore original parameters before optimization step
                model.load_state_dict(original_params)


            # if not using MAML
            else:
                # compute predictions and loss with original parameters
                output = model(batch)
                if isinstance(criterion, ReconstructionLoss):
                    loss = criterion(y_t=batch['y_t'].to(torch.device(device)), 
                                     y_t_hat_mean=output['y_t_hat_mean'],
                                     y_t_hat_var=output['y_t_hat_var'])
                elif isinstance(criterion, ReconstructionRegularizationLoss):
                    loss, rec, reg = criterion(y_t=batch['y_t'].to(torch.device(device)),
                                     y_t_hat_mean=output['y_t_hat_mean'],
                                     y_t_hat_var=output['y_t_hat_var'],
                                     full_z_dist=output['full_z_dist'],
                                     context_z_dist=output['context_z_dist'])
                elif isinstance(criterion, ApproximateReconstructionRegularizationLoss):
                    loss, rec, reg = criterion(y_t=batch['y_t'].to(torch.device(device)),
                                     y_t_hat_mean=output['y_t_hat_mean'],
                                     y_t_hat_var=output['y_t_hat_var'],
                                     full_z_dist=output['full_z_dist'],
                                     context_z_dist=output['context_z_dist'],
                                     full_z_sample=output['full_z_sample'])
                elif isinstance(criterion, nn.MSELoss):
                    if 'y' in batch:
                        loss = criterion(output['y_hat_mean'],
                                        batch['y'].to(torch.device(device)))
                    elif 'y_t' in batch:
                        loss = criterion(output['y_t_hat_mean'],
                                        batch['y_t'].to(torch.device(device)))
                    else:
                        raise RuntimeError
                elif isinstance(criterion, nn.BCELoss):
                    if 'y' in batch:
                        loss = criterion(output['p_hat'],
                                        batch['y'].to(torch.device(device)))
                    elif 'y_t' in batch:
                        loss = criterion(output['p_t_hat'],
                                        batch['y_t'].to(torch.device(device)))
                else:
                    raise ValueError('criterion class is not recognized.') 
                # compute gradients
                loss.backward()
                
            # take optimization step
            optimizer.step()

            # reduce the outer learning rate
            if use_scheduler:
                scheduler.step()

            # log
            if wandb_freq is not None:
                if i_batch % wandb_freq == 0:
                    epoch = i_epoch + i_batch / num_batches_in_epoch
                    wandb.log({'loss': loss.item(), 'epoch': epoch})
                    if (isinstance(criterion, ReconstructionRegularizationLoss) or
                        isinstance(criterion, ApproximateReconstructionRegularizationLoss)):
                        wandb.log({'reconstruction': rec.item(), 'epoch': epoch})
                        wandb.log({'regularization': reg.item(), 'epoch': epoch})
                    if ds_train.prediction_mode == 'binary_classification':
                        if 'y' in batch:
                            phi = compute_phi(batch['y'].to(torch.device(device)),
                                        output['p_hat'])
                        elif 'y_t' in batch:
                            phi = compute_phi(batch['y_t'].to(torch.device(device)),
                                        output['p_t_hat'])
                        wandb.log({'mean_phi': np.mean(phi), 'epoch': epoch})

            # freed space every `release_mem_freq` batches to avoid out-of-memory errors
            if release_mem_freq is not None:
                if i_batch % release_mem_freq == 0:
                    del output, batch
                    gc.collect()
                    torch.cuda.empty_cache()

        if save_model_at_epochs is not None:
            # in order to save the model, we need both the experiment name and id
            assert experiment_name is not None
            assert experiment_id is not None
            if i_epoch in save_model_at_epochs:
                save_model(model, experiment_name, f'{experiment_id}_{i_epoch}')
    
    if not in_place:
        return model


def set_subset_metadataset_attributes(subset_ds: Dataset, fullitem: dict,
                    datapoint_indices: Optional[np.ndarray]=None) -> None:
    if isinstance(subset_ds, MolecularGraphMetaDataset):
        if datapoint_indices is None:
            subset_ds.smiles = fullitem['smiles'][0]
            subset_ds.atoms = fullitem['atoms'][0]
            subset_ds.atoms_mask = fullitem['atoms_mask'][0]
            subset_ds.bonds = fullitem['bonds'][0]
            subset_ds.adjacencies = fullitem['adjacencies'][0]
            subset_ds.y = fullitem['y']
            subset_ds.functions = fullitem['functions']
            subset_ds.function_names = [fullitem['function_name']]
            subset_ds.mol_numeric_index = fullitem['mol_numeric_index']
            subset_ds.mask_labeled = fullitem['mask_labeled']
        else:
            subset_ds.smiles = fullitem['smiles'][0][datapoint_indices]
            subset_ds.atoms = fullitem['atoms'][0][datapoint_indices]
            subset_ds.atoms_mask = fullitem['atoms_mask'][0][datapoint_indices]
            subset_ds.bonds = fullitem['bonds'][0][datapoint_indices]
            subset_ds.adjacencies = fullitem['adjacencies'][0][datapoint_indices]
            subset_ds.y = fullitem['y'][:, datapoint_indices]
            subset_ds.functions = fullitem['functions'].iloc[datapoint_indices]
            subset_ds.function_names = [fullitem['function_name']]
            subset_ds.reset_mol_numeric_index()
            subset_ds.reset_mask_labeled()
    else:
        raise NotImplementedError

# TODO make this a dataset method!
def get_subset_metadataset(ds: Dataset, function_name: str,
                    datapoint_indices: Optional[np.ndarray]=None) -> Dataset:
    fullitem = ds.getfullitembyname(function_name)
    # initialize with dummy smiles to avoid re-featurizing every time
    dummy_smiles = pd.Series(['CCC'])
    if ds.num_functions() > 1:
        if isinstance(ds, MolecularGraphMetaDataset):
            # initialize dummy dataset with dummy smiles
            subset_ds = MolecularGraphMetaDataset(smiles=dummy_smiles,
                            functions=ds.functions,
                            num_contexts_range=ds.num_contexts_range,
                            num_targets_range=ds.num_targets_range,
                            split_mode=ds.split_mode,
                            max_num_atoms=ds.featurizer.max_num_atoms,
                            use_chirality=ds.featurizer.use_chirality,
                            use_stereochemistry=ds.featurizer.use_stereochemistry,
                            featurization_progress_bar=False,
                            y_len=ds.y_len)
            # populate attributes with the values of fullitem
            # (fullitem adds an additional dimension at the beginning so that
            # it looks is if it came from a batch; now we need to remove this
            # extra dimension when applicable in order to have the right dimensions
            # in the dataset)
            set_subset_metadataset_attributes(subset_ds=subset_ds,
                                        fullitem=fullitem,
                                        datapoint_indices=datapoint_indices)

        elif isinstance(ds, FingerprintMetaDataset):
            if datapoint_indices is not None:
                raise NotImplementedError
            # initialize dummy dataset with dummy smiles
            subset_ds = FingerprintMetaDataset(smiles=dummy_smiles,
                            functions=ds.functions,
                            num_contexts_range=ds.num_contexts_range,
                            num_targets_range=ds.num_targets_range,
                            split_mode=ds.split_mode,
                            featurization_progress_bar=False)
            # populate attributes with the values of fullitem
            # (fullitem adds an additional dimension at the beginning so that
            # it looks is if it came from a batch; now we need to remove this
            # extra dimension when applicable in order to have the right dimensions
            # in the dataset)
            subset_ds.smiles = fullitem['smiles'][0]
            subset_ds.x = fullitem['x'][0]
            subset_ds.y = fullitem['y']
            subset_ds.functions = fullitem['functions']
            subset_ds.function_names = [fullitem['function_name']]
            subset_ds.mol_numeric_index = fullitem['mol_numeric_index']
            subset_ds.mask_labeled = fullitem['mask_labeled']

        else:
            raise TypeError('Dataset should be of type FingerprintMetadataset, '
                            'MolecularGraphMetadataset or SinusoidMetadataset.')
    else:
        subset_ds = ds
    return subset_ds


def get_subset_sparse_metadataset(ds: Dataset,
                       function_name: str):
    '''
    Given a metadataset (FingerprintMetaDataset, MolecularGraphMetaDataset)
    containing many functions and a function name, return a metadataset of the
    same type with just that single function.
    '''
    subset_ds = deepcopy(ds)
    fullitem = deepcopy(ds.getfullitembyname(function_name, with_attributes=True))
    # function names (a list)
    subset_ds.function_names = [function_name]
    # functions (a dataframe)
    subset_ds.functions = fullitem['functions'][[function_name]]
    # y values (a tensor)
    subset_ds.y = fullitem['y']
    # smiles (a pandas series)
    subset_ds.smiles = fullitem['smiles'][0]
    # mol_numeric_index
    subset_ds.mol_numeric_index = fullitem['mol_numeric_index']
    # mol_mask_labeled
    subset_ds.mol_mask_labeled = fullitem['mol_mask_labeled']

    return subset_ds


def get_subset_simpledataset(ds: Dataset,
                       function_name: str):
    '''
    Given a metadataset (FingerprintMetaDataset or MolecularGraphMetaDataset)
    containing many functions and a function name, return a simpledataset 
    (FingerprintSimpleDataset or MolecularGraphSimpleDataset) of the same input
    type (fingerprints or graphs) with just that single function.
    '''

    assert isinstance(function_name, str)

    if isinstance(ds, FingerprintMetaDataset):
        subset_ds = FingerprintSimpleDataset(smiles=ds.smiles[:5],
                                             functions=ds.functions[[function_name]],
                                             progress_bar=False,
                                             prediction_mode=ds.prediction_mode)
        subset_ds.smiles = deepcopy(ds.smiles)
        subset_ds.x = deepcopy(ds.x)

    elif isinstance(ds, MolecularGraphMetaDataset):
        subset_ds = MolecularGraphSimpleDataset(smiles=ds.smiles[:5],
                                                functions=ds.functions[[function_name]],
                                                progress_bar=False,
                                                prediction_mode=ds.prediction_mode)
        subset_ds.smiles = deepcopy(ds.smiles)
        subset_ds.atoms = deepcopy(ds.atoms)
        subset_ds.atoms_mask = deepcopy(ds.atoms_mask)
        subset_ds.bonds = deepcopy(ds.bonds)
        subset_ds.adjacencies = deepcopy(ds.adjacencies)

    elif isinstance(ds, SinusoidMetaDataset):
        subset_ds = deepcopy(ds)
        subset_ds.function_names = [function_name]
        subset_ds.x = ds.getfullitembyname(function_name)['x']
        subset_ds.y = ds.getfullitembyname(function_name)['y']

    return subset_ds


def epochs_to_effective_epochs(num_epochs: int, num_datapoints: int,
                               mean_subset_size: int):
    """
    Get the number of effective epochs, i.e. the number of times that
    a datapoint is seen on average during training.
    """
    return num_epochs * (mean_subset_size / num_datapoints)

def effective_epochs_to_epochs(num_effective_epochs: int, num_datapoints: int,
                               num_targets_range: Tuple[int]) -> int:
    """
    Get the number of epochs, i.e. the number of times that we iterate
    over the dataset, given the number of effective epochs, i.e. the
    number of times that a datapoint is seen on average during training.
    """
    min_num_targets = num_targets_range[0]
    max_num_targets = num_targets_range[1]
    mean_subset_size = (max_num_targets + min_num_targets) / 2
    return int(num_effective_epochs * (num_datapoints / mean_subset_size))


def choose_criterion(criterion_name: Optional[str]=None):
    if criterion_name == 'ReconstructionLoss':
        criterion = ReconstructionLoss()
    elif criterion_name == 'ReconstructionRegularizationLoss':
        criterion = ReconstructionRegularizationLoss()
    elif criterion_name == 'ApproximateReconstructionRegularizationLoss':
        criterion = ApproximateReconstructionRegularizationLoss()
    elif criterion_name == 'MSELoss':
        criterion = nn.MSELoss()
    elif criterion_name == 'BCELoss':
        criterion = nn.BCELoss()
    elif criterion_name is None:
        criterion = None
    else:
        ValueError('Incorrect criterion_name')
    return criterion

def choose_criterion_from_model(model: nn.Module):
    if isinstance(model, MolecularGraphAttentionCNP):
        criterion = ReconstructionLoss()
    else:
        raise NotImplementedError
    return criterion


def choose_input_type(model_name: str) -> str:

    if model_name == 'FullyConnectedCNP':
        input_type = 'fingerprints'
    elif model_name == 'SiderFullyConnectedCNP':
        input_type = 'fingerprints'
    elif model_name == 'LincsFullyConnectedCNP':
        input_type = 'fingerprints'
    elif model_name == 'SiderAttentionCNP':
        input_type = 'fingerprints'
    elif model_name == 'FullyConnectedLNP':
        input_type = 'fingerprints'
    elif model_name == 'ApproximateGaussianProcessOnFingerprints':
        input_type = 'fingerprints'
    elif model_name == 'ExactGaussianProcessOnFingerprints':
        input_type = 'fingerprints'
    elif model_name == 'ExactGaussianProcessOnFingerprintsARD':
        input_type = 'fingerprints'
    elif model_name == 'ExactGaussianProcessOnCountFingerprints':
        input_type = 'count_fingerprints'
    elif model_name == 'ExactGaussianProcessOnCountFingerprintsARD':
        input_type = 'count_fingerprints'
    elif model_name == 'FSS':
        input_type = 'fingerprints'
    elif model_name == 'KNN':
        input_type = 'fingerprints'
    elif model_name == 'RF':
        input_type = 'fingerprints'
    elif model_name == 'XGB':
        input_type = 'fingerprints'
    elif model_name == 'Dummy':
        input_type = 'fingerprints'

    elif model_name == 'MolecularGraphAttentionNN':
        input_type = 'graphs'
    elif model_name == 'MolecularGraphAttentionCNP':
        input_type = 'graphs'
    elif model_name == 'LincsMolecularGraphAttentionCNP':
        input_type = 'graphs'
    elif model_name == 'SiderMolecularGraphAttentionCNP':
        input_type = 'graphs'
    elif model_name == 'MolecularGraphAttentionLNP':
        input_type = 'graphs'
    else:
        raise ValueError('Incorrect model_name')

    return input_type

def choose_function_strings(function_splits_type: str):
    if function_splits_type == 'adaptation':
        function_train_str = 'adaptrain'
        function_test_str = 'adaptest'
    elif function_splits_type == 'default':
        function_train_str = 'train'
        function_test_str = 'test'
    elif function_splits_type == 'all':
        function_train_str = ['train', 'test']
        function_test_str = ['train', 'test']
    else:
        raise ValueError('Incorrect value or combination of score_type and  of splits_type')
    return function_train_str, function_test_str


def choose_datapoint_strings(splits_type: str):
    if splits_type == 'random':
        datapoint_train_str = 'random_train'
        datapoint_test_str = 'random_test'
    else:
        raise NotImplementedError
    # else:
    #     raise ValueError('Incorrect value of splits_type')
    return datapoint_train_str, datapoint_test_str


def choose_model(model_name: str, example_dataset: Dataset,
                 device: str) -> nn.Module:

    num_atom_V_features = fixed_params['num_atom_V_features']
    num_bond_V_features = fixed_params['num_bond_V_features']
    num_QK_features = fixed_params['num_QK_features']
    mp_iterations = fixed_params['mp_iterations']
    use_layernorm = fixed_params['use_layernorm']
    r_len = fixed_params['r_len']
    z_len = fixed_params['z_len']
    prediction_mode = example_dataset.prediction_mode

    if model_name == 'FullyConnectedNN':
        if isinstance(example_dataset, SinusoidMetaDataset):
            model = FullyConnectedNN(x_len=1, y_len=1,
                                     linear_1_len=40, linear_2_len=40,
                                     linear_3_len=None,
                                     use_layernorm=False,
                                     leaky_relu=False,
                                     device=device)
        else:
            raise NotImplementedError

    elif model_name == 'FullyConnectedCNP':
        if isinstance(example_dataset, FingerprintMetaDataset):
            model = FullyConnectedCNP(r_len=r_len, y_len=1,
                                      use_layernorm=use_layernorm,
                                      device=device,
                                      prediction_mode=prediction_mode)
        elif isinstance(example_dataset, SinusoidMetaDataset):
            model = FullyConnectedCNP(r_len=r_len, x_len=1, y_len=1,
                                      use_layernorm=use_layernorm,
                                      device=device,
                                      prediction_mode=prediction_mode)

    elif model_name == 'SiderFullyConnectedCNP':
        model = FullyConnectedCNP(r_len=250, y_len=1,
                                  encoder_1_len=2000, encoder_2_len=1000,
                                  decoder_1_len=2000, decoder_2_len=1000,
                                  decoder_3_len=1000, decoder_4_len=1000,
                                    use_layernorm=use_layernorm,
                                    device=device,
                                    prediction_mode='binary_classification')

    elif model_name == 'LincsFullyConnectedCNP':
        model = FullyConnectedCNP(r_len=250, y_len=1,
                                  encoder_1_len=1000, encoder_2_len=500,
                                  decoder_1_len=1000, decoder_2_len=500,
                                  decoder_3_len=100, decoder_4_len=50,
                                    use_layernorm=use_layernorm,
                                    device=device,
                                    prediction_mode='regression')

    elif model_name == 'SiderAttentionCNP':
        model = AttentionCNP(r_len=250, y_len=1, QK_len=100,
                                  encoder_1_len=2000, encoder_2_len=1000,
                                  decoder_1_len=2000, decoder_2_len=1000,
                                  decoder_3_len=1000, decoder_4_len=1000,
                                    use_layernorm=use_layernorm,
                                    device=device,
                                    prediction_mode='binary_classification')


    elif model_name == 'FullyConnectedLNP':
        if isinstance(example_dataset, FingerprintMetaDataset):
            model = FullyConnectedLNP(r_len=r_len, z_len=z_len, y_len=1,
                                      use_layernorm=use_layernorm,
                                      device=device,
                                      prediction_mode=prediction_mode)
        elif isinstance(example_dataset, SinusoidMetaDataset):
            raise NotImplementedError

    elif model_name == 'MolecularGraphAttentionNN':
        num_atom_features = example_dataset.featurizer.num_atom_features
        num_bond_features = example_dataset.featurizer.num_bond_features
        if (isinstance(example_dataset, FingerprintSimpleDataset) or 
            isinstance(example_dataset, MolecularGraphSimpleDataset)):
            y_len = example_dataset.num_functions()
        else:
            y_len = 1
        model = MolecularGraphAttentionNN(num_atom_features=num_atom_features,
                                num_bond_features=num_bond_features,
                                num_atom_V_features=num_atom_V_features,
                                num_bond_V_features=num_bond_V_features,
                                num_QK_features=num_QK_features,
                                mp_iterations=mp_iterations,
                                y_len=y_len, use_layernorm=use_layernorm,
                                prediction_mode=prediction_mode,
                                device=device)
  
    elif model_name == 'MolecularGraphAttentionCNP':
        num_atom_features = example_dataset.featurizer.num_atom_features
        num_bond_features = example_dataset.featurizer.num_bond_features
        model = MolecularGraphAttentionCNP(num_atom_features=num_atom_features,
                                    num_bond_features=num_bond_features,
                                    num_atom_V_features=num_atom_V_features,
                                    num_bond_V_features=num_bond_V_features,
                                    num_QK_features=num_QK_features,
                                    mp_iterations=mp_iterations,
                                    r_len=r_len, y_len=1,
                                    use_layernorm=use_layernorm,
                                    device=device,
                                    prediction_mode=prediction_mode)

    elif model_name == 'LincsMolecularGraphAttentionCNP':
        num_atom_features = example_dataset.featurizer.num_atom_features
        num_bond_features = example_dataset.featurizer.num_bond_features
        model = MolecularGraphAttentionCNP(num_atom_features=num_atom_features,
                                    num_bond_features=num_bond_features,
                                    num_atom_V_features=1000,
                                    num_bond_V_features=1000,
                                    num_QK_features=50,
                                    mp_iterations=mp_iterations,
                                    r_len=1024, y_len=1,
                                    use_layernorm=use_layernorm,
                                    device=device,
                                    prediction_mode=prediction_mode,
                                    lincs_architecture=True)

    elif model_name == 'SiderMolecularGraphAttentionCNP':
        num_atom_features = example_dataset.featurizer.num_atom_features
        num_bond_features = example_dataset.featurizer.num_bond_features
        model = MolecularGraphAttentionCNP(num_atom_features=num_atom_features,
                                    num_bond_features=num_bond_features,
                                    num_atom_V_features=100,
                                    num_bond_V_features=100,
                                    num_QK_features=50,
                                    mp_iterations=mp_iterations,
                                    r_len=250, y_len=1,
                                    use_layernorm=use_layernorm,
                                    device=device,
                                    prediction_mode='binary_classification')

    elif model_name == 'MolecularGraphAttentionLNP':
        num_atom_features = example_dataset.featurizer.num_atom_features
        num_bond_features = example_dataset.featurizer.num_bond_features
        model = MolecularGraphAttentionLNP(num_atom_features=num_atom_features,
                                    num_bond_features=num_bond_features,
                                    num_atom_V_features=num_atom_V_features,
                                    num_bond_V_features=num_bond_V_features,
                                    num_QK_features=num_QK_features,
                                    mp_iterations=mp_iterations,
                                    r_len=r_len, z_len=r_len, y_len=1,
                                    use_layernorm=use_layernorm,
                                    device=device,
                                    prediction_mode=prediction_mode)

    elif model_name == 'ApproximateGaussianProcessOnFingerprints':
        model = ApproximateGaussianProcessOnFingerprints(basekernel=TanimotoKernel,
                                      num_inducing_points=500,
                                      device=device)

    elif model_name == 'ExactGaussianProcessOnFingerprints':
        model = ExactGaussianProcessOnFingerprints(basekernel=TanimotoKernel,
                                      device=device)

    elif model_name == 'ExactGaussianProcessOnFingerprintsARD':
        model = ExactGaussianProcessOnFingerprints(basekernel=TanimotoKernel,
                                      ard_num_dims=1024,  # 1024 is the length of our fingerprints
                                      device=device)

    elif model_name == 'ExactGaussianProcessOnCountFingerprints':
        model = ExactGaussianProcessOnFingerprints(device=device)


    elif model_name == 'ExactGaussianProcessOnCountFingerprintsARD':
        model = ExactGaussianProcessOnFingerprints(device=device,
                                                   ard_num_dims=1024)

    elif model_name == 'FSS':
        model = FSS(prediction_mode=prediction_mode)
        return model

    elif model_name == 'KNN':
        model = KNN(prediction_mode=prediction_mode)
        return model

    elif model_name == 'RF':
        model = RF(prediction_mode=prediction_mode)
        return model

    elif model_name == 'XGB':
        model = XGB(prediction_mode=prediction_mode)
        return model

    elif model_name == 'Dummy':
        model = Dummy(prediction_mode=prediction_mode)
        return model

    else:
        raise ValueError('Incorrect model_name')

    return model



def check_same_names(state_dict_1: OrderedDict, state_dict_2: OrderedDict) -> None:
    """
    Check that two Pytorch state dictionaries have the same names.
    """
    names_1, names_2 = state_dict_1.keys(), state_dict_2.keys()
    if len(names_1) != len(names_2):
        raise RuntimeError('State dictionaries have different length.')

    for name_1, name_2 in zip(names_1, names_2):
        if name_1 != name_2:
            raise RuntimeError('State dictionaries differ in names ' \
                               f'{name_1}, {name_2}.')


def check_same_shape(state_dict_1: OrderedDict, state_dict_2: OrderedDict) -> list:
    """
    Check that the shape of the parameters in the state dictionaries is the same.
    """
    differing_params_names = []
    for pair_1, pair_2 in zip(state_dict_1.items(), state_dict_2.items()):
        name_1, param_1 = pair_1
        name_2, param_2 = pair_2
        if name_1 == name_2:
            if param_1.shape != param_2.shape:
                print(f'Shape of parameter {name_1} in state dictionaries differs ' \
                      f'- {param_1.shape} vs {param_2.shape}')
                differing_params_names.append(name_1)
        else:
            raise RuntimeError('State dictionaries differ in names ' \
                               f'{name_1}, {name_2}.')
    return differing_params_names


def load_model(model: nn.Module, models_dir: str,
               repetition: int, device: str) -> nn.Module:
    """
    Load saved state dictionary (specified by a directory and a repetition number)
    to a model.

    The saved state dictionary usually corresponds to the model state dictionary
    exactly, but sometimes we may want to load a state dictionary that differs
    in the last layer only.
    
    from a slightly
    different model. For example, we may have pretrained a model using multi-task
    learning, so the model to load may have several heads, and we may want to fine-tune
    it on a single task, so the model we'll use has a single head.

    However, we assume that the state dict saved and the state dict of the model
    have the following pro the same number of parameter
    """
    saved_path = RESULTS_DIR / models_dir / f'model_{repetition}.pt'
    target_state_dict = torch.load(saved_path, map_location=device)
    origin_state_dict = deepcopy(model.state_dict())
    state_dict = origin_state_dict
    # check correspondence between origin and target state dictionaries
    check_same_names(origin_state_dict, target_state_dict)
    differing_params_names = check_same_shape(origin_state_dict, target_state_dict)
    # combine state dicts by taking
    # - corresponding params from the target state dict, and 
    # - differing params from the origin state dict
    for name in state_dict.keys():
        if name not in differing_params_names:
            state_dict[name] = target_state_dict[name]
    # load 
    model.load_state_dict(state_dict)
    return model


def datasets_to_metabatch(context_ds: Dataset, target_ds: Dataset) -> dict:
    context_batch = context_ds.getfullitem(0)
    context_batch.pop('function_name')   # TODO generalize to function_name or function_names
    target_batch = target_ds.getfullitem(0)
    target_batch.pop('function_name')
    batch = {}
    batch.update(add_suffix_to_keys(context_batch,'_c'))
    batch.update(add_suffix_to_keys(target_batch, '_t'))
    batch = cast_values_to_float(batch)
    return batch

def dataset_to_simplebatch(ds: Dataset) -> dict:
    batch = ds.getfullitem(0)
    batch.pop('function_name')   # TODO generalize to function_name or function_names
    batch = cast_values_to_float(batch)
    batch = {k: v[0] for k,v in batch.items()}
    return batch


def predict_dataset_with_simplemodel(model: nn.Module, ds: Dataset):
    """
    Function to evaluate normal models that don't differentiate
    between contexts and targets, and which are not finetuned
    for specific functions as part of the evaluation.

    ds should contain a single function
    """

    # create dataloader
    dl = get_dataloader_for_dataset(ds=ds, batch_size=32,
                                         shuffle=False)

    # compute predictions for entire dataset, one batch at a time
    all_output = {}

    for i, batch in enumerate(dl):
        with torch.no_grad():
            output = model(batch)
        if i == 0:
            all_output = {k:[v.float().cpu()] for k,v in output.items()}
        else:
            for k,v in output.items():
                all_output[k].append(v.float().cpu())

        # y_hat_mean.append(output['y_hat_mean'].float().cpu())  

    # y_hat_mean = torch.cat(y_hat_mean)

    for k,v in all_output.items():
        all_output[k] = torch.cat(v)

    return all_output


def print_num_labeled_datapoints_message(stat_ftrain_dtrain: Tuple[int],
                                         stat_ftrain_dtest: Tuple[int],
                                         stat_ftest_dtrain: Tuple[int],
                                         stat_ftest_dtest: Tuple[int]):
    stat_message = f"""
    ftrain_dtrain {stat_ftrain_dtrain}
    ftrain_dtest  {stat_ftrain_dtest}
    ftest_dtrain  {stat_ftest_dtrain}
    ftest_dtest   {stat_ftest_dtest}
    """
    print('Min and max number of labeled datapoint per function in dataset is')
    print(stat_message)
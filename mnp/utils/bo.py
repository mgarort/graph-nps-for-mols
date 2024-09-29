import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Callable, Optional, Union, List
import numpy as np
from tqdm import tqdm
from mnp.utils.scripting import (train_model, choose_criterion_from_model,
                                 effective_epochs_to_epochs, get_subset_metadataset)
from mnp.models.gp import (ApproximateGaussianProcessOnFingerprints,
                           ExactGaussianProcessOnFingerprints)
from copy import deepcopy


def lower_confidence_bound(y_hat_mean, y_hat_var, explored_indices, beta=1, 
                           num_to_select=1, **_):
    """
    Compute the lower confidence bound or LCB (like upper confidence bound or
    UCB, but for minimization) and identify the most promising datapoints that
    have not been explored yet.

    Args:
        - y_hat_mean: mean of the predictive distribution at each datapoint.
        - y_hat_var: variance of the predictive distribution at each datapoint.
        - explored_indices: indices of the datapoints in y_hat_mean and y_hat_var
            that have already been explored/measured.
        - beta: coefficient to weigh the variance in the computation of LCB.
        - num_to_select: number of most promising datapoints to select.
        - _** : for packing extra arguments given as a dictionary.

    Returns:
        The indices of the most promising unexplored datapoints accoding to
        the LCB.
    """
    # Make sure that the predictions are one-dimensional
    assert len(y_hat_mean.shape) == 1
    assert len(y_hat_var.shape) == 1
    # Compute LCB
    lcb = y_hat_mean - beta * torch.sqrt(y_hat_var)
    # Rank the datapoints among those not already selected
    n_datapoints = len(y_hat_mean)
    all_indices = list(range(n_datapoints))
    unexplored_indices = np.array([elem for elem in all_indices
                                   if elem not in explored_indices])
    # Get most promising indices within unexplored set, counting within unexplored
    num_to_select = min(num_to_select, len(unexplored_indices))
    _, best_unexplored_indices = lcb[unexplored_indices].topk(num_to_select,
                                                              largest=False)
    # Get most promising indices within unexplored set, counting within whole set
    best_unexplored_indices = unexplored_indices[best_unexplored_indices.detach().cpu().numpy()]
    return best_unexplored_indices


def greedy_selection(y_hat_mean, explored_indices, num_to_select=1, **_):
    """
    Identify the most promising datapoints that have not been explored yet,
    according to the mean prediction.

    Args:
        - y_hat_mean: mean of the predictive distribution at each datapoint.
        - y_hat_var: not used, but added for compatibility with lower_confidence_bound.
        - explored_indices: indices of the datapoints in y_hat_mean that have
          already been explored/measured.
        - num_to_select: number of most promising datapoints to select.
        - **_ : for packing extra arguments given as a dictionary.

    Returns:
        The indices of the most promising unexplored datapoints accoding to
        the mean prediction
    """
    # Make sure that the predictions are one-dimensional
    assert len(y_hat_mean.shape) == 1
    # Rank the datapoints among those not already selected
    n_datapoints = len(y_hat_mean)
    all_indices = list(range(n_datapoints))
    unexplored_indices = np.array([elem for elem in all_indices
                                   if elem not in explored_indices])
    # Get most promising indices within unexplored set, counting within unexplored
    num_to_select = min(num_to_select, len(unexplored_indices))
    _, best_unexplored_indices = y_hat_mean[unexplored_indices].topk(num_to_select,
                                                                     largest=False)
    # Get most promising indices within unexplored set, counting within whole set
    best_unexplored_indices = unexplored_indices[best_unexplored_indices.detach().cpu().numpy()]
    return best_unexplored_indices


def random_selection(y_hat_mean, explored_indices, num_to_select=1, random_seed=0, **_):
    """
    Select points that have not been explored yet at random.

    Args:
        - y_hat_mean: mean of the predictive distribution at each datapoint
          (used only for getting the number of total datapoints).
        - explored_indices: indices of the datapoints in y_hat_mean that have
          already been explored/measured.
        - num_to_select: number of most promising datapoints to select.
        - **_ : for packing extra arguments given as a dictionary.

    Returns:
        Random indices that have not been explored yet.
    """
    # Make sure that the predictions are one-dimensional
    assert len(y_hat_mean.shape) == 1
    # Get unexplored indices
    n_datapoints = len(y_hat_mean)
    np.random.seed(random_seed)
    all_shuffled_indices = np.random.permutation(n_datapoints)
    unexplored_indices = np.array([elem for elem in all_shuffled_indices
                                        if elem not in explored_indices])
    # Get random unexplored indices
    next_random_unexplored_indices = unexplored_indices[:num_to_select]
    return next_random_unexplored_indices


def get_bo_statistics(y, explored_indices, last_explored_indices) -> tuple:
    """
    At each iteration of Bayesian optimization, we could record:
    - the true y value and index of the top 1st molecule so far
    - the true y value and index of the top 25th molecule so far
    - the true y value and index of the best molecule(s) within the last explored
      (NOTE the index of the molecule(s) selected at the last step were received
      as a parameter, so this would not be new information, but we can return it
      anyway for consistency.)
    """
    # Make sure that the true valyes y are one-dimensional
    assert len(y.shape) == 1
    best = y[explored_indices].topk(1, largest=False)
    best_value, best_relative_index = best[0].item(), best[1].item()
    best_index = explored_indices[best_relative_index]
    # if we have explored less than 25 molecules, then choose the best within
    # those <25 molecules so far
    top_num = min(25, len(explored_indices))
    best_25th = y[explored_indices].topk(top_num, largest=False)
    best_25th_value, best_25th_relative_index = best_25th[0][-1].item(), best_25th[1][-1].item()
    best_25th_index = explored_indices[best_25th_relative_index]
    # best value and index within the last explored
    best_in_last = y[last_explored_indices].sort()
    best_in_last_value, best_in_last_relative_index = (best_in_last[0][0].item(),
                                                       best_in_last[1][0].item())
    best_in_last_index = last_explored_indices[best_in_last_relative_index]
    return (best_value, best_index,
            best_25th_value, best_25th_index,
            best_in_last_value, best_in_last_index)


def get_random_starting_indices(num_starting_samples, num_total, random_seed):
    np.random.seed(random_seed)
    all_indices = np.random.permutation(num_total)
    starting_indices = all_indices[:num_starting_samples]
    return starting_indices


def run_bo_trajectory(model: nn.Module,
                      ds: Dataset, function_name: str,
                      acquisition_function: Callable,
                      num_starting_indices: int,
                      num_selected_per_iter: int,
                      random_seed: int,
                      budget: Optional[int]=None,
                      verbose_freq: Union[None,int]=50,
                      num_longtuning_effective_epochs: int=0,
                      submodels_to_tune: Optional[Union[str, List[str]]]=None,
                      lr: Optional[float]=None,
                      min_num_longtuning_datapoints: Optional[int]=None):
    """
    - acquisition_function: a function such as lower_confidence_bound
    - molecular_function should be a dictionary with different keys for atoms, bonds,
        etc. In each key, the value has dimensions (1, num_datapoints, other_dimensions)
        (where the 1 is because it's just 1 function).
    """
    molecular_function = ds.getfullitembyname(function_name)
    _ = molecular_function.pop('function_name')
    _ = molecular_function.pop('functions')
    _ = molecular_function.pop('mask_labeled')
    _ = molecular_function.pop('mol_numeric_index')
    num_total = ds.num_datapoints(ds.get_idx_from_name(function_name))
    results = {}

    # initialize the trajectory as if we had carried out one iteration
    starting_indices = get_random_starting_indices(num_starting_samples=num_starting_indices,
                                                num_total=num_total,
                                                random_seed=random_seed)
    explored_indices = starting_indices
    next_index = starting_indices

    # trajectory loop
    # (previously we stopped the trajectory if we found the best value or the
    # 25th best value. However, now we are also recording the best value within
    # the last explored/selected molecules. Therefore, it is informative to
    # finish the exploration loop even if we have already found the best and the
    # 25th best values)
    with tqdm(total=budget, desc="Progress") as pbar:

        # the budget is the number of function evaluations / molecules we observe
        # during the experiment. In order to explore this number of molecules, we
        # have the following workflow:
        # - we start iteration 0 with some molecules selected at random. We log the
        #   statistics from those molecules, even though they were selected at
        #   random (just for info).
        # - at iteration n-th, we log the statistics from the molecules selected
        #   at iteration (n-1)-th, and select the molecules to be used in iteration
        #   (n+1)-th
        # - at iteration budget-th, we log the statistics from iteration (budget-1)-th,
        #   and select new molecules. However, these molecules will never be logged.
        # Therefore, at the end we have logs for:
        # - Iteration 0: info about the random molecules we started with.
        # - Iterations 1-st to budget-th: info about the molecules selected in `budget` iterations.
        # - Iteration (budget+1)-th: never logged so no info.
        for i in range(0,budget+1,num_selected_per_iter):

            # log the results from the previous iteration
            bo_statistics = get_bo_statistics(molecular_function['y'].squeeze(),
                                                        explored_indices,
                                                        next_index)
            best_value, best_index = bo_statistics[0], bo_statistics[1]
            best_25th_value, best_25th_index = bo_statistics[2], bo_statistics[3]
            best_in_last_value, best_in_last_index = bo_statistics[4], bo_statistics[5]
            results[f'best_value_{i}'] = best_value
            results[f'best_index_{i}'] = best_index
            results[f'best_25th_value_{i}'] = best_25th_value
            results[f'best_25th_index_{i}'] = best_25th_index
            results[f'best_in_last_value_{i}'] = best_in_last_value
            results[f'best_in_last_index_{i}'] = best_in_last_index

            # print information if verbose
            if verbose_freq is not None and i % verbose_freq == 0:
                print('Top 1 so far', best_value)
                print('Top 25 so far', best_25th_value)
                if num_longtuning_effective_epochs > 0:
                    if i < min_num_longtuning_datapoints:
                        print('Not yet reached # ')


            # make new predictions to select next points
            inputs = {}
            for k,v in molecular_function.items():
                inputs[k + '_c'] = molecular_function[k][:,explored_indices]
                inputs[k + '_t'] = molecular_function[k]

            # TODO modify so that you can take one fine-tuning step, to compare the graph CNP
            #      with the graph CNP + MAML

            # if the acquisition is random selection, the output of
            # the model does not matter, so we create dummy output
            if acquisition_function == random_selection:
                outputs = {'y_t_hat_mean': torch.zeros(1,num_total),
                        'y_t_hat_var': torch.zeros(1,num_total)}
            else:
                # if the model is a GPytorch GP, we need to train it, so
                # we need gradients
                if (isinstance(model, ApproximateGaussianProcessOnFingerprints) or
                    isinstance(model, ExactGaussianProcessOnFingerprints)):
                    outputs = model(inputs)

                # if the model is not a GPytorch GP, it is already trained,
                # so we don't need gradients
                else:
                    # save original params in case we will finetune
                    original_params = deepcopy(model.state_dict())
                    # finetune if desired
                    if num_longtuning_effective_epochs > 0 and i >= min_num_longtuning_datapoints:
                        train_ds = get_subset_metadataset(ds=ds,
                                                function_name=function_name,
                                                datapoint_indices=explored_indices)      
                        # calculate number of effective epochs
                        num_longtuning_epochs = effective_epochs_to_epochs(
                            num_effective_epochs=num_longtuning_effective_epochs,
                            num_datapoints=train_ds.num_datapoints(0),
                            num_targets_range=train_ds.num_targets_range)
                        train_model(model=model,
                                    ds_train=train_ds,
                                    batch_size=1,
                                    criterion=choose_criterion_from_model(model),
                                    num_epochs=num_longtuning_epochs,
                                    lr=lr,
                                    submodels_to_train=submodels_to_tune,
                                    num_contexts_range=train_ds.num_contexts_range,
                                    num_targets_range=train_ds.num_targets_range,
                                    release_mem_freq=10,
                                    use_tqdm=False,)
                    with torch.no_grad():
                        outputs = model(inputs)
                    # reload original params in case we have finetuned
                    model.load_state_dict(original_params)


            # Select next points
            acquisition_params = {'y_hat_mean': outputs['y_t_hat_mean'].squeeze(),
                                'y_hat_var': outputs['y_t_hat_var'].squeeze(),
                                'explored_indices': explored_indices,
                                'num_to_select': num_selected_per_iter,
                                'random_seed': random_seed}
            next_index = acquisition_function(**acquisition_params)
            explored_indices = np.append(explored_indices, next_index)

            # manually update the progress bar
            pbar.update(num_selected_per_iter)

    # log the smiles of the best molecule and the 25th best molecule of the
    # whole experiment
    final_best_index = results[f'best_index_{budget}']
    final_best_25th_index = results[f'best_25th_index_{budget}']
    results['final_best_smiles'] = molecular_function['smiles'][0, final_best_index]
    results['final_best_25th_smiles'] = molecular_function['smiles'][0, final_best_25th_index]

    # TODO log top1 smile and top 25 smile at the end of the experiment
    
    return results
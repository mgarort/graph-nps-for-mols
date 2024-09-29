import numpy as np
import torch


def vectors_shuffled_batch_collate_fn(shuffled_batch,
                                           num_contexts_range, num_targets_range):
    '''
    This function works together with the method get_ct_indices_for_one_function
    and the functions get_non_overlapping_ct_indices or get_overlapping_ct_indices
    in order to select batches of datapoints such that:

    - Datapoints selected are not the same across functions.
    - Within the same batch, all functions have the same number of datapoints.
    - Different batches can have different numbers of datapoints.
    - Within each function, the context datapoints and the target datapoints could
      be overlapping or non-overlapping, depending on our preferences.

    The way this works is the following:

    - First, get_ct_indices_for_one_function shuffles and __get_item__ shuffle
      the datapoints of each function so that each function in the batch presents
      datapoints in a different order.

    - Second, get_non_overlapping_ct_indices or get_overlapping_ct_indices get
      context and target indices that are non-intersecting or intersecting.

    - Third, the collate functions select the context and target datapoints with
      these indices.
    '''
    # Re-shuffle indices of contexts and targets
    # (NOTE datapoints are already shuffled for each individual function so that we
    # don't choose the same for different functions)
    max_num_contexts = num_contexts_range[1]
    max_num_targets = num_targets_range[1]
    shuffled_context_indices = np.random.permutation(max_num_contexts)
    shuffled_target_indices = np.random.permutation(max_num_targets)
    # Select some of those at random
    num_contexts = np.random.randint(low=num_contexts_range[0], 
                                     high=num_contexts_range[1] + 1)
    num_targets = np.random.randint(low=num_targets_range[0], 
                                    high=num_targets_range[1] + 1)
    context_indices = shuffled_context_indices[:num_contexts]
    target_indices = shuffled_target_indices[:num_targets]
     
    trimmed_batch = []
    for func in shuffled_batch:
        func = {'x_c': func['x_c'][context_indices],
                'x_t': func['x_t'][target_indices],
                'y_c': func['y_c'][context_indices],
                'y_t': func['y_t'][target_indices],
                }
        trimmed_batch.append(func)
    collated_batch = {}
    collated_batch['x_c'] = torch.stack([func['x_c'] for func in trimmed_batch])
    collated_batch['x_t'] = torch.stack([func['x_t'] for func in trimmed_batch])
    collated_batch['y_c'] = torch.stack([func['y_c'] for func in trimmed_batch])
    collated_batch['y_t'] = torch.stack([func['y_t'] for func in trimmed_batch])
    return collated_batch



def graphs_shuffled_batch_collate_fn(shuffled_batch,
                                     num_contexts_range, num_targets_range):
    '''
    This function works together with the method get_ct_indices_for_one_function
    and the functions get_non_overlapping_ct_indices or get_overlapping_ct_indices
    in order to select batches of datapoints such that:

    - Datapoints selected are not the same across functions.
    - Within the same batch, all functions have the same number of datapoints.
    - Different batches can have different numbers of datapoints.
    - Within each function, the context datapoints and the target datapoints could
      be overlapping or non-overlapping, depending on our preferences.

    The way this works is the following:

    - First, get_ct_indices_for_one_function shuffles and __get_item__ shuffle
      the datapoints of each function so that each function in the batch presents
      datapoints in a different order.

    - Second, get_non_overlapping_ct_indices or get_overlapping_ct_indices get
      context and target indices that are non-intersecting or intersecting.

    - Third, the collate functions select the context and target datapoints with
      these indices.
    '''
    # Re-shuffle indices of contexts and targets
    # (NOTE datapoints are already shuffled for each individual function so that we
    # don't choose the same for different functions)
    max_num_contexts = num_contexts_range[1]
    max_num_targets = num_targets_range[1]
    shuffled_context_indices = np.random.permutation(max_num_contexts)
    shuffled_target_indices = np.random.permutation(max_num_targets)
    # Select some of those at random
    num_contexts = np.random.randint(low=num_contexts_range[0], 
                                     high=num_contexts_range[1] + 1)
    num_targets = np.random.randint(low=num_targets_range[0], 
                                    high=num_targets_range[1] + 1)
    context_indices = shuffled_context_indices[:num_contexts]
    target_indices = shuffled_target_indices[:num_targets]

    trimmed_batch = []
    for func in shuffled_batch:
        func = {'atoms_c': func['atoms_c'][context_indices],
                'atoms_mask_c': func['atoms_mask_c'][context_indices],
                'bonds_c': func['bonds_c'][context_indices],
                'adjacencies_c': func['adjacencies_c'][context_indices],
                'y_c': func['y_c'][context_indices],
                'atoms_t': func['atoms_t'][target_indices],
                'atoms_mask_t': func['atoms_mask_t'][target_indices],
                'bonds_t': func['bonds_t'][target_indices],
                'adjacencies_t': func['adjacencies_t'][target_indices],
                'y_t': func['y_t'][target_indices],
                }
        trimmed_batch.append(func)

    collated_batch = {}
    collated_batch['atoms_c'] = torch.stack([func['atoms_c'] for 
                                             func in trimmed_batch])
    collated_batch['atoms_mask_c'] = torch.stack([func['atoms_mask_c'] for 
                                                  func in trimmed_batch])
    collated_batch['bonds_c'] = torch.stack([func['bonds_c'] for 
                                             func in trimmed_batch])
    collated_batch['adjacencies_c'] = torch.stack([func['adjacencies_c'] for 
                                                   func in trimmed_batch])
    collated_batch['y_c'] = torch.stack([func['y_c'] for 
                                         func in trimmed_batch])
    collated_batch['atoms_t'] = torch.stack([func['atoms_t'] for 
                                             func in trimmed_batch])
    collated_batch['atoms_mask_t'] = torch.stack([func['atoms_mask_t'] for 
                                                  func in trimmed_batch])
    collated_batch['bonds_t'] = torch.stack([func['bonds_t'] for 
                                             func in trimmed_batch])
    collated_batch['adjacencies_t'] = torch.stack([func['adjacencies_t'] for 
                                                   func in trimmed_batch])
    collated_batch['y_t'] = torch.stack([func['y_t'] for 
                                         func in trimmed_batch])
    return collated_batch
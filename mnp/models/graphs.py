# For models that use the molecular graph

import torch
import torch.nn as nn
from mnp.models.parents import SimpleModel, MetaModel
from mnp.models.vectors import FullyConnectedCNP, FullyConnectedLNP, FullyConnectedNN
from mnp.molecular_featurizer import MolecularFeaturizer
from typing import Union
import gc
import numpy as np


class AtomsFFNN(nn.Module):
    """
    Feed-forward NN that processes RDKit atom features.
    """

    def __init__(self, num_atom_features: int, num_atom_V_features: int,
                 num_hidden_neurons: int=50) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_atom_features,out_features=num_hidden_neurons)
        self.linear2 = nn.Linear(in_features=num_hidden_neurons,out_features=num_atom_V_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        return x


class BondsFFNN(nn.Module):
    """
    Feed-forward NN that processes RDKit bond features, and then aggregates them using the adjacency
    matrix so that they have atomic dimensionality instead.

    - Before aggregation, we have a tensor of shape:

        (batch_size, num_atoms, num_atoms, num_bond_features)

    - After aggregation, we have a tensor of shape:

        (batch_size, num_atoms, num_bond_features)
    """

    def __init__(self, num_bond_features: int, num_bond_V_features: int, 
                 num_hidden_neurons: int=50) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_bond_features,
                                 out_features=num_hidden_neurons)
        self.linear2 = nn.Linear(in_features=num_hidden_neurons,
                                 out_features=num_bond_V_features)

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # Process RDKit bond features
        x = self.linear1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        # Multiply by adjacency matrix and aggregate along atom dimension. That 
        # way each atom will only be influenced by the bonds it is involved in
        # NOTE einsum is the same as in the previous GNN model, but with one
        # additional dimension "f" for function
        x = torch.einsum('fijkl,fijk->fijl', x, adjacency_matrix)
        return x


class Attention(nn.Module):
    def __init__(self, num_atom_V_features: int=30, num_bond_V_features: int=10, 
                 num_QK_features: int=50, mp_iterations: int=3) -> None:
        """
        Update atom features with QKV attention coefficients, weighted so that only direct
        neighbours contribute to the update in each iteration.

            - num_atom_V_features: number of features of value vectors from atoms. 
                To be concatenated with value vectors from bonds.
            - num_bond_V_features: number of features of value vectors from bonds.
                To be concatenated with value vectors from atoms.
            - num_QK_features: number of features of query and key vectors
            - mp_iterations: number of message passing iterations
        """
        super().__init__()
        self.num_QK_features = torch.tensor(num_QK_features)
        self.mp_iterations = mp_iterations
        num_V_features = num_atom_V_features + num_bond_V_features
        self.query = nn.Linear(in_features=num_V_features, out_features=num_QK_features)
        self.key = nn.Linear(in_features=num_V_features, out_features=num_QK_features)
        

    def forward(self, values: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:

        for _ in range(self.mp_iterations):
            # Compute unnormalized attention coefficients between atoms
            query = torch.tanh(self.query(values))
            key = torch.tanh(self.key(values))
            # NOTE einsum is the same as for the vanilla GNN but with an additional dimension
            # "f" for functions or tasks
            e = torch.einsum('fijk,filk->fijl',query,key) / torch.sqrt(self.num_QK_features)
            # Use adjacency matrix to remove contributions from non-neighbours and from
            # artificial padding atoms, i.e. atoms filling atom dimension up to 132
            adjacency_log_weights = (adjacency_matrix - 1) * 1000
            e = e + adjacency_log_weights
            # Compute normalized attention coefficients
            alpha = nn.functional.softmax(e, dim=-1)
            # Update values
            # NOTE einsum is the same as for the vanilla GNN but with an additional dimension
            # "f" for functions or tasks
            values = torch.einsum('fijk,fikl->fijl', alpha.float(), values)
        return values


class MolecularGraphAttentionEncoder(nn.Module):

    """
    Graph NN with attention for molecular property prediction. It operates in the 
    following way:

    1. It calculates atomic representations by processing and aggregating the RDKit 
       atomic representations and the RDKit bond representations. Each atom includes 
       the RDKit bond representations of the bonds it is involved in.

    2. It updates the atomic representation by message passing on the molecular graph, 
       using attention to compute query-key-values (QKV) attention coefficients that 
       weigh the incoming messages.

    3. It produces an overall molecular representation by creating a "super-atom" that
       is connected to all atoms. The "super-atom" representation is initialized to 
       the sum of all atoms.

    4. It updates the "super-atom" representation by the same strategy of message 
       passing with attention as before.

    5. It further processes the "super-atom" molecular representation by passing it
       through a small feed-forward neural network (FFNN).

    6. It computes the final prediction from the final "super-atom" representation
       with a single linear layer. This favours that molecules with different labels 
       are well separated in the space of the final representation.
    """

    def __init__(self, num_atom_features: int, num_bond_features:int, 
                 num_atom_V_features: int=25, num_bond_V_features: int=25, 
                 num_QK_features: int=20, mp_iterations: int=3,
                 device: str='cuda') -> None:
        super().__init__()
        # Initialize model
        self.atoms_ffnn = AtomsFFNN(num_atom_features=num_atom_features, 
                                    num_atom_V_features=num_atom_V_features)
        self.bonds_ffnn = BondsFFNN(num_bond_features=num_bond_features, 
                                    num_bond_V_features=num_bond_V_features)
        self.atoms_attention = Attention(num_atom_V_features=num_atom_V_features, 
                                         num_bond_V_features=num_bond_V_features,
                                         num_QK_features=num_QK_features, 
                                         mp_iterations=mp_iterations)
        self.mol_attention = Attention(num_atom_V_features=num_atom_V_features, 
                                       num_bond_V_features=num_bond_V_features,
                                       num_QK_features=num_QK_features, 
                                       mp_iterations=mp_iterations)
        num_V_features = num_atom_V_features + num_bond_V_features
        self.num_V_features = num_V_features
        self.linear1 = nn.Linear(in_features=num_V_features, out_features=num_V_features)
        self.linear2 = nn.Linear(in_features=num_V_features, out_features=num_V_features)
        # Move to device
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, atoms: torch.Tensor, atoms_mask: torch.Tensor, 
                adjacency_matrix: torch.Tensor, bonds: torch.Tensor,
                max_points_per_iter: int=600) -> torch.Tensor:
        
        num_functions = atoms.shape[0]
        num_mols = atoms.shape[1]
        num_final_features = self.num_V_features

        if num_mols <= max_points_per_iter:
            return self._forward(atoms=atoms, atoms_mask=atoms_mask, 
                                 adjacency_matrix=adjacency_matrix,
                                 bonds=bonds)
        else:
            mol_h = torch.zeros(num_functions, num_mols, num_final_features).float()
            index_split = torch.tensor_split(torch.tensor(range(num_mols)),
                                             int(num_mols/max_points_per_iter))
            for each_split in index_split:
                mol_h[:, each_split] = self._forward(atoms=atoms[:,each_split],
                                        atoms_mask=atoms_mask[:,each_split], 
                                        adjacency_matrix=adjacency_matrix[:,each_split],
                                        bonds=bonds[:,each_split]).cpu().float()
            return mol_h


    def _forward(self, atoms: torch.Tensor, atoms_mask: torch.Tensor, 
                adjacency_matrix: torch.Tensor, bonds: torch.Tensor) -> torch.Tensor:

        # Move inputs to device
        atoms = atoms.to(self.device)
        atoms_mask = atoms_mask.to(self.device)
        adjacency_matrix = adjacency_matrix.to(self.device)
        bonds = bonds.to(self.device)
     
        # Obtain processed atom representation by passing RDKit atom features through FFNN
        atoms_values = self.atoms_ffnn(atoms)
        # Obtain processed bond representation by passing RDKit atom features through FNN,
        # and summing using adjacency matrix as weights. This way:
        # - We aggregate bond features into a tensor with atom dimensionality instead 
        #   of bond dimensionality
        # - For each atom, only the bonds linked to that atom are considered
        bonds_values = self.bonds_ffnn(bonds, adjacency_matrix)
        # Obtain values for QKV with atom dimensionality by concatenating the processed
        # atom features and the bond features (we can do this because bond features have 
        # been forced to atom dimensionality)
        values = torch.cat([atoms_values, bonds_values], dim=-1)
        # Update each atom's representation by message passing with attention
        atoms_h = self.atoms_attention(values, adjacency_matrix)
        # Zero-out padding atoms before the creation of the "super-atom"
        zeroed_atoms_h = atoms_h * atoms_mask
        # Create "super-atom" whose final updated features will be the molecular repesentation.
        # The "super-atom"
        # - is connected to all other atoms, so it can be updated with contributions from all atoms.
        # - is initialized to the sum of all atoms
        # NOTE einsum is the same as for the vanilla GNN but with an additional dimension
        # "f" for functions or tasks
        superatom_h = torch.einsum('fijk->fik',zeroed_atoms_h)[:,:,None,:]
        mol_h = torch.cat((superatom_h, atoms_h), dim=-2)
        # Create augmented adjacency matrix with super-atom
        # - The super-atom is connected to all other atoms
        # - All other atoms keep their connections as usual
        f_batch_size = atoms_h.shape[0]
        d_batch_size = atoms_h.shape[1]
        num_atoms = atoms_h.shape[2]
        aug_adjacency_matrix = torch.ones(f_batch_size, 
                                          d_batch_size, 
                                          num_atoms+1, 
                                          num_atoms+1).to(torch.device(self.device))
        aug_adjacency_matrix[:,:,1:,1:] = adjacency_matrix
        # Update super-atom representation to obtain molecular representation
        mol_h = self.mol_attention(mol_h, aug_adjacency_matrix)
        mol_h = mol_h[:,:,0]
        # Small FFNN to obtain final molecular representatin
        mol_h = self.linear1(mol_h)
        mol_h = nn.functional.leaky_relu(mol_h)
        mol_h = self.linear2(mol_h)
        mol_h = nn.functional.leaky_relu(mol_h)
        return mol_h.cpu()



class MolecularGraphAttentionNP(MetaModel):

    def __init__(self, num_atom_features: int, num_bond_features: int, 
                 num_atom_V_features: int, num_bond_V_features: int, 
                 num_QK_features: int, mp_iterations: int,
                 device: Union[str, torch.device]='cpu') -> None:

        super().__init__()

        self.encoder = MolecularGraphAttentionEncoder(num_atom_features=num_atom_features,
                            num_bond_features=num_bond_features,
                            num_atom_V_features=num_atom_V_features,
                            num_bond_V_features=num_bond_V_features,
                            num_QK_features=num_QK_features,
                            mp_iterations=mp_iterations,
                            device=device).float()
        self.device = torch.device(device)

    def forward(self, inputs):

        np_inputs = {}
        np_inputs['x_c'] = self.encoder(inputs['atoms_c'], inputs['atoms_mask_c'],
                                        inputs['adjacencies_c'], inputs['bonds_c'])
        np_inputs['y_c'] = inputs['y_c']
        np_inputs['x_t'] = self.encoder(inputs['atoms_t'], inputs['atoms_mask_t'], 
                                        inputs['adjacencies_t'], inputs['bonds_t'])
        del inputs
        gc.collect()
        return self.np(np_inputs)

    def to(self, device: Union[str,torch.device]):
        self.device = torch.device(device)
        self.encoder.device = torch.device(device)
        self.np.device = torch.device(device)
        super().to(device)




class MolecularGraphAttentionCNP(MolecularGraphAttentionNP):

    def __init__(self, num_atom_features: int, num_bond_features: int, 
                 num_atom_V_features: int, num_bond_V_features: int, 
                 num_QK_features: int, mp_iterations: int, r_len: int,
                 y_len: int, use_layernorm: bool=False,
                 prediction_mode: str='regression',
                 lincs_architecture: bool=False,
                 device: Union[str, torch.device]='cpu') -> None:

        super().__init__(num_atom_features=num_atom_features,
                         num_bond_features=num_bond_features, 
                         num_atom_V_features=num_atom_V_features,
                         num_bond_V_features=num_bond_V_features, 
                         num_QK_features=num_QK_features,
                         mp_iterations=mp_iterations,
                         device=device)

        encoding_len = num_atom_V_features + num_bond_V_features
        self.np = FullyConnectedCNP(x_len=encoding_len, r_len=r_len, 
                        y_len=y_len, use_layernorm=use_layernorm,
                        prediction_mode=prediction_mode,
                        lincs_architecture=lincs_architecture,
                        device=device).float()
        self.device = torch.device(device)
        self.to(self.device)
        self.prediction_mode = prediction_mode

    def forward(self, inputs: dict) -> dict:

        np_inputs = {}
        np_inputs['x_c'] = self.encoder(inputs['atoms_c'].float(), inputs['atoms_mask_c'].float(),
                                        inputs['adjacencies_c'].float(), inputs['bonds_c'].float())
        np_inputs['y_c'] = inputs['y_c'].float()
        np_inputs['x_t'] = self.encoder(inputs['atoms_t'].float(), inputs['atoms_mask_t'].float(), 
                                        inputs['adjacencies_t'].float(), inputs['bonds_t'].float())
        del inputs
        gc.collect()
        return self.np(np_inputs)

    def till_function_encoder(self, inputs: dict) -> dict:

        x = self.encoder(inputs['atoms'].float(), inputs['atoms_mask'].float(),
                         inputs['adjacencies'].float(), inputs['bonds'].float())
        y = inputs['y'].float()
        # Till datapoint encoder
        r_i = self.np.datapoint_encoder(x, y)
        r = self.np.function_encoder(r_i)
        return r


   
class MolecularGraphAttentionLNP(MolecularGraphAttentionNP):

    def __init__(self, num_atom_features: int, num_bond_features: int, 
                 num_atom_V_features: int, num_bond_V_features: int, 
                 num_QK_features: int, mp_iterations: int, r_len: int, z_len: int,
                 y_len: int, use_layernorm: bool=False,
                 prediction_mode: str='regression',
                 device: Union[str, torch.device]='cpu') -> None:

        super().__init__(num_atom_features=num_atom_features,
                         num_bond_features=num_bond_features, 
                         num_atom_V_features=num_atom_V_features,
                         num_bond_V_features=num_bond_V_features, 
                         num_QK_features=num_QK_features,
                         mp_iterations=mp_iterations,
                         device=device)

        encoding_len = num_atom_V_features + num_bond_V_features
        self.np = FullyConnectedLNP(x_len=encoding_len, r_len=r_len, z_len=z_len,
                                    y_len=y_len, use_layernorm=use_layernorm,
                                    prediction_mode=prediction_mode,
                                    device=device)
        self.device = torch.device(device)
        self.to(self.device)
        self.prediction_mode = prediction_mode

    def forward(self, inputs):

        np_inputs = {}
        np_inputs['x_c'] = self.encoder(inputs['atoms_c'], inputs['atoms_mask_c'],
                                        inputs['adjacencies_c'], inputs['bonds_c'])
        np_inputs['y_c'] = inputs['y_c']
        np_inputs['x_t'] = self.encoder(inputs['atoms_t'], inputs['atoms_mask_t'], 
                                        inputs['adjacencies_t'], inputs['bonds_t'])
        # if training, 'y_t' is given in the inputs
        if 'y_t' in inputs.keys():
            np_inputs['y_t'] = inputs['y_t']
        del inputs
        gc.collect()
        return self.np(np_inputs)



class MolecularGraphAttentionNN(SimpleModel):

    def __init__(self, num_atom_features: int, num_bond_features: int, 
                 num_atom_V_features: int, num_bond_V_features: int, 
                 num_QK_features: int, mp_iterations: int,
                 y_len: int, use_layernorm: bool=True,
                 prediction_mode: str='regression',
                 device: Union[str, torch.device]='cuda'):

        super().__init__()

        self.encoder = MolecularGraphAttentionEncoder(num_atom_features=num_atom_features,
                            num_bond_features=num_bond_features,
                            num_atom_V_features=num_atom_V_features,
                            num_bond_V_features=num_bond_V_features,
                            num_QK_features=num_QK_features,
                            mp_iterations=mp_iterations, device=device).float()
        self.device = torch.device(device)
        encoding_len = num_atom_V_features + num_bond_V_features
        self.nn = FullyConnectedNN(x_len=encoding_len, y_len=y_len,
                                   use_layernorm=use_layernorm,
                                   prediction_mode=prediction_mode,
                                   device=device).float()
        self.prediction_mode = prediction_mode
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, inputs):
        
        output = {}
        if 'atoms' in inputs:
            nn_inputs = {}
            nn_inputs['x'] = self.encoder(inputs['atoms'].float()[None,:,:,:],
                                            inputs['atoms_mask'].float()[None,:,:,:],
                                            inputs['adjacencies'].float()[None,:,:,:],
                                            inputs['bonds'].float()[None,:,:,:])
            # if predicting a single function (as in normal training regime),
            # we want output shape (num_datapoints, 1)
            if self.prediction_mode == 'regression':
                output['y_hat_mean'] = self.nn(nn_inputs)['y_hat_mean'][0]
            elif self.prediction_mode == 'binary_classification':
                output['p_hat'] = self.nn(nn_inputs)['p_hat'][0]
        if 'atoms_c' in inputs:
            nn_inputs = {}
            nn_inputs['x_c'] = self.encoder(inputs['atoms_c'].float(),
                                            inputs['atoms_mask_c'].float(),
                                            inputs['adjacencies_c'].float(),
                                            inputs['bonds_c'].float())
            # if predicting a batch of functions (as in metalearning),
            # we want output shape (num_functions, num_datapoints, 1)
            if self.prediction_mode == 'regression':
                output['y_c_hat_mean'] = self.nn(nn_inputs)['y_c_hat_mean']
            elif self.prediction_mode == 'binary_classification':
                output['p_c_hat'] = self.nn(nn_inputs)['p_c_hat']

            # shape = output['y_c_hat_mean'].shape
        if 'atoms_t' in inputs:
            nn_inputs = {}
            nn_inputs['x_t'] = self.encoder(inputs['atoms_t'].float(),
                                            inputs['atoms_mask_t'].float(),
                                            inputs['adjacencies_t'].float(),
                                            inputs['bonds_t'].float())
            # if predicting a batch of functions (as in metalearning),
            # we want output shape (num_functions, num_datapoints, 1)
            if self.prediction_mode == 'regression':
                output['y_t_hat_mean'] = self.nn(nn_inputs)['y_t_hat_mean']
            elif self.prediction_mode == 'binary_classification':
                output['p_t_hat'] = self.nn(nn_inputs)['p_t_hat']
        del inputs
        gc.collect()
        return output

    def to(self, device: Union[str,torch.device]):
        self.device = torch.device(device)
        self.encoder.device = torch.device(device)
        self.nn.device = torch.device(device)
        super().to(device)
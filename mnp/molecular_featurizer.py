import rdkit
import rdkit.Chem as Chem
from typing import List, Union, Tuple
import pandas as pd
import torch
import numpy as np


class MolecularFeaturizer():
    """
    Featurizer to represent molecular datasets. Given molecules in SMILES format, it
    creates a Pytorch with the following attributes:

    - atoms: tensor with atomic representation of size 
    
        (num_mols, max_num_atoms, num_atom_features)

      NOTE Hydrogens atoms are implicit.

    - atoms_mask: binary tensor indicating whether an atom in a molecule is real (1) or
        padded (0). Padding is necessary because not all molecules have the same number
        of atoms. Of size

        (num_mols, max_num_atoms, 1)

    - bonds: tensor with bond representation of size

        (num_mols, max_num_atoms, max_num_atoms)

        Note that not all atoms in a molecule form bonds between them. A slice of the
        `bonds` tensor corresponding to two atoms without a bond between them is popu-
        lated by 0s.

    - adjacencies: tensor with adjacency matrices of size

        (num_mols, max_num_atoms, max_num_atoms)

    - y: tensor with the molecular properties to predict as targets.
    """

    def __init__(self,
                 max_num_atoms: int=75,
                 use_chirality: bool=True,
                 use_stereochemistry: bool=True,
                 progress_bar: bool=True) -> None:
        """
        - max_num_atoms: maximum number of atoms that fit your molecule
          matrix representation.
            - In DOCKSTRING, we take this to be 75, since the maximum number of
              atoms of any molecule is 73.
            - In LD50, we take it to be 140, since the maximum number of atoms
              of any molecule is 139.
        - 
        """
        super().__init__()
        # General attributes
        self.use_chirality = use_chirality
        self.use_stereochemistry = use_stereochemistry
        self.progress_bar = progress_bar
        self.atoms_allowed = self.get_atoms_allowed()
        self.max_num_atoms = max_num_atoms


    def call_on_each_array_row(self, smiles: np.ndarray) -> tuple:
        if smiles.ndim == 1:
            features = self.get_atoms_bonds_and_adjacencies(smiles)
        elif smiles.ndim == 2:
            atoms = []
            atoms_mask = []
            bonds = []
            adjacencies = []
            for each_row in smiles:
                row_features = self.get_atoms_bonds_and_adjacencies(each_row)
                atoms.append(row_features[0])
                atoms_mask.append(row_features[1])
                bonds.append(row_features[2])
                adjacencies.append(row_features[3])
            atoms = torch.stack(atoms)
            atoms_mask = torch.stack(atoms_mask)
            bonds = torch.stack(bonds)
            adjacencies = torch.stack(adjacencies)
            features = atoms, atoms_mask, bonds, adjacencies
        return features


    def __call__(self, smiles: Union[list, pd.Series, np.ndarray]) -> tuple:
        """
        Featurize a list or a pandas series of smiles into
        atoms, atoms' mask, bonds and adjacencies.
        """
        if isinstance(smiles, np.ndarray):
            features = self.call_on_each_array_row(smiles)
        else:
            features = self.get_atoms_bonds_and_adjacencies(smiles)
        # features is a tuple of 
        #   atoms, atoms_mask, bonds, adjacencies
        return features


    def get_atoms_allowed(self) -> List[str]:
        """
        Identify atoms present in the dataset.
        """
        # TODO Change to simply returning ['C', 'O', 'S', 'N', 'Cl', 'F', 'Br', 'I', 'P', 'Unknown']
        # (if hydrogens implicit) or ['C', 'O', 'S', 'N', 'Cl', 'F', 'Br', 'I', 'P', 'H', 'Unknown']
        # (if hydrogens explicit), or maybe always considr hydrogens implicit
        # NOTE You'll need to pre-train your models for this. Probably during Slovakia would be
        # a good time for this
        atoms_allowed = ['Cl', 'C', 'N', 'O', 'S', 'F', 'Br', 'I', 'P']
        # TODO Add 'Unknown' at the end of the list, and use it as a bucket for all atoms that are not
        # explicitly listed
        return atoms_allowed


    def get_atoms_bonds_and_adjacencies(self, smiles: Union[list, pd.Series, np.ndarray]) -> torch.Tensor:
        """
        Compute inputs X for atoms and bonds:
        - atom features
        - binary mask that indicates whether an atom is really in 
          the molecule (1) or is padded (0).
        - bond features
        - adjacencies
        """
        if isinstance(smiles, list) or isinstance(smiles, np.ndarray):
            smiles = pd.Series(smiles)
        mols = smiles.map(Chem.MolFromSmiles)
        # Atoms
        if self.progress_bar:
            atoms_output = mols.progress_map(self.get_all_atom_features).to_list()
        else:
            atoms_output = mols.map(self.get_all_atom_features).to_list()
        atoms = [elem[0] for elem in atoms_output]
        is_real_atom = [elem[1] for elem in atoms_output]
        atoms = torch.tensor(np.stack(atoms), dtype=torch.float32)
        # Atoms' mask (to indicate whether real or padded atom)
        atoms_mask = torch.tensor(np.stack(is_real_atom), dtype=torch.bool)
        # Bonds
        if self.progress_bar:
            bonds = mols.progress_map(self.get_all_bond_features).to_list()
        else:
            bonds = mols.map(self.get_all_bond_features).to_list()
        bonds = torch.tensor(np.stack(bonds), dtype=torch.float32)
        # Adjacencies
        # map(self.get_adjacency_matrix).values gets an array of tensors, which
        # are stacked later on
        if self.progress_bar:
            adjacencies = mols.progress_map(self.get_adjacency_matrix).values
        else:
            adjacencies = mols.map(self.get_adjacency_matrix).values
        adjacencies = torch.tensor(np.stack(adjacencies))
        return atoms, atoms_mask, bonds, adjacencies

    @property
    def num_targets(self):
        return self.y.shape[1]

    # Featurization

    def one_hot_encoding(self, x: List, permitted_list: List) -> List[int]:
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        Adapted from 
        https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/

        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
        return binary_encoding

    # Atom features

    def get_atom_features(self, atom: rdkit.Chem.rdchem.Atom) -> np.ndarray:
        """
        Get RDKit features for one RDKit atom.
        Adapted from 
        https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
        """ 
        atom_type_enc = self.one_hot_encoding(str(atom.GetSymbol()), self.atoms_allowed)
        n_heavy_neighbors_enc = self.one_hot_encoding(int(atom.GetDegree()), 
                                                      [0, 1, 2, 3, 4, "MoreThanFour"])
        formal_charge_enc = self.one_hot_encoding(int(atom.GetFormalCharge()), 
                                                  [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        hybridisation_type_enc = self.one_hot_encoding(str(atom.GetHybridization()), 
                                                       ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        is_in_a_ring_enc = [int(atom.IsInRing())]
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
        atom_feature_vector = (atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + 
                               hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + 
                               atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled)                 
        if self.use_chirality:
            chirality_type_enc = self.one_hot_encoding(str(atom.GetChiralTag()), 
                                                       ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", 
                                                        "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc         
        n_hydrogens_enc = self.one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
        
        return np.array(atom_feature_vector)

    def get_all_atom_features(self, mol: rdkit.Chem.rdchem.Mol) -> Tuple[np.ndarray]:
        """
        Get features for all atoms in a molecule.
        """
        num_atoms = mol.GetNumAtoms()
        atoms_features = np.zeros((self.max_num_atoms, self.num_atom_features),
                                  dtype=np.float32)
        # Compute features for real atoms in the molecule
        for i in range(num_atoms):
            each_atom = mol.GetAtomWithIdx(i)
            atoms_features[i] = self.get_atom_features(each_atom)
        # Pad up to the maximum number of atoms with a made-up
        # value that is exclusive to the artificial padding atoms
        atoms_features[num_atoms:] = -9
        # Create boolean mask for real atoms. This will be useful later on to zero-out
        # contributions from artificial padding atoms
        is_real_atom = np.zeros((self.max_num_atoms,1),
                                 dtype=np.float32)
        is_real_atom[:num_atoms] = 1
        return atoms_features, is_real_atom

    @property
    def num_atom_features(self) -> int:
        """
        Get the number of features in our atom representation from RDKit.
        """
        example_smiles = 'CCC'
        example_mol = Chem.MolFromSmiles(example_smiles)
        example_atom = example_mol.GetAtoms()[0]
        features = self.get_atom_features(example_atom)
        num_atom_features = len(features)
        return num_atom_features

    # Adjacency matrix

    def get_adjacency_matrix(self, mol: rdkit.Chem.rdchem.Mol) -> torch.Tensor:
        """
        Get the adjacency (connectivity) matrix between the atoms of a molecule.
        """
        num_atoms = mol.GetNumAtoms()
        adjacency_matrix = np.zeros((self.max_num_atoms, self.max_num_atoms),
                                    dtype=np.float32)
        adjacency_matrix[:num_atoms, :num_atoms] = Chem.GetAdjacencyMatrix(mol)
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.bool)
        return adjacency_matrix

    # Bond features

    def get_bond_features(self, bond: rdkit.Chem.rdchem.Bond) -> np.ndarray:
        """
        Get RDKit features for one RDKit bond.
        Adapted from 
        https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
        """
        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                                        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type_enc = self.one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
        if self.use_stereochemistry:
            stereo_type_enc = self.one_hot_encoding(str(bond.GetStereo()), 
                                                    ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc
        return np.array(bond_feature_vector)

    def get_all_bond_features(self, mol: rdkit.Chem.rdchem.Mol) -> np.ndarray:
        num_bonds = mol.GetNumBonds()
        bond_features = np.zeros((self.max_num_atoms, self.max_num_atoms,
                                  self.num_bond_features),
                                 dtype=np.float32)
        (rows, cols) = np.nonzero(Chem.GetAdjacencyMatrix(mol))
        for each_from_atom, each_to_atom in zip(rows, cols):
            each_from_atom = int(each_from_atom)
            each_to_atom = int(each_to_atom)
            bond_features[each_from_atom, each_to_atom] = self.get_bond_features(mol.GetBondBetweenAtoms(each_from_atom,each_to_atom))
        return bond_features

    @property
    def num_bond_features(self) -> int:
        """
        Get the number of features in our bond representation from RDKit.
        """
        example_smiles = 'CCC'
        example_mol = Chem.MolFromSmiles(example_smiles)
        example_bond = example_mol.GetBonds()[0]
        features = self.get_bond_features(example_bond)
        num_bond_features = len(features)
        return num_bond_features
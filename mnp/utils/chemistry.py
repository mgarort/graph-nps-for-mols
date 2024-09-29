from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, RDKFingerprint, DataStructs
from rdkit.Chem.QED import qed
import logging
from typing import List, Iterable, Tuple
import numpy as np
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from tqdm import tqdm
tqdm.pandas()
from functools import partial


def smiles_to_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return qed(mol)


def get_rdkit_fingerprint_for_single_smiles(smiles: str) -> np.ndarray:
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        return None
    if mol:
        fp = RDKFingerprint(mol, maxPath=6)
        fp_list = list(map(int, fp.ToBitString()))
        return np.array(fp_list)


def single_mol_to_bit_morganfp(mol: Chem.Mol, nBits: int=1024, radius: int=3, 
                           return_bit_info: bool=False):
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bit_info)
    fp_array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    fp_array = fp_array.reshape(1,-1).astype(int)
    if return_bit_info:
        return fp_array, bit_info
    else:
        return fp_array

def single_mol_to_counts_morganfp(mol: Chem.Mol, nBits: int=1024, radius: int=3, 
                           return_bit_info: bool=False):
    bit_info = {}
    fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=nBits, bitInfo=bit_info)
    fp_array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    fp_array = fp_array.reshape(1,-1).astype(int)
    if return_bit_info:
        return fp_array, bit_info
    else:
        return fp_array


def many_smiles_to_morganfp(smiles: Iterable[str], progress_bar: bool,
                            counts: bool=False) -> np.ndarray:
    if not counts:
        single_mol_to_morganfp = single_mol_to_bit_morganfp
    else:
        single_mol_to_morganfp = single_mol_to_counts_morganfp
    smiles = pd.Series(smiles)
    if progress_bar:
        mols = smiles.progress_map(Chem.MolFromSmiles)
        mols = mols.progress_map(single_mol_to_morganfp)
    else:
        mols = smiles.map(Chem.MolFromSmiles)
        mols = mols.map(single_mol_to_morganfp)
    mols = np.concatenate(mols.values)
    return mols


def jaccard_coefficient(array1, array2):
    """
    Compute the Jaccard coeffient between two arrays
    """
    array1 = np.asarray(array1).astype(np.bool)
    array2 = np.asarray(array2).astype(np.bool)
    intersection = np.logical_and(array1, array2)
    union = np.logical_or(array1, array2)
    return np.sum(intersection,axis=1) / np.sum(union,axis=1).astype(np.float)


def tanimoto(fp1,fp2):
    """
    Compute Tanimoto similarity (Jaccard coefficient) between two fingerprints,
    which could be given in RDKit format (rdkit.DataStructs.cDataStructs.ExplicitBitVect)
    or as numpy arrays.
    """
    # fp1 = np.squeeze(fp1)
    # fp2 = np.squeeze(fp2)
    rdkit_format = DataStructs.cDataStructs.ExplicitBitVect
    if isinstance(fp1, np.ndarray) and isinstance(fp2, np.ndarray):
        return jaccard_coefficient(fp1,fp2)
    elif isinstance(fp1, rdkit_format) and isinstance(fp2, rdkit_format):
        return DataStructs.FingerprintSimilarity(fp1,fp2)
    else:
        raise RuntimeError('Fingerprints must be either all DataStructs.cDataStructs.ExplicitBitVect' \
                           'or all numpy.arrays, but not a mix.')


def compute_rdkit_closest_and_similarities(df_origin: pd.DataFrame,
                                    df_target: pd.DataFrame,
                                    smiles_type: str='standard_smiles',
                                    num_closest: int=10) -> Tuple[pd.DataFrame, ...]:
    """
    - "origin" refers to the molecules for which we want to find analogues.
    - "target" refers to the pool of molecules where we'll try to find analogues.
    """

    # Get dataframes among which to compute similarities
    # - "origin" refers to the molecules for which we want to find analogues
    # - "target" refers to the pool of molecules where we'll try to find analogues

    # Compute fingerprints of the desired type in the target dataset. One of
    # - RDKit fingerprints, path length 6 (recommended by Greg Landrum to find 
    #   analogues.
    # - Morgan fingerprints.
    df_target['fp'] = df_target[smiles_type].progress_map(get_rdkit_fingerprint_for_single_smiles)

    # Find closest compounds and similarity values. Use chunks of the origin dataset
    # instead of the whole dataset to avoid memory out-of-memory errors    
    df_origin['fp'] = df_origin[smiles_type].progress_map(get_rdkit_fingerprint_for_single_smiles)
    similarities = pd.DataFrame(index=df_origin.index, columns=df_target.index)

    for mol_index in tqdm(df_origin.index):
        mol_fp = df_origin.loc[mol_index,'fp']
        target_fp = np.vstack(df_target[['fp']].values.squeeze()).squeeze()
        tani = tanimoto(mol_fp,target_fp)
        similarities.loc[mol_index] = tani
        # Set self-similarity to -1 to avoid selecting self as similar compound
        if mol_index in similarities.columns:
            similarities.loc[mol_index,mol_index] = -1

    # Get only top num_closest compounds and similarity values
    order = np.argsort(-similarities,axis=1).iloc[:,:num_closest]
    closest_compounds = similarities.columns[order]
    closest_compounds = pd.DataFrame(closest_compounds, index=similarities.index)
    closest_similarities = np.take_along_axis(similarities.values,order.values,axis=1)
    closest_similarities = pd.DataFrame(closest_similarities, index=similarities.index)

    return closest_compounds, closest_similarities
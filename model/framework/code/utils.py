from multiprocessing import Pool
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FastFindRings


def _ebv_to_numpy(ebv):
    return np.frombuffer(ebv.ToBitString().encode("utf-8"), dtype=np.uint8) - ord("0")


def _counts_dict_to_folded_vector(counts, fp_size):
    v = np.zeros(fp_size, dtype=np.float32)
    for k, c in counts.items():
        v[int(k) % fp_size] += float(c)
    return v


def _mol_to_fp_vector(mol, which, fp_size, radius):
    w = which.lower()

    if w == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=fp_size, useFeatures=False, useChirality=True
        )
        return _ebv_to_numpy(fp).astype(np.float32)

    if w == "ecfp4":
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=fp_size, useFeatures=False, useChirality=True
        )
        return _ebv_to_numpy(fp).astype(np.float32)

    if w == "rdk":
        fp = Chem.RDKFingerprint(mol, fpSize=fp_size, maxPath=6)
        return _ebv_to_numpy(fp).astype(np.float32)

    if w == "pattern":
        fp = Chem.PatternFingerprint(mol, fpSize=fp_size)
        return _ebv_to_numpy(fp).astype(np.float32)

    if w == "morganc":
        counts = AllChem.GetMorganFingerprint(
            mol,
            radius,
            useChirality=True,
            useBondTypes=True,
            useFeatures=True,
            useCounts=True,
        ).GetNonzeroElements()
        return _counts_dict_to_folded_vector(counts, fp_size)

    if w == "rdkc":
        counts = AllChem.UnfoldedRDKFingerprintCountBased(mol, maxPath=6).GetNonzeroElements()
        return _counts_dict_to_folded_vector(counts, fp_size)

    raise ValueError(
        f"Unsupported which='{which}'. Supported: "
        f"'morgan', 'ecfp4', 'rdk', 'pattern', 'morganc', 'rdkc', and composites with '+' or '*'."
    )


def _smiles_to_fp(smi, fp_size, radius, is_smarts, which, sanitize=True):
    if is_smarts:
        mol = Chem.MolFromSmarts(str(smi), mergeHs=False)
    else:
        mol = Chem.MolFromSmiles(str(smi), sanitize=False)

    if mol is None:
        return np.zeros(fp_size, dtype=np.float32)

    if sanitize:
        Chem.SanitizeMol(mol, catchErrors=True)
        FastFindRings(mol)
    mol.UpdatePropertyCache(strict=False)

    if ("*" in which) or ("+" in which):
        concat = "*" in which
        split_sym = "*" if concat else "+"

        out = np.zeros(fp_size, dtype=np.float32)
        parts = which.split(split_sym)

        if concat:
            remaining = fp_size
            n_remaining = len(parts)
            cursor = 0
            for part in parts:
                part_size = remaining // n_remaining
                vec = _mol_to_fp_vector(mol, part, part_size, radius)
                out[cursor:cursor + len(vec)] += vec
                cursor += len(vec)
                remaining -= len(vec)
                n_remaining -= 1
        else:
            for part in parts:
                vec = _mol_to_fp_vector(mol, part, fp_size, radius)
                out[:len(vec)] += vec

        return np.log1p(out)

    return _mol_to_fp_vector(mol, which, fp_size, radius)


def convert_smiles_to_fp(list_of_smiles, fp_size=2048, is_smarts=False, which="morgan", radius=2, njobs=1, sanitize=True):
    smiles_list = list(list_of_smiles)

    if njobs and njobs > 1:
        args = [(s, fp_size, radius, is_smarts, which, sanitize) for s in smiles_list]
        with Pool(processes=njobs) as pool:
            fps = pool.starmap(_smiles_to_fp, args)
    else:
        fps = [_smiles_to_fp(s, fp_size, radius, is_smarts, which, sanitize) for s in smiles_list]

    return np.asarray(fps, dtype=np.float32)
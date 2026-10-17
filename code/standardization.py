from math import isnan

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


ORGANIC_ELEMENTS = frozenset(
    {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "B", "Si", "Se"}
)


def fragment_parent(mol):
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(fragments) == 1:
        return mol
    organic = []
    for fragment in fragments:
        symbols = {atom.GetSymbol() for atom in fragment.GetAtoms()}
        if "C" in symbols and not (symbols - ORGANIC_ELEMENTS):
            organic.append(fragment)
    if not organic:
        return None
    ranked = sorted(organic, key=lambda f: (-f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)))
    if len(ranked) > 1:
        best, runner_up = ranked[:2]
        if (best.GetNumHeavyAtoms() == runner_up.GetNumHeavyAtoms()
                and Chem.MolToSmiles(best) != Chem.MolToSmiles(runner_up)):
            return None
    return ranked[0]


def standardize_structure(value):
    if value is None or (isinstance(value, float) and isnan(value)) or not str(value).strip():
        return None
    mol = Chem.MolFromSmiles(str(value).strip(), sanitize=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        mol = fragment_parent(rdMolStandardize.Cleanup(mol))
        if mol is None:
            return None
        mol = rdMolStandardize.Reionize(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetRemoveSp3Stereo(False)
        enumerator.SetRemoveBondStereo(False)
        mol = enumerator.Canonicalize(mol)
        standardized = Chem.MolToSmiles(mol)
        return standardized if Chem.MolToInchiKey(mol) else None
    except Exception:
        return None

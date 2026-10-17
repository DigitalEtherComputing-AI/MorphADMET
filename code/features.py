from functools import lru_cache
import json
import hashlib
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdReducedGraphs, rdFingerprintGenerator as fg
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP

ROOT = Path(__file__).resolve().parents[1] / 'models/v6'
VIEWS = {
    'maplight': ['maplight'],
    'descriptors': ['maplight_desc', 'rdkit_extra', 'mordred'],
    'fingerprints': ['morgan2', 'morgan3', 'atompair', 'torsion', 'rdkitfp', 'maccs'],
}

@lru_cache(maxsize=1)
def contract():
    path=ROOT/'features.json'
    release=json.loads((ROOT/'manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=release['files']['features.json']:
        raise ValueError('V6 feature contract checksum mismatch')
    return json.loads(path.read_text())

def array(fp):
    a = np.zeros(fp.GetLength() if hasattr(fp, 'GetLength') else fp.GetNumBits(), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a

def descriptors(mol, names):
    functions = dict(Descriptors._descList)
    result = []
    for name in names:
        if name not in functions:
            raise RuntimeError(f'Required RDKit descriptor unavailable: {name}')
        try:
            result.append(float(functions[name](mol)))
        except Exception:
            result.append(np.nan)
    return np.asarray(result)

@lru_cache(maxsize=1)
def mordred_calculator():
    from mordred import Calculator, descriptors as md
    calc = Calculator(md, ignore_3D=True)
    if [str(d) for d in calc.descriptors] != contract()['mordred_names']:
        raise RuntimeError('Mordred descriptor order differs from the fitted models')
    return calc

class Chemistry:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError('Invalid SMILES')
        self.cache = {}

    def block(self, name):
        if name in self.cache:
            return self.cache[name]
        mol = self.mol
        generators = {
            'morgan2': lambda: fg.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
            'morgan3': lambda: fg.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True),
            'atompair': lambda: fg.GetAtomPairGenerator(fpSize=2048, includeChirality=True),
            'torsion': lambda: fg.GetTopologicalTorsionGenerator(fpSize=2048, includeChirality=True),
            'rdkitfp': lambda: fg.GetRDKitFPGenerator(fpSize=2048),
        }
        if name in generators:
            a = array(generators[name]().GetFingerprint(mol))
        elif name == 'maccs':
            a = array(MACCSkeys.GenMACCSKeys(mol))
        elif name == 'maplight':
            a = np.concatenate([
                array(fg.GetMorganGenerator(radius=2, fpSize=1024).GetCountFingerprint(mol)),
                array(GetAvalonCountFP(mol, nBits=1024)), rdReducedGraphs.GetErGFingerprint(mol),
                descriptors(mol, contract()['maplight_names'])])
        elif name == 'maplight_desc':
            a = self.block('maplight')[0, -200:]
        elif name == 'rdkit_extra':
            a = descriptors(mol, contract()[name + '_names'])
        elif name == 'mordred':
            a = np.asarray([float(v) if isinstance(v, (int, float, np.number)) else np.nan
                            for v in mordred_calculator()(mol)])
        else:
            raise ValueError(name)
        self.cache[name] = np.asarray(a, dtype=np.float64).reshape(1, -1)
        return self.cache[name]

    def structure(self, view):
        return np.concatenate([self.block(n) for n in VIEWS[view]], axis=1)

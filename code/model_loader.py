from functools import lru_cache
import hashlib
import io
import json
import pickle
from pathlib import Path
import numpy as np
import lightgbm
import torch
import model_types
from features import Chemistry

ROOT = Path(__file__).resolve().parents[1] / 'models/v6'
torch.set_num_threads(1)

@lru_cache(maxsize=1)
def manifest(): return json.loads((ROOT / 'manifest.json').read_text())

@lru_cache(maxsize=1024)
def _verify(name, size, modified):
    path = ROOT / name
    with path.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != manifest()['files'][name]:
        raise ValueError(f'V6 model checksum mismatch: {name}')
    return path

def verified_path(name):
    if Path(name).name != name: raise ValueError('Invalid model path')
    path = ROOT / name
    stat = path.stat()
    return _verify(name, stat.st_size, stat.st_mtime_ns)

CLASS_MAP = {
    ('models','Transform'): model_types.StructureTransform,
    ('models','TabularFit'): model_types.TabularFit,
    **{('neural',n): getattr(model_types,n) for n in ['GraphNetwork','NeuralFit']},
}

class Classes:
    def find_class(self, module, name):
        if (module,name) in CLASS_MAP: return CLASS_MAP[module,name]
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module,name)

class PickleReader(Classes, pickle.Unpickler): pass

@lru_cache(maxsize=16)
def load_fit(name):
    path = verified_path(name)
    with path.open('rb') as f:
        obj = PickleReader(f).load()
    model = obj.get('model') if isinstance(obj,dict) else getattr(obj,'model',obj)
    if hasattr(model,'n_jobs'): model.n_jobs = 1
    return obj

def neural_prediction(fit, chemistry):
    model = fit.model.cpu().eval()
    from chemprop import featurizers
    from chemprop.data import BatchMolGraph
    data = BatchMolGraph([featurizers.SimpleMoleculeMolGraphFeaturizer()(chemistry.mol)])
    with torch.inference_mode():
        p = model(data)
        return (torch.sigmoid(p) if fit.binary else p*fit.scale+fit.center).numpy()

class StructureModel:
    def __init__(self, metadata):
        self.metadata = metadata

    def predict_smiles(self, smiles):
        chemistry = structure_chemistry(smiles)
        total = 0.
        for member in self.metadata['members']:
            name, = member['files']
            fit = load_fit(name)
            if member['family'] == 'dmpnn':
                prediction = neural_prediction(fit, chemistry)
            else:
                prediction = fit.predict(chemistry.structure(member['config']['view']))
            total += member['weight'] * float(prediction[0])
        if not np.isfinite(total): raise ValueError('Nonfinite V6 prediction')
        return total

@lru_cache(maxsize=32)
def structure_chemistry(smiles):
    return Chemistry(smiles)


def load(endpoint):
    if endpoint.startswith('cyp'):
        endpoint = endpoint.removesuffix('_inhibition')
    return StructureModel(manifest()['models'][f'jump_source7/{endpoint}/morgan'])

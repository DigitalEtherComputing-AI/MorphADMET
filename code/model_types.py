import numpy as np
import torch
from torch import nn

class StructureTransform:

    def transform(self, x, k=0):
        z = np.asarray(x[:, self.valid], dtype=np.float64)
        z = np.where(np.isfinite(z), z, self.median)
        z = np.clip(z, self.low, self.high)
        stable = getattr(self, 'stable_columns', np.asarray([], dtype=int))
        if len(stable):
            ordinary = np.ones(len(self.valid), dtype=bool)
            ordinary[stable] = False
            z[:, ordinary] = (z[:, ordinary] - self.center[ordinary]) / self.scale[ordinary]
            magnitude = self.magnitude[stable]
            z[:, stable] = (z[:, stable] / magnitude - self.center[stable] / magnitude) / (self.scale[stable] / magnitude)
        else:
            z = (z - self.center) / self.scale
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite normalized features')
        order = self.order[:k] if k else self.order
        return np.asarray(np.clip(z[:, order], -25, 25), dtype=np.float32)

class TabularFit:

    def __init__(self, transform, model, k, binary, center=0.0, scale=1.0):
        self.transformer = transform
        self.model = model
        self.k = k
        self.binary = binary
        self.center = center
        self.scale = scale

    def predict(self, x):
        z = self.transformer.transform(x, self.k)
        return self.model.predict_proba(z)[:, 1] if self.binary else self.model.predict(z).ravel() * self.scale + self.center

class GraphNetwork(nn.Module):

    def forward(self, batch):
        return self.head(self.aggregate(self.encoder(batch), batch.batch)).ravel()

class NeuralFit:
    pass

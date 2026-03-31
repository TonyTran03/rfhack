# generator.py
import numpy as np
import pandas as pd
from collections import namedtuple
 
FeatureStats = namedtuple('FeatureStats', ['name', 'mean', 'std'])
 
 
def make_blobs(stats, n_samples, separation):
    n0 = n_samples // 2
    n1 = n_samples - n0
    p = len(stats)
    X = np.vstack([np.random.randn(n0, p), np.random.randn(n1, p) + separation])
    for i, s in enumerate(stats):
        X[:, i] = X[:, i] * s.std + s.mean
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    df = pd.DataFrame(X, columns=[s.name for s in stats])
    df['target'] = y.astype(int)
    return df
 
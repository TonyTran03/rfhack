from collections import namedtuple
from ..data.generators import FeatureStats, make_blobs
from .rf_wrapper import RFWrapper
from .auc_utils import sep_from_auc, split_xy
 
HackResult = namedtuple('result', ['df', 'auc', 'separation', 'iterations'])
 
 
class SingleDatasetHacker:
    def __init__(self, df, target_col='target', n_samples=1000, tol=0.005, max_iter=20):
        self.target_col = target_col
        self.n_samples = n_samples
        self.tol = tol
        self.max_iter = max_iter
        self.stats = self._extract_stats(df)
 
    def _extract_stats(self, df):
        X, _ = split_xy(df, self.target_col)
        return [FeatureStats(c, X[c].mean(), X[c].std()) for c in X.columns]

    def hack(self, target_auc):
        sep = sep_from_auc(target_auc)
        lo, hi = 0.0, sep * 3
        best = None
        for i in range(self.max_iter):
            mid = (lo + hi) / 2
            df = make_blobs(self.stats, self.n_samples, mid)
            score = RFWrapper.from_dataframe(df, self.target_col)
            if best is None or abs(score - target_auc) < abs(best.auc - target_auc):
                best = HackResult(df, score, mid, i + 1)
            if abs(score - target_auc) < self.tol:
                break
            if score < target_auc:
                lo = mid
            else:
                hi = mid
        return best
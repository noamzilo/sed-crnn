import pandas as pd
from sed_crnn.InferenceResult import InferenceResult
import numpy as np
from typing import List
import os
import json

class CRNNStatistics:
    """
    Collects and dumps statistics for SED-CRNN video inference runs.
    """
    def __init__(self):
        self._results: List[InferenceResult] = []
        self._df: pd.DataFrame = pd.DataFrame()

    def add_inference(self, result: InferenceResult) -> None:
        self._results.append(result)
        self._df = pd.DataFrame()  # Invalidate cached DataFrame

    def build_dataframe(self) -> pd.DataFrame:
        if self._df.empty:
            rows = []
            for r in self._results:
                rows.append({
                    "video_path": r.video_path,
                    "fold_id": r.fold_id,
                    "num_hits": len(r.hits),
                    "num_predicted_events": self._count_events(r.pred_video),
                    "num_gt_events": self._count_events(r.gt_video),
                })
            self._df = pd.DataFrame(rows)
        return self._df

    @staticmethod
    def _count_events(arr: np.ndarray, threshold: float = 0.5) -> int:
        # Count contiguous regions above threshold as events
        above = arr > threshold
        # Find rising edges (start of event)
        return int(np.diff(np.concatenate(([0], above.astype(int), [0]))).sum() // 2)

    @property
    def num_hits(self) -> pd.Series:
        df = self.build_dataframe()
        return pd.Series(df["num_hits"]) if "num_hits" in df.columns else pd.Series(dtype=int)

    @property
    def num_predicted_events(self) -> pd.Series:
        df = self.build_dataframe()
        return pd.Series(df["num_predicted_events"]) if "num_predicted_events" in df.columns else pd.Series(dtype=int)

    @property
    def num_gt_events(self) -> pd.Series:
        df = self.build_dataframe()
        return pd.Series(df["num_gt_events"]) if "num_gt_events" in df.columns else pd.Series(dtype=int)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.build_dataframe()

    def dump_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.build_dataframe()
        df.to_csv(path, index=False)

    def dump_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.build_dataframe()
        df.to_json(path, orient='records', indent=2) 
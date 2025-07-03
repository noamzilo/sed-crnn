import numpy as np
import pandas as pd
from typing import Any

class InferenceResult:
    def __init__(
        self,
        video_path: str,
        fold_id: int,
        hits: pd.DataFrame,
        gt_video: np.ndarray,
        pred_audio: np.ndarray,
        pred_video: np.ndarray,
        fps: float,
        nf: int,
        y: np.ndarray,
        mbe: np.ndarray,
    ):
        self.video_path = video_path
        self.fold_id = fold_id
        self.hits = hits
        self.gt_video = gt_video
        self.pred_audio = pred_audio
        self.pred_video = pred_video
        self.fps = fps
        self.nf = nf
        self.y = y
        self.mbe = mbe 
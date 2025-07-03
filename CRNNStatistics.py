import pandas as pd
from sed_crnn.InferenceResult import InferenceResult
import numpy as np
from typing import List
import os
import json
from sed_crnn.metrics import event_based_f1

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

    @staticmethod
    def _extract_intervals(binary_array: np.ndarray) -> List[tuple]:
        """Return list of (start_idx, end_idx) for contiguous True segments"""
        intervals = []
        in_interval = False
        for i, val in enumerate(binary_array):
            if val and not in_interval:
                start = i
                in_interval = True
            elif not val and in_interval:
                intervals.append((start, i - 1))
                in_interval = False
        if in_interval:
            intervals.append((start, len(binary_array)-1))
        return intervals

    def build_dataframe(self) -> pd.DataFrame:
        if self._df.empty:
            rows = []
            for r in self._results:
                pred_binary = r.pred_video > 0.5
                gt_binary = r.gt_video > 0.5
                pred_intervals = self._extract_intervals(pred_binary)
                gt_intervals = self._extract_intervals(gt_binary)
                f1, tp_events, fp_events, fn_events = event_based_f1(pred_intervals, gt_intervals, r.fps, tol_sec=0.25)
                num_gt_events = len(gt_intervals)
                detection_percent = 100.0 * tp_events / num_gt_events if num_gt_events > 0 else 0.0
                miss_percent = 100.0 * fn_events / num_gt_events if num_gt_events > 0 else 0.0
                fp_rate = 100.0 * fp_events / num_gt_events if num_gt_events > 0 else 0.0
                rows.append({
                    "video_path": r.video_path,
                    "fold_id": r.fold_id,
                    "num_hits": len(r.hits),
                    "num_predicted_events": len(pred_intervals),
                    "num_gt_events": num_gt_events,
                    "tp_events": tp_events,
                    "fn_events": fn_events,
                    "fp_events": fp_events,
                    "detection_percent": detection_percent,
                    "miss_percent": miss_percent,
                    "fp_rate": fp_rate,
                    "event_f1": f1,
                })
            self._df = pd.DataFrame(rows)
        return self._df

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.build_dataframe()

    @property
    def tp_events_sum(self) -> int:
        return int(self.dataframe["tp_events"].sum()) if "tp_events" in self.dataframe.columns else 0

    @property
    def fn_events_sum(self) -> int:
        return int(self.dataframe["fn_events"].sum()) if "fn_events" in self.dataframe.columns else 0

    @property
    def fp_events_sum(self) -> int:
        return int(self.dataframe["fp_events"].sum()) if "fp_events" in self.dataframe.columns else 0

    @property
    def detection_percent(self) -> float:
        total_gt = self.dataframe["num_gt_events"].sum()
        return 100.0 * self.tp_events_sum / total_gt if total_gt > 0 else 0.0

    @property
    def miss_percent(self) -> float:
        total_gt = self.dataframe["num_gt_events"].sum()
        return 100.0 * self.fn_events_sum / total_gt if total_gt > 0 else 0.0

    @property
    def fp_rate(self) -> float:
        total_gt = self.dataframe["num_gt_events"].sum()
        return 100.0 * self.fp_events_sum / total_gt if total_gt > 0 else 0.0

    @property
    def event_f1(self) -> float:
        # Aggregate event-based F1 over all videos
        total_tp = self.tp_events_sum
        total_fp = self.fp_events_sum
        total_fn = self.fn_events_sum
        denom = 2 * total_tp + total_fp + total_fn
        return 2 * total_tp / denom if denom > 0 else 0.0

    def dump_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.dataframe.to_csv(path, index=False)

    def dump_total_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        totals = self.summary
        pd.DataFrame([totals]).to_csv(path, index=False)

    def dump_json_with_totals(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "per_video": self.dataframe.to_dict(orient='records'),
            "totals": self.summary
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @property
    def summary(self) -> dict:
        return {
            "tp_events_sum": self.tp_events_sum,
            "fn_events_sum": self.fn_events_sum,
            "fp_events_sum": self.fp_events_sum,
            "detection_percent": self.detection_percent,
            "miss_percent": self.miss_percent,
            "fp_rate": self.fp_rate,
            "event_f1": self.event_f1,
            "total_gt_events": int(self.dataframe["num_gt_events"].sum()),
            "total_videos": len(self.dataframe),
        }

    def print_total_stats(self) -> None:
        s = self.summary
        print(f"Total videos: {s['total_videos']}")
        print(f"GT events: {s['total_gt_events']}")
        print(f"TP events: {s['tp_events_sum']} ({s['detection_percent']:.2f}%)")
        print(f"FN events: {s['fn_events_sum']} ({s['miss_percent']:.2f}%)")
        print(f"FP events: {s['fp_events_sum']} (FP rate: {s['fp_rate']:.2f}% of GT events)")
        print(f"Event-based F1: {s['event_f1']:.4f}")

    def print_stats_for_videos(self, video_names: List[str]) -> None:
        filtered = self.dataframe[self.dataframe["video_path"].isin(video_names)]
        if filtered.empty:
            print("No matching videos found.")
        else:
            print(filtered.to_string(index=False)) 
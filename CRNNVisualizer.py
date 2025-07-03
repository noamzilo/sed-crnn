#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRNNVisualizer.py

Handles SED-CRNN visualization and output generation.
Responsibilities:
- Frame-level dataframe creation
- Interval extraction and classification
- Plotting predictions and overlays
- CSV export
- Video overlay creation
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple
from sed_crnn.train_constants import *
import tempfile
from sed_crnn.metrics import event_based_f1

# Use cv2.VideoWriter_fourcc, suppress linter warning if needed
# type: ignore[attr-defined]
def fourcc(*args):
	return cv2.VideoWriter_fourcc(*args)  # type: ignore[attr-defined]

class CRNNVisualizer:
	"""
	Handles SED-CRNN visualization and output generation.
	"""
	def __init__(self, alpha: float = 0.5, prediction_threshold: float = 0.5):
		"""
		Initialize the CRNN visualizer.
		Args:
			alpha: Alpha blending factor for overlay
			prediction_threshold: Threshold for binary predictions
		"""
		self.alpha = alpha
		self.prediction_threshold = prediction_threshold

	def blend(self, frame: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
		"""Blend frame with color overlay."""
		overlay = np.full_like(frame, color, dtype=np.uint8)
		return cv2.addWeighted(frame, 1-self.alpha, overlay, self.alpha, 0)

	def create_frame_level_dataframe(self, pred_video, gt_video, fps, nf) -> pd.DataFrame:
		"""Create a dataframe with frame-level predictions and ground truth."""
		video_times = np.arange(nf) / fps
		df = pd.DataFrame({
			'frame': range(nf),
			'time': video_times,
			'prediction': pred_video,
			'ground_truth': gt_video,
			'pred_binary': pred_video > self.prediction_threshold,
			'gt_binary': gt_video > 0.5
		})
		return df

	@staticmethod
	def _extract_intervals(binary_array: np.ndarray) -> List[Tuple[int, int]]:
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
			intervals.append((start, len(binary_array) - 1))
		return intervals

	def create_intervals_dataframe(self, frame_df: pd.DataFrame, fps: float, tolerance_sec: float = 0.25):
		"""Create a dataframe with event intervals and their classification (TP/FP/FN) using event-based logic."""
		pred_intervals = self._extract_intervals(frame_df['pred_binary'].to_numpy(dtype=bool))
		gt_intervals = self._extract_intervals(frame_df['gt_binary'].to_numpy(dtype=bool))
		f1, tp, fp, fn = event_based_f1(pred_intervals, gt_intervals, fps, tol_sec=tolerance_sec)

		# Classify intervals
		matched_pred = set()
		matched_gt = set()
		pred_centers = [((s+e)/2)/fps for s, e in pred_intervals]
		gt_centers = [((s+e)/2)/fps for s, e in gt_intervals]
		for i, pc in enumerate(pred_centers):
			for j, gc in enumerate(gt_centers):
				if abs(pc - gc) <= tolerance_sec:
					matched_pred.add(i)
					matched_gt.add(j)
					break

		intervals_data = []
		for idx, (s, e) in enumerate(pred_intervals):
			intervals_data.append({
				'type': 'prediction',
				'index': idx + 1,
				'start_frame': s,
				'end_frame': e,
				'start_sec': s / fps,
				'end_sec': e / fps,
				'matched': idx in matched_pred,
				'classification': 'TP' if idx in matched_pred else 'FP'
			})
		for idx, (s, e) in enumerate(gt_intervals):
			intervals_data.append({
				'type': 'ground_truth',
				'index': idx + 1,
				'start_frame': s,
				'end_frame': e,
				'start_sec': s / fps,
				'end_sec': e / fps,
				'matched': idx in matched_gt,
				'classification': 'TP' if idx in matched_gt else 'FN'
			})
		intervals_df = pd.DataFrame(intervals_data)
		frame_df = frame_df.copy()
		frame_df['color'] = 'none'
		for _, interval in intervals_df.iterrows():
			s, e = interval['start_frame'], interval['end_frame']
			classification = interval['classification']
			if classification == 'TP':
				frame_df.loc[s:e+1, 'color'] = 'green'
			elif classification == 'FP':
				frame_df.loc[s:e+1, 'color'] = 'yellow'
			elif classification == 'FN':
				frame_df.loc[s:e+1, 'color'] = 'red'
		return intervals_df, frame_df, pred_intervals, gt_intervals, matched_pred, matched_gt

	def plot_predictions(self, frame_df: pd.DataFrame, intervals_df: pd.DataFrame, fps: float, save_path: str, y=None, mbe=None, pred_audio=None):
		"""Create prediction plot using the frame-level dataframe with colors, plus waveform and mel spectrogram."""
		# (Copy the plotting logic from the original class here)
		pass

	def dump_intervals_csv(self, intervals_df: pd.DataFrame, fps: float, out_dir: str, basename: str):
		"""Dump intervals to CSV files."""
		gt_df = intervals_df[intervals_df['type'] == 'ground_truth'].copy()
		gt_df.to_csv(os.path.join(out_dir, f'{basename}_ground_truth.csv'), index=False)
		pred_df = intervals_df[intervals_df['type'] == 'prediction'].copy()
		pred_df.to_csv(os.path.join(out_dir, f'{basename}_predictions.csv'), index=False)
		intervals_df.to_csv(os.path.join(out_dir, f'{basename}_intervals.csv'), index=False)

	def create_video_overlay(self, frame_df: pd.DataFrame, video_path: str, output_path: str, fps: float, width: int, height: int):
		"""Create video overlay using frame-level dataframe colors."""
		tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
		writer = cv2.VideoWriter(tmp_vid, int(fourcc('m','p','4','v')), fps, (width, height))
		cap = cv2.VideoCapture(video_path)
		for i, row in frame_df.iterrows():
			ret, frame = cap.read()
			if not ret:
				break
			color = row['color']
			if color == 'green':
				frame = self.blend(frame, (0, 255, 0))
			elif color == 'yellow':
				frame = self.blend(frame, (0, 255, 255))
			elif color == 'red':
				frame = self.blend(frame, (0, 0, 255))
			writer.write(frame)
		cap.release()
		writer.release()
		import subprocess
		subprocess.check_call([
			"ffmpeg", "-y", "-loglevel", "error",
			"-i", tmp_vid,
			"-i", video_path,
			"-c:v", "copy",
			"-map", "0:v:0", "-map", "1:a:0",
			"-shortest", output_path
		])
		os.remove(tmp_vid) 
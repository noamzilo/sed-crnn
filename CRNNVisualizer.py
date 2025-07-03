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

	def plot_predictions(self, frame_df, intervals_df, fps, save_path, y=None, mbe=None, pred_audio=None):
		"""Create prediction plot using the frame-level dataframe with colors, plus waveform and mel spectrogram."""
		fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})

		# Compute time axes
		if y is not None:
			t_audio = np.arange(len(y)) / SAMPLE_RATE
		if mbe is not None:
			t_mel = np.arange(mbe.shape[0]) / FPS_ORIG
		if frame_df is not None:
			t_video = frame_df['frame'].values / fps

		# 1. Waveform plot with background coloring (using time)
		if y is not None and pred_audio is not None:
			frame_colors = frame_df['color'].values
			samples_per_frame = int(np.ceil(len(y) / len(frame_df)))
			for i, color in enumerate(frame_colors):
				if color != 'none':
					start = i * samples_per_frame / SAMPLE_RATE
					end = min((i + 1) * samples_per_frame, len(y)) / SAMPLE_RATE
					axes[0].axvspan(start, end, alpha=0.2, color=color, zorder=0)
			axes[0].plot(t_audio, y, color='gray', linewidth=0.7, label='Waveform')
			axes[0].set_ylabel('Amplitude')
			axes[0].set_title('Audio Waveform with Event Coloring')
			# Overlay model prediction (upsampled to waveform length)
			pred_audio_upsampled = np.interp(
				np.linspace(0, len(pred_audio) - 1, num=len(y)),
				np.arange(len(pred_audio)),
				pred_audio
			)
			pred_scaled = pred_audio_upsampled * np.max(np.abs(y))
			axes[0].plot(t_audio, pred_scaled, color='b', alpha=0.7, label='Prediction (scaled)')
			# Overlay ground truth as a red mask at 1/8 of the plot height
			if frame_df is not None:
				gt_mask = frame_df['gt_binary'].values.astype(float)
				if len(gt_mask) != len(t_audio):
					gt_mask = np.interp(t_audio, t_video, gt_mask)
				y_min, y_max = axes[0].get_ylim()
				height = (y_max - y_min) / 8
				axes[0].fill_between(t_audio, y_min, y_min + gt_mask * height, color='red', alpha=0.4, label='GT Hit')
			axes[0].legend(loc='upper right')
			axes[0].set_xlim([0, t_audio[-1]])

		# 2. Mel spectrogram plot (aligned x, GT as filled mask, NO colorbar)
		if mbe is not None:
			img = axes[1].imshow(
				mbe.T, aspect='auto', origin='lower', interpolation='nearest', cmap='magma',
				extent=[t_mel[0], t_mel[-1], 0, mbe.shape[1]]
				)
			axes[1].set_ylabel('Mel Bin')
			axes[1].set_title('Mel Spectrogram with Ground Truth Overlay')
			# Overlay ground truth as a filled mask (hit=1/2 height, no hit=0)
			if len(t_mel) == len(frame_df):
				gt_mask = frame_df['gt_binary'].values.astype(float)
			else:
				gt_mask = np.interp(t_mel, t_video, frame_df['gt_binary'].values.astype(float))
			mel_height = mbe.shape[1]
			axes[1].fill_between(t_mel, 0, gt_mask * (mel_height / 8), color='white', alpha=0.4, label='Ground Truth')
			# No colorbar here!
			axes[1].set_xlim([0, t_audio[-1]])
			axes[1].legend(loc='upper right')

		# 3. Frame-level prediction-vs-gt plot (aligned x)
		ax = axes[2]
		pred_intervals = intervals_df[intervals_df['type'] == 'prediction']
		gt_intervals = intervals_df[intervals_df['type'] == 'ground_truth']
		for _, interval in pred_intervals.iterrows():
			s, e = interval['start_frame'], interval['end_frame']
			classification = interval['classification']
			t_s = s / fps
			t_e = (e + 1) / fps
			if classification == 'TP':
				ax.axvspan(t_s, t_e, alpha=0.4, color='green', zorder=0)
			elif classification == 'FP':
				ax.axvspan(t_s, t_e, alpha=0.4, color='yellow', zorder=0)
		for _, interval in gt_intervals.iterrows():
			s, e = interval['start_frame'], interval['end_frame']
			classification = interval['classification']
			t_s = s / fps
			t_e = (e + 1) / fps
			if classification == 'FN':
				ax.axvspan(t_s, t_e, alpha=0.4, color='red', zorder=0)
		# Overlay ground truth as a red mask at 1/8 of the plot height
		if frame_df is not None:
			gt_mask = frame_df['gt_binary'].values.astype(float)
			y_min, y_max = ax.get_ylim()
			height = (y_max - y_min) / 8
			ax.fill_between(t_video, y_min, y_min + gt_mask * height, color='red', alpha=0.4, label='GT Hit')
		# Plot prediction only (aligned x)
		ax.plot(t_video, frame_df['prediction'], 'b-', linewidth=1, label='Prediction')
		for idx, (_, interval) in enumerate(gt_intervals.iterrows()):
			s, e = interval['start_frame'], interval['end_frame']
			center = ((s + e) / 2) / fps
			ax.text(center, 1.05, str(idx + 1), color='blue', ha='center', va='bottom', fontsize=10, fontweight='bold')
		ax.axhline(
			self.prediction_threshold, color='black', linestyle='--', linewidth=1,
			label=f'Threshold={self.prediction_threshold}'
			)
		ax.set_xlabel('Time (seconds)')
		ax.set_ylabel('Score / Label')
		ax.set_ylim(-0.05, 1.15)
		ax.set_xlim([0, t_audio[-1]])
		ax.set_title('Hit Detection Predictions vs Ground Truth (with Tolerance)')
		ax.grid(True, alpha=0.3)
		ax.legend()

		# Set x-ticks at every second for all subplots and ensure tick labels are visible on all axes
		if y is not None:
			total_time = t_audio[-1]
		else:
			total_time = t_video[-1]
		max_sec = int(np.ceil(total_time))
		sec_ticks = np.arange(0, max_sec + 1, 1)
		for axx in axes:
			axx.set_xticks(sec_ticks)
			axx.set_xticklabels([str(int(t)) for t in sec_ticks])
			axx.tick_params(axis='x', which='both', labelbottom=True)

		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.close()
		print(f"✅ Saved plot to: {save_path}")
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRNNInferenceVisualizer.py

Unified SED-CRNN inference and visualization class.
Handles both single and batch video processing with consistent configuration.

Responsibilities:
- Model inference and prediction generation
- Video overlay creation with hit detection visualization
- Plot generation and CSV export
- Batch processing orchestration
- Train/val split organization
"""

import os
import glob
import subprocess
import tempfile
import math
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from sed_crnn.decorte_data_loader import load_decorte_dataset
from sed_crnn.audio_features import _ffmpeg_audio, _mbe
from sed_crnn.crnn_lightning import CRNNLightning
import sed_crnn.audio_features as af
from sed_crnn.train_constants import *
from scipy.interpolate import interp1d

class CRNNInferenceVisualizer:
	"""
	Unified SED-CRNN inference and visualization class.
	Processes single or multiple videos with consistent configuration.
	"""
	
	def __init__(self, ckpt_path: str, output_dir: str, alpha: float = 0.5, 
				 prediction_threshold: float = 0.5, device: str = None):
		"""
		Initialize the CRNN inference visualizer.
		
		Args:
			ckpt_path: Path to the model checkpoint
			output_dir: Base directory for outputs
			alpha: Alpha blending factor for overlay
			prediction_threshold: Threshold for binary predictions
			device: Device to use for inference (auto-detect if None)
		"""
		self.ckpt_path = ckpt_path
		self.output_dir = output_dir
		self.alpha = alpha
		self.prediction_threshold = prediction_threshold
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		
		# Load dataset metadata once
		self.meta_all = load_decorte_dataset()
		
		# Create output directories
		os.makedirs(output_dir, exist_ok=True)
	
	def blend(self, frame, color):
		"""Blend frame with color overlay."""
		overlay = np.full_like(frame, color, dtype=np.uint8)
		return cv2.addWeighted(frame, 1-self.alpha, overlay, self.alpha, 0)
	
	def sliding_windows(self, mbe: np.ndarray, win: int = SEQ_LEN_IN, stride: int = SEQ_LEN_OUT):
		"""Create sliding windows for inference."""
		wins, starts = [], []
		for s in range(0, mbe.shape[0] - win + 1, stride):
			wins.append(mbe[s:s + win].T)
			starts.append(s)
		return np.array(wins), np.array(starts)
	
	def create_frame_level_dataframe(self, pred_video, gt_video, fps, nf):
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
	
	def extract_intervals(self, binary_array):
		"""Extract (start, end) intervals from a binary 1D array."""
		intervals = []
		in_interval = False
		for i, val in enumerate(binary_array):
			if val and not in_interval:
				start = i
				in_interval = True
			elif not val and in_interval:
				end = i - 1
				intervals.append((start, end))
				in_interval = False
		if in_interval:
			intervals.append((start, len(binary_array) - 1))
		return intervals
	
	def create_intervals_dataframe(self, frame_df, fps, tolerance_sec=0.25):
		"""Create a dataframe with event intervals and their classification (TP/FP/FN)."""
		pred_intervals = self.extract_intervals(frame_df['pred_binary'].values)
		gt_intervals = self.extract_intervals(frame_df['gt_binary'].values)
		
		# Calculate centers for matching
		pred_centers = [((s + e) / 2) / fps for s, e in pred_intervals]
		gt_centers = [((s + e) / 2) / fps for s, e in gt_intervals]
		
		# Match predictions to ground truth
		matched_pred = set()
		matched_gt = set()
		for i, pc in enumerate(pred_centers):
			for j, gc in enumerate(gt_centers):
				if abs(pc - gc) <= tolerance_sec:
					matched_pred.add(i)
					matched_gt.add(j)
					break
		
		# Create intervals dataframe
		intervals_data = []
		
		# Add prediction intervals
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
		
		# Add ground truth intervals
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
		
		# Update frame-level dataframe with colors
		frame_df = frame_df.copy()
		frame_df['color'] = 'none'  # Default color
		
		# Apply colors based on intervals
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
	
	def plot_predictions(self, frame_df, intervals_df, fps, save_path):
		"""Create prediction plot using the frame-level dataframe with colors."""
		fig, ax = plt.subplots(figsize=(15, 5))
		
		# Get intervals for shading
		pred_intervals = intervals_df[intervals_df['type'] == 'prediction']
		gt_intervals = intervals_df[intervals_df['type'] == 'ground_truth']
		
		# Shade intervals based on classification
		for _, interval in pred_intervals.iterrows():
			s, e = interval['start_frame'], interval['end_frame']
			classification = interval['classification']
			if classification == 'TP':
				ax.axvspan(s, e+1, alpha=0.4, color='green', zorder=0)
			elif classification == 'FP':
				ax.axvspan(s, e+1, alpha=0.4, color='yellow', zorder=0)
		
		for _, interval in gt_intervals.iterrows():
			s, e = interval['start_frame'], interval['end_frame']
			classification = interval['classification']
			if classification == 'FN':
				ax.axvspan(s, e+1, alpha=0.4, color='red', zorder=0)
		
		# Plot prediction
		ax.plot(frame_df['frame'], frame_df['prediction'], 'b-', linewidth=1, label='Prediction')
		# Plot ground truth as blue line (0 or 1)
		ax.plot(frame_df['frame'], frame_df['ground_truth'], 'c-', linewidth=2, label='Ground Truth')
		
		# Annotate ground truth hits with index
		for idx, (_, interval) in enumerate(gt_intervals.iterrows()):
			s, e = interval['start_frame'], interval['end_frame']
			center = (s + e) // 2
			ax.text(center, 1.05, str(idx+1), color='blue', ha='center', va='bottom', fontsize=10, fontweight='bold')
		
		# Add dashed horizontal threshold line
		ax.axhline(self.prediction_threshold, color='black', linestyle='--', linewidth=1, label=f'Threshold={self.prediction_threshold}')
		ax.set_xlabel('Frame Number')
		ax.set_ylabel('Score / Label')
		ax.set_ylim(-0.05, 1.15)
		ax.set_title('Hit Detection Predictions vs Ground Truth (with Tolerance)')
		ax.grid(True, alpha=0.3)
		ax.legend()
		plt.tight_layout()
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.close()
		print(f"✅ Saved plot to: {save_path}")
	
	def dump_intervals_csv(self, intervals_df, fps, out_dir, basename):
		"""Dump intervals to CSV files."""
		# GT CSV
		gt_df = intervals_df[intervals_df['type'] == 'ground_truth'].copy()
		gt_df.to_csv(os.path.join(out_dir, f'{basename}_ground_truth.csv'), index=False)
		
		# Pred CSV
		pred_df = intervals_df[intervals_df['type'] == 'prediction'].copy()
		pred_df.to_csv(os.path.join(out_dir, f'{basename}_predictions.csv'), index=False)
		
		# Both CSV
		intervals_df.to_csv(os.path.join(out_dir, f'{basename}_intervals.csv'), index=False)
		print(f"✅ Saved CSVs to {out_dir}")
	
	def create_video_overlay(self, frame_df, video_path, output_path, fps, width, height):
		"""Create video overlay using frame-level dataframe colors."""
		print("🎬 Creating video overlay...")
		print("Unique frame colors in video:", frame_df['color'].unique())
		
		tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
		writer = cv2.VideoWriter(tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
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
			# else: do not blend
			writer.write(frame)

		cap.release()
		writer.release()

		# Remux original audio to keep sync
		subprocess.check_call([
			"ffmpeg", "-y", "-loglevel", "error",
			"-i", tmp_vid,
			"-i", video_path,
			"-c:v", "copy",
			"-map", "0:v:0", "-map", "1:a:0",
			"-shortest", output_path
		])
		os.remove(tmp_vid)
		print(f"✅ Saved {output_path}")
	
	def create_ground_truth_in_video_space(self, hits, fps, nf):
		"""Create ground truth labels in video frame space."""
		# Create ground truth in video frame space
		video_times = np.arange(nf) / fps
		gt_video = np.zeros(nf, dtype=np.float32)
		
		for _, h in hits.iterrows():
			start_time = h["start"]
			end_time = h["end"]
			
			# Find video frames that fall within this time range
			mask = (video_times >= start_time) & (video_times <= end_time)
			gt_video[mask] = 1.0
		
		return gt_video
	
	def create_predictions_in_video_space(self, pred_full, fps, nf):
		"""Convert audio frame predictions to video frame predictions."""
		audio_frames = len(pred_full)
		
		# Create time arrays using constants from train_constants.py
		audio_times = np.arange(audio_frames) / FPS_ORIG
		video_times = np.arange(nf) / fps
		
		# Interpolate predictions from audio time to video time
		pred_interpolator = interp1d(audio_times, pred_full, kind='linear', bounds_error=False, fill_value=0)
		pred_video = pred_interpolator(video_times)
		
		return pred_video
	
	def visualize_videos(self, video_paths: list, val_fold: int = 0) -> list:
		"""
		Process multiple videos and organize outputs into train/val folders.
		
		Args:
			video_paths: List of absolute paths to video files
			val_fold: Fold ID to use as validation set (0-3)
			
		Returns:
			list: Processing results for each video
		"""
		# Validate fold ID
		if not 0 <= val_fold <= 3:
			raise ValueError("val_fold must be between 0 and 3")
		
		# Create train/val output directories
		train_output_dir = os.path.join(self.output_dir, "train")
		val_output_dir = os.path.join(self.output_dir, "val")
		os.makedirs(train_output_dir, exist_ok=True)
		os.makedirs(val_output_dir, exist_ok=True)
		
		print("🚀 Starting video processing...")
		print(f"📁 Train output: {train_output_dir}")
		print(f"📁 Val output: {val_output_dir}")
		print(f"🎯 Validation fold: {val_fold}")
		print(f"🎬 Processing {len(video_paths)} videos")
		
		# Process each video
		results = []
		train_count = 0
		val_count = 0
		
		for video_path in video_paths:
			vname = os.path.basename(video_path)
			
			# Check if video is in metadata
			if vname not in self.meta_all:
				print(f"⚠️  Skipping {vname} - not in metadata")
				continue
			
			# Get fold assignment
			fold_id = self.meta_all[vname]["fold_id"]
			
			# Determine output directory based on fold
			basename = os.path.splitext(vname)[0]
			if fold_id == val_fold:  # Validation set
				output_dir = os.path.join(val_output_dir, basename)
				split = 'val'
				val_count += 1
			else:  # Training set
				output_dir = os.path.join(train_output_dir, basename)
				split = 'train'
				train_count += 1
			
			print(f"\n🎯 Processing {vname} (fold {fold_id}) -> {split} -> {output_dir}")
			
			try:
				# Process the video
				result = self._process_video_internal(video_path, output_dir)
				result['fold_id'] = fold_id
				result['split'] = split
				results.append(result)
				print(f"✅ Successfully processed {vname}")
				
			except Exception as e:
				print(f"❌ Error processing {vname}: {str(e)}")
				continue
		
		# Print summary
		self._print_summary(results, train_count, val_count, val_fold)
		
		return results
	
	def _process_video_internal(self, video_path: str, output_dir: str) -> dict:
		"""
		Internal method to process a single video and generate visualizations.
		
		Args:
			video_path: Path to the input video
			output_dir: Output directory
			
		Returns:
			dict: Processing results and paths
		"""
		os.makedirs(output_dir, exist_ok=True)
		
		# Load metadata & hits
		vname = os.path.basename(video_path)
		if vname not in self.meta_all:
			raise RuntimeError(f"{vname} not in Decorte metadata")
		
		meta = self.meta_all[vname]
		hits = meta["hits"]
		fold = meta["fold_id"]
		fold_cache = fold + 1	# scaler is 1-indexed
		
		print(f"🎯 Processing {vname} (fold {fold})")
		
		# Load scaler from ckpt dir or fallback to cache
		scaler_path = os.path.join(os.path.dirname(self.ckpt_path), f"scaler_fold{fold_cache}.joblib")
		if not os.path.exists(scaler_path):
			scaler_path = os.path.join(CACHE_DIR, f"scaler_fold{fold_cache}.joblib")
		scaler = af.load_scaler(scaler_path)
		print(f"✅ Loaded scaler from: {scaler_path}")
		
		# Decode audio → MBE → normalize
		y = _ffmpeg_audio(video_path, SAMPLE_RATE)
		mbe = _mbe(y, SAMPLE_RATE)
		mbe = af.normalize(mbe, scaler)
		
		# Prepare video I/O first to get fps and frame count
		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		nf   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		cap.release()
		
		print(f"📹 Video: {nf} frames at {fps} fps")
		print(f"🎵 Audio: {mbe.shape[0]} frames at {FPS_ORIG} fps")
		
		# Create ground truth in video space
		print("🎯 Creating ground truth in video space...")
		gt_video = self.create_ground_truth_in_video_space(hits, fps, nf)
		print(f"✅ Ground truth: {np.sum(gt_video)} active frames out of {nf}")
		
		# Inference windows
		win_x, win_starts = self.sliding_windows(mbe)
		tensor_x = torch.from_numpy(win_x).unsqueeze(1).float()
		loader = torch.utils.data.DataLoader(
			torch.utils.data.TensorDataset(tensor_x),
			batch_size = 64,
			shuffle    = False,
			pin_memory = True
		)
		
		# Run inference
		model = CRNNLightning.load_from_checkpoint(self.ckpt_path, fold_id=fold, art_dir="/tmp").to(self.device)
		trainer = Trainer(accelerator=self.device, devices=1, logger=False, enable_checkpointing=False)
		logits  = trainer.predict(model, loader)
		
		# Fix the type issue with logits
		if logits is not None:
			preds = torch.cat([torch.tensor(batch) for batch in logits], 0).sigmoid().squeeze(-1).cpu().numpy().reshape(-1)
		else:
			raise RuntimeError("No predictions returned from model")
		
		# Map to per-frame predictions in audio space
		pred_audio = np.zeros(mbe.shape[0], np.float32)
		for i, start in enumerate(win_starts):
			pred_audio[start:start + SEQ_LEN_OUT] = preds[i * SEQ_LEN_OUT : (i + 1) * SEQ_LEN_OUT]
		
		# Convert predictions to video space
		print("🔄 Converting predictions from audio space to video space...")
		pred_video = self.create_predictions_in_video_space(pred_audio, fps, nf)
		print(f"✅ Predictions: {np.sum(pred_video > self.prediction_threshold)} active frames out of {nf}")
		
		# Create prediction dataframe
		print("📊 Creating prediction dataframe...")
		df = self.create_frame_level_dataframe(pred_video, gt_video, fps, nf)
		print(f"✅ Created dataframe with {len(df)} frames")
		
		# Create plot
		print("📈 Creating prediction plot...")
		intervals_df, frame_df, pred_intervals, gt_intervals, matched_pred, matched_gt = self.create_intervals_dataframe(df, fps, tolerance_sec=0.25)
		
		basename = os.path.splitext(os.path.basename(video_path))[0]
		plot_path = os.path.join(output_dir, f"{basename}_predictions.png")
		self.plot_predictions(frame_df, intervals_df, fps, plot_path)
		
		# Dump CSVs
		self.dump_intervals_csv(intervals_df, fps, output_dir, basename)
		
		# Create video overlay
		video_out_path = os.path.join(output_dir, f"{basename}_overlay.mp4")
		self.create_video_overlay(frame_df, video_path, video_out_path, fps, w, h)
		
		# Return results
		return {
			'video_path': video_path,
			'output_dir': output_dir,
			'plot_path': plot_path,
			'video_out_path': video_out_path,
			'fold_id': fold,
			'num_hits': len(hits),
			'prediction_frames': np.sum(pred_video > self.prediction_threshold),
			'gt_frames': np.sum(gt_video),
			'intervals_df': intervals_df
		}
	
	def _print_summary(self, results, train_count, val_count, val_fold):
		"""Print processing summary."""
		print(f"\n📊 Processing Summary:")
		print(f"Total videos processed: {len(results)}")
		print(f"Train videos: {train_count}")
		print(f"Val videos: {val_count}")
		
		train_results = [r for r in results if r['split'] == 'train']
		val_results = [r for r in results if r['split'] == 'val']
		
		if train_results:
			train_hits = sum(r['num_hits'] for r in train_results)
			train_pred_frames = sum(r['prediction_frames'] for r in train_results)
			train_gt_frames = sum(r['gt_frames'] for r in train_results)
			print(f"Train total hits: {train_hits}")
			print(f"Train total prediction frames: {train_pred_frames}")
			print(f"Train total ground truth frames: {train_gt_frames}")
		
		if val_results:
			val_hits = sum(r['num_hits'] for r in val_results)
			val_pred_frames = sum(r['prediction_frames'] for r in val_results)
			val_gt_frames = sum(r['gt_frames'] for r in val_results)
			print(f"Val total hits: {val_hits}")
			print(f"Val total prediction frames: {val_pred_frames}")
			print(f"Val total ground truth frames: {val_gt_frames}")
		
		# Print fold distribution
		print(f"\n📈 Fold Distribution:")
		fold_counts = {}
		for result in results:
			fold = result['fold_id']
			fold_counts[fold] = fold_counts.get(fold, 0) + 1
		
		for fold in sorted(fold_counts.keys()):
			status = "VAL" if fold == val_fold else "TRAIN"
			print(f"  Fold {fold}: {fold_counts[fold]} videos ({status})")
		
		print(f"\n🎉 Batch processing complete!")
		print(f"📁 Check outputs in: {self.output_dir}") 
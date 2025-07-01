#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_visualizer.py

Generate a same-fps MP4 with synced audio and an alpha-blended
hit-detection overlay (green = TP, yellow = FP, red = FN).

✓	in-memory feature extraction (log-Mel)
✓	sliding-window inference with CRNNLightning
✓	alpha overlay per frame
✓	remuxes original audio to keep sync
✓	tabs only
✓	dataframe collection first
✓	plot with color-coded regions
"""

import os, subprocess, tempfile, math, cv2, torch, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from decorte_data_loader import load_decorte_dataset
from audio_features import _ffmpeg_audio, _mbe
from crnn_lightning import CRNNLightning
import audio_features as af
from train_constants import *
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────
# Paths (edit if needed)
# ─────────────────────────────────────────────────────────────
VIDEO_PATH		= "/home/noams/src/plai_cv/data/decorte/rallies/20230528_VIGO_04.mp4"
CKPT_PATH		= "/sed-crnn/train_artifacts/20250630_182940/fold1/epochepoch=015-valerval_er_1s=0.150.ckpt"
OUT_DIR			= "/home/noams/src/plai_cv/output/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)
BASENAME		= os.path.splitext(os.path.basename(VIDEO_PATH))[0]
VIDEO_OUT_PATH	= os.path.join(OUT_DIR, f"{BASENAME}_overlay.mp4")
PLOT_OUT_PATH	= os.path.join(OUT_DIR, f"{BASENAME}_predictions.png")

ALPHA			= 0.5
DEVICE			= "cuda" if torch.cuda.is_available() else "cpu"
PREDICTION_THRESHOLD = 0.5  # Configurable threshold for prediction

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def blend(frame, color):
	overlay = np.full_like(frame, color, dtype=np.uint8)
	return cv2.addWeighted(frame, 1-ALPHA, overlay, ALPHA, 0)

def sliding_windows(mbe: np.ndarray, win: int = SEQ_LEN_IN, stride: int = SEQ_LEN_OUT):
	wins, starts = [], []
	for s in range(0, mbe.shape[0] - win + 1, stride):
		wins.append(mbe[s:s + win].T)
		starts.append(s)
	return np.array(wins), np.array(starts)

def create_frame_level_dataframe(pred_video, gt_video, fps, nf):
	"""Create a dataframe with frame-level predictions and ground truth
	
	Args:
		pred_video: predictions in video frame space
		gt_video: ground truth in video frame space  
		fps: video frame rate
		nf: number of video frames
	"""
	# Create dataframe with video-space data
	video_times = np.arange(nf) / fps
	
	df = pd.DataFrame({
		'frame': range(nf),
		'time': video_times,
		'prediction': pred_video,
		'ground_truth': gt_video,
		'pred_binary': pred_video > PREDICTION_THRESHOLD,
		'gt_binary': gt_video > 0.5
	})
	
	return df

def extract_intervals(binary_array):
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

def create_intervals_dataframe(frame_df, fps, tolerance_sec=0.25):
	"""Create a dataframe with event intervals and their classification (TP/FP/FN)"""
	pred_intervals = extract_intervals(frame_df['pred_binary'].values)
	gt_intervals = extract_intervals(frame_df['gt_binary'].values)
	
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

def plot_predictions(frame_df, intervals_df, fps, save_path):
	"""Create prediction plot using the frame-level dataframe with colors"""
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
	ax.axhline(PREDICTION_THRESHOLD, color='black', linestyle='--', linewidth=1, label=f'Threshold={PREDICTION_THRESHOLD}')
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

def dump_intervals_csv(intervals_df, fps, out_dir, basename):
	"""Dump intervals to CSV files"""
	# GT CSV
	gt_df = intervals_df[intervals_df['type'] == 'ground_truth'].copy()
	gt_df.to_csv(os.path.join(out_dir, f'{basename}_ground_truth.csv'), index=False)
	
	# Pred CSV
	pred_df = intervals_df[intervals_df['type'] == 'prediction'].copy()
	pred_df.to_csv(os.path.join(out_dir, f'{basename}_predictions.csv'), index=False)
	
	# Both CSV
	intervals_df.to_csv(os.path.join(out_dir, f'{basename}_intervals.csv'), index=False)
	print(f"✅ Saved CSVs to {out_dir}")

def create_video_overlay(frame_df, video_path, output_path, fps, width, height):
	"""Create video overlay using frame-level dataframe colors"""
	print("🎬 Creating video overlay...")
	print("Unique frame colors in video:", frame_df['color'].unique())
	
	tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	writer = cv2.VideoWriter(tmp_vid, fourcc, fps, (width, height))
	cap = cv2.VideoCapture(video_path)

	for i, row in frame_df.iterrows():
		ret, frame = cap.read()
		if not ret: 
			break
		
		color = row['color']
		if color == 'green':
			frame = blend(frame, (0, 255, 0))
		elif color == 'yellow':
			frame = blend(frame, (0, 255, 255))
		elif color == 'red':
			frame = blend(frame, (0, 0, 255))
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

def create_ground_truth_in_video_space(hits, fps, nf):
	"""Create ground truth labels in video frame space"""
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

def create_predictions_in_video_space(pred_full, fps, nf):
	"""Convert audio frame predictions to video frame predictions"""
	audio_frames = len(pred_full)
	
	# Create time arrays using constants from train_constants.py
	audio_times = np.arange(audio_frames) / FPS_ORIG
	video_times = np.arange(nf) / fps
	
	# Interpolate predictions from audio time to video time
	pred_interpolator = interp1d(audio_times, pred_full, kind='linear', bounds_error=False, fill_value=0)
	pred_video = pred_interpolator(video_times)
	
	return pred_video

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
	# ── Load metadata & hits ─────────────────────────────
	vname = os.path.basename(VIDEO_PATH)
	meta_all = load_decorte_dataset()
	if vname not in meta_all:
		raise RuntimeError(f"{vname} not in Decorte metadata")
	meta = meta_all[vname]
	hits = meta["hits"]
	fold = meta["fold_id"]
	fold_cache = fold + 1	# scaler is 1-indexed

	# ── Load scaler from ckpt dir or fallback to cache ───
	scaler_path = os.path.join(os.path.dirname(CKPT_PATH), f"scaler_fold{fold_cache}.joblib")
	if not os.path.exists(scaler_path):
		scaler_path = os.path.join(CACHE_DIR, f"scaler_fold{fold_cache}.joblib")
	scaler = af.load_scaler(scaler_path)
	print(f"✅ Loaded scaler from: {scaler_path}")

	# ── Decode audio → MBE → normalize ───────────────────
	y = _ffmpeg_audio(VIDEO_PATH, SAMPLE_RATE)
	mbe = _mbe(y, SAMPLE_RATE)
	mbe = af.normalize(mbe, scaler)

	# ── Prepare video I/O first to get fps and frame count ─────────
	cap = cv2.VideoCapture(VIDEO_PATH)
	fps = cap.get(cv2.CAP_PROP_FPS)
	w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	nf   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()
	
	print(f"📹 Video: {nf} frames at {fps} fps")
	print(f"🎵 Audio: {mbe.shape[0]} frames at {FPS_ORIG} fps (from train_constants.py)")

	# ── Create ground truth in video space ─────────────────────────
	print("🎯 Creating ground truth in video space...")
	gt_video = create_ground_truth_in_video_space(hits, fps, nf)
	print(f"✅ Ground truth: {np.sum(gt_video)} active frames out of {nf}")

	# ── Inference windows ────────────────────────────────
	win_x, win_starts = sliding_windows(mbe)
	tensor_x = torch.from_numpy(win_x).unsqueeze(1).float()
	loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(tensor_x),
		batch_size = 64,
		shuffle    = False,
		pin_memory = True
	)

	# ── Run inference ────────────────────────────────────
	model = CRNNLightning.load_from_checkpoint(CKPT_PATH, fold_id=fold, art_dir="/tmp").to(DEVICE)
	trainer = Trainer(accelerator=DEVICE, devices=1, logger=False, enable_checkpointing=False)
	logits  = trainer.predict(model, loader)
	
	# Fix the type issue with logits
	if logits is not None:
		preds = torch.cat([torch.tensor(batch) for batch in logits], 0).sigmoid().squeeze(-1).cpu().numpy().reshape(-1)
	else:
		raise RuntimeError("No predictions returned from model")

	# ── Map to per-frame predictions in audio space ───────────────
	pred_audio = np.zeros(mbe.shape[0], np.float32)
	for i, start in enumerate(win_starts):
		pred_audio[start:start + SEQ_LEN_OUT] = preds[i * SEQ_LEN_OUT : (i + 1) * SEQ_LEN_OUT]

	# ── Convert predictions to video space ─────────────────────────
	print("🔄 Converting predictions from audio space to video space...")
	pred_video = create_predictions_in_video_space(pred_audio, fps, nf)
	print(f"✅ Predictions: {np.sum(pred_video > PREDICTION_THRESHOLD)} active frames out of {nf}")

	# ── Create prediction dataframe ────────────────────────────────
	print("📊 Creating prediction dataframe...")
	df = create_frame_level_dataframe(pred_video, gt_video, fps, nf)
	print(f"✅ Created dataframe with {len(df)} frames")
	
	# ── Create plot ──────────────────────────────────────
	print("📈 Creating prediction plot...")
	intervals_df, frame_df, pred_intervals, gt_intervals, matched_pred, matched_gt = create_intervals_dataframe(df, fps, tolerance_sec=0.25)
	plot_predictions(frame_df, intervals_df, fps, PLOT_OUT_PATH)
	# ── Dump CSVs ────────────────────────────────────────
	dump_intervals_csv(intervals_df, fps, OUT_DIR, BASENAME)

	# ── Create video overlay ──────────────────────────────
	create_video_overlay(frame_df, VIDEO_PATH, VIDEO_OUT_PATH, fps, w, h)

if __name__ == "__main__":
	main()

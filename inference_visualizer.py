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
✓	debugging with interval dataframes and statistics
✓	tabs only
"""

import os, subprocess, tempfile, math, cv2, torch, numpy as np, pandas as pd
from pytorch_lightning import Trainer
from decorte_data_loader import load_decorte_dataset
from decorte_datamodule import _ffmpeg_audio, _mbe
from crnn_lightning import CRNNLightning
from train_constants import *
from metrics import f1_overall_framewise, er_overall_framewise

# ─────────────────────────────────────────────────────────────
# Paths (edit if needed)
# ─────────────────────────────────────────────────────────────
VIDEO_PATH		= "/home/noams/src/plai_cv/data/decorte/rallies/20230528_VIGO_04.mp4"
CKPT_PATH		= "/home/noams/src/plai_cv/sed-crnn/train_artifacts/20250627_181038/fold3/epochepoch=022-valerval_er_1s=0.162.ckpt"
OUT_DIR			= "/home/noams/src/plai_cv/output/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)
BASENAME		= os.path.splitext(os.path.basename(VIDEO_PATH))[0]
VIDEO_OUT		= os.path.join(OUT_DIR, f"{BASENAME}_overlay.mp4")

ALPHA			= 0.5
DEVICE			= "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def blend(frame, color):
	overlay = np.full_like(frame, color, dtype=np.uint8)
	return cv2.addWeighted(frame, 1-ALPHA, overlay, ALPHA, 0)

def sliding_windows(mbe: np.ndarray, win: int = SEQ_LEN_IN, stride: int = SEQ_LEN_OUT):
	wins, starts = [], []
	for s in range(0, mbe.shape[0] - win + 1, stride):
		wins.append(mbe[s:s + win].T)		# (40, win)
		starts.append(s)
	return np.array(wins), np.array(starts)

def find_hit_intervals(hit_detection_frames, audio_feature_fps, video_fps):
	"""
	Convert hit detection frames to list of intervals in video frame coordinates.
	
	Args:
		hit_detection_frames: 1D numpy array where 1 = hit detected, 0 = no hit (from mel spectrogram frames)
		audio_feature_fps: FPS of the audio features (~43 Hz from mel spectrogram)
		video_fps: FPS of the video (typically 25 or 30 fps)
	
	Returns:
		List of (start_frame, end_frame) tuples in video coordinates
	"""
	intervals = []
	in_hit = False
	start_idx = 0
	
	for i, hit_detected in enumerate(hit_detection_frames):
		if hit_detected == 1 and not in_hit:
			# Start of hit
			in_hit = True
			start_idx = i
		elif hit_detected == 0 and in_hit:
			# End of hit
			in_hit = False
			end_idx = i - 1
			
			# Convert audio feature frame indices to video frame indices
			start_video_frame = int(start_idx * video_fps / audio_feature_fps)
			end_video_frame = int(end_idx * video_fps / audio_feature_fps)
			
			intervals.append((start_video_frame, end_video_frame))
	
	# Handle case where hit extends to end of array
	if in_hit:
		end_idx = len(hit_detection_frames) - 1
		start_video_frame = int(start_idx * video_fps / audio_feature_fps)
		end_video_frame = int(end_idx * video_fps / audio_feature_fps)
		intervals.append((start_video_frame, end_video_frame))
	
	return intervals

def compute_overlap_stats(ground_truth_intervals, prediction_intervals):
	"""
	Compute statistics about ground truth and prediction intervals.
	
	Args:
		ground_truth_intervals: List of (start, end) tuples for ground truth hits
		prediction_intervals: List of (start, end) tuples for predicted hits
	
	Returns:
		Dictionary with statistics
	"""
	# Convert to binary arrays for frame-wise metrics
	max_frame = max(
		max([end for _, end in ground_truth_intervals]) if ground_truth_intervals else 0,
		max([end for _, end in prediction_intervals]) if prediction_intervals else 0
	)
	
	ground_truth_frames = np.zeros(max_frame + 1, dtype=np.uint8)
	prediction_frames = np.zeros(max_frame + 1, dtype=np.uint8)
	
	for start, end in ground_truth_intervals:
		ground_truth_frames[start:end+1] = 1
	
	for start, end in prediction_intervals:
		prediction_frames[start:end+1] = 1
	
	# Compute metrics
	f1 = f1_overall_framewise(prediction_frames.reshape(-1, 1), ground_truth_frames.reshape(-1, 1))
	er = er_overall_framewise(prediction_frames.reshape(-1, 1), ground_truth_frames.reshape(-1, 1))
	
	# Count overlaps
	tp = np.logical_and(ground_truth_frames == 1, prediction_frames == 1).sum()
	fp = np.logical_and(ground_truth_frames == 0, prediction_frames == 1).sum()
	fn = np.logical_and(ground_truth_frames == 1, prediction_frames == 0).sum()
	tn = np.logical_and(ground_truth_frames == 0, prediction_frames == 0).sum()
	
	return {
		'num_gt_intervals': len(ground_truth_intervals),
		'num_pred_intervals': len(prediction_intervals),
		'gt_total_frames': ground_truth_frames.sum(),
		'pred_total_frames': prediction_frames.sum(),
		'tp_frames': tp,
		'fp_frames': fp,
		'fn_frames': fn,
		'tn_frames': tn,
		'f1_score': f1,
		'error_rate': er,
		'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
		'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
	}

def create_interval_dataframe(ground_truth_intervals, prediction_intervals):
	"""
	Create a sorted dataframe of all intervals for visualization.
	
	Args:
		ground_truth_intervals: List of (start, end) tuples for ground truth hits
		prediction_intervals: List of (start, end) tuples for predicted hits
	
	Returns:
		DataFrame with columns: start_frame, end_frame, type, color
	"""
	rows = []
	
	# Add ground truth intervals
	for start, end in ground_truth_intervals:
		rows.append({
			'start_frame': start,
			'end_frame': end,
			'type': 'ground_truth',
			'color': (0, 0, 255)  # Red for ground truth
		})
	
	# Add prediction intervals
	for start, end in prediction_intervals:
		rows.append({
			'start_frame': start,
			'end_frame': end,
			'type': 'prediction',
			'color': (0, 255, 255)  # Yellow for predictions
		})
	
	# Sort by start frame
	df = pd.DataFrame(rows)
	if not df.empty:
		df = df.sort_values('start_frame').reset_index(drop=True)
	
	return df

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
	# ── Load metadata & ground-truth hits ──────────────────
	vname = os.path.basename(VIDEO_PATH)
	meta_all = load_decorte_dataset()
	if vname not in meta_all:
		raise RuntimeError(f"{vname} not in Decorte metadata")
	meta   = meta_all[vname]
	hits   = meta["hits"]
	fold   = meta["fold_id"]

	# ── Decode audio & build MBE + label vector ────────────
	print("🔊 Extracting audio features...")
	y   = _ffmpeg_audio(VIDEO_PATH, SAMPLE_RATE)
	mbe = _mbe(y, SAMPLE_RATE)				# (frames, 40)
	lbl = np.zeros((mbe.shape[0], 1), np.float32)
	for _, h in hits.iterrows():
		s = int(math.floor(h["start"] * SAMPLE_RATE / HOP_LENGTH))
		e = int(math.ceil (h["end"]   * SAMPLE_RATE / HOP_LENGTH))
		lbl[s:e, 0] = 1.0

	# ── Sliding-window tensor dataset ──────────────────────
	print("🔄 Creating sliding windows...")
	win_x, win_starts = sliding_windows(mbe)
	tensor_x = torch.from_numpy(win_x).unsqueeze(1).float()		# (N,1,40,L)
	loader   = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(tensor_x),
		batch_size = 64,
		shuffle    = False,
		pin_memory = True
	)

	# ── Run inference ──────────────────────────────────────
	print("🤖 Running model inference...")
	model   = CRNNLightning.load_from_checkpoint(CKPT_PATH, fold_id=fold, art_dir="/tmp").to(DEVICE)
	trainer = Trainer(accelerator=DEVICE, devices=1, logger=False, enable_checkpointing=False)
	logits  = trainer.predict(model, loader)
	preds   = torch.cat(logits, 0).sigmoid().squeeze(-1).cpu().numpy().reshape(-1)

	print(f"📊 Raw predictions shape: {preds.shape}")
	print(f"📊 Raw predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
	print(f"📊 Raw predictions mean: {preds.mean():.4f}")
	print(f"📊 Raw predictions > 0.5: {(preds > 0.5).sum()} / {len(preds)}")

	# Map window predictions → per-MBE frame (SEQ_LEN_OUT rate)
	pred_accum = np.zeros(mbe.shape[0], np.float32)
	pred_mask = np.zeros(mbe.shape[0], np.uint8)

	for i, s in enumerate(win_starts):
		chunk = preds[i * SEQ_LEN_OUT: (i + 1) * SEQ_LEN_OUT]
		e = s + SEQ_LEN_OUT
		if e > mbe.shape[0]:
			chunk = chunk[:mbe.shape[0] - s]
			e = mbe.shape[0]
		pred_accum[s:e] = np.maximum(pred_accum[s:e], chunk)
		pred_mask[s:e] = 1  # mark that this region has predictions

	print(f"📊 Accumulated predictions shape: {pred_accum.shape}")
	print(f"📊 Accumulated predictions range: [{pred_accum.min():.4f}, {pred_accum.max():.4f}]")
	print(f"📊 Accumulated predictions mean: {pred_accum.mean():.4f}")
	print(f"📊 Accumulated predictions > 0.5: {(pred_accum > 0.5).sum()} / {len(pred_accum)}")

	# Get video properties
	cap   = cv2.VideoCapture(VIDEO_PATH)
	video_fps = cap.get(cv2.CAP_PROP_FPS)
	w, h  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	nf    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()

	audio_feature_fps = SAMPLE_RATE / HOP_LENGTH  # ~43 Hz

	print(f"🎬 Video: {nf} frames at {video_fps:.2f} fps")
	print(f"🔊 Audio features: {len(pred_accum)} frames at {audio_feature_fps:.2f} fps")

	# ── Create interval dataframes ──────────────────────────
	print("📋 Creating interval dataframes...")
	
	# Convert hit detection arrays to intervals
	ground_truth_hit_frames = (lbl.squeeze() > 0.5).astype(np.uint8)
	predicted_hit_frames = (pred_accum > 0.5).astype(np.uint8)
	
	gt_intervals = find_hit_intervals(ground_truth_hit_frames, audio_feature_fps, video_fps)
	pred_intervals = find_hit_intervals(predicted_hit_frames, audio_feature_fps, video_fps)
	
	# Create dataframe
	interval_df = create_interval_dataframe(gt_intervals, pred_intervals)
	
	# Compute statistics
	stats = compute_overlap_stats(gt_intervals, pred_intervals)
	
	# ── Log statistics ──────────────────────────────────────
	print("\n" + "="*60)
	print("📈 INTERVAL STATISTICS")
	print("="*60)
	print(f"Ground truth intervals: {stats['num_gt_intervals']}")
	print(f"Prediction intervals: {stats['num_pred_intervals']}")
	print(f"Ground truth frames: {stats['gt_total_frames']}")
	print(f"Prediction frames: {stats['pred_total_frames']}")
	print(f"True positives: {stats['tp_frames']}")
	print(f"False positives: {stats['fp_frames']}")
	print(f"False negatives: {stats['fn_frames']}")
	print(f"True negatives: {stats['tn_frames']}")
	print(f"F1 Score: {stats['f1_score']:.4f}")
	print(f"Error Rate: {stats['error_rate']:.4f}")
	print(f"Precision: {stats['precision']:.4f}")
	print(f"Recall: {stats['recall']:.4f}")
	print("="*60)
	
	# Save interval dataframe
	interval_csv = os.path.join(OUT_DIR, f"{BASENAME}_intervals.csv")
	interval_df.to_csv(interval_csv, index=False)
	print(f"💾 Saved interval dataframe: {interval_csv}")
	
	# Save statistics
	stats_csv = os.path.join(OUT_DIR, f"{BASENAME}_stats.csv")
	pd.DataFrame([stats]).to_csv(stats_csv, index=False)
	print(f"💾 Saved statistics: {stats_csv}")

	# ── Create video with interval-based coloring ───────────
	print("🎬 Creating video with interval-based coloring...")
	
	# Create frame-level color mapping
	frame_colors = {}
	
	# Process ground truth intervals (red)
	for start, end in gt_intervals:
		for frame in range(start, end + 1):
			if frame not in frame_colors:
				frame_colors[frame] = (0, 0, 255)  # Red
	
	# Process prediction intervals (yellow/green)
	for start, end in pred_intervals:
		for frame in range(start, end + 1):
			if frame in frame_colors:
				# Overlap with ground truth - make it green (TP)
				frame_colors[frame] = (0, 255, 0)  # Green
			else:
				# No overlap - make it yellow (FP)
				frame_colors[frame] = (0, 255, 255)  # Yellow

	# ── Prepare video I/O ──────────────────────────────────
	tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
	writer  = cv2.VideoWriter(tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))

	# ── Frame loop with overlay ────────────────────────────
	cap = cv2.VideoCapture(VIDEO_PATH)
	for i in range(nf):
		ret, frame = cap.read()
		if not ret: break
		
		if i in frame_colors:
			frame = blend(frame, frame_colors[i])
		
		writer.write(frame)

	cap.release()
	writer.release()

	# ── Remux original audio to keep sync ──────────────────
	print("🔊 Remuxing audio...")
	subprocess.check_call([
		"ffmpeg", "-y", "-loglevel", "error",
		"-i", tmp_vid,
		"-i", VIDEO_PATH,
		"-c:v", "copy",
		"-map", "0:v:0", "-map", "1:a:0",
		"-shortest", VIDEO_OUT
	])
	os.remove(tmp_vid)
	print(f"✅ Saved {VIDEO_OUT}")

if __name__ == "__main__":
	main()

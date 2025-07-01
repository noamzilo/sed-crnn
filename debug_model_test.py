#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_model_test.py

Debug script to test model behavior on test-set videos and analyze prediction issues.
"""

import os, math, torch, numpy as np, pandas as pd
from pytorch_lightning import Trainer
from decorte_data_loader import load_decorte_dataset
from decorte_datamodule import _ffmpeg_audio, _mbe, DecorteDataModule
from crnn_lightning import CRNNLightning
from train_constants import *
from metrics import f1_overall_framewise, er_overall_framewise
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
CKPT_PATH = "/sed-crnn/train_artifacts/20250629_190557/fold1/epochepoch=026-valerval_er_1s=0.142.ckpt"
CACHE_DIR = os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def sliding_windows(mbe: np.ndarray, win: int = SEQ_LEN_IN, stride: int = SEQ_LEN_OUT):
	"""Create sliding windows for inference"""
	wins, starts = [], []
	for s in range(0, mbe.shape[0] - win + 1, stride):
		wins.append(mbe[s:s + win].T)		# (40, win)
		starts.append(s)
	return np.array(wins), np.array(starts)

def analyze_predictions(pred_accum, lbl, vname, fold_id):
	"""Analyze prediction statistics"""
	print(f"\n📊 Prediction Analysis for {vname} (fold {fold_id}):")
	print(f"Ground truth frames: {lbl.sum():.0f}")
	print(f"Prediction frames > 0.5: {(pred_accum > 0.5).sum():.0f}")
	print(f"Prediction frames > 0.3: {(pred_accum > 0.3).sum():.0f}")
	print(f"Prediction frames > 0.1: {(pred_accum > 0.1).sum():.0f}")
	print(f"Prediction range: [{pred_accum.min():.4f}, {pred_accum.max():.4f}]")
	print(f"Prediction mean: {pred_accum.mean():.4f}")
	print(f"Prediction std: {pred_accum.std():.4f}")
	
	# Test different thresholds
	thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	print(f"\n📈 Threshold Analysis:")
	for thresh in thresholds:
		pred_binary = (pred_accum > thresh).astype(np.uint8)
		f1 = f1_overall_framewise(pred_binary.reshape(-1, 1), lbl)
		er = er_overall_framewise(pred_binary.reshape(-1, 1), lbl)
		print(f"  Threshold {thresh:.1f}: F1={f1:.4f}, ER={er:.4f}, Pred frames={(pred_binary > 0).sum()}")

def test_video_inference(video_path, fold_id, model, apply_normalization=True):
	"""Test inference on a specific video"""
	print(f"\n🎬 Testing inference on: {os.path.basename(video_path)}")
	
	# Load metadata
	meta_all = load_decorte_dataset()
	vname = os.path.basename(video_path)
	if vname not in meta_all:
		raise RuntimeError(f"{vname} not in Decorte metadata")
	
	meta = meta_all[vname]
	hits = meta["hits"]
	actual_fold = meta["fold_id"]
	
	print(f"Video fold: {actual_fold}, Model fold: {fold_id}")
	print(f"Number of ground truth hits: {len(hits)}")
	
	# Extract audio features
	print("🔊 Extracting audio features...")
	y = _ffmpeg_audio(video_path, SAMPLE_RATE)
	mbe = _mbe(y, SAMPLE_RATE)				# (frames, 40)
	
	# Apply normalization if requested (same as training)
	if apply_normalization:
		print("🔧 Applying StandardScaler normalization...")
		# Load the scaler from training data
		dm = DecorteDataModule(fold_id=fold_id, cache_dir=CACHE_DIR)
		dm.setup()
		
		# Get training data for scaler
		train_x = dm.train_ds.mel
		scaler = StandardScaler()
		scaler.fit(train_x)
		mbe = scaler.transform(mbe)
		print(f"Normalized MBE stats - mean: {mbe.mean():.4f}, std: {mbe.std():.4f}")
	
	# Create ground truth labels
	lbl = np.zeros((mbe.shape[0], 1), np.float32)
	for _, h in hits.iterrows():
		s = int(math.floor(h["start"] * SAMPLE_RATE / HOP_LENGTH))
		e = int(math.ceil (h["end"]   * SAMPLE_RATE / HOP_LENGTH))
		lbl[s:e, 0] = 1.0
	
	# Create sliding windows
	print("🔄 Creating sliding windows...")
	win_x, win_starts = sliding_windows(mbe)
	tensor_x = torch.from_numpy(win_x).unsqueeze(1).float()		# (N,1,40,L)
	loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(tensor_x),
		batch_size=64,
		shuffle=False,
		pin_memory=True
	)
	
	# Run inference
	print("🤖 Running model inference...")
	trainer = Trainer(accelerator=DEVICE, devices=1, logger=False, enable_checkpointing=False)
	logits = trainer.predict(model, loader)
	print("👀 First batch raw logits:")
	print(torch.cat(logits[:1], 0).squeeze().cpu().numpy().reshape(-1))
	preds = torch.cat(logits, 0).sigmoid().squeeze(-1).cpu().numpy().reshape(-1)
	print(f"Sigmoid min/max/mean: {preds.min():.4f} / {preds.max():.4f} / {preds.mean():.4f}")
	
	print(f"Raw predictions shape: {preds.shape}")
	print(f"Raw predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
	print(f"Raw predictions mean: {preds.mean():.4f}")
	
	# Map window predictions to per-MBE frame
	pred_accum = np.zeros(mbe.shape[0], np.float32)
	pred_mask = np.zeros(mbe.shape[0], np.uint8)
	
	for i, s in enumerate(win_starts):
		if i == 0:
			np.save("/tmp/infer_window.npy", win_x[0])
		chunk = preds[i * SEQ_LEN_OUT: (i + 1) * SEQ_LEN_OUT]
		e = s + SEQ_LEN_OUT
		if e > mbe.shape[0]:
			chunk = chunk[:mbe.shape[0] - s]
			e = mbe.shape[0]
		pred_accum[s:e] = np.maximum(pred_accum[s:e], chunk)
		pred_mask[s:e] = 1
	
	# Analyze predictions
	analyze_predictions(pred_accum, lbl, vname, actual_fold)
	
	return pred_accum, lbl, hits

def list_fold_videos():
	"""List videos by fold for debugging"""
	print("📋 Video fold assignments:")
	meta_all = load_decorte_dataset()
	
	fold_videos = {0: [], 1: [], 2: [], 3: []}
	for vname, info in meta_all.items():
		fold_videos[info["fold_id"]].append(vname)
	
	for fold_id in range(4):
		print(f"\nFold {fold_id}:")
		for video in sorted(fold_videos[fold_id]):
			print(f"  {video}")

def test_training_data_consistency():
	"""Test if training data is consistent"""
	print("\n🔍 Testing training data consistency...")
	
	dm = DecorteDataModule(fold_id=3, cache_dir=CACHE_DIR)
	dm.setup()
	
	print(f"Training dataset size: {len(dm.train_ds)}")
	print(f"Validation dataset size: {len(dm.val_ds)}")
	
	# Check a few training samples
	train_loader = dm.train_dataloader()
	batch = next(iter(train_loader))
	x, y = batch
	
	print(f"Training batch shapes: x={x.shape}, y={y.shape}")
	print(f"Training labels range: [{y.min():.4f}, {y.max():.4f}]")
	print(f"Training labels mean: {y.mean():.4f}")
	print(f"Training labels > 0.5: {(y > 0.5).sum().item()}")
	
	# Check validation samples
	val_loader = dm.val_dataloader()
	batch = next(iter(val_loader))
	x, y = batch
	
	print(f"Validation batch shapes: x={x.shape}, y={y.shape}")
	print(f"Validation labels range: [{y.min():.4f}, {y.max():.4f}]")
	print(f"Validation labels mean: {y.mean():.4f}")
	print(f"Validation labels > 0.5: {(y > 0.5).sum().item()}")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
	print("🔍 Model Debug Test")
	print("=" * 60)
	
	# List fold assignments
	list_fold_videos()
	
	# Test training data consistency
	test_training_data_consistency()
	
	# Load model
	print(f"\n🤖 Loading model from: {CKPT_PATH}")
	model = CRNNLightning.load_from_checkpoint(CKPT_PATH, fold_id=3, art_dir="/tmp").to(DEVICE)
	model.eval()
	for m in model.modules():
		if isinstance(m, torch.nn.Dropout):
			assert m.training is False
	for m in model.modules():
		if isinstance(m, torch.nn.BatchNorm2d):
			assert m.training is False
	
	# Test on a video from fold 3 (test set for this model)
	meta_all = load_decorte_dataset()
	fold3_videos = [vname for vname, info in meta_all.items() if info["fold_id"] == 3]
	
	if not fold3_videos:
		print("❌ No videos found in fold 3")
		return
	
	test_video = fold3_videos[0]
	test_video_path = meta_all[test_video]["video_meta"]["video_path"]
	
	print(f"\n🎯 Testing on fold 3 video: {test_video}")
	
	# Test without normalization first
	print("\n" + "="*50)
	print("TEST 1: Without normalization")
	print("="*50)
	pred_accum1, lbl1, hits1 = test_video_inference(test_video_path, 3, model, apply_normalization=False)
	
	# Test with normalization
	print("\n" + "="*50)
	print("TEST 2: With normalization")
	print("="*50)
	pred_accum2, lbl2, hits2 = test_video_inference(test_video_path, 3, model, apply_normalization=True)
	
	# Save detailed results
	results_dir = "/home/noams/src/plai_cv/output/debug_results"
	os.makedirs(results_dir, exist_ok=True)
	
	# Save predictions
	results_df = pd.DataFrame({
		'frame_idx': range(len(pred_accum1)),
		'prediction_no_norm': pred_accum1,
		'prediction_with_norm': pred_accum2,
		'ground_truth': lbl1.squeeze(),
		'pred_binary_0.5_no_norm': (pred_accum1 > 0.5).astype(np.uint8),
		'pred_binary_0.5_with_norm': (pred_accum2 > 0.5).astype(np.uint8)
	})
	
	results_path = os.path.join(results_dir, f"{test_video}_predictions.csv")
	results_df.to_csv(results_path, index=False)
	print(f"\n💾 Saved detailed results to: {results_path}")
	
	# Save ground truth hits
	hits_path = os.path.join(results_dir, f"{test_video}_hits.csv")
	hits1.to_csv(hits_path, index=False)
	print(f"💾 Saved ground truth hits to: {hits_path}")
	
	# Compare results
	print(f"\n📊 Comparison:")
	print(f"Without norm - F1: {f1_overall_framewise((pred_accum1 > 0.5).reshape(-1, 1), lbl1):.4f}")
	print(f"With norm    - F1: {f1_overall_framewise((pred_accum2 > 0.5).reshape(-1, 1), lbl2):.4f}")

if __name__ == "__main__":
	main() 
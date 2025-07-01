#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fold_based_visualizer.py

Flexible batch video processing app using SedRcnnInference class.
Processes all videos in the Decorte dataset and organizes outputs
into train/val folders based on fold assignments.

Allows specifying which fold to use as validation set.
Each video gets its own folder with:
- Video overlay with hit detection
- Prediction plots
- CSV files with intervals and classifications
"""

import os
import glob
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SedRcnnInference import SedRcnnInference
from decorte_data_loader import load_decorte_dataset

# ─────────────────────────────────────────────────────────────
# Configuration (edit these values as needed)
# ─────────────────────────────────────────────────────────────
CKPT_PATH = "/sed-crnn/train_artifacts/20250630_182940/fold1/epochepoch=015-valerval_er_1s=0.150.ckpt"
VIDEOS_DIR = "/home/noams/src/plai_cv/data/decorte/rallies"
OUTPUT_DIR = "/home/noams/src/plai_cv/output/batch_visualizations"
VAL_FOLD = 0  # Fold ID to use as validation set (0-3)
ALPHA = 0.5  # Alpha blending factor for video overlay
THRESHOLD = 0.5  # Prediction threshold for binary classification
DEVICE = ""  # Device to use (empty string for auto-detect)

def main():
	# Validate fold ID
	if not 0 <= VAL_FOLD <= 3:
		raise ValueError("VAL_FOLD must be between 0 and 3")
	
	# Create output directories
	train_output_dir = os.path.join(OUTPUT_DIR, "train")
	val_output_dir = os.path.join(OUTPUT_DIR, "val")
	os.makedirs(train_output_dir, exist_ok=True)
	os.makedirs(val_output_dir, exist_ok=True)
	
	print("🚀 Starting fold-based batch video processing...")
	print(f"📁 Videos directory: {VIDEOS_DIR}")
	print(f"📁 Train output: {train_output_dir}")
	print(f"📁 Val output: {val_output_dir}")
	print(f"🎯 Validation fold: {VAL_FOLD}")
	print(f"⚙️  Alpha: {ALPHA}, Threshold: {THRESHOLD}")
	
	# Initialize the inference visualizer
	visualizer = SedRcnnInference(
		ckpt_path=CKPT_PATH,
		output_base_dir=OUTPUT_DIR,
		alpha=ALPHA,
		prediction_threshold=THRESHOLD,
		device=DEVICE
	)
	
	# Load dataset metadata to get fold assignments
	meta_all = load_decorte_dataset()
	print(f"📊 Loaded metadata for {len(meta_all)} videos")
	
	# Get all video files
	video_files = glob.glob(os.path.join(VIDEOS_DIR, "*.mp4"))
	print(f"🎬 Found {len(video_files)} video files")
	
	# Process each video
	results = []
	train_count = 0
	val_count = 0
	
	for video_path in video_files:
		vname = os.path.basename(video_path)
		
		# Check if video is in metadata
		if vname not in meta_all:
			print(f"⚠️  Skipping {vname} - not in metadata")
			continue
		
		# Get fold assignment
		fold_id = meta_all[vname]["fold_id"]
		
		# Determine output directory based on fold
		if fold_id == VAL_FOLD:  # Validation set
			output_dir = os.path.join(val_output_dir, os.path.splitext(vname)[0])
			split = 'val'
			val_count += 1
		else:  # Training set
			output_dir = os.path.join(train_output_dir, os.path.splitext(vname)[0])
			split = 'train'
			train_count += 1
		
		print(f"\n🎯 Processing {vname} (fold {fold_id}) -> {split} -> {output_dir}")
		
		try:
			# Process the video
			result = visualizer.process_video(video_path, output_dir)
			result['fold_id'] = fold_id
			result['split'] = split
			results.append(result)
			print(f"✅ Successfully processed {vname}")
			
		except Exception as e:
			print(f"❌ Error processing {vname}: {str(e)}")
			continue
	
	# Print summary
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
		status = "VAL" if fold == VAL_FOLD else "TRAIN"
		print(f"  Fold {fold}: {fold_counts[fold]} videos ({status})")
	
	print(f"\n🎉 Batch processing complete!")
	print(f"📁 Check outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
	main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_visualizer.py

Batch video processing app using SedRcnnInference class.
Processes all videos in the Decorte dataset and organizes outputs
into train/val folders based on fold assignments.

Each video gets its own folder with:
- Video overlay with hit detection
- Prediction plots
- CSV files with intervals and classifications
"""

import os
import glob
from pathlib import Path
from SedRcnnInference import SedRcnnInference
from decorte_data_loader import load_decorte_dataset

def main():
	# Configuration
	CKPT_PATH = "/home/noams/src/plai_cv/sed-crnn/train_artifacts/20250630_182940/fold1/epochepoch=015-valerval_er_1s=0.150.ckpt"
	VIDEOS_DIR = "/home/noams/src/plai_cv/data/decorte/rallies"
	OUTPUT_BASE_DIR = "/home/noams/src/plai_cv/output/batch_visualizations"
	
	# Create output directories
	train_output_dir = os.path.join(OUTPUT_BASE_DIR, "train")
	val_output_dir = os.path.join(OUTPUT_BASE_DIR, "val")
	os.makedirs(train_output_dir, exist_ok=True)
	os.makedirs(val_output_dir, exist_ok=True)
	
	print("🚀 Starting batch video processing...")
	print(f"📁 Videos directory: {VIDEOS_DIR}")
	print(f"📁 Train output: {train_output_dir}")
	print(f"📁 Val output: {val_output_dir}")
	
	# Initialize the inference visualizer
	visualizer = SedRcnnInference(
		ckpt_path=CKPT_PATH,
		output_base_dir=OUTPUT_BASE_DIR,
		alpha=0.5,
		prediction_threshold=0.5
	)
	
	# Load dataset metadata to get fold assignments
	meta_all = load_decorte_dataset()
	print(f"📊 Loaded metadata for {len(meta_all)} videos")
	
	# Get all video files
	video_files = glob.glob(os.path.join(VIDEOS_DIR, "*.mp4"))
	print(f"🎬 Found {len(video_files)} video files")
	
	# Process each video
	results = []
	for video_path in video_files:
		vname = os.path.basename(video_path)
		
		# Check if video is in metadata
		if vname not in meta_all:
			print(f"⚠️  Skipping {vname} - not in metadata")
			continue
		
		# Get fold assignment
		fold_id = meta_all[vname]["fold_id"]
		
		# Determine output directory based on fold
		# For this example, we'll use fold 0 as validation, others as train
		# You can adjust this logic based on your specific fold assignments
		if fold_id == 0:  # Validation set
			output_dir = os.path.join(val_output_dir, os.path.splitext(vname)[0])
		else:  # Training set
			output_dir = os.path.join(train_output_dir, os.path.splitext(vname)[0])
		
		print(f"\n🎯 Processing {vname} (fold {fold_id}) -> {output_dir}")
		
		try:
			# Process the video
			result = visualizer.process_video(video_path, output_dir)
			result['fold_id'] = fold_id
			result['split'] = 'val' if fold_id == 0 else 'train'
			results.append(result)
			print(f"✅ Successfully processed {vname}")
			
		except Exception as e:
			print(f"❌ Error processing {vname}: {str(e)}")
			continue
	
	# Print summary
	print(f"\n📊 Processing Summary:")
	print(f"Total videos processed: {len(results)}")
	
	train_results = [r for r in results if r['split'] == 'train']
	val_results = [r for r in results if r['split'] == 'val']
	
	print(f"Train videos: {len(train_results)}")
	print(f"Val videos: {len(val_results)}")
	
	if train_results:
		train_hits = sum(r['num_hits'] for r in train_results)
		train_pred_frames = sum(r['prediction_frames'] for r in train_results)
		print(f"Train total hits: {train_hits}")
		print(f"Train total prediction frames: {train_pred_frames}")
	
	if val_results:
		val_hits = sum(r['num_hits'] for r in val_results)
		val_pred_frames = sum(r['prediction_frames'] for r in val_results)
		print(f"Val total hits: {val_hits}")
		print(f"Val total prediction frames: {val_pred_frames}")
	
	print(f"\n🎉 Batch processing complete!")
	print(f"📁 Check outputs in: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
	main() 
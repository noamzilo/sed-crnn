#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_visualizer.py

Single runner script for SED-CRNN video processing.
Handles both single video and batch processing with consistent configuration.

Usage:
    python run_visualizer.py
"""

# =============================================================================
# HARD-CODED CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Model checkpoint path
CHECKPOINT_PATH = "../../sed_crnn/train_artifacts/20250701_221947/fold1/epochepoch=022-valerval_er_1s=0.150.ckpt"

# Directory containing video files (for batch processing)
VIDEOS_DIR = "data/decorte/rallies"

# Single video path (for single video processing)
SINGLE_VIDEO_PATH = "data/decorte/rallies/20230528_VIGO_00.mp4"

# Base output directory
OUTPUT_DIR = "output/visualization"

# Validation fold (0-3) - only used for batch processing
VAL_FOLD = 0

# Alpha blending factor for video overlay (0.0-1.0)
ALPHA = 0.5

# Prediction threshold for binary classification
PREDICTION_THRESHOLD = 0.5

# Device to use (empty string for auto-detect, or "cuda", "cpu")
DEVICE = ""

# Processing mode: "single" or "batch"
MODE = "single"

# =============================================================================
# END CONFIGURATION
# =============================================================================

import os
import glob

from sed_crnn.CRNNInferenceVisualizer import CRNNInferenceVisualizer

def main():
	"""Main function to run video processing."""
	
	# Validate configuration
	assert os.path.isfile(CHECKPOINT_PATH), CHECKPOINT_PATH
	
	if MODE not in ["single", "batch"]:
		print(f"❌ Invalid mode: {MODE} (must be 'single' or 'batch')")
		return
	
	if not 0 <= VAL_FOLD <= 3:
		print(f"❌ Invalid validation fold: {VAL_FOLD} (must be 0-3)")
		return
	
	if not 0.0 <= ALPHA <= 1.0:
		print(f"❌ Invalid alpha value: {ALPHA} (must be 0.0-1.0)")
		return
	
	if not 0.0 <= PREDICTION_THRESHOLD <= 1.0:
		print(f"❌ Invalid prediction threshold: {PREDICTION_THRESHOLD} (must be 0.0-1.0)")
		return
	
	print("🚀 SED-CRNN Video Visualizer")
	print("=" * 50)
	print(f"📁 Checkpoint: {CHECKPOINT_PATH}")
	print(f"📁 Output: {OUTPUT_DIR}")
	print(f"🎯 Mode: {MODE}")
	print(f"⚙️  Alpha: {ALPHA}")
	print(f"⚙️  Threshold: {PREDICTION_THRESHOLD}")
	print(f"💻 Device: {DEVICE or 'auto-detect'}")
	print("=" * 50)
	
	# Create visualizer
	try:
		visualizer = CRNNInferenceVisualizer(
			ckpt_path=CHECKPOINT_PATH,
			output_dir=OUTPUT_DIR,
			alpha=ALPHA,
			prediction_threshold=PREDICTION_THRESHOLD,
			device=DEVICE
		)
	except Exception as e:
		print(f"❌ Failed to initialize visualizer: {str(e)}")
		return
	
	# Switch case to choose video paths
	video_paths = []
	
	if MODE == "single":
		assert os.path.isfile(SINGLE_VIDEO_PATH), SINGLE_VIDEO_PATH
		video_paths = [SINGLE_VIDEO_PATH]
		print(f"🎬 Single video: {SINGLE_VIDEO_PATH}")
		
	elif MODE == "batch":
		if not os.path.exists(VIDEOS_DIR):
			print(f"❌ Videos directory not found: {VIDEOS_DIR}")
			return
		video_paths = glob.glob(os.path.join(VIDEOS_DIR, "*.mp4"))
		if not video_paths:
			print(f"❌ No video files found in {VIDEOS_DIR}")
			return
		print(f"📁 Batch processing: {len(video_paths)} videos from {VIDEOS_DIR}")
		print(f"🎯 Val fold: {VAL_FOLD}")
	
	# Process videos
	try:
		results = visualizer.visualize_videos(video_paths, val_fold=VAL_FOLD)
		print(f"\n✅ Successfully processed {len(results)} videos!")
		
	except Exception as e:
		print(f"❌ Error during processing: {str(e)}")
		return

if __name__ == "__main__":
	main() 
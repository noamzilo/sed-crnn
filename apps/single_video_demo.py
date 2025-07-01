#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_video_demo.py

Demo script to verify SedRcnnInference class works with a single video.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SedRcnnInference import SedRcnnInference

def main():
	# Configuration
	CKPT_PATH = "/home/noams/src/plai_cv/sed-crnn/train_artifacts/20250630_182940/fold1/epochepoch=015-valerval_er_1s=0.150.ckpt"
	VIDEO_PATH = "/home/noams/src/plai_cv/data/decorte/rallies/20230528_VIGO_04.mp4"
	OUTPUT_DIR = "/home/noams/src/plai_cv/output/test_single_video"
	
	print("🧪 Testing SedRcnnInference with single video...")
	print(f"📁 Video: {VIDEO_PATH}")
	print(f"📁 Output: {OUTPUT_DIR}")
	
	# Initialize the inference visualizer
	visualizer = SedRcnnInference(
		ckpt_path=CKPT_PATH,
		output_base_dir=OUTPUT_DIR,
		alpha=0.5,
		prediction_threshold=0.5
	)
	
	try:
		# Process the video
		result = visualizer.process_video(VIDEO_PATH)
		
		print(f"\n✅ Test successful!")
		print(f"📊 Results:")
		print(f"  - Fold ID: {result['fold_id']}")
		print(f"  - Number of hits: {result['num_hits']}")
		print(f"  - Prediction frames: {result['prediction_frames']}")
		print(f"  - Ground truth frames: {result['gt_frames']}")
		print(f"  - Output directory: {result['output_dir']}")
		print(f"  - Plot saved to: {result['plot_path']}")
		print(f"  - Video saved to: {result['video_out_path']}")
		
	except Exception as e:
		print(f"❌ Test failed: {str(e)}")
		sys.exit(1)

if __name__ == "__main__":
	main() 
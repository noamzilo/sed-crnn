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
CHECKPOINT_PATH = "../../sed_crnn/train_artifacts/20250702_162812/fold1/epochepoch=021-valerval_er_1s=0.162.ckpt"

# Directory containing video files (for batch processing)
VIDEOS_DIR = "../../data/decorte/rallies"

# Single video path (for single video processing)
SINGLE_VIDEO_PATH = "../../data/decorte/rallies/20230528_VIGO_00.mp4"
SINGLE_VIDEO_PATH = "../../data/decorte/rallies/20231112_MALMO_04.mp4"

# Base output directory
OUTPUT_DIR = "../../output/visualization"

# Validation fold (0-3) - only used for batch processing
VAL_FOLD = 0

# Alpha blending factor for video overlay (0.0-1.0)
ALPHA = 0.5

# Prediction threshold for binary classification
PREDICTION_THRESHOLD = 0.5

# Device to use (empty string for auto-detect, or "cuda", "cpu")
DEVICE = ""

# Processing mode: "single" or "batch"
# MODE = "batch"
MODE = "single"

# =============================================================================
# END CONFIGURATION
# =============================================================================

import os
import glob
import pathlib
from sed_crnn.CRNNInference import CRNNInference
from sed_crnn.CRNNVisualizer import CRNNVisualizer
from sed_crnn.CRNNStatistics import CRNNStatistics
import cv2


def main():
	"""Main function to run video processing."""
	# Validate configuration
	assert os.path.isfile(CHECKPOINT_PATH), CHECKPOINT_PATH
	assert MODE in ["single", "batch"], f"Invalid mode: {MODE} (must be 'single' or 'batch')"
	assert 0 <= VAL_FOLD <= 3, f"Invalid validation fold: {VAL_FOLD} (must be 0-3)"
	assert 0.0 <= ALPHA <= 1.0, f"Invalid alpha value: {ALPHA} (must be 0.0-1.0)"
	assert 0.0 <= PREDICTION_THRESHOLD <= 1.0, f"Invalid prediction threshold: {PREDICTION_THRESHOLD} (must be 0.0-1.0)"

	print("🚀 SED-CRNN Video Visualizer")
	print("=" * 50)
	print(f"📁 Checkpoint: {CHECKPOINT_PATH}")
	print(f"📁 Output: {OUTPUT_DIR}")
	print(f"🎯 Mode: {MODE}")
	print(f"⚙️  Alpha: {ALPHA}")
	print(f"⚙️  Threshold: {PREDICTION_THRESHOLD}")
	print(f"💻 Device: {DEVICE or 'auto-detect'}")
	print("=" * 50)

	# Create inference and visualizer objects
	try:
		inference = CRNNInference(
			ckpt_path=CHECKPOINT_PATH,
			device=DEVICE if DEVICE else "cpu"
		)
		visualizer = CRNNVisualizer(
			alpha=ALPHA,
			prediction_threshold=PREDICTION_THRESHOLD
		)
		statistics = CRNNStatistics()
	except Exception as e:
		print(f"❌ Failed to initialize inference/visualizer: {str(e)}")
		return

	# Populate video_paths based on MODE
	if MODE == "single":
		video_paths = [SINGLE_VIDEO_PATH]
	elif MODE == "batch":
		assert os.path.isdir(VIDEOS_DIR), f"Videos directory does not exist: {VIDEOS_DIR}"
		video_paths = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
		assert video_paths, f"No .mp4 files found in directory: {VIDEOS_DIR}"
	else:
		assert False, f"Invalid MODE: {MODE}"

	# Unified file validity check
	for video_path in video_paths:
		assert os.path.isfile(video_path), f"Video file does not exist: {video_path}"

	for video_path in video_paths:
		try:
			# Inference
			pred_result = inference.process_video(video_path)
			# Output directory per video
			basename = os.path.splitext(os.path.basename(video_path))[0]
			output_dir = os.path.join(OUTPUT_DIR, basename)
			os.makedirs(output_dir, exist_ok=True)
			# Visualization
			frame_df = visualizer.create_frame_level_dataframe(
				pred_result.pred_video, pred_result.gt_video, pred_result.fps, pred_result.nf
			)
			intervals_df, frame_df, pred_intervals, gt_intervals, matched_pred, matched_gt = visualizer.create_intervals_dataframe(
				frame_df, pred_result.fps, tolerance_sec=0.25
			)
			plot_path = os.path.join(output_dir, f"{basename}_predictions.png")
			visualizer.plot_predictions(
				frame_df, intervals_df, pred_result.fps, plot_path,
				y=pred_result.y, mbe=pred_result.mbe, pred_audio=pred_result.pred_audio
			)
			visualizer.dump_intervals_csv(intervals_df, pred_result.fps, output_dir, basename)
			# Video overlay
			cap = cv2.VideoCapture(video_path)
			w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			cap.release()
			video_out_path = os.path.join(output_dir, f"{basename}_overlay.mp4")
			visualizer.create_video_overlay(frame_df, video_path, video_out_path, pred_result.fps, w, h)
			# Collect statistics (event-based)
			statistics.add_inference(pred_result)
			print(f"✅ Successfully processed {os.path.basename(video_path)}")
		except Exception as e:
			print(f"❌ Error processing {os.path.basename(video_path)}: {str(e)}")
			continue

	print(f"\n✅ Successfully processed {len(statistics.dataframe)} videos!")

	# Print output folders for each processed video
	print("\n📁 Output folders for processed videos:")
	for idx, row in enumerate(statistics.dataframe.itertuples(index=False, name=None), 1):
		video_path_str = str(row[0])
		video_base = os.path.splitext(os.path.basename(video_path_str))[0]
		print(f"  Video {idx} outputs: {os.path.abspath(os.path.join(OUTPUT_DIR, video_base))}")

	# Dump statistics to CSV in the output directory
	stats_csv_path = os.path.join(OUTPUT_DIR, "inference_statistics.csv")
	statistics.dump_csv(stats_csv_path)
	print(f"\n📊 Per-video statistics dumped to: {os.path.abspath(stats_csv_path)}")

	# Dump event-based totals to CSV and JSON
	totals_csv_path = os.path.join(OUTPUT_DIR, "inference_totals.csv")
	statistics.dump_total_csv(totals_csv_path)
	print(f"📊 Event-based totals dumped to: {os.path.abspath(totals_csv_path)}")

	totals_json_path = os.path.join(OUTPUT_DIR, "inference_statistics_with_totals.json")
	statistics.dump_json_with_totals(totals_json_path)
	print(f"📊 Per-video + totals JSON dumped to: {os.path.abspath(totals_json_path)}")

	# Print event-based summary to console
	print("\n🔎 Event-based summary:")
	statistics.print_total_stats()

	# Print per-video statistics
	print("\n📋 Per-video statistics:")
	print(statistics.dataframe.to_string(index=False))

	# Print separate stats for train and val
	val_fold = VAL_FOLD
	val_df = statistics.dataframe[statistics.dataframe["fold_id"] == val_fold]
	train_df = statistics.dataframe[statistics.dataframe["fold_id"] != val_fold]

	def print_group_stats(df, group_name):
		if df.empty:
			print(f"No videos in {group_name} set.")
			return
		total_gt = df["num_gt_events"].sum()
		tp = df["tp_events"].sum()
		fn = df["fn_events"].sum()
		fp = df["fp_events"].sum()
		detection_percent = 100.0 * tp / total_gt if total_gt > 0 else 0.0
		miss_percent = 100.0 * fn / total_gt if total_gt > 0 else 0.0
		fp_rate = 100.0 * fp / total_gt if total_gt > 0 else 0.0
		denom = 2 * tp + fp + fn
		event_f1 = 2 * tp / denom if denom > 0 else 0.0
		print(f"\n🔹 {group_name.capitalize()} set stats:")
		print(f"  Videos: {len(df)}")
		print(f"  GT events: {total_gt}")
		print(f"  TP events: {tp} ({detection_percent:.2f}%)")
		print(f"  FN events: {fn} ({miss_percent:.2f}%)")
		print(f"  FP events: {fp} (FP rate: {fp_rate:.2f}% of GT events)")
		print(f"  Event-based F1: {event_f1:.4f}")

	print_group_stats(train_df, "train")
	print_group_stats(val_df, "val")

	# Print where the total stats CSV was dumped
	print(f"\n📊 Event-based totals CSV dumped to: {os.path.abspath(totals_csv_path)}")

	# Print absolute paths in Windows format
	def to_windows_path(path: str) -> str:
		abs_path = os.path.abspath(path)
		# WSL home translation
		wsl_home = '/home/noams'
		windows_home = r'\\wsl.localhost\Ubuntu\home\noams'  # Single backslashes for UNC
		if abs_path.startswith(wsl_home):
			win_path = windows_home + abs_path[len(wsl_home):]
		else:
			win_path = abs_path
		return win_path.replace('/', '\\')

	print("\n🔎 Absolute Paths (Windows format):")
	print(f"  Checkpoint: {to_windows_path(CHECKPOINT_PATH)}")
	print(f"  Output Dir: {to_windows_path(OUTPUT_DIR)}")
	for idx, video_path in enumerate(video_paths, 1):
		print(f"  Video {idx}: {to_windows_path(video_path)}")

if __name__ == "__main__":
	main() 
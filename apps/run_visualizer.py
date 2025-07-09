#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_visualizer.py

Single runner script for SED-CRNN video processing.
Now logs metrics, params, and artifacts to MLflow.
"""

# =============================================================================
# HARD-CODED CONFIGURATION - EDIT THESE VALUES
# =============================================================================

CHECKPOINT_PATH = "../../sed_crnn/train_artifacts/20250702_162812/fold1/epochepoch=021-valerval_er_1s=0.162.ckpt"
VIDEOS_DIR = "../../data/decorte/rallies"
SINGLE_VIDEO_PATH = "../../data/decorte/rallies/20230528_VIGO_00.mp4"
# SINGLE_VIDEO_PATH = "../../data/decorte/rallies/20231112_MALMO_04.mp4"
# SINGLE_VIDEO_PATH = "../../data/decorte/rallies/20230903_FINLAND_08.mp4"
OUTPUT_DIR = "../../output/visualization"
VAL_FOLD = 0
ALPHA = 0.5
PREDICTION_THRESHOLD = 0.5
DEVICE = ""
MODE = "single"  # or "batch"

# =============================================================================
# END CONFIGURATION
# =============================================================================

import os
import glob
import pathlib
import mlflow

from sed_crnn.CRNNInference import CRNNInference
from sed_crnn.CRNNVisualizer import CRNNVisualizer
from sed_crnn.CRNNStatistics import CRNNStatistics
import cv2

def main():
	"""Main function to run video processing with MLflow logging."""

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

	# Set up MLflow run name and experiment
	run_name = f"visualizer_{os.path.basename(CHECKPOINT_PATH).replace('.ckpt','')}_{MODE}"
	mlflow.set_experiment("padel_inference")
	with mlflow.start_run(run_name=run_name):
		# Log key parameters
		mlflow.log_param("checkpoint_path", CHECKPOINT_PATH)
		mlflow.log_param("videos_dir", VIDEOS_DIR)
		mlflow.log_param("single_video_path", SINGLE_VIDEO_PATH)
		mlflow.log_param("output_dir", OUTPUT_DIR)
		mlflow.log_param("val_fold", VAL_FOLD)
		mlflow.log_param("alpha", ALPHA)
		mlflow.log_param("prediction_threshold", PREDICTION_THRESHOLD)
		mlflow.log_param("device", DEVICE or "auto-detect")
		mlflow.log_param("mode", MODE)

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
			mlflow.log_param("init_error", str(e))
			print(f"❌ Failed to initialize inference/visualizer: {str(e)}")
			return

		if MODE == "single":
			video_paths = [SINGLE_VIDEO_PATH]
		elif MODE == "batch":
			assert os.path.isdir(VIDEOS_DIR), f"Videos directory does not exist: {VIDEOS_DIR}"
			video_paths = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
			assert video_paths, f"No .mp4 files found in directory: {VIDEOS_DIR}"
		else:
			assert False, f"Invalid MODE: {MODE}"

		for video_path in video_paths:
			assert os.path.isfile(video_path), f"Video file does not exist: {video_path}"

		output_dirs = []

		# --- Process each video and log artifacts ---
		for video_path in video_paths:
			try:
				pred_result = inference.process_video(video_path)
				basename = os.path.splitext(os.path.basename(video_path))[0]
				output_dir = os.path.join(OUTPUT_DIR, basename)
				os.makedirs(output_dir, exist_ok=True)
				output_dirs.append(output_dir)
				# Visualization/artifacts
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
				intervals_csv_path = os.path.join(output_dir, f"{basename}_intervals.csv")
				visualizer.dump_intervals_csv(intervals_df, pred_result.fps, output_dir, basename)
				cap = cv2.VideoCapture(video_path)
				w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
				cap.release()
				video_out_path = os.path.join(output_dir, f"{basename}_overlay.mp4")
				visualizer.create_video_overlay(frame_df, video_path, video_out_path, pred_result.fps, w, h)
				statistics.add_inference(pred_result)
				# Log artifacts for this video
				for artifact_file in [
					plot_path,
					os.path.join(output_dir, f"{basename}_intervals.csv"),
					os.path.join(output_dir, f"{basename}_overlay.mp4"),
					os.path.join(output_dir, f"{basename}_predictions.csv"),
					os.path.join(output_dir, f"{basename}_ground_truth.csv"),
				]:
					if os.path.exists(artifact_file):
						mlflow.log_artifact(artifact_file, artifact_path=basename)
				print(f"✅ Successfully processed {os.path.basename(video_path)}")
			except Exception as e:
				mlflow.log_param(f"error_{os.path.basename(video_path)}", str(e))
				print(f"❌ Error processing {os.path.basename(video_path)}: {str(e)}")
				continue

		print(f"\n✅ Successfully processed {len(statistics.dataframe)} videos!")

		print("\n📁 Output folders for processed videos:")
		for idx, row in enumerate(statistics.dataframe.itertuples(index=False, name=None), 1):
			video_path_str = str(row[0])
			video_base = os.path.splitext(os.path.basename(video_path_str))[0]
			print(f"  Video {idx} outputs: {os.path.abspath(os.path.join(OUTPUT_DIR, video_base))}")

		# --- Dump and log summary artifacts ---
		stats_csv_path = os.path.join(OUTPUT_DIR, "inference_statistics.csv")
		statistics.dump_csv(stats_csv_path)
		if os.path.exists(stats_csv_path):
			mlflow.log_artifact(stats_csv_path)

		totals_csv_path = os.path.join(OUTPUT_DIR, "inference_totals.csv")
		statistics.dump_total_csv(totals_csv_path)
		if os.path.exists(totals_csv_path):
			mlflow.log_artifact(totals_csv_path)

		totals_json_path = os.path.join(OUTPUT_DIR, "inference_statistics_with_totals.json")
		statistics.dump_json_with_totals(totals_json_path)
		if os.path.exists(totals_json_path):
			mlflow.log_artifact(totals_json_path)

		# --- Print and log event-based summary ---
		print("\n🔎 Event-based summary:")
		statistics.print_total_stats()

		print("\n📋 Per-video statistics:")
		print(statistics.dataframe.to_string(index=False))

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
			# Log group metrics as MLflow metrics
			mlflow.log_metric(f"{group_name}_event_f1", event_f1)
			mlflow.log_metric(f"{group_name}_tp", tp)
			mlflow.log_metric(f"{group_name}_fn", fn)
			mlflow.log_metric(f"{group_name}_fp", fp)
			mlflow.log_metric(f"{group_name}_gt", total_gt)
			mlflow.log_metric(f"{group_name}_detection_percent", detection_percent)
			mlflow.log_metric(f"{group_name}_miss_percent", miss_percent)
			mlflow.log_metric(f"{group_name}_fp_rate", fp_rate)

		print_group_stats(train_df, "train")
		print_group_stats(val_df, "val")

		print(f"\n📊 Event-based totals CSV dumped to: {os.path.abspath(totals_csv_path)}")

		def to_windows_path(path: str) -> str:
			abs_path = os.path.abspath(path)
			wsl_home = '/home/noams'
			windows_home = r'\\wsl.localhost\Ubuntu\home\noams'
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

		print("  Output folders per video (Windows):")
		for idx, out_dir in enumerate(output_dirs, 1):
			print(f"    Video {idx} output: {to_windows_path(out_dir)}")

if __name__ == "__main__":
	main()

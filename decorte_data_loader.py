#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decorte dataset loader / validator
✓	only tabs for indentation
✓	no external tooling
✓	cross-validation-ready (k=4 by default)
"""

import os
import cv2
import pandas as pd
from typing import List, Dict, Tuple

# ────────────────────────────────────────────────────────────────────────────────
#  Paths (edit here if your tree moves)
# ────────────────────────────────────────────────────────────────────────────────
DATA_ROOT			= os.path.expanduser("~/src/plai_cv/data/decorte")
RALLIES_DIR			= os.path.join(DATA_ROOT, "rallies")			# videos
META_DIR			= os.path.join(DATA_ROOT, "metadata")		# .csv / .xlsx

RALLIES_CSV			= os.path.join(META_DIR, "rallies.csv")
HITS_CSV			= os.path.join(META_DIR, "hits.csv")
HIT_ASSIGN_XLSX		= os.path.join(META_DIR, "hit_assignments.xlsx")

VIDEO_EXTENSIONS	= [".mp4", ".MP4", ".avi", ".mkv"]

# ────────────────────────────────────────────────────────────────────────────────
#  Utility: order validation
# ────────────────────────────────────────────────────────────────────────────────
def _assert_monotone(group: pd.DataFrame,
					 starts_col: str,
					 ends_col: str,
					 file_label: str) -> None:
	start_vals = group[starts_col].values
	end_vals	= group[ends_col].values

	bad_start_idx = (start_vals[1:] < start_vals[:-1]).nonzero()[0]
	bad_end_idx   = (end_vals[1:]   < end_vals[:-1]).nonzero()[0]

	if len(bad_start_idx) > 0 or len(bad_end_idx) > 0:
		print(f"[ORDER ERROR] {file_label}: "
			  f"{len(bad_start_idx)} start, {len(bad_end_idx)} end")
		for i in bad_start_idx:
			print(f"\tstart row {i}: {start_vals[i]:.2f} > next {start_vals[i+1]:.2f}")
		for i in bad_end_idx:
			print(f"\tend   row {i}: {end_vals[i]:.2f} > next {end_vals[i+1]:.2f}")
		raise AssertionError(f"Monotonicity violated in {file_label}")

# ────────────────────────────────────────────────────────────────────────────────
#  Load raw metadata tables
# ────────────────────────────────────────────────────────────────────────────────
def _load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if not (os.path.exists(RALLIES_CSV) and os.path.exists(HITS_CSV)
			and os.path.exists(HIT_ASSIGN_XLSX)):
		raise FileNotFoundError("One or more metadata files missing in META_DIR")

	rallies_df			= pd.read_csv(RALLIES_CSV)
	hits_df				= pd.read_csv(HITS_CSV)
	assignments_df		= pd.read_excel(HIT_ASSIGN_XLSX)

	# canonical ordering
	hits_df = hits_df.sort_values(["filename", "start"]).reset_index(drop=True)
	assignments_df = assignments_df.sort_values(["video", "timestamp"]).reset_index(drop=True)

	# validate monotone order per video
	for vid, grp in hits_df.groupby("filename"):
		_assert_monotone(grp, "start", "end", f"HIT:{vid}")
	for vid, grp in assignments_df.groupby("video"):
		_assert_monotone(grp, "timestamp", "timestamp", f"ASSIGN:{vid}")

	return rallies_df, hits_df, assignments_df

# ────────────────────────────────────────────────────────────────────────────────
#  Video metadata (fps / frame count)
# ────────────────────────────────────────────────────────────────────────────────
def _scan_videos(video_dir: str = RALLIES_DIR) -> pd.DataFrame:
	records: List[Dict] = []

	for fname in os.listdir(video_dir):
		if not any(fname.endswith(ext) for ext in VIDEO_EXTENSIONS):
			continue

		path = os.path.join(video_dir, fname)
		cap = cv2.VideoCapture(path)
		if not cap.isOpened():
			print(f"[WARN] Cannot open {fname}, skipping")
			continue

		meta = dict(
			video_name			= os.path.splitext(fname)[0],
			video_ext			= os.path.splitext(fname)[1],
			video_name_ext		= fname,
			video_path			= path,
			fps					= cap.get(cv2.CAP_PROP_FPS),
			total_frames		= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
			width				= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
			height				= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		)
		cap.release()
		records.append(meta)

	return pd.DataFrame(records)

# ────────────────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────────────────
def load_decorte_dataset(k_folds: int = 4,
						 seed: int = 42) -> Dict[str, Dict]:
	"""
	Returns a dict keyed by video_name_ext with:
		• video_meta   : row from videos_df (Series)
		• hits         : DataFrame rows for that clip
		• assignments  : DataFrame rows (can be empty)
		• fold_id      : int in [0 .. k_folds-1]
	"""

	rallies_df, hits_df, assignments_df = _load_tables()
	videos_df = _scan_videos()

	# ------------------------------------------------------------------
	# Merge & sanity-check presence of every video in metadata
	# ------------------------------------------------------------------
	missing_meta = set(videos_df["video_name_ext"]) - set(hits_df["filename"])
	if len(missing_meta) > 0:
		print(f"[WARN] {len(missing_meta)} videos lack HIT rows")

	dataset: Dict[str, Dict] = {}
	for _, vid_row in videos_df.iterrows():
		vname_ext = vid_row["video_name_ext"]
		vname_no  = vid_row["video_name"]

		dataset[vname_ext] = dict(
			video_meta		= vid_row,
			hits			= hits_df[hits_df["filename"] == vname_ext].copy(),
			assignments		= assignments_df[assignments_df["video"] == vname_no].copy(),
			fold_id			= -1		# placeholder, filled next
		)

	# ------------------------------------------------------------------
	# Simple fold assignment: round-robin by sorted video name
	# (keeps deterministic splits, can swap for stratified later)
	# ------------------------------------------------------------------
	sorted_keys = sorted(dataset.keys())
	for idx, key in enumerate(sorted_keys):
		dataset[key]["fold_id"] = idx % k_folds

	# ------------------------------------------------------------------
	# Overview printout (validation)
	# ------------------------------------------------------------------
	n_hits_total = sum(len(v["hits"]) for v in dataset.values())
	n_assigns_total = sum(len(v["assignments"]) for v in dataset.values())
	print(f"[Decorte] videos={len(dataset)}  hits={n_hits_total}  assignments={n_assigns_total}")
	fold_sizes = [sum(1 for v in dataset.values() if v["fold_id"] == f) for f in range(k_folds)]
	print(f"[Decorte] fold distribution: {fold_sizes}")

	return dataset

# ────────────────────────────────────────────────────────────────────────────────
#  Quick CLI validation run
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
	ds = load_decorte_dataset(k_folds=4)
	print("Dataset keys:", list(ds.keys())[:5], "...")

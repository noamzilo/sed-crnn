#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP4 ➜ log-Mel ➜ per-fold .npz packs

✓	extracts audio from .mp4 using ffmpeg (no temp .wav)
✓	log-Mel: 40 bins, 2048 FFT, 50% hop
✓	per-video .npz (mbe, label)
✓	per-fold .npz (X_train, Y_train, X_test, Y_test)
✓	logs timing to console and feature_log.jsonl
✓	tabs only
"""

import os
import subprocess
import time
import json
import numpy as np
import pandas as pd
import librosa
from sklearn import preprocessing

from decorte_data_loader import load_decorte_dataset
import utils

# ────────────────────────────────────────────────────────────────
#  Parameters
# ────────────────────────────────────────────────────────────────
SR				= 44_100
NFFT			= 2048
HOP				= NFFT // 2			# 1024 samples → ~23.2ms hop
NB_MEL			= 40
CACHE_DIR		= os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
os.makedirs(CACHE_DIR, exist_ok=True)
LOG_PATH		= os.path.join(CACHE_DIR, "feature_log.jsonl")

# ────────────────────────────────────────────────────────────────
#  Audio extractor: ffmpeg piped to numpy
# ────────────────────────────────────────────────────────────────
def _ffmpeg_audio(path: str, sr: int = SR) -> np.ndarray:
	cmd = [
		"ffmpeg", "-v", "error",
		"-i", path,
		"-f", "f32le",
		"-ac", "1",
		"-ar", str(sr),
		"pipe:1"
	]
	raw = subprocess.check_output(cmd)
	return np.frombuffer(raw, dtype=np.float32)

# ────────────────────────────────────────────────────────────────
#  Mel feature extractor
# ────────────────────────────────────────────────────────────────
def _mbe(y: np.ndarray, sr: int) -> np.ndarray:
	spec, _ = librosa.core.spectrum._spectrogram(y=y, n_fft=NFFT,
		hop_length=HOP, power=1)
	mel_b = librosa.filters.mel(sr=sr, n_fft=NFFT, n_mels=NB_MEL)
	return np.log(np.dot(mel_b, spec)).T		# shape: (frames, 40)

# ────────────────────────────────────────────────────────────────
#  Load + Process
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
	print("▶ feature.py starting")

	ds = load_decorte_dataset(k_folds=4)
	per_video = {}

	for vname_ext, info in ds.items():
		base = os.path.splitext(vname_ext)[0]
		out_npz = os.path.join(CACHE_DIR, f"{base}_mon.npz")

		if os.path.exists(out_npz):
			data = np.load(out_npz)
			mbe, lbl = data['arr_0'], data['arr_1']
			print(f"[cached] {vname_ext} → {mbe.shape[0]} frames")
		else:
			print(f"[audio] extracting {vname_ext} … ", end="", flush=True)
			t0 = time.time()

			try:
				y = _ffmpeg_audio(info["video_meta"]["video_path"])
				mbe = _mbe(y, SR)
			except Exception as e:
				print(f"✖ error: {e}")
				continue

			lbl = np.zeros((mbe.shape[0], 1), dtype=np.float32)
			for _, hit in info["hits"].iterrows():
				s = int(np.floor(hit["start"] * SR / HOP))
				e = int(np.ceil(hit["end"] * SR / HOP))
				lbl[s:e, 0] = 1.0

			np.savez(out_npz, mbe, lbl)
			dt = time.time() - t0
			print(f"✔ {mbe.shape[0]} frames in {dt:.2f}s")

			with open(LOG_PATH, "a") as f:
				f.write(json.dumps({
					"video": vname_ext,
					"frames": int(mbe.shape[0]),
					"duration_sec": round(dt, 2),
					"saved": out_npz
				}) + "\n")

		per_video[vname_ext] = (mbe, lbl, info["fold_id"])

	# ─────────────────────────────────────────────────────────────
	#  Build per-fold npz packs
	# ─────────────────────────────────────────────────────────────
	print("▶ building per-fold datasets")

	fold_k = max(v[2] for v in per_video.values()) + 1
	for f in range(fold_k):
		X_train, Y_train, X_test, Y_test = None, None, None, None

		for vname, (mbe, lbl, fold) in per_video.items():
			if fold == f:
				X_test  = mbe if X_test is None else np.concatenate((X_test, mbe), axis=0)
				Y_test  = lbl if Y_test is None else np.concatenate((Y_test, lbl), axis=0)
			else:
				X_train = mbe if X_train is None else np.concatenate((X_train, mbe), axis=0)
				Y_train = lbl if Y_train is None else np.concatenate((Y_train, lbl), axis=0)

		# normalize
		scaler = preprocessing.StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

		out_fold = os.path.join(CACHE_DIR, f"mbe_mon_fold{f+1}.npz")
		np.savez(out_fold, X_train, Y_train, X_test, Y_test)
		print(f"[fold {f+1}] saved to {out_fold} | train={len(X_train)} test={len(X_test)}")

	print("✔ feature.py complete.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP4 ➜ log-Mel ➜ per-fold .npz packs

✓	extracts audio once via ffmpeg (no temp .wav)
✓	creates per-video .npz		 <cache>/<video>_mon.npz
✓	creates per-fold  .npz		 <cache>/mbe_mon_fold<k>.npz
✓	tabs only
"""

import os, subprocess, datetime, json
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
from sklearn import preprocessing

from decorte_data_loader import load_decorte_dataset
import utils		# existing utils.py

# ────────────────────────────────────────────────────────────────
#  Parameters
# ────────────────────────────────────────────────────────────────
SR				= 44_100
NFFT			= 2048
HOP				= NFFT // 2				# 50 % overlap  → 23.22 ms hop
NB_MEL			= 40
CACHE_DIR		= os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
os.makedirs(CACHE_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
#  Helpers
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

def _mbe(y: np.ndarray, sr: int) -> np.ndarray:
	spec, _ = librosa.core.spectrum._spectrogram(y=y, n_fft=NFFT,
		hop_length=HOP, power=1)
	mel_b = librosa.filters.mel(sr=sr, n_fft=NFFT, n_mels=NB_MEL)
	return np.log(np.dot(mel_b, spec)).T			# frames × 40

# ────────────────────────────────────────────────────────────────
#  Pass 1 – per-video extraction
# ────────────────────────────────────────────────────────────────
print("▶ pass 1 – audio extraction")
ds = load_decorte_dataset(k_folds=4)
per_video = {}					# name_ext → (mbe, label, fold)

for vname_ext, info in ds.items():
	out_npz = os.path.join(CACHE_DIR, f"{os.path.splitext(vname_ext)[0]}_mon.npz")
	if os.path.exists(out_npz):
		dmp = np.load(out_npz)
		mbe, lbl = dmp['arr_0'], dmp['arr_1']
	else:
		print(f"[audio] {vname_ext}")
		wav = _ffmpeg_audio(info["video_meta"]["video_path"])
		mbe = _mbe(wav, SR)									# (frames, 40)

		lbl = np.zeros((mbe.shape[0], 1), dtype=np.float32)
		for _, hit in info["hits"].iterrows():
			s = int(np.floor(hit["start"] * SR / HOP))
			e = int(np.ceil(hit["end"]   * SR / HOP))
			lbl[s:e, 0] = 1.0

		np.savez(out_npz, mbe, lbl)

	per_video[vname_ext] = (mbe, lbl, info["fold_id"])

print("✔ audio→npz done")

# ────────────────────────────────────────────────────────────────
#  Pass 2 – build per-fold packs with training normalisation
# ────────────────────────────────────────────────────────────────
print("▶ pass 2 – build fold packs")
fold_k = max(v[2] for v in per_video.values()) + 1
for f in range(fold_k):
	X_train, Y_train, X_test, Y_test = None, None, None, None

	for vname, (mbe, lbl, fold) in per_video.items():
		if fold == f:
			X_test  = mbe  if X_test  is None else np.concatenate((X_test,  mbe), 0)
			Y_test  = lbl  if Y_test  is None else np.concatenate((Y_test,  lbl), 0)
		else:
			X_train = mbe  if X_train is None else np.concatenate((X_train, mbe), 0)
			Y_train = lbl  if Y_train is None else np.concatenate((Y_train, lbl), 0)

	# scale
	scaler = preprocessing.StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test	= scaler.transform(X_test)

	out_fold = os.path.join(CACHE_DIR, f"mbe_mon_fold{f+1}.npz")
	np.savez(out_fold, X_train, Y_train, X_test, Y_test)
	print(f"  • fold {f+1}: {out_fold}")

print("✔ feature packs ready")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_features.py

Single source of truth for:
	• raw audio decode  (_ffmpeg_audio)
	• log-Mel extraction (_mbe)
	• StandardScaler fit / save / load
	• normalize() helper

Tabs ONLY.
"""

import os, subprocess, joblib, numpy as np, librosa
from sklearn.preprocessing import StandardScaler
from .train_constants import SAMPLE_RATE, HOP_LENGTH, N_MELS

# ───────────────────────────────────────────────
#  ffmpeg decode → mono float32
# ───────────────────────────────────────────────
def _ffmpeg_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
	cmd = ["ffmpeg","-v","error","-i",path,"-f","f32le","-ac","1","-ar",str(sr),"pipe:1"]
	return np.frombuffer(subprocess.check_output(cmd), dtype=np.float32)

# ───────────────────────────────────────────────
#  waveform → (n_frames, n_mels) log-Mel
# ───────────────────────────────────────────────
def _mbe(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray: # mel band energy - a standard feature engineering preprocessing step.
	n_fft = HOP_LENGTH * 2
	s = librosa.stft(y, n_fft=n_fft, hop_length=HOP_LENGTH)
	pwr = np.abs(s) ** 2
	mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=N_MELS)
	return np.log(np.dot(mel, pwr) + 1e-10).T				# (frames, 40)

# ───────────────────────────────────────────────
#  Scaler helpers
# ───────────────────────────────────────────────
def save_scaler(scaler: StandardScaler, path: str):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	joblib.dump(scaler, path)

def load_scaler(path: str) -> StandardScaler:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Scaler not found: {path}")
	return joblib.load(path)

def normalize(mbe: np.ndarray, scaler: StandardScaler) -> np.ndarray:
	return scaler.transform(mbe)

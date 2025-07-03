#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRNNInference.py

Handles SED-CRNN model inference and prediction generation.
Responsibilities:
- Model loading and inference
- Data preparation (sliding windows, normalization)
- Ground truth and prediction creation in video/audio space
- Core inference logic for single video
"""

import os
import torch
import numpy as np
from typing import Any, Dict
from sed_crnn.decorte_data_loader import load_decorte_dataset
from sed_crnn.audio_features import _ffmpeg_audio, _mbe
import sed_crnn.audio_features as af
from sed_crnn.crnn_lightning import CRNNLightning
from sed_crnn.train_constants import *
from scipy.interpolate import interp1d
from sed_crnn.InferenceResult import InferenceResult

class CRNNInference:
	"""
	Handles SED-CRNN model inference and prediction generation.
	"""
	def __init__(self, ckpt_path: str, device: str = ""):
		"""
		Initialize the CRNN inference engine.
		Args:
			ckpt_path: Path to the model checkpoint
			device: Device to use for inference (empty string for auto-detect)
		"""
		self.ckpt_path = ckpt_path
		self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
		self.meta_all = load_decorte_dataset()

	def sliding_windows(self, mbe: np.ndarray, win: int = SEQ_LEN_IN, stride: int = INFER_STRIDE):
		"""Create sliding windows for inference."""
		wins, starts = [], []
		for s in range(0, mbe.shape[0] - win + 1, stride):
			wins.append(mbe[s:s + win].T)
			starts.append(s)
		return np.array(wins), np.array(starts)

	def create_ground_truth_in_video_space(self, hits, fps, nf):
		"""Create ground truth labels in video frame space."""
		video_times = np.arange(nf) / fps
		gt_video = np.zeros(nf, dtype=np.float32)
		for _, h in hits.iterrows():
			start_time = h["start"]
			end_time = h["end"]
			mask = (video_times >= start_time) & (video_times <= end_time)
			gt_video[mask] = 1.0
		return gt_video

	def create_predictions_in_video_space(self, pred_full, fps, nf):
		"""Convert audio frame predictions to video frame predictions."""
		audio_frames = len(pred_full)
		audio_times = np.arange(audio_frames) / FPS_ORIG
		video_times = np.arange(nf) / fps
		pred_interpolator = interp1d(audio_times, pred_full, kind='linear', bounds_error=False, fill_value=0)
		pred_video = pred_interpolator(video_times)
		return pred_video

	def process_video(self, video_path: str) -> InferenceResult:
		"""
		Process a single video and generate predictions.
		Args:
			video_path: Path to the input video
		Returns:
			InferenceResult instance with predictions, ground truth, and metadata
		"""
		import cv2
		os.makedirs("/tmp", exist_ok=True)
		vname = os.path.basename(video_path)
		if vname not in self.meta_all:
			raise RuntimeError(f"{vname} not in Decorte metadata")
		meta = self.meta_all[vname]
		hits = meta["hits"]
		fold = meta["fold_id"]
		fold_cache = fold + 1
		scaler_path = os.path.join(os.path.dirname(self.ckpt_path), f"scaler_fold{fold_cache}.joblib")
		if not os.path.exists(scaler_path):
			scaler_path = os.path.join(CACHE_DIR, f"scaler_fold{fold_cache}.joblib")
		scaler = af.load_scaler(scaler_path)
		y = _ffmpeg_audio(video_path, SAMPLE_RATE)
		mbe = _mbe(y, SAMPLE_RATE)
		mbe = af.normalize(mbe, scaler)
		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		cap.release()
		gt_video = self.create_ground_truth_in_video_space(hits, fps, nf)
		win_x, win_starts = self.sliding_windows(mbe, win=SEQ_LEN_IN, stride=INFER_STRIDE)
		tensor_x = torch.from_numpy(win_x).unsqueeze(1).float()
		loader = torch.utils.data.DataLoader(
			torch.utils.data.TensorDataset(tensor_x),
			batch_size=64, shuffle=False, pin_memory=True
		)
		model = CRNNLightning.load_from_checkpoint(self.ckpt_path, fold_id=fold, art_dir="/tmp").to(self.device)
		from pytorch_lightning import Trainer
		trainer = Trainer(accelerator=self.device, devices=1, logger=False, enable_checkpointing=False)
		logits_batches = trainer.predict(model, loader)
		logits_per_frame = np.zeros(mbe.shape[0], dtype=np.float32)
		counts_per_frame = np.zeros(mbe.shape[0], dtype=np.float32)
		if logits_batches is not None:
			logits = torch.cat([torch.tensor(batch) for batch in logits_batches], 0).squeeze(-1).cpu().numpy().reshape(-1, SEQ_LEN_OUT)
			for i, start in enumerate(win_starts):
				end = start + SEQ_LEN_OUT
				logit_slice = logits[i]
				logits_per_frame[start:end] += logit_slice[:min(SEQ_LEN_OUT, mbe.shape[0] - start)]
				counts_per_frame[start:end] += 1
		else:
			raise RuntimeError("No predictions returned from model")
		mask = counts_per_frame > 0
		avg_logits = np.zeros_like(logits_per_frame)
		avg_logits[mask] = logits_per_frame[mask] / counts_per_frame[mask]
		pred_audio = 1 / (1 + np.exp(-avg_logits))
		pred_video = self.create_predictions_in_video_space(pred_audio, fps, nf)
		return InferenceResult(
			video_path=video_path,
			fold_id=fold,
			hits=hits,
			gt_video=gt_video,
			pred_audio=pred_audio,
			pred_video=pred_video,
			fps=fps,
			nf=nf,
			y=y,
			mbe=mbe
		) 
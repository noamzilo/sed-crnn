#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning DataModule for Decorte – now imports *all* preprocessing,
including StandardScaler, from audio_features.py so the exact same
code is shared with inference.

Tabs only.
"""

import os, random, time, json, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import joblib

from train_constants import *
from decorte_data_loader import load_decorte_dataset
import sed_crnn.audio_features as af								# ← shared helpers

# ───────────────────────────────────────────────
#  SpecAugment (unchanged)
# ───────────────────────────────────────────────
def _spec_augment(mel: np.ndarray):
	for _ in range(MASKS_PER_EX):
		if mel.shape[1] > TIME_MASK_W:
			t0 = np.random.randint(0, mel.shape[1]-TIME_MASK_W)
			mel[:,t0:t0+TIME_MASK_W]=0
		if mel.shape[0] > FREQ_MASK_W:
			f0 = np.random.randint(0, mel.shape[0]-FREQ_MASK_W)
			mel[f0:f0+FREQ_MASK_W,:]=0
	return mel

# ───────────────────────────────────────────────
#  Dataset
# ───────────────────────────────────────────────
class HitWindowDataset(Dataset):
	def __init__(self, mel: np.ndarray, lab: np.ndarray, augment: bool=False):
		self.mel, self.lab, self.augment = mel, lab, augment
		# Identify all positive and padded (0.5) frames
		pos_mask = (lab[:,0] >= 0.5)
		self.pos = np.where(lab[:,0]==1)[0].tolist()
		window = np.ones(SEQ_LEN_IN,dtype=np.uint8)
		# Negative indices: windows that do not touch any positive or padded frame
		neg_mask = np.convolve(pos_mask.astype(np.uint8), window, 'valid') == 0
		self.neg = np.where(neg_mask)[0].tolist()
		self.total = mel.shape[0]
	def __len__(self): return len(self.pos)*2
	def _rnd_pos(self):
		c = random.choice(self.pos)
		a,b = max(0,c-SEQ_LEN_IN+1), min(c,self.total-SEQ_LEN_IN)
		st = random.randint(a,b)
		# Time jitter
		if self.augment:
			jitter = random.randint(-JITTER_RANGE_FRAMES, JITTER_RANGE_FRAMES)
			st = min(max(0, st + jitter), self.total-SEQ_LEN_IN)
		return st
	def _rnd_neg(self):
		st = random.choice(self.neg)
		# Time jitter
		if self.augment:
			jitter = random.randint(-JITTER_RANGE_FRAMES, JITTER_RANGE_FRAMES)
			st = min(max(0, st + jitter), self.total-SEQ_LEN_IN)
		return st
	def __getitem__(self, idx):
		st = self._rnd_pos() if idx%2==0 else self._rnd_neg()
		if st+SEQ_LEN_IN>self.total: st=self.total-SEQ_LEN_IN
		x = self.mel[st:st+SEQ_LEN_IN].T
		if self.augment: x=_spec_augment(x.copy())
		if idx==0: np.save("/tmp/train_window.npy", x)
		y = self.lab[st:st+SEQ_LEN_IN].reshape(SEQ_LEN_OUT,-1).max(1,keepdims=True)
		return torch.from_numpy(x).unsqueeze(0).float(), torch.from_numpy(y).float()

# ───────────────────────────────────────────────
class DecorteDataModule(pl.LightningDataModule):
	def __init__(self, fold_id:int,
				 cache_dir:str=CACHE_DIR,
				 batch_size:int=BATCH_SIZE,
				 num_workers:int=NUM_WORKERS):
		super().__init__()
		self.fold_id, self.cache_dir = fold_id, cache_dir
		self.batch_size, self.num_workers = batch_size, num_workers
		os.makedirs(self.cache_dir, exist_ok=True)

	# paths
	def _per_video_npz(self, name): return os.path.join(self.cache_dir,f"{name}_mon.npz")
	def _per_fold_npz(self,f): return os.path.join(self.cache_dir,f"mbe_mon_fold{f}.npz")
	def _per_fold_scaler(self,f): return os.path.join(self.cache_dir,f"scaler_fold{f}.joblib")

	# ─────────────────────────────────────
	def _extract_and_cache_video(self, vname_ext, info):
		base = os.path.splitext(vname_ext)[0]
		target = self._per_video_npz(base)
		if os.path.exists(target): return
		y   = af._ffmpeg_audio(info["video_meta"]["video_path"])
		mbe = af._mbe(y)
		lbl = np.zeros((mbe.shape[0],1), np.float32)
		for _,h in info["hits"].iterrows():
			s = int(np.floor(h["start"]*SAMPLE_RATE/HOP_LENGTH))
			e = int(np.ceil (h["end"]  *SAMPLE_RATE/HOP_LENGTH))
			# Main hit region
			lbl[s:e,0]=1.
			# Label padding (augmentation): random extension by ±LABEL_PAD_RANGE_MS
			pad_frames = int((LABEL_PAD_RANGE_MS/1000) * SAMPLE_RATE / HOP_LENGTH)
			pad_left = random.randint(0, pad_frames)
			pad_right = random.randint(0, pad_frames)
			ps = max(0, s - pad_left)
			pe = min(lbl.shape[0], e + pad_right)
			# Only set to LABEL_PAD_VALUE if not already 1
			for i in range(ps, s):
				if lbl[i,0] < 1:
					lbl[i,0] = LABEL_PAD_VALUE
			for i in range(e, pe):
				if lbl[i,0] < 1:
					lbl[i,0] = LABEL_PAD_VALUE
		if mbe is None or lbl is None:
			raise ValueError("mbe and lbl must not be None before saving")
		mbe_arr = np.asarray(mbe)
		lbl_arr = np.asarray(lbl)
		np.savez(target, mbe=mbe_arr, lbl=lbl_arr)

	# ─────────────────────────────────────
	def _ensure_fold_npz(self):
		ds = load_decorte_dataset(k_folds=4)
		for v, info in ds.items():
			self._extract_and_cache_video(v, info)  # ensure per-video

		for f in range(1, 5):
			npz_path = self._per_fold_npz(f)
			scaler_path = self._per_fold_scaler(f)
			npz_exists = os.path.exists(npz_path)
			scaler_exists = os.path.exists(scaler_path)

			if npz_exists and scaler_exists:
				continue

			X_tr, Y_tr, X_te, Y_te = None, None, None, None
			for v, info in ds.items():
				data = np.load(self._per_video_npz(os.path.splitext(v)[0]))
				mbe = data['mbe']
				lbl = data['lbl']
				assert mbe is not None and lbl is not None, "mbe and lbl must not be None before stacking"
				if info["fold_id"] == f - 1:
					X_te = mbe if X_te is None else np.vstack((X_te, mbe))
					Y_te = lbl if Y_te is None else np.vstack((Y_te, lbl))
				else:
					X_tr = mbe if X_tr is None else np.vstack((X_tr, mbe))
					Y_tr = lbl if Y_tr is None else np.vstack((Y_tr, lbl))

			if not scaler_exists:
				scaler = StandardScaler().fit(X_tr)
				af.save_scaler(scaler, scaler_path)
			else:
				scaler = af.load_scaler(scaler_path)

			if not npz_exists:
				X_tr = scaler.transform(X_tr)
				X_te = scaler.transform(X_te)
				np.savez(npz_path, X_tr, Y_tr, X_te, Y_te)

	# ─────────────────────────────────────
	def setup(self, stage=None):
		self._ensure_fold_npz()
		pack = np.load(self._per_fold_npz(self.fold_id))
		aug = (stage not in ["validate","test"])
		self.train_ds = HitWindowDataset(pack['arr_0'], pack['arr_1'], augment=aug)
		self.val_ds   = HitWindowDataset(pack['arr_2'], pack['arr_3'], augment=False)

	def train_dataloader(self):
		return DataLoader(self.train_ds, self.batch_size, True, drop_last=True,
						  num_workers=self.num_workers, pin_memory=True)
	def val_dataloader(self):
		return DataLoader(self.val_ds, self.batch_size, False,
						  num_workers=self.num_workers, pin_memory=True)

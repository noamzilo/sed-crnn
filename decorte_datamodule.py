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
import audio_features as af								# ← shared helpers

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
		self.pos = np.where(lab[:,0]==1)[0].tolist()
		window = np.ones(SEQ_LEN_IN,dtype=np.uint8)
		self.neg = np.where(np.convolve((lab[:,0]==1).astype(np.uint8),
										window,'valid')==0)[0].tolist()
		self.total = mel.shape[0]
	def __len__(self): return len(self.pos)*2
	def _rnd_pos(self):
		c = random.choice(self.pos)
		a,b = max(0,c-SEQ_LEN_IN+1), min(c,self.total-SEQ_LEN_IN)
		return random.randint(a,b)
	def _rnd_neg(self): return random.choice(self.neg)
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
			lbl[s:e,0]=1.
		np.savez(target, mbe, lbl)

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
				mbe, lbl = np.load(self._per_video_npz(os.path.splitext(v)[0])).values()
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

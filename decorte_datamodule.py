#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning DataModule + Dataset for the Decorte hit-detection task
✓	added simple SpecAugment-style time+frequency masking (train-only)
✓	all previous behaviour unchanged
✓	tabs only
"""

import os, random, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# ────────────────────────────────────────────────────────────────
#  Constants (identical defaults + aug params)
# ────────────────────────────────────────────────────────────────
SEQ_LEN_IN		= 64
TIME_POOL = [5, 2, 2]
SEQ_LEN_OUT = int(256 // np.prod(TIME_POOL))
CACHE_DIR		= os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
BATCH_SIZE		= 128
NUM_WORKERS		= 4

# Augmentation hyper-params
TIME_MASK_W		= 8			# frames
FREQ_MASK_W		= 8			# mel bins
MASKS_PER_EX	= 2

# ────────────────────────────────────────────────────────────────
#  Utility
# ────────────────────────────────────────────────────────────────
def _find_clean_negatives(label_vec: np.ndarray):
	mask = (label_vec[:, 0] == 1).astype(np.uint8)
	window = np.ones(SEQ_LEN_IN, dtype=np.uint8)
	overlap = np.convolve(mask, window, mode='valid')
	return np.where(overlap == 0)[0]

def _load_all_npz(folder: str):
	folds = {}
	for i in range(1, 5):
		fp = os.path.join(folder, f"mbe_mon_fold{i}.npz")
		arr = np.load(fp)
		folds[i] = {
			"train_x": arr['arr_0'], "train_y": arr['arr_1'],
			"val_x":   arr['arr_2'], "val_y":   arr['arr_3'],
		}
		print(f"🔄 loaded into RAM → fold {i}  ({arr['arr_0'].nbytes / 1e6:0.1f} MB train)")
	return folds

# ────────────────────────────────────────────────────────────────
#  Simple SpecAugment
# ────────────────────────────────────────────────────────────────
def _spec_augment(mel: np.ndarray):
	for _ in range(MASKS_PER_EX):
		# time mask
		if mel.shape[1] > TIME_MASK_W:
			t0 = np.random.randint(0, mel.shape[1] - TIME_MASK_W)
			mel[:, t0:t0 + TIME_MASK_W] = 0.0
		# freq mask
		if mel.shape[0] > FREQ_MASK_W:
			f0 = np.random.randint(0, mel.shape[0] - FREQ_MASK_W)
			mel[f0:f0 + FREQ_MASK_W, :] = 0.0
	return mel

# ────────────────────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────────────────────
class HitWindowDataset(Dataset):
	def __init__(self, mel: np.ndarray, lab: np.ndarray, augment: bool = False):
		self.mel, self.lab = mel, lab
		self.augment = augment
		self.pos_frames = np.where(lab[:, 0] == 1)[0].tolist()
		self.neg_starts = _find_clean_negatives(lab).tolist()
		self.total_frames = mel.shape[0]

	def __len__(self):
		return len(self.pos_frames) * 2

	def _rand_pos(self):
		center = random.choice(self.pos_frames)
		a = max(0, center - SEQ_LEN_IN + 1)
		b = min(center, self.total_frames - SEQ_LEN_IN)
		return random.randint(a, b)

	def _rand_neg(self):
		return random.choice(self.neg_starts)

	def _pool_labels(self, lab_win):
		return lab_win.reshape(SEQ_LEN_OUT, -1).max(axis=1, keepdims=True)

	def __getitem__(self, idx):
		start = self._rand_pos() if idx % 2 == 0 else self._rand_neg()
		x = self.mel[start:start + SEQ_LEN_IN].T				# (40,64)
		if self.augment:										# ← new
			x = _spec_augment(x.copy())
		y = self._pool_labels(self.lab[start:start + SEQ_LEN_IN])
		return torch.from_numpy(x).unsqueeze(0).float(), torch.from_numpy(y).float()

# ────────────────────────────────────────────────────────────────
#  DataModule
# ────────────────────────────────────────────────────────────────
class DecorteDataModule(pl.LightningDataModule):
	def __init__(self, fold_id: int, cache_dir: str = CACHE_DIR,
				 batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
		super().__init__()
		self.fold_id, self.cache_dir = fold_id, cache_dir
		self.batch_size, self.num_workers = batch_size, num_workers

	def setup(self, stage=None):
		all_folds = _load_all_npz(self.cache_dir)
		fdata = all_folds[self.fold_id]
		self.train_ds = HitWindowDataset(fdata["train_x"], fdata["train_y"], augment=True)	# ← augment
		self.val_ds   = HitWindowDataset(fdata["val_x"],   fdata["val_y"],   augment=False)

	def train_dataloader(self):
		return DataLoader(self.train_ds, self.batch_size, shuffle=True,
						  drop_last=True, num_workers=self.num_workers,
						  pin_memory=True)

	def val_dataloader(self):
		return DataLoader(self.val_ds, self.batch_size, shuffle=False,
						  num_workers=self.num_workers, pin_memory=True)

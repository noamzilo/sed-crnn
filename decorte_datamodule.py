#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning DataModule + Dataset for the Decorte hit-detection task

Key upgrades
────────────
✓	Integrated *feature.py* logic — preprocessing now happens on-demand
✓	Will auto-create per-video and per-fold .npz if missing
✓	Provides `extract_video_to_npz()` for stand-alone use
✓	All training/inference codepaths still call HitWindowDataset exactly as before
✓	tabs only
"""

import os, random, subprocess, time, json, numpy as np, torch, librosa
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn import preprocessing

from train_constants import *
from decorte_data_loader import load_decorte_dataset		# ← unchanged

# ────────────────────────────────────────────────────────────────
#  Low-level audio & feature helpers
# ────────────────────────────────────────────────────────────────
def _ffmpeg_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
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
	n_fft = HOP_LENGTH * 2					# 2048
	s = librosa.stft(y, n_fft=n_fft, hop_length=HOP_LENGTH)
	power_spec = np.abs(s) ** 2
	mel_b = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=N_MELS)
	return np.log(np.dot(mel_b, power_spec)).T		# (frames, 40)

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
		if not os.path.exists(fp):
			raise FileNotFoundError(f"✖ expected {fp} – run code again to auto-create")
		arr = np.load(fp)
		folds[i] = {
			"train_x": arr['arr_0'], "train_y": arr['arr_1'],
			"val_x":   arr['arr_2'], "val_y":   arr['arr_3'],
		}
		print(f"🔄 loaded into RAM → fold {i} ({arr['arr_0'].nbytes / 1e6:0.1f} MB train)")
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

	def __len__(self): return len(self.pos_frames) * 2

	def _rand_pos(self):
		center = random.choice(self.pos_frames)
		a = max(0, center - SEQ_LEN_IN + 1)
		b = min(center, self.total_frames - SEQ_LEN_IN)
		return random.randint(a, b)

	def _rand_neg(self): return random.choice(self.neg_starts)

	def __getitem__(self, idx):

		start = self._rand_pos() if idx % 2 == 0 else self._rand_neg()
		if start + SEQ_LEN_IN > self.total_frames:
			start = max(0, self.total_frames - SEQ_LEN_IN)

		x = self.mel[start:start + SEQ_LEN_IN].T					# (40, SEQ_LEN_IN)
		if self.augment: x = _spec_augment(x.copy())
		if idx == 0:
			np.save("/tmp/train_window.npy", x.numpy())

		lab_win = self.lab[start:start + SEQ_LEN_IN]
		y = lab_win.reshape(SEQ_LEN_OUT, -1).max(axis=1, keepdims=True)

		return torch.from_numpy(x).unsqueeze(0).float(), torch.from_numpy(y).float()

# ────────────────────────────────────────────────────────────────
#  DataModule  (owns preprocessing now)
# ────────────────────────────────────────────────────────────────
class DecorteDataModule(pl.LightningDataModule):
	def __init__(self, fold_id: int,
				 cache_dir: str = CACHE_DIR,
				 batch_size: int = BATCH_SIZE,
				 num_workers: int = NUM_WORKERS):
		super().__init__()
		self.fold_id, self.cache_dir = fold_id, cache_dir
		self.batch_size, self.num_workers = batch_size, num_workers
		os.makedirs(self.cache_dir, exist_ok=True)

	# ───────── Internal prep helpers ─────────
	def _per_video_npz_path(self, vname_no_ext): return os.path.join(self.cache_dir, f"{vname_no_ext}_mon.npz")
	def _per_fold_npz_path(self, fold): return os.path.join(self.cache_dir, f"mbe_mon_fold{fold}.npz")
	def _log_path(self): return os.path.join(self.cache_dir, "feature_log.jsonl")

	def _extract_and_cache_video(self, vname_ext, info):
		base = os.path.splitext(vname_ext)[0]
		out_npz = self._per_video_npz_path(base)

		if os.path.exists(out_npz): return np.load(out_npz)['arr_0'].shape[0]

		print(f"[audio] extracting {vname_ext} … ", end="", flush=True)
		t0 = time.time()
		y = _ffmpeg_audio(info["video_meta"]["video_path"])
		mbe = _mbe(y, SAMPLE_RATE)

		lbl = np.zeros((mbe.shape[0], 1), dtype=np.float32)
		for _, hit in info["hits"].iterrows():
			s = int(np.floor(hit["start"] * SAMPLE_RATE / HOP_LENGTH))
			e = int(np.ceil (hit["end"]   * SAMPLE_RATE / HOP_LENGTH))
			lbl[s:e, 0] = 1.0

		np.savez(out_npz, mbe, lbl)
		dt = time.time() - t0
		print(f"✔ {mbe.shape[0]} frames in {dt:.2f}s")

		with open(self._log_path(), "a") as f:
			f.write(json.dumps({
				"video": vname_ext,
				"frames": int(mbe.shape[0]),
				"duration_sec": round(dt, 2),
				"saved": out_npz
			}) + "\n")
		return mbe.shape[0]

	def _ensure_fold_npz(self):
		target = self._per_fold_npz_path(self.fold_id)
		if os.path.exists(target): return						# already cached

		print(f"▶ preprocessing fold data – generating {target}")
		ds = load_decorte_dataset(k_folds=4)
		per_video = {}

		for vname_ext, info in ds.items():
			n_frames = self._extract_and_cache_video(vname_ext, info)
			base = os.path.splitext(vname_ext)[0]
			mbe, lbl = np.load(self._per_video_npz_path(base)).values()
			per_video[vname_ext] = (mbe, lbl, info["fold_id"])

		# Build fold packs
		fold_k = max(v[2] for v in per_video.values()) + 1
		for f in range(fold_k):
			X_tr, Y_tr, X_te, Y_te = None, None, None, None
			for _, (mbe, lbl, fold) in per_video.items():
				if fold == f:
					X_te  = mbe if X_te is None else np.concatenate((X_te, mbe), axis=0)
					Y_te  = lbl if Y_te is None else np.concatenate((Y_te, lbl), axis=0)
				else:
					X_tr = mbe if X_tr is None else np.concatenate((X_tr, mbe), axis=0)
					Y_tr = lbl if Y_tr is None else np.concatenate((Y_tr, lbl), axis=0)

			scaler = preprocessing.StandardScaler()
			X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)

			out_fold = self._per_fold_npz_path(f + 1)
			np.savez(out_fold, X_tr, Y_tr, X_te, Y_te)
			print(f"[fold {f+1}] saved {out_fold} | train={len(X_tr)} test={len(X_te)}")

	# ───────── public Lightning hooks ─────────
	def setup(self, stage=None):
		self._ensure_fold_npz()								# ← on-demand preparer
		all_folds = _load_all_npz(self.cache_dir)
		fdata = all_folds[self.fold_id]

		aug = (stage != "validate" and stage != "test")
		self.train_ds = HitWindowDataset(fdata["train_x"], fdata["train_y"], augment=aug)
		self.val_ds   = HitWindowDataset(fdata["val_x"],   fdata["val_y"],   augment=False)

	def train_dataloader(self):
		return DataLoader(self.train_ds, self.batch_size, shuffle=True,
						  drop_last=True, num_workers=self.num_workers,
						  pin_memory=True)

	def val_dataloader(self):
		return DataLoader(self.val_ds, self.batch_size, shuffle=False,
						  num_workers=self.num_workers, pin_memory=True)

	# ───────── Optional helper for external callers ─────────
	@staticmethod
	def extract_video_to_npz(video_path:str, hits_df, out_npz:str):
		"""
		Simple stand-alone convenience wrapper so other scripts (e.g. inference)
		can generate a per-video .npz with **identical** logic.
		`hits_df` must have 'start' & 'end' columns in seconds.
		"""
		y = _ffmpeg_audio(video_path, SAMPLE_RATE)
		mbe = _mbe(y, SAMPLE_RATE)

		lbl = np.zeros((mbe.shape[0], 1), dtype=np.float32)
		for _, hit in hits_df.iterrows():
			s = int(np.floor(hit["start"] * SAMPLE_RATE / HOP_LENGTH))
			e = int(np.ceil (hit["end"]   * SAMPLE_RATE / HOP_LENGTH))
			lbl[s:e, 0] = 1.0
		np.savez(out_npz, mbe, lbl)
		print(f"📝 saved {out_npz}  ({mbe.shape[0]} frames)")

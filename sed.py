#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced CRNN training (hit / no-hit)

• Random 1 : 1 sampling *inside every __getitem__ call*
  – positives  = windows that contain ≥1 hit
  – negatives  = windows that contain 0 hits (strict)
  – both are drawn fresh for every batch → temporal augmentation
• Positive window is *randomly centred* around a hit (not always centred).
• Sequence length 64  (≈1.49 s input)   pool=[2,2,2] → 185 ms resolution
• BCE-With-Logits loss (no sigmoid in the model)
• Tab-only indentation
"""

from __future__ import print_function
import os, sys, random, datetime, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import metrics
from decorte_data_loader import load_decorte_dataset

# ───────────────────────── parameters ──────────────────────────
SR              = 44_100
NFFT            = 2048
HOP             = NFFT // 2          # 23.22 ms per frame
SEQ_LEN         = 64                 # 64 frames ≈ 1.49 s
POOL_SIZES      = [2, 2, 2]
BATCH_SIZE      = 128
MAX_EPOCH       = 200
PATIENCE        = 40
CACHE_DIR       = os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
ART_DIR         = os.path.expanduser(f"~/src/plai_cv/sed-crnn/train_artifacts/{datetime.datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(ART_DIR, exist_ok=True)
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FRAMES_1_SEC    = int(SR / (NFFT / 2.0))
# ───────────────────────────────────────────────────────────────

def load_fold_npz(fold_id):
	fp = os.path.join(CACHE_DIR, f"mbe_mon_fold{fold_id}.npz")
	d = np.load(fp)
	return d['arr_0'], d['arr_1'], d['arr_2'], d['arr_3']   # Xtr, Ytr, Xte, Yte

# ---------- helper: build negative-window start positions ----------
def valid_neg_starts(label_vec):
	"""
	Return all window start indices whose SEQ_LEN window contains only zeros.
	label_vec shape  (frames, 1)
	"""
	hit_mask = (label_vec[:,0] == 1).astype(np.uint8)
	# convolution trick: any overlap → sum > 0
	window = np.ones(SEQ_LEN, dtype=np.uint8)
	overlap = np.convolve(hit_mask, window, mode='valid')
	return np.where(overlap == 0)[0]

# ---------- dataset ----------
class WindowDataset(Dataset):
	def __init__(self, mel, lbl):
		self.mel   = mel        # (frames, 40)
		self.lbl   = lbl        # (frames, 1)
		self.pos_frames = np.where(lbl[:,0] == 1)[0].tolist()
		self.neg_starts = valid_neg_starts(lbl).tolist()
		self.frames = mel.shape[0]

	def __len__(self):
		# dataset length defined as 2× (# positive frames)  (balance)
		return len(self.pos_frames)*2

	def _rand_pos_window(self):
		hit = random.choice(self.pos_frames)
		min_start = max(0, hit - SEQ_LEN + 1)
		max_start = min(hit, self.frames - SEQ_LEN)
		start = random.randint(min_start, max_start)
		return start

	def _rand_neg_window(self):
		# keep trying until we pick a clean negative
		start = random.choice(self.neg_starts)
		return start

	def __getitem__(self, idx):
		if idx % 2 == 0:          # positive sample
			start = self._rand_pos_window()
		else:                     # negative sample
			start = self._rand_neg_window()

		x_win = self.mel[start:start+SEQ_LEN].T           # (40, 64)
		y_win = self.lbl[start:start+SEQ_LEN].T           # (1, 64)
		return torch.from_numpy(x_win).unsqueeze(0).float(), torch.from_numpy(y_win).float()

# ---------- CRNN ----------
class CRNN(nn.Module):
	def __init__(self, cnn_filt=128, dropout=0.5):
		super().__init__()
		self.cnn, self.bn, self.pool = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
		ch = 1
		for p in POOL_SIZES:
			self.cnn.append(nn.Conv2d(ch, cnn_filt, 3, padding=1))
			self.bn.append(nn.BatchNorm2d(cnn_filt))
			self.pool.append(nn.MaxPool2d((1,p)))
			ch = cnn_filt
		self.drop = nn.Dropout(dropout)
		# compute flat size
		with torch.no_grad():
			d = torch.zeros(1,1,40,SEQ_LEN)
			for c,b,p in zip(self.cnn,self.bn,self.pool):
				d = self.drop(p(torch.relu(b(c(d)))))
			flat = d.permute(0,2,1,3).reshape(1,40,-1).shape[-1]
		self.gru = nn.GRU(flat, 32, num_layers=2, batch_first=True, bidirectional=True)
		self.fc  = nn.Linear(64, 1)   # 32*2 → 1

	def forward(self, x):
		for c,b,p in zip(self.cnn,self.bn,self.pool):
			x = self.drop(p(torch.relu(b(c(x)))))
		x = x.permute(0,2,1,3)                # B,F,C,T
		B,F,C,T = x.shape
		x = x.reshape(B,F,C*T)
		x,_ = self.gru(x)
		return self.fc(x)                     # logits

# ---------- training loop ----------
def run_epoch(model, loader, loss_fn, opt=None):
	train = opt is not None
	model.train() if train else model.eval()
	total, preds, labels = 0., [], []
	for xb,yb in loader:
		xb,yb = xb.to(DEVICE), yb.to(DEVICE)
		if train: opt.zero_grad()
		out = model(xb)
		loss = loss_fn(out, yb)
		if train:
			loss.backward(); opt.step()
		total += loss.item()
		preds.append(torch.sigmoid(out).detach().cpu().numpy())
		labels.append(yb.cpu().numpy())
	return total/len(loader), np.concatenate(preds), np.concatenate(labels)

# ---------- main ----------
def main():
	print("[artifacts] →", ART_DIR)
	fold_npzs = [os.path.join(CACHE_DIR, f"mbe_mon_fold{i}.npz") for i in range(1,5)]
	best_ers = []

	for fold_id, fp in enumerate(fold_npzs, 1):
		npy = np.load(fp)
		Xtr,Ytr,Xte,Yte = npy['arr_0'], npy['arr_1'], npy['arr_2'], npy['arr_3']

		tr_ds = WindowDataset(Xtr, Ytr)
		vl_ds = WindowDataset(Xte, Yte)
		tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
		vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE)

		model = CRNN().to(DEVICE)
		opt   = optim.Adam(model.parameters(), lr=1e-3)
		crit  = nn.BCEWithLogitsLoss()

		best_er, best_ep, no_imp = 1e9, 0, 0
		tr_curve, vl_curve = [], []

		for ep in range(MAX_EPOCH):
			tr_loss,_,_ = run_epoch(model, tr_ld, crit, opt)
			vl_loss, vl_pred, vl_lab = run_epoch(model, vl_ld, crit)

			pred_bin = vl_pred > 0.5
			sc = metrics.compute_scores(pred_bin, vl_lab, frames_in_1_sec=FRAMES_1_SEC)
			tr_curve.append(tr_loss); vl_curve.append(vl_loss)

			print(f"[F{fold_id} E{ep:03}] tl={tr_loss:.4f} vl={vl_loss:.4f} "
			      f"f1={sc['f1_overall_1sec']:.4f} er={sc['er_overall_1sec']:.4f}")

			# overwrite plot
			plt.figure(); plt.plot(tr_curve,label='train'); plt.plot(vl_curve,label='val'); plt.legend(); plt.grid(True)
			plt.savefig(os.path.join(ART_DIR, f"fold{fold_id}_loss.png")); plt.close()

			# checkpoint
			if sc['er_overall_1sec'] < best_er:
				best_er, best_ep, no_imp = sc['er_overall_1sec'], ep, 0
				torch.save(model.state_dict(), os.path.join(ART_DIR, f"best_fold{fold_id}.pt"))
			else:
				no_imp += 1

			if no_imp > PATIENCE: break

		best_ers.append(best_er)
		print(f"[fold {fold_id}] best ER={best_er:.4f} at epoch {best_ep}")

	print("AVG ER across folds:", np.mean(best_ers))

if __name__ == "__main__":
	main()

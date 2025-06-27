#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRNN hit / no-hit training   (channels-first)

✓	pools only along time – preserves frequency
✓	random 1 : 1 pos / neg sampling per batch
✓	labels down-sampled to pooled-time steps (T' = 8)
✓	BCEWithLogitsLoss
✓	logs + checkpoints in ~/src/plai_cv/sed-crnn/train_artifacts/<timestamp>
"""

from __future__ import print_function
import os, random, datetime, json, math
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import metrics, utils
from decorte_data_loader import load_decorte_dataset

# ─────────── constants ───────────
SR				= 44_100
NFFT			= 2048
HOP				= NFFT // 2				# 23.22 ms
SEQ_LEN			= 64					# input frames per window
POOL_SIZES		= [2,2,2]				# (1,2) pooling ×3 → T' = 64 / 8 = 8
T_PRIME			= SEQ_LEN // math.prod(POOL_SIZES)	# 8
BATCH_SIZE		= 128
EPOCHS			= 200
PATIENCE		= 40
CACHE_DIR		= os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
ART_DIR			= os.path.expanduser(f"~/src/plai_cv/sed-crnn/train_artifacts/{datetime.datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(ART_DIR, exist_ok=True)
DEVICE			= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FRAMES_ORIG_SEC	= int(SR / HOP)			# ≈43
FRAMES_SEC_DWN	= max(1, FRAMES_ORIG_SEC // math.prod(POOL_SIZES))	# 5

# ─────────── helper: build valid negative starts ───────────
def valid_neg_starts(label_vec):
	hit = (label_vec[:,0] == 1).astype(np.uint8)
	window = np.ones(SEQ_LEN, dtype=np.uint8)
	overlap = np.convolve(hit, window, mode='valid')
	return np.where(overlap == 0)[0]

# ─────────── dataset ───────────
class WindowDS(Dataset):
	def __init__(self, mel, lbl):
		self.mel = mel								# (frames, 40)
		self.lbl = lbl								# (frames, 1)
		self.pos_frames = np.where(lbl[:,0]==1)[0].tolist()
		self.neg_starts = valid_neg_starts(lbl).tolist()
		self.total_frames = mel.shape[0]

	def __len__(self):							# balanced length
		return len(self.pos_frames)*2

	def _rand_pos_start(self):
		h = random.choice(self.pos_frames)
		min_st = max(0, h - SEQ_LEN + 1)
		max_st = min(h, self.total_frames - SEQ_LEN)
		return random.randint(min_st, max_st)

	def _rand_neg_start(self):
		return random.choice(self.neg_starts)

	def _downsample_lbl(self, win_lbl):
		# max-pool over every 8 frames → length 8
		return win_lbl.reshape(T_PRIME, -1).max(axis=1, keepdims=True)

	def __getitem__(self, idx):
		start = self._rand_pos_start() if idx % 2 == 0 else self._rand_neg_start()
		x = self.mel[start:start+SEQ_LEN].T					# (40,64)
		y = self.lbl[start:start+SEQ_LEN]					# (64,1)
		y_ds = self._downsample_lbl(y)						# (8,1)
		# return channels_first input
		return torch.from_numpy(x).unsqueeze(0).float(), torch.from_numpy(y_ds).float()

# ─────────── model ───────────
class CRNN(nn.Module):
	def __init__(self, cnn_filt=128, dropout=0.5):
		super().__init__()
		self.convs, self.bns, self.pools = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
		ch = 1
		for p in POOL_SIZES:
			self.convs.append(nn.Conv2d(ch, cnn_filt, kernel_size=3, padding=1))
			self.bns.append(nn.BatchNorm2d(cnn_filt))
			self.pools.append(nn.MaxPool2d(kernel_size=(1,p)))		# pool on time only
			ch = cnn_filt
		self.drop = nn.Dropout(dropout)

		# flat feature size after conv stack (C_out × F)
		with torch.no_grad():
			dummy = torch.zeros(1,1,40,SEQ_LEN)
			for c,b,p in zip(self.convs,self.bns,self.pools):
				dummy = self.drop(p(torch.relu(b(c(dummy)))))
			dummy = dummy.permute(0,3,1,2)				# [B,T',C,F]
			self.flat = dummy.shape[2]*dummy.shape[3]	# C×F

		self.gru = nn.GRU(self.flat, 32, num_layers=2, batch_first=True, bidirectional=True)
		self.fc  = nn.Linear(64, 1)					# 32*2 → 1

	def forward(self, x):							# x: [B,1,40,64]
		for c,b,p in zip(self.convs,self.bns,self.pools):
			x = self.drop(p(torch.relu(b(c(x)))))
		x = x.permute(0,3,1,2)						# [B,T',C,F]
		B,T,C,F = x.shape
		x = x.reshape(B,T,C*F)						# [B,T',flat]
		x,_ = self.gru(x)							# [B,T',64]
		return self.fc(x)							# logits [B,T',1]

# ─────────── train / eval helpers ───────────
def run_epoch(model, loader, loss_fn, opt=None):
	train = opt is not None
	model.train() if train else model.eval()
	total, preds, lbls = 0., [], []
	for xb,yb in loader:
		xb,yb = xb.to(DEVICE), yb.to(DEVICE)
		if train: opt.zero_grad()
		out = model(xb)
		loss = loss_fn(out, yb)
		if train: loss.backward(); opt.step()
		total += loss.item()
		preds.append(torch.sigmoid(out).detach().cpu().numpy())
		lbls.append(yb.cpu().numpy())
	return total/len(loader), np.concatenate(preds), np.concatenate(lbls)

# ─────────── main training loop ───────────
def main():
	print("[artifacts] →", ART_DIR)
	folds = [os.path.join(CACHE_DIR, f"mbe_mon_fold{i}.npz") for i in range(1,5)]
	best_ers = []

	for fid, npz_path in enumerate(folds,1):
		npy = np.load(npz_path)
		Xtr,Ytr,Xte,Yte = npy['arr_0'],npy['arr_1'],npy['arr_2'],npy['arr_3']

		tr_ds, vl_ds = WindowDS(Xtr,Ytr), WindowDS(Xte,Yte)
		tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
		vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False)

		model = CRNN().to(DEVICE)
		opt   = optim.Adam(model.parameters(), lr=1e-3)
		crit  = nn.BCEWithLogitsLoss()

		best_er, best_ep, no_imp, tr_curve, vl_curve = 1e9, 0, 0, [], []

		for ep in range(EPOCHS):
			tr_loss,_,_ = run_epoch(model, tr_ld, crit, opt)
			vl_loss, pred, gt = run_epoch(model, vl_ld, crit)

			tr_curve.append(tr_loss); vl_curve.append(vl_loss)
			bin_pred = pred > 0.5
			sc = metrics.compute_scores(bin_pred, gt, frames_in_1_sec=FRAMES_SEC_DWN)
			print(f"[F{fid} E{ep:03}] tl={tr_loss:.4f} vl={vl_loss:.4f} "
				  f"f1={sc['f1_overall_1sec']:.4f} er={sc['er_overall_1sec']:.4f}")

			# plot curve
			plt.figure(); plt.plot(tr_curve,label='train'); plt.plot(vl_curve,label='val')
			plt.grid(True); plt.legend(); plt.tight_layout()
			plt.savefig(os.path.join(ART_DIR,f"loss_fold{fid}.png")); plt.close()

			# checkpoint
			if sc['er_overall_1sec'] < best_er:
				best_er, best_ep, no_imp = sc['er_overall_1sec'], ep, 0
				torch.save(model.state_dict(), os.path.join(ART_DIR,f"best_fold{fid}.pt"))
			else:
				no_imp += 1
			if no_imp > PATIENCE: break

		best_ers.append(best_er)
		print(f"[fold {fid}] best ER={best_er:.4f} @ ep={best_ep}")

	print("AVG ER over folds:", np.mean(best_ers))

if __name__ == "__main__":
	main()

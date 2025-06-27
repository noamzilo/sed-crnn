#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRNN hit / no–hit training (channels_first)

• pools only along time (stride-2 three times) → 64 → 8 frames
• balanced 1 : 1 pos/neg sampling per batch
• labels max-pooled to 8 steps to match model output
• BCEWithLogitsLoss
• every artifact path is printed
• epoch timing + progress printed
"""

import os, random, datetime, math, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import metrics, utils

# ─────────── configuration ───────────
SAMPLE_RATE				= 44_100
N_FFT					= 2048
HOP_LENGTH				= N_FFT // 2
FPS_ORIG				= int(SAMPLE_RATE / HOP_LENGTH)		# ≈43 fps

SEQ_LEN_IN				= 64									# input frames
TIME_POOL				= [2, 2, 2]							# pool only on time
SEQ_LEN_OUT				= SEQ_LEN_IN // math.prod(TIME_POOL)	# 8 frames
FPS_OUT					= FPS_ORIG // math.prod(TIME_POOL)	# 5 fps

BATCH_SIZE				= 128
MAX_EPOCHS				= 200
EARLY_STOP				= 40
NUM_WORKERS				= 4

CACHE_DIR				= os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
ART_DIR					= os.path.expanduser(f"~/src/plai_cv/sed-crnn/train_artifacts/{datetime.datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(ART_DIR, exist_ok=True)
DEVICE					= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────── utils ───────────
def log_save(path: str):
	print(f"📁 Saved → {path}")

def find_clean_negatives(label_vec: np.ndarray):
	mask = (label_vec[:,0] == 1).astype(np.uint8)
	window = np.ones(SEQ_LEN_IN, dtype=np.uint8)
	overlap = np.convolve(mask, window, mode='valid')
	return np.where(overlap == 0)[0]

# ─────────── dataset ───────────
class HitWindowDataset(Dataset):
	def __init__(self, mel: np.ndarray, lab: np.ndarray):
		self.mel, self.lab = mel, lab
		self.pos_frames = np.where(lab[:,0] == 1)[0].tolist()
		self.neg_starts = find_clean_negatives(lab).tolist()
		self.total_frames = mel.shape[0]

	def __len__(self): return len(self.pos_frames)*2

	def _rand_pos(self):
		center = random.choice(self.pos_frames)
		a = max(0, center - SEQ_LEN_IN + 1)
		b = min(center, self.total_frames - SEQ_LEN_IN)
		return random.randint(a, b)

	def _rand_neg(self): return random.choice(self.neg_starts)

	def _pool_labels(self, lab_win):
		return lab_win.reshape(SEQ_LEN_OUT, -1).max(axis=1, keepdims=True)

	def __getitem__(self, idx):
		start = self._rand_pos() if idx % 2 == 0 else self._rand_neg()
		x = self.mel[start:start+SEQ_LEN_IN].T						# (40,64)
		y = self._pool_labels(self.lab[start:start+SEQ_LEN_IN])		# (8,1)
		return torch.from_numpy(x).unsqueeze(0).float(), torch.from_numpy(y).float()

# ─────────── model ───────────
class TimePooledCRNN(nn.Module):
	def __init__(self, conv_channels=128, dropout=0.5):
		super().__init__()
		self.convs, self.bns, self.pools = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
		ch = 1
		for p in TIME_POOL:
			self.convs.append(nn.Conv2d(ch, conv_channels, 3, padding=1))
			self.bns.append(nn.BatchNorm2d(conv_channels))
			self.pools.append(nn.MaxPool2d(kernel_size=(1,p)))
			ch = conv_channels
		self.drop = nn.Dropout(dropout)

		with torch.no_grad():
			d = torch.zeros(1,1,40,SEQ_LEN_IN)
			for c,b,p in zip(self.convs,self.bns,self.pools):
				d = self.drop(p(torch.relu(b(c(d)))))
			d = d.permute(0,3,1,2)					# [B,T',C,F]
			self.flat = d.shape[2]*d.shape[3]		# C×F

		self.gru = nn.GRU(self.flat, 32, num_layers=2,
						  batch_first=True, bidirectional=True)
		self.fc = nn.Linear(64, 1)

	def forward(self, x):							# x [B,1,40,64]
		for c,b,p in zip(self.convs,self.bns,self.pools):
			x = self.drop(p(torch.relu(b(c(x)))))
		x = x.permute(0,3,1,2)						# [B,T',C,F]
		B,T,C,F = x.shape
		x = x.reshape(B,T,C*F)						# [B,T',features]
		x,_ = self.gru(x)
		return self.fc(x)							# logits [B,T',1]

# ─────────── load all folds once ───────────
def load_all_npz(folder):
	folds = {}
	for i in range(1,5):
		fp = os.path.join(folder, f"mbe_mon_fold{i}.npz")
		arr = np.load(fp)
		folds[i] = {
			"train_x": arr['arr_0'], "train_y": arr['arr_1'],
			"val_x":   arr['arr_2'], "val_y":   arr['arr_3'],
		}
		print(f"🔄 loaded into RAM → fold {i}  ({arr['arr_0'].nbytes/1e6:0.1f} MB train)")
	return folds

# ─────────── epoch runner ───────────
def run_epoch(model, loader, loss_fn, optim=None):
	train = optim is not None
	model.train() if train else model.eval()
	total, preds, labels = 0., [], []
	for xb,yb in loader:
		xb,yb = xb.to(DEVICE), yb.to(DEVICE)
		if train: optim.zero_grad()
		out = model(xb)
		loss = loss_fn(out, yb)
		if train: loss.backward(); optim.step()
		total += loss.item()
		preds.append(torch.sigmoid(out).detach().cpu().numpy())
		labels.append(yb.detach().cpu().numpy())
	return total/len(loader), np.concatenate(preds), np.concatenate(labels)

# ─────────── training loop ───────────
def main():
	print(f"ARTIFACTS → {ART_DIR}")
	all_folds = load_all_npz(CACHE_DIR)
	error_rates = []

	for fold_id in range(1,5):
		fdata = all_folds[fold_id]
		train_ds = HitWindowDataset(fdata["train_x"], fdata["train_y"])
		val_ds   = HitWindowDataset(fdata["val_x"],   fdata["val_y"])
		train_ld = DataLoader(train_ds, BATCH_SIZE, True,  drop_last=True,
							  num_workers=NUM_WORKERS, pin_memory=True)
		val_ld   = DataLoader(val_ds,   BATCH_SIZE, False,
							  num_workers=NUM_WORKERS, pin_memory=True)

		model = TimePooledCRNN().to(DEVICE)
		optimizer = optim.Adam(model.parameters(), lr=1e-3)
		criterion = nn.BCEWithLogitsLoss()

		best_er, best_epoch, no_imp = float('inf'), 0, 0
		train_curve, val_curve = [], []
		start_train = time.time()

		for epoch in range(1, MAX_EPOCHS+1):
			ep_start = time.time()
			train_loss, train_pred, train_true = run_epoch(model, train_ld, criterion, optimizer)
			val_loss,   val_pred,   val_true   = run_epoch(model, val_ld,   criterion)

			train_curve.append(train_loss); val_curve.append(val_loss)

			train_bin = train_pred > 0.5
			val_bin   = val_pred   > 0.5
			train_scores = metrics.compute_scores(train_bin, train_true, frames_in_1_sec=FPS_OUT)
			val_scores   = metrics.compute_scores(val_bin,   val_true,   frames_in_1_sec=FPS_OUT)

			elapsed_ep = time.time() - ep_start
			total_elapsed = time.time() - start_train
			print(f"[Fold {fold_id}] [Epoch {epoch}/{MAX_EPOCHS}] "
				  f"Δt={elapsed_ep:.1f}s total={total_elapsed/60:.1f} min | "
				  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
				  f"train_f1={train_scores['f1_overall_1sec']:.3f} "
				  f"val_f1={val_scores['f1_overall_1sec']:.3f} | "
				  f"val_ER={val_scores['er_overall_1sec']:.3f}")

			# save loss curve every epoch
			plt.figure(figsize=(5,3))
			plt.plot(train_curve,label='train'); plt.plot(val_curve,label='val'); plt.grid()
			plt.xlabel('epoch'); plt.ylabel('BCE loss'); plt.legend()
			plot_path = os.path.join(ART_DIR, f"loss_fold{fold_id}.png")
			plt.tight_layout(); plt.savefig(plot_path); plt.close()
			log_save(plot_path)

			# checkpoint best ER
			if val_scores['er_overall_1sec'] < best_er:
				best_er, best_epoch, no_imp = val_scores['er_overall_1sec'], epoch, 0
				model_path = os.path.join(ART_DIR, f"best_fold{fold_id}.pt")
				torch.save(model.state_dict(), model_path); log_save(model_path)
			else:
				no_imp += 1
			if no_imp > EARLY_STOP: break

		error_rates.append(best_er)
		print(f"✔️ Fold {fold_id} best ER={best_er:.3f} @ epoch={best_epoch}")

	print(f"\n🧮 Average ER across folds: {np.mean(error_rates):.3f}")

if __name__ == "__main__":
	main()

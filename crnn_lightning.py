#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightningModule – Decorte hit-detection model
✓	architecture & loss exactly as §4.2.1 (Table 2) of the paper
✓	conv depth 64, TIME_POOL = [5,2,2]  (→ overall pool ×20)
✓	Bi-GRU #1 = 32 units, Bi-GRU #2 = 16 units
✓	time-distributed dense → 16 → 1 (sigmoid)
✓	Binary Focal Loss (α = 0.25, γ = 2.0)
✓	L2 weight-decay, ReduceLROnPlateau, dropout 0.4
✓	tabs only
"""

import os, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch, torch.nn as nn
import pytorch_lightning as pl
import metrics
from train_constants import SEQ_LEN_IN, TIME_POOL, SEQ_LEN_OUT, SAMPLE_RATE, HOP_LENGTH, FPS_ORIG, FPS_OUT

# ────────────────────────────────────────────────────────────────
#  Binary Focal BCE (α & γ from original focal-loss paper)
# ────────────────────────────────────────────────────────────────
class FocalBCELoss(nn.Module):
	def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
		super().__init__()
		self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

	def forward(self, logits, targets):
		if logits.shape != targets.shape:
			print(f"🔥 shape mismatch: logits={logits.shape}, targets={targets.shape}")
		pt = torch.sigmoid(logits)
		pt = torch.where(targets == 1, pt, 1 - pt)
		loss = -self.alpha * (1 - pt) ** self.gamma * pt.log()
		return loss.mean() if self.reduction == "mean" else loss.sum()
# ────────────────────────────────────────────────────────────────
#  CNN + GRU (backbone copied from Table 2)
# ────────────────────────────────────────────────────────────────
class TimePooledCRNN(nn.Module):
	def __init__(self, dropout=0.4):
		super().__init__()

		self.time_pool_layers = []
		self.conv_stack = nn.Sequential()
		in_channels = 1
		conv_depth = 64
		for i, pool_factor in enumerate(TIME_POOL):
			self.conv_stack.append(nn.Conv2d(in_channels, conv_depth, kernel_size=3, padding=1))
			self.conv_stack.append(nn.BatchNorm2d(conv_depth))
			self.conv_stack.append(nn.ReLU())
			self.conv_stack.append(nn.MaxPool2d((1, pool_factor)))
			in_channels = conv_depth
		self.conv_stack.append(nn.Dropout(dropout))

		# dummy forward to infer flattened shape
		with torch.no_grad():
			dummy = torch.zeros(1, 1, 40, SEQ_LEN_IN)
			dummy = self.conv_stack(dummy)			# → (1,64,40,T_out)
			dummy = dummy.permute(0, 3, 1, 2)		# → (1,T_out,64,40)
			T_actual = dummy.shape[1]
			self.flat = dummy.shape[2] * dummy.shape[3]	# 64 × 40 = 2560 if unchanged

		assert T_actual == SEQ_LEN_OUT, f"❌ CNN downsampling mismatch: expected SEQ_LEN_OUT={SEQ_LEN_OUT}, but got T={T_actual} from TIME_POOL={TIME_POOL}"

		self.gru1 = nn.GRU(self.flat, 32, bidirectional=True, batch_first=True)
		self.gru2 = nn.GRU(64, 16, bidirectional=True, batch_first=True)
		self.dense1 = nn.Linear(32, 16)
		self.dense2 = nn.Linear(16, 1)

	def forward(self, x):					# x [B,1,40,T]
		x = self.conv_stack(x)				# [B,64,40,T_out]
		x = x.permute(0, 3, 1, 2)			# [B,T_out,64,40]
		B, T, C, F = x.shape
		x = x.reshape(B, T, C * F)			# [B,T,2560]
		x, _ = self.gru1(x)
		x, _ = self.gru2(x)
		x = torch.relu(self.dense1(x))
		return self.dense2(x)				# [B,T_out,1]# logits [B,12,1]

# ────────────────────────────────────────────────────────────────
#  Lightning wrapper
# ────────────────────────────────────────────────────────────────
class CRNNLightning(pl.LightningModule):
	def __init__(self, fold_id: int, art_dir: str,
				 lr=1e-3, weight_decay=1e-4, dropout=0.4):
		super().__init__()
		self.save_hyperparameters(ignore=["art_dir"])
		self.art_dir, self.model = art_dir, TimePooledCRNN(dropout)
		self.loss_fn = FocalBCELoss()
		self.tr_losses, self.val_losses = [], []
		self.preds, self.trues = [], []

	def forward(self, x): return self.model(x)

	def training_step(self, batch, _):
		x, y = batch; loss = self.loss_fn(self(x), y)
		self.log("train_loss", loss, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch, _):
		x, y = batch
		logits = self(x); loss = self.loss_fn(logits, y)
		self.log("val_loss", loss, on_epoch=True, prog_bar=True)
		self.preds.append(torch.sigmoid(logits).cpu())
		self.trues.append(y.cpu())

	def on_validation_epoch_end(self):
		p = torch.cat(self.preds).numpy()
		t = torch.cat(self.trues).numpy()
		self.preds.clear();
		self.trues.clear()

		scores = metrics.compute_scores(p > 0.5, t, FPS_OUT)
		er = scores["er_overall_1sec"]
		f1 = scores["f1_overall_1sec"]

		self.log("val_er", er, prog_bar=True)
		self.log("val_f1", f1, prog_bar=True)

		train_loss = self.trainer.callback_metrics.get("train_loss_epoch")
		val_loss = self.trainer.callback_metrics.get("val_loss_epoch")

		# initialize buffers on first run
		if not hasattr(self, "val_f1s"):
			self.val_f1s, self.val_ers = [], []
			self.best_f1, self.best_er, self.best_loss = -1, 1, float("inf")
			self.best_f1_epoch = self.best_er_epoch = self.best_loss_epoch = -1

		epoch = self.current_epoch

		# save all values
		if train_loss is not None and val_loss is not None:
			self.tr_losses.append(train_loss.item())
			self.val_losses.append(val_loss.item())

		self.val_f1s.append(f1)
		self.val_ers.append(er)

		# update bests
		if val_loss is not None and val_loss.item() < self.best_loss:
			self.best_loss = val_loss.item()
			self.best_loss_epoch = epoch
		if f1 > self.best_f1:
			self.best_f1 = f1
			self.best_f1_epoch = epoch
		if er < self.best_er:
			self.best_er = er
			self.best_er_epoch = epoch

		# inline print summary
		loss_str = f"{val_loss.item():.3f} (best={self.best_loss:.3f}@{self.best_loss_epoch})" if val_loss is not None else "?"
		f1_str = f"{f1:.3f} (best={self.best_f1:.3f}@{self.best_f1_epoch})"
		er_str = f"{er:.3f} (best={self.best_er:.3f}@{self.best_er_epoch})"
		print(f"📊 Epoch {epoch} | loss={loss_str} | F1={f1_str} | ER={er_str}")

		# Plot
		plt.figure(figsize=(12, 4))

		plt.subplot(1, 3, 1)
		plt.plot(self.tr_losses, label="train")
		plt.plot(self.val_losses, label="val")
		plt.title("Focal Loss")
		plt.xlabel("Epoch");
		plt.grid();
		plt.legend()

		plt.subplot(1, 3, 2)
		plt.plot(self.val_f1s, label="F1 score")
		plt.axhline(self.best_f1, linestyle="--", color="gray", linewidth=0.8)
		plt.title("F1 (1s block)")
		plt.xlabel("Epoch");
		plt.grid();
		plt.legend()

		plt.subplot(1, 3, 3)
		plt.plot(self.val_ers, label="ER")
		plt.axhline(self.best_er, linestyle="--", color="gray", linewidth=0.8)
		plt.title("Error Rate (1s block)")
		plt.xlabel("Epoch");
		plt.grid();
		plt.legend()

		os.makedirs(self.art_dir, exist_ok=True)
		path = os.path.join(self.art_dir, f"metrics_fold{self.hparams.fold_id}.png")
		plt.tight_layout();
		plt.savefig(path);
		plt.close()
		print(f"📈 Saved metrics → {path}")

	def configure_optimizers(self):
		opt = torch.optim.Adam(self.parameters(),
							   lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
														   factor=0.5, patience=10)
		return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

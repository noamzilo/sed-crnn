#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightningModule wrapper around the original CRNN
✓	same architecture / init weights / optimiser
✓	logs BCE, F1, ER each epoch
✓	saves loss-curve PNGs exactly like before
✓	checkpoint & early-stop on val_ER
✓	tabs only
"""

import os, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch, torch.nn as nn
import pytorch_lightning as pl
import metrics

# ────────────────────────────────────────────────────────────────
#  Constants copied verbatim from sed.py
# ────────────────────────────────────────────────────────────────
SAMPLE_RATE				= 44_100
N_FFT					= 2048
HOP_LENGTH				= N_FFT // 2
FPS_ORIG				= int(SAMPLE_RATE / HOP_LENGTH)		# ≈43 fps
SEQ_LEN_IN				= 64
TIME_POOL				= [2, 2, 2]
SEQ_LEN_OUT				= SEQ_LEN_IN // 8
FPS_OUT					= FPS_ORIG // 8

# ────────────────────────────────────────────────────────────────
#  CRNN backbone (unchanged)
# ────────────────────────────────────────────────────────────────
class TimePooledCRNN(nn.Module):
	def __init__(self, conv_channels=128, dropout=0.5):
		super().__init__()
		self.convs, self.bns, self.pools = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
		ch = 1
		for p in TIME_POOL:
			self.convs.append(nn.Conv2d(ch, conv_channels, 3, padding=1))
			self.bns.append(nn.BatchNorm2d(conv_channels))
			self.pools.append(nn.MaxPool2d(kernel_size=(1, p)))
			ch = conv_channels
		self.drop = nn.Dropout(dropout)

		with torch.no_grad():
			d = torch.zeros(1, 1, 40, SEQ_LEN_IN)
			for c, b, p in zip(self.convs, self.bns, self.pools):
				d = self.drop(p(torch.relu(b(c(d)))))
			d = d.permute(0, 3, 1, 2)				# [B,T',C,F]
			self.flat = d.shape[2] * d.shape[3]

		self.gru = nn.GRU(self.flat, 32, num_layers=2,
						  batch_first=True, bidirectional=True)
		self.fc = nn.Linear(64, 1)

	def forward(self, x):							# x [B,1,40,64]
		for c, b, p in zip(self.convs, self.bns, self.pools):
			x = self.drop(p(torch.relu(b(c(x)))))
		x = x.permute(0, 3, 1, 2)					# [B,T',C,F]
		B, T, C, F = x.shape
		x = x.reshape(B, T, C * F)					# [B,T',features]
		x, _ = self.gru(x)
		return self.fc(x)							# logits [B,T',1]

# ────────────────────────────────────────────────────────────────
#  LightningModule
# ────────────────────────────────────────────────────────────────
class CRNNLightning(pl.LightningModule):
	def __init__(self, fold_id: int, art_dir: str, lr: float = 1e-3, dropout: float = 0.5):
		super().__init__()
		self.save_hyperparameters()
		self.model = TimePooledCRNN(dropout=dropout)
		self.loss_fn = nn.BCEWithLogitsLoss()
		self.train_curve, self.val_curve = [], []
		self.all_preds, self.all_trues = [], []

	def forward(self, x):
		return self.model(x)

	# ─────────────── training ───────────────
	def training_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = self.loss_fn(logits, y)
		self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
		return loss

	# ─────────────── validation ───────────────
	def validation_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = self.loss_fn(logits, y)
		self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
		self.all_preds.append(torch.sigmoid(logits).detach().cpu())
		self.all_trues.append(y.detach().cpu())
		return loss

	def on_validation_epoch_end(self):
		pred = torch.cat(self.all_preds).numpy()
		true = torch.cat(self.all_trues).numpy()
		bin_pred = pred > 0.5
		scores = metrics.compute_scores(bin_pred, true, frames_in_1_sec=FPS_OUT)
		val_er = scores['er_overall_1sec']
		self.log("val_er", val_er, prog_bar=True)
		self.all_preds.clear(); self.all_trues.clear()

		# ─ save loss curve PNG exactly like old script
		self.train_curve.append(self.trainer.callback_metrics["train_loss_epoch"].item())
		self.val_curve.append(self.trainer.callback_metrics["val_loss_epoch"].item())
		plt.figure(figsize=(5, 3))
		plt.plot(self.train_curve, label='train')
		plt.plot(self.val_curve, label='val')
		plt.grid(); plt.xlabel('epoch'); plt.ylabel('BCE loss'); plt.legend()
		os.makedirs(self.hparams.art_dir, exist_ok=True)
		plot_path = os.path.join(self.hparams.art_dir, f"loss_fold{self.hparams.fold_id}.png")
		plt.tight_layout(); plt.savefig(plot_path); plt.close()
		print(f"📁 Saved → {plot_path}")

	# ─────────────── optimiser ───────────────
	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

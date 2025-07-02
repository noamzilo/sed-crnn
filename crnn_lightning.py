#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crnn_lightning.py — Decorte hit-detection LightningModule
with extra debug logging and per-epoch logits dumps.

Tabs only.
"""

import os, matplotlib

from . import metrics

matplotlib.use('Agg')				# no GUI required
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .train_constants import (
	SEQ_LEN_IN, TIME_POOL, SEQ_LEN_OUT,
	FPS_OUT, N_MELS, CONV_DEPTH,
	GRU1_UNITS, GRU2_UNITS, DENSE1_UNITS
)

EPS = 1e-12

# ───────────────────────────────────────────────
#  Focal BCE loss
# ───────────────────────────────────────────────
class FocalBCELoss(nn.Module):
	def __init__(self, alpha=.25, gamma=2., reduction="mean"):
		super().__init__()
		self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
	def forward(self, logits, targets):
		pt = torch.sigmoid(logits)
		pt = torch.where(targets == 1, pt, 1 - pt)
		loss = -self.alpha * (1 - pt)**self.gamma * torch.log(pt + EPS)
		return loss.mean() if self.reduction == "mean" else loss.sum()

# ───────────────────────────────────────────────
#  Backbone
# ───────────────────────────────────────────────
class TimePooledCRNN(nn.Module):
	def __init__(self, dropout=0.4):
		super().__init__()
		self.conv_stack = nn.Sequential()
		in_c = 1
		for pool in TIME_POOL:
			self.conv_stack.append(nn.Conv2d(in_c, CONV_DEPTH, 3, padding=1))
			self.conv_stack.append(nn.BatchNorm2d(CONV_DEPTH))
			self.conv_stack.append(nn.ReLU())
			self.conv_stack.append(nn.MaxPool2d((1, pool)))
			in_c = CONV_DEPTH
		self.conv_stack.append(nn.Dropout(dropout))

		with torch.no_grad():
			d = torch.zeros(1, 1, N_MELS, SEQ_LEN_IN)
			d = self.conv_stack(d).permute(0, 3, 1, 2)
			self.T_out = d.shape[1]
			self._flat = int(np.prod(d.shape[2:]))
		assert self.T_out == SEQ_LEN_OUT, f"Down-sampling mismatch {self.T_out}≠{SEQ_LEN_OUT}"

		self.gru1 = nn.GRU(self._flat, GRU1_UNITS, bidirectional=True, batch_first=True)
		self.gru2 = nn.GRU(2*GRU1_UNITS, GRU2_UNITS, bidirectional=True, batch_first=True)
		self.d1   = nn.Linear(2*GRU2_UNITS, DENSE1_UNITS)
		self.d2   = nn.Linear(DENSE1_UNITS, 1)

	def forward(self, x):						# x: [B,1,40,T_in]
		x = self.conv_stack(x)					# [B,C,40,T_out]
		x = x.permute(0, 3, 1, 2)				# [B,T,C,40]
		b, t, c, f = x.shape
		x = x.reshape(b, t, c*f)				# [B,T,flat]
		x, _ = self.gru1(x)
		x, _ = self.gru2(x)
		x = torch.relu(self.d1(x))
		return self.d2(x)						# logits [B,T,1]

# ───────────────────────────────────────────────
#  Lightning wrapper
# ───────────────────────────────────────────────
class CRNNLightning(pl.LightningModule):
	def __init__(self, fold_id:int, art_dir:str,
				 lr=1e-3, weight_decay=1e-4, dropout=0.4):
		super().__init__()
		self.save_hyperparameters(ignore=["art_dir"])
		self.art_dir = art_dir
		self.model   = TimePooledCRNN(dropout)
		self.loss_fn = FocalBCELoss()

		self._buf = {m: {"preds":[], "trues":[], "losses":[]} for m in ["train","val"]}
		self.track = {k:[] for k in [
			"loss_tr","loss_val",
			"f1_1s_tr","f1_1s_val","er_1s_tr","er_1s_val",
			"f1_fr_tr","f1_fr_val","er_fr_tr","er_fr_val"
		]}

	# ──────────────────────────
	def forward(self, x): return self.model(x)

	def predict_step(self, batch, batch_idx):
		if isinstance(batch,(tuple,list)):
			x = batch[0]
		else:
			x = batch
		return self(x)

	# ────────────────────────── helpers
	def _collect(self, logits, y, loss, mode):
		self._buf[mode]["preds"].append(torch.sigmoid(logits))
		self._buf[mode]["trues"].append(y)
		self._buf[mode]["losses"].append(loss.detach())

	def _aggregate(self, mode):
		p_t = torch.cat(self._buf[mode]["preds"])
		t_t = torch.cat(self._buf[mode]["trues"])
		loss = torch.stack(self._buf[mode]["losses"]).mean().item()
		for k in self._buf[mode]: self._buf[mode][k].clear()

		p = p_t.detach().cpu().numpy()
		t = t_t.detach().cpu().numpy()
		p_bin = (p > 0.5).astype(np.uint8)
		t_bin = t.astype(np.uint8)

		tn = np.logical_and(p_bin==0, t_bin==0).sum()
		fp = np.logical_and(p_bin==1, t_bin==0).sum()
		fn = np.logical_and(p_bin==0, t_bin==1).sum()
		tp = np.logical_and(p_bin==1, t_bin==1).sum()
		cm = np.array([[tn,fp],[fn,tp]])

		f1_fr = metrics.f1_overall_framewise(p_bin, t_bin)
		er_fr = metrics.er_overall_framewise(p_bin, t_bin)
		f1_1s = metrics.f1_overall_1sec(p_bin, t_bin, FPS_OUT)
		er_1s = metrics.er_overall_1sec(p_bin, t_bin, FPS_OUT)

		return dict(loss=loss, f1_frame=f1_fr, er_frame=er_fr,
					f1_1s=f1_1s, er_1s=er_1s, cm=cm)

	def _plot_epoch(self, ep, tr, val):
		os.makedirs(self.art_dir, exist_ok=True)
		plt.figure(figsize=(14,6))
		def _line(ax,tr,val,title):
			ax.plot(tr,label="train"); ax.plot(val,label="val")
			ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(); ax.legend()
		_line(plt.subplot(2,3,1), self.track["loss_tr"], self.track["loss_val"], "Focal Loss")
		_line(plt.subplot(2,3,2), self.track["f1_1s_tr"], self.track["f1_1s_val"], "F1 (1 s)")
		_line(plt.subplot(2,3,3), self.track["er_1s_tr"], self.track["er_1s_val"], "ER (1 s)")
		_line(plt.subplot(2,3,6), self.track["f1_fr_tr"], self.track["f1_fr_val"], "F1 (frame)")
		def _cm(ax,M,title):
			ax.imshow(M,cmap="Blues")
			for i in range(2):
				for j in range(2):
					ax.text(j,i,f"{M[i,j]}",ha="center",va="center",
							color="white" if M[i,j]>M.max()/2 else "black")
			ax.set_xticks([0,1]); ax.set_yticks([0,1])
			ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
			ax.set_xlabel("Pred"); ax.set_ylabel("True"); ax.set_title(title)
		_cm(plt.subplot(2,3,4), tr["cm"], f"Train CM (e{ep})")
		_cm(plt.subplot(2,3,5), val["cm"], f"Val CM (e{ep})")
		out = os.path.join(self.art_dir, f"metrics_fold{self.hparams.fold_id}.png")
		plt.tight_layout(); plt.savefig(out); plt.close()
		print(f"📈 Saved metrics → {out}")

	# ────────────────────────── Lightning hooks
	def training_step(self, batch, _):
		x,y = batch
		logits = self(x)
		loss = self.loss_fn(logits, y)
		self._collect(logits,y,loss,"train")
		self.log("train_loss", loss, on_epoch=True, prog_bar=True)

		if self.global_step % 100 == 0:
			print(f"Train step={self.global_step}")
			print("Sigmoid sample:\n",
				  torch.sigmoid(logits[:4]).detach().cpu().numpy().reshape(-1, SEQ_LEN_OUT))
		return loss

	def on_train_epoch_end(self):
		tr = self._aggregate("train")
		self.track["loss_tr"].append(tr["loss"])
		self.track["f1_1s_tr"].append(tr["f1_1s"]); self.track["er_1s_tr"].append(tr["er_1s"])
		self.track["f1_fr_tr"].append(tr["f1_frame"]); self.track["er_fr_tr"].append(tr["er_frame"])
		self._last_train = tr

	def validation_step(self, batch, _):
		x,y = batch
		logits = self(x)
		loss = self.loss_fn(logits, y)
		self._collect(logits,y,loss,"val")
		self.log("val_loss", loss, on_epoch=True, prog_bar=True)

		if not hasattr(self,"_val_logged"):
			np.savez(os.path.join(self.art_dir,
					 f"logits_e{self.current_epoch:03d}.npz"),
					 logits=logits[:8].detach().cpu().numpy())
			self._val_logged = True

	def on_validation_epoch_start(self):
		self._val_logged = False

	def on_validation_epoch_end(self):
		val = self._aggregate("val")
		self.track["loss_val"].append(val["loss"])
		self.track["f1_1s_val"].append(val["f1_1s"]); self.track["er_1s_val"].append(val["er_1s"])
		self.track["f1_fr_val"].append(val["f1_frame"]); self.track["er_fr_val"].append(val["er_frame"])
		self.log("val_er_1s", val["er_1s"], prog_bar=True)
		self.log("val_f1_1s", val["f1_1s"], prog_bar=True)

		if not hasattr(self, "_last_train"):	# sanity-check pass
			self._last_train = val.copy()

		self._plot_epoch(self.current_epoch, self._last_train, val)

	def configure_optimizers(self):
		opt = torch.optim.Adam(self.parameters(),
				lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
				opt, mode='min', factor=.5, patience=10)
		return {"optimizer":opt,
				"lr_scheduler":{"scheduler":sched,"monitor":"val_loss"}}

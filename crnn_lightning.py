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
from train_constants import (
	SEQ_LEN_IN, TIME_POOL, SEQ_LEN_OUT,
	SAMPLE_RATE, HOP_LENGTH, FPS_ORIG, FPS_OUT,
	N_MELS, CONV_DEPTH, GRU1_UNITS, GRU2_UNITS, DENSE1_UNITS
)

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
		self.conv_stack = nn.Sequential()
		in_channels = 1
		for i, pool_factor in enumerate(TIME_POOL):
			self.conv_stack.append(nn.Conv2d(in_channels, CONV_DEPTH, kernel_size=3, padding=1))
			self.conv_stack.append(nn.BatchNorm2d(CONV_DEPTH))
			self.conv_stack.append(nn.ReLU())
			self.conv_stack.append(nn.MaxPool2d((1, pool_factor)))
			in_channels = CONV_DEPTH
		self.conv_stack.append(nn.Dropout(dropout))

		# dummy forward to infer flattened shape
		with torch.no_grad():
			dummy = torch.zeros(1, 1, N_MELS, SEQ_LEN_IN)
			dummy = self.conv_stack(dummy)
			dummy = dummy.permute(0, 3, 1, 2)
			T_actual = dummy.shape[1]
			self.flat = dummy.shape[2] * dummy.shape[3]

		assert T_actual == SEQ_LEN_OUT, f"❌ CNN downsampling mismatch: expected SEQ_LEN_OUT={SEQ_LEN_OUT}, but got T={T_actual} from TIME_POOL={TIME_POOL}"

		self.gru1 = nn.GRU(self.flat, GRU1_UNITS, bidirectional=True, batch_first=True)
		self.gru2 = nn.GRU(2 * GRU1_UNITS, GRU2_UNITS, bidirectional=True, batch_first=True)
		self.dense1 = nn.Linear(2 * GRU2_UNITS, DENSE1_UNITS)
		self.dense2 = nn.Linear(DENSE1_UNITS, 1)

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
		self.train_preds, self.train_trues = [], []

	def forward(self, x): return self.model(x)

	def training_step(self, batch, _):
		x, y = batch
		logits = self(x)
		loss = self.loss_fn(logits, y)

		# store for metrics (keep on GPU)
		with torch.no_grad():
			self.train_preds.append(torch.sigmoid(logits))
			self.train_trues.append(y)

		self.log("train_loss", loss, on_epoch=True, prog_bar=True)
		return loss

	def on_training_epoch_end(self, outputs):
		# concatenate along batch axis (dim=0)
		preds = torch.cat(self.train_preds).detach().cpu().numpy()
		trues = torch.cat(self.train_trues).detach().cpu().numpy()

		self.train_preds.clear()
		self.train_trues.clear()

		f1_1s = metrics.f1_overall_1sec(preds > 0.5, trues, FPS_OUT)
		er_1s = metrics.er_overall_1sec(preds > 0.5, trues, FPS_OUT)
		f1_frame = metrics.f1_overall_framewise(preds > 0.5, trues)
		er_frame = metrics.er_overall_framewise(preds > 0.5, trues)

		self.log("train_f1_1s", f1_1s, prog_bar=False)
		self.log("train_er_1s", er_1s, prog_bar=False)
		self.log("train_f1_framewise", f1_frame, prog_bar=False)
		self.log("train_er_framewise", er_frame, prog_bar=False)

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

		# val metrics
		val_scores = metrics.compute_scores(p > 0.5, t, FPS_OUT)
		val_f1_1s, val_er_1s = val_scores["f1_overall_1sec"], val_scores["er_overall_1sec"]
		val_f1_frame = metrics.f1_overall_framewise(p > 0.5, t)
		val_er_frame = metrics.er_overall_framewise(p > 0.5, t)

		# log to Lightning
		self.log("val_f1_1s", val_f1_1s, prog_bar=True)
		self.log("val_er_1s", val_er_1s, prog_bar=True)
		self.log("val_f1_framewise", val_f1_frame, prog_bar=False)
		self.log("val_er_framewise", val_er_frame, prog_bar=False)

		# get all train + val losses + metrics
		train_loss = self.trainer.callback_metrics.get("train_loss_epoch")
		val_loss = self.trainer.callback_metrics.get("val_loss_epoch")

		train_f1_1s = self.trainer.callback_metrics.get("train_f1_1s", 0.0)
		train_er_1s = self.trainer.callback_metrics.get("train_er_1s", 0.0)
		train_f1_frame = self.trainer.callback_metrics.get("train_f1_framewise", 0.0)
		train_er_frame = self.trainer.callback_metrics.get("train_er_framewise", 0.0)

		# initialize tracking
		if not hasattr(self, "track"):
			self.track = dict(
				tr_losses=[], val_losses=[],
				train_f1_1s=[], val_f1_1s=[],
				train_er_1s=[], val_er_1s=[],
				train_f1_frame=[], val_f1_frame=[],
				train_er_frame=[], val_er_frame=[],
				best={}
			)
			for k in ["train_loss", "val_loss", "f1_1s", "er_1s", "f1_frame", "er_frame"]:
				self.track["best"][k] = {"val": None, "epoch": -1}

		epoch = self.current_epoch

		# store
		if train_loss is not None:
			self.track["tr_losses"].append(train_loss.item())
		if val_loss is not None:
			self.track["val_losses"].append(val_loss.item())

		self.track["train_f1_1s"].append(train_f1_1s)
		self.track["train_er_1s"].append(train_er_1s)
		self.track["train_f1_frame"].append(train_f1_frame)
		self.track["train_er_frame"].append(train_er_frame)

		self.track["val_f1_1s"].append(val_f1_1s)
		self.track["val_er_1s"].append(val_er_1s)
		self.track["val_f1_frame"].append(val_f1_frame)
		self.track["val_er_frame"].append(val_er_frame)

		# best tracking
		def update_best(name, val, maximize):
			best = self.track["best"][name]["val"]
			if best is None or (maximize and val > best) or (not maximize and val < best):
				self.track["best"][name] = {"val": val, "epoch": epoch}

		if train_loss is not None:
			update_best("train_loss", train_loss.item(), maximize=False)
		if val_loss is not None:
			update_best("val_loss", val_loss.item(), maximize=False)
		update_best("f1_1s", val_f1_1s, maximize=True)
		update_best("er_1s", val_er_1s, maximize=False)
		update_best("f1_frame", val_f1_frame, maximize=True)
		update_best("er_frame", val_er_frame, maximize=False)

		def fmt(name, value):
			best = self.track["best"][name]
			best_val = best['val']
			best_epoch = best['epoch']
			if best_val is None:
				return f"{value:.3f} (best=NA@NA)"
			return f"{value:.3f} (best={best_val:.3f}@{best_epoch})"

		log = f"📊 Epoch {epoch:3d} | "
		log += f"train_loss={fmt('train_loss', train_loss.item() if train_loss else 0)} | "
		log += f"val_loss={fmt('val_loss', val_loss.item() if val_loss else 0)} | "
		log += f"F1@1s={fmt('f1_1s', val_f1_1s)} | ER@1s={fmt('er_1s', val_er_1s)} | "
		log += f"F1@frame={fmt('f1_frame', val_f1_frame)} | ER@frame={fmt('er_frame', val_er_frame)}"
		print(log)

		# plots
		def plot_metric(ax, train, val, name, best=None):
			ax.plot(train, label="train")
			ax.plot(val, label="val")
			if best is not None and best["val"] is not None:
				ax.axhline(best["val"], linestyle="--", color="gray", linewidth=0.8)
			ax.set_title(name)
			ax.set_xlabel("Epoch")
			ax.grid()
			ax.legend()

		plt.figure(figsize=(12, 6))

		ax1 = plt.subplot(2, 2, 1)
		plot_metric(
			ax1, self.track["tr_losses"], self.track["val_losses"], "Focal Loss", self.track["best"]["val_loss"]
			)

		ax2 = plt.subplot(2, 2, 2)
		plot_metric(ax2, self.track["train_f1_1s"], self.track["val_f1_1s"], "F1 (1s)", self.track["best"]["f1_1s"])

		ax3 = plt.subplot(2, 2, 3)
		plot_metric(ax3, self.track["train_er_1s"], self.track["val_er_1s"], "ER (1s)", self.track["best"]["er_1s"])

		ax4 = plt.subplot(2, 2, 4)
		plot_metric(
			ax4, self.track["train_f1_frame"], self.track["val_f1_frame"], "F1 (frame)", self.track["best"]["f1_frame"]
			)

		os.makedirs(self.art_dir, exist_ok=True)
		path = os.path.join(self.art_dir, f"metrics_fold{self.hparams.fold_id}.png")
		plt.tight_layout();
		plt.savefig(path);
		plt.close()
		print(f"📈 Saved metrics → {path}")

		# best tracking
		def update_best(name, val, maximize):
			best = self.track["best"][name]["val"]
			if best is None or (maximize and val > best) or (not maximize and val < best):
				self.track["best"][name] = {"val": val, "epoch": epoch}

		if train_loss is not None:
			update_best("train_loss", train_loss.item(), maximize=False)
		if val_loss is not None:
			update_best("val_loss", val_loss.item(), maximize=False)
		update_best("f1_1s", val_f1_1s, maximize=True)
		update_best("er_1s", val_er_1s, maximize=False)
		update_best("f1_frame", val_f1_frame, maximize=True)
		update_best("er_frame", val_er_frame, maximize=False)

		#print
		def fmt(name, value):
			best = self.track["best"][name]
			best_val = best["val"]
			best_epoch = best["epoch"]
			if best_val is None:
				return f"{value:.3f} (best=NA@NA)"
			return f"{value:.3f} (best={best_val:.3f}@{best_epoch})"

		log = f"📊 Epoch {epoch:3d} | "
		log += f"train_loss={fmt('train_loss', train_loss.item() if train_loss else 0)} | "
		log += f"val_loss={fmt('val_loss', val_loss.item() if val_loss else 0)} | "
		log += f"F1@1s={fmt('f1_1s', val_f1_1s)} | ER@1s={fmt('er_1s', val_er_1s)} | "
		log += f"F1@frame={fmt('f1_frame', val_f1_frame)} | ER@frame={fmt('er_frame', val_er_frame)}"
		print(log)

		# plots
		def plot_metric(ax, train, val, name, best=None):
			ax.plot(train, label="train")
			ax.plot(val, label="val")
			if best is not None and best["val"] is not None:
				ax.axhline(best["val"], linestyle="--", color="gray", linewidth=0.8)
			ax.set_title(name)
			ax.set_xlabel("Epoch")
			ax.grid()
			ax.legend()

		plt.figure(figsize=(12, 6))

		ax1 = plt.subplot(2, 2, 1)
		plot_metric(
			ax1, self.track["tr_losses"], self.track["val_losses"], "Focal Loss", self.track["best"]["val_loss"]
			)

		ax2 = plt.subplot(2, 2, 2)
		plot_metric(ax2, self.track["train_f1_1s"], self.track["val_f1_1s"], "F1 (1s)", self.track["best"]["f1_1s"])

		ax3 = plt.subplot(2, 2, 3)
		plot_metric(ax3, self.track["train_er_1s"], self.track["val_er_1s"], "ER (1s)", self.track["best"]["er_1s"])

		ax4 = plt.subplot(2, 2, 4)
		plot_metric(
			ax4, self.track["train_f1_frame"], self.track["val_f1_frame"], "F1 (frame)", self.track["best"]["f1_frame"]
			)

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

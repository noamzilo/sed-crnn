#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-fold training launcher (Lightning) + tee logging.

• Everything written to console is also saved to
  <fold_dir>/train.log.
• First mel window from the training loader is dumped to
  /tmp/train_window.npy  (handled in HitWindowDataset).
• First 8 logits of each validation epoch are stored in
  <fold_dir>/logits_eXXX.npz  (handled in CRNNLightning).
"""

import os, sys, io, datetime, torch, numpy as np, pytorch_lightning as pl, joblib
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from decorte_datamodule import DecorteDataModule
from crnn_lightning import CRNNLightning
torch.set_float32_matmul_precision('medium')

# ─────────────────────────────────────────────────────────────
#  Tee helper
# ─────────────────────────────────────────────────────────────
class _Tee(io.TextIOBase):
	def __init__(self, *streams):	self.streams = streams
	def write(self, data):
		for s in self.streams:
			s.write(data)
			s.flush()
	def flush(self):	pass

def _tee_to_file(path: str):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	f = open(path, "w")
	sys.stdout = _Tee(sys.__stdout__, f)
	sys.stderr = _Tee(sys.__stderr__, f)

# ─────────────────────────────────────────────────────────────
MAX_EPOCHS		= 200
EARLY_STOP		= 20
CACHE_DIR		= os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
ART_DIR_ROOT	= os.path.expanduser(f"~/src/plai_cv/sed-crnn/train_artifacts/{datetime.datetime.now():%Y%m%d_%H%M%S}")
DEVICE_TYPE		= 'gpu' if torch.cuda.is_available() else 'cpu'

os.makedirs(ART_DIR_ROOT, exist_ok=True)
print(f"ARTIFACTS → {ART_DIR_ROOT}")

error_rates = []
for fold_id in range(1, 2):
	art_dir = os.path.join(ART_DIR_ROOT, f"fold{fold_id}")
	_tee_to_file(os.path.join(art_dir, "train.log"))

	dm	   = DecorteDataModule(fold_id=fold_id, cache_dir=CACHE_DIR)
	model  = CRNNLightning(fold_id=fold_id, art_dir=art_dir)

	ckpt = ModelCheckpoint(
		dirpath		= art_dir,
		monitor		= 'val_er_1s',
		mode		= 'min',
		save_top_k	= -1,
		save_last	= True,
		filename	= "epoch{epoch:03d}-valer{val_er_1s:.3f}"
	)
	early = EarlyStopping(monitor='val_er_1s', mode='min', patience=EARLY_STOP)

	trainer = pl.Trainer(
		max_epochs			= MAX_EPOCHS,
		accelerator			= DEVICE_TYPE,
		devices				= 1,
		deterministic		= True,
		callbacks			= [ckpt, early],
		log_every_n_steps	= 50,
		gradient_clip_val	= 1.0
	)
	trainer.fit(model, datamodule=dm)

	if ckpt.best_model_score is not None:
		best_er = ckpt.best_model_score.item()
		print(f"✔️ Fold {fold_id} best ER={best_er:.3f}")
		error_rates.append(best_er)

print(f"\n🧮 Average ER across folds: {np.mean(error_rates):.3f}")

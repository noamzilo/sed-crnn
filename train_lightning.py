#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-fold training launcher (Lightning)
✓	switched EarlyStopping patience 40 → 20
✓	added gradient_clip_val = 1.0
✓	rest unchanged
✓	tabs only
"""

import os, datetime, torch, numpy as np, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from decorte_datamodule import DecorteDataModule
from crnn_lightning import CRNNLightning
torch.set_float32_matmul_precision('medium')

MAX_EPOCHS		= 200
EARLY_STOP		= 20					# ← tighter
CACHE_DIR		= os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
ART_DIR_ROOT	= os.path.expanduser(f"~/src/plai_cv/sed-crnn/train_artifacts/{datetime.datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(ART_DIR_ROOT, exist_ok=True)
DEVICE_TYPE		= 'gpu' if torch.cuda.is_available() else 'cpu'

print(f"ARTIFACTS → {ART_DIR_ROOT}")

error_rates = []
for fold_id in range(1, 2):
	art_dir = os.path.join(ART_DIR_ROOT, f"fold{fold_id}")
	dm = DecorteDataModule(fold_id=fold_id, cache_dir=CACHE_DIR)
	model = CRNNLightning(fold_id=fold_id, art_dir=art_dir)

	ckpt = ModelCheckpoint(
		dirpath=art_dir,
		monitor='val_er_1s',
		mode='min',
		save_top_k=-1,  # Save all
		save_last=True,  # Optional: saves `last.ckpt`
		save_weights_only=False,
		filename="epoch{epoch:03d}-valer{val_er_1s:.3f}"
	)
	early_stopping	 = EarlyStopping(monitor='val_er_1s', mode='min', patience=EARLY_STOP)

	trainer = pl.Trainer(
		max_epochs=MAX_EPOCHS,
						 accelerator=DEVICE_TYPE,
						 devices=1,
						 deterministic=True,
						 callbacks=[ckpt, early_stopping],
						 log_every_n_steps=50,
						 gradient_clip_val=1.0,
						 limit_train_batches=1.,
						 overfit_batches=0.,
						 )		# ← new

	trainer.fit(model, datamodule=dm)
	if ckpt.best_model_score is not None:
		best_er = ckpt.best_model_score.item()
		print(f"✔️ Fold {fold_id} best ER={best_er:.3f}")
		error_rates.append(best_er)
	else:
		print(f"⚠️ Fold {fold_id} has no best_model_score — skipping ER logging")

print(f"\n🧮 Average ER across folds: {np.mean(error_rates):.3f}")

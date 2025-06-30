#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference visualizer – now loads the *exact* StandardScaler that
the DataModule saved and normalizes MBE before feeding the model.

Tabs only.
"""

import os, sys, io, subprocess, tempfile, math, cv2, torch, numpy as np, pandas as pd
from pytorch_lightning import Trainer
from decorte_data_loader import load_decorte_dataset
from train_constants import *
import audio_features as af												# ← shared helpers

# ────────────────── tee logging (unchanged) ──────────────────
class _Tee(io.TextIOBase):
	def __init__(self,*s): self.s=s
	def write(self,d): [x.write(d) or x.flush() for x in self.s]
	def flush(self): pass
def _tee(path):
	os.makedirs(os.path.dirname(path),exist_ok=True)
	f=open(path,"w"); sys.stdout=_Tee(sys.__stdout__,f); sys.stderr=_Tee(sys.__stderr__,f)

# ────────────────── user paths ──────────────────
VIDEO_PATH = "/home/noams/src/plai_cv/data/decorte/rallies/20230528_VIGO_04.mp4"
CKPT_PATH  = "/home/noams/src/plai_cv/sed-crnn/train_artifacts/20250630_182940/fold1/epochepoch=015-valerval_er_1s=0.150.ckpt"
OUT_DIR    = "/home/noams/src/plai_cv/output/visualizations"; os.makedirs(OUT_DIR,exist_ok=True)

BASENAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
VIDEO_OUT= os.path.join(OUT_DIR,f"{BASENAME}_overlay.mp4")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA    = 0.5

_tee(os.path.join(os.path.dirname(CKPT_PATH),f"{BASENAME}_inference.log"))

from crnn_lightning import CRNNLightning

# ────────────────── helpers ────────────────── (blend, intervals)* kept identical to your last version
def _blend(f,c):
	ov = np.full_like(f,c); return cv2.addWeighted(f,1-ALPHA,ov,ALPHA,0)
def _sliding(mbe):
	w,s=[],[]
	for i in range(0,mbe.shape[0]-SEQ_LEN_IN+1,SEQ_LEN_OUT):
		w.append(mbe[i:i+SEQ_LEN_IN].T); s.append(i)
	return np.array(w),np.array(s)
def _spans(v,ffps,vfps):
	r,ins,s0=[],False,0
	for i,x in enumerate(v):
		if x and not ins: ins,s0=True,i
		elif not x and ins: ins=False; r.append((int(s0*vfps/ffps),int((i-1)*vfps/ffps)))
	if ins: r.append((int(s0*vfps/ffps),int((len(v)-1)*vfps/ffps)))
	return r

# ────────────────── main ──────────────────
def main():
	meta = load_decorte_dataset()[os.path.basename(VIDEO_PATH)]
	fold = meta["fold_id"] + 1									# folds are 1-indexed in cache
	scaler_path = os.path.join(os.path.dirname(CKPT_PATH), f"scaler_fold{fold}.joblib")
	if not os.path.exists(scaler_path):
		# fallback to cache copy
		scaler_path = os.path.join(CACHE_DIR, f"scaler_fold{fold}.joblib")
	scaler = af.load_scaler(scaler_path)
	print(f"Loaded scaler {scaler_path}")

	# audio → mbe → normalize
	y   = af._ffmpeg_audio(VIDEO_PATH)
	mbe = af._mbe(y)
	mbe = af.normalize(mbe, scaler)

	win_x, win_st = _sliding(mbe)
	if len(win_x)==0: raise RuntimeError("clip too short")
	np.save("/tmp/infer_window.npy", win_x[0])

	loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(torch.from_numpy(win_x).unsqueeze(1).float()),
		batch_size=64, shuffle=False, pin_memory=True)

	model = CRNNLightning.load_from_checkpoint(CKPT_PATH, fold_id=fold-1, art_dir="/tmp").to(DEVICE)
	trainer = Trainer(accelerator=DEVICE, devices=1, logger=False, enable_checkpointing=False)
	logits  = trainer.predict(model, loader)
	preds   = torch.cat(logits,0).sigmoid().squeeze(-1).cpu().numpy().reshape(-1)
	np.savez(os.path.join(os.path.dirname(CKPT_PATH),f"{BASENAME}_logits.npz"),logits=logits[0].cpu())

	accum = np.zeros(mbe.shape[0]);			# map back to frame rate
	for i,s in enumerate(win_st):
		ch=preds[i*SEQ_LEN_OUT:(i+1)*SEQ_LEN_OUT]; e=min(s+SEQ_LEN_OUT,len(accum))
		accum[s:e]=np.maximum(accum[s:e],ch[:e-s])

	# … (video overlay & stats identical to previous version) …

if __name__ == "__main__":
	main()

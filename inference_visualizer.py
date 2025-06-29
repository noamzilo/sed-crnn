#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_visualizer.py

Generate a same-fps MP4 with synced audio and an alpha-blended
hit-detection overlay (green = TP, yellow = FP, red = FN).

✓	in-memory feature extraction (log-Mel)
✓	sliding-window inference with CRNNLightning
✓	alpha overlay per frame
✓	remuxes original audio to keep sync
✓	tabs only
"""

import os, subprocess, tempfile, math, cv2, torch, numpy as np
from pytorch_lightning import Trainer
from decorte_data_loader import load_decorte_dataset
from decorte_datamodule import _ffmpeg_audio, _mbe
from crnn_lightning import CRNNLightning
from train_constants import *

# ─────────────────────────────────────────────────────────────
# Paths (edit if needed)
# ─────────────────────────────────────────────────────────────
VIDEO_PATH		= "/home/noams/src/plai_cv/data/decorte/rallies/20230528_VIGO_04.mp4"
CKPT_PATH		= "/home/noams/src/plai_cv/sed-crnn/train_artifacts/20250627_181038/fold3/epochepoch=022-valerval_er_1s=0.162.ckpt"
OUT_DIR			= "/home/noams/src/plai_cv/output/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)
BASENAME		= os.path.splitext(os.path.basename(VIDEO_PATH))[0]
VIDEO_OUT_PATH		= os.path.join(OUT_DIR, f"{BASENAME}_overlay.mp4")

ALPHA			= 0.5
DEVICE			= "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def blend(frame, color):
	overlay = np.full_like(frame, color, dtype=np.uint8)
	return cv2.addWeighted(frame, 1-ALPHA, overlay, ALPHA, 0)

def sliding_windows(mbe: np.ndarray, win: int = SEQ_LEN_IN, stride: int = SEQ_LEN_OUT):
	wins, starts = [], []
	for s in range(0, mbe.shape[0] - win + 1, stride):
		wins.append(mbe[s:s + win].T)		# (40, win)
		starts.append(s)
	return np.array(wins), np.array(starts)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
	# ── Load metadata & ground-truth hits ──────────────────
	vname = os.path.basename(VIDEO_PATH)
	meta_all = load_decorte_dataset()
	if vname not in meta_all:
		raise RuntimeError(f"{vname} not in Decorte metadata")
	meta   = meta_all[vname]
	hits   = meta["hits"]
	fold   = meta["fold_id"]

	# ── Decode audio & build MBE + label vector ────────────
	y   = _ffmpeg_audio(VIDEO_PATH, SAMPLE_RATE)
	mbe = _mbe(y, SAMPLE_RATE)				# (frames, 40)
	lbl = np.zeros((mbe.shape[0], 1), np.float32)
	for _, h in hits.iterrows():
		s = int(math.floor(h["start"] * SAMPLE_RATE / HOP_LENGTH))
		e = int(math.ceil (h["end"]   * SAMPLE_RATE / HOP_LENGTH))
		lbl[s:e, 0] = 1.0

	# ── Sliding-window tensor dataset ──────────────────────
	win_x, win_starts = sliding_windows(mbe)
	tensor_x = torch.from_numpy(win_x).unsqueeze(1).float()		# (N,1,40,L)
	loader   = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(tensor_x),
		batch_size = 64,
		shuffle    = False,
		pin_memory = True
	)

	# ── Run inference ──────────────────────────────────────
	model   = CRNNLightning.load_from_checkpoint(CKPT_PATH, fold_id=fold, art_dir="/tmp").to(DEVICE)
	trainer = Trainer(accelerator=DEVICE, devices=1, logger=False, enable_checkpointing=False)
	logits  = trainer.predict(model, loader)
	preds   = torch.cat(logits, 0).sigmoid().squeeze(-1).cpu().numpy().reshape(-1)

	# Map window predictions → per-MBE frame (SEQ_LEN_OUT rate)
	pred_full = np.zeros(mbe.shape[0], np.float32)
	for i, s in enumerate(win_starts):
		pred_full[s:s + SEQ_LEN_OUT] = preds[i * SEQ_LEN_OUT : (i + 1) * SEQ_LEN_OUT]

	# ── Prepare video I/O ──────────────────────────────────
	cap   = cv2.VideoCapture(VIDEO_PATH)
	fps   = cap.get(cv2.CAP_PROP_FPS)
	w, h  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	nf    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	rep          = int(math.ceil(nf / len(pred_full)))
	pred_aligned = np.repeat(pred_full.squeeze(), rep)[:nf]
	gt_aligned   = np.repeat(lbl.squeeze(),        rep)[:nf]

	tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
	writer  = cv2.VideoWriter(tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

	# ── Frame loop with overlay ────────────────────────────
	for i in range(nf):
		ret, frame = cap.read()
		if not ret: break
		p = pred_aligned[i] > 0.5
		t = gt_aligned[i]   > 0.5
		color = (0,255,0) if t and p else \
				(0,255,255) if p and not t else \
				(0,0,255) if t and not p else None
		if color is not None:
			frame = blend(frame, color)
		writer.write(frame)

	cap.release()
	writer.release()

	# ── Remux original audio to keep sync ──────────────────
	subprocess.check_call([
		"ffmpeg", "-y", "-loglevel", "error",
		"-i", tmp_vid,
		"-i", VIDEO_PATH,
		"-c:v", "copy",
		"-map", "0:v:0", "-map", "1:a:0",
		"-shortest", VIDEO_OUT_PATH
	])
	os.remove(tmp_vid)
	print(f"✅ Saved {VIDEO_OUT_PATH}")

if __name__ == "__main__":
	main()

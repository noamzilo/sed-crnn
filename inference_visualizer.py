#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize model predictions on a Decorte rally video
✓	Auto-looks up video metadata from Decorte dataset
✓	Extracts features via DecorteDataModule
✓	Infers using LightningModule + Trainer.predict()
✓	Overlays prediction quality per frame
✓	tabs only
"""

import os, cv2, torch, numpy as np
from pytorch_lightning import Trainer
from decorte_datamodule import DecorteDataModule, HitWindowDataset
from decorte_data_loader import load_decorte_dataset
from crnn_lightning import CRNNLightning
from train_constants import *
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
VIDEO_PATH		= "/home/noams/src/plai_cv/data/decorte/rallies/20230528_VIGO_04.mp4"
CKPT_PATH		= "/home/noams/src/plai_cv/sed-crnn/train_artifacts/20250627_181038/fold3/epochepoch=022-valerval_er_1s=0.162.ckpt"
OUTPUT_PATH		= f"/home/noams/src/plai_cv/output/visualizations/{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_overlay.mp4"
TMP_NPZ_PATH	= "/tmp/vigo04_tmp.npz"

assert os.path.isfile(VIDEO_PATH), VIDEO_PATH
assert os.path.isfile(CKPT_PATH), CKPT_PATH
os.makedirs(OUTPUT_PATH, exist_ok=True)

FRAME_SIZE		= (1280, 720)
ALPHA			= 0.5
DEVICE			= "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# Video I/O
# ─────────────────────────────────────────────────────────────
def load_video_frames(path):
	cap = cv2.VideoCapture(path)
	frames = []
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if FRAME_SIZE:
			frame = cv2.resize(frame, FRAME_SIZE)
		frames.append(frame)
	cap.release()
	return frames

def blend_color(frame, color):
	overlay = np.full_like(frame, color, dtype=np.uint8)
	return cv2.addWeighted(frame, 1 - ALPHA, overlay, ALPHA, 0)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
	vname = os.path.basename(VIDEO_PATH)
	ds = load_decorte_dataset()
	if vname not in ds:
		raise ValueError(f"✖ {vname} not found in Decorte dataset")

	meta = ds[vname]
	hits_df = meta["hits"]
	fold_id = meta["fold_id"]

	if len(hits_df) == 0:
		raise ValueError(f"✖ no hit annotations for {vname}")

	print(f"📼 Visualizing {vname} (fold {fold_id}, {len(hits_df)} hits)")

	# Step 1: generate .npz features
	DecorteDataModule.extract_video_to_npz(VIDEO_PATH, hits_df, TMP_NPZ_PATH)

	# Step 2: load raw video frames
	frames = load_video_frames(VIDEO_PATH)

	# Step 3: prepare dataset
	npz = np.load(TMP_NPZ_PATH)
	mbe, labels = npz["arr_0"], npz["arr_1"]
	ds = HitWindowDataset(mbe, labels, augment=False)
	dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

	# Step 4: load model + run predict
	model = CRNNLightning.load_from_checkpoint(CKPT_PATH, fold_id=fold_id, art_dir="/tmp").to(DEVICE)
	trainer = Trainer(accelerator=DEVICE, devices=1, logger=False, enable_checkpointing=False)
	preds = trainer.predict(model, dataloaders=dl)

	# Step 5: postprocess predictions and ground truth
	preds_concat = torch.cat(preds, dim=0).squeeze(-1).cpu().numpy()
	gt_concat = torch.cat([y.squeeze(-1) for _, y in dl], dim=0).cpu().numpy()

	n_frames = len(frames)
	rep = int(np.ceil(n_frames / len(preds_concat)))
	pred_aligned = np.repeat(preds_concat, rep)[:n_frames]
	gt_aligned = np.repeat(gt_concat, rep)[:n_frames]

	# Step 6: generate overlay
	out_frames = []
	for i, frame in enumerate(frames):
		p, t = pred_aligned[i] > 0.5, gt_aligned[i] > 0.5
		color = (
			(0, 255, 0) if t and p else
			(0, 255, 255) if not t and p else
			(0, 0, 255) if t and not p else
			None
		)
		out_frames.append(blend_color(frame, color) if color else frame)

	# Step 7: save video
	ffmpeg_write_video(OUTPUT_PATH, np.stack(out_frames), FPS_OUT, FRAME_SIZE, codec="libx264")
	print(f"✅ Done: {OUTPUT_PATH}")

if __name__ == "__main__":
	main()

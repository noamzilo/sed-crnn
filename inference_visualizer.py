#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a same-fps MP4 with alpha-blended hit overlay *and*
full tee-logging.

• Console output saved to  <ckpt_dir>/<video>_inference.log
• First inference mel window → /tmp/infer_window.npy
• First 8 raw logits → <ckpt_dir>/<video>_logits.npz
"""

import os, sys, io, subprocess, tempfile, math, cv2, torch, numpy as np, pandas as pd
from pytorch_lightning import Trainer
from decorte_data_loader import load_decorte_dataset
from decorte_datamodule import _ffmpeg_audio, _mbe
from crnn_lightning import CRNNLightning
from train_constants import *
from metrics import f1_overall_framewise, er_overall_framewise

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
#  USER-EDITABLE PATHS
# ─────────────────────────────────────────────────────────────
VIDEO_PATH = "/home/noams/src/plai_cv/data/decorte/rallies/20230528_VIGO_04.mp4"
CKPT_PATH  = "/home/noams/src/plai_cv/sed-crnn/train_artifacts/20250629_213009/fold1/epochepoch=023-valerval_er_1s=0.153.ckpt"
OUT_DIR	   = "/home/noams/src/plai_cv/output/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

BASENAME   = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
VIDEO_OUT  = os.path.join(OUT_DIR, f"{BASENAME}_overlay.mp4")
DEVICE	   = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA	   = 0.5

_tee_to_file(os.path.join(os.path.dirname(CKPT_PATH), f"{BASENAME}_inference.log"))

# ─────────────────────────────────────────────────────────────
#  Helper functions (unchanged core)
# ─────────────────────────────────────────────────────────────
def _blend(frame, color):
	overlay = np.full_like(frame, color, dtype=np.uint8)
	return cv2.addWeighted(frame, 1-ALPHA, overlay, ALPHA, 0)

def _sliding_windows(mbe: np.ndarray, win: int = SEQ_LEN_IN, stride: int = SEQ_LEN_OUT):
	wins, starts = [], []
	for s in range(0, mbe.shape[0] - win + 1, stride):
		wins.append(mbe[s:s+win].T)
		starts.append(s)
	return np.array(wins), np.array(starts)

def _find_intervals(hit_frames, feat_fps, vid_fps):
	spans, inside, s0 = [], False, 0
	for i,v in enumerate(hit_frames):
		if v and not inside:	inside, s0 = True, i
		elif not v and inside:
			inside = False
			spans.append((int(s0*vid_fps/feat_fps), int((i-1)*vid_fps/feat_fps)))
	if inside:
		spans.append((int(s0*vid_fps/feat_fps), int((len(hit_frames)-1)*vid_fps/feat_fps)))
	return spans

def _compute_stats(gt_int, pr_int):
	mx = max(max([e for _,e in gt_int], default=0), max([e for _,e in pr_int], default=0))
	gt = np.zeros(mx+1, np.uint8); pr = np.zeros_like(gt)
	for s,e in gt_int:	gt[s:e+1]=1
	for s,e in pr_int:	pr[s:e+1]=1
	tp = np.logical_and(gt==1, pr==1).sum()
	fp = np.logical_and(gt==0, pr==1).sum()
	fn = np.logical_and(gt==1, pr==0).sum()
	tn = np.logical_and(gt==0, pr==0).sum()
	return dict(
		num_gt=len(gt_int), num_pr=len(pr_int),
		gt_frames=int(gt.sum()), pr_frames=int(pr.sum()),
		tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
		f1=f1_overall_framewise(pr.reshape(-1,1), gt.reshape(-1,1)),
		er=er_overall_framewise(pr.reshape(-1,1), gt.reshape(-1,1)),
		precision=tp/(tp+fp) if tp+fp else 0.,
		recall=tp/(tp+fn) if tp+fn else 0.
	)

def _make_interval_df(gt_int, pr_int):
	rows=[]
	for s,e in gt_int: rows.append(dict(start=s,end=e,type="gt",color=(0,0,255)))
	for s,e in pr_int: rows.append(dict(start=s,end=e,type="pred",color=(0,255,255)))
	if not rows:	return pd.DataFrame()
	return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
def main():
	# ── metadata & GT hits ─────────────────────────────────
	meta = load_decorte_dataset()[os.path.basename(VIDEO_PATH)]
	hits, fold = meta["hits"], meta["fold_id"]

	# ── audio → MBE ───────────────────────────────────────
	print("🔊 Extracting audio features …")
	y   = _ffmpeg_audio(VIDEO_PATH, SAMPLE_RATE)
	mbe = _mbe(y, SAMPLE_RATE)

	# GT label vector
	lbl = np.zeros((mbe.shape[0], 1), np.float32)
	for _,h in hits.iterrows():
		s = int(math.floor(h["start"]*SAMPLE_RATE/HOP_LENGTH))
		e = int(math.ceil (h["end"]  *SAMPLE_RATE/HOP_LENGTH))
		lbl[s:e,0]=1.

	# sliding windows
	win_x, win_starts = [], []
	for s in range(0, mbe.shape[0]-SEQ_LEN_IN+1, SEQ_LEN_OUT):
		win   = mbe[s:s+SEQ_LEN_IN].T
		if not win_x:	np.save("/tmp/infer_window.npy", win)
		win_x.append(win);	win_starts.append(s)
	win_x = np.array(win_x)

	loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(torch.from_numpy(win_x).unsqueeze(1).float()),
		batch_size=64, shuffle=False, pin_memory=True
	)

	# inference
	model   = CRNNLightning.load_from_checkpoint(CKPT_PATH, fold_id=fold, art_dir="/tmp").to(DEVICE)
	trainer = Trainer(accelerator=DEVICE, devices=1, logger=False, enable_checkpointing=False)
	logits  = trainer.predict(model, loader)

	np.savez(os.path.join(os.path.dirname(CKPT_PATH), f"{BASENAME}_logits.npz"),
			 logits=torch.cat(logits[:1],0).cpu().numpy())

	preds = torch.cat(logits,0).sigmoid().squeeze(-1).cpu().numpy().reshape(-1)

	# map back to frame rate
	accum = np.zeros(mbe.shape[0], np.float32)
	for i,s in enumerate(win_starts):
		ch = preds[i*SEQ_LEN_OUT:(i+1)*SEQ_LEN_OUT]
		e = min(s+SEQ_LEN_OUT, mbe.shape[0])
		accum[s:e] = np.maximum(accum[s:e], ch[:e-s])

	# video info
	cap = cv2.VideoCapture(VIDEO_PATH)
	vid_fps = cap.get(cv2.CAP_PROP_FPS)
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()

	feat_fps = SAMPLE_RATE / HOP_LENGTH

	gt_int  = _find_intervals(lbl.squeeze()>0.5, feat_fps, vid_fps)
	pr_int  = _find_intervals(accum>0.5,        feat_fps, vid_fps)
	stats   = _compute_stats(gt_int, pr_int)
	df_int  = _make_interval_df(gt_int, pr_int)

	print("\n" + "="*60 + "\n📈 INTERVAL STATISTICS\n" + "="*60)
	for k,v in stats.items(): print(f"{k.replace('_',' ').title():>18}: {v}")
	print("="*60)

	df_int.to_csv(os.path.join(OUT_DIR, f"{BASENAME}_intervals.csv"), index=False)
	pd.DataFrame([stats]).to_csv(os.path.join(OUT_DIR, f"{BASENAME}_stats.csv"), index=False)

	# frame colour LUT
	clr={}
	for s,e in gt_int:	[clr.setdefault(f,(0,0,255)) for f in range(s,e+1)]
	for s,e in pr_int:
		for f in range(s,e+1):
			clr[f]=(0,255,0) if f in clr else (0,255,255)

	# write video
	tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
	wrt = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), vid_fps, (w,h))
	cap = cv2.VideoCapture(VIDEO_PATH)
	for i in range(nf):
		ret,frm = cap.read()
		if not ret:	break
		if i in clr:	frm = _blend(frm, clr[i])
		wrt.write(frm)
	cap.release();	wrt.release()

	subprocess.check_call([
		"ffmpeg","-y","-loglevel","error",
		"-i", tmp, "-i", VIDEO_PATH,
		"-c:v","copy","-map","0:v:0","-map","1:a:0",
		"-shortest", VIDEO_OUT
	])
	os.remove(tmp)
	print(f"✅ Saved {VIDEO_OUT}")

if __name__ == "__main__":
	main()

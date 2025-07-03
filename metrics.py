import numpy as np
from . import utils
from typing import List, Tuple

#####################
# Scoring functions
#
# Code blocks taken from Toni Heittola's repository: http://tut-arg.github.io/sed_eval/
#
# Implementation of the Metrics in the following paper:
# Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, 'Metrics for polyphonic sound event detection',
# Applied Sciences, 6(6):162, 2016
#####################

def _ensure_numeric(arr):
	if isinstance(arr, np.ndarray) and arr.dtype == bool:
		return arr.astype(np.uint8)
	if hasattr(arr, "dtype") and str(arr.dtype) == "bool":
		return arr.astype(np.uint8)
	return arr

def f1_overall_framewise(O, T):
	if len(O.shape) == 3:
		O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
	O = _ensure_numeric(O)
	T = _ensure_numeric(T)
	TP = ((2 * T - O) == 1).sum()
	Nref, Nsys = T.sum(), O.sum()
	prec = float(TP) / float(Nsys + utils.eps)
	recall = float(TP) / float(Nref + utils.eps)
	return 2 * prec * recall / (prec + recall + utils.eps)

def er_overall_framewise(O, T):
	if len(O.shape) == 3:
		O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
	O = _ensure_numeric(O)
	T = _ensure_numeric(T)
	FP = np.logical_and(T == 0, O == 1).sum(1)
	FN = np.logical_and(T == 1, O == 0).sum(1)

	S = np.minimum(FP, FN).sum()
	D = np.maximum(0, FN - FP).sum()
	I = np.maximum(0, FP - FN).sum()

	Nref = T.sum()
	return (S + D + I) / (Nref + 0.0)

def f1_overall_1sec(O, T, block_size):
	if len(O.shape) == 3:
		O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
	O, T = _ensure_numeric(O), _ensure_numeric(T)
	new_size = int(np.ceil(O.shape[0] / block_size))
	O_block = np.zeros((new_size, O.shape[1]))
	T_block = np.zeros((new_size, O.shape[1]))
	for i in range(new_size):
		O_block[i, :] = np.max(O[i * block_size : i * block_size + block_size], axis=0)
		T_block[i, :] = np.max(T[i * block_size : i * block_size + block_size], axis=0)
	return f1_overall_framewise(O_block, T_block)

def er_overall_1sec(O, T, block_size):
	if len(O.shape) == 3:
		O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
	O, T = _ensure_numeric(O), _ensure_numeric(T)
	new_size = int(O.shape[0] / block_size)
	O_block = np.zeros((new_size, O.shape[1]))
	T_block = np.zeros((new_size, O.shape[1]))
	for i in range(new_size):
		O_block[i, :] = np.max(O[i * block_size : i * block_size + block_size], axis=0)
		T_block[i, :] = np.max(T[i * block_size : i * block_size + block_size], axis=0)
	return er_overall_framewise(O_block, T_block)

def compute_scores(pred, y, frames_in_1_sec=50):
	return {
		'f1_overall_1sec': f1_overall_1sec(pred, y, frames_in_1_sec),
		'er_overall_1sec': er_overall_1sec(pred, y, frames_in_1_sec),
	}


def event_based_f1(pred_intervals: List[Tuple[int, int]], gt_intervals: List[Tuple[int, int]], fps: float, tol_sec: float = 0.25):
	"""
	Compute event-based F1 score given predicted and ground truth event intervals.
	Returns (f1, tp, fp, fn)
	"""
	pred_centers = [((s+e)/2)/fps for s, e in pred_intervals]
	gt_centers = [((s+e)/2)/fps for s, e in gt_intervals]
	matched_pred = set()
	matched_gt = set()
	for i, pc in enumerate(pred_centers):
		for j, gc in enumerate(gt_centers):
			if abs(pc - gc) <= tol_sec:
				matched_pred.add(i)
				matched_gt.add(j)
				break
	tp = len(matched_gt)
	fn = len(gt_intervals) - tp
	fp = len(pred_intervals) - len(matched_pred)
	denom = 2 * tp + fp + fn
	f1 = 2 * tp / denom if denom > 0 else 0.0
	return f1, tp, fp, fn

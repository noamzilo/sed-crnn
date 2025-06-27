from __future__ import print_function
import os
import numpy as np
import time
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import metrics
import utils

sys.setrecursionlimit(10000)


# ----------------------------- Data utilities -----------------------------

def load_data(_feat_folder, _mono, _fold=None):
	feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
	dmp = np.load(feat_file_fold)
	_X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'], dmp['arr_1'], dmp['arr_2'], dmp['arr_3']
	return _X_train, _Y_train, _X_test, _Y_test


def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
	# split into sequences
	_X = utils.split_in_seqs(_X, _seq_len)
	_Y = utils.split_in_seqs(_Y, _seq_len)

	_X_test = utils.split_in_seqs(_X_test, _seq_len)
	_Y_test = utils.split_in_seqs(_Y_test, _seq_len)

	_X = utils.split_multi_channels(_X, _nb_ch)
	_X_test = utils.split_multi_channels(_X_test, _nb_ch)
	return _X, _Y, _X_test, _Y_test


# ----------------------------- Model -----------------------------

class CRNN(nn.Module):
	def __init__(self, in_ch, in_freq, in_time, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb, dropout_rate, n_classes):
		super(CRNN, self).__init__()
		self.cnn = nn.ModuleList()
		self.bns = nn.ModuleList()
		self.pools = nn.ModuleList()

		ch = in_ch
		for p in _cnn_pool_size:
			self.cnn.append(nn.Conv2d(ch, _cnn_nb_filt, kernel_size=3, padding=1))
			self.bns.append(nn.BatchNorm2d(_cnn_nb_filt))
			self.pools.append(nn.MaxPool2d(kernel_size=(1, p)))
			ch = _cnn_nb_filt

		self.dropout = nn.Dropout(dropout_rate)

		# shape after convs/pools to compute flattened size
		with torch.no_grad():
			dummy = torch.zeros(1, in_ch, in_freq, in_time)
			for conv, bn, pool in zip(self.cnn, self.bns, self.pools):
				dummy = pool(nn.functional.relu(bn(conv(dummy))))
				dummy = self.dropout(dummy)
			dummy = dummy.permute(0, 2, 1, 3)  # (B, freq, ch, time)
			B, F, C, T = dummy.shape
			self.flat_features = C * T

		self.gru_stack = nn.ModuleList()
		gru_in = self.flat_features
		for h in _rnn_nb:
			self.gru_stack.append(nn.GRU(gru_in, h, batch_first=True, bidirectional=True))
			gru_in = h

		self.fc_stack = nn.ModuleList()
		for f in _fc_nb:
			self.fc_stack.append(nn.Linear(gru_in, f))
			gru_in = f

		self.classifier = nn.Linear(gru_in, n_classes)

	def forward(self, x):
		for conv, bn, pool in zip(self.cnn, self.bns, self.pools):
			x = pool(nn.functional.relu(bn(conv(x))))
			x = self.dropout(x)

		x = x.permute(0, 2, 1, 3)  # (B, freq, ch, time)
		B, F, C, T = x.shape
		x = x.reshape(B, F, C * T)  # (B, seq_len, feat)

		for gru in self.gru_stack:
			x, _ = gru(x)

		for fc in self.fc_stack:
			x = self.dropout(fc(x))

		x = torch.sigmoid(self.classifier(x))
		return x


# ----------------------------- Training helpers -----------------------------

def train_epoch(model, loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	for xb, yb in loader:
		xb = xb.to(device)
		yb = yb.to(device)
		optimizer.zero_grad()
		output = model(xb)
		loss = criterion(output, yb)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	preds = []
	labels = []
	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device)
			yb = yb.to(device)
			output = model(xb)
			loss = criterion(output, yb)
			running_loss += loss.item()
			preds.append(output.cpu().numpy())
			labels.append(yb.cpu().numpy())
	return (running_loss / len(loader),
			np.concatenate(preds, axis=0),
			np.concatenate(labels, axis=0))


# ----------------------------- Main script -----------------------------

if __name__ == '__main__':

	is_mono = True  # True: mono-channel input, False: binaural input
	feat_folder = '/scratch/asignal/sharath/DCASE2017/TUT-sound-events-2017-development/feat/'
	__fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))

	nb_ch = 1 if is_mono else 2
	batch_size = 128
	seq_len = 64
	nb_epoch = 500
	patience = int(0.25 * nb_epoch)

	sr = 44100
	nfft = 2048
	frames_1_sec = int(sr / (nfft / 2.0))

	print('\n\nUNIQUE ID: {}'.format(__fig_name))
	print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
		nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

	__models_dir = 'models/'
	utils.create_folder(__models_dir)

	# CRNN parameters
	cnn_nb_filt = 128
	cnn_pool_size = [2, 2, 2]
	rnn_nb = [32, 32]
	fc_nb = [32]
	dropout_rate = 0.5
	print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format(
		cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	criterion = nn.BCELoss()

	avg_er = list()
	avg_f1 = list()
	for fold in [1, 2, 3, 4]:
		print('\n\n----------------------------------------------')
		print('FOLD: {}'.format(fold))
		print('----------------------------------------------\n')

		X, Y, X_test, Y_test = load_data(feat_folder, is_mono, fold)
		X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)

		# Dataset shapes -> torch tensors
		X_train_t = torch.tensor(X, dtype=torch.float32)
		Y_train_t = torch.tensor(Y, dtype=torch.float32)
		X_test_t = torch.tensor(X_test, dtype=torch.float32)
		Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

		train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True,
								  drop_last=True)
		test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=batch_size, shuffle=False)

		in_ch, in_freq, in_time = X.shape[1], X.shape[2], X.shape[3]
		n_classes = Y.shape[-1]
		model = CRNN(in_ch, in_freq, in_time, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate, n_classes)
		model.to(device)
		optimizer = optim.Adam(model.parameters(), lr=1e-3)

		best_epoch, pat_cnt, best_er, f1_for_best_er = 0, 0, 99999, None
		tr_loss, val_loss = [0] * nb_epoch, [0] * nb_epoch
		f1_overall_1sec_list, er_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch
		posterior_thresh = 0.5

		for i in range(nb_epoch):
			print('Epoch : {} '.format(i), end='')
			tr_loss[i] = train_epoch(model, train_loader, criterion, optimizer, device)
			val_loss[i], pred_np, Y_np = evaluate(model, test_loader, criterion, device)

			pred_thresh = pred_np > posterior_thresh
			score_list = metrics.compute_scores(pred_thresh, Y_np, frames_in_1_sec=frames_1_sec)

			f1_overall_1sec_list[i] = score_list['f1_overall_1sec']
			er_overall_1sec_list[i] = score_list['er_overall_1sec']
			pat_cnt += 1

			# Confusion matrix
			test_pred_cnt = np.sum(pred_thresh, 2)
			Y_test_cnt = np.sum(Y_np, 2)
			conf_mat = confusion_matrix(Y_test_cnt.reshape(-1), test_pred_cnt.reshape(-1))
			conf_mat = conf_mat / (utils.eps + np.sum(conf_mat, 1)[:, None].astype('float'))

			if er_overall_1sec_list[i] < best_er:
				best_er = er_overall_1sec_list[i]
				f1_for_best_er = f1_overall_1sec_list[i]
				torch.save(model.state_dict(),
						   os.path.join(__models_dir, '{}_fold_{}_model.pt'.format(__fig_name, fold)))
				best_epoch = i
				pat_cnt = 0

			print('tr Er : {}, val Er : {}, F1_overall : {}, ER_overall : {} Best ER : {}, best_epoch: {}'.format(
				tr_loss[i], val_loss[i], f1_overall_1sec_list[i], er_overall_1sec_list[i], best_er, best_epoch))

			# Plot training curves
			if i == nb_epoch - 1 or pat_cnt > patience:
				plot.figure()
				plot.subplot(211)
				plot.plot(range(i + 1), tr_loss[:i + 1], label='train loss')
				plot.plot(range(i + 1), val_loss[:i + 1], label='val loss')
				plot.legend()
				plot.grid(True)

				plot.subplot(212)
				plot.plot(range(i + 1), f1_overall_1sec_list[:i + 1], label='f')
				plot.plot(range(i + 1), er_overall_1sec_list[:i + 1], label='er')
				plot.legend()
				plot.grid(True)

				plot.savefig(os.path.join(__models_dir, __fig_name + '_fold_{}.png'.format(fold)))
				plot.close()

			if pat_cnt > patience:
				break

		avg_er.append(best_er)
		avg_f1.append(f1_for_best_er)
		print('saved model for the best_epoch: {} with best_f1: {} f1_for_best_er: {}'.format(
			best_epoch, best_er, f1_for_best_er))

	print('\n\nMETRICS FOR ALL FOUR FOLDS: avg_er: {}, avg_f1: {}'.format(avg_er, avg_f1))
	print('MODEL AVERAGE OVER FOUR FOLDS: avg_er: {}, avg_f1: {}'.format(np.mean(avg_er), np.mean(avg_f1)))

import numpy as np
from math import floor, ceil
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from pyeer.eer_info import get_eer_stats
import sys

def get_similarity(X_n, Y_n):
    return np.dot(X_n, Y_n)

def get_scores(embeddings, labels):

	label_pairs_done = set()
	(true_scores, false_scores) = (np.array([]), np.array([]))

	for (idx_1, (embedding_1, label_1)) in enumerate(zip(embeddings, labels)):
		for (idx_2, (embedding_2, label_2)) in enumerate(zip(embeddings, labels)):
			if idx_2 > idx_1:
				if label_1 == label_2:
					score = get_similarity(embedding_1, embedding_2)
					true_scores = np.hstack((true_scores, score))
				else:
					if (label_1, label_2) not in label_pairs_done:
						score = get_similarity(embedding_1, embedding_2)
						false_scores = np.hstack((false_scores, score))
						label_pairs_done.add((label_1, label_2))

	print('Matching pairs:', true_scores.shape[0])
	print('Imposter pairs:', false_scores.shape[0])

	return true_scores, false_scores


def get_auth_metrics_mcc(total_embeddings, total_labels, phase):
    ts, fs = get_scores(total_embeddings, total_labels)
    mini = min(np.min(ts), np.min(fs))
    maxi = max(np.max(ts), np.max(fs))
    tsn = (ts - mini) / (maxi - mini)
    fsn = (fs - mini) / (maxi - mini)
    res = get_eer_stats(tsn, fsn)
    print("FRR @ FAR 0.1% : ", np.round(res.fmr1000 * 100, 6))
    print("FRR @ FAR 1.0% : ", np.round(res.fmr100 * 100, 6))
    print("FRR @ FAR 0.0% : ", np.round(res.fmr0 * 100, 6))
    print("Verification EER : ", np.round(res.eer * 100, 6))

    mcc_ts, mcc_fs = np.loadtxt('mcc_true_scores_normalized'), np.loadtxt('mcc_false_scores_normalized')
    mini = min(np.min(mcc_ts), np.min(mcc_fs))
    maxi = max(np.max(mcc_ts), np.max(mcc_fs))
    mcc_ts = (mcc_ts - mini) / (maxi - mini)
    mcc_fs = (mcc_fs - mini) / (maxi - mini)

    res = get_eer_stats(mcc_ts, mcc_fs)
    print("Only MCC FRR @ FAR 0.1% : ", np.round(res.fmr1000 * 100, 6))
    print("Only MCC FRR @ FAR 1.0% : ", np.round(res.fmr100 * 100, 6))
    print("Only MCC FRR @ FAR 0.0% : ", np.round(res.fmr0 * 100, 6))
    print("Only MCC Verification EER : ", np.round(res.eer * 100, 6))

    avg_ts, avg_fs = ((tsn + mcc_ts) / 2), ((fsn + mcc_fs) / 2)
    res = get_eer_stats(avg_ts, avg_fs)
    print("MCC Avg. FRR @ FAR 0.1% : ", np.round(res.fmr1000 * 100, 6))
    print("MCC Avg. FRR @ FAR 1.0% : ", np.round(res.fmr100 * 100, 6))
    print("MCC Avg. FRR @ FAR 0.0% : ", np.round(res.fmr0 * 100, 6))
    print("MCC Avg. Verification EER : ", np.round(res.eer * 100, 6))

    wavg_ts, wavg_fs = (0.2*tsn + 0.8*mcc_ts), (0.2*fsn + 0.8*mcc_fs)
    res = get_eer_stats(wavg_ts, wavg_fs)
    print("MCC WAvg. FRR @ FAR 0.1% : ", np.round(res.fmr1000 * 100, 6))
    print("MCC WAvg. FRR @ FAR 1.0% : ", np.round(res.fmr100 * 100, 6))
    print("MCC WAvg. FRR @ FAR 0.0% : ", np.round(res.fmr0 * 100, 6))
    print("MCC WAvg. Verification EER : ", np.round(res.eer * 100, 6))

    wavg_ts, wavg_fs = (0.1*tsn + 0.9*mcc_ts), (0.1*fsn + 0.9*mcc_fs)
    res = get_eer_stats(wavg_ts, wavg_fs)
    print("MCC WAvg2. FRR @ FAR 0.1% : ", np.round(res.fmr1000 * 100, 6))
    print("MCC WAvg2. FRR @ FAR 1.0% : ", np.round(res.fmr100 * 100, 6))
    print("MCC WAvg2. FRR @ FAR 0.0% : ", np.round(res.fmr0 * 100, 6))
    print("MCC WAvg2. Verification EER : ", np.round(res.eer * 100, 6))

    sys.stdout.flush()

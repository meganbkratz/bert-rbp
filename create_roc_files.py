"""Use non-training data to calculate AUROC curves. Save a file that includes the FP, TP, FN, TN, n_pos and n_neg for each threshold."""
import os, sys, argparse
import numpy as np
from analyze_RNA import load_probabilities

parser = argparse.ArgumentParser()

parser.add_argument("RBP", type=str)
parser.add_argument("--prediction_path", type=str, help="The path to the prediction file. (must have 'positives' and 'negatives')")
parser.add_argument("--save_dir", type=str, help="Where to save the generated text file (will be named '%RBP_eval_performance.csv')")

args = parser.parse_args()

#p_eval = os.path.join(dataset_path, RBP, 'nontraining_sample_finetune', 'model_mbk_nontraining_model_output.pk')
#p_train = os.path.join(dataset_path, RBP, 'training_sample_finetune', 'train_%s_bertrbp_output.pk'%RBP)


probs=load_probabilities(args.prediction_path)

pos = probs['positives']
neg = probs['negatives']

data = np.zeros(100, dtype=[
	('threshold', float),
	('true_positives', int),
	('false_positives', int),
	('true_negatives',int),
	('false_negatives',int),
	('pos_hist', int),
	('neg_hist', int)
	])

pos_hist, bins = np.histogram(pos, bins=100, range=(0,1))
neg_hist, bins = np.histogram(neg, bins=100, range=(0,1))

for i,t in enumerate(np.arange(0,1,0.01)):
	data[i]['threshold'] = t
	data[i]['true_positives'] = len(pos[pos >= t])
	data[i]['false_positives'] = len(neg[neg >= t])
	data[i]['true_negatives'] = len(neg[neg < t])
	data[i]['false_negatives'] = len(pos[pos < t])
	data[i]['pos_hist'] = pos_hist[i]
	data[i]['neg_hist'] = neg_hist[i]

header = str(data.dtype.names).strip('(').strip(')').strip()
filename = os.path.join(args.save_dir, '%s_eval_performance.csv'%(args.RBP))
np.savetxt(filename, data, delimiter=',', header=header, fmt=['%.2f']+['%i']*6, comments='')

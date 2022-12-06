"""Use non-training data to calculate AUROC curves. Save a file that includes the FP, TP, FN, TN, n_pos and n_neg for each threshold."""
import os, sys
import numpy as np
from analyze_RNA import load_probabilities

dataset_path = '/proj/magnuslb/users/mkratz/bert-rbp/datasets'

RBP = sys.argv[1]

p_eval = os.path.join(dataset_path, RBP, 'nontraining_sample_finetune', 'dev_%s_bertrbp_output.pk'%RBP)
p_train = os.path.join(dataset_path, RBP, 'training_sample_finetune', 'train_%s_bertrbp_output.pk'%RBP)


for k, v in {'eval': p_eval, 'train':p_train}.items():
	probs=load_probabilities(v)

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
	filename = '/proj/magnuslb/users/mkratz/data/RBP_performance/%s_%s_performance.csv'%(RBP, k)
	np.savetxt(filename, data, delimiter=',', header=header, fmt=['%.2f']+['%i']*6, comments='')

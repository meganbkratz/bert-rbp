import os, sys, itertools
import numpy as np
from analyze_RNA import load_probabilities
from sklearn.metrics import roc_auc_score

dataset_path = '/proj/magnuslb/users/mkratz/bert-rbp/datasets'

metrics = ['auc_roc']
save_file = '/proj/magnuslb/users/mkratz/bert-rbp/results.csv'
with open(save_file, 'w') as f:
	s = 'rbp,' + ','.join('_'.join(x) for x in itertools.product(['train', 'eval'], metrics)) + '\n'
	f.write(s)

def calculate_metrics(data):
	pos_probs = data['positives']
	neg_probs = data['negatives']

	pos_labels = [1]*len(pos_probs)
	neg_labels = [0]*len(neg_probs)

	pos_preds = pos_probs >= 0.5
	neg_preds = neg_probs >= 0.5

	probs = np.concatenate((pos_probs, neg_probs))
	labels = np.concatenate((pos_labels, neg_labels))
	preds = np.concatenate((pos_preds, neg_preds))

	## keys in results should match names in 'metrics' list above
	results = {	
		'auc_roc': roc_auc_score(labels, probs)
	}

	return results

for RBP in sorted(os.listdir(dataset_path)):
	p_eval = os.path.join(dataset_path, RBP, 'nontraining_sample_finetune', 'dev_%s_bertrbp_output.pk'%RBP)
	p_train = os.path.join(dataset_path, RBP, 'training_sample_finetune', 'train_%s_bertrbp_output.pk'%RBP)

	eval_data = load_probabilities(p_eval) 
	train_data = load_probabilities(p_train)

	eval_metrics = calculate_metrics(eval_data)
	train_metrics = calculate_metrics(train_data)

	with open(save_file, 'a') as f:
		f.write(RBP+',')
		for dataset in [train_metrics, eval_metrics]:
			for m in metrics:
				f.write('%0.3f'%dataset[m]+',')
		f.write('\n')



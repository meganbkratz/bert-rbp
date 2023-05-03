import os
import numpy as np
import pyqtgraph as pg
from util import calculate_precision_recall_fscore

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


# ### the goal here is to find thresholds where the rate of identifying positives (recall) is high and the False Discovery Rate (1-precision) is low.

# ## accuracy 
# 	- the fraction of predictions the model got right
# 	= N correct predictions / N samples  (fine when dataset is balanced, not informative when dataset is unbalanced)
# ## precision - !!!!
# 	- the proportion of positive predictions that is correct (opposite of false discovery rate, i think)
# 	- = TP / (TP + FP)
# ## recall (same as True Positive Rate)
# 	- the fraction of actual positives that was identified correctly
# 	- = TP / (TP + FN)
# ## F1
#	- = 2*TP / (2*TP + FP + FN)
#	- = (1+ b**2) * TP / ((1+b**2)*TP + b**2*FN + FP) - b=2 favors recall over precision, b=0.5 favors precision over recall
# 	- a combination metric of recall and precision - can be weighted to favor recall or precision more
# 	- doesnt look at TN - but wikipedia suggests mcc does


##So, for each threshold we want to calculate precision, recall, F1, and F0.5

def load_performance_data(filename):
	stats = np.genfromtxt(filename, delimiter=',', names=True, dtype=[float]+[int]*6)
	return stats

def summary(metrics, min_precision=0.95, min_recall=0.2):
	F_max = np.argwhere(metrics['F0.2_score'] == metrics['F0.2_score'].max())
	valids = list(metrics['threshold'][np.argwhere(np.logical_and(metrics['precision'] >= min_precision, metrics['recall'] >= min_recall)).flatten()])
	return {
		'valid_thresholds' : valids,
		'min_precision' : min_precision,
		'min_recall': min_recall,
		'F0.2_max_threshold': metrics['threshold'][F_max][0][0],
		'F0.2_max': metrics['F0.2_score'].max(),
		}

def write_summary_file(metrics, stringencies, summary_file):
	string1 = ' \t'
	string2 = 'RBP \t'
	for i, s in enumerate(stringencies):
		string1 += '\t'.join(['stringency_%i'%i,'min_precision:%0.3f'%s[0], 'min_recall:%0.3f'%s[1],'\t', ''])
		string2 += '\t'.join(['F0.2_max_threshold', 'F0.2_max', 'valid_thresholds', 'N_thresholds_found', '\t'])

	with open(summary_file, 'w') as f:
		f.write(string1 + '\n')
		f.write(string2 + '\n')
		#f.write('\t'.join([' '].extend([['stringency_%i'%i,'min_precision:%0.3f'%s[0], 'min_recall:%0.3f'%s[1],''] for i,s in enumerate(stringencies)])))
		#f.write('\t'.join(['RBP', ['F0.2_max_threshold', 'F0.2_max', 'valid_thresholds', ' ']*len(stringencies)]) + '\n')

	for rbp in sorted(metrics.keys()):
		summaries = [summary(metrics[rbp]['metrics'], min_precision=s[0], min_recall=s[1]) for s in stringencies]
		with open(summary_file, 'a') as f:
			f.write(rbp+'\t')
			for s in summaries:
				f.write('\t'.join(['%0.2f'%s['F0.2_max_threshold'], '%0.3f'%s['F0.2_max'], '%s'%str(s['valid_thresholds']), '%i'%len(s['valid_thresholds']), '\t']))
			f.write('\n')

def plot_metric(metrics, key=None, rbp=None):
	p = pg.plot(title=rbp, labels={'left':key, 'bottom':'threshold'})
	if rbp is None:
		rbps = list(metrics.keys())
	else:
		rbps = [rbp]

	passes = []
	fails = []
	for r in rbps:
		s = summary(metrics[r]['metrics'], min_precision=0.9, min_recall=0.1)
		if len(s['valid_thresholds']) > 0:
			pen = pg.mkPen((0,0,0,255))
			passes.append(metrics[r]['metrics'][key])
			#pen='b'
		else:
			pen = pg.mkPen('y')
			fails.append(metrics[r]['metrics'][key])
			#pen = 'r'
		p.plot(metrics[r]['metrics']['threshold'], metrics[r]['metrics'][key], pen=pen)

	a_passes = np.array(passes)
	a_fails = np.array(fails)

	pass_avg = np.nanmean(a_passes, axis=0)
	fail_avg = np.nanmean(a_fails, axis=0)

	p.plot(np.arange(0, 1, 0.01), fail_avg, pen=pg.mkPen('r', width=2))
	p.plot(np.arange(0, 1, 0.01), pass_avg, pen=pg.mkPen('b', width=3))

	return p

def write_RBP_list(metrics, filename):
	import yaml

	valids = []
	invalids = []

	for rbp in metrics.keys():
		s = summary(metrics[rbp]['metrics'], min_precision=0.9, min_recall=0.1)
		if len(s['valid_thresholds']) > 0:
			valids.append(rbp)
		else:
			invalids.append(rbp)

	with open(filename, 'w') as f:
		yaml.dump({'valid_RBPs':valids, 'invalid_RBPs':invalids, 'criteria':{'min_precision':0.9, 'min_recall':0.1}}, f)



if __name__ == '__main__':

	pg.dbg()

	filenames = ['/home/megan/work/smithlab/data/RBP_performance/3mer_kyamada/KHSRP_eval_performance.csv']
	performance_dir = '/home/megan/work/smithlab/data/RBP_performance/3mer_kyamada'

	metrics = {}
	#for f in filenames:
	for f in os.listdir(performance_dir):
		if f.split('_')[-1] != 'performance.csv':
			continue
		stats = load_performance_data(os.path.join(performance_dir, f))
		#stats = load_performance_data(f)
		rbp = f.split('_')[0]
		m = calculate_precision_recall_fscore(stats)
		metrics[rbp] = {'metrics': m}

	stringencies = [(0.95, 0.2), (0.95, 0.1), (0.9, 0.2), (0.9, 0.1), (0.85, 0.2), (0.85, .1), (0.8, 0.2), (0.8, 0.1)]
	summary_file = os.path.join(performance_dir, 'dynamic_thresholds_2.tsv')
	#write_summary_file(metrics, stringencies, summary_file)

	prec_plot = plot_metric(metrics, key='precision', rbp=None)
	recall_plot = plot_metric(metrics, key='recall', rbp=None)

	write_RBP_list(metrics, '/home/megan/work/smithlab/bert_manuscript/figure_1/valid_rbps.yaml')

	







	# p = pg.plot()
	# p.addLegend()

	# p.plot(x=metrics['threshold'], y=metrics['precision'], pen='b', label='precision')
	# p.plot(x=metrics['threshold'], y=metrics['recall'], pen='c', label='recall')
	# p.plot(x=metrics['threshold'], y=metrics['F1_score'], pen='g', label='F1_score')
	# p.plot(x=metrics['threshold'], y=metrics['F0.5_score'], pen='y', label='F0.5_score')
	# p.plot(x=metrics['threshold'], y=metrics['F0.2_score'], pen='r', label='F0.1_score')

	# p.show()



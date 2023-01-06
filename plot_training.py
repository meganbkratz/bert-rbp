import os, argparse
import pyqtgraph as pg
import numpy as np
import torch


def parse_metrics(metrics):
	collected = {}

	## convert from list of dicts to dict of lists
	keys = list(metrics[0].keys())
	for k in keys:
		if k == 'metrics':
			for k2 in metrics[0][k].keys():
				collected[k2] = []
		collected[k]=[]
	for i, m in enumerate(metrics):
		for k in keys:
			if k == 'metrics':
				for k2 in metrics[0]['metrics'].keys():
					collected[k2].append(m['metrics'][k2])
			collected[k].append(m[k])

	## create x-axis of epoch * step
	n_epochs = len(set(collected['epoch']))
	n_steps_per_epoch = collected['step'][-1] + 1
	x = [(collected['epoch'][i]*n_steps_per_epoch)+(collected['step'][i]+1) for i in range(len(collected['epoch']))]
	collected['total_steps'] = x

	## find starts of epochs in total_steps
	indices = np.argwhere(np.diff(collected['epoch']) == 1).flatten()
	epoch_starts = [n_steps_per_epoch * i for i in range(len(indices)+1)]
	collected['epoch_starts'] = epoch_starts

	return collected

def plot_metrics(metrics, RBP):
	training = parse_metrics(metrics['training'])
	val = parse_metrics(metrics['validation'])

	pl = pg.GraphicsLayout()
	plots = {}
	keys = ['loss', 'auroc', 'accuracy']
	for k in keys:
		plots[k]=pl.addPlot(title="%s %s during training"%(RBP, k), labels={'bottom':'training_batch', 'left':k})
		plots[k].addLegend()
		plots[k].plot(x=training['total_steps'], y=training[k], pen='b', name='training')
		plots[k].plot(x=val['total_steps'], y=val[k], pen='g', name='validation')
		plots[k].addItem(pg.VTickGroup(xvals=training['epoch_starts'], yrange=[0,0.2], pen='r'),name='epoch starts')
		#plots[k].setYRange(0.5, 0.8, padding=0.05)

	return pl, plots, training, val




if __name__ == '__main__':
	app = pg.mkQApp()

	parser = argparse.ArgumentParser()
	parser.add_argument("RBP", type=str, help="The name of the RNA binding protien (RBP) to train.")
	parser.add_argument("--debug", action="store_true", help="If True, opens the pyqtgraph debug console.")
	parser.add_argument("--metric", type=str, default='auroc', help="The metric to plot (default is auroc)")
	args = parser.parse_args()

	valid_metrics = ['loss', 'auroc', 'accuracy']
	if args.metric not in valid_metrics:
		raise Exception("not sure how to plot metric %s. options are %s"%(args.metric, valid_metrics))

	if args.debug:
		pg.dbg()

	mw = pg.GraphicsView()
	mw.setWindowTitle("%s %s" % (args.RBP, args.metric))
	plot_layout = pg.GraphicsLayout()
	mw.setCentralItem(plot_layout)

	learning_rates = ['0.00001', '0.00002', '0.00005', '0.0001', '0.0002']
	dropouts = ['0.01', '0.1', '0.2', '0.4']

	for lr in learning_rates:
		for d in dropouts:
			metrics_file = 'longleaf/bert-rbp/datasets/%s/grid_search/5_epoch/%s/training_metrics.bin' % (args.RBP, 'LR%s_D%s_model'%(lr, d))
			if not os.path.exists(metrics_file):
				print('couldnt find %s'%metrics_file)
				continue
			metrics = torch.load(metrics_file)

			pl, plots, training, val = plot_metrics(metrics, args.RBP)
			plots[args.metric].setTitle("lr: %s, dropout:%s, end roc:%.3f"%(lr, d, val['auroc'][-1]))
			plot_layout.addItem(plots[args.metric])
		plot_layout.nextRow()
	mw.show()

## notes:
#   AARS - 0.74
#	TIAL1 - 0.84
# 	ZC3H11A - 0.74
#	RBFOX2 - 0.77
#	HNRNPK - 0.87
# 	KHSRP - 0.86
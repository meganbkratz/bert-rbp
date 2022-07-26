import os
import pyqtgraph as pg
import pyqtgraph.exporters
from analyze_RNA import *

pg.dbg()

#RBPs = ['GRWD1', 'HNRNPU', 'RBFOX2', 'HNRNPA1', 'KHSRP', 'ZC3H11A', 'TIAL1']
RBPs = ['TIAL1', 'KHSRP']
data_dir = "/home/megan/work/lnc_rna/data/sequences/OIP5-AS1/human"
file_name_format="OIP5-AS1_genomic_sequence_%s_probabilities.pk"
#RBPs = ['TIAL1']

# probs = {}
# for r in RBPs:
# 	p = os.path.join('nontraining_data_probabilities', 'dev_%s_probabilities.pk'%r)
# 	probs[r]=load_probabilities(p)

# plots={}
# for r in RBPs:
# 	plots[r]=plot(probs[r], r)

sequence_probs = {}
for r in RBPs:
	p = os.path.join(data_dir, file_name_format%r)
	sequence_probs[r] = load_probabilities(p)

sequence_plots = {}
for r in RBPs:
	sequence_plots[r] = plot(sequence_probs[r], r, 'OIP5-AS1')


### resize and export
for k in RBPs:
	p = sequence_plots[k]
	p.resize(1500,800)
	p.setRange(yRange=(0,1.2))
	# exporter = pg.exporters.ImageExporter(p.plotItem)
	# exporter.parameters()['background'] = pg.mkColor('w')
	# export_path=os.path.join(data_dir, file_name_format.rsplit('.', 1)[0]%k)
	# exporter.export(export_path+'_fullPlot.png')
	# svg_exporter = pg.exporters.SVGExporter(p.plotItem)
	# svg_exporter.export(export_path+'_fullPlot.svg')
	# svg_exporter.parameters()['background']=pg.mkColor('w')
	# csv_exporter = pg.exporters.CSVExporter(p.plotItem)
	# csv_exporter.export(export_path+'.csv')

	# # zoom
	# p.setRange(xRange=(30,1350))
	# exporter.export(export_path +'_zoomedPlot.png')

	# #p=plots[k]
	# #exporter = pg.exporters.ImageExporter(p.plotItem)
	# #exporter.export('/home/megan/work/lnc_rna/data/RBP_analysis/%s_histogram.png'%k)


def create_roc_plot(probs, label=""):
	pos = probs['positives']
	neg = probs['negatives']
	p = pg.plot(title=label)
	p.addLegend(offset=(-10,-10))
	p.setLabel('bottom', "False positive rate")
	p.setLabel('left', "True positive rate")
	p.plot(x=[0,1], y=[0,1]) ## plot randomness line

	best_threshold=None
	minimum_distance=1
	for i,t in enumerate(np.arange(0,1,0.01)):
		tpr = len(pos[pos >= t])/len(pos)
		fpr = len(neg[neg >= t])/len(neg)
		name=None
		if i % 10 == 0:
			name = np.round(t, decimals=2)

		p.plot(x=[fpr], y=[tpr], pen=None, symbolPen=None, symbolBrush=pg.intColor(i, hues=120), name=name)
		distance = (pg.Point(0,1) - pg.Point(fpr,tpr)).length()
		if distance < minimum_distance:
			minimum_distance = distance
			best_threshold = t
			best_points = (fpr, tpr)

	p.setTitle("%s roc curve (best threshold = %0.2f)" %(label, best_threshold))
	p.plot([best_points[0]], [best_points[1]], pen=None, symbolPen=pg.mkPen('w', width=3), symbolBrush=None, name='best threshold')

	return p


# roc_plots={}
# for r in RBPs:
# 	roc_plots[r] = create_roc_plot(probs[r], r)
# 	exporter = pg.exporters.ImageExporter(roc_plots[r].plotItem)
# 	exporter.export('/home/megan/work/lnc_rna/data/RBP_analysis/%s_roc_plot.png'%r)

	
# sequences={}
# for rbp in RBPs:
# 	sequences[rbp]= {'pos':None, 'neg':None}
# 	sequences[rbp]['pos'] = load_fasta_sequences('/home/megan/work/lnc_rna/code/bert-rbp/RBP_training_data/%s.subset.positive.fa'%rbp)
# 	sequences[rbp]['neg'] = load_fasta_sequences('/home/megan/work/lnc_rna/code/bert-rbp/RBP_training_data/%s.subset.neg.fa'%rbp)


# probs = {}
# plots = {}
# for rbp in RBPs:
# 	probs[rbp]={}
# 	probs[rbp]['pos'] = load_probabilities('/home/megan/work/lnc_rna/data/RBP_analysis/%s.subset.positive_probabilities.pk'%rbp)
# 	probs[rbp]['neg'] = load_probabilities('/home/megan/work/lnc_rna/data/RBP_analysis/%s.subset.negative_probabilities.pk'%rbp)

# 	plots[rbp+'_pos'] = plot_test_probabilities(probs[rbp]['pos'], rbp+" positive")
# 	plots[rbp+'_neg'] = plot_test_probabilities(probs[rbp]['neg'], rbp+" negative")

# for k, p in plots.items():
# 	p.resize(1200,600)
# 	p.setRange(yRange=(0,1.2))

# 	exporter = pg.exporters.ImageExporter(p.plotItem)
# 	exporter.export('/home/megan/work/lnc_rna/data/RBP_analysis/%s_example.png'%k)


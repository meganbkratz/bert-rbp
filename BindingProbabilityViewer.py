import os, argparse, re
import pyqtgraph as pg 
import numpy as np
from plotting_helpers import load_probabilities
import scipy.signal
import config
from region_analysis import find_binding_regions
from Bio import SeqIO
import util
## set plots to have white backgrounds
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

model_types = ['3mer_kyamada', '6mer_mbk']

class FileLoader(pg.QtWidgets.QWidget):
	def __init__(self, parent=None, baseDir=None):
		pg.QtWidgets.QWidget.__init__(self, parent)

		self.layout = pg.QtWidgets.QVBoxLayout()
		self.setLayout(self.layout)

		self.baseDirBtn = pg.QtWidgets.QPushButton("Set base directory...")
		self.layout.addWidget(self.baseDirBtn)
		self.fileTree = pg.QtWidgets.QTreeWidget()
		self.fileTree.setAcceptDrops(False)
		self.fileTree.setDragEnabled(False)
		self.fileTree.setEditTriggers(pg.QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
		self.layout.addWidget(self.fileTree)

		self.baseDirBtn.clicked.connect(self.baseDirBtnClicked)

		self.baseDir = None
		if baseDir is not None:
			if not os.path.isdir(baseDir) and os.path.isfile(baseDir):
				baseDir = os.path.dirname(baseDir)
			if os.path.basename(baseDir) == "" and baseDir[-1] == "/":
				baseDir = baseDir[:-1]
			self.setBaseDir(baseDir)

	def baseDirBtnClicked(self):
		baseDir = self.baseDir if self.baseDir is not None else ""
		newBaseDir = pg.FileDialog.getExistingDirectory(self, caption="Select base directory...", directory=baseDir)
		if os.path.exists(newBaseDir):
			self.setBaseDir(newBaseDir)

	def setBaseDir(self, baseDir):
		self.baseDir = baseDir
		self.fileTree.clear()

		item = pg.QtWidgets.QTreeWidgetItem(self.fileTree, [os.path.basename(baseDir)])
		item.setExpanded(True)
		self.fileTree.addTopLevelItem(item)

		for path in [os.path.join(baseDir, x) for x in os.listdir(baseDir)]:
			self.addFileItem(path, item)

	def addFileItem(self, path, root):
		if os.path.isfile(path):
			item = pg.QtWidgets.QTreeWidgetItem(root, [os.path.basename(path)])
			item.path = os.path.abspath(path)
			if path[-3:] != '.pk':
				item.setDisabled(True)
			root.addChild(item)
		elif os.path.isdir(path):
			item = pg.QtWidgets.QTreeWidgetItem(root, [os.path.basename(path)])
			root.addChild(item)
			for f in sorted(os.listdir(path)):
				self.addFileItem(os.path.join(path,f), item)
		elif os.path.islink(path):
			item = pg.QtWidgets.QTreeWidgetItem(root, [os.path.basename(path)])
			item.path = os.path.abspath(path)
			if not os.path.exists(path):
				item.setDisabled(True)
			root.addChild(item)

		else:
			raise Exception("Why are we here?")


class NonscientificAxisItem(pg.AxisItem):
	###Need an axis that doesn't convert big numbers to SI notation so that we can display DNA addresses on plots

	def tickStrings(self, values, scale, spacing):
		if self.logMode:
			return self.logTickStrings(values, scale, spacing)

		places = max(0, np.ceil(-np.log10(spacing*scale)))
		strings = []
		for v in values:
			vs = v * scale
			vstr = ("%%0.%df" % places) % vs
			strings.append(vstr)
		return strings

def besselFilter(data, cutoff, order=1, dt=None, btype='low', bidir=True):
	"""return data passed through bessel filter"""

	if dt is None:
		dt = 1.0

	b,a = scipy.signal.bessel(order, cutoff * dt, btype=btype) 

	if bidir:
		d1 = scipy.signal.lfilter(b, a, scipy.signal.lfilter(b, a, data)[::-1])[::-1]
	else:
		d1 = scipy.signal.lfilter(b, a, data)

	return d1

	
def createSymbol(character):
	mysymbol = pg.QtGui.QPainterPath()
	mysymbol.addText(0, 0, pg.QtGui.QFont("San Serif", 10), character)
	br = mysymbol.boundingRect()
	scale = min(1. / br.width(), 1. / br.height())
	tr = pg.QtGui.QTransform()
	tr.scale(scale, scale)
	tr.translate(-br.x() - br.width()/2., -br.y() - br.height()/2.)
	mysymbol = tr.map(mysymbol)
	return mysymbol


class BindingProbabilityViewer(pg.QtWidgets.QWidget):

	def __init__(self, probability_file):
		pg.QtWidgets.QWidget.__init__(self)
		self.resize(1500,1000)

		self.layout = pg.QtWidgets.QGridLayout()
		self.layout.setContentsMargins(3,3,3,3)
		self.setLayout(self.layout)

		self.fileLoader = FileLoader(self, probability_file)

		self.plot_layout = pg.GraphicsLayout()
		self.genomePlot = self.plot_layout.addPlot(labels={'left':('binding probability')}, axisItems={'bottom':NonscientificAxisItem('bottom', text='DNA nucleotide number')}, title='Genome-Aligned probabilities')
		#self.genomePlot = self.plot_layout.addPlot(labels={'left':('binding probability')}, title='Genome-Aligned probabilities')
		self.plot_layout.nextRow()
		self.rnaPlot = self.plot_layout.addPlot(labels={'left':'binding probability - model output', 'bottom':'RNA nucleotide number'}, title='RNA-aligned probabilities')
		self.genomePlot.getAxis('bottom').enableAutoSIPrefix(False)
		self.plot_layout.nextRow()
		self.attentionPlot = self.plot_layout.addPlot(labels={'left': 'attention', 'bottom': 'RNA nucleotide number'}, title="Attention")
		self.attentionPlot.setXLink(self.rnaPlot)

		for p in [self.genomePlot, self.rnaPlot]:
			p.addLegend()
			p.showGrid(True,True)

		view = pg.GraphicsView()
		view.setCentralItem(self.plot_layout)
		view.setContentsMargins(0,0,0,0)

		self.probability_file = None 
		self.sequences = None 


		self.ctrl = pg.QtWidgets.QWidget()
		grid = pg.QtWidgets.QGridLayout()
		self.ctrl.setLayout(grid)

		self.spliceTree = pg.TreeWidget()
		self.showBtn = pg.QtWidgets.QPushButton("Show all splices")
		self.hideBtn = pg.QtWidgets.QPushButton("Hide all splices")
		self.thresholdCheck = pg.QtWidgets.QCheckBox("Threshold:")
		self.thresholdSpin = pg.SpinBox(value=0.95, bounds=[0,0.99], minStep=0.01)
		self.regionCheck = pg.QtWidgets.QCheckBox("Show regions - min length:")
		self.regionSpin = pg.SpinBox(value=3, bounds=[1, None], int=True)
		self.recallSpin = pg.SpinBox(value=0.1, bounds=[0,0.99], minStep=0.01)
		self.precisionSpin = pg.SpinBox(value=0.9, bounds=[0,0.99], minStep=0.01)
		self.showSequenceChk = pg.QtWidgets.QCheckBox("Show sequences")
		self.showAttentionChk = pg.QtWidgets.QCheckBox("Show attention (can be slow)")
		#self.showSequenceChk.setChecked(True)
		self.showFilterChk = pg.QtWidgets.QCheckBox("Show filter line")
		self.showFilterChk.setChecked(True)
		label1 = pg.QtWidgets.QLabel("Minimum Recall:")
		label1.setAlignment(pg.QtCore.Qt.AlignRight)
		label2 = pg.QtWidgets.QLabel("Minimum Precision:")
		label2.setAlignment(pg.QtCore.Qt.AlignRight)
		label3 = pg.QtWidgets.QLabel("Suggested Thresholds:")
		label3.setAlignment(pg.QtCore.Qt.AlignCenter)
		label4 = pg.QtWidgets.QLabel("lowest valid:")
		label4.setAlignment(pg.QtCore.Qt.AlignRight)
		label5 = pg.QtWidgets.QLabel("highest valid F0.2:")
		label5.setAlignment(pg.QtCore.Qt.AlignRight)
		self.fprLabel = pg.QtWidgets.QLabel("")
		self.tprLabel = pg.QtWidgets.QLabel("")
		self.fdrLabel = pg.QtWidgets.QLabel("")
		self.lowestThresholdLabel = pg.QtWidgets.QLabel("")
		self.highFScoreLabel=pg.QtWidgets.QLabel("")
		#self.metricsPlot = pg.PlotWidget(labels={'left':"True Positive Rate (TPR)", 'bottom':"False Positive Rate (FPR)"}, title="ROC")
		self.metricsPlot = pg.PlotWidget(labels={'bottom':"threshold"})
		self.metricsPlot.addLegend()
		self.metricsPlot.showGrid(True, True)
		self.metricsPlot.setMaximumSize(300,300)
		self.histogramPlot = pg.PlotWidget(labels={'left':'count', 'bottom':'model output'}, title='Control data histogram')
		self.histogramPlot.addLegend()
		self.histogramPlot.setMaximumSize(300,200)
		self.thresholdLine = pg.InfiniteLine(pos=self.thresholdSpin.value())

		grid.addWidget(self.showBtn, 0,0,1,1)
		grid.addWidget(self.hideBtn, 0,1,1,1)
		grid.addWidget(self.spliceTree, 1,0,3,2)
		grid.addWidget(self.showSequenceChk, 4, 0, 1,1)
		grid.addWidget(self.showFilterChk, 5,0,1,1)
		grid.addWidget(self.thresholdCheck, 6,0,1,1)
		grid.addWidget(self.thresholdSpin, 6,1,1,1)
		grid.addWidget(self.regionCheck, 7,0,1,1)
		grid.addWidget(self.regionSpin, 7,1,1,1)
		grid.addWidget(label1, 8,0,1,1)
		grid.addWidget(self.recallSpin, 8,1,1,1)
		grid.addWidget(label2, 9,0,1,1)
		grid.addWidget(self.precisionSpin, 9,1,1,1)
		grid.addWidget(label3, 10,0)
		grid.addWidget(label4, 11,0)
		grid.addWidget(self.lowestThresholdLabel, 11,1)
		grid.addWidget(label5, 12,0)
		grid.addWidget(self.highFScoreLabel,12,1)
		#grid.addWidget(self.fdrLabel, 8,1,1,1)
		grid.addWidget(self.metricsPlot, 13,0,2,2)
		grid.addWidget(self.histogramPlot,15,0,2,2)
		grid.addWidget(self.showAttentionChk, 17, 0, 1,1)
		grid.setRowStretch(0,20)
		#grid.setContentsMargins(1,1,1,1)
		#grid.setSpacing(1)

		self.h_splitter = pg.QtWidgets.QSplitter(pg.QtCore.Qt.Horizontal)
		self.layout.addWidget(self.h_splitter)
		self.h_splitter.addWidget(self.fileLoader)
		self.h_splitter.addWidget(view)
		self.h_splitter.addWidget(self.ctrl)
		self.h_splitter.setStretchFactor(1,10)

		self.loadFile(probability_file)
		
		self.fileLoader.fileTree.currentItemChanged.connect(self.newFileSelected)
		self.spliceTree.itemChanged.connect(self.spliceTreeItemChanged)
		self.thresholdSpin.sigValueChanged.connect(self.thresholdValueChanged)
		self.thresholdCheck.stateChanged.connect(self.thresholdValueChanged)
		self.regionCheck.stateChanged.connect(self.plot_probabilities)
		self.regionSpin.sigValueChanged.connect(self.plot_probabilities)
		self.showSequenceChk.stateChanged.connect(self.plot_probabilities)
		self.showFilterChk.stateChanged.connect(self.plot_probabilities)
		self.precisionSpin.sigValueChanged.connect(self.calculate_thresholds)
		self.recallSpin.sigValueChanged.connect(self.calculate_thresholds)
		self.showAttentionChk.stateChanged.connect(self.plot_probabilities)

		self.showBtn.clicked.connect(self.showAllSplices)
		self.hideBtn.clicked.connect(self.hideAllSplices)

	def newFileSelected(self, new, old):
		if hasattr(new, 'path'):
			self.loadFile(new.path)
			self.probability_file = new.path

	def loadFile(self, probability_file):
		if os.path.isdir(probability_file):
			return
		self.probs = load_probabilities(probability_file)

		self.spliceTree.clear()
		self.showAttentionChk.setChecked(False)
		self.splices = {}
		for k in self.probs.keys():
			if k not in ['indices', 'genomic_indices', 'metainfo', 'attention']:
				treeItem = pg.QtWidgets.QTreeWidgetItem([k])
				treeItem.setFlags(treeItem.flags() | pg.QtCore.Qt.ItemFlag.ItemIsUserCheckable)
				treeItem.setCheckState(0, pg.QtCore.Qt.CheckState.Checked)
				self.spliceTree.invisibleRootItem().addChild(treeItem)
				self.splices[k]=treeItem

		self.rbp = self.parseRBP(filename=probability_file)
		self.metricsPlot.setTitle(self.rbp)
		self.model_type = self.parse_model_type(filename=probability_file)
		self.probability_file = probability_file
		self.metrics = None
		self.rbp_stats = self.loadPerformanceData()
		self.plot_rbp_stats()
		self.calculate_thresholds()
		self.sequences = None 

		self.plot_probabilities()

	def load_sequence(self):
		#if self.sequences is not None:
		#	return self.sequences
		fasta = self.find_fasta_file(self.probability_file)
		if fasta is None:
			self.sequences = {}
			return {}

		splices = [x for x in SeqIO.parse(fasta, 'fasta')]

		sequences = {}

		for splice in splices:
			dna = np.array([s for s in splice.seq])
			mask = np.char.isupper(dna)
			rna = dna[mask]
			indices = np.argwhere(mask)[:,0]

			chromosome, start, end = util.parse_dna_range(splice.description)
			sequences[splice.id] = {'sequence':rna, 'dna_indices':indices+start}

		self.sequences = sequences
		return sequences

	def find_fasta_file(self, prob_file):
		species_dir = os.path.dirname(os.path.dirname(prob_file))
		for f in os.listdir(species_dir):
			if f[-13:] == 'genomic.fasta':
				return os.path.join(species_dir, f)
		print("Unable to find .fasta file for %s. Can't plot sequences" % prob_file)


	def showAllSplices(self):
		self.setAllSplices(True)
		self.plot_probabilities()

	def hideAllSplices(self):
		self.setAllSplices(False)
		self.plot_probabilities()

	def setAllSplices(self, b):
		if b:
			checkState = pg.QtCore.Qt.CheckState.Checked 
		else:
			checkState = pg.QtCore.Qt.CheckState.Unchecked

		try:
			self.spliceTree.blockSignals(True)
			for i in range(self.spliceTree.topLevelItemCount()):
				self.spliceTree.topLevelItem(i).setCheckState(0, b)
		finally:
			self.spliceTree.blockSignals(False)


	def parseRBP(self, filename=None):
		rbp = self.probs.get('metainfo', {}).get('rbp_name')
		if rbp is not None:
			return rbp

		if filename is not None:
			pattern = re.compile('_[A-Z0-9]*_')
			names = pattern.findall(filename)
			if len(names) == 1:
				return names[0].strip('_')
			else:
				print("Could not parse RBP name from filename: {}. Found {} matches: {}".format(filename, len(names), names))

	def parse_model_type(self, filename):
		global model_types
		if filename is None:
			filename = self.probablity_file

		#check if model_type is in the filename
		model_type = []
		for typ in model_types:
			if typ in filename:
				model_type.append(typ)
		if len(model_type) == 1:
			return model_type[0]

		#check if model_type is the name of the directory we're in
		dir_name = os.path.basename(os.path.dirname(filename))
		if dir_name in model_types:
			return dir_name

		
		print("Unable to determine model_type for file:%s" % filename)


	# def loadPerformanceData(self):
	#   try:
	#       if self.rbp is None:
	#           print("Could not load performance data, no RBP found.")
	#           return

	#       filename = None
	#       if 'RBP_performance' in os.listdir(self.fileLoader.baseDir):
	#           filename = os.path.join(self.fileLoader.baseDir, 'RBP_performance', self.rbp+'_eval_performance.csv')
	#       if filename is None or not os.path.exists(filename):
	#           # try:
	#           #   filename = os.path.join(config.rbp_performance_dir, self.rbp+'_eval_performance.csv')
	#           # except TypeError:
	#           #   if config.rbp_performance_dir is None:
	#           #       raise Exception("Please specify the rbp_performance_dir directory in your config.yaml file.")
	#           #   raise

	#       if not os.path.exists(filename):
	#           print("Could not find perfomance data for {}. Looked here:{}".format(self.rbp, filename))
	#           return
	#       return np.genfromtxt(filename, delimiter=',', names=True, dtype=[float]+[int]*6)
	#   except:
	#       print("could not load performance data")


	def loadPerformanceData(self):
		#print("loading performance data is disabled for right now.")
		#return
		
		if self.rbp is None:
			self.rbp = self.parseRBP(self.probability_file)
		if self.rbp is None:
			print("Could not load performance data, don't know which RBP we're using.")
			return

		if self.model_type is None:
			self.model_type = self.parse_model_type(self.probability_file)
		if self.model_type is None:
			print("Could not load performance data, could not find model type")
			return

		filename = None
		if 'RBP_performance' in os.listdir(self.fileLoader.baseDir):
			filename = os.path.join(self.fileLoader.baseDir, 'RBP_performance', self.model_type, self.rbp+'_eval_performance.csv')
		if filename is None or not os.path.exists(filename):
			performance_dir = config.rbp_performance_dir
			if performance_dir is None:
				print("Could not find performance directory in either fileloader base directory ({base_dir}) or in config. Please specify the rbp_performance_dir directory in your config.yaml file.".format(base_dir=self.fileLoader.baseDir))
				return
			filename = os.path.join(config.rbp_performance_dir, self.model_type, self.rbp+'_eval_performance.csv')
			if not os.path.exists(filename):
				print('Could not find performance data for {} (model:{}). Looked here:{}'.format(self.rbp, self.model_type, filename))
				return

		stats = np.genfromtxt(filename, delimiter=',', names=True, dtype=[float]+[int]*6)
		print("Loading performance data for {}, model:{} (file:{})".format(self.rbp, self.model_type, filename))
		return stats


	def plot_rbp_stats(self):
		self.metricsPlot.clear()
		self.histogramPlot.clear()
		self.histogramPlot.addItem(self.thresholdLine)
		if self.rbp_stats is None:
			return
		if self.metrics is None:
			self.metrics = util.calculate_precision_recall_fscore(self.rbp_stats)

		#self.fpr = self.rbp_stats['false_positives']/(self.rbp_stats['false_positives']+self.rbp_stats['true_negatives'])
		#self.tpr = self.rbp_stats['true_positives']/(self.rbp_stats['true_positives']+self.rbp_stats['false_negatives'])
		#self.fdr = self.rbp_stats['false_positives']/(self.rbp_stats['false_positives']+self.rbp_stats['true_positives'])

		#self.metricsPlot.plot(x=self.fpr, y=self.tpr, pen=None, symbol='o', symbolPen=None, symbolBrush='r')
		self.metricsPlot.plot(x=self.metrics['threshold'], y=self.metrics['precision'], pen='b', name='precision')
		self.metricsPlot.plot(x=self.metrics['threshold'], y=self.metrics['recall'], pen='c', name='recall')
		self.metricsPlot.plot(x=self.metrics['threshold'], y=self.metrics['F0.2_score'], pen='g', name='F0.2 score')

		#i = np.argwhere(self.rbp_stats['threshold'] == self.thresholdSpin.value())[0]
		#self.thresholdMarker = self.metricsPlot.plot(x=self.fpr[i], y=self.tpr[i], pen=None, symbol='o', symbolPen=None, symbolBrush='k')


		x = list(self.rbp_stats['threshold']) + [1.]
		self.histogramPlot.plot(x=x, y=self.rbp_stats['pos_hist'], stepMode=True, pen='b', name='positives')
		self.histogramPlot.plot(x=x, y=self.rbp_stats['neg_hist'], stepMode=True, pen='r', name='negatives')
		self.thresholdValueChanged()

	def calculate_thresholds(self):
		if self.rbp_stats is None:
			return

		prec = self.precisionSpin.value()
		recall = self.recallSpin.value()
		lowest = util.get_threshold(self.rbp_stats, min_precision=prec, min_recall=recall, mode='lowest')
		f_score = util.get_threshold(self.rbp_stats, min_precision=prec, min_recall=recall, mode='high_f0.2')

		if lowest == None:
			self.lowestThresholdLabel.setText("None")
			self.highFScoreLabel.setText("None")
		else:
			self.lowestThresholdLabel.setText("{}".format(lowest))
			self.highFScoreLabel.setText("{}".format(f_score))

	# def calculate_metrics(self, stats):
	# 	data = np.zeros(100, dtype=[
	# 		('threshold', float),
	# 		('precision', float),
	# 		('recall', float),
	# 		#('F1_score', float),
	# 		#('F0.5_score', float),
	# 		('F0.2_score', float)
	# 		])

	# 	for i, x in enumerate(stats):
	# 		threshold, TP, FP, TN, FN, p, n = x
	# 		if ((TP + FP) == 0):
	# 			data[i]['threshold'] = -1
	# 			continue
	# 		data[i]['threshold'] = threshold
	# 		data[i]['precision'] = TP / (TP + FP)
	# 		data[i]['recall'] = TP / (TP + FN)
	# 		#data[i]['F1_score'] = (2*TP) / (2*TP + FP + FN)
	# 		#data[i]['F0.5_score'] = fscore(TP, FP, FN, b=0.5)
	# 		b=0.2
	# 		data[i]['F0.2_score'] = (1 + b**2) * TP / ((1+b**2)*TP + (b**2 * FN) + FP)

	# 	return data[data['threshold'] != -1]


	def spliceTreeItemChanged(self, item, col):
		if col == 0:
			self.plot_probabilities()

	def thresholdValueChanged(self):
		if self.rbp_stats is None:
			return
		value = self.thresholdSpin.value()

		i = np.argwhere(self.rbp_stats['threshold'] == value)[0]
		#self.thresholdMarker.setData(x=self.fpr[i], y=self.tpr[i])
		#self.fprLabel.setText("%.6f"%self.fpr[i])
		#self.tprLabel.setText("%.6f"%self.tpr[i])
		#self.fdrLabel.setText("%.6f"%self.fdr[i])
		self.thresholdLine.setPos(value)

		self.plot_probabilities()


	def plot_probabilities(self):
		probs = self.probs

		self.genomePlot.clear()
		self.rnaPlot.clear()
		self.attentionPlot.clear()
		#self.attention_plots = []

		alpha = 255
		hues = len(probs.keys())
		colors = {}

		applyFilter=self.showFilterChk.isChecked() ## todo: make this a gui option
		showRegions=self.regionCheck.isChecked()
		showSequences=self.showSequenceChk.isChecked()
		if showSequences:
			symbols={
				'A':createSymbol('A'),
				'C':createSymbol('C'),
				'T':createSymbol('U'),
				'G':createSymbol('G')}


		useThreshold = self.thresholdCheck.isChecked()
		threshold = self.thresholdSpin.value()
		if showRegions:
			regions = find_binding_regions(probs, threshold=threshold, n_contiguous=self.regionSpin.value())
		if showSequences:
			sequences = self.load_sequence()


		for i,k in enumerate(sorted(self.splices.keys())):
			if k[-10:] == '_unspliced':
				spliced_key = k[:-10]
				symbol = 't1'
			else:
				spliced_key = k
				symbol = 'o'

			color = pg.intColor(i, hues)

			if self.splices[k].checkState(0) == pg.QtCore.Qt.CheckState.Unchecked:
				continue
			if applyFilter:
				filtered = besselFilter(probs[k], 0.1, dt=1)
			if useThreshold:
				thresholdMask = np.argwhere(probs[k]>=threshold)[:,0]
				alpha = 100
			if probs['indices'].get('dna_indices') is not None:
				start = probs['indices'].get('metainfo', {}).get(spliced_key, {}).get('range_start', 0)
				# workaround for an svg export bug that produces errors when axis numbers are very big (so we make them small)
				# 	- for normal operation the following block can be commented out, this is only necessary when exporting
				#	  svg files for figures
				# start_dna = probs['indices'].get('metainfo', {}).get(k, {}).get('range_start', 0)
				# start = start_dna % 100
				# offset = start_dna - start
				# print("dna_index offset for {splice}: {start}".format(splice=k, start=offset))
				# color=pg.mkColor('r')

				dna_indices = np.array(probs['indices']['dna_indices'][k]) + start
				self.genomePlot.plot(x=dna_indices, y=probs[k], symbolBrush=pg.mkBrush(color.red(), color.green(), color.blue(), alpha), name=k, pen=None, symbolPen=None, symbol=symbol)
				if applyFilter:
					connect = np.ones(len(dna_indices))
					breaks = np.argwhere(np.diff(dna_indices) > 10)[:,0]
					connect[breaks] = 0
					self.genomePlot.plot(x=dna_indices, y=filtered, pen=color, connect=connect)
				if useThreshold:
					self.genomePlot.plot(x=dna_indices[thresholdMask], y=probs[k][thresholdMask], pen=None, symbolBrush=color, symbolPen='k', symbol=symbol)
				if showRegions:
					x = np.array([a for r in regions[k] for a in r['dna_indices']]) + start
					self.genomePlot.plot(x=x, y=[1.04+0.01*i]*len(x), connect='pairs', pen={'color':color, 'width':5})
				if showSequences:
					for n in ['A', 'C', 'T', 'G']:
						x = sequences[k]['dna_indices'][np.argwhere(sequences[k]['sequence'] == n)[:,0]]
						self.genomePlot.plot(x=x, y=[1.1+0.04*i]*len(x), pen=None, symbol=symbols[n], symbolPen=color, symbolBrush=color)
			if spliced_key != k: # don't plot introns on RNA plot
				continue

			rna_indices = np.array(probs['indices']['rna_indices'][k])
			self.rnaPlot.plot(x=rna_indices, y=probs[k], symbolBrush=pg.mkBrush(color.red(), color.green(), color.blue(), alpha), name=k, pen=None, symbolPen=None)
			if applyFilter:
				self.rnaPlot.plot(x=rna_indices, y=filtered, pen=color)
			if useThreshold:
				self.rnaPlot.plot(x=rna_indices[thresholdMask], y=probs[k][thresholdMask], symbolBrush=color, pen=None, symbolPen='k')
			if showRegions:
				x = [a for r in regions[k] for a in r['rna_indices']]
				self.rnaPlot.plot(x=x, y=[1.04+0.01*i]*len(x), connect='pairs', pen={'color':color, 'width':5})
			if showSequences:
				for n in ['A', 'C', 'G', 'T']:
					x = np.argwhere(sequences[k]['sequence'] == n)[:,0]
					self.rnaPlot.plot(x=x, y=[1.1+0.04*i]*len(x), pen=None, symbol=symbols[n], symbolPen=color, symbolBrush=color)

			if self.showAttentionChk.isChecked():
				if probs.get('attention') is None:
					continue
				att = np.sum(probs['attention'][k], axis=1)  # sum across attention heads
				avg = np.zeros([3, rna_indices[-1]+int(att.shape[-1]/2)]) #array where we sum attention and counts
				avg[2] = np.arange(rna_indices[-1]+int(att.shape[-1]/2))
				for j in range(att.shape[0]):
					x = np.arange(att.shape[-1]) + (rna_indices[j]-int(att.shape[-1]/2))
					self.attentionPlot.plot(x=x[2:-2], y = att[j][2:-2])
					avg[0, x[2]:x[-2]] += att[j][2:-2]
					avg[1, x[2]:x[-2]] += 1
				mask = avg[1] != 0
				avg2 = avg[0,mask]/avg[1,mask]
				self.attentionPlot.plot(x=avg[2][mask], y=avg2, pen=pg.mkPen('g', width=2))

		
if __name__ == '__main__':
	app = pg.mkQApp()

	parser = argparse.ArgumentParser()

	parser.add_argument('probability_file', type=str, help="The path to the probability file to load (or directory to start in).")
	parser.add_argument('--debug', action='store_true', help="Run the pyqtgraph debug console")

	args = parser.parse_args()

	if args.debug:
		pg.dbg()

	mw = BindingProbabilityViewer(args.probability_file)
	mw.show()



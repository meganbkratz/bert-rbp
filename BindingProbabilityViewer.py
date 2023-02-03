import os, argparse, re
import pyqtgraph as pg 
import numpy as np
from plotting_helpers import load_probabilities
import scipy.signal
import config
from region_analysis import find_binding_regions
from Bio import SeqIO
from util import parse_dna_range

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
			self.setBaseDir(baseDir)

	def baseDirBtnClicked(self):
		baseDir = self.baseDir if self.baseDir is not None else ""
		newBaseDir = pg.FileDialog.getExistingDirectory(self, caption="Select base directory...", directory=baseDir)
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
		self.plot_layout.nextRow()
		#self.sequencePlot = self.plot_layout.addPlot(axisItems={'bottom':NonscientificAxisItem('bottom', text='DNA nucleotide number')}, title="sequence")
		self.rnaPlot = self.plot_layout.addPlot(labels={'left':'binding probability', 'bottom':'RNA nucleotide number'}, title='RNA-aligned probabilities')
		self.genomePlot.getAxis('bottom').enableAutoSIPrefix(False)
		#self.sequencePlot.getAxis('bottom').enableAutoSIPrefix(False)

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
		label1 = pg.QtWidgets.QLabel("FPR:")
		label1.setAlignment(pg.QtCore.Qt.AlignRight)
		label2 = pg.QtWidgets.QLabel("TPR:")
		label2.setAlignment(pg.QtCore.Qt.AlignRight)
		label3 = pg.QtWidgets.QLabel("False Discovery Rate:")
		label3.setAlignment(pg.QtCore.Qt.AlignRight)
		self.fprLabel = pg.QtWidgets.QLabel("")
		self.tprLabel = pg.QtWidgets.QLabel("")
		self.fdrLabel = pg.QtWidgets.QLabel("")
		self.rocPlot = pg.PlotWidget(labels={'left':"True Positive Rate (TPR)", 'bottom':"False Positive Rate (FPR)"}, title="ROC")
		self.rocPlot.setMaximumSize(300,300)
		self.histogramPlot = pg.PlotWidget(labels={'left':'count', 'bottom':'model output'}, title='Control data histogram')
		self.histogramPlot.addLegend()
		self.histogramPlot.setMaximumSize(300,200)
		self.thresholdLine = pg.InfiniteLine(pos=self.thresholdSpin.value())

		grid.addWidget(self.showBtn, 0,0,1,1)
		grid.addWidget(self.hideBtn, 0,1,1,1)
		grid.addWidget(self.spliceTree, 1,0,3,2)
		grid.addWidget(self.thresholdCheck, 4,0,1,1)
		grid.addWidget(self.thresholdSpin, 4,1,1,1)
		grid.addWidget(self.regionCheck, 5,0,1,1)
		grid.addWidget(self.regionSpin, 5,1,1,1)
		grid.addWidget(label1, 6,0,1,1)
		grid.addWidget(self.fprLabel, 6,1,1,1)
		grid.addWidget(label2, 7,0,1,1)
		grid.addWidget(self.tprLabel, 7,1,1,1)
		grid.addWidget(label3, 8,0,1,1)
		grid.addWidget(self.fdrLabel, 8,1,1,1)
		grid.addWidget(self.rocPlot, 9,0,2,2)
		grid.addWidget(self.histogramPlot,11,0,2,2)
		grid.setRowStretch(0,10)

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
		self.splices = {}
		for k in self.probs.keys():
			if k not in ['indices', 'genomic_indices', 'metainfo']:
				treeItem = pg.QtWidgets.QTreeWidgetItem([k])
				treeItem.setFlags(treeItem.flags() | pg.QtCore.Qt.ItemFlag.ItemIsUserCheckable)
				treeItem.setCheckState(0, pg.QtCore.Qt.CheckState.Checked)
				self.spliceTree.invisibleRootItem().addChild(treeItem)
				self.splices[k]=treeItem

		self.rbp = self.parseRBP(filename=probability_file)
		self.model_type = self.parse_model_type(filename=probability_file)
		self.probability_file = probability_file
		self.rbp_stats = self.loadPerformanceData()
		self.plot_rbp_stats()
		self.sequences = None 

		self.plot_probabilities()

	def load_sequence(self):
		if self.sequences is not None:
			return self.sequences
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

			chromosome, start, end = parse_dna_range(splice.description)
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
				print("Could not find performance directory in either fileloader base directory or in config. Please specify the rbp_performance_dir directory in your config.yaml file.")
				return
			filename = os.path.join(config.rbp_performance_dir, self.model_type, self.rbp+'_eval_performance.csv')
			if not os.path.exists(filename):
				print('Could not find performance data for {} (model:{}). Looked here:{}'.format(self.rbp, self.model_type, filename))
				return

		stats = np.genfromtxt(filename, delimiter=',', names=True, dtype=[float]+[int]*6)
		print("Loading performance data for {}, model:{} (file:{})".format(self.rbp, self.model_type, filename))
		return stats


	def plot_rbp_stats(self):
		self.rocPlot.clear()
		self.histogramPlot.clear()
		self.histogramPlot.addItem(self.thresholdLine)
		if self.rbp_stats is None:
			return

		self.fpr = self.rbp_stats['false_positives']/(self.rbp_stats['false_positives']+self.rbp_stats['true_negatives'])
		self.tpr = self.rbp_stats['true_positives']/(self.rbp_stats['true_positives']+self.rbp_stats['false_negatives'])
		self.fdr = self.rbp_stats['false_positives']/(self.rbp_stats['false_positives']+self.rbp_stats['true_positives'])

		self.rocPlot.plot(x=self.fpr, y=self.tpr, pen=None, symbol='o', symbolPen=None, symbolBrush='r')

		i = np.argwhere(self.rbp_stats['threshold'] == self.thresholdSpin.value())[0]
		self.thresholdMarker = self.rocPlot.plot(x=self.fpr[i], y=self.tpr[i], pen=None, symbol='o', symbolPen=None, symbolBrush='k')


		x = list(self.rbp_stats['threshold']) + [1.]
		self.histogramPlot.plot(x=x, y=self.rbp_stats['pos_hist'], stepMode=True, pen='b', name='positives')
		self.histogramPlot.plot(x=x, y=self.rbp_stats['neg_hist'], stepMode=True, pen='r', name='negatives')
		self.thresholdValueChanged()

	def spliceTreeItemChanged(self, item, col):
		if col == 0:
			self.plot_probabilities()

	def thresholdValueChanged(self):
		if self.rbp_stats is None:
			return
		value = self.thresholdSpin.value()

		i = np.argwhere(self.rbp_stats['threshold'] == value)[0]
		self.thresholdMarker.setData(x=self.fpr[i], y=self.tpr[i])
		self.fprLabel.setText("%.6f"%self.fpr[i])
		self.tprLabel.setText("%.6f"%self.tpr[i])
		self.fdrLabel.setText("%.6f"%self.fdr[i])
		self.thresholdLine.setPos(value)

		self.plot_probabilities()


	def plot_probabilities(self):
		probs = self.probs

		self.genomePlot.clear()
		self.rnaPlot.clear()

		pens = ['r','g','b','m','c','k','y']
		alpha = 255
		hues = len(probs.keys())

		applyFilter=True ## todo: make this a gui option
		showRegions=self.regionCheck.isChecked()
		showSequences=True
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
		for i,k in enumerate(self.splices.keys()):
			if self.splices[k].checkState(0) == pg.QtCore.Qt.CheckState.Unchecked:
				continue
			color = pg.intColor(i, hues)
			if applyFilter:
				filtered = besselFilter(probs[k], 0.1, dt=1)
			if useThreshold:
				thresholdMask = np.argwhere(probs[k]>=threshold)[:,0]
				alpha = 100
			if probs['indices'].get('dna_indices') is not None:
				start = probs['indices'].get('metainfo', {}).get(k, {}).get('range_start', 0)
				dna_indices = np.array(probs['indices']['dna_indices'][k]) + start
				self.genomePlot.plot(x=dna_indices, y=probs[k], symbolBrush=pg.intColor(i, hues, alpha=alpha), name=k, pen=None, symbolPen=None)
				if applyFilter:
					connect = np.ones(len(dna_indices))
					breaks = np.argwhere(np.diff(dna_indices) > 10)[:,0]
					connect[breaks] = 0
					self.genomePlot.plot(x=dna_indices, y=filtered, pen=color, connect=connect)
				if useThreshold:
					self.genomePlot.plot(x=dna_indices[thresholdMask], y=probs[k][thresholdMask], pen=None, symbolBrush=color, symbolPen='k')
				if showRegions:
					x = np.array([a for r in regions[k] for a in r['dna_indices']]) + start
					self.genomePlot.plot(x=x, y=[1.04+0.02*i]*len(x), connect='pairs', pen={'color':color, 'width':5})
				if showSequences:
					for n in ['A', 'C', 'T', 'G']:
						x = sequences[k]['dna_indices'][np.argwhere(sequences[k]['sequence'] == n)[:,0]]
						self.genomePlot.plot(x=x, y=[-0.02-0.04*i]*len(x), pen=None, symbol=symbols[n], symbolPen=color, symbolBrush=color)
			rna_indices = np.array(probs['indices']['rna_indices'][k])
			self.rnaPlot.plot(x=rna_indices, y=probs[k], symbolBrush=pg.intColor(i, hues, alpha=alpha), name=k, pen=None, symbolPen=None)
			if applyFilter:
				self.rnaPlot.plot(x=rna_indices, y=filtered, pen=color)
			if useThreshold:
				self.rnaPlot.plot(x=rna_indices[thresholdMask], y=probs[k][thresholdMask], symbolBrush=color, pen=None, symbolPen='k')
			if showRegions:
				x = [a for r in regions[k] for a in r['rna_indices']]
				self.rnaPlot.plot(x=x, y=[1.04+0.02*i]*len(x), connect='pairs', pen={'color':color, 'width':5})
			if showSequences:
				for n in ['A', 'C', 'G', 'T']:
					x = np.argwhere(sequences[k]['sequence'] == n)[:,0]
					self.rnaPlot.plot(x=x, y=[-0.02-0.04*i]*len(x), pen=None, symbol=symbols[n], symbolPen=color, symbolBrush=color )



		
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



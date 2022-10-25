import os, argparse, re, sys
import pyqtgraph as pg 
import numpy as np
from plotting_helpers import load_probabilities
import scipy.signal
import config

print('Finished imports')
os.environ['QT_MAC_WANTS_LAYER'] = '1'
print('set environment variable')
print("PyqtGraph info:")
pg.systemInfo()

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
			for f in os.listdir(path):
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


class BindingProbabilityViewer(pg.QtWidgets.QWidget):

	def __init__(self, probability_file):
		pg.QtWidgets.QWidget.__init__(self)
		self.resize(1500,1000)

		self.layout = pg.QtWidgets.QGridLayout()
		self.layout.setContentsMargins(3,3,3,3)
		self.setLayout(self.layout)

		self.fileLoader = FileLoader(self, os.path.dirname(probability_file))

		self.plot_layout = pg.GraphicsLayout()
		self.genomePlot = self.plot_layout.addPlot(labels={'left':('binding probability')}, axisItems={'bottom':NonscientificAxisItem('bottom', text='DNA nucleotide number')}, title='Genome-Aligned probabilities')
		self.plot_layout.nextRow()
		self.rnaPlot = self.plot_layout.addPlot(labels={'left':'binding probability', 'bottom':'RNA nucleotide number'}, title='RNA-aligned probabilities')
		self.genomePlot.getAxis('bottom').enableAutoSIPrefix(False)

		for p in [self.genomePlot, self.rnaPlot]:
			p.addLegend()
			p.showGrid(True,True)

		view = pg.GraphicsView()
		view.setCentralItem(self.plot_layout)
		view.setContentsMargins(0,0,0,0)

		self.ctrl = pg.QtWidgets.QWidget()
		grid = pg.QtWidgets.QGridLayout()
		self.ctrl.setLayout(grid)

		self.spliceTree = pg.TreeWidget()
		self.thresholdCheck = pg.QtWidgets.QCheckBox("Threshold:")
		self.thresholdSpin = pg.SpinBox(value=0.95, bounds=[0,1], minStep=0.01)
		label1 = pg.QtWidgets.QLabel("FPR:")
		label1.setAlignment(pg.QtCore.Qt.AlignRight)
		label2 = pg.QtWidgets.QLabel("TPR:")
		label2.setAlignment(pg.QtCore.Qt.AlignRight)
		self.fprLabel = pg.QtWidgets.QLabel("")
		self.tprLabel = pg.QtWidgets.QLabel("")
		self.rocPlot = pg.PlotWidget(labels={'left':"True Positive Rate (TPR)", 'bottom':"False Positive Rate (FPR)"}, title="ROC")
		self.rocPlot.setMaximumSize(300,300)
		self.histogramPlot = pg.PlotWidget(labels={'left':'count', 'bottom':'model output'}, title='Control data histogram')
		self.histogramPlot.addLegend()
		self.histogramPlot.setMaximumSize(300,200)
		self.thresholdLine = pg.InfiniteLine(pos=self.thresholdSpin.value())

		grid.addWidget(self.spliceTree, 0,0, 3,2)
		grid.addWidget(self.thresholdCheck, 3, 0, 1,1)
		grid.addWidget(self.thresholdSpin, 3,1,1,1)
		grid.addWidget(label1, 4,0,1,1)
		grid.addWidget(self.fprLabel, 4,1,1,1)
		grid.addWidget(label2, 5,0,1,1)
		grid.addWidget(self.tprLabel, 5,1,1,1)
		grid.addWidget(self.rocPlot, 6,0,2,2)
		grid.addWidget(self.histogramPlot,8,0,2,2)
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

	def newFileSelected(self, new, old):
		if hasattr(new, 'path'):
			self.loadFile(new.path)

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
		self.rbp_stats = self.loadPerformanceData()
		self.plot_rbp_stats()

		self.plot_probabilities()

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

	def loadPerformanceData(self):
		if self.rbp is None:
			print("Could not load performance data, no RBP found.")
			return

		filename = None
		if 'RBP_performance' in os.listdir(self.fileLoader.baseDir):
			filename = os.path.join(self.fileLoader.baseDir, 'RBP_performance', self.rbp+'_performance.csv')
		if filename is None or not os.path.exists(filename):
			try:
				filename = os.path.join(config.rbp_performance_dir, self.rbp+'_performance.csv')
			except TypeError:
				if config.rbp_performance_dir is None:
					raise Exception("Please specify the rbp_performance_dir directory in your config.yaml file.")
				raise

		if not os.path.exists(filename):
			print("Could not find perfomance data for {}. Looked here:{}".format(self.rbp, filename))
			return
		return np.genfromtxt(filename, delimiter=',', names=True, dtype=[float]+[int]*6)

	def plot_rbp_stats(self):
		self.rocPlot.clear()
		self.histogramPlot.clear()
		self.histogramPlot.addItem(self.thresholdLine)
		if self.rbp_stats is None:
			return

		self.fpr = self.rbp_stats['false_positives']/(self.rbp_stats['false_positives']+self.rbp_stats['true_negatives'])
		self.tpr = self.rbp_stats['true_positives']/(self.rbp_stats['true_positives']+self.rbp_stats['false_negatives'])

		self.rocPlot.plot(x=self.fpr, y=self.tpr, pen=None, symbol='o', symbolPen=None, symbolBrush='r')

		i = np.argwhere(self.rbp_stats['threshold'] == self.thresholdSpin.value())[0]
		self.thresholdMarker = self.rocPlot.plot(x=self.fpr[i], y=self.tpr[i], pen=None, symbol='o', symbolPen=None, symbolBrush='w')


		x = list(self.rbp_stats['threshold']) + [1.]
		self.histogramPlot.plot(x=x, y=self.rbp_stats['pos_hist'], stepMode=True, pen='b', name='positives')
		self.histogramPlot.plot(x=x, y=self.rbp_stats['neg_hist'], stepMode=True, pen='r', name='negatives')

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
		self.thresholdLine.setPos(value)

		self.plot_probabilities()


	def plot_probabilities(self):
		probs = self.probs

		self.genomePlot.clear()
		self.rnaPlot.clear()

		pens = ['r','g','b','m','c','w','y']
		alpha = 255
		hues = len(probs.keys())

		applyFilter=True ## todo: make this a gui option
		useThreshold = self.thresholdCheck.isChecked()
		threshold = self.thresholdSpin.value()
		for i,k in enumerate(self.splices.keys()):
			if self.splices[k].checkState(0) == pg.QtCore.Qt.CheckState.Unchecked:
				continue
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
					self.genomePlot.plot(x=dna_indices, y=filtered, pen=pg.intColor(i, hues))
				if useThreshold:
					self.genomePlot.plot(x=dna_indices[thresholdMask], y=probs[k][thresholdMask], pen=None, symbolBrush=pg.intColor(i, hues), symbolPen='w')
			rna_indices = np.array(probs['indices']['rna_indices'][k])
			self.rnaPlot.plot(x=rna_indices, y=probs[k], symbolBrush=pg.intColor(i, hues, alpha=alpha), name=k, pen=None, symbolPen=None)
			if applyFilter:
				self.rnaPlot.plot(x=rna_indices, y=filtered, pen=pg.intColor(i, hues))
			if useThreshold:
				self.rnaPlot.plot(x=rna_indices[thresholdMask], y=probs[k][thresholdMask], symbolBrush=pg.intColor(i, hues), pen=None, symbolPen='w')


		
if __name__ == '__main__':
	print('entering main')
	app = pg.mkQApp()

	parser = argparse.ArgumentParser()

	parser.add_argument('probability_file', type=str, help="The path to the probability file to load (or directory to start in).")

	args = parser.parse_args()

	print("parsed args")

	#pg.dbg()

	#print('setup pg.dbg()')

	mw = BindingProbabilityViewer(args.probability_file)
	print('created mw.')
	mw.show()

	print("ready.")

	if not sys.flags.interactive:
		print('starting qt event loop')
		app.exec_()
		print('exited qt eventloop')
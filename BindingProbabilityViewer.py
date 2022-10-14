import os, argparse
import pyqtgraph as pg 
import numpy as np
from plotting_helpers import load_probabilities
import scipy.signal

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
		grid.addWidget(self.spliceTree)

		self.h_splitter = pg.QtWidgets.QSplitter(pg.QtCore.Qt.Horizontal)
		self.layout.addWidget(self.h_splitter)
		self.h_splitter.addWidget(self.fileLoader)
		self.h_splitter.addWidget(view)
		self.h_splitter.addWidget(self.ctrl)


		self.loadFile(probability_file)
		self.fileLoader.fileTree.currentItemChanged.connect(self.newFileSelected)
		self.spliceTree.itemChanged.connect(self.spliceTreeItemChanged)

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

		self.plot_probabilities()

	def spliceTreeItemChanged(self, item, col):
		if col == 0:
			self.plot_probabilities()

	def plot_probabilities(self):
		probs = self.probs

		self.genomePlot.clear()
		self.rnaPlot.clear()

		pens = ['r','g','b','m','c','w','y']

		applyFilter=True ## todo: make this a gui option
		for i,k in enumerate(self.splices.keys()):
			if self.splices[k].checkState(0) == pg.QtCore.Qt.CheckState.Unchecked:
				continue
			if applyFilter:
				filtered = besselFilter(probs[k], 0.1, dt=1)
			if probs['indices'].get('dna_indices') is not None:
				start = probs['indices'].get('metainfo', {}).get(k, {}).get('range_start', 0)
				self.genomePlot.plot(x=np.array(probs['indices']['dna_indices'][k])+start, y=probs[k], symbolBrush=pg.mkColor(pens[i]), name=k, pen=None, symbolPen=None)
				if applyFilter:
					self.genomePlot.plot(x=np.array(probs['indices']['dna_indices'][k])+start, y=filtered, pen=pens[i])
			self.rnaPlot.plot(x=probs['indices']['rna_indices'][k], y=probs[k], symbolBrush=pg.mkColor(pens[i]), name=k, pen=None, symbolPen=None)
			if applyFilter:
				self.rnaPlot.plot(x=probs['indices']['rna_indices'][k], y=filtered, pen=pens[i])

		
if __name__ == '__main__':
	app = pg.mkQApp()

	parser = argparse.ArgumentParser()

	parser.add_argument('probability_file', type=str, help="The path to the probability file to load (or directory to start in).")

	args = parser.parse_args()

	pg.dbg()

	mw = BindingProbabilityViewer(args.probability_file)
	mw.show()



import os, argparse
import pyqtgraph as pg 
from plotting_helpers import load_probabilities

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
		newBaseDir = pg.FileDialog.getOpenFileName(self, caption="Select base directory...", directory=baseDir)
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
			root.addChild(item)
		elif os.path.isdir(path):
			item = pg.QtWidgets.QTreeWidgetItem(root, [os.path.basename(path)])
			root.addChild(item)
			for f in os.listdir(path):
				self.addFileItem(f, item)
		else:
			raise Exception("Why are we here?")


class BindingProbabilityViewer(pg.QtWidgets.QWidget):

	def __init__(self, probability_file):
		pg.QtWidgets.QWidget.__init__(self)

		self.layout = pg.QtWidgets.QGridLayout()
		self.layout.setContentsMargins(3,3,3,3)
		self.setLayout(self.layout)

		self.label = pg.QtWidgets.QLabel('File: %s' %probability_file)
		self.genomePlot = pg.PlotWidget(labels={'left':'binding probability', 'bottom':'DNA nucleotide number'}, title='Genome-Aligned probabilities')
		self.genomePlot.addLegend()
		self.rnaPlot = pg.PlotWidget(labels={'left':'binding probability', 'bottom':'RNA nucleotide number'}, title='RNA-aligned probabilities')
		self.rnaPlot.addLegend()

		self.fileLoader = FileLoader(self, os.path.dirname(probability_file))
		#self.layout.addWidget(self.label, ,0)
		self.layout.addWidget(self.fileLoader, 0,0)
		self.layout.addWidget(self.genomePlot, 0,1)
		self.layout.addWidget(self.rnaPlot, 1,1)


		self.loadFile(probability_file)
		self.fileLoader.fileTree.currentItemChanged.connect(self.newFileSelected)

	def newFileSelected(self, new, old):
		self.loadFile(new.path)

	def loadFile(self, probability_file):
		self.probs = load_probabilities(probability_file)
		self.plot_probabilities()

	def plot_probabilities(self):
		probs = self.probs

		self.genomePlot.clear()
		self.rnaPlot.clear()

		pens = ['r','g','b','m','c','w','y']
		for i,k in enumerate(probs.keys()):
			if k in ['indices', 'genomic_indices']:
				continue
			if probs['indices'].get('dna_indices') is not None:
				self.genomePlot.plot(x=probs['indices']['dna_indices'][k], y=probs[k], symbolBrush=pg.mkColor(pens[i]), name=k, pen=None, symbolPen=None)
			self.rnaPlot.plot(x=probs['indices']['rna_indices'][k], y=probs[k], symbolBrush=pg.mkColor(pens[i]), name=k, pen=None, symbolPen=None)

		
if __name__ == '__main__':
	app = pg.mkQApp()

	parser = argparse.ArgumentParser()

	parser.add_argument('probability_file', type=str, help="The path to the probability file to load.")

	args = parser.parse_args()

	mw = BindingProbabilityViewer(args.probability_file)
	mw.show()



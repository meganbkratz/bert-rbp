import argparse
import pyqtgraph as pg 
from plotting_helpers import load_probabilities


class BindingProbabilityViewer(pg.QtGui.QWidget):

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

		self.layout.addWidget(self.label, 0,0)
		self.layout.addWidget(self.genomePlot, 1,0)
		self.layout.addWidget(self.rnaPlot, 2,0)

		self.newFileSelected(probability_file)

	def newFileSelected(self, probability_file):
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



import pyqtgraph as pg
import numpy as np

def load_probabilities(file_name):
    print("Loading probabilities from %s" %file_name)
    import pickle
    with open(file_name, 'rb') as f:
        probs = pickle.load(f)
    return probs

def plot_test_probabilities(dataset, label="", rna='example_set'):
    """Creates a plot of probability of binding, for several 101 nucleotide controls and a longer segment made of those controls concatenated together."""
    concatenated = dataset.get('concatenated', dataset.get('concatenated_neg'))

    dataset.pop('genomic_indices')
    x = np.arange(0, concatenated.shape[0]) * 10 + 50 ## *10 because we roll a new segment to test every 10 nucleotides, and + 50 to center on the middle of the RNA
    keys = list(dataset.keys())[:-1] ## take off the concatenated key

    p = pg.plot()
    p.setLabel('bottom', 'nucleotide number')
    p.setLabel('left', 'binding probability')
    p.setTitle("%s %s binding probabilites"%(label, rna))
    p.showGrid(True, True)
    p.addLegend()
    p.plot(x=x, y=concatenated, pen=None, symbol='o', symbolBrush=(255,255,255,100), name='concatenated sample')

    x = [100*i+50 for i, k in enumerate(keys)]
    data = [dataset[k][0] for k in keys]
    p.plot(x, data, symbol='o', pen=None, name='individual samples')
    for i,k in enumerate(keys):
        x = 100*i + 50
        p.plot(x=[x], y=dataset[k], symbol='o')

    return p

def plot_probabilities(dataset, label="", rna=''):
    p = pg.plot()
    p.setLabel('bottom', 'nucleotide')
    p.setLabel('left', 'binding probability')
    p.addLegend()
    p.showGrid(True,True)
    p.setTitle("%s %s binding probabilities"%(label, rna))
    pens = ['r','g','b','m','c','w','y']

    keys = list(dataset.keys())
    for i, k in enumerate(keys):
        if k == 'genomic_indices':
            continue
        if dataset.get('genomic_indices') is not None:
            x = dataset['genomic_indices'][k]
        else:
            x = np.arange(0, dataset[k].shape[0]) * 10 + 50 ## *10 because we roll a new segment to test every 10 nucleotides, and + 50 to center on the middle of the RNA
        p.plot(x=x, y=dataset[k], pen=None, symbol='o', symbolBrush=pens[i%len(pens)], symbolPen=None, name=k)

    return p

def plot_nontraining_data(dataset, label="", rna=None):
    ## plot histograms
    p = pg.plot()
    p.setTitle("%s probability distribution for known samples" % label)
    p.setLabel('bottom', 'probability')
    p.setLabel('left', 'count')
    p.addLegend()
    p.showGrid(True,True)
    pos, x = np.histogram(dataset['positives'], bins=100, range=[0,1])
    neg, x = np.histogram(dataset['negatives'], bins=100, range=[0,1])
    p.plot(x, pos, stepMode=True, pen='b', name='binding')
    p.plot(x, neg, stepMode=True, pen='r', name='non-binding')
    return p

def plot(probs, label=None, rna=''):
    if 'positives' in probs.keys():
        return plot_nontraining_data(probs, label=label, rna=rna)
    elif 'concatenated' in probs.keys():
        return plot_test_probabilities(probs, label=label, rna=rna)
    else:
        return plot_probabilities(probs, label, rna=rna)
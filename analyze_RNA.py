import os, argparse
import torch
from Bio import SeqIO
from src.transformers_DNABERT import BertConfig, BertForSequenceClassification, DNATokenizer
from src.transformers_DNABERT.data.processors.utils import InputExample, DataProcessor
from src.transformers_DNABERT import glue_convert_examples_to_features as convert_examples_to_features
from motif.motif_utils import seq2kmer
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from util import parse_dna_range
import numpy as np
import config
try:
    import pyqtgraph as pg
    HAVE_PYQTGRAPH = True
except ImportError:
    HAVE_PYQTGRAPH = False

#pg.dbg()


#### notes
#tensor.to() - move a tensor from cpu (default) to gpu
#output_mode is always classification



##### Map:
# - load data, split it into kmers, load into a tensor
# - load model
# - send data and model to gpu -- .to()
# - output = model(input)  <- run data through model
# - do softmax on output to get probablility?
# - return probability?

MAX_LENGTH = 101

#sequence_file='/home/megan/work/lnc_rna/data/sequences/OIP5-AS1_sequences.fasta'
#pos_test_file='/home/megan/work/lnc_rna/data/sequences/TIAL1_pos.fa'
#neg_test_file='/home/megan/work/lnc_rna/data/sequences/TIAL1_neg.fa'
#pos_test_file='/home/megan/work/lnc_rna/code/bert-rbp/RBP_training_data/TIAL1.positive.fa'
#neg_test_file='/home/megan/work/lnc_rna/code/bert-rbp/RBP_training_data/TIAL1.negative.fa'
#nontrain_tsv_file='/proj/magnuslb/users/mkratz/bert-rbp/datasets/TIAL1/nontraining_sample_finetune/dev.tsv'
#train_tsv_file='/proj/magnuslb/users/mkratz/bert-rbp/datasets/TIAL1/training_sample_finetune/dev.tsv'
#nontrain_tsv_file='/home/megan/work/lnc_rna/code/bert-rbp/datasets/TIAL1/nontraining_sample_finetune/dev.tsv'




#model_path = "/home/megan/work/lnc_rna/code/bert-rbp/datasets/TIAL1/finetuned_model"
#model_path = "/proj/magnuslb/users/mkratz/bert-rbp/datasets/TIAL1/finetuned_model"
#tokenizer = DNATokenizer.from_pretrained(model_path)




def load_fasta_sequences(f, tokenizer, n_kmer):
    """Given a FASTA file with one or more splices, return a Tensor Dataset for each splice"""

    splices = [x for x in SeqIO.parse(f, 'fasta')]

    global MAX_LENGTH

    datasets = OrderedDict()
    dataset_indices = {'rna_indices':{}}
    for splice in splices:
        examples=[]
        indices=[]
        i = 0
        while i <= len(splice.seq)-MAX_LENGTH:
            kmer_sequence = seq2kmer(str(splice.seq[i:i+MAX_LENGTH].upper()), n_kmer).replace('U', 'T')
            examples.append(InputExample(splice.id+'_%i'%i, text_a=kmer_sequence, label='0')) 
            indices.append(i+MAX_LENGTH/2)
            i += 10

        features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=["0", "1"],
                max_length=MAX_LENGTH,
                output_mode="classification",
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        #all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        #all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        #all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        ### for using the model to predict, we only need the input_ids
        ## -- input_ids is a tensor or shape(batch_size, sequence_length) and it's indicies of tokens in the vocabulary (ie [2,54,18,etc])
        ## -- attention_mask is for if we use padding tokens - 1 is not masked, 0 is masked
        ## -- so a questions is if our the attention mask and token ids here are necessary

        datasets[splice.id] = all_input_ids
        dataset_indices['rna_indices'][splice.id] = indices
        #datasets[splice.id] = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    datasets['indices'] = dataset_indices
    return datasets

def load_fasta_genome(filename, tokenizer, n_kmer):
    splices = [x for x in SeqIO.parse(filename, 'fasta')]

    global MAX_LENGTH

    datasets = OrderedDict()
    dataset_indices = {'dna_indices':{}, 'rna_indices':{}, 'metainfo':{}}
    for splice in splices:
        examples = []
        dna_indices = []
        rna_indices = []
        dna = np.array([s for s in splice.seq])
        mask = np.char.isupper(dna)
        rna = "".join(dna[mask])
        indices = np.argwhere(mask)
        i = 0
        while i <= len(rna)-MAX_LENGTH:
            kmer_sequence = seq2kmer(str(rna[i:i+MAX_LENGTH]), n_kmer).replace('U', 'T')
            examples.append(InputExample(splice.id+'_%i'%i, text_a=kmer_sequence, label='0'))
            dna_indices.append(indices[int(i+MAX_LENGTH/2)][0])
            rna_indices.append(int(i+MAX_LENGTH/2))
            i += 10

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=["0", "1"],
            max_length=MAX_LENGTH,
            output_mode="classification",
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
            )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)

        chromosome, start, end = parse_dna_range(splice.description)

        datasets[splice.id] = all_input_ids
        dataset_indices['metainfo'][splice.id] = {'desc':splice.description, 'chromosome':chromosome, 'range_start':start, 'range_end':end}
        dataset_indices['dna_indices'][splice.id] = dna_indices
        dataset_indices['rna_indices'][splice.id] = rna_indices

    datasets['indices'] = dataset_indices
    return datasets


def load_tsv_sequences(filename, tokenizer):
    """Load a .tsv file of labeled sequences already split into kmers (generated by generate_datasets.py).
     Return a dict with a positive dataset and a negative dataset."""

    global MAX_LENGTH

    data = DataProcessor._read_tsv(filename) ## produces a list of ['kmer sequence', 'label']
    negatives=[]
    positives=[]
    for i, entry in enumerate(data):
        if entry[1] == '0':
            negatives.append(InputExample('uid_%i'%i, text_a=entry[0], label='0')) ## label needs to be filled in, but we don't use it
        elif entry[1] == '1':
            positives.append(InputExample('uid_%i'%i, text_a=entry[0], label='0'))

    neg_features = convert_examples_to_features(
        negatives,
        tokenizer,
        label_list=["0", "1"],
        max_length=MAX_LENGTH,
        output_mode="classification",
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        )

    pos_features = convert_examples_to_features(
        positives,
        tokenizer,
        label_list=["0", "1"],
        max_length=MAX_LENGTH,
        output_mode="classification",
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        )

    neg_input_ids = torch.tensor([f.input_ids for f in neg_features], dtype=torch.long)
    pos_input_ids = torch.tensor([f.input_ids for f in pos_features], dtype=torch.long)

    return {'positives':pos_input_ids, 'negatives':neg_input_ids}


def predict(dataset, model_path):
    """Make binding predictions for all tensors in the dataset, using the model at model_path.
    Arguments:
    dataset    a dictionary of {name:[tensor1, tensor2, ...]}(as produced by load_tsv_sequences or load_fasta_sequences)
    model_path (str) path to the model to use for prediction

    Returns a dictionary of {name:[prob1, prob2, ...]}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=1)

    results = OrderedDict()

    for name, data in dataset.items():
        if name in ['indices']:
            continue
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        predictions=None

        for batch in dataloader:
            batch = batch.to(device)

            with torch.no_grad():
                outputs = model(input_ids=batch)
                #_, logits = outputs[:2] ## see modeling_bert.py line 390 for description of outputs -- right now we only get (and need) logits
                logits = outputs[0]

            preds = logits.detach().cpu().numpy()
            if predictions is None:
                predictions = preds
            else:
                predictions = np.append(predictions, preds, axis=0)

        probs = softmax(torch.tensor(predictions, dtype=torch.float32)).numpy()

        results[name] = probs[:,1]

    results['indices'] = dataset.get('indices') ## just pass these through here
    return results

# def plot_test_probabilities(dataset, label="", rna='example_set'):
#     """Creates a plot of probability of binding (model output), for several 101 nucleotide controls and a longer segment made of those controls concatenated together."""
#     concatenated = dataset.get('concatenated', dataset.get('concatenated_neg'))

#     dataset.pop('genomic_indices')
#     x = np.arange(0, concatenated.shape[0]) * 10 + 50 ## *10 because we roll a new segment to test every 10 nucleotides, and + 50 to center on the middle of the RNA
#     keys = list(dataset.keys())[:-1] ## take off the concatenated key

#     p = pg.plot()
#     p.setLabel('bottom', 'nucleotide number')
#     p.setLabel('left', 'model output')
#     p.setTitle("%s %s model output"%(label, rna))
#     p.showGrid(True, True)
#     p.addLegend()
#     p.plot(x=x, y=concatenated, pen=None, symbol='o', symbolBrush=(255,255,255,100), name='concatenated sample')

#     x = [100*i+50 for i, k in enumerate(keys)]
#     data = [dataset[k][0] for k in keys]
#     p.plot(x, data, symbol='o', pen=None, name='individual samples')
#     for i,k in enumerate(keys):
#         x = 100*i + 50
#         p.plot(x=[x], y=dataset[k], symbol='o')

#     return p

# def plot_probabilities(dataset, label="", rna='', index_mode=None):
#     if mode not in ['rna', 'dna']:
#         raise Exception("please use the index_mode argument to choose x-axis mode. options are 'rna' or 'dna'")
#     if mode == 'dna' and dataset.get('indices', {}).get('dna_indices') is None:
#         raise Exception('DNA indices are not present in dataset')
#     p = pg.plot()
#     p.setLabel('bottom', 'nucleotide')
#     p.setLabel('left', 'model output')
#     p.addLegend()
#     p.showGrid(True,True)
#     p.setTitle("%s %s binding probabilities"%(label, rna))
#     pens = ['r','g','b','m','c','w','y']

#     keys = list(dataset.keys())
#     if mode == 'dna':
#         indices = dataset['indices']['dna_indices']
#     elif dataset.get('indices', {}).get('rna_indices') is not None:
#         indices = dataset['indices']['rna_indices']
#     else:
#         indices = None

#     for i, k in enumerate(keys):
#         if k in 'indices':
#             continue
#         if indices == None:
#             x = np.arange(0, dataset[k].shape[0]) * 10 + 50 ## *10 because we roll a new segment to test every 10 nucleotides, and + 50 to center on the middle of the RNA
#         else:
#             x = indices[k]
#         p.plot(x=x, y=dataset[k], pen=None, symbol='o', symbolBrush=pens[i%len(pens)], symbolPen=None, name=k)

#     return p

# def plot_nontraining_data(dataset, label="", rna=None):
#     ## plot histograms
#     p = pg.plot()
#     p.setTitle("%s model output distribution for known samples" % label)
#     p.setLabel('bottom', 'output')
#     p.setLabel('left', 'count')
#     p.addLegend()
#     p.showGrid(True,True)
#     pos, x = np.histogram(dataset['positives'], bins=100, range=[0,1])
#     neg, x = np.histogram(dataset['negatives'], bins=100, range=[0,1])
#     p.plot(x, pos, stepMode=True, pen='b', name='binding')
#     p.plot(x, neg, stepMode=True, pen='r', name='non-binding')
#     return p

# def plot(probs, label=None, rna=''):
#     if 'positives' in probs.keys():
#         return plot_nontraining_data(probs, label=label, rna=rna)
#     elif 'concatenated' in probs.keys():
#         return plot_test_probabilities(probs, label=label, rna=rna)
#     else:
#         return plot_probabilities(probs, label, rna=rna)

def save_probabilities(probs, file_name):
    print("Saving model output in %s"%file_name)
    import pickle
    with open(file_name, 'wb') as f:
        pickle.dump(probs, f)

def load_probabilities(file_name):
    print("Loading model output from %s" %file_name)
    import pickle
    with open(file_name, 'rb') as f:
        probs = pickle.load(f)
    return probs

#pos_data = load_sequences(pos_test_file)
#neg_data = load_sequences(neg_test_file)
#oip_data = load_sequences(sequence_file)

#pos = predict(pos_data, model_path)
#neg = predict(neg_data, model_path)
#oip = predict(oip_data, model_path)

#p_pos = plot_test_probabilities(pos, label="positive test set")
#p_neg = plot_test_probabilities(neg, label="negative test set")
#oip_plot = plot_probablities(oip, label="OIP5-AS1_splice")

#dataset = load_tsv_sequences(nontrain_tsv_file)
#training_dataset = load_tsv_sequences(train_tsv_file)
#probs = predict(dataset, model_path)
#training_probs = predict(training_dataset, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #arguments:
    #  - RBP - required I think
    #  - RNA sequences file -- might be able to guess or have a default be the non-training data for the RBP
    #  - save - whether to save the results to be plotted later
    #  - plot - whether to try to plot now

    parser.add_argument("RBP", type=str, help="The name of the RNA binding protien (RBP) to use.")
    #parser.add_argument("--genome", action="store_true", help="if True, the supplied sequence file is a genomic file rather than just RNA")
    parser.add_argument("--sequence_path", default=None, type=str, required=False, help="(optional) The path to the sequence file to use. If not specified, the non-training data for the RBP will be used")
    #parser.add_argument("--save", action="store_true", help="(optional) if true, save binding probabilities as .pk (pickle) files")
    #parser.add_argument("--plot", action="store_true", help="(optional) if true, create plots of probabilites")
    #parser.add_argument("--run_new_prediction", action="store_true", help="(optional) If true, run a new set of predictions, else only run predictions if we don't find saved ones")
    #parser.add_argument("--plot_only", action="store_true", help="(optional) if true, load a previously saved set of outputs and create plots, requires --probability_path")
    #parser.add_argument("--probability_path", default=None, type=str, help="(optional) path to a previously saved model output file (required if --plot_only is True")
    parser.add_argument("--model_path", default=None, type=str, required=True, help="The path to the model to use")
    parser.add_argument("--save_path", default=None, type=str, required=True, help="Where to save the output data.")
    parser.add_argument("--kmer", type=int, default=3)

    args = parser.parse_args()

    # if args.plot_only:
    #     if args.probability_path is None:
    #         raise Exception("No --probability_path specified. Please specify the path to the probability file to use at runtime")
    #     import pyqtgraph as pg
    #     probs = load_probabilities(args.probability_path)
    #     p = plot(probs, label=args.RBP)
    #     quit()

    #############################################################
    # Everything below only happens if --plot_only is not True  #
    #############################################################

    ### Make sure we have a path to RBP

    # if config.dataset_directory is None:
    #     raise Exception('No dataset_directory specified in config.yml. Please fill in the path to the RBP datasets directory')
    # dataset_path = os.path.normpath(config.dataset_directory)

    # if args.RBP is None:
    #     raise Exception('No RBP specified. Options are: %s' %str(os.listdir(dataset_path)))
    # elif args.RBP not in os.listdir(dataset_path):
    #     raise Exception("No data available for %s. Options are: %s" %(args.RBP, str(os.listdir(dataset_path))))

    ### make sure we have a trained model
    #model_path = os.path.join(dataset_path, args.RBP, 'model_to_use')
    model_path = args.model_path
    if not os.path.exists(model_path):
        raise Exception('Could not find model at "%s".' % model_path)


    ## find and load our sequence data
    if args.sequence_path is None:
        sequence_path = os.path.join(dataset_path, args.RBP, 'nontraining_sample_finetune', 'dev.tsv')
    else:
        sequence_path = args.sequence_path

    if not os.path.exists(sequence_path):
        raise Exception('Could not find sequence data at "%s". Path does not exist.' %sequence_path)

    ## check if a saved probability file exists, ask if we want to use it
    #save_file = sequence_path.split('.', 1)[0] + "_%s_bertrbp_output.pk"%args.RBP
    #if os.path.exists(save_file) and not args.run_new_prediction:
    #    probs = load_probabilities(save_file)
    #else:
    tokenizer=DNATokenizer.from_pretrained(model_path) ## need the tokenizer to load the sequences
    # if args.genome:
    #     print("Loading genomic sequence data from %s" %sequence_path)
    #     dataset= load_fasta_genome(sequence_path, tokenizer, args.kmer)
    if sequence_path[-3:] == '.fa' or sequence_path[-6:] == '.fasta':
        try:
            dataset= load_fasta_genome(sequence_path, tokenizer, args.kmer)
            print("Loaded genomic sequence data from %s" %sequence_path)
        except:
            print("Loading RNA sequence data from fasta file: %s"%sequence_path)
            dataset = load_fasta_sequences(sequence_path, tokenizer, args.kmer)
    elif sequence_path[-4:] == '.tsv':
        print("Loading RNA sequence data from .tsv file: %s"%sequence_path)
        dataset = load_tsv_sequences(sequence_path, tokenizer)
    else:
        raise Exception("Not sure how to load sequence from '%s'. It doesn't seem to be a .fa, .fasta, or .tsv file" % sequence_path)
    print("Running probability predictions against model at %s ....."%model_path)
    probs = predict(dataset, model_path)
    probs['metainfo'] = {'model_path': model_path, 'sequence_path':sequence_path}

    save_file = args.save_path
    save_probabilities(probs, save_file)

    #if args.plot:
    #    import pyqtgraph as pg
    #    p = plot(probs, label=args.RBP)


#### refactor goals:
#   - remove plotting
#   - command line specification for which model to use
#   - default is genome
#   - default is saving
#   - default is running a new prediction
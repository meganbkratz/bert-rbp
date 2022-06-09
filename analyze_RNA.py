import os, argparse
import torch
from Bio import SeqIO
from src.transformers_DNABERT import BertConfig, BertForSequenceClassification, DNATokenizer
from src.transformers_DNABERT.data.processors.utils import InputExample, DataProcessor
from src.transformers_DNABERT import glue_convert_examples_to_features as convert_examples_to_features
from motif.motif_utils import seq2kmer
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
import numpy as np
#import pyqtgraph as pg
import config

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




def load_fasta_sequences(f):
    """Given a FASTA file with one or more splices, return a Tensor Dataset for each splice"""

    splices = [x for x in SeqIO.parse(f, 'fasta')]

    global tokenizer, MAX_LENGTH

    datasets = OrderedDict()
    for splice in splices:
        examples=[]
        i = 0
        while i <= len(splice.seq)-MAX_LENGTH:
            kmer_sequence = seq2kmer(str(splice.seq[i:i+MAX_LENGTH]), 3).replace('U', 'T')
            examples.append(InputExample(splice.id+'_%i'%i, text_a=kmer_sequence, label='0')) 
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
        #datasets[splice.id] = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return datasets

def load_tsv_sequences(filename):
    """Load a .tsv file of labeled sequences already split into kmers (generated by generate_datasets.py).
     Return a dict with a positive dataset and a negative dataset."""

    global tokenizer, MAX_LENGTH

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

    return results

def plot_test_probabilities(dataset, label=""):
    """Creates a plot of probability of binding, for several 101 nucleotide controls and a longer segment made of those controls concatenated together."""
    concatenated = dataset.get('concatenated', dataset.get('concatenated_neg'))

    x = np.arange(0, concatenated.shape[0]) * 10 + 50 ## *10 because we roll a new segment to test every 10 nucleotides, and + 50 to center on the middle of the RNA
    keys = list(dataset.keys())[:-1] ## take off the concatenated key

    p = pg.plot()
    p.setLabel('bottom', 'nucleotide number')
    p.setLabel('left', 'binding probability')
    p.setTitle("%s RNA binding probabilites"%label)
    p.showGrid(True, True)
    p.addLegend()
    p.plot(x=x, y=concatenated, pen=None, symbol='o', symbolBrush=(255,255,255,100), name='concatenated samples')

    x = [100*i+50 for i, k in enumerate(keys)]
    data = [dataset[k] for k in keys]
    p.plot(x, data, symbol='o', pen=None, name='individual samples')
    for i,k in enumerate(keys):
        x = 100*i + 50
        p.plot(x=[x], y=dataset[k], symbol='o')



    return p

def plot_probablities(dataset, label=""):
    p = pg.plot()
    p.setLabel('bottom', 'nucleotide')
    p.setLabel('left', 'binding probability')
    p.addLegend()
    p.showGrid(True,True)
    p.setTitle("%s example binding probabilities"%label)
    pens = ['r','y','g','b','m','c','w']

    keys = list(dataset.keys())
    for i, k in enumerate(keys):
        x = np.arange(0, dataset[k].shape[0]) * 10 + 50 ## *10 because we roll a new segment to test every 10 nucleotides, and + 50 to center on the middle of the RNA
        p.plot(x=x, y=dataset[k], pen=None, symbol='o', symbolBrush=pens[i%len(pens)], symbolPen=None, name=k)

    return p

def plot_nontraining_data(dataset, label=""):
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

def plot(probs, label=None):
    if 'positives' in probs.keys():
        return plot_nontraining_data(probs, label=label)
    elif 'concatenated' in probs.keys():
        return plot_test_probabilities(probs, label=label)
    else:
        return plot_probabilities(probs, label)

def save_probabilities(probs, file_name):
    print("Saving probabilities in %s"%file_name)
    import pickle
    with open(file_name, 'wb') as f:
        pickle.dump(probs, f)

def load_probabilities(file_name):
    print("Loading probabilities from %s" %file_name)
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
    parser.add_argument("--sequence_path", default=None, type=str, required=False, help="(optional) The path to the sequence file to use. If not specified, the non-training data for the RBP will be used")
    parser.add_argument("--save", action="store_true", help="(optional) if true, save binding probabilities as .pk (pickle) files")
    parser.add_argument("--plot", action="store_true", help="(optional) if true, create plots of probabilites")
    parser.add_argument("--run_new_prediction", action="store_true", help="(optional) If true, run a new set of predictions, else only run predictions if we don't find saved ones")
    parser.add_argument("--plot_only", action="store_true", help="(optional) if true, load a previously saved set of probabilities and create plots, requires --probability_path")
    parser.add_argument("--probability_path", default=None, type=str, help="(optional) path to a previously saved probabilities file (required if --plot_only is True")

    args = parser.parse_args()

    if args.plot_only:
        if args.probability_path is None:
            raise Exception("No --probability_path specified. Please specify the path to the probability file to use at runtime")
        import pyqtgraph as pg
        probs = load_probabilities(args.probability_path)
        p = plot(probs, label=args.RBP)
        quit()

    #############################################################
    # Everything below only happens if --plot_only is not True  #
    #############################################################

    ### Make sure we have a path to RBP

    if config.dataset_directory is None:
        raise Exception('No dataset_directory specified in config.yml. Please fill in the path to the RBP datasets directory')
    dataset_path = os.path.normpath(config.dataset_directory)

    if args.RBP is None:
        raise Exception('No RBP specified. Options are: %s' %str(os.listdir(dataset_path)))
    elif args.RBP not in os.listdir(dataset_path):
        raise Exception("No data available for %s. Options are: %s" %(args.RBP, str(os.listdir(dataset_path))))

    ### make sure we have a trained model
    model_path = os.path.join(dataset_path, args.RBP, 'finetuned_model')
    if not os.path.exists(model_path):
        raise Exception('Could not find model at "%s". Please make sure you have a trained model' % model_path)


    ## find and load our sequence data
    if args.sequence_path is None:
        sequence_path = os.path.join(dataset_path, args.RBP, 'nontraining_sample_finetune', 'dev.tsv')
    else:
        sequence_path = args.sequence_path

    if not os.path.exists(sequence_path):
        raise Exception('Could not find sequence data at "%s". Path does not exist.' %sequence_path)

    ## check if a saved probability file exists, ask if we want to use it
    save_file = sequence_path.split('.', 1)[0] + "_%s_probabilities.pk"%args.RBP
    if os.path.exists(save_file) and not args.run_new_prediction:
        probs = load_probabilities(save_file)
    else:
        tokenizer=DNATokenizer.from_pretrained(model_path) ## need the tokenizer to load the sequences
        if sequence_path[-3:] == '.fa' or sequence_path[-6:] == '.fasta':
            print("Loading RNA sequence data from fasta file: %s"%sequence_path)
            dataset = load_fasta_sequences(sequence_path)
        elif sequence_path[-4:] == '.tsv':
            print("Loading RNA sequence data from .tsv file: %s"%sequence_path)
            dataset = load_tsv_sequences(sequence_path)
        else:
            raise Exception("Not sure how to load sequence from '%s'. It doesn't seem to be a .fa, .fasta, or .tsv file" % sequence_path)
        print("Running probability predictions.....")
        probs = predict(dataset, model_path)


    if args.save:
        save_file = sequence_path.split('.', 1)[0] + "_%s_probabilities.pk" % args.RBP
        save_probabilities(probs, save_file)

    if args.plot:
        import pyqtgraph as pg
        p = plot(probs, label=args.RBP)

    
import os
import argparse
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from Bio import SeqIO
from src.transformers_DNABERT import BertForSequenceClassification, DNATokenizer
from src.transformers_DNABERT.data.processors.utils import InputExample, DataProcessor, InputFeatures
from src.transformers_DNABERT import glue_convert_examples_to_features as convert_examples_to_features
from motif.motif_utils import seq2kmer, seq2kmer_aslist
from util import parse_dna_range, FastaParsingError
import config


MAX_LENGTH = 101


def load_fasta_sequences(f, tokenizer, n_kmer):
    """Given a FASTA file with one or more splices, return a Tensor Dataset for each splice"""
    global MAX_LENGTH

    datasets = OrderedDict()
    dataset_indices = {'rna_indices': {}}
    for splice in SeqIO.parse(f, 'fasta'):
        examples = []
        indices = []
        i = 0
        while i <= len(splice.seq)-MAX_LENGTH:
            kmer_sequence = seq2kmer(str(splice.seq[i:i+MAX_LENGTH].upper()), n_kmer).replace('U', 'T')
            examples.append(InputExample(splice.id+'_%i' % i, text_a=kmer_sequence, label='0'))
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

        datasets[splice.id] = all_input_ids
        dataset_indices['rna_indices'][splice.id] = indices
    datasets['indices'] = dataset_indices
    return datasets


def load_fasta_genome(filename, tokenizer, n_kmer):
    global MAX_LENGTH

    datasets = OrderedDict()
    dataset_indices = {'dna_indices': {}, 'rna_indices': {}, 'metainfo': {}}
    for splice in SeqIO.parse(filename, 'fasta'):
        chromosome, start, end = parse_dna_range(splice.description)

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
            examples.append(InputExample(splice.id+'_%i' % i, text_a=kmer_sequence, label='0'))
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

        datasets[splice.id] = all_input_ids
        dataset_indices['metainfo'][splice.id] = {
            'desc': splice.description,
            'chromosome': chromosome,
            'range_start': start,
            'range_end': end,
            }
        dataset_indices['dna_indices'][splice.id] = dna_indices
        dataset_indices['rna_indices'][splice.id] = rna_indices

    datasets['indices'] = dataset_indices
    return datasets

def load_fasta_with_introns(filename, tokenizer, n_kmer):
    max_length = 101
    spacing = 10

    datasets = OrderedDict()
    dataset_indices = {'dna_indices': {}, 'rna_indices': {}, 'metainfo': {}}
    for splice in SeqIO.parse(filename, 'fasta'):
        chromosome, start, end = parse_dna_range(splice.description)

        ### load unspliced sequence
        dna_sequence = str(splice.seq)
        features = []
        indices = []
        i = 0
        while i <= len(dna_sequence) - max_length:
            kmer_dna_sequence = seq2kmer_aslist(dna_sequence[i:i+max_length].upper().replace('U', 'T'), n_kmer)
            inputs = tokenizer.encode_plus(kmer_dna_sequence, None, add_special_tokens=True, max_length=max_length)
            dna_input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            indices.append(int(i+max_length/2))

            features.append(InputFeatures(input_ids=dna_input_ids))
            i += spacing

        dna_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        datasets[splice.id+'_unspliced'] = dna_input_ids
        dataset_indices['dna_indices'][splice.id+'_unspliced'] = indices 
        dataset_indices['rna_indices'][splice.id+'_unspliced'] = indices
        dataset_indices['metainfo'][splice.id] = {
            'desc': splice.description,
            'chromosome': chromosome,
            'range_start': start,
            'range_end': end,
            }


        ### load spliced sequence
        features = []
        dna_indices = []
        rna_indices = []
        dna = np.array([s for s in splice.seq])
        mask = np.char.isupper(dna)
        rna = "".join(dna[mask])
        indices = np.argwhere(mask)
        i = 0
        while i <= len(rna)-max_length:
            kmer_sequence = seq2kmer_aslist(rna[i:i+max_length].replace('U', 'T'), n_kmer)
            inputs = tokenizer.encode_plus(kmer_sequence, None, add_special_tokens=True, max_length=max_length)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            rna_indices.append(int(i+max_length/2))
            dna_indices.append(indices[int(i+max_length/2)][0])

            features.append(InputFeatures(input_ids=input_ids))
            i += spacing

        rna_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)

        datasets[splice.id] = rna_input_ids
        dataset_indices['dna_indices'][splice.id] = dna_indices
        dataset_indices['rna_indices'][splice.id] = rna_indices

    datasets['indices']=dataset_indices
    return datasets

    



def load_tsv_sequences(filename, tokenizer):
    """Load a .tsv file of labeled sequences already split into kmers (generated by generate_datasets.py).
     Return a dict with a positive dataset and a negative dataset."""

    global MAX_LENGTH

    data = DataProcessor._read_tsv(filename)  # produces a list of ['kmer sequence', 'label']
    negatives = []
    positives = []
    for i, entry in enumerate(data):
        # in both cases label needs to be filled in, but we don't use it
        if entry[1] == '0':
            negatives.append(InputExample('uid_%i' % i, text_a=entry[0], label='0'))
        elif entry[1] == '1':
            positives.append(InputExample('uid_%i' % i, text_a=entry[0], label='0'))

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

    return {'positives': pos_input_ids, 'negatives': neg_input_ids}


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
        predictions = None

        for batch in dataloader:
            batch = batch.to(device)

            with torch.no_grad():
                outputs = model(input_ids=batch)
                logits = outputs[0]  # because we don't supply labels, we only get logits (not loss)

            preds = logits.detach().cpu().numpy()
            if predictions is None:
                predictions = preds
            else:
                predictions = np.append(predictions, preds, axis=0)

        probs = softmax(torch.tensor(predictions, dtype=torch.float32)).numpy()

        results[name] = probs[:, 1]

    results['indices'] = dataset.get('indices')  # just pass these through here
    return results


def save_probabilities(probs, file_name):
    print("Saving model output in %s" % file_name)
    import pickle
    with open(file_name, 'wb') as f:
        pickle.dump(probs, f)


def load_probabilities(file_name, quiet=False):
    if not quiet:
        print("Loading model output from %s" % file_name)
    import pickle
    with open(file_name, 'rb') as f:
        probs = pickle.load(f)
    return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("RBP", type=str, help="The name of the RNA binding protien (RBP) to use.")
    parser.add_argument("--sequence_path", default=None, type=str, required=False, help="(optional) The path to \
        the sequence file to use. If not specified, the non-training data for the RBP will be used")
    parser.add_argument("--model_path", default=None, type=str, required=True, help="The path to the model to use")
    parser.add_argument("--save_path", default=None, type=str, required=True, help="Where to save the output data.")
    parser.add_argument("--kmer", type=int, default=3)

    args = parser.parse_args()

    # make sure we have a trained model
    model_path = args.model_path
    if not os.path.exists(model_path):
        raise Exception('Could not find model at "%s".' % model_path)

    # find and load our sequence data
    if args.sequence_path is None:
        sequence_path = os.path.join(config.dataset_path, args.RBP, 'nontraining_sample_finetune', 'dev.tsv')
    else:
        sequence_path = args.sequence_path

    if not os.path.exists(sequence_path):
        raise Exception('Could not find sequence data at "%s". Path does not exist.' % sequence_path)

    tokenizer = DNATokenizer.from_pretrained(model_path)  # need the tokenizer to load the sequences

    if sequence_path[-3:] == '.fa' or sequence_path[-6:] == '.fasta':
        try:
            #dataset = load_fasta_genome(sequence_path, tokenizer, args.kmer)
            dataset = load_fasta_with_introns(sequence_path, tokenizer, args.kmer)
            print("Loaded genomic sequence data from %s" % sequence_path)
        except FastaParsingError:
            print("Loading RNA sequence data from fasta file: %s" % sequence_path)
            dataset = load_fasta_sequences(sequence_path, tokenizer, args.kmer)
    elif sequence_path[-4:] == '.tsv':
        print("Loading RNA sequence data from .tsv file: %s" % sequence_path)
        dataset = load_tsv_sequences(sequence_path, tokenizer)
    else:
        raise Exception("Not sure how to load sequence from '%s'. It doesn't seem to be \
        a .fa, .fasta, or .tsv file" % sequence_path)
    print("Running probability predictions against model at %s ....." % model_path)
    probs = predict(dataset, model_path)
    probs['metainfo'] = {'model_path': model_path, 'sequence_path': sequence_path}

    save_file = args.save_path
    save_probabilities(probs, save_file)

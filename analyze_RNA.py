import torch
from Bio import SeqIO
from src.transformers_DNABERT import BertConfig, BertForSequenceClassification, DNATokenizer
from src.transformers_DNABERT.data.processors.utils import InputExample
from src.transformers_DNABERT import glue_convert_examples_to_features as convert_examples_to_features
from motif.motif_utils import seq2kmer
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
import numpy as np
import pyqtgraph as pg

pg.dbg()


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

sequence_file='/home/megan/work/lnc_rna/data/sequences/OIP5-AS1_sequences.fasta'
#pos_test_file='/home/megan/work/lnc_rna/data/sequences/TIAL1_pos.fa'
#neg_test_file='/home/megan/work/lnc_rna/data/sequences/TIAL1_neg.fa'
pos_test_file='/home/megan/work/lnc_rna/code/bert-rbp/RBP_training_data/TIAL1.positive.fa'
neg_test_file='/home/megan/work/lnc_rna/code/bert-rbp/RBP_training_data/TIAL1.negative.fa'



model_path = "/home/megan/work/lnc_rna/code/bert-rbp/datasets/TIAL1/finetuned_model"
tokenizer = DNATokenizer.from_pretrained(model_path)




def load_sequences(f):
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


def predict(dataset, model_path):
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

        results[name] = probs

    return results

def plot_test_probabilities(dataset, label=None):
    """Creates a plot of probability of binding, for several 101 nucleotide controls and a longer segment made of those controls concatenated together."""
    concatenated = dataset.get('concatenated', dataset.get('concatenated_neg'))

    x = np.arange(0, concatenated.shape[0]) * 10 + 50 ## *10 because we roll a new segment to test every 10 nucleotides, and + 50 to center on the middle of the RNA
    keys = list(dataset.keys())[:-1] ## take off the concatenated key

    p = pg.plot()
    p.plot(x=x, y=concatenated[:,1], pen=None, symbol='o', symbolBrush=(255,255,255,100))

    for i,k in enumerate(keys):
        x = 100*i + 50
        p.plot(x=[x], y=dataset[k][:,1], symbol='o')

    p.setLabel('bottom', 'nucleotide')
    p.setLabel('left', 'binding probability')
    if label is not None:
        p.setTitle(label)

    return p

def plot_probablities(dataset, label):
    p = pg.plot()
    p.setLabel('bottom', 'nucleotide')
    p.setLabel('left', 'binding probability')
    p.addLegend()
    if label is not None:
        p.setTitle(label)
    pens = ['r','y','g','b','m','c','w']

    keys = list(dataset.keys())
    for i, k in enumerate(keys):
        x = np.arange(0, dataset[k].shape[0]) * 10 + 50 ## *10 because we roll a new segment to test every 10 nucleotides, and + 50 to center on the middle of the RNA
        p.plot(x=x, y=dataset[k][:,1], pen=None, symbol='o', symbolBrush=pens[i%len(pens)], name=k)

    return p



pos_data = load_sequences(pos_test_file)
neg_data = load_sequences(neg_test_file)
#oip_data = load_sequences(sequence_file)

pos = predict(pos_data, model_path)
neg = predict(neg_data, model_path)
#oip = predict(oip_data, model_path)

#p_pos = plot_test_probabilities(pos, label="positive test set")
#p_neg = plot_test_probabilities(neg, label="negative test set")
#oip_plot = plot_probablities(oip, label="OIP5-AS1_splice")


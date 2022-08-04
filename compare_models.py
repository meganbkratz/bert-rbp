import os, argparse
import config
from collections import OrderedDict
from src.transformers_DNABERT import DNATokenizer
from analyze_RNA import load_tsv_sequences, predict, save_probabilities

parser = argparse.ArgumentParser()
parser.add_argument("RBP", type=str, help="The name of the RNA binding protien (RBP) to use.")

args = parser.parse_args()

RBP = args.RBP

models = {"4_gpu":{}, "6_gpu":{}, "8_gpu":{}}

models['4_gpu']['model_path'] = os.path.join(config.dataset_directory, RBP, "finetuned_model_4gpus")
models['6_gpu']['model_path'] = os.path.join(config.dataset_directory, RBP, "finetuned_model_6gpus")
models['8_gpu']['model_path'] = os.path.join(config.dataset_directory, RBP, "finetuned_model_8gpus")

sequence_path = os.path.join(config.dataset_directory, RBP, "nontraining_sample_finetune", "dev.tsv")

for k in models.keys():
	tokenizer=DNATokenizer.from_pretrained(models[k]['model_path']) ## need the tokenizer to load the sequences
	dataset = load_tsv_sequences(sequence_path, tokenizer)
	probs = predict(dataset, models[k]['model_path'])
	save_file = sequence_path.rsplit('.', 1)[0] + "_%s_%s_probabilities.pk" % (RBP, k)
	save_probabilities(probs, save_file)
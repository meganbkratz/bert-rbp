"""Because the whole genome lncRNA file is so big, we need a different approach 
than is used in analyze_RNA, which is good for analyzing smaller sets of 
sequences that the user wants to compare (such as different splices of a single 
gene).

Approach:
- for single RBP:
	-for each sequence:
		- run it through model
		- run region analysis
		- save in big .csv

"""
import os, argparse
import torch
from Bio import SeqIO
from src.transformers_DNABERT import BertConfig, BertForSequenceClassification, DNATokenizer
from src.transformers_DNABERT.data.processors.utils import InputExample, DataProcessor
from src.transformers_DNABERT import glue_convert_examples_to_features as convert_examples_to_features
from motif.motif_utils import seq2kmer, seq2kmer_aslist
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
import util, region_analysis
import numpy as np

def load_splice(splice, tokenizer, n_kmer=3):
	max_length = 101
	spacing = 10

	pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

	indices = []
	inputs = []
	i = 0
	while i <= len(splice.seq)-max_length:
		sequence = seq2kmer_aslist(str(splice.seq[i:i+max_length]).upper().replace('U', 'T'), n_kmer)
		input_ids = tokenizer.encode_plus(sequence, None, add_special_tokens=True, max_length=max_length)['input_ids']
		#features.append(InputFeatures(input_ids=inputs['input_ids']))
		inputs.append(input_ids)
		indices.append(i+int(max_length/2))
		i += spacing

	all_input_ids = torch.as_tensor(inputs, dtype=torch.long)

	return {splice.id:all_input_ids, 'indices':{'rna_indices':{splice.id:indices}}}



def predict(dataset, model):
	"""Make binding predictions for all tensors in the dataset, using the model at model_path.
	Arguments:
	dataset    a dictionary of {name:[tensor1, tensor2, ...]}(as produced by load_tsv_sequences or load_fasta_sequences)
	model_path (str) path to the model to use for prediction

	Returns a dictionary of {name:[prob1, prob2, ...]}
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	model.eval()

	softmax = torch.nn.Softmax(dim=1)

	results = OrderedDict()

	for name, data in dataset.items():
		if name in ['indices']:
			continue
		dataloader = DataLoader(data, batch_size=32, shuffle=False, pin_memory=True)
		predictions=None

		for batch in dataloader:
			batch = batch.to(device)

			with torch.no_grad():
				outputs = model(input_ids=batch)
				#_, logits = outputs[:2] ## see modeling_bert.py line 390 for description of outputs -- right now we only get (and need) logits
				logits = outputs[0]

			torch.cuda.synchronize()
			preds = logits.detach().cpu().numpy()
			#preds = logits.detach().numpy()
			if predictions is None:
				predictions = preds
			else:
				predictions = np.append(predictions, preds, axis=0)

		probs = softmax(torch.tensor(predictions, dtype=torch.float32)).numpy()

		results[name] = probs[:,1]

	results['indices'] = dataset.get('indices') ## just pass these through here
	return results

		

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument("RBP", type=str, help="The name of the RNA binding protien (RBP) to use.")
	parser.add_argument("--sequence_path", default=None, type=str, required=True, help="The path to the sequence file to use.")
	parser.add_argument("--model_path", default=None, type=str, required=True, help="The path to the model to use")
	parser.add_argument("--save_dir", default=None, type=str, required=True, help="Where to save the output data.")
	parser.add_argument("--kmer", type=int, default=3)

	args = parser.parse_args()

	print("""Optimizations:
				loading:
					-- using seq2kmer_aslist function
			    predicting:
			    	-- using pin_memory=True
		""")

	tokenizer = DNATokenizer.from_pretrained(args.model_path)
	model = BertForSequenceClassification.from_pretrained(args.model_path)

	model_stats = util.load_performance_data(os.path.join(args.model_path, args.RBP+"_eval_performance.csv"))
	threshold = util.get_threshold(model_stats, min_precision=0.9, min_recall=0.1, mode='high_f0.2')
	n_contiguous = 3

	summary_file = os.path.join(args.save_dir, args.RBP+"_binding_regions_summary.csv")
	region_file = os.path.join(args.save_dir, args.RBP+"_binding_regions.csv")

	with open(summary_file, 'w') as f:
		f.write("sequence_file:, %s, \n" % args.sequence_path)
		f.write("RBP, Splice, Splice_index, n_regions, model_type, threshold, n_contiguous, \n")

	with open(region_file, 'w') as f:
		f.write("sequence_file, %s, \n" % args.sequence_path)
		f.write("RBP, Splice, Splice_index, model_type, threshold, n_contiguous, mean_binding_probability, rna_coordinates, region_length \n")

	for i, splice in enumerate(SeqIO.parse(args.sequence_path, 'fasta')):
		dataset = load_splice(splice, tokenizer, args.kmer)
		probs = predict(dataset, model)
		regions = region_analysis.find_binding_regions(probs, threshold=threshold, n_contiguous=n_contiguous)

		with open(summary_file, 'a') as f:
			f.write("{rbp}, {splice}, {splice_index}, {n_regions}, {model_type}, {threshold}, {n_contiguous}, \n".format(
				rbp=args.RBP, 
				splice=splice.id, 
				splice_index = i,
				n_regions=len(regions[splice.id]), 
				model_type="3mer_kyamada", 
				threshold=threshold, 
				n_contiguous=n_contiguous))
		with open(region_file, 'a') as f:
			for splice in sorted(regions.keys()):
				#chrom = probs['indices']['metainfo'][splice]['chromosome']
				#start = probs['indices']['metainfo'][splice]['range_start']
				for r in regions[splice]:
					f.write("{rbp}, {splice}, {splice_index}, {model_type}, {threshold}, {n_contiguous}, {probability}, {coordinates}, {region_len}, \n".format(
						rbp=args.RBP,
						splice=splice,
						splice_index = i, 
						model_type="3mer_kyamada",
						threshold=threshold,
						n_contiguous=n_contiguous,
						probability=r['mean_probability'],
						# coordinates="chr{chrom}:{start}-{end}".format(
						#     chrom=chrom,
						#     start=r['dna_indices'][0]+start,
						#     end=r['dna_indices'][1]+start),
						coordinates = r['rna_indices'],
						#prob_file_indices=r['indices'],
						region_len=r['region_length']
						))


#### TODO:
# 	two slow places:
#		- in predict, copying the tensor to the cpu accounts for about half the total time
#			- not true: if we add torch.cuda.synchronize, it's the synchronize that's taking the time, which actually means it's the other gpu operations that are taking the time.
#			- things to try:
#				- increase the batch size
#				- increase the number of gpus
#		- load_splice accounts for most of the other half

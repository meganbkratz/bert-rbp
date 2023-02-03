import sys, os, time, datetime, argparse
import torch
import numpy as np
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from src.transformers_DNABERT import BertConfig, BertForSequenceClassification, DNATokenizer, AdamW
from src.transformers_DNABERT.data.processors.utils import InputExample, DataProcessor, InputFeatures
from src.transformers_DNABERT.data.processors.glue import DnaPromProcessor
from src.transformers_DNABERT import glue_convert_examples_to_features as convert_examples_to_features
from src.transformers_DNABERT import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def load_dataset(data_file, tokenizer):
	"""Return a torch TensorDataset from a kmerized .tsv file (where column 1 is the sequence and column 2 is the label)"""

	print("Loading dataset from: %s" % data_file)
	label_map = {"0":0, "1":1}
	max_length = 101
	pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

	features = []

	for line in DataProcessor._read_tsv(data_file)[1:]:
		sequence = line[0]
		label = label_map[line[1]]
		inputs = tokenizer.encode_plus(sequence, None, add_special_tokens=True, max_length=max_length)
		input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		attention_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding_length = max_length - len(input_ids)

		input_ids = input_ids + ([pad_token] * padding_length)
		attention_mask = attention_mask + ([0] * padding_length)
		token_type_ids = token_type_ids + ([0] * padding_length)

		assert len(input_ids) == max_length
		assert len(attention_mask) == max_length
		assert len(token_type_ids) == max_length

		features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label))

	# Convert to Tensors and build dataset
	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
	all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
	all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

	dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
	print('    success!')
	return dataset

def train(model_path, tokenizer, training_dataset, eval_dataset, **kwargs):
	#args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	print("Training model......")
	params = {
		#'batch_size_per_gpu': 32,
		'n_epochs': 3,
		'weight_decay': 0.01, #from train_and_test.sh
		'warmup_percent': 0.1, #from train_and_test.sh
		'learning_rate': 2e-4,
		'batch_size':128,
		'dropout':0.1,
		#'gradient_accumulation_steps': 1, 
		#'adam_epsilon': 1e-8,
	}
	params.update(kwargs)
	print('Training hyperparameters:', params)

	config = BertConfig.from_pretrained(
		model_path,
		num_labels=2,
	)

	#print("orig config:", config)
	config.hidden_dropout_prob = params['dropout']
	config.attention_probs_dropout_prob = params['dropout']
	#config.rnn_dropout = params['dropout']
	print("adjusted config:", config)
	model = BertForSequenceClassification.from_pretrained(model_path, config=config)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#n_gpu = torch.cuda.device_count()
	model.to(device)

	#batch_size = params['batch_size_per_gpu'] * n_gpu
	train_dataloader = DataLoader(training_dataset, sampler=RandomSampler(training_dataset), batch_size=params['batch_size']) ## might not need the sampler here, and just use shuffle=True?
	#eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=params['batch_size'])

	# Prepare optimizer and schedule (linear warmup and decay)
	## AdamW (more or less) takes all the parameters in the model and adjusts them down the gradient, then apply weight decay to the weights that don't move so that they eventually go to 0
	no_decay = ["bias", "LayerNorm.weight"] ## <- we're not gonna do weight decay for these ones
	optimizer_grouped_parameters = [
		{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": params['weight_decay'],}, # <- apply weight decay
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},]  # <- don't apply weight decay

	optimizer = AdamW(optimizer_grouped_parameters, lr=params['learning_rate']) 
	total_steps = len(train_dataloader) * params['n_epochs']
	warmup_steps = int(params['warmup_percent'] * total_steps)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

	#if n_gpu > 1:
	#       model = torch.nn.DataParallel(model)

	training_metrics = []
	validation_metrics = []

	## measure metrics before any training
	training_metrics.append({"name": 'epoch{}_step{}'.format(0, -1), 'metrics':evaluate(model, training_dataset), 'epoch':0, 'step':-1})
	validation_metrics.append({"name": 'epoch{}_step{}'.format(0, -1), 'metrics':evaluate(model, eval_dataset), 'epoch':0, 'step':-1})


	t0 = time.time()
	t_last = t0
	### do the training
	for epoch in range(params['n_epochs']):

		for step, batch in enumerate(train_dataloader):
			#print('step:%i'%step)
			model.train()
			model.zero_grad()  

			batch = tuple(t.to(device) for t in batch)
			inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3].unsqueeze(1)}
			#print('train - run through model')
			#for k,v in inputs.items():
			#       print("    ",k,':', v.shape)
			outputs = model(**inputs) # <- this is the line the warning is coming from
			loss = outputs[0] # model outputs are always tuple in transformers (see doc)
			#print('   loss:', loss)

			loss = loss.mean()

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			optimizer.step()
			scheduler.step()

			if (step+1) % 10 == 0:
				training_metrics.append({"name": 'epoch{}_step{}'.format(epoch, step), 'metrics':evaluate(model, training_dataset), 'epoch':epoch, 'step':step})
				validation_metrics.append({"name": 'epoch{}_step{}'.format(epoch, step), 'metrics':evaluate(model, eval_dataset), 'epoch':epoch, 'step':step})

				print("Epoch%i_step%i: train_loss_1=%f, validation_loss:%f, train_auroc:%f, validation_auroc:%f"%(epoch, step+1, training_metrics[-1]['metrics']['loss'], validation_metrics[-1]['metrics']['loss'],training_metrics[-1]['metrics']['auroc'], validation_metrics[-1]['metrics']['auroc']))
				#print("                                train_auroc:%f, validation_auroc:%f" % (training_metrics[-1]['metrics']['auroc'], validation_metrics[-1]['metrics']['auroc']))
				total_time = time.time() - t0
				elapsed_time = time.time() - t_last
				t_last = time.time()
				print('         total_time:', str(datetime.timedelta(seconds=int(total_time))), 'time_since_last_check:', str(datetime.timedelta(seconds=int(elapsed_time))))
	return model, {'training':training_metrics, 'validation':validation_metrics, 'hyperparameters':params}  

def evaluate(model, dataset):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#n_gpu = torch.cuda.device_count()

	model.eval()

	dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=128)
	total_loss = 0
	#import IPython

	preds = None
	for step, batch in enumerate(dataloader):
		batch = tuple(t.to(device) for t in batch)

		inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3].unsqueeze(1)}
		with torch.no_grad():
			#print('evaluate - run through model')
			outputs = model(**inputs)
		loss = outputs[0]
		logits = outputs[1]
		#IPython.embed()
		total_loss += loss.item() * len(batch[3])

		if preds is None:
			preds = logits.detach().cpu().numpy()
			labels = inputs["labels"].detach().cpu().numpy()
		else:
			preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
			labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)

	avg_loss = total_loss / len(dataset)
	probs = torch.nn.Softmax(dim=1)(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
	auc = roc_auc_score(labels, probs)
	#import IPython
	#IPython.embed()
	accuracy = (np.argmax(preds, axis=1) == labels.flatten()).mean()

	return {'loss': avg_loss, 'auroc':auc, 'accuracy':accuracy}





if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("RBP", type=str, help="The name of the RNA binding protien (RBP) to train.")
	parser.add_argument("--save_path", type=str, help="The path where the model will be saved.")
	parser.add_argument("--learning_rate", default=2e-4, help="The initial learning rate for the AdamW optimizer.")
	parser.add_argument("--dropout_rate", default=0.1)
	parser.add_argument("--kmer", type=int, default=3)
	args = parser.parse_args()

	if args.kmer == 3:
		model_path = '/proj/magnuslb/users/mkratz/bert-rbp/dnabert/3-new-12w-0/'
		data_dir = '/proj/magnuslb/users/mkratz/bert-rbp/datasets/%s/training_sample_finetune/' % args.RBP
	elif args.kmer == 6:
		model_path = '/proj/magnuslb/users/mkratz/bert-rbp/dnabert/6-new-12w-0/'
		data_dir = '/proj/magnuslb/users/mkratz/bert-rbp/datasets_6/%s/training_sample_finetune/' % args.RBP
	elif args.kmer == 5:
		model_path = '/proj/magnuslb/users/mkratz/bert-rbp/dnabert/5-new-12w-0/'
		data_dir = '/proj/magnuslb/users/mkratz/bert-rbp/datasets_5/%s/training_sample_finetune/' % args.RBP
	elif args.kmer == 4:
		model_path = '/proj/magnuslb/users/mkratz/bert-rbp/dnabert/4-new-12w-0/'
		data_dir = '/proj/magnuslb/users/mkratz/bert-rbp/datasets_4/%s/training_sample_finetune/' % args.RBP
	else:
		raise Exception('Only 3,4,5 and 6 kmer models are currently supported')

	#save_path = '/proj/magnuslb/users/mkratz/bert-rbp/datasets/%s/new_trained_model_2e-5LR_0.2dropout' % args.RBP
	save_path = args.save_path
	print('save_path', save_path)
	print('base_model', model_path)
	print('data_dir', data_dir)

	## save model and metrics
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	else:
		pass
		#print('%s already exists. quitting...')
		#sys.exit()

	tokenizer = DNATokenizer.from_pretrained(model_path)

	training_data = load_dataset(os.path.join(data_dir, 'train.tsv'), tokenizer)
	eval_data = load_dataset(os.path.join(data_dir, 'dev.tsv'), tokenizer)

	model, metrics = train(model_path, tokenizer, training_data, eval_data, learning_rate=float(args.learning_rate), dropout=float(args.dropout_rate))



	# Save a trained model, configuration and tokenizer using `save_pretrained()`.
	# They can then be reloaded using `from_pretrained()`
	#model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
	model.save_pretrained(save_path)
	tokenizer.save_pretrained(save_path)
	torch.save(metrics, os.path.join(save_path, 'training_metrics.bin'))
	with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as f:
		json.dump(metrics['hyperparameters'], f)




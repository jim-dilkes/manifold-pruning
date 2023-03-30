import os
from collections import defaultdict
import pickle as pkl
import argparse
import numpy as np

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering
from pruning.masked.utils.arch import apply_neuron_mask

parser = argparse.ArgumentParser(description='Extract linguistic features from Transformer.')
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# Input
parser.add_argument('--dataset_file', type=str, default="data/experiments/FINAL_Q.pkl",
                    help='Input pickle file with the relevant dataset. Each line contains the ambiguous word '
                    'tag and question with form ||word_1, tag_1|,...,|word_k, tag_k|| '
                    'and a boolean value for whether the question is adversarial or '
                    'not.')
parser.add_argument('--tag_file', type=str,
                    default="data/experiments/relevant_pos_tags_1.txt",
                    help='Input file with all POS tags used for Manifold Analysis.')
parser.add_argument('--sample', type=str,
                    default="data/experiments/sample_seed_1.pkl",
                    help='Input file containing the line index, '
                         'word index and tag of the randomly sampled dataset (output from '
                         'prepare_data.py.')
parser.add_argument('--pruning_metric', type=str, default=None,
                    choices=['random', 'mac', 'latency'],
                    help='Input a supported pruning metric')
parser.add_argument('--pruned_percentage', type=int,
                    default=0,
                    help='Percentage of the model that has been pruned.')
parser.add_argument('--masks_dir', type=str, default=None,
                    help='Directory with model masks.')
# Output
parser.add_argument('--feature_dir', type=str, default='data/features',
                    help='Output feature data directory.')

# Parameters
parser.add_argument('--pretrained_model_name', type=str, 
                    default=os.path.join(os.getcwd(), 'models/bert-base-uncased-squad2'),
                    choices=['bert-base-cased', 'openai-gpt', 'distilbert-base-uncased',
                             'roberta-base', 'albert-base-v1'], help='Pretrained model name.')
parser.add_argument('--mask', action='store_true', default=False,
                    help='Boolean indicating whether to mask relevant word.')
parser.add_argument('--random_init', action='store_true', default=False,
                    help='Boolean indication whether to randomly initialize the model.')


args = parser.parse_args()
print(args)

print('Extracting Features')

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
config = AutoConfig.from_pretrained(args.pretrained_model_name, output_hidden_states=True)
if args.random_init: # random initialization of the model
    model = AutoModelForQuestionAnswering.from_config(config)
else:
    model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model_name, config=config)

manifold_vectors = defaultdict(dict)
with open(args.tag_file) as f:
    for tag in f:
        tag = tag.strip().lower()
        for layer in range(1,config.num_hidden_layers+1):
            manifold_vectors[layer][tag] = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# load masks
if args.pruning_metric:
    head_mask = torch.load(os.path.join(args.masks_dir, "head_mask.pt"),
                            map_location=device)
    neuron_mask = torch.load(os.path.join(args.masks_dir, "neuron_mask.pt"),
                                map_location=device)
    handles = apply_neuron_mask(model, neuron_mask)

line_word_tag_map = pkl.load(open(args.sample, 'rb+'))

dfile = pkl.load(open(args.dataset_file, "rb"))
for line_idx,line in enumerate(dfile):
    if line_idx in line_word_tag_map:
        #skips empty strings for words and tags. line[2] is where sentences are stored.
        word_list, tags = [word[0].lower() for word in line[2] if len(word) == 2], [word[1].lower() for word in line[2] if len(word) == 2]

        for word_idx in line_word_tag_map[line_idx]:
            tag = line_word_tag_map[line_idx][word_idx].lower()
            if args.mask:
                # replace the word_idx location with mask token
                word_list[word_idx] = tokenizer.mask_token
            if args.pretrained_model_name == 'openai-gpt':
                split_word_idx = []
            else:
                split_word_idx = [-1]
            # tokenization - assign the same id for all sub words of a same word
            word_tokens = []
            for split_id, split_word in enumerate(word_list):
                tokens = tokenizer.tokenize(split_word)
                word_tokens.extend(tokens)
                split_word_idx.extend([split_id] * len(tokens))
            if args.pretrained_model_name != 'openai-gpt':
                split_word_idx.append(len(word_list))
            input_ids = torch.Tensor([tokenizer.encode(word_tokens, add_special_tokens=True, is_split_into_words=True)]).long()
            input_ids = input_ids.to(device)
            with torch.no_grad():
                if args.pruning_metric:
                    model_output = model(input_ids, head_mask=head_mask)[-1]
                else:
                    model_output = model(input_ids)[-1]
            for layer in range(1,config.num_hidden_layers+1):
                layer_output = model_output[layer][0]
                vector_idcs = np.argwhere(np.array(split_word_idx) == word_idx).reshape(-1)
                token_vector = layer_output[vector_idcs].mean(0).cpu().reshape(-1,1).numpy()
                if manifold_vectors[layer][tag] is None:
                    manifold_vectors[layer][tag] = token_vector
                else:
                    manifold_vectors[layer][tag] = np.hstack((manifold_vectors[layer][tag],
                                                                token_vector))

os.makedirs(args.feature_dir, exist_ok=True)
for layer in range(1,config.num_hidden_layers+1):
    pkl.dump(list(manifold_vectors[layer].values()), open(os.path.join(args.feature_dir,
                                                                  str(layer)+'.pkl'), 'wb+'))

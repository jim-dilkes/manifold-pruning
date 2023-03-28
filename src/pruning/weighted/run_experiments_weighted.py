import pickle as pkl
import json
import os
import numpy as np
from bert import extract_features_manifolds as extract_features
from bert import tokenization
import random
from collections import defaultdict
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

def _get_experiment_dir(model_name, experiment_number):
    return f"experiments\\experiment{experiment_number}_prune_{model_name}"

def feature_extract(model_name, experiment_number):
    model_dir = f"models\\weighted\\bert-prune-{model_name}-squad"
    experiment_dir = _get_experiment_dir(model_name, experiment_number)
    input_pkl_filename = "FINAL_Q.pkl"
    relevant_pos_filename = f"relevant_pos_tags_{experiment_number}.txt"
    sample_filename = f"sample_seed_{experiment_number}.pkl"

    num_hidden_layers = 12

    ### LOAD AND RESTRUCTURE INPUTS ###

    # Experiment directories
    inputs_dir = os.path.join(experiment_dir, "input")
    intermediate_dir = os.path.join(experiment_dir, "intermediate")
    features_dir = os.path.join(experiment_dir, "features")
    mftma_dir = os.path.join(experiment_dir, "mftma-analysis")

    # Make the directories if they don't exist
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(mftma_dir, exist_ok=True)

    # Load the data
    input_pkl_file = os.path.join(inputs_dir, input_pkl_filename)
    sample_pkl_file = os.path.join(inputs_dir, sample_filename)

    tagged_data_obj = pkl.load(open(input_pkl_file, 'rb'))
    relevant_pos = open(os.path.join(inputs_dir, relevant_pos_filename), 'r').read().splitlines()
    line_word_tag_map = pkl.load(open(sample_pkl_file, 'rb'))

    INPUT_examples = []
    INPUT_sentences = []
    INPUT_key_tag = []

    for ex in tagged_data_obj:
            tags = []
            words = []
            for word_tag in ex[2]:
                # Some words are not tagged (looks like only empty strings)
                if len(word_tag)==2:
                    words.append(word_tag[0].lower())
                    tags.append(word_tag[1].lower())
            sentence = " ".join(words)

            INPUT_examples.append({"words": words, "tags": tags})
            INPUT_sentences.append(sentence)
            INPUT_key_tag.append(f"{ex[0]}^{ex[1]}")
        
        
    print(f"Found {len(INPUT_examples)} examples")

    sentences_file = os.path.join(intermediate_dir, "bert_input_sentences.txt")

    # Write the input sentences to a file
    with open(sentences_file, "w", encoding="utf-8") as f:
        for sentence in INPUT_sentences:
            f.write(f"{sentence}\n")

    ### EXTRACT FEATURES ###

    # Model files
    bert_config_file = os.path.join(model_dir, "bert_config.json")
    vocab_file = os.path.join(model_dir, "vocab.txt")
    init_checkpoint = model_dir

    # Layer indices
    num_layers = 12
    layers = range(num_layers)
    layers_str = ",".join([str(l) for l in layers])

    # Target file
    features_file = os.path.join(intermediate_dir, "bert_features.json")

    # Extracts features of the sentences in file sentences_file, records them
    #   in file features_file
    extract_features.extract(input_file=sentences_file,
                            output_file=features_file,
                            bert_config_file=bert_config_file,
                            init_checkpoint=init_checkpoint,
                            vocab_file=vocab_file,
                            layers=layers_str
                            )
    
    ### RESTRUCTURE FEATURES ###

    # Load the previously saved features
    features_jsons = []
    with open(features_file, 'r') as fp:
        features_jsons.extend(json.loads(line) for line in fp)
        
    # Getting feature vectors out of features_jsons:
    # features_jsons[i]['features'][j]['layers'][k]['values']
    # i = example
    # j = token
    # k = layer  

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    # initialise the dict structure
    manifold_vectors = defaultdict(dict)
    for tag in relevant_pos:
        tag = tag.strip().lower()
        for layer in range(1, num_hidden_layers + 1):
            manifold_vectors[layer][tag] = None

            
    # Iterate over the examples + their data structure
    # Fill in the corresponding elements in manifold_vectors

    # Relying on the feature extraction above running on ALL lines -> indexing preserved
    for line_idx, (features_dict, key_tag) in enumerate(zip(features_jsons, INPUT_examples)): 
        if line_idx in line_word_tag_map:
            word_list = key_tag['words']
            for word_idx in line_word_tag_map[line_idx]: 
                tag = line_word_tag_map[line_idx][word_idx]
                # appending -1 to the 1st and last position to represent CLS and SEP tokens 
                split_word_idx = [-1]
                # tokenization - assign the same id for all sub words of a same word
                for split_id, split_word in enumerate(word_list):
                    tokens = tokenizer.tokenize(split_word)
                    split_word_idx.extend([split_id] * len(tokens))
                split_word_idx.append(-1)
                
                vector_idcs = np.argwhere(np.array(split_word_idx) == word_idx).reshape(-1)         
                tokens_features = [features_dict['features'][i]['layers'] for i in vector_idcs]
                
                # iterating through layers in mftma encoding (1-12)
                for layer in range(1, num_hidden_layers + 1):     
                    tokens_layer_features = [token_features[layer-1]['values'] for token_features in tokens_features]
                    # take the mean of token features of the same word to represent the word as a single feature vector
                    token_vector = np.mean(tokens_layer_features, axis=0).reshape(-1,1)
                    if manifold_vectors[layer][tag] is None:
                        manifold_vectors[layer][tag] = token_vector
                    else:
                        manifold_vectors[layer][tag] = np.hstack((manifold_vectors[layer][tag], token_vector))                    

    ### STORE FEATURE ARRAYS ###

    for layer in range(1,num_hidden_layers+1):
        pkl.dump(list(manifold_vectors[layer].values()), open(os.path.join(features_dir,
                                                                  str(layer)+'.pkl'), 'wb+'))
        
def mftma_analysis(feature_dir, mftma_analysis_dir, num_layers=12, kappa=1e-8, n_t=200, n_reps=1):

    for layer in range(1,num_layers+1):
        print(f'MFTMA analysis, layer {str(layer)}')
        class_encodings = pkl.load(
            open(os.path.join(feature_dir, f'{str(layer)}.pkl'), 'rb+')
        )

        a, r, d, r0, K = manifold_analysis_corr(class_encodings, kappa, n_t, n_reps=n_reps)

        mftma_analysis_data = {'a': a, 'r': r, 'd': d, 'r0': r0, 'K': K}
        pkl.dump(
            mftma_analysis_data,
            open(
                os.path.join(mftma_analysis_dir, f'{str(layer)}.pkl'), 'wb+'
            ),
        )

def run_experiments(model_names, experiment_numbers, mftma_only=False):
    # model_name = pruning percentage
    # example: 
    # from run_experiments_weighted import run_experiments
    # run_experiments([30,40,50], [1])

    for experiment_number in experiment_numbers:
        for model_name in model_names:
            experiment_dir = _get_experiment_dir(model_name, experiment_number)
            features_dir = os.path.join(experiment_dir, "features")
            mftma_dir = os.path.join(experiment_dir, "mftma-analysis")

            if not mftma_only:
                feature_extract(model_name, experiment_number)
            mftma_analysis(features_dir, mftma_dir)


def generate_plot(experiment_number, model_names=None, num_layers=12):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    if not model_names:
        model_names = [30,40,50]

    for model_name in model_names:
        experiment_dir = _get_experiment_dir(model_name, experiment_number)
        mftma_dir = os.path.join(experiment_dir, "mftma-analysis")

        capacities = []
        radii = []
        dimensions = []
        correlations = []

        for layer in range(1,num_layers+1):
            temp_data = pkl.load(open(os.path.join(mftma_dir,str(layer)+'.pkl'), 'rb+'))
            a = 1 / np.mean(1 / temp_data['a'])
            r = np.mean(temp_data['r'])
            d = np.mean(temp_data['d'])
            r0 = temp_data['r0']
            if layer == 1:
                norm_a = a
                norm_r = r
                norm_d = d
                norm_r0 = r0

            a /= norm_a
            r /= norm_r
            d /= norm_d
            r0 /= norm_r0
            print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(
                'LAYER_' + str(layer), a, r, d, r0))

            capacities.append(a)
            radii.append(r)
            dimensions.append(d)
            correlations.append(r0)

        axes[0].plot(capacities, linewidth=5)
        axes[1].plot(radii, linewidth=5)
        axes[2].plot(dimensions, linewidth=5)
        axes[3].plot(correlations, linewidth=5)

    axes[0].set_ylabel(r'$\alpha_M$', fontsize=18)
    axes[1].set_ylabel(r'$R_M$', fontsize=18)
    axes[2].set_ylabel(r'$D_M$', fontsize=18)
    axes[3].set_ylabel(r'$\rho_{center}$', fontsize=18)

    xticklabels = [i for i in range(1,num_layers+1)]
    for ax in axes:
        ax.set_xticks([i for i, _ in enumerate(xticklabels)])
        ax.set_xlabel('Layer')
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show()
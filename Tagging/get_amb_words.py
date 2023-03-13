"""
Using entropy, we score ambiguous words by their presence in 
multiple POS tags. This returns a list of words, and their scores,
to then use for MFMA.
"""

import json
from collections import Counter
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

modules = ["Train", "Test", "Validation"]
open_word_tag = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "NN", "NNS",
                 "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "FW"]


parser = argparse.ArgumentParser(
    description='Generate list of ambiguous words by POS tags.')
parser.add_argument('--N', type=int, default=2,
                    help='Minimum number of occurences of POS tags word is included in.')

args = parser.parse_args()
print(args)


def pos_tag_filter(pos_dict, N):
    """
    Returns a set of words who are present in atleast 
    N POS tags:

    Input:
    pos_dict (POS_Tag: List of words): Dictionary containing POS tags
    and a list of words (potentially duplicates) who have that corresponding
    tag somewhere in the corpus.
    N (int): Minimum number of POS tags the word is present in

    Output:
    ambiguous_words: Set of words that are present in atleast
    N POS tags
    """
    full_words = []
    for key in pos_dict.keys():
        full_words.extend(pos_dict[key])
    full_words = list(set(full_words))
    tmp = {key: list(set(pos_dict[key])) for key in pos_dict.keys()}
    ambiguous_words = []
    # Filtering by minimum number of instances
    for word in tqdm(full_words):
        cnt = 0
        for key in tmp.keys():
            if word in tmp[key]:
                cnt += 1
            if cnt >= N:
                ambiguous_words.append(word)
                break

    return ambiguous_words


for env in modules:
    file_name = env + "/" + env + "_frequency_bin.json"
    with open(file_name, "r") as f:
        json_file = json.load(f)
    open_tag_dict = {key: json_file[key] for key in open_word_tag}

    ambiguous_words = pos_tag_filter(open_tag_dict, 2)
    ambiguous_words_pos = {}
    for key in open_word_tag:
        ambiguous_words_pos[key] = [
            word for word in open_tag_dict[key] if word in ambiguous_words]

    n = len(ambiguous_words)
    k = len(ambiguous_words_pos.keys())
    df = pd.DataFrame(np.zeros((n, k)), columns=open_word_tag,
                      index=ambiguous_words)

    for tag in open_word_tag:
        words_count = Counter(ambiguous_words_pos[tag])
        vals = np.zeros(n)
        for indx, word in enumerate(ambiguous_words):
            vals[indx] = words_count.get(word, 0.0)
        df[tag] = vals

    df["TotalCount"] = np.sum(df.values, axis=1)
    df = df[df["TotalCount"] > 10]
    df = df.drop(columns=["TotalCount"])

    word_tag_matrix = df.values
    n, k = word_tag_matrix.shape

    tag_prob = np.sum(word_tag_matrix, axis=0)
    word_norm = (word_tag_matrix.T/word_tag_matrix.sum(axis=1)).T
    word_score = -np.sum(word_norm * np.log2(word_norm,
                         out=np.zeros_like(word_norm), where=(word_norm != 0)), axis=1)

    res = pd.DataFrame(np.zeros((n, 2)), columns=["Word", "Score"])
    res["Word"] = df.index
    res["Score"] = word_score
    res = res.sort_values(by=["Score"], ascending=False)
    res.to_csv(env+"/"+env+"_word_amb_entr_score.csv", index=False)

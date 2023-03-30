import random
import pickle as pkl
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='Prepare data for MFTMA analysis by sampling the '
                                             'dataset such that each manifold '
                                             'contains between min_manifold_size and '
                                             'max_manifold_size number of samples.')

# Input
parser.add_argument('--dataset_file', type=str, default="data/experiments/FINAL_Q.pkl",
                    help='Input pickle file with the relevant dataset. Each line contains the ambiguous word '
                    'tag and question with form ||word_1, tag_1|,...,|word_k, tag_k|| '
                    'and a boolean value for whether the question is adversarial or '
                    'not. ')
parser.add_argument('--tag_file', type=str,
                    default="data/experiments/all_pos.txt",
                    help='Input file with all possible base POS tags. ')

# Output
parser.add_argument('--sample', type=str, default="data/experiments/sample_seed_1.pkl",
                    help='Output file containing the  line index, '
                         'word index and tag of the randomly sampled dataset.')
parser.add_argument('--relevant_tags', type=str,
                    default="data/experiments/relevant_pos_tags_1.txt",
                    help='Output file to store relevant POS tags. ')

# Parameters
parser.add_argument('--max_manifold_size', type=int, default=50,
                    help='The maximal number of words per manifold.')
parser.add_argument('--min_manifold_size', type=int, default=5,
                    help='The minimal number of words per manifold.')
parser.add_argument('--seed', type=int, default=0,
                    help='Randomization seed.')

args = parser.parse_args()

print(args)

random.seed(args.seed)

"""
Open dataset file and get tags and words - save them to a dictionary
Cycle through data file with rows |amb_word,amb_tag||word_1, tag_1|, ..., 
|word_k, tag_k||, bool adversarial |. Retrieve relevant pos tags as
data_tags and the ambiguous word tags as amb_tags. 
Use the and operator on all possible relevant pos_tags from relevant_pos.txt
and get a final list of pos_tags stored in train_question_final_pos.txt
"""
data = pkl.load(open(args.dataset_file, "rb"))
data_tags = []
amb_tags = []
for line in data:
    amb_tags.append(line[0].lower() + "^" + line[1].lower())
    for tmp in line[2]:
        if len(tmp) == 2: 
            data_tags.append(tmp[-1].lower())

with open(args.tag_file, "r", newline='\r\n') as f:
    pos_tags = [tag.split('\r\n')[0].lower() for tag in f.readlines()]
    pos_tags.extend(amb_tags)
pos_tags = set(pos_tags)   
data_tags = set(data_tags)
print(pos_tags)
print(data_tags)
relevant_tags = pos_tags & data_tags
with open(args.relevant_tags, 'w+') as f:
    s = '\n'.join(relevant_tags)
    f.write(s)
    
# open dataset file and get tags and words - save them to a dictionary
tag2location = defaultdict(list)
for line_idx, line in enumerate(data):
    tags = [word_tag[-1].lower() for word_tag in line[2] if (line[0].lower() != word_tag[0].lower()) and (len(word_tag) == 2)]
    tags += [line[0].lower() + "^" + line[1].lower()]
    for word_idx, tag in enumerate(tags):
        if tag in relevant_tags:
            tag2location[tag].append((line_idx, word_idx))

word_count = 0
line_word_tag_map = defaultdict(dict)
for tag in tag2location:
    if len(tag2location[tag]) > args.max_manifold_size:
        locations = random.sample(tag2location[tag], args.max_manifold_size)
    elif len(tag2location[tag]) >= args.min_manifold_size:
        locations = tag2location[tag]
    else:
        continue
    for location in locations:
        line_idx, word_idx = location
        word_count += 1
        line_word_tag_map[line_idx][word_idx] = tag

print('Number of Words', word_count)
print('Number of Manifolds', len(tag2location))
pkl.dump(line_word_tag_map, open(args.sample,'wb+'))

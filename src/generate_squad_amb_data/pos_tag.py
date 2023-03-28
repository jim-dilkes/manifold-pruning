"""
Tags SQUAD questions stored in Train/Validation/Test - _QA.txt using spacy.
"""

import spacy
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")

modules = ["Train", "Validation", "Test"]

for env in modules:
    file_name = env+"_QA.txt"
    with open(env+"/"+file_name, "r", newline="\n") as f:
        lines = [line.rstrip() for line in f.readlines()]
    pos_tags = []
    for indx, line in tqdm(enumerate(lines)):
        doc = nlp(line)
        pos_tag = ' '.join([token.text+":"+token.tag_ for token in doc])
        pos_tags.append(pos_tag)
    pos_tags = [tmp+"\n" for tmp in pos_tags]
    with open(env+"/"+env+"_freq_POS.txt", "w") as f:
        f.writelines(pos_tags)

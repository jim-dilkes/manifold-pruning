## Loads the part (10%?) of the Penn Treebank that is available in NLTK
## Transforms it in to the required format for perpare_data.py

from nltk.corpus import BracketParseCorpusReader, LazyCorpusLoader
import nltk

# Load the treebank
nltk.download('treebank')
treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')

# Reformat the treebank to work with the feature extraction script
reformatted_treebank = []
for sent in treebank.tagged_sents():
    words, tags = zip(*sent)
    reformatted_treebank.append(" ".join(words) + ' \t ' + " ".join(tags))

    
# Save the reformatted treebank
with open('dataset/ptb_pos.txt', 'w') as f:
    f.writelines("\n".join(reformatted_treebank))


## Word^Tag

# Reformat the treebank data to work with the feature extraction script
# with tags like {word}^{tag}
reformatted_treebank = [\
    " ".join([word for word, _ in sent])
    + "\t"
    + " ".join([f"{word.lower()}^{tag}" for word, tag in sent])
    for sent in treebank.tagged_sents()
]

# Save the reformatted treebank
with open('dataset/ptb_word_pos.txt', 'w') as f:
    f.writelines("\n".join(reformatted_treebank))
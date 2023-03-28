#!/bin/bash

## This script wont run as is, but will work as a guide to help run the python commands 
# correctly

## This is the first script that should be executed on the raw downloaded wiki and 
# bookcorpus data


# Sentencize the data in batches
# First arg is the current batch number, second arg is the total batch numbers 
# They are split by taking every nth line, where n is the total batch numbers
# Arg 3 is the directory of the wiki data
# Arg 4 is the directory of the book data
# Arg 5 is the output file
python scripts/preprocess_pretrain_data.py 0 4 {wiki_dir} {book_dir}/out_txts > {drive_data}/sentencized_0.txt

# Randomly shuffle and split the data into files (which will later be assigned to train, val, and test sets)
# Arg 1 is the directory of the sentencized data
# Arg 2 is the number of files to split the data into
# The new data is in the same directory as the sentencized data
#   - a number is appended to the original file name
python scripts/shuffle_and_split.py data/sentencized_0 50
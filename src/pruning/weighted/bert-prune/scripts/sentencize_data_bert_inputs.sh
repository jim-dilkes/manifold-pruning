#!/bin/bash

i=$1 
splits_start=$2
splits_end=$3
for (( j=$splits_start; $j < $splits_end; ++j ))
do
    echo Doing $i $j
    python bert/create_pretraining_data.py --input_file data/sentencized/sentencized_$i\_$j --output_file data/pretraining/bert_in/sentencized_$i\_$j.tfrecord --vocab_file uncased_L-12_H-768_A-12/vocab.txt --do_lower_case True --max_seq_length 128 --max_predictions_per_seq 20 --masked_lm_prob 0.15 --random_seed 12345 --dupe_factor 5
done

python scripts/train_test_split.py /home/azureuser/cloudfiles/code/Users/jimdilkes/bert-prune/data/pretraining/bert_in /home/azureuser/cloudfiles/code/Users/jimdilkes/bert-prune/data/pretraining/train /home/azureuser/cloudfiles/code/Users/jimdilkes/bert-prune/data/pretraining/test /home/azureuser/cloudfiles/code/Users/jimdilkes/bert-prune/data/pretraining/dev
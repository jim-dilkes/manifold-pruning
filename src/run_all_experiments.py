import argparse
import os

parser = argparse.ArgumentParser(description='Run all experiments for a given pruning metric.')

parser.add_argument('--pruning_metric', type=str, default=None,
                    choices=['random', 'mac', 'latency'],
                    help='Input a supported pruning metric')
args = parser.parse_args()

pruned_percentages = range(0, 100, 10)
experiment_ids = range(1,6)

for id in experiment_ids:
    for perc in pruned_percentages:

        if perc == 0:
            os.system(f'python feature_extract.py \
                    --dataset_file data/experiments/FINAL_Q.pkl \
                    --tag_file data/experiments/relevant_pos_tags_{id}.txt \
                    --sample data/experiments/sample_seed_{id}.pkl \
                    --feature_dir data/features/train_{id}')
            os.system(f'python mftma_analysis.py \
                    --feature_dir data/features/train_{id} \
                    --mftma_analysis_dir results/masked/train_{id}')

        else:
            os.system(f'python feature_extract.py \
                    --dataset_file data/experiments/FINAL_Q.pkl \
                    --tag_file data/experiments/relevant_pos_tags_{id}.txt \
                    --sample data/experiments/sample_seed_{id}.pkl \
                    --pruning_metric {args.pruning_metric} \
                    --pruned_percentage {perc} \
                    --feature_dir data/features/train_{id}')
            
            os.system(f'python mftma_analysis.py \
                    --pruning_metric {args.pruning_metric} \
                    --pruned_percentage {perc} \
                    --feature_dir data/features/train_{id} \
                    --mftma_analysis_dir results/masked/train_{id}')

        
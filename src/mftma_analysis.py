import os
from time import time
import argparse
import pickle as pkl

from mftma.manifold_analysis_correlation import manifold_analysis_corr

parser = argparse.ArgumentParser(description='MFTMA analysis over layers.')

# Input
parser.add_argument('--feature_dir', type=str, default='data/features',
                    help='Input feature data directory.')
parser.add_argument('--pruning_metric', type=str, default=None,
                    choices=['random', 'mac', 'latency', None],
                    help='Input a supported pruning metric')
parser.add_argument('--pruned_percentage', type=int,
                    default=0,
                    help='Percentage of the model that has been pruned.')

#Output
parser.add_argument('--mftma_analysis_dir', type=str, default='results/mftma-analysis',
                    help='Location to output MFTMA analysis directory.')

parser.add_argument('--num_layers', type=int, default=12, help='Number of hidden layers.')

# MFTMA parameters
parser.add_argument('--kappa', type=float, default=1e-8, help='Margin size to use in the '
                                                              'analysis (kappa > 0).')
parser.add_argument('--n_t', type=int, default=1000, help='Number of gaussian vectors to sample '
                                                          'per manifold.')

parser.add_argument('--n_reps', type=int, default=10, help='Number of repetitions.')

args = parser.parse_args()
print(args)

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if args.pruning_metric:
    feature_dir = os.path.join(parent_dir, args.feature_dir, args.pruning_metric, f"{args.pruned_percentage}Pruned")
else:
    feature_dir = os.path.join(parent_dir, args.feature_dir, "0Pruned")
    
for layer in range(1,args.num_layers+1):
    start_time = time()
    print(f'============ layer {layer} ============')
    class_encodings = pkl.load(
        open(os.path.join(feature_dir, f'{str(layer)}.pkl'), 'rb+')
    )

    a, r, d, r0, K = manifold_analysis_corr(class_encodings, args.kappa, args.n_t, n_reps=args.n_reps)
    print(f'Finished layer {layer} in {time() - start_time:.3f} s')

    mftma_analysis_data = {'a': a, 'r': r, 'd': d, 'r0': r0, 'K': K}
 
    if args.pruning_metric:
        mftma_analysis_dir = os.path.join(parent_dir, args.mftma_analysis_dir, args.pruning_metric, f"{args.pruned_percentage}Pruned")
    else:
        mftma_analysis_dir = os.path.join(parent_dir, args.mftma_analysis_dir, "0Pruned")
    
    os.makedirs(mftma_analysis_dir, exist_ok=True)
    pkl.dump(
        mftma_analysis_data,
        open(
            os.path.join(mftma_analysis_dir, f'{str(layer)}.pkl'), 'wb+'
        ),
    )

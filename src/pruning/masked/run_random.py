import os
import numpy as np

random_constraints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("Random constraints:", random_constraints)

for h_c in random_constraints:
    for n_c in random_constraints:
        cmd = f"python3 main.py --model_name bert-base-uncased-squad2 --ckpt_dir bert-base-uncased-squad2 --task_name squad_v2 --metric random --rnd_constraint_heads {h_c} --rnd_constraint_neurons {n_c}"
        print(cmd)
        os.system(cmd)


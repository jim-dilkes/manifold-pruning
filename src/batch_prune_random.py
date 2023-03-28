import os

random_constraints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("Random constraints:", random_constraints)

for c in random_constraints:
    cmd = f"python3 src/prune_masked.py --metric random --constraint_heads {c} --constraint_neurons {c}"
    print(cmd)
    os.system(cmd)

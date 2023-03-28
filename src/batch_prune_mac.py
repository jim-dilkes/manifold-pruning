import os

mac_constraints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("MAC constraints:", mac_constraints)

for c in mac_constraints:
    cmd = f"python3 src/prune_masked.py --metric mac --constraint {c}"
    print(cmd)
    os.system(cmd)

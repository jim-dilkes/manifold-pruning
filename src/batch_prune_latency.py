import os

latency_constraints = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("Latency constraints:", latency_constraints)

for c in latency_constraints:
    cmd = f"python3 src/prune_masked.py --mha_lut models/luts/bert-base-uncased-squad2/mha_lut.pt --ffn_lut models/luts/bert-base-uncased-squad2/ffn_lut.pt --metric latency --constraint {c}"
    print(cmd)
    os.system(cmd)

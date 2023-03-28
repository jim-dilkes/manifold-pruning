import os
import numpy as np

latency_constraints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("Latency constraints:", latency_constraints)

for c in latency_constraints:
    cmd = f"python3 main.py --model_name bert-base-uncased-squad2 --ckpt_dir bert-base-uncased-squad2 --task_name squad_v2 --mha_lut outputs/mha_lut.pt --ffn_lut outputs/ffn_lut.pt --metric latency --constraint {c}"
    print(cmd)
    os.system(cmd)


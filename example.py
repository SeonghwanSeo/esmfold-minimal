import time

import biotite.structure.io as bsio
import torch

from esmfold import load_esmfold

print("Loading ESMFold model...")
model = load_esmfold().eval().cuda()

print("Model loaded. Running inference...")
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

st = time.time()
with torch.no_grad():
    output = model.infer_pdb(sequence)
end = time.time()
print(f"Inference completed in {end - st:.2f} seconds.")

with open("result.pdb", "w") as f:
    f.write(output)

struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
print(struct.b_factor.mean())  # this will be the pLDDT
# 88.3

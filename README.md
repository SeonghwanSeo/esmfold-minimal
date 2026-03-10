# Easy-to-install ESMFold.

This repository provides easy-to-install ESMFold models for protein structure prediction, without complicated openfold dependencies.
This repository is a fork of [ESM](https://github.com/facebookresearch/esm/tree/main) by Facebook Research, allowing users to easily install and use ESMFold for protein structure prediction.

I note that this implementation is operated with [cuEquivariance kernel](https://docs.nvidia.com/cuda/cuequivariance/).

## Installation

```bash
# Install with pip
pip install git+https://github.com/SeonghwanSeo/esmfold-minimal
# Optional: for biotite dependency
pip install biotite
```


## Usage

Our implementation also provides similar python and command line interfaces as the original ESMFold.

### Python

```python
import biotite.structure.io as bsio
import torch

from esmfold import load_esmfold

print("Loading ESMFold model...")
model = load_esmfold().eval().cuda()

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

print("Model loaded. Running inference...")
with torch.no_grad():
    output = model.infer_pdb(sequence)

print("Inference complete. Saving output...")
with open("result.pdb", "w") as f:
    f.write(output)

# Require additional biotite dependency
struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
print('Predicted LDDT:', struct.b_factor.mean().item())
# 88.3
```

### Command Line

```bash
run-esmfold -i /path/to/input.fasta -o /path/to/out/pdb/
```

```
usage: run-esmfold [-h] -i FASTA -o PDB [--num-recycles NUM_RECYCLES]
                [--max-tokens-per-batch MAX_TOKENS_PER_BATCH]
                [--chunk-size CHUNK_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -i FASTA, --fasta FASTA
                        Path to input FASTA file
  -o PDB, --pdb PDB     Path to output PDB directory
  --num-recycles NUM_RECYCLES
                        Number of recycles to run. Defaults to number used in
                        training (4).
  --max-tokens-per-batch MAX_TOKENS_PER_BATCH
                        Maximum number of tokens per gpu forward-pass. This
                        will group shorter sequences together for batched
                        prediction. Lowering this can help with out of memory
                        issues, if these occur on short sequences.
  --chunk-size CHUNK_SIZE
                        Chunks axial attention computation to reduce memory
                        usage from O(L^2) to O(L). Equivalent to running a for
                        loop over chunks of of each dimension. Lower values
                        will result in lower memory usage at the cost of
                        speed. Recommended values: 128, 64, 32. Default: None.
```

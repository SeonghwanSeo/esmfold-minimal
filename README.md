# Easy-to-install ESMFold.

This is a performance-optimized, easy-to-install version of [ESMFold](https://github.com/facebookresearch/esm/tree/main) (Facebook Research) for protein structure prediction.

**Key features:**
* 📦 **Minimal Dependencies:** No complicated openfold dependencies.
* 🚀 **High Performance:** Supports cuEquivariance kernels for optimized triangle attention and multiplicative update operations.
* ⚡ **Fast Inference:** Native support for bfloat16 (bf16) precision to reduce memory usage and speed up prediction on modern GPUs.

## Installation

```bash
# Install with pip
pip install "esmfold @ git+https://github.com/SeonghwanSeo/esmfold-minimal"
# For cuEquivariance kernel support, install with the `cu-equivariance` extra
pip install "esmfold[cueq] @ git+https://github.com/SeonghwanSeo/esmfold-minimal"

# Optional: for biotite dependency to run the example script.
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

model.set_cuequivariance_kernel(True)  # Enable cuEquivariance kernel 
model.set_precision("bf16")  # Use bf16 precision for faster inference

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

print("Model loaded. Running inference...")
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
usage: run-esmfold [-h] -i INPUT_FASTA -o OUT_DIR [-m MODEL_DIR] 
                [--num-recycles NUM_RECYCLES] 
                [--max-tokens-per-batch MAX_TOKENS_PER_BATCH] [--no-kernel] 
                [--precision {fp32,bf16}] [--chunk-size CHUNK_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FASTA, --input-fasta INPUT_FASTA
                        Path to input FASTA file
  -o OUT_DIR, --out-dir OUT_DIR
                        Path to output PDB directory  
  --num-recycles NUM_RECYCLES
                        Number of recycles to run. Defaults to number used in
                        training (4).
  --max-tokens-per-batch MAX_TOKENS_PER_BATCH
                        Maximum number of tokens per gpu forward-pass. This
                        will group shorter sequences together for batched
                        prediction. Lowering this can help with out of memory
                        issues, if these occur on short sequences.
  --no-kernel           Whether to disable the cuequivariance kernel
  --precision {fp32,bf16}
                        Precision to run the model in. `bf16` can reduce memory 
                        usage and speed up inference on supported hardware.
```

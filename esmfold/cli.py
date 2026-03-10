import argparse
import logging
import sys
import typing as T
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def create_batched_sequence_datasest(
    sequences: list[tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[tuple[list[str], list[str]], None, None]:
    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-fasta",
        help="Path to input FASTA file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        help="Path to output PDB directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        help="Parent path to Pretrained ESM data directory. ",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1536,
        help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
        "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
        "short sequences.",
    )
    # TODO: test this option works well
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). "
        "Equivalent to running a for loop over chunks of of each dimension. Lower values will "
        "result in lower memory usage at the cost of speed. Recommended values: 128, 64, 32. "
        "Default: None.",
    )
    return parser


def run(args):
    from timeit import default_timer as timer

    import torch

    from esmfold import load_esmfold
    from esmfold.esm.data import read_fasta

    input_fasta = Path(args.input_fasta)
    out_dir = Path(args.out_dir)

    if not input_fasta.exists():
        raise FileNotFoundError(f"Input fasta file {input_fasta} does not exist.")

    # Read fasta and sort sequences by length
    logger.info(f"Reading sequences from {input_fasta}")
    all_sequences = sorted(
        read_fasta(input_fasta), key=lambda header_seq: len(header_seq[1])
    )
    logger.info(
        f"Loaded {len(all_sequences)} sequences, with lengths from {len(all_sequences[0][1])} to {len(all_sequences[-1][1])}."
    )

    if out_dir.exists():
        # if output directory is not empty, assume work has already been done and skip prediction.
        if not out_dir.is_dir():
            raise NotADirectoryError(f"Output path {out_dir} is not a directory.")
        filtered_sequences = []
        for header, seq in all_sequences:
            output_file = out_dir / f"{header}.pdb"
            if not output_file.exists():
                filtered_sequences.append((header, seq))
        if len(filtered_sequences) < len(all_sequences):
            logger.info(
                f"Output directory {out_dir} already contains {len(all_sequences) - len(filtered_sequences)} PDB files. "
                f"Predicting structures for remaining {len(filtered_sequences)} sequences."
            )
        all_sequences = filtered_sequences
    else:
        out_dir.mkdir(parents=True)

    # Use pre-downloaded ESM weights from model_pth.
    logger.info("Loading model")
    if args.model_dir is not None:
        # if pretrained model path is available
        torch.hub.set_dir(args.model_dir)

    model = load_esmfold().eval().cuda()
    model.set_chunk_size(args.chunk_size)
    logger.info("Starting Predictions")
    batched_sequences = create_batched_sequence_datasest(
        all_sequences, args.max_tokens_per_batch
    )

    num_completed = 0
    num_sequences = len(all_sequences)
    for headers, sequences in batched_sequences:
        start = timer()
        try:
            output = model.infer(sequences, num_recycles=args.num_recycles)
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                if len(sequences) > 1:
                    logger.error(
                        f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                        "Try lowering `--max-tokens-per-batch`."
                    )
                else:
                    # If the batch size is already 1, all the remaining sequences will also fail.
                    # So we exit the loop and skip the remaining sequences.
                    logger.error(
                        f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}. "
                        "Try lowering `--chunk-size` to reduce memory usage."
                    )
                return
            raise e

        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        tottime = timer() - start
        time_string = f"{tottime / len(headers):0.1f}s"
        if len(sequences) > 1:
            time_string = time_string + f" (amortized, batch size {len(sequences)})"
        for header, seq, pdb_string, mean_plddt, ptm in zip(
            headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        ):
            output_file = out_dir / f"{header}.pdb"
            output_file.write_text(pdb_string)
            num_completed += 1
            logger.info(
                f"Predicted structure for {header} with length {len(seq)}, "
                f"pLDDT {mean_plddt:0.1f}, "
                f"pTM {ptm:0.3f} in {time_string}. "
                f"{num_completed} / {num_sequences} completed."
            )


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

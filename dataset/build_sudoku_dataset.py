from typing import Optional, List
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from .common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"
    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    """
    Performs random transformations that preserve Sudoku validity.
    """
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # Shuffle the 3 bands (3 rows each) and shuffle rows within each band.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping.
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        # Apply the position mapping and copy to ensure memory contiguity
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)

def convert_subset(set_name: str, config: DataProcessConfig):
    """
    Downloads raw CSV data and converts it to fixed-shape .npy files for TPU.
    """
    inputs = []
    labels = []
    
    print(f"Downloading and reading {set_name} data from HuggingFace...")
    csv_path = hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset")
    
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                # Standard Sudoku is always 81 characters
                assert len(q) == 81 and len(a) == 81
                
                # Convert string representation to 9x9 uint8 arrays
                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    # Subsampling logic for training set
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Augmentation
    num_augments = config.num_aug if set_name == "train" else 0
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    print(f"Processing and augmenting {set_name} puzzles...")
    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            # Standard Sudoku uses a dummy identifier 0
            results["puzzle_identifiers"].append(0)
            
        results["group_indices"].append(puzzle_id)
        
    # Helper to convert list of arrays into a single TPU-optimized NumPy block
    def _seq_to_numpy(seq):
        # TPU OPTIMIZATION: Use int32 explicitly. 
        # XLA handles 32-bit integers faster and uses less HBM memory than 64-bit.
        arr = np.concatenate(seq).reshape(len(seq), -1).astype(np.int32)
        
        assert np.all((arr >= 0) & (arr <= 9))
        # Add 1 to shift range to [1, 10]. 0 is strictly reserved for PADDING.
        return arr + 1
    
    print(f"Finalizing {set_name} tensors...")
    final_results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Generate Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,              # Static shape for 9x9 Sudoku
        vocab_size=11,           # 0 (Pad) + 1-10 (Digits)
        pad_id=0,
        ignore_label_id=0,       # Targets matching the pad ID are ignored
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(final_results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(final_results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save logic
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    print(f"Subset {set_name} saved to {save_dir}")

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """
    Main entry point for dataset generation.
    """
    convert_subset("train", config)
    convert_subset("test", config)
    
    # Save identifiers helper
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    print("Pre-processing complete.")

if __name__ == "__main__":
    cli()
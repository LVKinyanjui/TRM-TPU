import os
import json
import math
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic
import torch
from torch.utils.data import IterableDataset, get_worker_info
from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

# ... (keep _sample_batch as it is) ...

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        # ... (Keep the metadata merging logic exactly as you have it) ...
        self.config = config
        self.split = split
        # (Assuming metadata loading logic here)
        
        # Calculate local batch size
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas
        self._data = None
        self._iters = 0

    def _lazy_load_dataset(self):
        # ... (Keep your existing _lazy_load_dataset logic) ...
        pass

    def _collate_batch(self, batch):
        """
        Ensures batch is ALWAYS (local_batch_size, seq_len)
        """
        # 1. Convert to int32
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # 2. Convert ignore labels
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # 3. STRICT PADDING
        # We check the actual size of the first dimension
        current_batch_size = batch["puzzle_identifiers"].shape[0]
        
        if current_batch_size < self.local_batch_size:
            pad_size = self.local_batch_size - current_batch_size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            # Pad the first dimension (batch dimension) to local_batch_size
            batch = {
                k: np.pad(
                    v, 
                    ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), 
                    mode='constant', 
                    constant_values=pad_values[k]
                ) for k, v in batch.items()
            }

        # 4. Final cast to Torch
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_train(self):
        worker_info = get_worker_info()
        
        for set_name, dataset in self._data.items():
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            # Shuffle groups
            group_order = np.concatenate([
                rng.permutation(dataset["group_indices"].size - 1) 
                for _ in range(self.config.epochs_per_iter)
            ])

            # --- WORKER SHARDING ---
            # If num_workers > 1, each worker takes a slice of the groups
            if worker_info is not None:
                per_worker = int(math.ceil(len(group_order) / float(worker_info.num_workers)))
                iter_start = worker_info.id * per_worker
                iter_end = min(iter_start + per_worker, len(group_order))
                group_order = group_order[iter_start:iter_end]

            start_index = 0
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Ensure we have a full global batch before splitting
                if batch_puzzle_indices.size < self.config.global_batch_size:
                    # In training, we drop the last incomplete batch to keep shapes 100% consistent
                    break

                # Slice for this specific TPU rank
                rank_start = self.config.rank * self.local_batch_size
                rank_end = rank_start + self.local_batch_size
                
                b_idx = batch_indices[rank_start:rank_end]
                p_idx = batch_puzzle_indices[rank_start:rank_end]

                batch = self._collate_batch({
                    "inputs": dataset["inputs"][b_idx],
                    "labels": dataset["labels"][b_idx],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][p_idx]
                })

                yield set_name, batch, self.config.global_batch_size

    def _iter_test(self):
        worker_info = get_worker_info()
        for set_name, dataset in self._data.items():
            total_examples = len(dataset["inputs"])
            
            # Divide total test examples across TPU ranks first
            # Then divide that rank's work across CPU workers
            indices = np.arange(total_examples)
            
            # Simple sharding logic
            if worker_info is not None:
                # This is a bit complex: we need to ensure rank-sharding AND worker-sharding
                # The easiest way: only use 1 worker for test sets to preserve order,
                # OR shard indices here:
                indices = indices[worker_info.id::worker_info.num_workers]

            start_index = 0
            while start_index < len(indices):
                # Grab a batch of indices for this specific rank
                # (Note: In test mode, we usually want to process every example exactly once)
                end_index = min(start_index + self.local_batch_size, len(indices))
                batch_indices = indices[start_index:end_index]
                
                # Puzzle IDs for these indices
                p_idx = np.searchsorted(dataset["puzzle_indices"], batch_indices, side="right") - 1
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][p_idx]
                })

                # Always yield local_batch_size so TPU doesn't recompile
                yield set_name, batch, self.local_batch_size
                start_index += self.local_batch_size

    def __iter__(self):
        # REMOVED the assert num_workers == 1
        self._lazy_load_dataset()
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
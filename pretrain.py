import os
import math
import copy
import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
from torch.utils.data import DataLoader

# --- TPU SPECIFIC IMPORTS ---
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
# ----------------------------

# [Import your local classes - assumed same as before]
from adam_atan2 import AdamATan2 # Use the Pure PyTorch version provided earlier
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
# ... (keep other imports and Config classes from original) ...

def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=4, # Increased for TPU
        prefetch_factor=4,
        persistent_workers=True
    )
    return dataloader, dataset.metadata

def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    device = xm.xla_device() # Get TPU core
    
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    # Instantiate on TPU
    model = model_cls(model_cfg).to(device)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)

    # NOTE: torch.compile(model) is generally not used on TPU 
    # as XLA is a compiler itself. Skip for now.

    if rank == 0:
        load_checkpoint(model, config)

    # Wait for rank 0 to finish loading
    xm.rendezvous('init_model')

    # Optimizers (Same logic, but use the TPU-compatible versions we discussed)
    # ... [Optimizer setup logic] ...
    
    return model, optimizers, optimizer_lrs

def train_batch(config, train_state, batch, global_batch_size, rank, world_size):
    train_state.step += 1
    device = xm.xla_device()

    # TPU Note: Batch is already on device because we use MpDeviceLoader
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    ((1 / global_batch_size) * loss).backward()

    # TPU Optimizer Step
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        
        # xm.optimizer_step handles the all-reduce (gradient syncing)
        xm.optimizer_step(optim)
        optim.zero_grad()

    # CRITICAL: This tells TPU to actually execute the graph
    xm.mark_step()

    if len(metrics) and rank == 0:
        # Reconstruct metrics logic (xm.all_reduce could be used here if needed)
        # For simplicity, returning metrics as is; note that xm.all_reduce is needed 
        # for mathematically accurate global logging.
        return {f"train/{k}": v.item() for k, v in metrics.items()}

def _train_worker(rank, hydra_config):
    """
    This is the function that runs on EACH of the 8 TPU cores.
    """
    # 1. Setup TPU Environment
    device = xm.xla_device()
    WORLD_SIZE = xm.xrt_world_size()
    RANK = xm.get_ordinal() # This is the global rank (0-7)

    # 2. Load Config (Only rank 0 logs to wandb)
    config = PretrainConfig(**hydra_config)
    
    # 3. Data Loaders
    train_epochs_per_iter = config.eval_interval or config.epochs
    raw_loader, train_meta = create_dataloader(config, "train", RANK, WORLD_SIZE, epochs_per_iter=train_epochs_per_iter)
    
    # Wrap with MpDeviceLoader: This sends data to TPU in the background
    train_loader = pl.MpDeviceLoader(raw_loader, device)

    # 4. Model & State
    train_state = init_train_state(config, train_meta, RANK, WORLD_SIZE)
    
    if RANK == 0:
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump())
        pbar = tqdm.tqdm(total=train_state.total_steps)

    # 5. Training Loop
    total_iters = config.epochs // train_epochs_per_iter
    for _iter_id in range(total_iters):
        train_state.model.train()
        
        # In TPU XLA, the loader handles epoch-shuffling via MpDeviceLoader
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, RANK, WORLD_SIZE)
            
            if RANK == 0 and metrics:
                wandb.log(metrics, step=train_state.step)
                pbar.update(1)

        # 6. Evaluation (Simplified for TPU)
        if _iter_id >= config.min_eval_interval:
            xm.mark_step() # Ensure training is finished before eval starts
            # ... call evaluate() logic here ...
            # Inside evaluate, remember to wrap eval_loader in MpDeviceLoader
            xm.mark_step()

    xm.rendezvous('finalize')
    if RANK == 0:
        wandb.finish()

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    # Convert Hydra DictConfig to a plain dict so it can be pickled by xmp.spawn
    config_dict = OmegaConf.to_container(hydra_config, resolve=True)
    
    # Start 8 processes (one per TPU core)
    # nprocs=8 is standard for TPU v5e-8
    xmp.spawn(_train_worker, args=(config_dict,), nprocs=8, start_method='fork')

if __name__ == "__main__":
    main()
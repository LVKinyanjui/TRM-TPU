# --- NEW IMPORTS ---
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
# -------------------

# ... [Keep your Config classes and Imports as they are] ...

def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # ... [Keep config setup] ...
    
    # TPU change: Use XLA device
    device = xm.xla_device()

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model: nn.Module = model_cls(model_cfg).to(device)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
    
    # TPU Change: torch.compile(model) is generally not needed on TPU 
    # because XLA is already a compiler. If you use it, use backend="openxla"
    # For now, let's disable it for stability.

    if rank == 0:
        load_checkpoint(model, config)

    # TPU Change: XLA handles broadcasting automatically, but if you need a manual one:
    xm.rendezvous('model_init') 

    # Optimizers (Ensure these are standard PyTorch or XLA-friendly)
    # Note: CastedSparseEmbeddingSignSGD_Distributed must be pure PyTorch to work on XLA
    # ... [Optimizer Init Logic] ...
    
    return model, optimizers, optimizer_lrs

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    device = xm.xla_device()

    # TPU Change: Batch is already on device if using MpDeviceLoader
    # batch = {k: v.to(device) for k, v in batch.items()}

    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    scaled_loss = (1 / global_batch_size) * loss
    scaled_loss.backward()

    # TPU Change: xm.optimizer_step handles the All-Reduce across TPU cores
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        
        # This is the "TPU way" to step and sync
        xm.optimizer_step(optim)
        optim.zero_grad()

    # TPU Change: Trigger the graph execution
    xm.mark_step()

    # Reduce metrics
    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        
        # TPU Change: xm.all_reduce
        metric_values = xm.all_reduce(xm.REDUCE_SUM, metric_values)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {f"train/{k}": metric_values[i] / (global_batch_size if k.endswith("loss") else 1) for i, k in enumerate(metric_keys)}
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

def evaluate(config, train_state, eval_loader, eval_metadata, evaluators, rank, world_size, cpu_group):
    # Wrap the loader for TPU
    device = xm.xla_device()
    tpu_eval_loader = pl.MpDeviceLoader(eval_loader, device)
    
    with torch.no_grad():
        for set_name, batch, global_batch_size in tpu_eval_loader:
            # TPU Change: Logic is similar to train, but use xm.mark_step() 
            # after the batch loop or periodically to clear the graph.
            # ... [Inference logic] ...
            xm.mark_step() 
    # ...
    return reduced_metrics

def save_train_state(config, train_state):
    if config.checkpoint_path is None: return
    # TPU Change: xm.save ensures tensors are moved to CPU and un-sharded
    save_obj = train_state.model.state_dict()
    xm.save(save_obj, os.path.join(config.checkpoint_path, f"step_{train_state.step}"))

# --- MULTIPROCESSING LAUNCHER ---
def _mp_fn(index, hydra_config):
    # This replaces your 'launch' logic for TPU
    # index is the LOCAL_RANK (0-7 on a v5e-8)
    
    # TPU v5e Setup
    device = xm.xla_device()
    WORLD_SIZE = xm.xrt_world_size()
    RANK = xm.get_ordinal()
    
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.manual_seed(config.seed + RANK)

    # Loaders
    raw_train_loader, train_metadata = create_dataloader(...)
    # Wrap for TPU
    train_loader = pl.MpDeviceLoader(raw_train_loader, device)

    # ... [Init train state, EMA, etc] ...

    for _iter_id in range(total_iters):
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, RANK, WORLD_SIZE)
            # ... [Logging] ...

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    # Launch 8 processes (one per TPU core on a v5e-8)
    xmp.spawn(_mp_fn, args=(hydra_config,), nprocs=8, start_method='fork')

if __name__ == "__main__":
    launch()

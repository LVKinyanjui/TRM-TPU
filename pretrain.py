import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import wandb
import tqdm

def train_batch(config, train_state, batch, global_batch_size, rank, world_size):
    """
    Executes a single training step on a TPU core.
    """
    train_state.model.train()
    
    # 1. Forward Pass
    # Batch is already on device thanks to pl.MpDeviceLoader
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    # 2. Backward Pass
    # We scale by global batch size because XLA optimizer_step averages gradients across cores
    (loss / world_size).backward()

    # 3. Optimizer Step
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        
        # xm.optimizer_step handles the 'All-Reduce' of gradients across the 8 cores
        xm.optimizer_step(optim)
        optim.zero_grad()

    # 4. Trigger TPU Execution
    # This is critical to prevent the graph from growing infinitely in memory
    xm.mark_step()

    # 5. Global Metric Reduction (for logging)
    # We reduce the metrics across all 8 TPU cores so Rank 0 sees the global average
    if len(metrics) > 0:
        metric_keys = sorted(metrics.keys())
        metric_values = torch.stack([metrics[k].to(torch.float32) for k in metric_keys])
        
        # Average metrics across all TPU cores
        reduced_metrics = xm.mesh_reduce('metrics_reduce', metric_values, torch.mean)
        
        if rank == 0:
            return {k: v.item() for k, v in zip(metric_keys, reduced_metrics)}
    return None

def evaluate(config, train_state, eval_loader, rank, world_size):
    """
    Executes evaluation across all TPU cores.
    """
    train_state.model.eval()
    device = xm.xla_device()
    
    # Wrap the loader for TPU parallelism
    tpu_eval_loader = pl.MpDeviceLoader(eval_loader, device)
    
    all_metrics = []
    
    with torch.no_grad():
        for set_name, batch, _ in tpu_eval_loader:
            carry = train_state.model.initial_carry(batch)
            # Standard inference loop
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=set(config.eval_save_outputs)
                )
                if all_finish: break
            
            # Record metrics (summing them for later averaging)
            metric_keys = sorted(metrics.keys())
            all_metrics.append(torch.stack([metrics[k].to(torch.float32) for k in metric_keys]))
            
            # Periodically mark step to clear the TPU graph during long evaluations
            xm.mark_step()

    # Aggregate metrics across all batches and all TPU cores
    if all_metrics:
        total_metrics = torch.stack(all_metrics).mean(dim=0)
        global_metrics = xm.mesh_reduce('eval_reduce', total_metrics, torch.mean)
        
        if rank == 0:
            return {f"eval/{k}": v.item() for k, v in zip(metric_keys, global_metrics)}
    return None

def _train_worker(rank, hydra_config_dict):
    """
    Main loop running on each of the 8 TPU cores.
    """
    # Initialize TPU core
    device = xm.xla_device()
    WORLD_SIZE = xm.xrt_world_size()
    RANK = xm.get_ordinal()
    
    # Reconstruct config
    config = PretrainConfig(**hydra_config_dict)
    
    # 1. Dataset & Loaders
    train_epochs_per_iter = config.eval_interval or config.epochs
    raw_train_loader, train_metadata = create_dataloader(
        config, "train", RANK, WORLD_SIZE, epochs_per_iter=train_epochs_per_iter
    )
    # Background loader that pre-fetches to TPU memory
    train_loader = pl.MpDeviceLoader(raw_train_loader, device)

    # 2. Model & Optimizers
    train_state = init_train_state(config, train_metadata, RANK, WORLD_SIZE)
    
    # 3. Rank 0 Setup (Logging & EMA)
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump())
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
    
    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # 4. The Training Loop
    total_iters = config.epochs // train_epochs_per_iter
    for iter_id in range(total_iters):
        
        # --- TRAINING PHASE ---
        for set_name, batch, _ in train_loader:
            metrics = train_batch(config, train_state, batch, config.global_batch_size, RANK, WORLD_SIZE)
            
            if RANK == 0 and metrics:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(1)
            
            if config.ema:
                ema_helper.update(train_state.model)
                # Note: EMA updates are local, xm.mark_step in train_batch handles sync

        # --- EVALUATION PHASE ---
        if iter_id >= config.min_eval_interval:
            # Synchronize all cores before starting eval
            xm.rendezvous('eval_start')
            
            # If using EMA, create a copy for evaluation
            if config.ema:
                eval_model = ema_helper.ema_copy(train_state.model)
                eval_state = copy.copy(train_state)
                eval_state.model = eval_model
            else:
                eval_state = train_state
            
            eval_metrics = evaluate(config, eval_state, eval_loader, RANK, WORLD_SIZE)
            
            if RANK == 0 and eval_metrics:
                wandb.log(eval_metrics, step=train_state.step)
                
            # Checkpoint (xm.save is mandatory for TPU)
            if RANK == 0 and (config.checkpoint_every_eval or (iter_id == total_iters - 1)):
                save_path = os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt")
                xm.save(train_state.model.state_dict(), save_path)
            
            xm.rendezvous('eval_end')

    if RANK == 0:
        wandb.finish()
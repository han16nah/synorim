import argparse
import bdb
import importlib
import pdb
import shutil
import traceback
from pathlib import Path

import torch
import omegaconf
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import exp

import wandb
import ray
from ray import tune, train
from ray.air.integrations.wandb import WandbLoggerCallback


def train_epoch(net_model, train_loader, optimizer, scheduler, writer):
    global global_step
    # Initialize wandb
    wandb = setup_wandb(net_model.hparams)

    net_model.train()
    net_model.hparams.is_training = True

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, data in enumerate(pbar):
        data = exp.to_target_device(data, args.device)
        optimizer.zero_grad()
        loss = net_model.training_step(data, batch_idx)
        loss.backward()
        net_model.on_after_backward()
        optimizer.step()
        scheduler.step()
        net_model.log('learning_rate', scheduler.get_last_lr()[0])
        wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
        pbar.set_postfix_str(f"Loss = {loss.item():.2f}")
        wandb.log({"train_loss": loss.item()})
        tune.report(train_loss=loss.item(), global_step=global_step)

        net_model.write_log(writer, global_step)
        global_step += 1


def validate_epoch(net_model, val_loader, optimizer, writer, epoch_idx):
    global metric_val_best

    net_model.eval()
    net_model.hparams.is_training = False

    pbar = tqdm(val_loader, desc='Validation')
    for batch_idx, data in enumerate(pbar):
        data = exp.to_target_device(data, args.device)
        with torch.no_grad():
            net_model.validation_step(data, batch_idx)

    log = net_model.write_log(writer, global_step)
    metric_val = log['val_loss']

    model_state = {
        'state_dict': net_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch_idx, 'val_loss': metric_val
    }
    wandb.log(model_state)

    if metric_val < metric_val_best:
        metric_val_best = metric_val
        torch.save(model_state, train_log_dir / f"best.pth")
    torch.save(model_state, train_log_dir / f"newest.pth")


def train_example(config_tunable):
    # Train and validate within a protected loop.
    net_module = importlib.import_module("models." + model_args.model).Model
    net_model = net_module(model_args)

    print(" >>>> ======= MODEL HYPER-PARAMETERS ======= <<<< ")
    print(OmegaConf.to_yaml(net_model.hparams, resolve=True))
    print("Save Directory is in:", train_log_dir)
    print(" >>>> ====================================== <<<< ")

    # Copy the model definition and config.
    shutil.copy(f"{model_dir}/{model_args.model.replace('.', '/')}.py", train_log_dir / "model.py")
    OmegaConf.save(model_args, train_log_dir / "config.yaml")

    # Load dataset
    train_loader = net_model.train_dataloader()
    val_loader = net_model.val_dataloader()

    # Load training specs
    optimizers, schedulers = net_model.configure_optimizers()
    assert len(optimizers) == 1 and len(schedulers) == 1
    optimizer, scheduler = optimizers[0], schedulers[0]
    assert scheduler['interval'] == 'step'
    scheduler = scheduler['scheduler']

    # TensorboardX writer
    tb_logdir = train_log_dir / "tensorboard"
    tb_logdir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    # Move to target device
    args.device = torch.device(args.device)
    net_model = exp.to_target_device(net_model, args.device)
    net_model.device = args.device

    global_step = 0
    metric_val_best = 1e6
    try:
        for epoch_idx in range(100):
            # update net_module.hparams with the values from config_tunable
            for key, value in config_tunable.items():
                setattr(net_model.hparams, key, value)
            train_epoch(net_model, train_loader, optimizer, scheduler, writer)
            validate_epoch(net_model, val_loader, optimizer, writer, epoch_idx)
    except Exception as ex:
        if not isinstance(ex, bdb.BdbQuit):
            traceback.print_exc()
            pdb.post_mortem(ex.__traceback__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synorim Training script')
    parser.add_argument('config', type=str, help='Path to the config file.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to run on.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    args = parser.parse_args()

    exp.seed_everything(0)

    model_args = exp.parse_config_yaml(Path(args.config))
    # make checkpoint path absolute
    try:
        model_args.desc_checkpoint = Path(model_args.desc_checkpoint).expanduser().resolve().as_posix()
    except (KeyError, omegaconf.errors.ConfigAttributeError):
        pass
    print(model_args)
    try:
        model_args.train_kwargs.base_folder = Path(model_args.train_kwargs.base_folder).expanduser().resolve().as_posix()
    except (KeyError, omegaconf.errors.ConfigAttributeError) as e:
        print("Could not write base_folder to absolute path.")
        print(e)
        

    config_tunable = {
        "voxel_size": tune.grid_search(model_args.voxel_size)
    }

    if model_args.model == "basis_net":
        config_tunable.update({
        "gt_align_prob": tune.grid_search(model_args.gt_align_prob),
        "ctc_weight": tune.grid_search(model_args.ctc_weight),
        "smoothness_weight": tune.grid_search(model_args.smoothness_weight),
        "n_match_th": tune.grid_search(model_args.n_match_th),
    })

    train_log_dir = Path("/mnt/sds-hd/sd23k005/Hannah/synorim/out") / model_args.name
    train_log_dir.mkdir(exist_ok=True, parents=True)
    model_dir = (Path.cwd() / "models").as_posix()

    ray.init(_temp_dir="/gpfs/bwfor/home/hd/hd_hd/hd_wq452/tmp/ray")
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_example),
            resources={"cpu": 1, "gpu": 1}
        ),
        param_space=config_tunable,
        run_config=train.RunConfig(
            callbacks=[WandbLoggerCallback(project="synorim")]
        )
)
    tuner.fit()

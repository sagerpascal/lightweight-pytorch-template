import os
import shutil
import time
from pathlib import Path
import logging
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader import get_loaders
from losses.loss import get_loss
from metrics import get_metrics
from models.model import get_model
from tools.optimizers import get_optimizer, get_lr
from utils.ddp import setup, cleanup
from utils.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def setup_wandb(conf):
    """ Initialize logging with wandb.ai """
    # TODO: define project name
    run = wandb.init(project="<Project Name>", job_type='train')
    wandb.run.save()
    return run


def wandb_log_settings(conf, loader_train, loader_val):
    """ Log the configuration """
    add_logs = {
        'size training set': len(loader_train.dataset),
        'size validation set': len(loader_val.dataset),
    }

    wandb.config.update({**conf, **add_logs})


def wandb_log_epoch(n_epoch, lr, best_loss, train_logs, valid_logs):
    """ Log the metrics from an epoch """
    logs = {
        'epoch': n_epoch,
        'learning rate': lr,
        'smallest loss': best_loss,
    }
    for k, v in train_logs.items():
        logs[k + " train"] = v
    for k, v in valid_logs.items():
        logs[k + " valid"] = v
    wandb.log(logs)


def save_model(model, model_path, model_name, save_wandb=False):
    """ Save a model on the filesystem and on wandb.ai if available """
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if save_wandb:
        filename = 'model.pth'
        if os.path.exists(filename):
            os.remove(filename)
        torch.save(model.state_dict(), filename)
        wandb.save(filename)
        shutil.copy(filename, Path(model_path) / model_name)
    else:
        if os.path.exists(model_name):
            os.remove(model_name)
        torch.save(model.state_dict(), Path(model_path) / model_name)


def _save_logs(store, logs, mode, rank):
    for k, v in logs.items():
        store.set("{}:{}-{}".format(k, mode, rank), str(v))


def save_logs_in_store(store, rank, train_logs, valid_logs):
    """ Save the logs in a TCP-Store to synchronize across multiple distributed instances """
    _save_logs(store, train_logs, 'train', rank)
    _save_logs(store, valid_logs, 'valid', rank)


def _get_average_logs(conf, store, logs, mode):
    avg_logs = {}
    for k, v in logs.items():
        key, val = k.split(':')[0], 0.
        for rank in range(conf['env']['world_size']):
            val += float(store.get("{}:{}-{}".format(k, mode, rank)))
        avg_logs[key] = val / conf['env']['world_size']
    return avg_logs


def calculate_average_logs(conf, store, train_logs, valid_logs):
    """ Calculate the average metrics values across multiple distributed instances """
    train_logs_avg = _get_average_logs(conf, store, train_logs, 'train')
    valid_logs_avg = _get_average_logs(conf, store, valid_logs, 'valid')
    return train_logs_avg, valid_logs_avg


def train(rank=None, mport=None, store_port=None, world_size=None, conf=None):
    """ Run the training """

    is_main_process = not conf['env']['use_data_parallel'] or conf['env']['use_data_parallel'] and rank == 0

    if conf['env']['use_data_parallel']:
        torch.cuda.manual_seed_all(42)
        setup(mport, rank, world_size)
        logger.info("Running DDP on rank {}".format(rank))
        device = rank
        model = get_model(conf, device)
        store = dist.TCPStore("127.0.0.1",
                              port=store_port,
                              world_size=conf['env']['world_size'],
                              is_master=is_main_process,
                              )
        model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        device = conf['device']
        model = get_model(conf, device)
        model.to(device)

    loader_train, loader_val, _ = get_loaders(conf, device)
    loss = get_loss(conf)
    optimizer = get_optimizer(conf, model)
    metrics = get_metrics(conf, device)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=conf['lr_scheduler']['step_size'],
                                                   gamma=conf['lr_scheduler']['gamma'])

    train_epoch = TrainEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=is_main_process,
        conf=conf
    )

    valid_epoch = ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=is_main_process,
        conf=conf
    )

    wandb_run = None
    if conf['use_wandb'] and is_main_process:
        wandb_run = setup_wandb(conf)
        wandb_log_settings(conf, loader_train, loader_val)

    best_loss = 999999999999
    count_not_improved = 0

    for i in range(conf['train']['max_number_of_epochs']):

        if conf['env']['use_data_parallel']:
            loader_train.sampler.set_epoch(i)
            loader_val.sampler.set_epoch(i)
            dist.barrier()

        train_logs = train_epoch.run(loader_train, i)
        valid_logs = valid_epoch.run(loader_val, i)

        if conf['env']['use_data_parallel']:
            save_logs_in_store(store, rank, train_logs, valid_logs)
            dist.barrier()

        if is_main_process:
            if conf['env']['use_data_parallel']:
                train_logs, valid_logs = calculate_average_logs(conf, store, train_logs, valid_logs)

            if valid_logs['loss'] < best_loss:
                best_loss = valid_logs['loss']
                model_name = wandb.run.name if conf['use_wandb'] else 'tsc_acf'
                model_name = "{}.pth".format(model_name)
                model_path = '/workspace/data_pa/trained_models'

                save_model(model, model_path, model_name, save_wandb=conf['use_wandb'])
                logger.info("Model saved (loss={})".format(best_loss))
                count_not_improved = 0

                if conf['env']['use_data_parallel']:
                    model_fp = Path(model_path) / model_name
                    store.set("model_filename", str(model_fp.resolve()))
                    store.set("model_update_flag", str(True))

            else:
                count_not_improved += 1
                if conf['env']['use_data_parallel']:
                    store.set("model_update_flag", str(False))

            if conf['use_wandb']:
                wandb_log_epoch(i, get_lr(optimizer), best_loss, train_logs, valid_logs)

        if conf['env']['use_data_parallel']:
            dist.barrier()  # Other processes have to load model saved by process 0
            if not is_main_process and bool(store.get("model_update_flag")):
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                filename = store.get("model_filename").decode("utf-8")
                model.load_state_dict(torch.load(filename, map_location=map_location))

        if (i + 1) % conf['train']['backup_frequency'] == 0 and is_main_process:
            model_name = "{}-backup-{}.pth".format(wandb.run.name, i + 1) if conf[
                'use_wandb'] else 'model-backup-{}.pth'.format(i + 1)
            save_model(model, '/workspace/data_pa/trained_models', model_name, save_wandb=False)
            logger.info("Model saved as backup after {} epochs".format(i))

        if is_main_process and train_logs['loss'] < 0.0001 or conf['train'][
            'early_stopping'] and count_not_improved >= 5:
            logger.info("early stopping after {} epochs".format(i))
            if conf['env']['use_data_parallel']:
                # TODO: Fixme
                raise KeyboardInterrupt
            break

        if conf['lr_scheduler']['activate']:
            lr_scheduler.step()

    if wandb_run is not None:
        wandb_run.finish()

    dist.barrier()
    del store
    cleanup()

    if is_main_process:
        time.sleep(10)  # wait for other processes to terminate
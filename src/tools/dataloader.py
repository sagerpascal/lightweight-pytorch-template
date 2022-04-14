from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.collate import collate_fn
from datasets.dataset import BaseDataset


def _get_loader(conf, train_set, val_set, test_set, rank=None):
    """ Returns the dataloader according to the run configuration """

    if "cuda" in conf['device']:
        pin_memory = True
    else:
        pin_memory = False

    if conf['env']['use_data_parallel']:
        train_sampler = DistributedSampler(train_set,
                                           num_replicas=conf['env']['world_size'],
                                           rank=rank,
                                           shuffle=True)
        valid_sampler = DistributedSampler(val_set,
                                           num_replicas=conf['env']['world_size'],
                                           rank=rank,
                                           shuffle=False)
        if test_set is not None:
            test_sampler = DistributedSampler(test_set,
                                              num_replicas=conf['env']['world_size'],
                                              rank=rank,
                                              shuffle=False)
        else:
            test_sampler = None

        train_loader_shuffle, val_loader_shuffle, test_loader_shuffle = False, False, False

    else:
        train_sampler, valid_sampler, test_sampler = None, None, None
        train_loader_shuffle, val_loader_shuffle, test_loader_shuffle = True, False, False

    train_loader = DataLoader(
        train_set,
        batch_size=conf['train']['batch_size'],
        shuffle=train_loader_shuffle,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=conf['dataloader']['num_workers'],
        pin_memory=pin_memory,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=conf['train']['batch_size'],
        shuffle=val_loader_shuffle,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=conf['dataloader']['num_workers'],
        pin_memory=pin_memory,
        sampler=valid_sampler,
    )
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=conf['train']['batch_size'],
            shuffle=test_loader_shuffle,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=conf['dataloader']['num_workers'],
            pin_memory=pin_memory,
            sampler=test_sampler,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def get_loaders(conf, device):
    """ Returns the dataloader according to the run configuration """

    train_set = BaseDataset(conf, "train")
    val_set = BaseDataset(conf, "val")
    test_set = BaseDataset(conf, "test")
    return _get_loader(conf, train_set, val_set, test_set, rank=device)

import datetime
import os

import torch.distributed as dist


def setup(mport, rank, world_size):
    """ Setup Distributed Data Parallel """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = mport

    # initialize the process group
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=300), rank=rank, world_size=world_size)


def cleanup():
    """ Cleanup Distributed Data Parallel """
    dist.destroy_process_group()

import torch


def get_loss(conf):
    """ Returns the loss according to the run configuration """

    # Try to load PyTorch Basic Loss
    try:
        loss = getattr(torch.nn, conf['train']['loss'])
        return loss()
    except AttributeError:
        pass

    # load custom loss
    if conf['train']['loss'] == 'my_custom_loss':
        raise NotImplementedError()  # TODO

    else:
        raise AttributeError(f"Unknown loss: {conf['train']['loss']}")

from torch import optim


def get_lr(optimizer):
    """ Returns the learning rate of an optimizer """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_optimizer(conf, model):
    """ Returns the optimizer according to run configuration """
    if conf['optimizer']['type'] == 'adam':
        return optim.Adam(model.parameters(),
                          lr=conf['optimizer']['lr'],
                          weight_decay=conf['optimizer']['adam']['weight_decay'])
    
    elif conf['optimizer']['type'] == 'sgd':
        return optim.SGD(model.parameters(),
                         lr=conf['optimizer']['lr'],
                         momentum=conf['optimizer']['sgd']['momentum'])

    else:
        raise AttributeError("Unsupported optimizer in config file: {}".format(conf['optimizer']['type']))
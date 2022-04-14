import torchmetrics

# For other metrics see: https://torchmetrics.readthedocs.io/en/stable/references/modules.html

class Accuracy(torchmetrics.Accuracy):
    __name__ = 'Accuracy'


def get_metrics(conf, device):
    """ Returns the metrics used to evaluate the results """
    # TODO: add needed Metrics
    metrics = [
        Accuracy(),
    ]

    for m in metrics:
        m.to(device)

    return metrics

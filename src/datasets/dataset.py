import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, conf, mode, augmentation=None):
        raise NotImplementedError()

        self.conf = conf
        self.augmentation = augmentation
        self.mode = mode

        if mode == "train":
            self.data = None  # TODO

        elif mode == "valid":
            self.data = None  # TODO

        elif mode == "test":
            self.data = None  # TODO

    def __getitem__(self, item):
        x, y = self.data[item]

        if self.augmentation is not None:
            x, y = self.augmentation(x, y)

        return torch.as_tensor(x), torch.as_tensor(y)

    def __len__(self):
        return len(self.data)

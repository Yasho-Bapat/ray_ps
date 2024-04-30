import os
import torch
from torchvision import datasets, transforms
from filelock import FileLock

def get_data_loader():
    '''Safely downloads data. Returns training/validation set dataloader.'''
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # FileLock helps because multiple workers wanting to download the data can lead to unnecessary overwrites as DataLoader is not threadsafe.

    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            ),
            batch_size=128,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("~/data", train=False, transform=mnist_transforms),
            batch_size=128,
            shuffle=True,
        )
    return train_loader, test_loader
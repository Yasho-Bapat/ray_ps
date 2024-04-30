import os
import torch
from torchvision import datasets, transforms
from filelock import FileLock

'''This has general utility functions like downloading datasets from an online repository and the evaluate function.
In an implementation, general utility functions can be written here.'''
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


def evaluate(model, test_loader):
    #Evaluates the accuracy of the model on a validation dataset. The same function is used to evaluate the models when they are run on the individual machines and on the parameter server
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            if batch_idx * len(data) > 1024:
                break
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


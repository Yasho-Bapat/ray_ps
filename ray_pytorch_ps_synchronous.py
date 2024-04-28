import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
import argparse
import time
import ray


def get_data_loader():
    """Safely downloads data. Returns training/validation set dataloader."""
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
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
    """Evaluates the accuracy of the model on a validation dataset."""
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
    
class ConvNet(nn.Module):
    """Small ConvNet for MNIST."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
                
@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        '''standardized_gradients = []
        for gradient_zip in zip(*gradients):
            print("Shape of gradient array is: ",np.stack(gradient_zip).shape)
            sum_gradient = np.stack(gradient_zip).sum(axis=0)
            # Append the standardized gradient to the list
            standardized_gradients.append(mean_gradient)'''
            
        '''summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]'''
        '''avg_gradients = [
            np.stack(gradient_zip).mean(axis=0) for gradient_zip in zip(*gradients)
        ]'''
        num_workers = len(gradients)
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        #self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

'''class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        print(gradients)
        #summed_gradients = [np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients) # this is the aggregation part. FIX]
        summed_gradients = self.weighted_sum_gradients(gradients)
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()
        
    def weighted_sum_gradients(self, gradients):
        max_shape = np.max([np.array(grad).shape for grad in gradients], axis=0)

        # Pad gradients to the maximum shape
        padded_gradients = []
        for grad in gradients:
            grad_np = np.array(grad)
            pad_width = [(0, max_shape[i] - grad_np.shape[i]) for i in range(len(max_shape))]
            padded_grad = np.pad(grad_np, pad_width, mode='constant', constant_values=0)
            padded_gradients.append(padded_grad)

        # Calculate weighted sum
        num_gradients = len(padded_gradients)
        weighted_gradients = np.zeros_like(padded_gradients[0])
        for grad in padded_gradients:
            weighted_gradients += grad * (1 / num_gradients)  # Equal weights for now

        return weighted_gradients

    def get_weights(self):
        return self.model.get_weights()'''

@ray.remote
class DataWorker(object):
    def __init__(self):
        self.model = ConvNet()
        self.data_iterator = iter(get_data_loader()[0])
        self.train_loader = get_data_loader()[0]
        self.test_loader = get_data_loader()[1]
        
    def compute_gradients(self, weights):
        self.model.set_weights(weights)
        batch_size = self.train_loader.batch_size  # Batch size per iteration
        num_epochs = len(self.train_loader)  # Total number of batches per epoch
        print("Batch size on each worker is ",batch_size)
        print("Number of epochs is ",num_epochs)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        return self.model.get_gradients()
   


if __name__ == '__main__':

    '''num_workers = 2
    iterations = 5
    model = ConvNet()
    test_loader = get_data_loader()[1]


    ray.init(ignore_reinit_error=True)
    ps = ParameterServer.remote(1e-2)
    workers = [DataWorker.remote() for i in range(num_workers)]

    print("Running synchronous parameter server training.")
    current_weights = ps.get_weights.remote()
    accuracy = 0
    i = 0
    while accuracy < 90:
        #print(f'Iteration {i}')
        gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
        # Calculate update after all gradients are available.
        current_weights = ps.apply_gradients.remote(*gradients)
        # Evaluate the current model.
        model.set_weights(ray.get(current_weights))
        accuracy = evaluate(model, test_loader)
        print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))
        i += 1

    print("Final accuracy is {:.1f}.".format(accuracy))
    # Clean up Ray resources and processes before the next example.
    ray.shutdown()'''
    
    
    print("Running Asynchronous Parameter Server Training.")
    start_time = time.time()
    
    num_workers = 2 #try to keep this equal to the number of machines available
    model = ConvNet()
    test_loader = get_data_loader()[1]
        
    ray.init(ignore_reinit_error=True)
    ps = ParameterServer.remote(1e-2)
    workers = [DataWorker.remote() for i in range(num_workers)]
    
    current_weights = ps.get_weights.remote()

    gradients = {}
    for worker in workers:
        gradients[worker.compute_gradients.remote(current_weights)] = worker

    accuracy = 0
    i = 0
    while i<200:
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients.
        current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
        gradients[worker.compute_gradients.remote(current_weights)] = worker

        if i % 10 == 0:
            # Evaluate the current model after every 10 updates.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))
        i += 1

    print("Final accuracy is {:.1f}.".format(accuracy))
    end_time = time.time()
    print("Time of job on server ",end_time-start_time)

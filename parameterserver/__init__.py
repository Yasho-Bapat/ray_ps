import torch
import numpy as np
import model
import ray

'''This defines the Parameter Server class which has the function which performs gradient aggregation.'''
@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = model.ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        num_workers = len(gradients)
        summed_gradients = [np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)] # this is where the actual gradient aggregation occurs.
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()
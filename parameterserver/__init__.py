import torch
import modeldef
import ray

'''This defines the Parameter Server class which has the function which performs gradient aggregation.'''
@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = modeldef.ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        '''Takes gradients from models trained on individual machines and aggregates them here.'''

        import parameterserver.apply_gradients as ag
        return ag.apply_gradients(gradients)

    def get_weights(self):
        return self.model.get_weights()
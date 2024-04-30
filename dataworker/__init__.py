import torch.nn.functional as F
import ray
import model
import utilities

'''This describes the code that will run on the individual machines, taking the gradients from the PS as inputs.'''
@ray.remote
class DataWorker(object):
    def __init__(self):
        self.model = model.ConvNet()
        self.data_iterator = iter(utilities.get_data_loader()[0])
        self.train_loader = utilities.get_data_loader()[0]
        self.test_loader = utilities.get_data_loader()[1]

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

import ray
import modeldef
import utilities

'''This describes the code that will run on the individual machines, taking the gradients from the PS as inputs.'''
@ray.remote
class DataWorker(object):
    def __init__(self):
        self.model = modeldef.ConvNet()
        self.data_iterator = iter(utilities.get_data_loader()[0])
        self.train_loader = utilities.get_data_loader()[0]
        self.test_loader = utilities.get_data_loader()[1]

    def compute_gradients(self, weights):
        import dataworker.compute_gradients as cg
        return cg.compute_gradients(weights)

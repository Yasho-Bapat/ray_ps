import parameterserver
import numpy as np

def apply_gradients(*gradients):
    num_workers = len(gradients)
    summed_gradients = [np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)]  # this is where the actual gradient aggregation occurs.
    parameterserver.optimizer.zero_grad()
    parameterserver.model.set_gradients(summed_gradients)
    parameterserver.optimizer.step()
    return parameterserver.model.get_weights()
import torch.nn.functional as F
import dataworker
import utilities

def compute_gradients(weights):
    dataworker.model.set_weights(weights)
    batch_size = dataworker.train_loader.batch_size  # Batch size per iteration
    num_epochs = len(dataworker.train_loader)  # Total number of batches per epoch
    print("Batch size on each worker is ",batch_size)
    print("Number of epochs is ",num_epochs)
    try:
        data, target = next(dataworker.data_iterator)
    except StopIteration:  # When the epoch ends, start a new epoch.
        data_iterator = iter(utilities.get_data_loader()[0])
        data, target = next(data_iterator)
    data.model.zero_grad()
    output = dataworker.model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    return dataworker.model.get_gradients()
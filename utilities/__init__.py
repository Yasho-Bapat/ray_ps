'''This has general utility functions like downloading datasets from an online repository and the evaluate function.
In an implementation, general utility functions can be written here.'''
def get_data_loader():
    '''Safely downloads data. Returns training/validation set dataloader.'''
    import utilities.get_data_loader as get
    return get.get_data_loader()


def evaluate(model, test_loader):
    '''Evaluates the accuracy of passed modeldef. The same evaluate() method is used to determine the accuracy of the modeldef after
    aggregation and when trained on individual machines.'''

    import utilities.eval as eval
    return eval.evaluate(model, test_loader)
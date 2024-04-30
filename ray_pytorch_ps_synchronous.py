import time
import ray
import model, parameterserver, dataworker, utilities

'''This example outlines synchronous training. That means that the gradients are aggregated iteratively once they are all available'''


# Differences between synchronous and asynchronous are in the main function. Both use the same aggregation function (apply_gradients). What's different is how and when the aggregation function is called.


if __name__ == '__main__':

    num_workers = 2  # try to keep this equal to the number of machines available
    iterations = 200  # keep this number high. it is variable, dependent upon the model you are running.
    model = model.ConvNet()
    test_loader = utilities.get_data_loader()[1]

    print("Running Synchronous Parameter Server Training.")
    start_time = time.time() # start measuring time

    ray.init(ignore_reinit_error=True) # reinitialization can throw an error if the ignore_reinit_error flag is set to False (default)
    ps = parameterserver.ParameterServer.remote(1e-2)
    workers = [dataworker.DataWorker.remote() for i in range(num_workers)]

    current_weights = ps.get_weights.remote()
    accuracy = 0
    for i in range(iterations):
        gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
        # Calculate update after all gradients are available.
        current_weights = ps.apply_gradients.remote(*gradients)
        # Evaluate the current model.
        model.set_weights(ray.get(current_weights))
        accuracy = utilities.evaluate(model, test_loader)
        print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))
    end_time = time.time() # stop measuring time
    print("Time of job on server ",end_time-start_time)
  
    # Clean up Ray resources and processes.
    ray.shutdown()

import time
import ray
import modeldef, parameterserver, dataworker, utilities

'''This example outlines asynchronous training. This means that gradients are aggregated in parallel, as soon as they are available. The aggregation is not dependent upon whether the other nodes have finished or not.
    This could, ideally, be a bit faster than synchronous'''

# Differences between synchronous and asynchronous are in the main function. Both use the same aggregation function (apply_gradients). What's different is how and when the aggregation function is called. 

if __name__ == '__main__':


    num_workers = 2 #try to keep this equal to the number of machines available
    iterations = 200 # keep this number high. it is variable, dependent upon the modeldef you are running.
    model = modeldef.ConvNet()
    test_loader = utilities.get_data_loader()[1]

    print("Running Asynchronous Parameter Server Training.")
    start_time = time.time()

    ray.init(ignore_reinit_error=True) # reinitialization can throw an error if the ignore_reinit_error flag is set to False (default)
    ps = parameterserver.ParameterServer.remote(1e-2)
    workers = [dataworker.DataWorker.remote() for i in range(num_workers)]

    current_weights = ps.get_weights.remote()
    accuracy = 0
    gradients = {}

    for worker in workers:
        gradients[worker.compute_gradients.remote(current_weights)] = worker

    for i in range(iterations):
        ready_gradient_list, _ = ray.wait(list(gradients)) # ray.wait is an asynchronous call. the .wait() function ensures that whenever anything is possible in gradients, it is immediately used.
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients.
        current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
        gradients[worker.compute_gradients.remote(current_weights)] = worker

        if i % 10 == 0:
            # Evaluate the current modeldef after every 10 updates.
            model.set_weights(ray.get(current_weights))
            accuracy = utilities.evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))
    end_time = time.time()
    print("Time of job on server ",end_time-start_time)

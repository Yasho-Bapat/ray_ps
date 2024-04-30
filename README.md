# Parameter Server for Distributed ML using Ray
By utilizing the Parameter Server model, we distribute the workload across multiple nodes, facilitating parallel processing. The code provided in this repository demonstrates the training of a simple CNN model on the MNIST dataset, consisting of 60,000 images. In the parameter server framework, a centralized server (or group of server nodes) maintains global shared parameters of a machine-learning model (e.g., a neural network) while the data and computation of calculating updates (i.e., gradient descent updates) are distributed over worker nodes.

Parameter Server (ps) will: 
  1.  Receive gradients and apply them to its model.
  2.  Send the updated model back to the workers.

Worker will: 
  1. Continuously evaluate data and send gradients to the parameter server.
  2. Synchronize its model with the Parameter Server model weights.

Each Iteration
- weights are sent by the ps to the workers
- workers apply those weights to their local model and train
- the gradients generated are sent back to the ps
- ps aggregates the gradients, and updates its weights accordingly

There can be multiple parameter servers and multiple workers. Both numbers can be specified in the code. Multiple PS and worker processes can run on the same device, but this should be done decided based on the capabilities of the device. 

We are utilising [Ray](https://docs.ray.io/en/latest/) (an open-source unified compute framework that makes it easy to scale AI and Python workloads) to build the edge clusters and facilitate the distributed training.

### Using Ray to create a Ray cluster
A Ray cluster consists of a single head node and any number of connected worker nodes. Users can submit jobs for execution on the Ray cluster.

##### Ray Head Node
- `ray start --head --port=6379   #specify the port number on the head node`
- `ray job submit --working-dir <working directory> -- python3 <python file>`
- `ray stop`

##### Ray Client Node
- `ray start --address=10.8.1.153:6379  #IP:port of the head node`
- `ray stop`

### Synchronous and Asynchronous Training
Synchronous parameter server training involves a tightly synchronized communication pattern between workers and the parameter server. In this approach, all workers wait for each other to complete their gradients computation before updating the parameters. This ensures consistency across all workers and guarantees that every parameter update is based on the most recent global gradient. On the other hand, asynchronous parameter server training allows workers to update parameters independently without waiting for others. Each worker computes gradients based on its local data and updates the parameters asynchronously. 

For the purpose of the project we have elected to use asynchronous training given the heterogeneous nature of the architecture, however we have provided working code for both synchronous and asynchronous approaches.

### Implementation
In this implementation, there are 4 modules (dataworker, modeldef, parameterserver and utilities). These 4 are core libraries required for the distributed system to run. These are described below:
- `dataworker`: This module contains the methods that will be running on the dataworker nodes in the system. Specifically, the `compute_gradients()` function, which takes current weights and trains the model to increase its accuracy and returns the updated gradients.  
- `modeldef`: This module contains the definition of the actual machine learning model that will be running on the devices. This is to be changed according to use case.
- `parameterserver`: This module contains the actual code running on the PS, which does gradient aggregation, specifically the `apply_gradients` function.
- `utilities`: This module contains general utility functions required for running, debugging and evaluating the program. As of this example, it contains a method to obtain data (from an online repository) and the function used for evaluating the model. More methods can be added as required. 
### Integration with ILP
For integrating this work with the output of the ILP algorithm, **node affinity scheduling** can be used. This is a method by which particular tasks can be assigned to particular nodes in the cluster. In principle, the ILP program should return a list of nodes (as Node IDs), which will be running the ML process. Then these node IDs can be passed to the tasks and the node affinity scheduler individually. [Node Affinity Scheduling Strategy](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy.html) is one of the many scheduling strategies that Ray offers, but this is the one which could be used for maximum customizability. For more information on Ray's scheduling strategies, visit [this page](https://docs.ray.io/en/latest/ray-core/scheduling/index.html).

### Documentation referred
- [Ray Cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html#cluster-key-concepts)
- [Ray Parameter Server](https://docs.ray.io/en/latest/ray-core/examples/plot_parameter_server.html)

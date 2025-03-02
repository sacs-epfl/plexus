# Plexus: Practical Federated Learning without a Server
Repository for the source code of our paper *[Practical Federated Learning without a Server](https://arxiv.org/pdf/2302.13837)* published at [The Workshop on Machine Learning and System 2025](https://euromlsys.eu/#).

## Abstract

Federated Learning (FL) enables end-user devices to collaboratively train ML models without sharing raw data, thereby preserving data privacy.
In FL, a central parameter server coordinates the learning process by iteratively aggregating the trained models received from clients. 
Yet, deploying a central server is not always feasible due to hardware unavailability, infrastructure constraints, or operational costs.
We present Plexus, a fully decentralized FL system for large networks that operates without the drawbacks originating from having a central server.
Plexus distributes the responsibilities of model aggregation and sampling among participating nodes while avoiding network-wide coordination.
We evaluate Plexus using realistic traces for compute speed, pairwise latency and network capacity.
Our experiments on three common learning tasks and with up to 1000 nodes empirically show that Plexus reduces time-to-accuracy by 1.4-1.6x, communication volume by 15.8-292x and training resources needed for convergence by 30.5-77.9x compared to conventional decentralized learning algorithms.

## Installation

Start by cloning the repository recursively (since Plexus depends on the PyIPv8 networking library):

```
git clone git@github.com:sacs-epfl/plexus.git --recursive
```

Install the required dependencies (preferably in a virtual environment to avoid conflicts with existing libraries):

```
pip install -r requirements.txt
```

In our paper, we evaluate Plexus using the CIFAR-10, CelebA and FEMNIST datasets.
For CIFAR-10 we use `torchvision`. The FEMNIST and CelebA datasets have to be downloaded manually and we refer the reader to the [decentralizepy framework](https://github.com/sacs-epfl/decentralizepy) that uses the same datasets.

## Running

You can run Plexus for different datasets using the following commands:

```
# CIFAR-10
python3 -u simulations/plexus/cifar10.py \
    --peers 1000 \
    --sample-size 13 \
    --accuracy-logging-interval 10 \
    --success-fraction 0.8 \
    --duration 180000 \
    --bypass-model-transfers \
    --seed 42 \
    --log-level INFO

# CelebA
python3 -u simulations/plexus/celeba.py \
    --peers 500 \
    --sample-size 13 \
    --accuracy-logging-interval 10 \
    --success-fraction 0.8 \
    --duration 180000 \
    --bypass-model-transfers \
    --seed 42 \
    --log-level INFO

# FEMNIST
python3 -u simulations/plexus/femnist.py \
    --peers 355 \
    --sample-size 13 \
    --accuracy-logging-interval 10 \
    --success-fraction 0.8 \
    --duration 720000 \
    --bypass-model-transfers \
    --seed 42 \
    --log-level INFO
```

You can modify the path to the datasets with the `--dataset-base-path` argument.
You can speed up the experiment executing by enabling cuda by passing `--train-device-name "cuda:0" --accuracy-device-name "cuda:0"`.
Since the experiment outputs quite a lot of logging, we also recommend to increase the log level for longer experiments.

To run the Federated Learning (FL) baseline, it suffices to pass the `--fix-aggregator` flag to the above commands, which will select a random peer to act as server and sets the bandwidth of this server to infinity.
The D-PSGD baseline can be run as follows:

```
# CIFAR-10
python3 -u simulations/dl/cifar10.py \
    --peers 1000 \
    --seed 42 \
    --duration 180000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --dl-round-timeout 60 \
    --topology k-regular \
    --k 10 \
    --log-level INFO

# CelebA
python3 -u simulations/dl/celeba.py \
    --peers 500 \
    --seed 42 \
    --duration 180000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --dl-round-timeout 60 \
    --topology k-regular \
    --k 10 \
    --log-level INFO

# FEMNIST
python3 -u simulations/dl/femnist.py \
    --peers 355 \
    --seed 42 \
    --duration 720000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --dl-round-timeout 60 \
    --topology k-regular \
    --k 10 \
    --log-level INFO
```

While the above commands use a `k-regular` topology, Plexus supports the `ring`, `k-regular` and `exp-one-peer` topologies.
For the `k-regular` topology, the `k` argument is required which indicates the number of incoming and outgoing edges in the topology.

The Gossip Learning (GL) baseline can be run as follows:

```
# CIFAR-10
python3 -u simulations/gl/cifar10.py \
    --peers 1000 \
    --seed 42 \
    --duration 180000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --accuracy-logging-interval-is-in-sec \
    --log-level INFO

# CelebA
python3 -u simulations/gl/celeba.py \
    --peers 500 \
    --seed 42 \
    --duration 180000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --accuracy-logging-interval-is-in-sec \
    --log-level INFO

# FEMNIST
python3 -u simulations/gl/femnist.py \
    --peers 355 \
    --seed 42 \
    --duration 720000 \
    --bypass-model-transfers \
    --accuracy-logging-interval 7200 \
    --accuracy-logging-interval-is-in-sec \
    --log-level INFO
```

We provide several scripts in the `scripts` directory to help running these experiments.
For example, `run_parallel_with_seeds.sh` can run the above experiments in parallel for different seeds.

## Reference

If you find our work useful, you can cite us as follows:

```
@inproceedings{devos2025plexus,
  title={Practical Federated Learning without a Server},
  author={Dhasade, Akash and Kermarrec, Anne-Marie and Lavoie, Erick and Pouwelse, Johan and Sharma, Rishi and de Vos, Martijn},
  booktitle={Proceedings of the 5th Workshop on Machine Learning and Systems},
  year={2025}
}
```

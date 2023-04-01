# Artifact Evaluation Guide

This document provides a guide on how to exercise this software artifact to
reproduce key results from the paper **Communication-Efficient Graph Neural
Networks with Probabilistic Neighborhood Expansion Analysis and Caching**
published at MLSys 2023.  The artifact covers two classes of experiments, which
are accessible through the following directories in this repository:

- The `experiments` directory contains scripts that streamline the process of
  executing _multi-GPU experiments_ in a distributed system and measure GNN
  training performance.  It also includes a convenience script for replicating
  the communication-volume simulation experiments in the paper.
- The `caching` directory contains the code for caching and
  communication-volume simulations, and a script for running custom simulation
  experiments.

This artifact attempts to streamline the process of exercising the
functionality in SALIENT++ by providing pathways for users who do not have
access to a distributed cluster of GPU nodes or who wish to skip time consuming
data preprocessing steps. These options are discussed in more depth later in
this document, but we summarize these options here so that users can determine,
in advance, how they would like to exercise the artifact.

- The distributed performance experiments in `experiments` can be run either on
  a SLURM cluster of GPU nodes or locally on a single machine with multiple
  GPUs.  To run locally on a multi-GPU machine, pass the flag `--run_local 1`.
- Helper scripts are provided for downloading (a) pre-generated partitions and
  (b) pre-processed partitioned datasets.
- The simulation experiments all run on a single machine and do not use a GPU.


## Contents

- [Outline of repository structure](#directories-in-this-repo)
- [Setup](#setup)
- [Simulation experiments](#simulation-experiments-caching-impact-on-communication-volume)
- [Distributed experiments](#distributed-gnn-training-performance-experiments)
- [Acknowledgements](#acknowledgements)


## Outline of repository structure

The contents of this artifact repo are organized in directories as follows:

- `caching/`: Code for simulated communication experiments, computing vertex
  inclusion probability (VIP) weights, and identifying remote vertices to be
  cached locally.
- `dataset/`: Initially empty, this is the default location for
  downloading/storing GNN training datasets (raw and pre-processed) and
  partition label tensors.
- `driver/`: Driver code for running GNN computations with SALIENT++.
- `experiments/`: Scripts for performing local simulation and distributed
  execution experiments that reproduce the results presented in the paper, and
  for downloading the corresponding input datasets.
- `fast_sampler/`: Code the SALIENT neighbor sampler module.
- `fast_trainer/`: Code for the SALIENT++ distributed GNN training
  infrastructure.
- `workflows/`: Additional instructions for running custom distributed
  experiments, running experiments on a single multi-GPU machine, and manual
  partitioning and pre-processing of datasets.


## Setup

### Install SALIENT++

Refer to [INSTALL.md](INSTALL.md) for instructions on how to install SALIENT++
and all dependencies in order to exercise this artifact.

In what follows, we use `${SALIENT_ROOT}` to refer to the root of your clone of
this repository.

### Utility scripts

We provide utility scripts for automatically configuring experiments based on
your execution environment (e.g., number of cores and available disk space) and
downloading pre-processed and partitioned datasets.

#### Configuration

Use the `experiments/configure_for_environment.py` script to set
configuration parameters:

```bash
cd ${SALIENT_ROOT}/experiments
python configure_for_environment.py --max_num_nodes <N> --force_diskspace_limit <S>
```

The usage of this script is provided below.

```
usage: configure_for_environment.py [-h] [--force_diskspace_limit FORCE_DISKSPACE_LIMIT] --max_num_nodes MAX_NUM_NODES

Configure SALIENT++ distributed experiments.

optional arguments:
  -h, --help            show this help message and exit
  --force_diskspace_limit FORCE_DISKSPACE_LIMIT
                        Override normal behavior which will analyze disk space using filesystem commands. The script will attempt, but not guarantee, to
                        select a subset of datasets that will fit within the specified disk-space constraints. Specify an integer representing a soft limit on
                        the number of GB used for storage.
  --max_num_nodes MAX_NUM_NODES
                        Specify the maximum number of nodes (GPUs) you will use for experiments. This determines which preprocessed partitioned datasets are
                        downloaded --- e.g., 8-way partitioned datasets will not be downloaded if you specify a value here of less than 8.
```

If you plan to run experiments on up to 8 GPUs, and want to use no more than
500GB of space then you would run the following command,

```bash
python configure_for_environment.py --max_num_nodes 8 --force_diskspace_limit 500
```

which would result in configuring your environment to download all partitioned
datasets except MAG240. You can manually edit the contents of
`configuration_files/feasible_datasets.cfg` if you wish to override the
decisions made by the configuration script.

#### Download pre-processed datasets

The script `experiments/download_datasets_fast.py` can be used to
download preprocessed partitioned datasets. 

```bash
python download_datasets_fast.py
```

This process is optional, but if you choose to skip it you will need to follow
additional instructions for computing graph partitions and generating reordered
partitioned datasets. If you wish to generate partitioned datasets yourself,
you should follow the instructions in [INSTALL.md](INSTALL.md) for installing
optional dependencies related to METIS and read the instructions for
partitioning and reordering datasets at
[workflows/partition_and_reorder_1.md](workflows/partition_and_reorder_1.md).

This script accepts two optional flags:

- `--skip_confirmation`: Do not prompt for confirmation before downloading a
  dataset.  The datasets downloaded by this script are determined by the
  contents of `configuration_files/feasible_datasets.cfg` generated by
  `configure_for_environment.py`. This script downloads rather large files and
  may ask for confirmation before each download. Since many such large files
  will be downloaded by this script (e.g., the papers100M dataset will be
  downloaded 4 times if `--max_num_nodes=8` for 1, 2, 4, and 8 partitions), you
  may find it useful to skip the confirmation prompts.

- `--simulation_downloads_only`: If you wish to only run the simulation
  experiments, you do not need to download the pre-processed partitioned
  datasets. This flag opts out of downloading the preprocessed partitioned
  datasets.

## Simulation experiments: caching impact on communication volume

This section provides instructions for running local GNN training simulation
experiments to evaluate the inter-partition communication volume with different
caching policies and replication factors.

### Preset experiments with ogbn-papers100M

To perform simulation experiments with the same parameters that were used
to generate the results in Figure 2 of the MLSys 2023 paper, run the provided
`experiments/run_sim_experiments_paper.sh` bash script:

```bash
bash ./run_sim_experiments_paper.sh
```

The script will run a set of caching simulation experiments, reporting progress
and results along the way, and then print tables that show the average
per-epoch communication volume in number of remote vertices for each experiment
configuration.

**NOTE:** This script assumes that you have downloaded the ogbn-papers100M
dataset and associated 8-way partition labels using the utility scripts
described earlier.  The script runs locally on CPU cores, requires
approximately 160 GB of RAM, and may take a few hours to complete.

### Custom simulation experiments

`caching/experiment_communication_caching.py`

The script `experiment/run_sim_experiments.sh` allows you to run custom
simulation experiments with a variety of options.  For example, to run the
simulation experiments with the `ogbn-products` dataset and 4 partitions (all
downloaded using the utility script described earlier), using default options,
execute the following:

```bash
bash ./run_sim_experiments.sh --dataset_name ogbn-products --dataset_dir ../dataset --partition_labels_dir ../dataset/partition-labels --num_partitions 4
```

This script will run a number of simulated GNN training epochs, evaluate the
effectiveness of different caching policies with varying replication factor,
and print and store the results.  For example:

```
Loading 'ogbn-products', partitions=4
Path is ../dataset/ogbn-products
Simulating vertex accesses (10 epochs)
[...]
Average per-epoch communication in # of vertices
- ogbn-products, 4 partitions, 10 epochs, fanout (15,10,5), minibatch size 1024
+-------+----------------+----------------+---------------------+-----------+------------------+
| alpha | vip-analytical | vip-simulation | num-paths-reachable | halo-1hop | degree-reachable |
+-------+----------------+----------------+---------------------+-----------+------------------+
|     0 |       1.67E+07 |       1.67E+07 |            1.67E+07 |  1.67E+07 |         1.67E+07 |
|  0.01 |       1.57E+07 |       1.56E+07 |            1.60E+07 |  1.64E+07 |         1.62E+07 |
|  0.05 |       1.29E+07 |       1.29E+07 |            1.40E+07 |  1.54E+07 |         1.50E+07 |
|   0.1 |       1.07E+07 |       1.08E+07 |            1.20E+07 |  1.40E+07 |         1.39E+07 |
|   0.2 |       7.95E+06 |       8.01E+06 |            9.30E+06 |  1.16E+07 |         1.22E+07 |
|   0.5 |       3.88E+06 |       4.06E+06 |            4.94E+06 |  6.50E+06 |         8.76E+06 |
|   1.0 |       1.34E+06 |       1.62E+06 |            1.95E+06 |  5.71E+06 |         4.69E+06 |
+-------+----------------+----------------+---------------------+-----------+------------------+
Saving communication results (results-simulation/sim-comm-ogbn-products-partitions-4-minibatch-1024-fanout-15-10-5-epochs-10.pobj)
```

#### Simulation experiment script options

The bash script `experiment/run_sim_experiments.sh` is simply a wrapper for the
`caching/experiment_communication_caching.py` Python script.  The latter
supports several options that control the simulation and caching computations.
These are described in the script's command-line argument documentation, which
is echoed by passing the `--help` flag to the script:

```
usage: experiment_communication_caching.py [-h] --dataset_name DATASET_NAME --dataset_dir DATASET_DIR --partition_labels_dir PARTITION_LABELS_DIR --num_partitions
                                           NUM_PARTITIONS [--fanouts FANOUTS [FANOUTS ...]] [--minibatch_size MINIBATCH_SIZE] [--num_epochs_eval NUM_EPOCHS_EVAL]
                                           [--use_sim_accesses_file USE_SIM_ACCESSES_FILE] [--cache_schemes CACHE_SCHEMES [CACHE_SCHEMES ...]]
                                           [--replication_factors REPLICATION_FACTORS [REPLICATION_FACTORS ...]] [--num_epochs_vip_sim NUM_EPOCHS_VIP_SIM]
                                           [--num_workers_sampler NUM_WORKERS_SAMPLER] [--output_prefix OUTPUT_PREFIX] [--store_sim_accesses STORE_SIM_ACCESSES]

Run VIP caching communication simulation experiments.

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        Dataset name (default: None)
  --dataset_dir DATASET_DIR
                        Path to the partitioned dataset directory (default: None)
  --partition_labels_dir PARTITION_LABELS_DIR
                        Path to the partition-label tensors directory (default: None)
  --num_partitions NUM_PARTITIONS
                        Number of partitions (default: None)
  --fanouts FANOUTS [FANOUTS ...]
                        Layer-wise sampling fanouts (default: [15, 10, 5])
  --minibatch_size MINIBATCH_SIZE
                        Minibatch size (default: 1024)
  --num_epochs_eval NUM_EPOCHS_EVAL
                        Number of simulated epochs for communication evaluation (default: 10)
  --cache_schemes CACHE_SCHEMES [CACHE_SCHEMES ...]
                        VIP caching scheme names (default: ['degree-reachable', 'num-paths-reachable', 'halo-1hop', 'vip-simulation', 'vip-analytical'])
  --replication_factors REPLICATION_FACTORS [REPLICATION_FACTORS ...]
                        Cache replication factors (alpha) (default: [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
  --num_epochs_vip_sim NUM_EPOCHS_VIP_SIM
                        Number of epochs for simulation-based VIP weight estimation (default: 2)
  --num_workers_sampler NUM_WORKERS_SAMPLER
                        Number of CPU workers used by the SALIENT fast sampler (default: 20)
  --output_prefix OUTPUT_PREFIX
                        Prefix name for communication simulation results output .pobj file. (default: experiments/results-simulation/sim-comm)
  --store_sim_accesses STORE_SIM_ACCESSES
                        set to 1 to store vertex-wise access statistics after simulation. (default: False)
  --use_sim_accesses_file USE_SIM_ACCESSES_FILE
                        If specified, skip evaluation simulation and use vertex accesses from file (default: None)
```

The last two options, `--store_sim_accesses` and `--use_sim_accesses_file`, can
be used to avoid repeating the simulation computations.  This is useful, for
example, if post-evaluating different caching policy configurations for a long
simulation (e.g., to measure the average communication over 100 epochs).

### Parsing stored results

To parse and display stored simulation experiment results, use the
`caching/parse_communication_volume_results.py` script:

```bash
python caching/parse_communication_volume_results.py --path experiments/results-simulation/sim-comm-ogbn-products-partitions-4-minibatch-1024-fanout-15-10-5-epochs-10.pobj
```

## Distributed GNN training performance experiments

This section will walk you through the process of running a specific set of
prepared experiment scripts.  For custom experiments, please see the
instructions in
[workflows/run_custom_distributed_experiments.md](workflows/run_custom_distributed_experiments.md).

### Configuration for SLURM cluster or local execution

If you are running on a SLURM cluster, the `exp_driver.py` script will generate
a script that will then be executed using `sbatch`.

You must edit the `SLURM_CONFIG` variable at the top of `exp_driver.py` to provide cluster-specific parameters to execute your distributed experiments.

```python
SLURM_CONFIG="""#SBATCH -J sb_{JOB_NAME}
#SBATCH -o {OUTPUT_FILE_DIRECTORY}/%x-%A.%a.out
#SBATCH -e {OUTPUT_FILE_DIRECTORY}/%x-%A.%a.err
#SBATCH --nodes=1
#SBATCH -t 0-4:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -a 1-{NUM_MACHINES}
#SBATCH --qos=high
#SBATCH --export=ALL
#SBATCH --partition=queue1
"""
```

The variables `{JOB_NAME}`, `{OUTPUT_FILE_DIRECTORY}`, and `{NUM_NODES}` will
be populated automatically by the `exp_driver.py` script. You must set the
rest of the SBATCH parameters to appropriate values for your cluster. Note that
that you must ensure that you configure your job so that at most one process is
launched per machine, which may necessitate the use of the `--exclusive` flag.

Note that, our experiment helper scripts are configured to work most smoothly in two cases: 
(a) a cluster where there is one machine per-GPU; and (b) a single machine with multiple GPUs. If you want to run
in the multi-machine multi-GPU setting, you must modify the default parameter value for ``--num_devices_per_node`` in
``exp_driver.py``. The nomenclature these scripts use is confusing, so let us explain this case in more detail. 
Note that, the ``--num_nodes`` parameter in ``exp_driver.py`` and other scripts in the ``experiments`` directory
does not refer to the number of machines used in the distributed experiment when the number of devices per node is greater than 1. Instead,
it will refer to the number of GPUs (and thus partitions) being used for the experiment. As such, all experiments must use a value for ``--num_nodes`` that is divisible by the number of devices per-machine.

## List of prepared distributed-training experiments

There are 4 prepared experiments:

- **A.**  Impact of different optimizations
  (`experiment_optimization_impact.py`). Can be used to reproduce Table 1 and
  Figure 4.
- **B.**  Per-epoch runtime versus replication factor
  (`experiment_vary_replication_factor.py`). Can be used to reproduce Figure 5
  and Figure 7.
- **C.**  Per-epoch runtime versus percentage of data on GPU
  (`experiment_vary_gpu_percent.py`). Can be used to reproduce Figure 6.
- **D.**  Accuracy on datasets (`experiment_accuracy.py`). Can be used to
  verify accuracy of trained models using SALIENT++.

For all of these experiments, you can display a table of results for all
experiments run so far by adding the flag `--parse_results_only True`. You can
also run all experiments on a single machine with multiple GPUs, instead of
running on a SLURM cluster, by passing the flag `--run_local 1`.

### A. Impact of different optimizations (`experiment_optimization_impact.py`)

You can compare the performance of SALIENT and different versions of SALIENT++
that incorporate key optimizations. The `partition` system version adapts
SALIENT to operate on partitioned datasets, instead of fully replicating all
data, but does not incorporate additional optimizations to pipeline
communication or cache data. The `pipeline` version extends `partition` by
incorporating a multi-stage pipeline to overlap the host-device and network
communication involved with communicating feature data across
machines. Finally, the `vip` version extends `pipeline` by using
vertex-inclusion probability analysis to cache the feature data of frequently
accessed vertices to reduce communication volume.

The following command 

```bash
python experiment_optimization_impact.py --dataset_name ogbn-papers100M --dataset_dir ../dataset --num_nodes_list 8 4 2
```

runs the experiment on the ogbn-papers100M dataset on 8, 4 and 2 nodes. The
following output is generated on a cluster of 8 AWS instances of type
g5.8xlarge.

```
+----------------------------+----------------+-----------+-------------+--------------------+---------------+--------------------+
|          Dataset           | System version | Num Nodes | GPU Percent | Replication Factor | Pipeline Comm |   Per-Epoch time   |
+----------------------------+----------------+-----------+-------------+--------------------+---------------+--------------------+
| ../dataset/ogbn-papers100M |    salient     |     2     |     0.0     |        0.0         |      N/A      | 11407.285349486707 |
| ../dataset/ogbn-papers100M |    salient     |     4     |     0.0     |        0.0         |      N/A      | 5632.410749585575  |
| ../dataset/ogbn-papers100M |    salient     |     8     |     0.0     |        0.0         |      N/A      | 3059.607648131748  |
| ../dataset/ogbn-papers100M |   partition    |     2     |     0.0     |        0.0         |       No      | 41399.77080273359  |
| ../dataset/ogbn-papers100M |   partition    |     4     |     0.0     |        0.0         |       No      | 17489.229134771766 |
| ../dataset/ogbn-papers100M |   partition    |     8     |     0.0     |        0.0         |       No      | 11581.732730285885 |
| ../dataset/ogbn-papers100M |    pipeline    |     2     |     0.0     |        0.0         |      Yes      | 20176.131764118767 |
| ../dataset/ogbn-papers100M |    pipeline    |     4     |     0.0     |        0.0         |      Yes      | 8787.786316200367  |
| ../dataset/ogbn-papers100M |    pipeline    |     8     |     0.0     |        0.0         |      Yes      | 5801.716621940748  |
| ../dataset/ogbn-papers100M |      vip       |     2     |     0.15    |        0.08        |      Yes      | 10757.24265893497  |
| ../dataset/ogbn-papers100M |      vip       |     4     |     0.15    |        0.16        |      Yes      | 5724.088441804775  |
| ../dataset/ogbn-papers100M |      vip       |     8     |     0.15    |        0.32        |      Yes      | 3153.999767107889  |
+----------------------------+----------------+-----------+-------------+--------------------+---------------+--------------------+
```

### B. Per-epoch runtime versus replication factor (`experiment_vary_replication_factor.py`)

This experiment allows you to specify a list of different replication factors
and measures SALIENT++'s per-epoch runtime for each value.

For example, the command below runs SALIENT++ on the ogbn-papers100M dataset on
8 nodes with replication factors 0, 1, 2, 4, 8, 16, 32, and 64 (measured as
percentages of the number of vertices in each local partition)

```bash
python experiment_vary_replication_factor.py --dataset_name ogbn-papers100M --dataset_dir ../dataset --num_nodes 8 --replication_factor_list 0 1 2 4 8 16 32 64
```

After completion, the following table is generated.

```
+----------------------------+-----------+-------------+--------------------+--------------------+
|          Dataset           | Num Nodes | GPU Percent | Replication Factor |   Per-Epoch time   |
+----------------------------+-----------+-------------+--------------------+--------------------+
| ../dataset/ogbn-papers100M |     8     |     0.15    |        0.0         | 7042.021358306416  |
| ../dataset/ogbn-papers100M |     8     |     0.15    |        0.01        | 5836.294363993427  |
| ../dataset/ogbn-papers100M |     8     |     0.15    |        0.02        | 5090.2936436006175 |
| ../dataset/ogbn-papers100M |     8     |     0.15    |        0.04        | 4289.161108649853  |
| ../dataset/ogbn-papers100M |     8     |     0.15    |        0.08        | 3622.2004357820583 |
| ../dataset/ogbn-papers100M |     8     |     0.15    |        0.16        | 3570.064393489104  |
| ../dataset/ogbn-papers100M |     8     |     0.15    |        0.32        | 3241.3765479219664 |
| ../dataset/ogbn-papers100M |     8     |     0.15    |        0.64        | 3289.4505012460672 |
+----------------------------+-----------+-------------+--------------------+--------------------+
```

You can modify the `--gpu_percent` parameter used for this experiment by
passing the desired value as a command line argument: `--gpu_percent <val>`.

### C. Per-epoch runtime versus percentage of data on GPU (`experiment_vary_gpu_percent.py`)

The script `experiment_vary_gpu_percent.py` measures the per-epoch runtime of
SALIENT++ when storing a different percentage of the node features in GPU
memory. This script provides the option to perform a comparison between two
differently ordered datasets. 

Note that, presently, to save time and space our scripts do not download the non-reordered versions of different datasets. Instructions for generating a dataset that is not reordered are provided in [workflows/partition_and_reorder_1.md](workflows/partition_and_reorder_1.md).

The command 

```bash
python experiment_vary_gpu_percent.py --dataset_name ogbn-papers100M --dataset_dir_viporder ../dataset --dataset_dir_inputorder ../dataset-4constraint-inputorder --num_nodes 4 --replication_factor 15
```

generates the following output

```
+---------------------------------------------------+-------------+-----------+--------------------+-------------+-------------------+
|                      Dataset                      |    System   | Num Nodes | Replication Factor | GPU Percent |   Per-Epoch time  |
+---------------------------------------------------+-------------+-----------+--------------------+-------------+-------------------+
|             ../dataset/ogbn-papers100M            |  VIP Order  |     4     |        0.15        |     0.0     | 6033.232439877879 |
|             ../dataset/ogbn-papers100M            |  VIP Order  |     4     |        0.15        |     0.1     | 5759.152126575303 |
|             ../dataset/ogbn-papers100M            |  VIP Order  |     4     |        0.15        |     0.25    |  5746.39854423174 |
|             ../dataset/ogbn-papers100M            |  VIP Order  |     4     |        0.15        |     0.5     | 5666.521418532564 |
|             ../dataset/ogbn-papers100M            |  VIP Order  |     4     |        0.15        |     0.75    | 5719.900165736365 |
|             ../dataset/ogbn-papers100M            |  VIP Order  |     4     |        0.15        |     1.0     | 5756.726035469419 |
| ../dataset-4constraint-inputorder/ogbn-papers100M | Input Order |     4     |        0.15        |     0.0     | 6401.780735645651 |
| ../dataset-4constraint-inputorder/ogbn-papers100M | Input Order |     4     |        0.15        |     0.1     | 6050.808943205409 |
| ../dataset-4constraint-inputorder/ogbn-papers100M | Input Order |     4     |        0.15        |     0.25    | 6097.511649903014 |
| ../dataset-4constraint-inputorder/ogbn-papers100M | Input Order |     4     |        0.15        |     0.5     | 5982.476283267244 |
| ../dataset-4constraint-inputorder/ogbn-papers100M | Input Order |     4     |        0.15        |     0.75    |  5787.13487790525 |
| ../dataset-4constraint-inputorder/ogbn-papers100M | Input Order |     4     |        0.15        |     1.0     | 5797.535189236618 |
+---------------------------------------------------+-------------+-----------+--------------------+-------------+-------------------+
```

### D. Trained model accuracy (`experiment_accuracy.py`)

To verify accuracy of models, you can run the `experiment_accuracy.py`
script. This script accepts an optional argument `--make_deterministic` which
will configure CUDNN and CUBLAS to use deterministic algorithms. Note that this
may result in reduced performance in some cases as some deterministic
algorithms introduce device-to-host data transfers that can impact the
effectiveness of pipelining.

The following command can be used to generate accuracy measurements for MAG240,
ogbn-papers100M, and ogbn-products for training on 8 nodes deterministically.

```bash
python experiment_accuracy.py --dataset_name ogbn-papers100M ogbn-products --dataset_dir ../dataset --num_nodes 8 --make_deterministic
```

After completion, this script generates the following output.

```
+----------------------------+-----------+-------------+--------------------+--------------------+--------------------+--------------------+
|          Dataset           | Num Nodes | GPU Percent | Replication Factor |   Per-Epoch time   |   Valid accuracy   |   Test accuracy    |
+----------------------------+-----------+-------------+--------------------+--------------------+--------------------+--------------------+
|     ../dataset/MAG240      |     8     |     0.15    |        0.15        | 5001.481605186866  | 0.6509510683776062 |        N/A         |
| ../dataset/ogbn-papers100M |     8     |     0.15    |        0.15        | 3653.0829548643387 | 0.6764778669221251 | 0.6467681885619909 |
|  ../dataset/ogbn-products  |     8     |     0.15    |        0.15        | 817.7131605893053  | 0.9116801871678153 | 0.7850490558228288 |
+----------------------------+-----------+-------------+--------------------+--------------------+--------------------+--------------------+
```

# Acknowledgements

This research was sponsored by MIT-IBM Watson AI Lab and in part by the United
States Air Force Research Laboratory and the United States Air Force Artificial
Intelligence Accelerator and was accomplished under Cooperative Agreement
Number FA8750-19-2-1000. The views and conclusions contained in this document
are those of the authors and should not be interpreted as representing the
official policies, either expressed or implied, of the United States Air Force
or the U.S. Government. The U.S. Government is authorized to reproduce and
distribute reprints for Government purposes notwithstanding any copyright
notation herein.

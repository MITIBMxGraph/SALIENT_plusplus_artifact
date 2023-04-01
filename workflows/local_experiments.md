


## Single-Machine Multi-GPU Experiments

Our experimental scripts can be run on a single machine with multiple GPUs.
These experiments are designed to exercise the same functionality as the
distributed multi-machine experiments --- i.e., a single-machine 4-GPU
experiment will use the same partitioned dataset and communication patterns as
an experiment across 4 machines with 1-GPU each. One exception to this
principle is for experiments involving the `salient` baseline system which will
store the graph feature data in shared memory such that it is accessible to all
processes on the same host.

### General instructions for running experiments locally on a single machine.

You can run the provided distributed experiments locally on a single machine with multiple GPUs by defining the command-line argument ``--run_local 1``. You may additionally wish to specify the number of sampling workers per-GPU by setting the command line argument ``--num_samplers_per_gpu N`` to use N sampling workers per-GPU. If ``--num_samplers_per_gpu`` is not specified (or is set to 0) the number of sampling workers will be the default value (15) for ``--num_workers`` in ``exp_driver.py``. A reasonable rule of thumb is to set the number of samplers per-GPU to be between ``(# Physical CPUs) / # GPUs`` and ``3/2 (# Hardware Threads) / # GPUs``.

## Instructions for individual experiments

### Experiment A: Impact of different optimizations (experiment_optimization_impact.py)

The script `experiment_optimization_impact.py` can be used to compare the
per-epoch runtime of different system designs: `salient`, `partition`,
`pipeline`, and `vip`.

```bash
usage: experiment_optimization_impact.py [-h] --dataset_name DATASET_NAME --dataset_dir DATASET_DIR --salient_dataset_dir SALIENT_DATASET_DIR --num_nodes_list
                                         [NUM_NODES_LIST ...] [--parse_results_only PARSE_RESULTS_ONLY] [--joblist_directory JOBLIST_DIRECTORY]
                                         [--run_local RUN_LOCAL] [--num_samplers_per_gpu NUM_SAMPLERS_PER_GPU]

run distributed experiment

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        name of dataset
  --dataset_dir DATASET_DIR
                        directory to partitioned dataset
  --salient_dataset_dir SALIENT_DATASET_DIR
                        directory to regular dataset for vanilla SALIENT
  --num_nodes_list [NUM_NODES_LIST ...]
                        list number of nodes to run on
  --parse_results_only PARSE_RESULTS_ONLY
                        set --parse_results_only True to display a table of results generated from a previous run.
  --joblist_directory JOBLIST_DIRECTORY
                        Location to place generated joblists.
  --run_local RUN_LOCAL
                        Set to 1 to run locally on a single machine with multiple gpus.
  --num_samplers_per_gpu NUM_SAMPLERS_PER_GPU
                        If set to 0 (default), default value in exp_driver.py is used. You may need to change this to be a smaller number in the multi-gpu
                        single-machine setting.
```


For the ogbn-products dataset, you can execute the experiments for 2 and 4 GPUs with the following command. 

```bash
python experiment_optimization_impact.py --dataset_name ogbn-products --dataset_dir ../dataset-4constraint --salient_dataset_dir ../dataset --num_nodes_list 2 4 --run_local 1
``` 

Results are shown below for an execution of the above script on an AWS g5.24xlarge instance which has 48 physical cores (96 hardware threads) and 4 GPUs.

```bash
+--------------------------------------+----------------+----------------+-------------+--------------------+---------------+--------------------+
|               Dataset                | System version |   Num Nodes    | GPU Percent | Replication Factor | Pipeline Comm |   Per-Epoch time   |
+--------------------------------------+----------------+----------------+-------------+--------------------+---------------+--------------------+
|       ../dataset/ogbn-products       |    salient     | 2 (local exec) |     0.0     |        0.0         |      N/A      | 1757.8975135953062 |
|       ../dataset/ogbn-products       |    salient     | 4 (local exec) |     0.0     |        0.0         |      N/A      | 1537.9751914770652 |
| ../dataset-4constraint/ogbn-products |   partition    | 2 (local exec) |     0.0     |        0.0         |       No      | 3114.618688107675  |
| ../dataset-4constraint/ogbn-products |   partition    | 4 (local exec) |     0.0     |        0.0         |       No      | 2465.2895663368204 |
| ../dataset-4constraint/ogbn-products |    pipeline    | 2 (local exec) |     0.0     |        0.0         |      Yes      | 2001.8269405881356 |
| ../dataset-4constraint/ogbn-products |    pipeline    | 4 (local exec) |     0.0     |        0.0         |      Yes      | 1677.554101601036  |
| ../dataset-4constraint/ogbn-products |      vip       | 2 (local exec) |     0.15    |        0.08        |      Yes      | 1847.0062099651743 |
| ../dataset-4constraint/ogbn-products |      vip       | 4 (local exec) |     0.15    |        0.16        |      Yes      | 1463.4360417996843 |
+--------------------------------------+----------------+----------------+-------------+--------------------+---------------+--------------------+
```

### Experiment B: Vary replication factor (experiment_vary_replication_factor.py)

This experiment illustrates the impact of increasing the replication factor on per-epoch runtime. 

```bash
usage: experiment_vary_replication_factor.py [-h] --dataset_name DATASET_NAME --dataset_dir DATASET_DIR [--num_nodes NUM_NODES] [--gpu_percent GPU_PERCENT]
                                             [--parse_results_only PARSE_RESULTS_ONLY] [--joblist_directory JOBLIST_DIRECTORY] [--run_local RUN_LOCAL]
                                             [--num_samplers_per_gpu NUM_SAMPLERS_PER_GPU]

run distributed experiment

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        name of dataset
  --dataset_dir DATASET_DIR
                        directory to partitioned dataset, ordered by VIP weights.
  --num_nodes NUM_NODES
                        Number of nodes to run on (default: 4)
  --gpu_percent GPU_PERCENT
                        Fixed gpu_percent to use during experiments (default: 0.15)
  --parse_results_only PARSE_RESULTS_ONLY
                        set --parse_results_only True to display a table of results generated from a previous run.
  --joblist_directory JOBLIST_DIRECTORY
                        Location to place generated joblists.
  --run_local RUN_LOCAL
                        Set to 1 to run locally on a single machine with multiple gpus.
  --num_samplers_per_gpu NUM_SAMPLERS_PER_GPU
                        If set to 0 (default), default value in exp_driver.py is used. You may need to change this to be a smaller number in the multi-gpu
                        single-machine setting.
```

For the ogbn-products dataset, you can execute the experiments on 4 GPUs with the following command. 

```bash
python experiment_vary_replication_factor.py --dataset_name ogbn-products --dataset_dir ../dataset-4constraint --num_nodes 4 --gpu_percent 1.0
```

Which on an AWS g5.24xlarge instance produces the following results.

```bash
+--------------------------------------+----------------+-------------+--------------------+--------------------+
|               Dataset                |   Num Nodes    | GPU Percent | Replication Factor |   Per-Epoch time   |
+--------------------------------------+----------------+-------------+--------------------+--------------------+
| ../dataset-4constraint/ogbn-products | 4 (local exec) |     1.0     |        0.0         | 1336.7483591292466 |
| ../dataset-4constraint/ogbn-products | 4 (local exec) |     1.0     |        0.01        | 1311.0996954402783 |
| ../dataset-4constraint/ogbn-products | 4 (local exec) |     1.0     |        0.02        | 1296.024363665945  |
| ../dataset-4constraint/ogbn-products | 4 (local exec) |     1.0     |        0.04        | 1272.1185978466851 |
| ../dataset-4constraint/ogbn-products | 4 (local exec) |     1.0     |        0.08        | 1590.2893943951156 |
| ../dataset-4constraint/ogbn-products | 4 (local exec) |     1.0     |        0.16        | 1216.3020619253318 |
| ../dataset-4constraint/ogbn-products | 4 (local exec) |     1.0     |        0.32        | 1175.8965575017241 |
| ../dataset-4constraint/ogbn-products | 4 (local exec) |     1.0     |        0.64        | 1161.2499198913574 |
+--------------------------------------+----------------+-------------+--------------------+--------------------+
```

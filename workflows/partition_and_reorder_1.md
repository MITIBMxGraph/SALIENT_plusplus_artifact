# Generating partition labels and generating partitioned datasets

This workflow describes how to generate partition labels for a dataset using METIS, and how to generate a partitioned dataset using the partition labels. The scripts for generating partition labels assume you have followed the instructions in [INSTALL.md](../INSTALL.md) for installing METIS. Note that you should **not** install METIS yourself or use a package manager to install METIS: our scripts assume that you have installed METIS using the specific repositories and processes outlined in the [INSTALL.md](../INSTALL.md) file.

**NOTE** All of the commands in this section will assume that your working directory is the root directory of the `SALIENT_plusplus_artifact` repository.

## Downloading existing partition-labels and preprocessed datasets

The utility script ``experiments/download_datasets_fast.py`` provides an optional command line argument ``--simulation_downloads_only`` which will (a) download preprocessed (unpartitioned) OGB datasets in the FastDataset format; and (b) download pre-generated partition labels for the OGB datasets. These partition labels will be located in the ``dataset/partition-labels`` directory. If you so choose, you can use these pre-generated partition labels to create partitioned datasets without needing to run METIS. To avoid confusion, we suggest that you use a separate directory, ``dataset/my-partition-labels``, to store any partition labels that you generate yourself.

## Generating partition labels

We provide utility scripts for generating partition labels for a graph dataset. These utility scripts require that the initial unpartitioned dataset be in SALIENT's FastDataset format. However, any dataset that is obtainable from OGB will be automatically downloaded and formatted into the FastDataset format the first time you attempt to use it.

To partition the dataset, use the `partitioners/run_4constraint_partition.py` script:

```
usage: run_4constraint_partition.py [-h] [--dataset_name DATASET_NAME] [--output_directory OUTPUT_DIRECTORY] [--dataset_dir DATASET_DIR]
                                   [--num_parts NUM_PARTS]
```

For example, to generate a 4-way partitioning of the ogbn-products dataset you may run the following command:

```bash
python -m partitioners.run_4constraint_partition --dataset_name ogbn-products --output_directory ./dataset/my-partition-labels/ --dataset_dir ./dataset --num_parts 4
```

After completion, the partition labels will be written to disk at ``./dataset/my-partition-labels/ogbn-products-4.pt``.

### Advice for partitioning larger datasets

Partitioning is fairly fast on relatively small datasets like ogbn-products, but for larger datasets like ogbn-papers100M it can take several hours and substantial memory to complete successfully. A machine with 512GB of RAM should be able to partition the larger datasets without running out of memory.

If you have the ability to create swap files and are using a machine with large SSD storage volumes, then it is possible to partition the larger datasets on a machine with much less memory. In our own workflows, we generate partition labels on machines with approximately 128GB of memory with an additional 384GB swapfile on an NVME drive. You can create a swapfile on Ubuntu 20.04 with the following commands:

```bash
# Assumes that the directory /scratch exists on your system and uses a fast SSD for storage with at least 512GB
cd /scratch/
sudo fallocate -l 512G swapfile
sudo chmod 600 swapfile
sudo mkswap swapfile
sudo swapon swapfile
```

You can check that the swapfile was enabled correctly by looking at the available swap space using a tool like ``htop``.

### Time estimates

Partitioning should complete within a few minutes for ogbn-products, but may take several hours on the papers100M and MAG240 datasets. In general, a higher number of partitions results in longer execution time --- e.g., generating 1024 partitions may take a whole day, whereas generating 8 or 16 should complete within 2-4 hours.


## Generating partitioned datasets

We provide a utility script for generating a reordered and partitioned dataset
from an input dataset and corresponding partition labels. 
The script `partitioners/reorder_data.py` reorders vertices first by their partition
ID and then based on their internal access frequency using VIP analysis. You can run 
`partitioners/reorder_data.py` with the additional flag ``--disable_vip`` to only reorder
vertices by their partition ID.

### (a) Reorder by partition and VIP weight

```
usage: reorder_data.py [-h] --dataset_name DATASET_NAME --path_to_part PATH_TO_PART --output_path OUTPUT_PATH --dataset_dir DATASET_DIR [--fanouts [FANOUTS ...]] [--disable_vip]

generate a reordered dataset from OGB dataset, partition file, and frequency analysis.

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        Name of the ogb dataset
  --path_to_part PATH_TO_PART
                        Path to the file containing the partitions
  --output_path OUTPUT_PATH
                        Location to save.
  --dataset_dir DATASET_DIR
                        The dataset directory
  --fanouts [FANOUTS ...]
                        Training fanouts
  --disable_vip         Disables the use of vertex-inclusion probabilities to order vertices within each partition.
```

Example on products:

```bash
python -m partitioners.reorder_data --dataset_name ogbn-products --path_to_part ./dataset/my-partition-labels/ogbn-products-4.pt --dataset_dir ./dataset --output_path dataset-4constraint
```

### (b) Reorder by partition only

The process for generating partitions which do not order by VIP is the same, but you need to pass the flag ``--disable_vip`` and use a different output path. Example:

```bash
python -m partitioners.reorder_data --disable_vip --dataset_name ogbn-products --path_to_part ./dataset/my-partition-labels/ogbn-products-4.pt --dataset_dir ./dataset --output_path dataset-noreorder/
```

In the above example, the reordered dataset will be stored in a new datasets directory, `./dataset-noreorder`. 


## State after running examples

If you ran all of the examples on this page for ogbn-products, you should have the following filesystem state.

```shell-session
$ ls ./dataset/my-partition-labels/
ogbn-products-4.pt

$ ls -R dataset-noreorder
dataset-noreorder:
metis-reordered-k4

dataset-noreorder/metis-reordered-k4:
ogbn-products

dataset-noreorder/metis-reordered-k4/ogbn-products:
col.pt  meta_info.pt  name.pt  num_parts.pt  part_offsets.pt  rowptr.pt  split_idx.pt  split_idx_parts.pt  x0.pt  x1.pt  x2.pt  x3.pt  y.pt
```




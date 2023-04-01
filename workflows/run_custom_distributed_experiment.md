# General instructions for running distributed experiments.

This document will provide general instructions for running customized distributed experiments on a SLURM cluster (or on a single machine with multiple GPUs). The primary script we provide for this purpose is ``distributed_experiments/exp_driver.py`` which is a driver script for executing distributed experiments. This driver script is used by the other experimental scripts provided as artifacts. This document describes how to use ``exp_driver.py`` directly to run custom experiments not covered by the provided experiments.

**Assumptions** This document assumes you have already generated (or downloaded) partitioned datasets and that they are located in the <root_dir>/dataset directory. In our examples, we will assume that you are running ``exp_driver.py`` from the ``distributed_experiments`` directory.

## Local Execution (single machine, multi-gpu)

If you are running on a single machine with multiple GPUs, you should append the additional command-line argument ``--run_local 1`` to all commands. This will cause ``exp_driver.py`` to generate a bash script that will be executed locally.

## Configuration for SLURM Cluster.
If you are running on a SLURM cluster, the ``exp_driver.py`` script will generate a script that will be executed using ``sbatch``. 

You must edit the ``SLURM_CONFIG`` variable at the top of ``exp_driver.py`` to provide cluster-specific parameters to execute your distributed experiments.

```python
SLURM_CONFIG="""#SBATCH -J sb_{JOB_NAME}
#SBATCH -o {OUTPUT_FILE_DIRECTORY}/%x-%A.%a.out
#SBATCH -e {OUTPUT_FILE_DIRECTORY}/%x-%A.%a.err
#SBATCH --nodes=1
#SBATCH -t 0-4:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -a 1-{NUM_NODES}
#SBATCH --qos=high
#SBATCH --export=ALL
#SBATCH --partition=queue1
"""
```

Note that the variables ``{JOB_NAME}``, ``{OUTPUT_FILE_DIRECTORY}``, and ``{NUM_NODES}`` will be formatted automatically by the ``exp_driver.py`` script. You must modify the SBATCH parameters to the correct values for your cluster. Note that that you must ensure that you configure your job so that at most one process is launched per machine, which may necessitate the use of the ``--exclusive`` flag.

## Run distributed experiment

You can run a experiment using the script exp_driver.py 

Example usage of exp_driver.py below
	
	python exp_driver.py --dataset_name ogbn-products --dataset_dir ../dataset --job_name test-job --num_nodes 4 --gpu_percent 0.999 --replication_factor 15 --joblist test_job_list.txt
	SBATCH_COMMAND: sbatch distributed_job_output/test-job_2022-10-21_06:24:58.509356/run.sh
	TAIL_COMMAND: tail -f distributed_job_output/test-job_2022-10-21_06:24:58.509356/logs/*.1.*
	CompletedProcess(args=['sbatch', 'distributed_job_output/test-job_2022-10-21_06:24:58.509356/run.sh'], returncode=0, stdout=b'Submitted batch job 7221\n', stderr=b'')
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"RUNNING"', '"RUNNING"', '"RUNNING"', '"RUNNING"']
	['"COMPLETED"', '"COMPLETED"', '"COMPLETED"', '"COMPLETED"']
	Done with job
	

**All output directories are timestamped, so you don't need to worry about naming conflicts**
But you still need to be able to get/parse output files easily. For this reason, the ***Job List*** argument (--joblist) is used.

The --joblist argument specifies a file where jobs will be appended. If you are running a collection of jobs
that, for example, only vary a particular parameter. You should group those jobs together by using the same
joblist file. This will make it easier for you to write parsing scripts that collect data from the logical
group of jobs that you ran.


## Manual running 

You can run exp_driver.py manually. Once you see the command print out the SBATCH_COMMAND and TAIL_COMMAND, you do not
need to stay in the interactive session (you can control+C the script). If you wish to view the output file manually, you
can copy and paste the "TAIL_COMMAND" that is provided. If you want to inspect the exact sbatch script that is generated
by the script, you can view it by looking at the SBATCH_COMMAND file.


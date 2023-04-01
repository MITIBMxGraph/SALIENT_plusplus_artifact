import os
import time
import datetime
import argparse
from pathlib import Path
import subprocess
from exp_utils.parser import parse_joblist, ExperimentResultGroup
from exp_utils.utils import verify_dataset_exists, run_experiment

parser = argparse.ArgumentParser(description="run distributed experiment")
parser.add_argument("--dataset_name", help="name of dataset", type=str, required=True, nargs="*")
parser.add_argument("--dataset_dir", help="directory to partitioned dataset, ordered by VIP weights.", type=str, required=True)
parser.add_argument("--num_nodes", help="Number of nodes to run on (default: 4)", type=int, required=False, default=[8], nargs="*")
parser.add_argument("--gpu_percent", help="Fixed gpu_percent to use during experiments (default: 0.15)", type=float, required=False, default=0.15)
parser.add_argument("--replication_factor_list", help="List of replication factors to run experiments on. Replication factors are given as integers N, where N/100 represents the fraction of data replicated in the system. Note that N may be greater than 100 when the number of partitions is greater than 2.", type=int, required=False, default=[0,1,2,4,8,32,64], nargs="*")
parser.add_argument("--parse_results_only", help="list number of nodes to run on", type=bool, required=False, default=False)
parser.add_argument("--joblist_directory", help="Location to place generated joblists.", type=str, required=False, default="./experiment_joblists")
parser.add_argument("--run_local", help="Set to 1 to run locally on a single machine with multiple gpus.", type=int, required=False, default=0)
parser.add_argument("--num_samplers_per_gpu", help="If set to 0 (default), default value in exp_driver.py is used. You may need to change this to be a smaller number in the multi-gpu single-machine setting.", type=int, required=False, default=0)
args = parser.parse_args()

if not args.parse_results_only:
    for num_nodes in args.num_nodes:
        for dataset_name in args.dataset_name:
            for rfactor in args.replication_factor_list:
                #run_experiment(dataset_name=dataset_name, num_nodes=num_nodes, gpu_percent=args.gpu_percent, replication_factor=rfactor, dataset_dir=args.dataset_dir, num_epochs=10, job_list_file="experiment-varyrfactor-" + args.dataset_name + ".txt")
                gpupercent = args.gpu_percent
                n_nodes = num_nodes
                run_experiment(args, dataset_name=dataset_name, num_nodes=n_nodes, gpu_percent=gpupercent, replication_factor=rfactor, dataset_dir=args.dataset_dir, num_epochs=10, job_list_file="experiment-varyrfactor.txt")


result_list = parse_joblist(Path(args.joblist_directory) / "experiment-varyrfactor.txt")
result_group = ExperimentResultGroup(result_list, sort_by=['dataset_name', 'dataset_dir', 'num_nodes', 'gpu_percent', 'replication_factor'])

import prettytable
tab = prettytable.PrettyTable()
tab.field_names = ['Dataset', 'Num Nodes', 'GPU Percent', 'Replication Factor', 'Per-Epoch time']

for i in range(result_group.size()):
    row = [result_group.get_attribute('dataset_dir',i) + '/' + result_group.get_attribute('dataset_name',i)] +\
            [str(result_group.get_attribute('num_nodes',i)) + ("" if int(result_group.get_attribute('run_local',i))==0 else " (local exec)")] +\
            [str(result_group.get_attribute('gpu_percent',i))]+\
            [str(result_group.get_attribute('replication_factor',i))]+\
            [result_group.get_total_time(i)]
    tab.add_row(row)
print(tab.get_string())

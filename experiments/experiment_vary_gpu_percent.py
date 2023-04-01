import os
import time
import datetime
import argparse
from pathlib import Path
import subprocess
from exp_utils.parser import parse_joblist, ExperimentResultGroup
from exp_utils.utils import verify_dataset_exists, run_experiment

parser = argparse.ArgumentParser(description="run distributed experiment")
parser.add_argument("--dataset_name", help="name of dataset", type=str, required=True)
parser.add_argument("--dataset_dir_viporder", help="directory to partitioned dataset, ordered by VIP weights.", type=str, required=True)
parser.add_argument("--dataset_dir_inputorder", help="directory to partitioned dataset, with default input ordering.", type=str, required=True)
parser.add_argument("--num_nodes", help="Number of nodes to run on (default: 4)", type=int, required=False, default=4)
parser.add_argument("--replication_factor", help="Fixed replication factor to use during experiments (default: 15)", type=int, required=False, default=15)
parser.add_argument("--parse_results_only", help="list number of nodes to run on", type=bool, required=False, default=False)
parser.add_argument("--joblist_directory", help="Location to place generated joblists.", type=str, required=False, default="./experiment_joblists")
parser.add_argument("--run_local", help="Set to 1 to run locally on a single machine with multiple gpus.", type=int, required=False, default=0)
parser.add_argument("--num_samplers_per_gpu", help="If set to 0 (default), default value in exp_driver.py is used. You may need to change this to be a smaller number in the multi-gpu single-machine setting.", type=int, required=False, default=0)
parser.add_argument("--gpu_percent_list", help="List of gpu percentages to run", type=float, required=False, default=[0, 0.1, 0.25, 0.5, 0.75, 1.0], nargs="*")
args = parser.parse_args()

if not args.parse_results_only:
    for gpu_percent in args.gpu_percent_list:
        run_experiment(args,dataset_name=args.dataset_name, num_nodes=args.num_nodes, gpu_percent=gpu_percent, replication_factor=args.replication_factor, dataset_dir=args.dataset_dir_inputorder, num_epochs=10, job_list_file="experiment-varygpupercent.txt", label="noreorder")
    
    for gpu_percent in args.gpu_percent_list:
        run_experiment(args,dataset_name=args.dataset_name, num_nodes=args.num_nodes, gpu_percent=gpu_percent, replication_factor=args.replication_factor, dataset_dir=args.dataset_dir_viporder, num_epochs=10, job_list_file="experiment-varygpupercent.txt", label="vip")


result_list = parse_joblist(Path(args.joblist_directory) / "experiment-varygpupercent.txt")
result_group = ExperimentResultGroup(result_list, sort_by=['dataset_name', 'dataset_dir', 'label', 'num_nodes', 'replication_factor', 'gpu_percent'])

import prettytable
tab = prettytable.PrettyTable()
tab.field_names = ['Dataset', 'System', 'Num Nodes', 'Replication Factor', 'GPU Percent', 'Per-Epoch time']

for i in range(result_group.size()):
    row = [result_group.get_attribute('dataset_dir',i) + '/' + result_group.get_attribute('dataset_name',i)] +\
            ["VIP Order" if result_group.get_attribute('label',i) == 'vip' else "Input Order"] +\
            [str(result_group.get_attribute('num_nodes',i)) + ("" if int(result_group.get_attribute('run_local',i))==0 else " (local exec)")] +\
            [str(result_group.get_attribute('replication_factor',i))]+\
            [str(result_group.get_attribute('gpu_percent',i))]+\
            [result_group.get_total_time(i)]
    tab.add_row(row)
print(tab.get_string())


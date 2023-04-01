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
parser.add_argument("--num_nodes", help="Number of nodes to run on (default: 4)", type=int, required=True, default=8)
parser.add_argument("--gpu_percent", help="Fixed gpu_percent to use during experiments (default: 0.15)", type=float, required=False, default=0.15)
parser.add_argument("--epochs", help="Number of epochs to train.", type=int, required=False, default=30)
parser.add_argument("--replication_factor", help="Fixed replication factor expressed as integer N to use during experiments (default: 15)", type=int, required=False, default=15)
parser.add_argument("--parse_results_only", help="list number of nodes to run on", type=bool, required=False, default=False)
parser.add_argument("--joblist_directory", help="Location to place generated joblists.", type=str, required=False, default="./experiment_joblists")
parser.add_argument("--run_local", help="Set to 1 to run locally on a single machine with multiple gpus.", type=int, required=False, default=0)
parser.add_argument("--num_samplers_per_gpu", help="If set to 0 (default), default value in exp_driver.py is used. You may need to change this to be a smaller number in the multi-gpu single-machine setting.", type=int, required=False, default=0)
parser.add_argument("--make_deterministic",
                   help="Make the training/inference deterministic. Comes with a performance penalty --- due, in part, to device to host data transfers to pageable memory by deterministic versions of algorithms.",
                   action="store_true")
args = parser.parse_args()

if not args.parse_results_only:
    for dataset_name in args.dataset_name:
        run_experiment(args, dataset_name=dataset_name, num_nodes=args.num_nodes, gpu_percent=args.gpu_percent, replication_factor=args.replication_factor, dataset_dir=args.dataset_dir, num_epochs=args.epochs, job_list_file="experiment-accuracy.txt")

result_list = parse_joblist(Path(args.joblist_directory) / "experiment-accuracy.txt")
result_group = ExperimentResultGroup(result_list, sort_by=['dataset_name', 'dataset_dir', 'num_nodes', 'gpu_percent', 'replication_factor'])

import prettytable
tab = prettytable.PrettyTable()
tab.field_names = ['Dataset', 'Num Nodes', 'GPU Percent', 'Replication Factor', 'Per-Epoch time', 'Valid accuracy', 'Test accuracy']

for i in range(result_group.size()):
    row = [result_group.get_attribute('dataset_dir',i) + '/' + result_group.get_attribute('dataset_name',i)] +\
            [str(result_group.get_attribute('num_nodes',i)) + ("" if int(result_group.get_attribute('run_local',i))==0 else " (local exec)")] +\
            [str(result_group.get_attribute('gpu_percent',i))]+\
            [str(result_group.get_attribute('replication_factor',i))]+\
            [result_group.get_total_time(i)] +\
            [str(result_group.get_valid_accuracy(i))]+\
            [str(result_group.get_test_accuracy(i))]

    tab.add_row(row)
print(tab.get_string())


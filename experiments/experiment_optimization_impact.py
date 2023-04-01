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
parser.add_argument("--dataset_dir", help="directory to partitioned dataset", type=str, required=True)
parser.add_argument("--salient_dataset_dir", help="directory to regular dataset for vanilla SALIENT", type=str, required=False, default = None)
parser.add_argument("--num_nodes_list", help="list number of nodes to run on", type=int, required=True, default=[2,4,8], nargs="*")
parser.add_argument("--parse_results_only", help="if set to 1, just parse existing results", type=bool, required=False, default=False)
parser.add_argument("--joblist_directory", help="Location to place generated joblists.", type=str, required=False, default="./experiment_joblists")
parser.add_argument("--run_local", help="Set to 1 to run locally on a single machine with multiple gpus.", type=int, required=False, default=0)
parser.add_argument("--num_samplers_per_gpu", help="If set to 0 (default), default value in exp_driver.py is used. You may need to change this to be a smaller number in the multi-gpu single-machine setting.", type=int, required=False, default=0)
parser.add_argument("--vip_only",
                   help="Only run VIP-version of code.",
                   action="store_true")
args = parser.parse_args()

if args.salient_dataset_dir == None:
    args.salient_dataset_dir = args.dataset_dir

if not args.parse_results_only:
    for num_nodes in args.num_nodes_list:
        if not args.vip_only:
            run_experiment(args,dataset_name=args.dataset_name, num_nodes=num_nodes, gpu_percent=0.0, replication_factor=0, dataset_dir=args.salient_dataset_dir, job_name="experiment-opt-impact", num_epochs=10, job_list_file="optimpact_"+args.dataset_name+".txt", pipeline_disabled=True, distribute_data = 0, label="salient")
            run_experiment(args,dataset_name=args.dataset_name, num_nodes=num_nodes, gpu_percent=0.0, replication_factor=0, dataset_dir=args.dataset_dir, job_name="experiment-opt-impact", num_epochs=10, job_list_file="optimpact_"+args.dataset_name+".txt", pipeline_disabled=True, label="partition")
            run_experiment(args,dataset_name=args.dataset_name, num_nodes=num_nodes, gpu_percent=0, replication_factor=0, dataset_dir=args.dataset_dir, job_name="experiment-opt-impact", num_epochs=10, job_list_file="optimpact_"+args.dataset_name+".txt", pipeline_disabled=False, label="pipeline")
        gpupercent = 0.15
        rfactor = 4*num_nodes
        if args.dataset_name.find("MAG240") != -1:
            rfactor = 2*num_nodes
            gpupercent = 0.1
        run_experiment(args, dataset_name=args.dataset_name, num_nodes=num_nodes, gpu_percent=gpupercent, replication_factor=rfactor, dataset_dir=args.dataset_dir, job_name="experiment-opt-impact", num_epochs=10, job_list_file="optimpact_"+args.dataset_name+".txt", pipeline_disabled=False, label="vip")


result_list = parse_joblist(str(Path(args.joblist_directory) / ("optimpact_"+args.dataset_name+".txt")))
result_group = ExperimentResultGroup(result_list, sort_by=['label', 'dataset_name', 'dataset_dir', 'num_nodes',  'gpu_percent', 'replication_factor'], attr_sortkey_map = {'label' : {'salient' : 0, 'partition' : 1, 'pipeline' : 2, 'vip' : 3}})


import prettytable
tab = prettytable.PrettyTable()
tab.field_names = ['Dataset', 'System version', 'Num Nodes', 'GPU Percent', 'Replication Factor', 'Pipeline Comm', 'Per-Epoch time']

for i in range(result_group.size()):
    row = [result_group.get_attribute('dataset_dir',i) + '/' + result_group.get_attribute('dataset_name',i)] +\
            [result_group.get_attribute('label',i)] +\
            [str(result_group.get_attribute('num_nodes',i)) + ("" if int(result_group.get_attribute('run_local',i))==0 else " (local exec)")] +\
            [str(result_group.get_attribute('gpu_percent',i))]+\
            [str(result_group.get_attribute('replication_factor',i))]+\
            ["N/A" if result_group.get_attribute('label',i) == 'salient' else ("Yes" if not result_group.get_attribute('pipeline_disabled',i) else "No")]+\
            [result_group.get_total_time(i)]
    tab.add_row(row)
print(tab.get_string())


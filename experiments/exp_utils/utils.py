import os
import pathlib
from pathlib import Path
import colorama
from colorama import Fore, Back, Style
import subprocess

# Requires args.run_local, args.make_deterministic, args.num_samplers_per_gpu, args.joblist_directory
def run_experiment(args, *pos_args, job_list_file, dataset_name, num_nodes, gpu_percent, replication_factor, dataset_dir, job_name=None, num_epochs, pipeline_disabled=False, distribute_data = 1, label="nolabel"):
    if not verify_dataset_exists(dataset_name = dataset_name, dataset_dir = dataset_dir, num_nodes = num_nodes, distribute_data = distribute_data):
        print("Skipping experiment due to not being able to verify the requested dataset exists on disk.")
        return

    if job_name == None:
        job_name = job_list_file.replace(".txt", "")

    job_list_file = str(Path(args.joblist_directory) / job_list_file)

    additional_args = []
    #num_samplers_per_gpu = ""
    if hasattr(args,"num_samplers_per_gpu") and args.num_samplers_per_gpu != 0:
        #num_samplers_per_gpu = "--num_samplers " + str(args.num_samplers_per_gpu)
        additional_args = additional_args + f"--num_samplers {args.num_samplers_per_gpu}".split(" ")
    
    if dataset_name.find("MAG240") != -1:
        additional_args = additional_args + "--train_fanouts 25 15 --test_fanouts 25 15 --valid_fanouts 25 15 --num_hidden 1024".split(" ")

    if pipeline_disabled:
        additional_args = additional_args + f"--pipeline_disabled {pipeline_disabled}".split(" ")

    if hasattr(args, "make_deterministic") and args.make_deterministic:
        additional_args.append("--make_deterministic")


    #if dataset_name.find("MAG240") != -1:
    #    if pipeline_disabled:
    #        cmd = f"python exp_driver.py --dataset_name {dataset_name} --num_nodes {num_nodes} --gpu_percent {gpu_percent} --replication_factor {replication_factor} --dataset_dir {dataset_dir} --job_name {job_name} --num_epochs {num_epochs} --joblist {job_list_file} --test_epoch_frequency {num_epochs} --pipeline_disabled {pipeline_disabled} --distribute_data {distribute_data} --label={label} --run_local {args.run_local} {num_samplers_per_gpu} --train_fanouts 25 15 --test_fanouts 25 15 --valid_fanouts 25 15 --num_hidden 1024"
    #    else:
    #        cmd = f"python exp_driver.py --dataset_name {dataset_name} --num_nodes {num_nodes} --gpu_percent {gpu_percent} --replication_factor {replication_factor} --dataset_dir {dataset_dir} --job_name {job_name} --num_epochs {num_epochs} --joblist {job_list_file} --test_epoch_frequency {num_epochs} --distribute_data {distribute_data} --label={label} --run_local {args.run_local} {num_samplers_per_gpu} --train_fanouts 25 15 --test_fanouts 25 15 --valid_fanouts 25 15 --num_hidden 1024"
    #else:
    #    if pipeline_disabled:
    #        cmd = f"python exp_driver.py --dataset_name {dataset_name} --num_nodes {num_nodes} --gpu_percent {gpu_percent} --replication_factor {replication_factor} --dataset_dir {dataset_dir} --job_name {job_name} --num_epochs {num_epochs} --joblist {job_list_file} --test_epoch_frequency {num_epochs} --pipeline_disabled {pipeline_disabled} --distribute_data {distribute_data} --label={label} --run_local {args.run_local} {num_samplers_per_gpu}"
    #    else:
    run_local = 0
    if hasattr(args,"run_local"):
        run_local = args.run_local
    cmd = f"python exp_driver.py --dataset_name {dataset_name} --num_nodes {num_nodes} --gpu_percent {gpu_percent} --replication_factor {replication_factor} --dataset_dir {dataset_dir} --job_name {job_name} --num_epochs {num_epochs} --joblist {job_list_file} --test_epoch_frequency {num_epochs} --distribute_data {distribute_data} --label={label} --run_local {run_local}"


    cmd_list = []
    for x in cmd.strip().split(' '):
        if len(x.strip()) > 0:
            cmd_list.append(x)
    for x in additional_args:
        if len(x.strip()) > 0:
            cmd_list.append(x)
    print("Running command: " + " ".join(cmd_list))
    subprocess.run(cmd_list)



def verify_dataset_exists(*posargs, dataset_name, dataset_dir, num_nodes, distribute_data):
    if distribute_data == 1 and num_nodes < 2:
        print(Fore.RED + f'[Error] You are running with distribute_data=1 and num_nodes < 2. This experiment will not run.'+ Style.RESET_ALL)
        return False
    elif distribute_data == 0:
        # verify that the local data exists.
        actual_dataset_dir = Path(dataset_dir) / dataset_name
        if actual_dataset_dir.exists():
            return True
        else:
            print(Fore.YELLOW + f'[Warning] There does not exist a {num_nodes}-way partitioned dataset for {dataset_name} in {dataset_dir}. Expected to find this dataset at path: {actual_dataset_dir}'+ Style.RESET_ALL)
            print(Fore.YELLOW + f'[Warning] You will either need to download the appropriate partitioned dataset or follow the instructions in the README for generating a partitioned dataset.'+ Style.RESET_ALL)
            return False
    else:
        partitioned_subdir_name = f'metis-reordered-k{num_nodes}'
        actual_dataset_dir = Path(dataset_dir) / partitioned_subdir_name / dataset_name
        if actual_dataset_dir.exists():
            return True
        else:
            print(Fore.YELLOW + f'[Warning] There does not exist a {num_nodes}-way partitioned dataset for {dataset_name} in {dataset_dir}. Expected to find this dataset at path: {actual_dataset_dir}'+ Style.RESET_ALL)
            print(Fore.YELLOW + f'[Warning] You will either need to download the appropriate partitioned dataset or follow the instructions in the README for generating a partitioned dataset.'+ Style.RESET_ALL)
            return False





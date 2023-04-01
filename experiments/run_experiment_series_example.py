import subprocess
import time

def run_experiment(*pos_args, job_list_file, dataset_name, num_nodes, gpu_percent, replication_factor, dataset_dir, job_name, num_epochs):
    cmd = f"python exp_driver.py --dataset_name {dataset_name} --num_nodes {num_nodes} --gpu_percent {gpu_percent} --replication_factor {replication_factor} --dataset_dir {dataset_dir} --job_name {job_name} --num_epochs {num_epochs} --joblist {job_list_file}"
    print("Running command: " + cmd)
    subprocess.run(cmd.split(' '))


for rfactor in [0, 2, 4, 8, 16, 32]:
    time.sleep(5)
    run_experiment(dataset_name="ogbn-products", num_nodes=4, gpu_percent=0.999, replication_factor=rfactor, dataset_dir="tfk_partitioned_dataset", job_name="vary-replication-products", num_epochs=10, job_list_file="products_vary_rfactor_4nodes.txt")

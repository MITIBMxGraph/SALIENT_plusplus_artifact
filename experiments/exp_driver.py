import os
import time
import datetime
import argparse
from pathlib import Path
import subprocess


SLURM_CONFIG="""#SBATCH -J sb_{JOB_NAME}
#SBATCH -o {OUTPUT_FILE_DIRECTORY}/%x-%A.%a.out
#SBATCH -e {OUTPUT_FILE_DIRECTORY}/%x-%A.%a.err
#SBATCH --nodes=1
#SBATCH -t 0-1:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -a 1-{NUM_MACHINES}
#SBATCH --qos=high
#SBATCH --export=ALL
#SBATCH --partition=queue1
"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run distributed experiment")
    parser.add_argument("--dataset_name", help="name of dataset", type=str, required=True)
    parser.add_argument("--dataset_dir", help="directory to partitioned dataset", type=str, required=True)
    parser.add_argument("--job_name", help="short name of job (no spaces)", type=str, required=True)
    parser.add_argument("--num_nodes", help="number of nodes to run on", type=int, required=True)
    parser.add_argument("--gpu_percent", help="percent of data to put on GPU", type=float, required=True)
    parser.add_argument("--replication_factor", help="percentage of data to replicate to reduce comm. Integer valid between 0 and 100", type=int, required=True)
    parser.add_argument("--train_batch_size", help="size of training batch", type=int, default=1024)
    parser.add_argument("--valid_batch_size", help="size of valid batch size", type=int, default=1024)
    parser.add_argument("--test_batch_size", help="size of test batch size", type=int, default=1024)
    parser.add_argument("--train_fanouts", help="training fanouts", type=int, default=[15, 10, 5], nargs="*")
    parser.add_argument("--valid_fanouts", help="valid fanouts", type=int, default=[20, 20, 20], nargs="*")
    parser.add_argument("--test_fanouts", help="test fanouts", type=int, default=[20, 20, 20], nargs="*")
    parser.add_argument("--test_epoch_frequency", help="Number of epochs before running on validation test set.", type=int, default=1)
    parser.add_argument("--num_epochs", help="num epochs", type=int, default=25)
    parser.add_argument("--num_hidden", help="num hidden features", type=int, default=256)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.001)
    parser.add_argument("--model_name", help="name of model", type=str, default="SAGE")
    parser.add_argument("--interactive", help="run interactively and start to tail the output", type=bool, default=True)
    parser.add_argument("--joblist", type=str, default="job_list.txt")
    parser.add_argument("--distributed_job_root", help="root dir for collections of jobs to go into", type=str, default="./distributed_job_output")
    parser.add_argument("--num_samplers", help="num sampling workers", type=int, default=15)
    parser.add_argument("--pipeline_disabled", help="disable the pipeline", type=bool, default=False)
    parser.add_argument("--distribute_data", help="Distribute the data among devices/machines according to a graph partitioning.", type=int, default=1)
    parser.add_argument("--run_local", help="Run on the local machine.", type=int, default=0)
    parser.add_argument("--label", help="Label is only used for describing a subset of jobs with an human-readable name.", type=str, default=0)
    parser.add_argument("--num_devices_per_node", help="Num devices per node.", type=int, default=1)
    parser.add_argument("--make_deterministic",
                        help="Make the training/inference deterministic. Comes with a performance penalty --- due, in part, to device to host data transfers to pageable memory by deterministic versions of algorithms.",
                        action="store_true")

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    partitioned_dataset = os.path.realpath(args.dataset_dir)
    job_name = args.job_name
    job_root_dir = Path(args.distributed_job_root)
    
    def get_job_directory(job_name):
        time_string = str(datetime.datetime.now()).replace(" ", "_")
        return job_name +"_"+time_string
    
    script_dir = Path(script_dir)
    job_dir = job_root_dir / Path(get_job_directory(job_name))
    log_dir = job_dir / Path("logs")
    output_root_dir = job_dir / Path("experiment_testbed")
    nodelist_dir = job_dir / Path("nodelist_experiment_testbed") 
    
    sbatch_file = job_dir / Path('run.sh')
    job_root_dir.mkdir(exist_ok=True) 
    
    job_dir.mkdir()
    log_dir.mkdir()
    nodelist_dir.mkdir()
    
    output_root_dir.mkdir(exist_ok=True)
    if args.num_nodes % args.num_devices_per_node != 0:
        print("[Error] num_nodes must be divisible by num_devices_per_node")
        exit(1)
    VARS = dict()
    VARS["NUM_NODES"] = args.num_nodes
    VARS["NUM_MACHINES"] = args.num_nodes // args.num_devices_per_node
    VARS["print NF"] = "{print NF}"
    VARS["OUTPUT_FILE_DIRECTORY"] = str(log_dir)
    VARS["JOB_NAME"] = args.job_name
    VARS["TRAIN_BATCH_SIZE"] = args.train_batch_size
    VARS["VALID_BATCH_SIZE"] = args.valid_batch_size
    VARS["TEST_BATCH_SIZE"] = args.test_batch_size
    VARS["GPU_PERCENT"] = args.gpu_percent
    VARS["SCRIPT_DIR"] = str(script_dir) + "/"
    VARS["PYTHONPATH"] = "$SCRIPT_DIR/../"
    if args.distribute_data:
        if not args.run_local:
            VARS["PARTITIONED_FEATURE_DATASET_ROOT"] = args.dataset_dir+"/metis-reordered-k$NUM_NODES"
        else:
            VARS["PARTITIONED_FEATURE_DATASET_ROOT"] = args.dataset_dir+"/metis-reordered-k"+str(args.num_nodes)
    else:
        VARS["PARTITIONED_FEATURE_DATASET_ROOT"] = args.dataset_dir

    VARS["OUTPUT_ROOT"] = str(job_dir) # "$SCRIPT_DIR/distributed_job_output"
    VARS["EPOCHS"]=args.num_epochs
    VARS["TEST_EPOCH_FREQUENCY"]=args.test_epoch_frequency
    VARS["CACHE_CREATION_EPOCHS"]=2
    VARS["REPLICATION_FACTOR"]=args.replication_factor
    VARS["EXECUTION_MODE"]="computation"
    VARS["COMPUTATION_MODE"]="frequency_cache"
    VARS["DATASET_NAME"]=args.dataset_name
    VARS["LOAD_BALANCE_SCHEME"] = "federated"
    VARS["TRAIN_FANOUTS"]=" ".join([str(x) for x in args.train_fanouts])
    VARS["TEST_FANOUTS"]= " ".join([str(x) for x in args.valid_fanouts])
    VARS["NUM_SAMPLING_WORKERS"]=args.num_samplers
    VARS["HIDDEN_SIZE"]=args.num_hidden
    VARS["LEARNING_RATE"]=args.learning_rate
    VARS["MODEL_NAME"]=args.model_name
    VARS["TRAIN_MAX_NUM_BATCHES"]=48
    VARS["NUM_LAYERS"] = str(len(args.train_fanouts))
    VARS["MAKE_DETERMINISTIC"] = "--make_deterministic" if args.make_deterministic else ""
    VARS["DISTRIBUTE_DATA"] = 1 if args.distribute_data else 0


    if args.pipeline_disabled:
        VARS["PIPELINE_DISABLED"] = "--pipeline_disabled"
    else: 
        VARS["PIPELINE_DISABLED"] = ""


    VARS["ONE_NODE_DDP"] = ""
    preVARS = dict()
    preVARS["SLURM_CONFIG"] = SLURM_CONFIG

    if args.run_local:
        VARS["SLURM_CONFIG"] = ""
        VARS["NUM_DEVICES_PER_NODE"] = args.num_nodes #args.num_devices_per_node
        VARS["ONE_NODE_DDP"] = "--one_node_ddp"
    else:
        for x in VARS.keys():
            VARS[x] = str(VARS[x])
        VARS["NUM_DEVICES_PER_NODE"] = args.num_devices_per_node
        preVARS["SLURM_CONFIG"] = preVARS["SLURM_CONFIG"].format(**VARS)
        VARS["SLURM_CONFIG"] = preVARS["SLURM_CONFIG"]
    for x in VARS.keys():
        VARS[x] = str(VARS[x])
    text = """#!/bin/bash
{SLURM_CONFIG}

# Configure the option below based on the nodes you requested from the cluster.
# 
# If you specified '#SBATCH --gres=gpu:k' then set NUM_DEVICES_PER_NODE=k
# If you specified '#SBATCH -a 1-N' then set NUM_NODES=N

if [ $SLURM_ARRAY_TASK_COUNT ]; then
NUM_DEVICES_PER_NODE={NUM_DEVICES_PER_NODE}
NUM_NODES={NUM_NODES}
#$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
#NUM_NODES=$SLURM_ARRAY_TASK_COUNT
else
NUM_DEVICES_PER_NODE={NUM_DEVICES_PER_NODE}
NUM_NODES=1
fi

# Most architectures train with fanouts 15,10,5
TRAIN_BATCH_SIZE={TRAIN_BATCH_SIZE}

# Typical validation and test fanouts are 20,20,20 so use a smaller batch size.
VALID_BATCH_SIZE={VALID_BATCH_SIZE}
TEST_BATCH_SIZE={TEST_BATCH_SIZE}


#
# Obtain script directory from ./run_distributed.sh and setup other paths.
# 	No need to change.
#
SCRIPT_DIR={SCRIPT_DIR} 

# Set JOB_NAME used for this script
export SLURMD_NODENAME=`hostname`
export PYTHONPATH={PYTHONPATH}

PARTITIONED_FEATURE_DATASET_ROOT={PARTITIONED_FEATURE_DATASET_ROOT}
OUTPUT_ROOT={OUTPUT_ROOT}

NSYS_ENABLE=0
NSYS_PROFILE_NAME="8nodes_nopipeline_nocache_federated_15_15_15_%h_%p"
#NSYS_PROFILE_CMD="/usr/local/cuda/bin/nsys profile -f true --cuda-flush-interval 360000 -o $SCRIPT_DIR/nsys_profiles/$NSYS_PROFILE_NAME"
NSYS_PROFILE_CMD="/home/ubuntu/miniconda3/envs/salient_dist13/pkgs/cuda-toolkit/bin/nsys profile -f true -o $SCRIPT_DIR/nsys_profiles/$NSYS_PROFILE_NAME"
if [ $NSYS_ENABLE -eq 1 ]; then NSYS="$NSYS_PROFILE_CMD"; else NSYS=""; fi

touch $OUTPUT_ROOT/nodelist_experiment_testbed/$SLURMD_NODENAME

EPOCHS={EPOCHS}
TEST_EPOCH_FREQUENCY={TEST_EPOCH_FREQUENCY}
CACHE_CREATION_EPOCHS={CACHE_CREATION_EPOCHS}
REPLICATION_FACTOR={REPLICATION_FACTOR}
EXECUTION_MODE="{EXECUTION_MODE}"
COMPUTATION_MODE="{COMPUTATION_MODE}"
DATASET_NAME="{DATASET_NAME}"
LOAD_BALANCE_SCHEME="{LOAD_BALANCE_SCHEME}"
TRAIN_FANOUTS="{TRAIN_FANOUTS}"
TEST_FANOUTS="{TEST_FANOUTS}"
NUM_SAMPLING_WORKERS={NUM_SAMPLING_WORKERS}
HIDDEN_SIZE={HIDDEN_SIZE}
LEARNING_RATE={LEARNING_RATE}
MODEL_NAME="{MODEL_NAME}"
TRAIN_MAX_NUM_BATCHES={TRAIN_MAX_NUM_BATCHES}
NUM_LAYERS={NUM_LAYERS}
DISTRIBUTE_DATA={DISTRIBUTE_DATA}


#sudo tc qdisc del dev enp0s5 root
#sudo tc qdisc add dev enp0s5 root fq limit 10240 flow_limit 1024 quantum 9015 maxrate 1gbit pacing
#sudo tc qdisc add dev enp0s5 root fq limit 10240 flow_limit 1024 maxrate 25gbit pacing
#sudo tc qdisc add dev enp0s5 root tbf latency 1msec burst 100mb rate 10gbit
#sudo tc qdisc add dev enp0s5 root tbf latency 200usec burst 500mb rate 10gbit

#sudo nvidia-smi --reset-gpu-clocks
#sudo nvidia-smi --compute-mode=1 

NCCL_NSOCKS_PERTHREAD=1 NCCL_SOCKET_NTHREADS=1 PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 $NSYS python -m driver.main \
    $DATASET_NAME experiment_testbed --load_balance_scheme $LOAD_BALANCE_SCHEME \
    --train_type serial --max_num_devices_per_node $NUM_DEVICES_PER_NODE \
    --total_num_nodes {NUM_MACHINES} --test_epoch_frequency $TEST_EPOCH_FREQUENCY \
    --execution_mode $EXECUTION_MODE --computation_mode $COMPUTATION_MODE \
    --cache_creation_epochs $CACHE_CREATION_EPOCHS \
    --cache_size $REPLICATION_FACTOR --epochs $EPOCHS \
    --overwrite_job_dir --output_root $OUTPUT_ROOT \
    --dataset_root=$PARTITIONED_FEATURE_DATASET_ROOT \
    --train_batch_size $TRAIN_BATCH_SIZE --test_batch_size $VALID_BATCH_SIZE \
    --lr $LEARNING_RATE --num_workers=$NUM_SAMPLING_WORKERS \
    --hidden_features=$HIDDEN_SIZE --num_layers=$NUM_LAYERS \
    --train_fanouts $TRAIN_FANOUTS \
    --batchwise_test_fanouts $TEST_FANOUTS \
    --final_test_fanouts $TEST_FANOUTS \
    --final_test_batchsize $TEST_BATCH_SIZE \
    --model_name $MODEL_NAME --ddp_dir $OUTPUT_ROOT/nodelist_experiment_testbed/ \
    --trials 1 --distribute_data $DISTRIBUTE_DATA {ONE_NODE_DDP} --performance_stats --train_max_num_batches $TRAIN_MAX_NUM_BATCHES --gpu_percent {GPU_PERCENT} {PIPELINE_DISABLED} {MAKE_DETERMINISTIC}

    """
    text = text.format(**VARS)
    
    
    
    open(str(sbatch_file), 'w+').write(text)
    if args.interactive:
        if args.run_local:
            print("LOCAL JOB COMMAND: bash " + str(sbatch_file))
            output = subprocess.run(["bash", str(sbatch_file)], check=True, capture_output=False)
            time_string = str(datetime.datetime.now()).replace(" ", "_")
            job_id = time_string  
        else:
            print("SBATCH_COMMAND: sbatch " + str(sbatch_file))
            print("TAIL_COMMAND: tail -f " + str(log_dir) + "/*.1.*")
            output = subprocess.run(["sbatch", str(sbatch_file)], check=True, capture_output=True)
            print (output)
            job_id = int(str(output.stdout).strip().split(" ")[-1].replace("'", "").replace("\\n", "").strip())
        open(args.joblist, '+a').write(str([job_id, args, os.path.realpath(str(log_dir)), os.path.realpath(str(job_dir))]) + '\n')

        while True and not args.run_local:
            time.sleep(5)
            output = subprocess.run(["squeue", "-j", str(job_id), "--format=\"%T\"", "--states=all"], check=True, capture_output=True)
            lines = str(output.stdout).replace("\\n", "\n").split("\n")
            lines = lines[1:-1]
            is_done = len(lines) > 0
            for x in lines:
                if x.find("COMPLETED") == -1:
                    is_done = False
            print(lines)
            if is_done:
                print("Done with job")
                break
        # Wait for it to get into waiting state
    else:
        print("Non interactive mode is not supported yet.")
        quit()
     


    #print(text)

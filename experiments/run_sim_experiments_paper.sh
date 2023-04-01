#!/usr/bin/env bash

DATASET_NAME=ogbn-papers100M
DATASET_DIR=../dataset
PARTITION_LABELS_DIR=../dataset/partition-labels
NUM_PARTITIONS=8
MINIBATCH_SIZE=1024
NUM_EPOCHS_EVAL=5
NUM_EPOCHS_VIP_SIM=2
CACHE_SCHEMES="degree-reachable num-paths-reachable halo-1hop vip-simulation vip-analytical"
REPLICATION_FACTORS="0 0.01 0.05 0.10 0.20 0.50 1.00"
OUTPUT_PREFIX=results-simulation/sim-comm
FANOUTS=("5 5 5" "15 10 5" "20 20 20")

PYTHON_MODULE_SIM=caching.experiment_communication_caching
PYTHON_MODULE_SHOW_RESULTS=caching.parse_communication_volume_results

THIS_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# run simulation experiments
for fanout in "${FANOUTS[@]}"; do
    PYTHONPATH="$THIS_SCRIPT_DIR/.." python -m $PYTHON_MODULE_SIM \
        --dataset_name $DATASET_NAME \
        --dataset_dir $DATASET_DIR \
        --partition_labels_dir $PARTITION_LABELS_DIR \
        --num_partitions $NUM_PARTITIONS \
        --fanouts $fanout \
        --minibatch_size $MINIBATCH_SIZE \
        --num_epochs_eval $NUM_EPOCHS_EVAL \
        --num_epochs_vip_sim $NUM_EPOCHS_VIP_SIM \
        --output_prefix $OUTPUT_PREFIX
done

# redisplay all results tables
echo ""
echo "=================================================="
echo " RESULTS"
echo "=================================================="
echo ""
for fanout in "${FANOUTS[@]}"; do
    FILENAME="$OUTPUT_PREFIX-$DATASET_NAME\
        -partitions-$NUM_PARTITIONS-minibatch-$MINIBATCH_SIZE\
        -fanout-${fanout// /-}-epochs-$NUM_EPOCHS_EVAL.pobj"
    FILENAME=${FILENAME// /}
    echo "*** $FILENAME ***"
    PYTHONPATH="$THIS_SCRIPT_DIR/.." python -m $PYTHON_MODULE_SHOW_RESULTS --path "$FILENAME"
    echo ""
done

exit 0

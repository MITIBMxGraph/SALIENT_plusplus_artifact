import argparse


class LoadFromFile (argparse.Action):
    # def __call__ (self, parser, namespace, values, option_string = None):
    #    with values as f:
    #        # parse arguments in the file and store them in the target namespace
    #        parser.parse_args(f.read().split(), namespace)

    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            contents = f.read()
            # parse arguments in the file and store them in a blank namespace
            data = parser.parse_args(contents.split(), namespace=None)
            for k, v in vars(data).items():
                if k not in ["dataset_name", "job_name"]:
                    setattr(namespace, k, v)


def make_parser():
    parser = argparse.ArgumentParser(description="Start an experiment")
    parser.add_argument("dataset_name",
                        help="Name of the OGB dataset",
                        type=str)
    parser.add_argument("--gpu_percent", help="percent of data on gpu. note min fraction is 1/1000 and max fraction is 999/1000", type=float)
    parser.add_argument("job_name",
                        help="Name of the Job",
                        type=str)
    parser.add_argument("--config_file", type=open,
                        action=LoadFromFile, default=None)
    parser.add_argument("--dataset_root",
                        help="Dataset root path",
                        type=str, default=f"fast_dataset/")
    parser.add_argument("--output_root",
                        help="The root of output storage",
                        type=str, default=f"job_output/")
    parser.add_argument("--datacollector_root",
                        help="The root of output when collecting batch execution data (e.g. which vertices get cache hits).",
                        type=str, default=f"data_collector/")
    parser.add_argument("--datacollector_save",
                        help="Whether to write collected data to file (e.g. which vertices get cache hits).",
                        action="store_true")
    parser.add_argument("--pipeline_disabled",
                        help="Whether to disable pipelining",
                        action="store_true")
    parser.add_argument("--ddp_dir",
                        help="Coordination directory for ddp multinode jobs",
                        type=str, default=f"NONE")
    parser.add_argument("--do_test_run",
                        help="Only run inference on the test set",
                        action="store_true")
    parser.add_argument("--make_deterministic",
                        help="Make the training/inference deterministic. Comes with a performance penalty --- due, in part, to device to host data transfers to pageable memory by deterministic versions of algorithms.",
                        action="store_true")
    parser.add_argument("--performance_stats",
                        help="Collect detailed performance statistics.",
                        action="store_true")
    parser.add_argument("--do_test_run_filename",
                        help="The filename of model to load for the test run",
                        type=str, default=f"NONE", nargs='*')
    parser.add_argument("--overwrite_job_dir",
                        help="If a job directory exists, delete it",
                        action="store_true")
    parser.add_argument("--total_num_nodes",
                        help="Total number of nodes to use",
                        type=int, default=1)
    parser.add_argument("--patience",
                        help="Patience to use in the LR scheduler",
                        type=int, default=1000)
    parser.add_argument("--max_num_devices_per_node",
                        help="Maximum number of devices per node",
                        type=int, default=1)
    parser.add_argument("--execution_mode",
                        help="Whether to run computation with the model (train, test, etc.) or simulate communication (no forward/backward passes).",
                        choices=["computation", "communication_simulation"], default="computation")
    parser.add_argument("--computation_mode",
                        help="If execution_mode == 'computation', whether to run normally or create a cache before computation and then use it to allievate communicaiton.",
                        choices=["normal", "frequency_cache"], default="normal")
    parser.add_argument("--communication_simulation_mode",
                        help="""
                            If execution_mode == 'communication_simulation', whether to simulate:
                                normal execution,
                                after creating a cache,
                            """,
                        choices=["normal", "frequency_cache"], default="normal")
    parser.add_argument("--load_balance_scheme",
                        help="""
                             Fully Random (default):
                                Minibatches are fully random. At the beginning of each epoch the rank 0 machine shuffles all training nodes and scatters
                                1/kth of the shuffled training indices to each of the k machines. Each iteration each machine will then compute a microbatch
                                composed of the next minibatch_size/k vertices it gathered
                             Federated:
                                Minibatches are not fully random. At the beginning of each epoch, each of the k machines shuffles the training ids on its
                                partition, then each iteration each machine will compute a microbatch composed of the next minibatch_size/k vertices that it
                                shuffled locally. Note: If the training nodes are not partitioned equally this can also lead to an uneven number of iterations/microbatches
                                for each machine. Currently we force even number of iterations which can lead to uneven minibatch sizes for different machines.
                            """,
                        choices=["fully_random", "federated"], default="fully_random")
    parser.add_argument("--hidden_features",
                        help="Number of hidden features",
                        type=int, default=256)
    parser.add_argument("--num_layers",
                        help="Number of layers",
                        type=int, default=3)
    parser.add_argument("--lr",
                        help="Learning rate",
                        type=float, default=0.003)
    parser.add_argument("--num_workers",
                        help="Number of workers",
                        type=int, default=70)
    parser.add_argument("--one_node_ddp",
                        help="Do DDP when total_num_nodes=1",
                        action="store_true")
    parser.add_argument("--use_lrs",
                        help="Use learning rate scheduler",
                        action="store_true")
    parser.add_argument("--trials",
                        help="Number of trials to run",
                        type=int, default=10)
    parser.add_argument("--epochs",
                        help="Total number of epochs to train or to simulate communication",
                        type=int, default=21)
    parser.add_argument("--cache_creation_epochs",
                        help="Total number of epochs to run before computation or communication simulation to determine which vertices to cache.",
                        type=int, default=0)
    parser.add_argument("--cache_size",
                        help="""
                            UPDATE: basically replication factor but just haven't updated the names yet.
                            It is a percentage to replicate.
                            E.g. passing in 30 here will give an intended replication factor of 30%.
                            Note: this is different from the effective replication factor.
                            """,
                        type=int, default=0)

    parser.add_argument("--cache_strategy",
                        help="""
                            UPDATE: basically replication factor but just haven't updated the names yet.
                            It is a percentage to replicate.
                            E.g. passing in 30 here will give an intended replication factor of 30%.
                            Note: this is different from the effective replication factor.
                            """,
                        type=str, default="vip")

    parser.add_argument("--model_name",
                        help="Name of the model to use",
                        type=str, default="SAGE")
    # See driver/main.py/get_model_type() for available choices
    parser.add_argument("--train_sampler",
                        help="Training sampler",
                        type=str, default="FastSampler")
    parser.add_argument("--verbose",
                        help="Print log entries to stdout",
                        action="store_true")
    parser.add_argument("--train_batch_size",
                        help="Size of training batches",
                        type=int, default=1024)
    parser.add_argument("--train_max_num_batches",
                        help="Max number of training batches waiting in queue",
                        type=int, default=100)
    parser.add_argument("--train_fanouts",
                        help="Training fanouts",
                        type=int, default=[15, 10, 5], nargs="*")
    parser.add_argument("--train_prefetch",
                        help="Prefetch for training",
                        type=int, default=1),
    parser.add_argument("--distribute_data",
                        help="Distribute the node feature data across GPUs and use appropriate samplers/transferers.",
                        type=int, default=0)
    parser.add_argument("--train_type",
                        help="Training Type",
                        type=str, default="serial",
                        choices=("serial", "dp"))
    parser.add_argument("--test_epoch_frequency",
                        help="Number of epochs to train before testing occurs",
                        type=int, default=20)
    parser.add_argument("--test_batch_size",
                        help="Size of testing batches",
                        type=int, default=4096)
    parser.add_argument("--test_max_num_batches",
                        help="Max number of testing batches waiting in queue",
                        type=int, default=50)
    parser.add_argument("--batchwise_test_fanouts",
                        help="Testing fanouts",
                        type=int, default=[20, 20, 20], nargs="*")
    parser.add_argument("--test_prefetch",
                        help="Prefetch for testing",
                        type=int, default=1)
    parser.add_argument("--test_type",
                        help="Testing type",
                        type=str, default="batchwise",
                        choices=("layerwise", "batchwise"))

    parser.add_argument("--final_test_batchsize",
                        help="Size of testing batches",
                        type=int, default=1024)
    parser.add_argument("--final_test_fanouts",
                        help="Testing fanouts",
                        type=int, default=[20, 20, 20], nargs="*")

    return parser

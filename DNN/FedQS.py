import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
sys.path.append('code')
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import tensorflow as tf
from FedAvg_tfd import FedAvg_tfd



import json
import wandb
import pprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


dataset ="inat"
train_group = "nat"
test_group = "G1"

if dataset == "CIFAR":
    K = 10
elif dataset == "CIFAR100":
    K = 100
elif dataset == "Tinyimagenet":
    K = 200
elif dataset == "femnist":
    K = 62
elif dataset == "inat":
    N = 9275
    K = 1203
elif dataset == "FLAIR":
    N = 41131
    K = 1628

r = int(0.3 * N)
T = 2000
nworkers = 1

base_policy_list = [ "uniform", "KL", "CBS", "POC", "ODFL"]
balanced_policy_list = [f"{policy}_balanced" for policy in base_policy_list]
policy_name = base_policy_list[-1]

def get_run_config():

    if policy_name == "uniform":
        sweep_id = "ezpsha5s"
    elif policy_name == "CBS":
        sweep_id = "3walnpcs"
    elif policy_name == "POC":
        sweep_id = "t3gtgj5b"
    else:
        sweep_id = None

    if sweep_id is None:
        run_config = {
            'epoch': 2000, 
            'n_class': 1203, 
            'n_steps': 100, 
            'n_clients': 9275, 
            'batch_size': 8, 
            'n_available': 2782, 
            'policy_name': policy_name, 
            'dataset_name': 'inat', 
            'params_value': 0.1, 
            'learning_rate': 0.01, 
            'n_participants': 5
        }
    else:
        api = wandb.Api()
        sweep = api.sweep(f"sctpimming/FedQS-sweep-tosho/sweeps/{sweep_id}")
        best_run = sweep.best_run()
        run_config = best_run.config
    return run_config


if __name__ == "__main__":
    run_config = get_run_config()
    project_name = "FedQS-inat120k"
    wandb.init(project=project_name, config=run_config)
    FedAvg_tfd()
    wandb.finish()

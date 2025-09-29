import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
sys.path.append('code')


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

base_policy_list = [ "uniform", "KL", "CBS", "POC", "ODFL"]
balanced_policy_list = [f"{policy}_balanced" for policy in base_policy_list]
policy_name = base_policy_list[2]

def get_sweep_config():
    sweep_config = {
        'method': 'grid'
    }
    metric = {
        'name': 'global_test_loss',
        'goal': 'minimize'   
    }
    
    sweep_config['metric'] = metric
    sweep_config["early_terminate"] = {
        'type': 'hyperband',
        'min_iter': 200, # Minimum number of epochs to run before considering stopping        
    }
    
    parameters_dict = {
        'learning_rate': {
            "values": [0.001, 0.01, 0.1]
        },
        'batch_size': {
            "values": [8]
        },
        'n_participants': {
            "values": [5]
        },
        'n_steps':{
            "values": [50, 100, 250, 500]
        },
        'params_value':{
            "values": [0.1, 1, 10, 100, 1000]
        },
        # Fixed hyper parameter
        'policy_name':{
            "value": policy_name
        },
        'epoch':{
            "value": T
        },
        "n_clients":{
            "value": N
        },
        "n_available":{
            "value": r
        },
        "n_class":{
            "value": K
        },
        "dataset_name":{
            "value": dataset
        }
    }

    sweep_config['parameters'] = parameters_dict
    return sweep_config


if __name__ == "__main__":
    sweep_config = get_sweep_config()
    pprint.pprint(sweep_config)
    
    sweep_id = "atm70hpf"
    project_name = "FedQS-sweep-tosho"
    
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project="FedQS-sweep-tosho")
    
    wandb.agent(sweep_id, project = project_name, function=FedAvg_tfd)

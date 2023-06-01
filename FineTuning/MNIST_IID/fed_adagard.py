from asociita.components.orchestrator.generic_orchestrator import Orchestrator
from asociita.components.nodes.federated_node import FederatedNode
from asociita.models.pytorch.mnist import MnistNet
from asociita.models.pytorch.cifar10 import CifarNet
from asociita.datasets.fetch_data import load_data
from asociita.utils.helpers import Helpers
from asociita.datasets.shard_transformation import Shard_Transformation
import copy
import os
import sys
from shutil import rmtree
import matplotlib.pyplot as plt
import random
import numpy as np
import csv

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)


def prepare_repository(root: str,
                       structure: list[str],
                       remove_existing: bool = True):
    """Creates a one-level structure for an experimental repository.
    Args:
        root (path-like object): root directory
        structure (list[str]): list containing folder names
        remove_exisitng (bool): if True, will remove exisitng folders"""
    for folder_name in structure:
        path = os.path.join(root, 'results', folder_name)
        if os.path.exists(path):
            if remove_existing == True:
                rmtree(path)
                os.mkdir(path)
                os.mkdir(os.path.join(path, 'images'))
            else:
                raise "Unable to create repository structure - indicated folders already exist."
        else:
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'images'))


if __name__ == "__main__":
    # 1. Preparing simulation repository 
    root_path = os.path.join(os.getcwd(), "FineTuning", "MNIST_IID")
    os.chdir(root_path)
    metrics_save_path = os.path.join("results")
    transformations = ['FedAdagard']
    prepare_repository(os.getcwd(), transformations)
    
    
    # Loading configurations
    data_configuration = Helpers.load_from_json(os.path.join(os.getcwd(), "dataset_configurations", 'baseline_dataset.json'))
    simulation_configuration = Helpers.load_from_json(os.path.join(os.getcwd(), "simulation_configurations", "FedAdagard.json"))
    number_of_nodes = simulation_configuration["orchestrator"]["number_of_nodes"]
    simulation_configuration["orchestrator"]["nodes"] = [node for node in range(number_of_nodes)]
    current_save_path = os.path.join(os.getcwd(), 'results', 'FedAdagard')
    simulation_configuration["orchestrator"]["metrics_save_path"] = current_save_path
    simulation_configuration["orchestrator"]["archiver"]["metrics_savepath"] = current_save_path

    
    
    model_learning_rates = [10 ** (-i/2) for i in range(10)]
    server_learning_rates = [10 ** (-i/2) for i in range(10)]
    taus = [10 ** (-i) for i in range(5)]
    
    for tau in taus:
        for server_learning_rate in server_learning_rates:
            for model_learning_rate in model_learning_rates:
                simulation_configuration['nodes']["model_settings"]["learning_rate"] = model_learning_rate
                simulation_configuration["orchestrator"]["optimizer"]["learning_rate"] = server_learning_rate
                simulation_configuration["orchestrator"]["optimizer"]["tau"] = server_learning_rate

                simulation_configuration["orchestrator"]["archiver"]["orchestrator_filename"] = f"modellr_{model_learning_rate}__serverlr_{server_learning_rate}__tau{tau}_central_testset.csv"
                simulation_configuration["orchestrator"]["archiver"]["central_on_local_filename"] = f"modellr_{model_learning_rate}__serverlr_{server_learning_rate}__tau{tau}_local_testsets.csv"
                # DATA: Loading the data
                data = load_data(data_configuration)
                # DATA: Selecting data for the orchestrator
                orchestrator_data = data[0]
                # DATA: Selecting data for nodes
                nodes_data = data[1]
                model = MnistNet()
                orchestraotr = Orchestrator(settings=simulation_configuration)
                orchestraotr.prepare_orchestrator(model=model,
                                                validation_data=orchestrator_data)
                orchestraotr.train_protocol(nodes_data=nodes_data)
                    
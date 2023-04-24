# Project: Classification with Use of Neural Networks in the Keras Environment
# Application: Experimental application for neural network comparison with use of Keras
# Author: Michal Pyšík
# File: experiment.py

import numpy as np
from dataset import Dataset
from architectures import create_network


# Class encapsulating the selected experiment
class Experiment:
    # Prepares the experiment selected by the given data type
    # Creates the selected models and prepares the given dataset
    def __init__(self, data_type, selected_networks):
        print(
            "\n\n*****************************************************************\n\n"
        )
        if not any(selected_networks):
            print(
                "Error: Please select at least one model, then select the experiment again!"
            )
            return
        print("Preparing the dataset for the", data_type, "experiment...")
        self.data_type = data_type
        self.dataset = Dataset(data_type)
        print("Creating selected models...\n")
        self.networks = self.create_networks(data_type, selected_networks)
        print("All selected models have been successfully created.\n")

    # Creates the selected models based on the given data type
    def create_networks(self, data_type, selected_networks):
        networks = []

        if data_type != "Sequential":
            for i in range(4):
                if not selected_networks[i]:
                    networks.append(None)
                    continue
                network = create_network(
                    data_type, i, self.dataset.sample_shape, self.dataset.num_classes
                )
                networks.append(network)

        else:  # Sequential
            if selected_networks[0]:
                sequential_mlp = create_network(
                    data_type,
                    0,
                    self.dataset.vectorized_sample_shape,
                    self.dataset.num_classes,
                )
            else:
                sequential_mlp = None
            networks.append(sequential_mlp)

            for i in range(1, 4):
                if not selected_networks[i]:
                    networks.append(None)
                    continue
                network = create_network(
                    data_type,
                    i,
                    self.dataset.padded_sample_shape,
                    self.dataset.num_classes,
                    top_words=self.dataset.top_words,
                    max_review_len=self.dataset.max_review_len,
                )
                networks.append(network)

        return networks

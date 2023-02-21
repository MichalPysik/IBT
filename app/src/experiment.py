from dataset import Dataset
from architectures import create_network
import numpy as np


class Experiment:
    def __init__(self, data_type, selected_networks):
        print(
            "\n\n*****************************************************************\n\n"
        )
        if not any(selected_networks):
            print("Please select at least one neural network, then select the experiment again.")
            return
        print("Preparing the dataset for the", data_type, "experiment...")
        self.data_type = data_type
        self.dataset = Dataset(data_type)
        print("Creating selected neural networks...\n")
        self.networks = self.create_networks(data_type, selected_networks)
        print("All selected neural networks have been successfully created.\n")
        self.finished_epochs = 0

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

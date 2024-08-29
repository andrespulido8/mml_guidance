import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


class SimpleDNN(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        nodes_per_layer,
        output_size=2,
        activation_fn="relu",
    ):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, nodes_per_layer))
            else:
                self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
        self.layers.append(nn.Linear(nodes_per_layer, output_size))
        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function")

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)

    def predict(self, x):
        return self.forward(torch.from_numpy(x.astype(np.float32))).detach().numpy()


def main():
    path = os.path.expanduser("~/mml_ws/src/mml_guidance/hardware_data/")
    # print the files in the directory
    df = pd.read_csv(path + "converted_occlusion_training_data.csv")
    X = df.iloc[:, :-2].values
    y = df.iloc[:, -2:].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    return 0


if __name__ == "__main__":
    main()

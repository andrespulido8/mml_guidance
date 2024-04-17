import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import os


class SimpleDNN(nn.Module):
    def __init__(
        self, input_size, num_layers, nodes_per_layer, output_size, activation_fn
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


def train(model, criterion, optimizer, X_train, y_train, epochs=100):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}")
    return losses


def evaluate(model, criterion, X_test, y_test):
    with torch.no_grad():
        outputs = model.forward(X_test)
        loss = criterion(outputs, y_test)
    return loss.item()


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

    # Define the hyperparameters to tune
    param_grid = {
        "num_layers": [2, 4, 8],
        "nodes_per_layer": [20, 40, 80],
        "lr": [0.01, 0.001, 0.0001],
    }
    grid = ParameterGrid(param_grid)

    best_loss = float("inf")
    best_params = None

    for params in grid:
        print("Training with parameters: ", params)
        model = SimpleDNN(
            input_size=X_train.shape[1],
            num_layers=params["num_layers"],
            nodes_per_layer=params["nodes_per_layer"],
            output_size=2,
            activation_fn="relu",
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        losses = train(model, criterion, optimizer, X_train, y_train, epochs=100)
        test_loss = evaluate(model, criterion, X_test, y_test)
        print("Test Loss: ", test_loss, "\n")
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = params
            torch.save(model.state_dict(), "simple_dnn_best.pth")
        plt.plot(
            losses,
            label=f"num_layers={params['num_layers']}, nodes_per_layer={params['nodes_per_layer']}, lr={params['lr']}",
        )
        print(
            f"Test Loss for num_layers={params['num_layers']}, nodes_per_layer={params['nodes_per_layer']}, lr={params['lr']}: {test_loss}"
        )
    print(f"Best parameters: {best_params}")
    print(f"Best test loss: {best_loss}")

    # second figure showing the predicted direction of motion and the actual direction of motion
    model = SimpleDNN(
        input_size=X_train.shape[1],
        num_layers=best_params["num_layers"],
        nodes_per_layer=best_params["nodes_per_layer"],
        output_size=2,
        activation_fn="relu",
    )
    model.load_state_dict(torch.load("simple_dnn_best.pth"))
    model.eval()
    y_pred = model(X_test).detach().numpy()
    # figure with legends
    plt.figure()
    # plot an arrow from the last point of X_test to the predicted point and the actual point
    for i in range(X_test.shape[0]):
        plt.arrow(
            X_test[i, -2],
            X_test[i, -1],
            y_pred[i, 0] - X_test[i, -2],
            y_pred[i, 1] - X_test[i, -1],
            color="red",
        )
        plt.arrow(
            X_test[i, -2],
            X_test[i, -1],
            y_test[i, 0] - X_test[i, -2],
            y_test[i, 1] - X_test[i, -1],
            color="blue",
        )
    plt.arrow(
        X_test[0, -2],
        X_test[0, -1],
        y_pred[0, 0] - X_test[i, -2],
        y_pred[0, 1] - X_test[i, -1],
        color="red",
        label="Predicted",
    )
    plt.arrow(
        X_test[0, -2],
        X_test[0, -1],
        y_test[0, 0] - X_test[i, -2],
        y_test[0, 1] - X_test[i, -1],
        color="blue",
        label="Actual",
    )
    occlusions = np.array(
        [[-1.75, -0.75, -1.1, -0.1], [-0.15, 0.85, -0.3, 0.7]]
    )  # [x_min, x_max, y_min, y_max]
    # plot the squares representing occlusions
    for occlusion in occlusions:
        plt.plot(
            [occlusion[0], occlusion[1], occlusion[1], occlusion[0], occlusion[0]],
            [occlusion[2], occlusion[2], occlusion[3], occlusion[3], occlusion[2]],
            color="black",
        )

    # min and max x and y values
    plt.xlim([-2, 1.0])
    plt.ylim([-1.5, 2])
    plt.legend()
    plt.show()
    return 0


if __name__ == "__main__":
    main()

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
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


def parameter_search(
    ModelClass, param_grid, X_train, y_train, X_test, y_test, epochs, weights_filename
):
    grid = ParameterGrid(param_grid)

    best_loss = float("inf")
    best_params = None

    for params in grid:
        print("Training with parameters: ", params)
        # initialize the model with the given parameters, make it so that it can take any parameter for any model
        model = ModelClass(**params)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        losses = train(model, criterion, optimizer, X_train, y_train, epochs)
        test_loss = evaluate(model, criterion, X_test, y_test)
        print("Test Loss: ", test_loss, "\n")
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = params
            torch.save(model.state_dict(), weights_filename)
        plt.plot(
            losses,
            label=f"{params}",
        )

    plt.legend()
    plt.show()

    print(f"\nBest parameters: {best_params}")
    print(f"Best test loss: {best_loss}")

    return best_params


def plot_NN_output(X_test, y_pred, y_test, is_velocities, df=None):
    # font sizes for the plot labels
    font = {"family": "serif", "weight": "normal", "size": 40}
    sns.set_theme()
    sns.set_style("white")
    sns.set_context("paper", font_scale=3)

    # figure with legends
    fig, ax = plt.subplots()

    occlusions = np.array(
        [[-1.75, -0.75, -1.1, -0.1], [-0.15, 0.85, -0.3, 0.7]]
    )  # [x_min, x_max, y_min, y_max]
    # plot the squares representing occlusions
    # for occlusion in occlusions:
    #    plt.plot(
    #        [occlusion[0], occlusion[1], occlusion[1], occlusion[0], occlusion[0]],
    #        [occlusion[2], occlusion[2], occlusion[3], occlusion[3], occlusion[2]],
    #        color="black",
    #    )
    # plot an arrow from the last point of X_test to the predicted point and the actual point

    if not is_velocities:
        # Calculate arrow directions for predictions and actual values
        pred_dx = y_pred[:, 0] - X_test[:, -2]
        pred_dy = y_pred[:, 1] - X_test[:, -1]
        actual_dx = y_test[:, 0] - X_test[:, -2]
        actual_dy = y_test[:, 1] - X_test[:, -1]

        # Plot path
        # ax.plot(X_test[:, 0], X_test[:, 1], color="black", linestyle="--", label="Path", alpha=0.5)

        # Plot predicted arrows
        plt.quiver(
            X_test[:, -2],
            X_test[:, -1],
            2 * pred_dx,
            2 * pred_dy,
            color="red",
            label="Predicted",
        )

        # Plot actual arrows
        plt.quiver(
            X_test[:, -2],
            X_test[:, -1],
            2 * actual_dx,
            2 * actual_dy,
            color="blue",
            label="Actual",
        )

    else:
        # plot the predicted and actual velocities as arrows from the positions defined in df
        _, Xpos_test, _, _ = train_test_split(
            df.iloc[:, :-2].values, df.iloc[:, -2:], test_size=0.2, random_state=42
        )
        for i in range(Xpos_test.shape[0] - 1):
            plt.arrow(
                Xpos_test[i, -2],
                Xpos_test[i, -1],
                y_pred[i, -2] * 0.33,
                y_pred[i, -1] * 0.33,
                color="red",
            )
            plt.arrow(
                Xpos_test[i, -2],
                Xpos_test[i, -1],
                y_test[i, 0] * 0.33,
                y_test[i, 1] * 0.33,
                color="blue",
            )

    # axis labels
    plt.xlabel("$x_g$ [m]")
    plt.ylabel("$y_g$ [m]")
    # min and max x and y values
    plt.xlim([-2, 1.0])
    plt.ylim([-1.5, 2])
    # equal aspect ratio
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.show()


def main():
    is_velocities = False
    path = os.path.expanduser("~/mml_ws/src/mml_guidance/sim_data/training_data/")
    # print the files in the directory
    if is_velocities:
        df_vel = pd.read_csv(path + "converted_training_data.csv")
    df = pd.read_csv(path + "converted_noisy_training_data.csv")
    X = df.iloc[:, :-2].values if not is_velocities else df_vel.iloc[:, :-2].values
    y = df.iloc[:, -2:].values if not is_velocities else df_vel.iloc[:, -2:].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    # Define the hyperparameters to tune
    # param_grid = {
    #    "input_size": [X_train.shape[1]],
    #    "num_layers": [2, 4, 8],
    #    "nodes_per_layer": [20, 40, 80],
    # }

    # ModelClass = SimpleDNN

    # epochs = 100
    # best_params = parameter_search(
    #    ModelClass,
    #    param_grid,
    #    X_train,
    #    y_train,
    #    X_test,
    #    y_test,
    #    epochs,
    #    weights_filename="occlusion_vel_dnn_best.pth",
    # )

    # second figure showing the predicted direction of motion and the actual direction of motion
    model = SimpleDNN(
        input_size=X_train.shape[1],
        num_layers=2,
        nodes_per_layer=80,
        output_size=2,
        activation_fn="relu",
    )
    model.load_state_dict(
        torch.load(path + "/../../scripts/mml_network/models/" + "perfect_dnn_best.pth")
    )
    model.eval()
    y_pred = model(X_test).detach().numpy()
    X_test = X_test.detach().numpy()

    plot_NN_output(X_test, y_pred, y_test, is_velocities, df)
    return 0


if __name__ == "__main__":
    main()

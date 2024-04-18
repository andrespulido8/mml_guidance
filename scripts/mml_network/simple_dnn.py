import math
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=100, dropout=0.01):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=100, dropout=0.01):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=10)

    def forward(self, src):
        output = self.pos_encoder(src)
        for layer in self.layers:
            output = layer(output)
        return output


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=100,
        dropout=0.01,
        output_size=2,
    ):
        super().__init__()
        self.input_projection = nn.Linear(2, d_model)
        self.transformer = TransformerEncoder(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = x.view(
            batch_size, sequence_length // 2, 2
        )  # Reshape to (batch_size, 10, 2)
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.output_projection(x[:, -1, :])  # Use the output of the last time step
        return x

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

    # model = simple_dnn.SimpleDNN(
    #    input_size=20,
    #    num_layers=2,
    #    nodes_per_layer=80,
    #    output_size=2,
    #    activation_fn="relu",
    # )
    model = TransformerModel(d_model=12, nhead=2, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    losslist = train(model, criterion, optimizer, X_train, y_train, epochs=100)
    torch.save(model.state_dict(), "transformer.pth")

    # plot the loss
    plt.plot(losslist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # load the saved weights
    # model.load_state_dict(torch.load("transformer.pth"))

    # Evaluation
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        print(f"Test Loss: {test_loss.item()}")

    # model.load_state_dict(torch.load("simple_dnn_best.pth"))
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

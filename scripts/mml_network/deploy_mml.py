import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from data_functions import get_data
from training_functions import Trainer
from transformer_functions import TransAm

plt.rcParams["figure.figsize"] = (16, 12) # (w, h)

input_window = 9 # number of input steps
output_window = 1 # number of prediction steps
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


fname='andres_bag_1.csv'
close = pd.read_csv('./data/'+fname, header=0,index_col=0, squeeze=False).to_numpy()
print(close.shape)

train_data, val_data = get_data(close, 0.1, input_window, output_window, device, scale_data = False) # 60% train, 40% test split
model = TransAm(in_dim=2, feature_size = 256, num_layers=2).to(device)
model.load_state_dict(torch.load('./models/current.pth'))
model.eval()

# This block is only included because I put forecast_seq in the trainer class
criterion = nn.MSELoss() # Loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
trainer = Trainer(model, optimizer, criterion, scheduler, train_data, input_window, batch_size)

test_result, truth = trainer.forecast_seq(val_data)
print(test_result.shape)
plt.figure()
plt.plot(truth[:,0], truth[:,1], '.', color='red', alpha=0.7)
plt.plot(test_result[:,0], test_result[:,1], '.', color='blue', linewidth=0.7)
plt.title('Actual vs Forecast')
plt.legend(['Actual', 'Forecast'])

fig, ax = plt.subplots(2,1)
ax[0].plot(truth[:,0], '.', color='red', alpha=0.7)
ax[0].plot(test_result[:,0], '.', color='blue', linewidth=0.7)
ax[1].plot(truth[:,1], '.', color='red', alpha=0.7)
ax[1].plot(test_result[:,1], '.', color='blue', linewidth=0.7)
plt.title('Actual vs Forecast')
plt.legend(['Actual', 'Forecast'])
plt.xlabel('Time Steps')
plt.show()

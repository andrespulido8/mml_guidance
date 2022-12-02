import torch
import torch.nn as nn
import numpy as np

from mml_network.data_functions import get_data, get_batch
#from training_functions import Trainer
from mml_network.transformer_functions import TransAm

class Motion_Model():
    def __init__(self, model_file):
        self.input_window = 9 # number of input steps
        self.output_window = 1 # number of prediction steps
        batch_size = 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        #train_data, val_data = get_data(close, 0.1, input_window, output_window, device, scale_data = False) # 60% train, 40% test split
        self.model = TransAm(in_dim=2, feature_size = 256, num_layers=2).to(device)
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

        # This block is only included because I put forecast_seq in the trainer class
        #criterion = nn.MSELoss() # Loss function
        #optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        #self.trainer = Trainer(model, optimizer, criterion, scheduler, train_data, input_window, batch_size)

    def predict(self, particles):
        with torch.no_grad():
                output = self.model(torch.from_numpy(particles[:,:2]).float())
                forecast_seq = torch.cat((particles[1,:,:],output[-1, :,:].cpu()), 0)

        return forecast_seq
        
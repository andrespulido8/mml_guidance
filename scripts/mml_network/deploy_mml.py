import torch
import torch.nn as nn
import numpy as np

from data_functions import get_data
from training_functions import Trainer
from transformer_functions import TransAm

class Motion_Model():
    def __init__(self):
        input_window = 9 # number of input steps
        output_window = 1 # number of prediction steps
        batch_size = 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        #train_data, val_data = get_data(close, 0.1, input_window, output_window, device, scale_data = False) # 60% train, 40% test split
        self.model = TransAm(in_dim=2, feature_size = 256, num_layers=2).to(device)
        self.model.load_state_dict(torch.load('./models/current.pth'))
        self.model.eval()

        # This block is only included because I put forecast_seq in the trainer class
        criterion = nn.MSELoss() # Loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        self.trainer = Trainer(model, optimizer, criterion, scheduler, train_data, input_window, batch_size)

    def predict(self, particles):

        #test_result, truth = trainer.forecast_seq(val_data)
            def forecast_seq(self, sequences):
        """Sequences data has to been windowed and passed through device"""
        start_timer = time.time()
        self.model.eval() 
        forecast_seq = torch.empty((0,2))    
        actual = torch.empty((0,2))
        batch_size = 100
        with torch.no_grad():
            #for i in range(0, len(sequences) - 1):
            for i in range(0, len(sequences)-1,batch_size):
                data, target = get_batch(sequences, i, batch_size, self.input_window)
                output = self.model(data)            
                #forecast_seq = torch.cat((forecast_seq,torch.unsqueeze(output[-1, :,-2:], 0).cpu()), 0)
                forecast_seq = torch.cat((forecast_seq,output[-1, :,-2:].cpu()), 0)
                a = len(target.shape)
                if a == 3:
                    #actual = torch.cat((actual, torch.unsqueeze(target[-1,:,-2:], 0).cpu()), 0)
                    actual = torch.cat((actual, target[-1,:,-2:].cpu()), 0)
                else:
                    actual = torch.cat((actual, torch.unsqueeze(target[-1,-2:], 0).cpu()), 0)
                
        timed = time.time()-start_timer
        print(f"{timed} sec")

        return forecast_seq, actual
        